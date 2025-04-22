import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from models.lstm import LSTM
from models.transformer import StaticEmbedder, CrossAttnDecoder


class ATTN_LSTM(eqx.Module):
    """Model that uses attention to mix time frequencies.

    Attributes
    ----------
    active_source: dict
        Boolean indicating if this data source (and encoder) are to be used.
    encoders: dict
        Encoders, one for each dynamic data source.
    static_embedder: StaticEmbedder
        Embedder for static data.
    decoders: dict
        Decoders, one for each cross-attention or self-attention.
    head: eqx.nn.Linear
        Linear layer that maps the output of the decoders to the target
        variable(s).
    target: list
        Names of the target variables.
    """
    active_source: dict[str:bool]
    static_embedder: StaticEmbedder
    cross_source: str
    cross_vars: list[str]
    proj: dict[str:eqx.nn.Linear]
    attn: dict[str:eqx.Module]
    lstm: LSTM
    target: list[str]

    def __init__(self,
                 *,
                 target: list,
                 seq_length: int,
                 dynamic_sizes: dict,
                 static_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 seed: int,
                 dropout: float,
                 time_aware: dict,
                 active_source: dict = {}):
        """Initializes an LSTM_MLP_ATTN model.

        Parameters
        ----------
        target: list
            The names of the target variables.
        seq_length: int
            The length of the input sequence.
        dynamic_sizes: dict
            A dictionary of the sizes of the dynamic features.
        static_size: int
            The number of static features.
        hidden_size: int
            The size of the hidden state.
        num_layers: int
            The number of layers in the LSTM and MLP.
        num_heads: int
            The number of heads in the attention.
        seed: int
            A seed for the random number generator.
        dropout: float
            The dropout rate.
        time_aware: dict
            A dictionary of booleans indicating whether each dynamic feature is
            time-aware.
        active_source: dict, optional
            A dicitonary of booleans indicating whether each dyanmic source
            (collection of features), and the accompanying encoder will be used.
            Defaults to using all sources.
        """
        self.target = target
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 4)

        # Encoder for static data if used.
        entity_aware = static_size > 0
        if entity_aware:
            self.static_embedder = StaticEmbedder(static_size, hidden_size, dropout,
                                                  keys[0])
            static_size = hidden_size
        else:
            self.static_embedder = None
            static_size = 0

        # Default all sources to true. Optional specification in the model args.
        if len(active_source) == 0:
            for source in dynamic_sizes.keys():
                active_source[source] = True
        self.active_source = active_source

        self.proj = {}
        proj_keys = jax.random.split(keys[1], len(dynamic_sizes))
        for (var_name, var_size), var_key in zip(dynamic_sizes.items(), proj_keys):
            self.proj[var_name] = eqx.nn.Linear(var_size, hidden_size, key=var_key)

        # Cross-attn or Self-attn
        self.attn = {}
        self.cross_source = list(dynamic_sizes.keys())[0]
        if len(dynamic_sizes) > 1:
            self.cross_vars = list(dynamic_sizes.keys())[1:]
        else:
            self.cross_vars = [self.cross_source]

        # Set up each cross-attention block
        attn_keys = jax.random.split(keys[2], len(self.cross_vars))
        for var_name, var_key in zip(self.cross_vars, attn_keys):
            self.attn[var_name] = CrossAttnDecoder(seq_length, hidden_size, hidden_size,
                                                   hidden_size, num_layers, num_heads,
                                                   dropout, entity_aware, var_key,
                                                   False)

        # LSTM
        self.lstm = LSTM(hidden_size * len(self.attn),
                         hidden_size,
                         out_size=len(target),
                         dropout=dropout,
                         key=keys[3])

    def finetune_update(self, *, active_source: dict):
        """Updates the model configuration after initialization.
        
        These updates must not break the forward call of the model. Only some
        things can reasonably change to ensure it does not break.

        Parameters
        ----------
        active_source: dict
            Boolean indicating if this data source (and encoder) are to be used.
        """
        for source, active in active_source.items():
            if source in self.active_source:
                self.active_source[source] = active
            else:
                raise ValueError(f"Source '{source}' not found in active_source.")
        print(self.active_source)

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        """The forward pass of the data through the model 

        Parameters
        ----------
        data: dict[str, Array | dict[str, Array]]
            The input data.
        key: PRNGKeyArray
            A PRNG key used to apply the model.

        Returns
        -------
        Array
            The output of the model.
        """
        keys = jax.random.split(key, 3)

        # Static embedding
        if self.static_embedder:
            static_bias = self.static_embedder(data['static'], keys[0])
        else:
            static_bias = None

        # Attention
        source_proj = jax.vmap(self.proj[self.cross_source])(
            data['dynamic'][self.cross_source])
        attn_keys = jax.random.split(keys[1], len(self.attn))
        attn_data = []

        for (var_name, attn_block), var_key in zip(self.attn.items(), attn_keys):
            if not self.active_source[var_name]:
                continue

            cross_var_data = data['dynamic'][var_name]
            mask = ~jnp.any(jnp.isnan(cross_var_data), axis=1)
            x_d = jnp.where(jnp.expand_dims(mask, 1), data['dynamic'][var_name], 0.0)
            cross_proj = jax.vmap(self.proj[var_name])(x_d)

            attn_data.append(
                attn_block(source_proj, cross_proj, static_bias, mask, var_key))

        attn_data = jnp.concatenate(attn_data, axis=0)

        return self.lstm(attn_data, keys[2])
