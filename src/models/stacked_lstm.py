import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .lstm import BidirectionalLSTM, LSTM


class STACKED_LSTM(eqx.Module):
    in_proj: eqx.nn.Linear
    static_proj: eqx.nn.Linear
    in_lstm: BidirectionalLSTM
    in_head: eqx.nn.Linear
    out_lstm: LSTM
    target: list[str]
    seq2seq: bool = eqx.field(static=True)

    def __init__(self,
                 *,
                 in_targets: list[str],
                 out_targets: list[str],
                 dynamic_size: int,
                 static_size: int,
                 hidden_size: int,
                 seed: int,
                 dropout: float,
                 seq2seq: bool = False):

        self.target = in_targets + out_targets
        self.seq2seq = seq2seq
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 5)

        # Embedding layer for dynamic data
        self.in_proj = eqx.nn.Linear(dynamic_size, hidden_size, key=keys[0])

        # Embedding layer for static data if used.
        entity_aware = static_size > 0
        if entity_aware:
            self.static_proj = eqx.nn.Linear(static_size, hidden_size, key=keys[1])
            static_embed_size = hidden_size
        else:
            self.static_proj = None
            static_embed_size = 0

        # First LSTM layer
        self.in_lstm = BidirectionalLSTM(hidden_size + static_embed_size,
                                         hidden_size,
                                         None,
                                         return_all=True,
                                         dropout=dropout,
                                         key=keys[2])
        self.in_head = eqx.nn.Linear(hidden_size * 2, len(in_targets), key=keys[3])

        # Second LSTM layer
        self.out_lstm = LSTM(hidden_size * 2,
                             hidden_size,
                             out_size=len(out_targets),
                             return_all=seq2seq,
                             dropout=dropout,
                             key=keys[4])

    def __call__(self, data: dict[str, Array | dict[str, Array]], key: PRNGKeyArray):
        keys = jax.random.split(key, 2)

        # Replace NaN values with 0s in the dynamic data
        x_d = jnp.nan_to_num(data['dynamic']['era5'], nan=0.0)
        x = jax.vmap(self.in_proj)(x_d)

        if self.static_proj:
            x_s = self.static_proj(data['static'])
            x_s_tiled = jnp.tile(x_s, (x.shape[0], 1))
            x = jnp.concat([x, x_s_tiled], axis=1)

        x_in = self.in_lstm(x, keys[0])
        y_out = self.out_lstm(x_in, keys[1])

        if self.seq2seq:
            y_in = jax.vmap(self.in_head)(x_in)
        else:
            y_in = self.in_head(x_in[-1, ...])

        out = jnp.concat([y_in, y_out], axis=-1)
        return out
