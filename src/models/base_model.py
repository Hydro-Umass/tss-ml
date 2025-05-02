import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class BaseModel(eqx.Module):
    head: eqx.nn.Linear
    target: list[str]

    def __init__(self, hidden_size, dense_size, target, *, key):
        if dense_size is not None:
            self.head = eqx.nn.Linear(hidden_size, dense_size, key=key)
        else:
            self.head = None
        self.target = target

    def __call__(self, data: dict[str, Array | dict[str, Array]],
                 key: PRNGKeyArray | None) -> Array:
        raise NotImplementedError

    def finetune_update(self, **kwargs):
        raise NotImplementedError
