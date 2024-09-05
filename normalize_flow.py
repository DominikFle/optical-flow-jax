import jax.numpy as jnp


def normalize_flow(flow: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize the flow values to be between 0 and 1.
    """
    flow_min = -1.0
    flow_max = 1.0
    normalized_flow = (flow - flow_min) / (flow_max - flow_min)
    return normalized_flow
