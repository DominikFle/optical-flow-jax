import jax.numpy as jnp


def to_colorful_grayscale(image: jnp.ndarray):
    """
    Expects an image with shape (H,W)
    Returns an image with shape (H,W,3) with the same values in each channel
    """
    image = jnp.tile(jnp.expand_dims(image, axis=-1), (1, 1, 3))
    return image
