import jax.numpy as jnp
import cv2
import numpy as np


def visualize_optical_flow(
    flow_field: jnp.ndarray,  # (H/Kh,W/Kw,2)
    original_image: jnp.ndarray,  # (H,W,3)
    kernel_size=(10, 10),  # (Kh,Kw)
    overscale_factor=1,
) -> np.ndarray:
    """
    Visualize the optical flow field on top of the original image using arrows.
    """
    H_Kh = flow_field.shape[0]
    W_Kw = flow_field.shape[1]
    Kh = kernel_size[0]
    Kw = kernel_size[1]
    image_to_visualize = (np.array(original_image) * 255).astype(np.uint8)
    for i in range(H_Kh):
        for j in range(W_Kw):
            flow = flow_field[i, j]
            cv2.arrowedLine(
                image_to_visualize,
                (j * Kw, i * Kh),  # swapped because opencv uses (x,y) instead of (h,w)
                (
                    int(j * Kw + flow[0] * overscale_factor),
                    int(i * Kh + flow[1] * overscale_factor),
                ),
                (0, 0, 255),
                3,
            )
    return image_to_visualize
