import jax.numpy as jnp
import jax.scipy as jsp


def warp_img_with_flow(img: jnp.ndarray, flow: jnp.ndarray) -> jnp.ndarray:
    """
    Warp the image with the flow field.
    """
    H, W = img.shape
    H_flow, W_flow = flow.shape[:2]
    assert H == H_flow and W == W_flow
    x = jnp.arange(W)
    y = jnp.arange(H)
    grid_x, grid_y = jnp.meshgrid(x, y)
    grid_x = grid_x + flow[:, :, 0]
    print(grid_x)
    grid_y = grid_y + flow[:, :, 1]

    coordinate_cloud = jnp.stack([grid_y, grid_x], axis=-1).reshape(
        -1
    )  # decide if y or x is first --> (H*W,2)
    value_cloud = img.reshape(-1)  # --> (H*W)
    interpolator = jsp.interpolate.LinearNDInterpolator(coordinate_cloud, value_cloud)
    warped_img = jsp.ndimage.map_coordinates(
        img, (grid_y, grid_x), order=1, mode="nearest"
    )
    return warped_img


def test_warp_img_with_flow():
    import numpy as np

    size = 5
    img = np.ones((size, size))
    img[1, 1] = 4
    print(img)
    flow = np.zeros((size, size, 2))
    flow[1, 1, 0] = 1  # move the pixel at (1,1) to one to the right
    flow[1, 2, 0] = 1  # move the pixel at (1,2) to one to the right
    warped_img = warp_img_with_flow(img, flow)
    assert warped_img.shape == img.shape
    print(warped_img)


if __name__ == "__main__":
    test_warp_img_with_flow()
