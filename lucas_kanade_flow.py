import jax.numpy as jnp

from image_gradients import grad_t, grad_x, grad_y


def patchified_sum_pooling(
    img: jnp.ndarray, kernel_size=(10, 10), return_num_patches=False
):
    """
    This function is basically sum pooling with the kernel size as the pooling size.
    """

    M, N = img.shape
    K = kernel_size[0]
    L = kernel_size[1]

    MK = M // K
    NL = N // L
    summed_patches = img[: MK * K, : NL * L].reshape(MK, K, NL, L).sum(axis=(1, 3))
    if return_num_patches:
        return summed_patches, MK, NL
    return summed_patches


def lucas_canade_flow(img0, img1, kernel_size=(3, 3)):
    I_x = 1 / 2 * (grad_x(img0, normalize=False) + grad_x(img1, normalize=False))
    I_y = 1 / 2 * (grad_y(img0, normalize=False) + grad_y(img1, normalize=False))
    I_t = grad_t(img0, img1, normalize=False)

    I_xI_t = I_x * I_t
    I_yI_t = I_y * I_t
    I_xI_x = I_x * I_x
    I_yI_y = I_y * I_y
    I_xI_y = I_x * I_y

    I_xI_x_sum, H, W = patchified_sum_pooling(
        I_xI_x, kernel_size=kernel_size, return_num_patches=True
    )
    I_yI_y_sum = patchified_sum_pooling(I_yI_y, kernel_size=kernel_size)
    I_xI_y_sum = patchified_sum_pooling(I_xI_y, kernel_size=kernel_size)
    I_xI_t_sum = patchified_sum_pooling(I_xI_t, kernel_size=kernel_size)
    I_yI_t_sum = patchified_sum_pooling(I_yI_t, kernel_size=kernel_size)

    AT_A = jnp.stack([I_xI_x_sum, I_xI_y_sum, I_xI_y_sum, I_yI_y_sum], axis=-1).reshape(
        H, W, 2, 2
    )
    dangerous_inv = jnp.linalg.inv(AT_A)
    AT_A_inv = jnp.nan_to_num(dangerous_inv)
    ATB = -jnp.stack([I_xI_t_sum, I_yI_t_sum], axis=-1).reshape(H, W, 2)
    u_flow = jnp.einsum("ijkl,ijl->ijk", AT_A_inv, ATB)
    return u_flow
