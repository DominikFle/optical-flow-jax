import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np

# Special transform functions (we'll understand what these are very soon!)
from jax import grad, jit, vmap, pmap

# JAX's low level API
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)
from jax import lax
import matplotlib.pyplot as plt

from image_gradients import grad_t, grad_x, grad_y


def get_img(index, in_gray=False):
    path_to_imgs = f"C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\nuimage_{index}.jpg"
    img = plt.imread(path_to_imgs) / 255.0
    if in_gray:
        img = jnp.mean(img, axis=-1)
    return img


img0 = get_img(0, in_gray=True)
img1 = get_img(1, in_gray=True)
img2 = get_img(2, in_gray=True)

grad_filter_x = jnp.array([[-1, 0, 1]]) / 2
grad_filter_y = jnp.array([[-1], [0], [1]])

print(img0.shape)
print(grad_filter_x.shape)
I_x = 1 / 2 * (grad_x(img0, normalize=False) + grad_x(img1, normalize=False))
I_y = 1 / 2 * (grad_y(img0, normalize=False) + grad_y(img1, normalize=False))
I_t = grad_t(img0, img1, normalize=False)
print(I_x.shape)
print(I_x.min(), I_x.max())
plt.imsave(
    "C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\nuimage_grad_x.jpg",
    I_x,
)
plt.imsave(
    "C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\nuimage_grad_y.jpg",
    I_y,
)
plt.imsave(
    "C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\nuimage_grad_t.jpg",
    I_t,
)
I_xI_t = I_x * I_t
I_yI_t = I_y * I_t
I_xI_x = I_x * I_x
I_yI_y = I_y * I_y
I_xI_y = I_x * I_y


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


def direct_inverse(A):
    """Compute the inverse of matrices in an array of shape (N,N,M)"""
    return np.linalg.inv(A.transpose(2, 0, 1)).transpose(1, 2, 0)


ones = jnp.ones((10, 10))
print(patchified_sum_pooling(ones, kernel_size=(10, 10)))
I_xI_x_sum, H, W = patchified_sum_pooling(
    I_xI_x, kernel_size=(10, 10), return_num_patches=True
)
I_yI_y_sum = patchified_sum_pooling(I_yI_y, kernel_size=(10, 10))
I_xI_y_sum = patchified_sum_pooling(I_xI_y, kernel_size=(10, 10))
I_xI_t_sum = patchified_sum_pooling(I_xI_t, kernel_size=(10, 10))
I_yI_t_sum = patchified_sum_pooling(I_yI_t, kernel_size=(10, 10))

# AT = jnp.stack([I_x, I_y], axis=-1).reshape(H, W, 2)
AT_A = jnp.stack([I_xI_x_sum, I_xI_y_sum, I_xI_y_sum, I_yI_y_sum], axis=-1).reshape(
    H, W, 2, 2
)
# AT_A_inv = jnp.linalg.inv(AT_A)
AT_A_inv = jnp.linalg.inv(jnp.array([[1, 2], [3, 4]]))
print(AT_A_inv.shape)
raise ValueError("Stop here")
ATB = -jnp.stack([I_xI_t_sum, I_yI_t_sum], axis=-1).reshape(H, W, 2)

u_flow = np.einsum("ijkl,ijl->ijk", AT_A_inv, ATB)
# ATB = jnp.einsum("ijkl,ijl->ijk", AT, B)
# def matmul(A, B):
#     return jnp.ma
# print(type(B))
# print(AT_A[0, 0])

plt.imsave(
    "C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\nuimage_I_xI_x_sum.jpg",
    I_xI_x_sum,
)
plt.imsave(
    "C:\\Users\\Domi\\Desktop\\Programmmieren\\OpticalFlowJax\\images\\u_flow_x.jpg",
    u_flow[:, :, 0],
)
