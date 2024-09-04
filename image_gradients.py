import jax.numpy as jnp
import jax.scipy as jsp


def grad_x(img, normalize=False):
    grad_filter_x = jnp.array([[-1, 0, 1]]) / 2
    grad_img_x = jsp.signal.convolve(img, grad_filter_x, mode="same")
    if normalize:
        grad_img_x = grad_img_x + 1 / 2
    return grad_img_x


def grad_y(img, normalize=False):
    grad_filter_y = jnp.array([[-1], [0], [1]])
    grad_img_y = jsp.signal.convolve(img, grad_filter_y, mode="same")
    if normalize:
        grad_img_y = grad_img_y + 1 / 2
    return grad_img_y


def grad_t(img0, img1, normalize=False):
    grad_img_t = (img1 - img0) / 2
    if normalize:
        grad_img_t = grad_img_t + 1 / 2
    return grad_img_t
