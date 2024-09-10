import cv2
import jax.numpy as jnp
import numpy as np

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


def test_patchified_sum_pooling():
    import numpy as np

    src = np.ones((12, 12))
    src[:3, :3] = 4
    summed_patches = patchified_sum_pooling(src, kernel_size=(3, 3))
    # assert np.all(summed_patches == 9)
    assert summed_patches[0, 0] == 36
    print(summed_patches.shape)
    print(summed_patches)
    # print(jnp.gradient(jnp.array([[0, 1], [0, 1]])))


def lucas_canade_flow(img0, img1, kernel_size=(3, 3), verbose=False):
    print(jnp.gradient(img0))
    I_grad = jnp.gradient(img0) + jnp.gradient(img1)
    I_y, I_x = I_grad[0] / 2, I_grad[1] / 2
    # I_x = 1 / 2 * (grad_x(img0, normalize=False) + grad_x(img1, normalize=False))
    # I_y = 1 / 2 * (grad_y(img0, normalize=False) + grad_y(img1, normalize=False))
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
    if verbose:
        num_nan = jnp.sum(jnp.isnan(dangerous_inv))
        print(f"Number of NaNs in AT_A_inv: {num_nan} of {dangerous_inv.size}")
    AT_A_inv = jnp.nan_to_num(dangerous_inv)
    ATB = -jnp.stack([I_xI_t_sum, I_yI_t_sum], axis=-1).reshape(H, W, 2)
    u_flow = jnp.einsum("ijkl,ijl->ijk", AT_A_inv, ATB)
    return u_flow


def lucas_canade_flow_naive(
    img0, img1, kernel_size=(3, 3), useOwnGrad=False, verbose=False, usePatches=True
):
    if not useOwnGrad:
        I_grad = jnp.gradient(img0) + jnp.gradient(img1)
        I_y, I_x = I_grad[0] / 2, I_grad[1] / 2
        I_grad = jnp.gradient(img0)
        I_y, I_x = I_grad[0], I_grad[1]
    else:
        I_x = 1 / 2 * (grad_x(img0, normalize=False) + grad_x(img1, normalize=False))
        I_y = 1 / 2 * (grad_y(img0, normalize=False) + grad_y(img1, normalize=False))
    I_t = grad_t(img0, img1, normalize=False, grad_factor=1)
    if usePatches:
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
        ATA = np.zeros((H, W, 2, 2))
        ATA_inv = np.zeros((H, W, 2, 2))
        ATB = np.zeros((H, W, 2))
        for i in range(H):
            for j in range(W):
                ATB[i, j] = -np.array([I_xI_t_sum[i, j], I_yI_t_sum[i, j]])
                ATA[i, j] = np.array(
                    [
                        [I_xI_x_sum[i, j], I_xI_y_sum[i, j]],
                        [I_xI_y_sum[i, j], I_yI_y_sum[i, j]],
                    ]
                )
                ATA_inv[i, j] = np.nan_to_num(np.linalg.inv(ATA[i, j]))

        # AT_A = jnp.stack([I_xI_x_sum, I_xI_y_sum, I_xI_y_sum, I_yI_y_sum], axis=-1).reshape(
        #     H, W, 2, 2
        # )
        # dangerous_inv = jnp.linalg.inv(AT_A)
        # if verbose:
        #     num_nan = jnp.sum(jnp.isnan(dangerous_inv))
        #     print(f"Number of NaNs in AT_A_inv: {num_nan} of {dangerous_inv.size}")
        # AT_A_inv = jnp.nan_to_num(dangerous_inv)
        # ATB = -jnp.stack([I_xI_t_sum, I_yI_t_sum], axis=-1).reshape(H, W, 2)
        u_flow = jnp.einsum("ijkl,ijl->ijk", ATA_inv, ATB)

    if not usePatches:
        H, W = img0.shape
        print("H,W: ", H, W)
        u_flow = np.zeros((H // kernel_size[0], W // kernel_size[1], 2))
        for i in range(0, H, kernel_size[0]):
            for j in range(0, W, kernel_size[1]):
                dy = kernel_size[0]
                dx = kernel_size[1]
                i_end = i + dy
                j_end = j + dx
                Ix_in_win = I_x[i:i_end, j:j_end].flatten()
                Iy_in_win = I_y[i:i_end, j:j_end].flatten()
                It_in_win = I_t[i:i_end, j:j_end].flatten()
                B = np.zeros((Ix_in_win.shape[0], 2))
                B[:, 0] = Ix_in_win
                B[:, 1] = Iy_in_win
                dP = -np.linalg.inv(B.T @ B) @ B.T @ It_in_win
                print(dP.shape)
                i_uflow = i // kernel_size[0]
                j_uflow = j // kernel_size[1]
                print(i_uflow, j_uflow)
                print(u_flow.shape)
                u_flow[i_uflow, j_uflow, :] = dP
                # ä Extracat window
                # Ix_in_win = Ix(i-w:i+w, j-w:j+w);
                # Iy_in_win = Iy(i-w:i+w, j-w:j+w);
                # It_in_win  = It(i-w:i+w, j-w:j+w);
                # Ix_in_win = Ix_in_win(:);
                # Iy_in_win = Iy_in_win(:);
                # It_in_win = It_in_win(:);

                # % Calculate the flow
                # B = [Ix_in_win Iy_in_win];
                # dP = -inv(B'*B) * B' * It_in_win;
                # du(row,col)=dP(1);
                # dv(row,col)=dP(2);
    return u_flow


def test_lucas_canade_flow(test_farneback=False):
    import numpy as np

    src = np.random.random((14, 14)) * 255
    img0 = src[:12, :12]
    img1 = src[2:, 2:]
    # img0[1, 1] = 255
    # img1[1, 2] = 255
    kernel_size = (4, 4)
    u_flow = lucas_canade_flow(img0, img1, kernel_size=kernel_size, verbose=True)
    u_flow_naive = lucas_canade_flow_naive(
        img0, img1, kernel_size=kernel_size, verbose=True
    )
    u_flow_naive_ownGrad = lucas_canade_flow_naive(
        img0, img1, kernel_size=kernel_size, verbose=True, useOwnGrad=True
    )
    u_flow_naive_no_patches = lucas_canade_flow_naive(
        img0,
        img1,
        kernel_size=kernel_size,
        verbose=True,
        useOwnGrad=False,
        usePatches=False,
    )
    print(u_flow_naive_ownGrad.shape)
    print("uflow_naive_ownGrad_x: ", u_flow_naive_ownGrad[:, :, 0])
    print("uflow_naive_ownGrad_y: ", u_flow_naive_ownGrad[:, :, 1])
    print(u_flow.shape)
    print("uflow_x: ", u_flow[:, :, 0])
    print("uflow_y: ", u_flow[:, :, 1])
    print(u_flow_naive.shape)
    print("uflow_naive_x: ", u_flow_naive[:, :, 0])
    print("uflow_naive_y: ", u_flow_naive[:, :, 1])

    print(u_flow_naive_no_patches.shape)
    print("uflow_naive_noPatches_x: ", u_flow_naive_no_patches[:, :, 0])
    print("uflow_naive_noPatches_y: ", u_flow_naive_no_patches[:, :, 1])
    # print(u_flow)
    if test_farneback:
        img0 = src[:10, :10] * 255
        img1 = src[1:, 1:] * 255
        u_flow_farneback = cv2.calcOpticalFlowFarneback(
            img0,
            img1,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=1,
        )
        # round flow to 0.01
        u_flow_farneback = np.round(u_flow_farneback, 2)

        print("uflow_farneback: ", u_flow_farneback.shape)
        print("uflow_farneback_x: ", u_flow_farneback[:, :, 0])
        print("uflow_farneback_y: ", u_flow_farneback[:, :, 1])


if __name__ == "__main__":
    test_lucas_canade_flow()
    # test_patchified_sum_pooling()
