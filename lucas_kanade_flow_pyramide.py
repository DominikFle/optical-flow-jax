import jax.numpy as jnp
import jax
import jax.scipy as jsp
from lucas_kanade_flow import lucas_canade_flow


def lucas_kanade_flow_pyramide(
    img0: jnp.ndarray, img1: jnp.ndarray, pyramide_levels=3, kernel_size=(3, 3)
):
    # Create the pyramide
    pyramide = {1: (img0, img1)}  # zoomlevel : (img0,img1)
    for i in range(1, pyramide_levels):
        zoom_level = i * 2
        img0 = jax.image.resize(
            pyramide[i][0],
            (pyramide[i][0].shape[0] // 2, pyramide[i][0].shape[1] // 2),
            "linear",
        )
        img1 = jax.image.resize(
            pyramide[i][1],
            (pyramide[i][1].shape[0] // 2, pyramide[i][1].shape[1] // 2),
            "linear",
        )
        pyramide[zoom_level] = (img0, img1)

    # Perform the lucas kanade flow on the pyramide and warp image of the next level and iterate
    u_flow = None
    for zoom_level in reversed(sorted(pyramide.keys())):
        img0, img1 = pyramide[zoom_level]
        if u_flow is not None:
            # Warp the image of the next level
            # test = jsp.interpolate.RegularGridInterpolator() use this for warping
            img0_warped = warp  #
        else:
            img0_warped = img0  # first level

        u_flow = lucas_canade_flow(
            img0_warped, img1, kernel_size=kernel_size
        )  # u_flow: (H/Kh,W/Kw,2)
        # upsample the flow field to the pyramide above (2*H,2*W,2)
        u_flow = jax.image.resize(u_flow, (img0.shape[0], img0.shape[1]), "linear")
