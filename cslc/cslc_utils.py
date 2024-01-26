import numpy as np
from shapely.geometry import box
from scipy.signal import convolve


def bounds(img):
    """
    Returns shapely box with image bounds
    """
    return box(
        img.geotrans[0],
        img.geotrans[3] + img.geotrans[5] * img.y_pixels,
        img.geotrans[0] + img.geotrans[1] * img.x_pixels,
        img.geotrans[3]
    )


def group_into_stacks(imgs):
    """
    Group set of images into burst stacks - eliminate duplicates
    """
    stack = {}
    for ii in sorted(imgs, key=lambda x: x.name):
        if ii.provider_id not in stack:
            stack[ii.provider_id] = {}
        burst_stack = stack[ii.provider_id]
        if ii.acquired.date() not in burst_stack:
            burst_stack[ii.acquired.date()] = ii
    return stack


def get_kernel(size, num_conv):
    assert type(num_conv) == int, "num_conv must be an integer"
    k0 = np.ones((size, size))
    k = k0
    for i in range(num_conv):
        if i > 3:
            k = convolve(k, k0, mode="same")
        else:
            k = convolve(k, k0)
    k /= np.sum(k)

    return k.astype(np.float32)


def smoothed_dot_prod(z1, z2, k, real=False):
    """
    For each pixel compute the dot product of the two
    scenes within the weighted neighborhood given by k

    Parameters
    ----------
    z1 : array
        First complex scene
    z2 : array
        Second complex scene
    k : array
        2D kernel for smoothing

    Returns
    -------
    zz : array
        Smoothed dot product.
    """
    if not real:
        return convolve(z1 * np.conj(z2), k, mode="same")
    else:
        return convolve(np.abs(z1) * np.abs(z2), k, mode="same")
