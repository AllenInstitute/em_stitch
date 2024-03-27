import time
import logging

import cv2
import numpy as np
from scipy import ndimage

import renderapi

from ..utils import utils as common_utils

logger = logging.getLogger(__name__)


def split_inverse_tform(tform, src, block_size):
    """
    Split and inverse transform the source array using the provided transformation.

    Parameters
    ----------
    tform : Any
        Transformation object.
    src : np.ndarray
        Source array.
    block_size : int
        Size of each block.

    Returns
    -------
    np.ndarray
        Inverse transformed array.
    """
    nsplit = np.ceil(float(src.shape[0]) / float(block_size))
    split_src = np.array_split(src, nsplit, axis=0)
    dst = []
    for s in split_src:
        dst.append(tform.inverse_tform(s))
    dst = np.concatenate(dst)
    return dst


def maps_from_tform(tform, width, height, block_size=10000, res=32):
    """
    Generate maps and a mask for remapping based on the provided transformation.

    Parameters
    ----------
    tform : Any
        Transformation object.
    width : int
        Width of the map.
    height : int
        Height of the map.
    block_size : int, optional
        Size of each block, by default 10000.
    res : int, optional
        cell resolution, by default 32.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing map1, map2, and mask arrays.

    """
    t0 = time.time()

    x = np.arange(0, width + res, res)
    y = np.arange(0, height + res, res)
    src = common_utils.src_from_xy(x, y)
    idst = split_inverse_tform(tform, src, block_size)
    ix = idst[:, 0].reshape(y.size, x.size)
    iy = idst[:, 1].reshape(y.size, x.size)

    fx = np.arange(0, width)
    fy = np.arange(0, height)
    src = np.flipud(common_utils.src_from_xy(
        fx, fy, transpose=False).astype('float32'))
    src[0, :] *= (float(ix.shape[0] - 1) / y.max())
    src[1, :] *= (float(ix.shape[1] - 1) / x.max())

    dx = ndimage.map_coordinates(ix, src, order=1)
    dy = ndimage.map_coordinates(iy, src, order=1)

    map1 = dx.reshape((fy.size, fx.size)).astype('float32')
    map2 = dy.reshape((fy.size, fx.size)).astype('float32')

    # actually do it, to find a mask
    mask = np.ones_like(map1)
    mask = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST)
    mask = np.uint8(mask * 255)
    t1 = time.time()
    logger.info(" created maps for remap:\n  took %0.1f seconds" % (t1 - t0))
    return map1, map2, mask


def estimate_stage_affine(t0, t1):
    """
    Estimate affine transformation between two sets of translations.

    Parameters
    ----------
    t0 : Iterable
        List of transformations (t0).
    t1 : Iterable
        List of transformations (t1).

    Returns
    -------
    renderapi.transform.AffineModel
        Estimated affine transformation.

    """
    src = np.array([t.tforms[0].translation for t in t0])
    dst = np.array([t.tforms[1].translation for t in t1])
    aff = renderapi.transform.AffineModel()
    aff.estimate(src, dst)
    return aff


def remove_weighted_matches(matches, weight=0.0):
    """
    Remove matches with specified weight.

    Parameters
    ----------
    matches : List[Dict[str, Any]]
        List of matches.
    weight : float, optional
        Weight threshold, by default 0.0.
    """
    for m in matches:
        ind = np.invert(np.isclose(np.array(m['matches']['w']), weight))
        m['matches']['p'] = np.array(m['matches']['p'])[:, ind].tolist()
        m['matches']['q'] = np.array(m['matches']['q'])[:, ind].tolist()
        m['matches']['w'] = np.array(m['matches']['w'])[ind].tolist()
    return
