import numpy as np
import cv2
import renderapi
import time
from scipy import ndimage
import json


def get_z_from_metafile(metafile):
    offsets = [
            {
              "load": "Tape147",
              "offset": 100000
            },
            {
              "load": "Tape148",
              "offset": 110000
            },
            {
              "load": "Tape148B",
              "offset": 110000
            },
            {
              "load": "Tape148A",
              "offset": 110000
            },
            {
              "load": "Tape149",
              "offset": 120000
            },
            {
              "load": "Tape151",
              "offset": 130000
            },
            {
              "load": "Tape162",
              "offset": 140000
            },
            {
              "load": "Tape127",
              "offset": 150000
            }]

    loads = np.array([i['load'] for i in offsets])

    with open(metafile, 'r') as f:
        j = json.load(f)
    try:
        tape = int(j[0]['metadata']['media_id'])
        offset = offsets[
                np.argwhere(loads == 'Tape%d' % tape).flatten()[0]]['offset']
    except ValueError:
        offset = 0
    grid = int(j[0]['metadata']['grid'])
    return offset + grid


def split_inverse_tform(tform, src, block_size):
    nsplit = np.ceil(float(src.shape[0]) / float(block_size))
    split_src = np.array_split(src, nsplit, axis=0)
    dst = []
    for s in split_src:
        dst.append(tform.inverse_tform(s))
    dst = np.concatenate(dst)
    return dst


def maps_from_tform(tform, width, height, logger, block_size=10000, res=32):
    t0 = time.time()

    x = np.arange(0, width + res, res)
    y = np.arange(0, height + res, res)
    src = src_from_xy(x, y)
    idst = split_inverse_tform(tform, src, block_size)
    ix = idst[:, 0].reshape(y.size, x.size)
    iy = idst[:, 1].reshape(y.size, x.size)

    fx = np.arange(0, width)
    fy = np.arange(0, height)
    src = np.flipud(src_from_xy(fx, fy, transpose=False).astype('float32'))
    src[0, :] *= ((ix.shape[0] - 1) / y.max())
    src[1, :] *= ((ix.shape[1] - 1) / x.max())

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


def src_from_xy(x, y, transpose=True):
    xt, yt = np.meshgrid(x, y)
    src = np.vstack((xt.flatten(), yt.flatten())).astype('float')
    if not transpose:
        return src
    return src.transpose()


def estimate_stage_affine(t0, t1):
    src = np.array([t.tforms[0].translation for t in t0])
    dst = np.array([t.tforms[1].translation for t in t1])
    aff = renderapi.transform.AffineModel()
    aff.estimate(src, dst)
    return aff


def pointmatch_filter(
        match, n_clusters=None, ransacReprojThreshold=10,
        n_cluster_pts=15, n_min_ignore=3, model='Affine'):
    """filter point matches via Similarity Model with
       local clustering, if specified.

    Parameters
    ----------
    match: dict
        pointmatch dict
    n_clusters: int
        number of clusters. If None, will be set by n_cluster_pts
    ransacReprojThreshold: float
        passed to cv2.estimateAffinePartial2D
    n_cluster_pts: int
        sets number of clusters by points. If n_cluster is specified,
        determines stopping criteria for finding clusters.
    n_min_ignore: int
        if the number of inliers <= this setting, all weights set to zero
        for the cluster

    Returns
    -------
    p: numpy array
        N x 2 array from match['matches']['p']
    q: numpy array
        N x 2 array from match['matches']['q']
    w: numpy array
        1.0/0.0 for inliers/outliers. Can be used in pointmatch
        dict match['matches']['w'] = w.tolist()
    labels: numpy array
        labels from kmeans clustering
    """

    p = np.array(match['matches']['p']).transpose().astype('float32')
    q = np.array(match['matches']['q']).transpose().astype('float32')

    # find some clusters with a minimum number of points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    npts = len(match['matches']['w'])
    if n_clusters is None:
        n_clusters = npts // n_cluster_pts
    while True:
        _, labels, _ = cv2.kmeans(
                p,
                n_clusters,
                None,
                criteria,
                10,
                cv2.KMEANS_RANDOM_CENTERS)
        ulab, cnts = np.unique(labels, return_counts=True)
        if np.all(cnts >= n_cluster_pts) | (n_clusters == 1):
            break
        n_clusters -= 1

    labels = labels.flatten()

    if model == 'Affine':
        mfun = cv2.estimateAffine2D
    elif model == 'Similarity':
        mfun = cv2.estimateAffinePartial2D

    # run RANSAC on each cluster and track inliers
    w = np.zeros(npts)
    for u in ulab:
        cind = np.argwhere(labels == u).flatten()
        _, b = mfun(
                p[cind],
                q[cind],
                ransacReprojThreshold=ransacReprojThreshold)
        b = b.flatten()
        if np.count_nonzero(b) > n_min_ignore:
            w[cind[b != 0]] = 1.0

    return p, q, w, labels


def remove_weighted_matches(matches, weight=0.0):
    for m in matches:
        ind = np.invert(np.isclose(np.array(m['matches']['w']), weight))
        m['matches']['p'] = np.array(m['matches']['p'])[:, ind].tolist()
        m['matches']['q'] = np.array(m['matches']['q'])[:, ind].tolist()
        m['matches']['w'] = np.array(m['matches']['w'])[ind].tolist()
    return
