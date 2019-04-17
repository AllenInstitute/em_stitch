import numpy as np
import cv2


def src_from_xy(x, y, transpose=True):
    xt, yt = np.meshgrid(x, y)
    src = np.vstack((xt.flatten(), yt.flatten())).astype('float')
    if not transpose:
        return src
    return src.transpose()


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
