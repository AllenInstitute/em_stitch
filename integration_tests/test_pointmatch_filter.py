from em_stitch.utils.utils import pointmatch_filter, src_from_xy
import renderapi
import numpy as np
from shapely.geometry import Polygon, Point
from skimage.transform import PolynomialTransform as skPoly


def dummy_tilespec(width=1000, height=1000, transforms=[]):
    t = renderapi.tilespec.TileSpec()
    t.width = width
    t.height = height
    t.tforms = transforms
    return t


def random_affine(da=0.02, do=100):
    return renderapi.transform.AffineModel(
            M00=np.random.randn() * da + 1,
            M11=np.random.randn() * da + 1,
            M10=np.random.randn() * da,
            M01=np.random.randn() * da,
            B0=np.random.randn() * do,
            B1=np.random.randn() * do)


def random_polynomial(dparams=[100, 0.02, 0.00005]):
    params = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]]).astype('float')
    params[:, 0] += np.random.randn(2) * dparams[0]
    params[:, 1:3] += np.random.randn(2, 2) * dparams[1]
    params[:, 3:] += np.random.randn(2, 3) * dparams[2]
    return renderapi.transform.Polynomial2DTransform(
            params=params)


def inverse(tspec, src):
    if isinstance(tspec.tforms[0], renderapi.transform.AffineModel):
        itf = tspec.tforms[0].invert()
    elif isinstance(
            tspec.tforms[0], renderapi.transform.Polynomial2DTransform):
        x = np.linspace(0, tspec.width, 10)
        y = np.linspace(0, tspec.height, 10)
        src = src_from_xy(x, y)
        dst = tspec.tforms[0].tform(src)
        # renderapi doesn't have a good estimator, use skimage
        invtf = skPoly()
        invtf.estimate(dst, src, order=2)
        itf = renderapi.transform.Polynomial2DTransform(
                params=invtf.params)
    return itf.tform(src).transpose().tolist()


def random_transform(tftype):
    if tftype == 'affine':
        return random_affine()
    elif tftype == 'polynomial':
        return random_polynomial()


def dummy_match(npts=100, tform_type='affine'):
    t0 = dummy_tilespec(transforms=[random_transform(tform_type)])
    t1 = dummy_tilespec(transforms=[random_transform(tform_type)])
    p0 = Polygon(t0.bbox_transformed(ndiv_inner=5))
    p1 = Polygon(t1.bbox_transformed(ndiv_inner=5))
    overlap = p0.intersection(p1)
    pts = []
    ow = overlap.bounds[2] - overlap.bounds[0]
    oh = overlap.bounds[3] - overlap.bounds[1]
    while len(pts) < npts:
        pt = Point(
                np.random.rand() * ow + overlap.bounds[0],
                np.random.rand() * oh + overlap.bounds[1])
        if overlap.contains(pt):
            pts.append(pt)
    src = np.array([np.array(p) for p in pts])
    match = {}
    for k in ['pId', 'qId', 'pGroupId', 'qGroupId']:
        match[k] = k
    match['matches'] = {
            'p': inverse(t0, src),
            'q': inverse(t1, src),
            'w': [1.0] * npts
            }
    return match


def test_one_cluster_filter():
    # check the size and shape of things
    m = dummy_match()
    p, q, w, labels = pointmatch_filter(m, n_clusters=1)
    unique_labels = np.unique(labels)
    assert np.all(p.shape == q.shape)
    assert p.shape[0] == w.size
    assert p.shape[0] == len(m['matches']['p'][0])
    assert q.shape[0] == len(m['matches']['q'][0])
    assert w.size == len(m['matches']['w'])
    assert unique_labels.size == 1
    assert np.all(np.isclose(w, 1.0))

    # make a small change to the match
    m['matches']['p'][0][12] += 3.0
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, ransacReprojThreshold=10.0)
    assert np.all(np.isclose(w, 1.0))

    # make a larger change that should be filtered
    m['matches']['p'][0][12] += 20.0
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, ransacReprojThreshold=10.0)
    assert np.count_nonzero(np.isclose(w, 0.0)) == 1

    # make a really small match, which will be filtered because of size
    m = dummy_match(npts=5)
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, n_min_ignore=5)
    assert np.all(np.isclose(w, 0.0))

    # example is exact affine, so we can dial down the ransac parameter
    m = dummy_match()
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, ransacReprojThreshold=0.1)
    assert np.all(np.isclose(w, 1.0))

    # but with Similarity instead of Affine, will not match all
    m = dummy_match(npts=100)
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, model='Similarity', ransacReprojThreshold=0.1)
    assert np.any(np.isclose(w, 0.0))

    # and, with underlying non-linearity, affine won't match all either
    m = dummy_match(npts=100, tform_type='polynomial')
    p, q, w, labels = pointmatch_filter(
            m, n_clusters=1, model='Affine', ransacReprojThreshold=0.1)
    assert np.any(np.isclose(w, 0.0))


def test_n_cluster_filter():
    # start with underlying non-linearity
    npts = 100
    m = dummy_match(npts=npts, tform_type='polynomial')

    # in general, the number of inliers should increase
    # with more clusters. There is some randomness in RANSAC
    # so, we'll just check that the trend is such
    ncs = np.arange(1, 6)
    nin = np.zeros_like(ncs)
    for i in range(ncs.size):
        p, q, w, labels = pointmatch_filter(
                m,
                n_clusters=ncs[i],
                model='Affine',
                ransacReprojThreshold=10.0)
        nin[i] = np.count_nonzero(np.isclose(w, 1.0))
    lp = np.polyfit(ncs, nin, 1)
    assert lp[0] > 0

    # above was 100 pts divided into up to 5 clusters
    # or, about 20 pts per cluster
    # we can force it into a smaller number
    # n_cluster_pts
    npts = 100
    nc_ask = 5
    m = dummy_match(npts=npts, tform_type='polynomial')
    p, q, w, labels = pointmatch_filter(
            m,
            n_clusters=nc_ask,
            model='Affine',
            ransacReprojThreshold=10.0,
            n_cluster_pts=25)
    nc_get = np.unique(labels).size
    assert nc_get < nc_ask

    # or, we can set n_clusters=None and have the function decide
    m = dummy_match(npts=npts, tform_type='polynomial')
    p, q, w, labels = pointmatch_filter(
            m,
            n_clusters=None,
            model='Affine',
            ransacReprojThreshold=10.0,
            n_cluster_pts=25)
    nc_get = np.unique(labels).size
    assert nc_get < 5
