from em_stitch.utils.utils import src_from_xy
from em_stitch.lens_correction.utils import (
        maps_from_tform,
        split_inverse_tform,
        remove_weighted_matches,
        estimate_stage_affine)
import renderapi
import numpy as np
import copy


def test_remove_weighted():
    npts = 100
    match = {
            'pId': 'a',
            'qId': 'b',
            'pGroupId': 'c',
            'qGroupId': 'd',
            'matches': {
                'p': (np.random.rand(2, npts) * 1000).tolist(),
                'q': (np.random.rand(2, npts) * 1000).tolist(),
                'w': np.random.randint(0, 2, npts).astype('float').tolist()
                }}
    orig = copy.deepcopy(match)
    remove_weighted_matches([match])
    assert np.all(np.isclose(match['matches']['w'], 1.0))
    keep = np.argwhere(
            np.isclose(np.array(orig['matches']['w']), 1.0)).flatten()
    assert keep.size == len(match['matches']['w'])

    for pq in ['p', 'q']:
        pqorig = np.array(orig['matches'][pq])[:, keep]
        pqfilt = np.array(match['matches'][pq])
        assert np.all(pqorig == pqfilt)


def test_estimate_affine():
    t0locs = np.random.rand(3, 9) * 1000
    t0locs[2, :] = 1.0
    M = np.array([
        [1.1, 0.02, 134],
        [-0.01, 1.02, -434],
        [0.0, 0.0, 1.0]])
    t1locs = M.dot(t0locs)
    t0specs = []
    t1specs = []
    for i in range(t0locs.shape[1]):
        t0specs.append(renderapi.tilespec.TileSpec(
                tforms=[
                    renderapi.transform.AffineModel(
                        B0=t0locs[0, i],
                        B1=t0locs[1, i])]))
        # t1specs have a lens correction transform
        t1specs.append(renderapi.tilespec.TileSpec(
                tforms=[
                    renderapi.transform.AffineModel(),
                    renderapi.transform.AffineModel(
                        B0=t1locs[0, i],
                        B1=t1locs[1, i])]))
    e = estimate_stage_affine(t0specs, t1specs)
    assert np.all(np.isclose(e.M, M))


def test_split_inverse():
    # the function just splits things up
    # so one doesn't run out of memory,
    # typically a problem with ThinPlateSpline
    tf = renderapi.transform.AffineModel()
    tf.M = np.array([
        [1.1, 0.02, 134],
        [-0.01, 1.02, -434],
        [0.0, 0.0, 1.0]])

    src = np.random.rand(1000000, 2) * 1000

    dst = tf.inverse_tform(src)
    dst_split_1 = split_inverse_tform(tf, src, block_size=2000000)
    assert np.all(dst_split_1 == dst)

    dst_split_2 = split_inverse_tform(tf, src, block_size=2000)
    assert np.all(dst_split_2 == dst)


def test_make_maps():
    # identity transform
    width = height = 1000
    tform = renderapi.transform.AffineModel()
    map1, map2, mask = maps_from_tform(
            tform, width, height)
    x = np.arange(width)
    y = np.arange(height)
    xt, yt = np.meshgrid(x, y)
    assert np.all(np.isclose(map1, xt))
    assert np.all(np.isclose(map2, yt))
    assert np.all(mask == 255)

    # something more interesting
    tform.M = np.array([
        [1.1, 0.02, 134],
        [-0.01, 1.02, -434],
        [0.0, 0.0, 1.0]])
    # the following looks easy but for ThinPlateSpline
    # and larger images, it blows up
    # so, the function we're testing does it better
    src = src_from_xy(x, y)
    dst = tform.inverse_tform(src)
    dx = dst[:, 0].reshape((y.size, x.size))
    dy = dst[:, 1].reshape((y.size, x.size))

    # check that the results match the easy way
    map1, map2, mask = maps_from_tform(
            tform, width, height)
    assert np.all(np.isclose(map1, dx))
    assert np.all(np.isclose(map2, dy))
    assert np.any(mask == 0)
