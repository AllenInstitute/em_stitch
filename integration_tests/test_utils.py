from em_stitch.utils.utils import correction_grid
import json
import renderapi
import os
import glob
import numpy as np

test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')


def test_correction_grid():
    data_dir = os.path.join(test_files_dir, "montage_example")
    meta = glob.glob(os.path.join(
        data_dir, '_metadata*.json'))[0]
    with open(meta, 'r') as f:
        j = json.load(f)
    jtform = j[2]['sharedTransform']

    npts = 20
    src, dst = correction_grid(jtform, npts=npts)
    delta = dst - src
    mag = np.linalg.norm(delta, axis=1)
    assert mag.size == npts**2

    tf = renderapi.transform.ThinPlateSplineTransform(json=jtform)
    src, dst = correction_grid(tf, npts=npts)
    delta = dst - src
    mag = np.linalg.norm(delta, axis=1)
    assert mag.size == npts**2
