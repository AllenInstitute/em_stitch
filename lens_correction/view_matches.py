import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from meta_to_collection import main
import sys
import logging
from lens_correction.lens_correction_solver import maps_from_tform
import renderapi
import cv2

logger = logging.getLogger()


def find_ind(matches, ids):
    # helper to find the index of matches
    # ind = find_ind(matches, ['_71_64', '_72_64'])
    for i in range(len(matches)):
        m = matches[i]
        pq = (ids[0] in m['pId']) & (ids[1] in m['qId'])
        qp = (ids[0] in m['qId']) & (ids[1] in m['pId'])
        if pq | qp:
            ind = i
    return ind


def get_ims_and_coords(m, ddir):
    pname = os.path.join(ddir, m['pId'] + '.tif')
    qname = os.path.join(ddir, m['qId'] + '.tif')
    pim = plt.imread(pname, 0)
    qim = plt.imread(qname, 0)
    p = np.array(m['matches']['p']).transpose()
    q = np.array(m['matches']['q']).transpose()
    w = np.array(m['matches']['w'])
    return pim, qim, p, q, w, pname, qname


def plot_ims_and_coords(
        pim, qim, p, q, w, pname=None, qname=None, fignum=1, tform=None):
    if tform:
        map1, map2, mask = maps_from_tform(
                tform, pim.shape[1], pim.shape[0], logger)
        pim = cv2.remap(pim, map1, map2, cv2.INTER_NEAREST)
        qim = cv2.remap(qim, map1, map2, cv2.INTER_NEAREST)

    f, a = plt.subplots(1, 2, clear=True, num=fignum)

    a[0].imshow(pim, cmap='gray')
    nind = np.isclose(w, 0.0)
    ind = np.invert(nind)
    a[0].scatter(p[ind, 0], p[ind, 1], marker='x', color='r')
    a[0].scatter(p[nind, 0], p[nind, 1], marker='x', color='k')
    a[0].set_title(os.path.basename(pname), fontsize=8)

    a[1].imshow(qim, cmap='gray')
    a[1].scatter(q[ind, 0], q[ind, 1], marker='x', color='r')
    a[1].scatter(q[nind, 0], q[nind, 1], marker='x', color='k')
    a[1].set_title(os.path.basename(qname), fontsize=8)
    return


ddir = '/data/em-131fs3/lctest/T4_07/20190314150250_reference/0'
ddir = '/data/em-131fs3/lctest/T4_07/20190315112830_reference/0'
ddir = '/data/em-131fs3/lctest/T4_07/20190315142205_reference/0'
collection = os.path.join(ddir, 'collection.json')
tform_path = os.path.join(ddir, 'lens_corr_transform.json')
ddir = '/data/em-131fs3/lctest/T4_06/001931/0'
collection = os.path.join(ddir, 'match_collection.json')
tform_path = os.path.join(ddir, 'lens_corr_transform.json')
tform = None
if os.path.isfile(tform_path):
    with open(tform_path, 'r') as f:
        tformj = json.load(f)
    tform = renderapi.transform.ThinPlateSplineTransform(json=tformj)

if not os.path.isfile(collection):
    main([ddir, '-o', collection])

mcp = glob.glob(collection)[0]
with open(mcp, 'r') as f:
    matches = json.load(f)

if len(sys.argv) > 1:
    try:
        ind = int(sys.argv[1])
    except TypeError as e:
        print('not an integer', str(e))

print('using index %d' % ind)

pim, qim, p, q, w, pname, qname = get_ims_and_coords(matches[ind], ddir)
plot_ims_and_coords(
        pim, qim, p, q, w, pname=pname, qname=qname, tform=tform, fignum=2)
plt.show()
