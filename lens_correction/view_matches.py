import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.cm as cm
from meta_to_collection import MetaToCollection, main


def get_ims_and_coords(m, ddir):
    pname = os.path.join(ddir, m['pId'] + '.tif')
    qname = os.path.join(ddir, m['qId'] + '.tif')
    pim = plt.imread(pname, 0)
    qim = plt.imread(qname, 0)
    p = np.array(m['matches']['p']).transpose()
    q = np.array(m['matches']['q']).transpose()
    return pim, qim, p, q, pname, qname


def plot_ims_and_coords(pim, qim, p, q, pname=None, qname=None, fignum=1):
    f, a = plt.subplots(1, 2, clear=True, num=fignum)

    a[0].imshow(pim, cmap='gray')
    a[0].scatter(p[:, 0], p[:, 1], marker='x', color='r')
    a[0].set_title(pname, fontsize=8)

    a[1].imshow(qim, cmap='gray')
    a[1].scatter(q[:, 0], q[:, 1], marker='x', color='r')
    a[1].set_title(qname, fontsize=8)
    return

ddir = '/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction8/000000/0'
collection = os.path.join(ddir, 'test_collection.json')
main([ddir, '-o', collection])

mcp = glob.glob(collection)[0]
with open(mcp, 'r') as f:
    matches = json.load(f)

pim, qim, p, q, pname, qname = get_ims_and_coords(matches[84], ddir)
plot_ims_and_coords(pim, qim, p, q, pname=pname, qname=qname)
plt.show()
