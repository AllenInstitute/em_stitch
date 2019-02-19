import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import matplotlib.cm as cm


def get_ims_and_coords(m, ddir):
    pim = plt.imread(os.path.join(ddir, m['pId'] + '.tif'), 0)
    qim = plt.imread(os.path.join(ddir, m['qId'] + '.tif'), 0)
    p = np.array(m['matches']['p']).transpose()
    q = np.array(m['matches']['q']).transpose()
    return pim, qim, p, q


def plot_ims_and_coords(pim, qim, p, q, fignum=1):
    f, a = plt.subplots(1, 2, clear=True, num=fignum)
    a[0].imshow(pim, cmap='gray')
    a[1].imshow(qim, cmap='gray')
    a[0].scatter(p[:, 0], p[:, 1], marker='x', color='r')
    a[1].scatter(p[:, 0], p[:, 1], marker='x', color='r')
    return


ddir = '/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction8/000000/0'
mcp = glob.glob(os.path.join(ddir, 'montage_collection.json'))[0]
with open(mcp, 'r') as f:
    matches = json.load(f)

pim, qim, p, q = get_ims_and_coords(matches[87], ddir)
plot_ims_and_coords(pim, qim, p, q)
plt.show()
