from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from lens_correction.utils import maps_from_tform
from EMaligner import jsongz
import renderapi
import cv2
from em_stitch.plots.schemas import ViewMatchesSchema
from argschema import ArgSchemaParser

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

    f, a = plt.subplots(1, 2, clear=True, num=fignum, figsize=(11, 8))

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
    return f, a


example = {
        "data_dir": "/data/em-131fs3/lctest/T3_OL5pct/005270/0"
        }


class ViewMatches(ArgSchemaParser):
    default_schema = ViewMatchesSchema

    def run(self):
        cpath = self.args.get(
                'collection_path',
                os.path.join(
                    self.args['data_dir'],
                    self.args['collection_basename']))
        if (
                (not os.path.isfile(cpath)) &
                (os.path.splitext(cpath)[-1] == '.json')):
            cpath += '.gz'

        self.matches = jsongz.load(cpath)
        self.get_transform()

        if self.args['view_all']:
            inds = np.arange(len(self.matches))
        else:
            inds = [self.args['match_index']]

        with PdfPages(self.args['pdf_out']) as pdf:
            for ind in inds:
                pim, qim, p, q, w, pname, qname = get_ims_and_coords(
                        self.matches[ind],
                        self.args['data_dir'])
                f, a = plot_ims_and_coords(
                        pim, qim, p, q, w, pname=pname,
                        qname=qname, tform=self.tform, fignum=2)
                pdf.savefig(f)
                if self.args['show']:
                    plt.show()

        print('wrote %s' % os.path.abspath(self.args['pdf_out']))

    def get_transform(self):
        self.tform = None
        if 'transform_file' in self.args:
            self.tform = renderapi.transform.ThinPlateSplineTransform(
                    json=jsongz.load(self.args['transform_file']))
        else:
            for fbase in self.args['resolved_tiles']:
                fpath = os.path.join(
                        self.args['data_dir'],
                        fbase)
                if os.path.isfile(fpath):
                    resolved = renderapi.resolvedtiles.ResolvedTiles(
                            json=jsongz.load(fpath))
                    self.tform = resolved.transforms[0]
                    break


if __name__ == '__main__':
    vmod = ViewMatches(input_data=example)
    vmod.run()
