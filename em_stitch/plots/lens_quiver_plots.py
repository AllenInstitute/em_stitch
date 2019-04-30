import renderapi
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import os
import re
import warnings
import datetime
from em_stitch.utils.utils import src_from_xy
from em_stitch.plots.schemas import LensQuiverSchema
from argschema import ArgSchemaParser
from EMaligner import jsongz
warnings.simplefilter(action='ignore', category=FutureWarning)

example = {
        "transform_list": [
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429113358_reference/0/resolvedtiles.json.gz",
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429121307_reference/0/resolvedtiles.json.gz",
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429122415_reference/0/resolvedtiles.json.gz",
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429134601_reference/0/resolvedtiles.json.gz",
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429143510_reference/0/resolvedtiles.json.gz",
            "/data/em-131fs3/lctest/T3_OL5_5pct/20190429153110_reference/0/resolvedtiles.json.gz",
            ]
        }


def plot_lens_changes(
        lcs, arrow_scale=10.0, num=1, pdfname='lens_changes.pdf'):
    scale = 1.0 / arrow_scale
    with PdfPages(pdfname) as pdf:
        for lc in lcs:
            tstamp = re.findall('\d+_reference', lc)[0].split('_')[0]
            dt = datetime.datetime.strptime(tstamp, '%Y%m%d%H%M%S')
            dtsf = dt.strftime('%Y-%m-%d %H:%M:%S')
            ddir = os.path.dirname(lc)
            mfile = glob.glob(os.path.join(ddir, '_metadata*.json'))[0]
            with open(mfile, 'r') as f:
                meta = json.load(f)
            obj_focus = meta[0]['metadata']['objective_focus']
            print(obj_focus)
            with open(lc, 'r') as f:
                tf = renderapi.transform.ThinPlateSplineTransform(
                        json=json.load(f))
                sz = tf.srcPts.max(axis=1)
                src = src_from_xy(
                        np.linspace(0, sz[0], 20),
                        np.linspace(0, sz[1], 20))
                dst = tf.tform(src)
                delta = dst - src
                rmax = np.linalg.norm(delta, axis=1).max()
                fig, axes = plt.subplots(
                        1, 1, num=1, clear=True, figsize=(11, 8))
                axes.quiver(
                        src[:, 0], src[:, 1], delta[:, 0], delta[:, 1],
                        angles='xy', scale=scale, scale_units='xy')
                axes.invert_yaxis()
                axes.set_aspect('equal')
                axes.set_title(
                        dtsf + '\n' + lc + "\nfull transform max: %0.1f "
                        "pixels\narrow scale = %0.1f\nobjective focus: %d" %
                        (rmax, arrow_scale, obj_focus))
                pdf.savefig(fig)


def load_transform(path):
    j = jsongz.load(path)

    try:
        return renderapi.transform.ThinPlateSplineTransform(
                json=j)
    except KeyError:
        res = renderapi.resolvedtiles.ResolvedTiles(json=j)
        return res.transforms[0]


def grid_from_tform(tform, xpts=20, ypts=20):
    xmin = tform.srcPts[0, :].min()
    xmax = tform.srcPts[0, :].max()
    ymin = tform.srcPts[1, :].min()
    ymax = tform.srcPts[1, :].max()

    x = np.linspace(xmin, xmax, xpts)
    y = np.linspace(ymin, ymax, ypts)

    return src_from_xy(x, y)


def lens_quiver(ax, tform, src, arrow_scale=1.0, title=None):
    dst = tform.tform(src)
    delta = dst - src

    rmax = np.linalg.norm(delta, axis=1).max()

    scale1 = 1.0 / arrow_scale

    ax.quiver(
            src[:, 0],
            src[:, 1],
            delta[:, 0],
            delta[:, 1],
            angles='xy',
            scale=scale1,
            scale_units='xy')
    if not ax.axes.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect('equal')
    if not title:
        title = ''
    title += "\nfull transform max: %0.1f " % rmax
    title += "pixels\narrow scale = %0.1f" % (1/scale1)
    ax.set_title(title, fontsize=8)


class LensQuiverPlots(ArgSchemaParser):
    default_schema = LensQuiverSchema

    def run(self):
        tforms = [load_transform(p) for p in self.args['transform_list']]

        grid = grid_from_tform(
                tforms[0],
                xpts=self.args['n_grid_pts'],
                ypts=self.args['n_grid_pts'])

        nrow, ncol = self.args['subplot_shape']
        figsize = (10, 10)
        f, a = plt.subplots(
                nrow, ncol, num=self.args['fignum'], figsize=figsize,
                clear=True, sharex=True, sharey=True)
        if not isinstance(a, np.ndarray):
            a = np.array([a])

        iplot = 0
        with PdfPages(self.args['pdf_out']) as pdf:
            for tform, p in zip(tforms, self.args['transform_list']):
                lens_quiver(
                        a[iplot], tform, grid,
                        self.args['arrow_scale'], title=p)
                iplot += 1
                if iplot == (nrow * ncol):
                    pdf.savefig(f)
                    if tform != tforms[-1]:
                        fn = f.number + 1
                        f, a = plt.subplots(
                                nrow, ncol, num=fn, figsize=figsize,
                                clear=True, sharex=True, sharey=True)
                        if not isinstance(a, np.ndarray):
                            a = np.array([a])
                        iplot = 0

        print('wrote %s' % os.path.abspath(self.args['pdf_out']))

        if self.args['show']:
            plt.show()


if __name__ == "__main__":
    qmod = LensQuiverPlots(input_data=example)
    qmod.run()
