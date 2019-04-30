from matplotlib.backends.backend_pdf import PdfPages
from argschema import ArgSchemaParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import renderapi
import numpy as np
import json
from EMaligner import jsongz
from em_stitch.plots.schemas import MontagePlotsSchema
import os

example = {
        "collection_path": "/data/em-131fs3/lctest/T4.2019.04.29b/001738/0/collection.json.gz",
        "resolved_path": "/data/em-131fs3/lctest/T4.2019.04.29b/001738/0/resolvedtiles_AffineModel_0.json.gz"
        }


def tspec_transform(tspec, mpq, shared=None):
    xy = np.array(mpq).transpose()
    for tf in tspec.tforms:
        if isinstance(tf, renderapi.transform.ReferenceTransform):
            # matches already in LC units
            continue
        else:
            xy = tf.tform(xy)
    return xy


def make_xyres(matches, resolved):
    tids = np.array([t.tileId for t in resolved.tilespecs])
    xy = []
    res = []
    mxy = []
    mres = []
    for m in matches:
        pind = np.argwhere(m['pId'] == tids).flatten()
        qind = np.argwhere(m['qId'] == tids).flatten()
        if (pind.size != 1) & (qind.size != 1):
            continue
        pxy = tspec_transform(
                resolved.tilespecs[pind[0]],
                m['matches']['p'],
                shared=resolved.transforms)
        qxy = tspec_transform(
                resolved.tilespecs[qind[0]],
                m['matches']['q'],
                shared=resolved.transforms)
        w = np.array(m['matches']['w']) != 0.0
        xy.append(0.5 * (pxy[w] + qxy[w]))
        res.append(pxy[w] - qxy[w])
        w = np.invert(w)
        mxy.append(0.5 * (pxy[w] + qxy[w]))
        mres.append(pxy[w] - qxy[w])
    return (np.concatenate(xy), np.concatenate(res),
            np.concatenate(mxy), np.concatenate(mres))


def one_plot(f, ax, xy, c, vmin=-100, vmax=100, title=None, colorbar=True,fontsize=10):
    col = c
    if vmin == vmax:
        col = 'k'
    s = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=col,
            marker='s',
            s=2.5,
            vmin=vmin,
            vmax=vmax)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.5)
    if colorbar:
        f.colorbar(s, cax=cax)
    else:
        cax.axis('off')
    ax.set_title(title, fontsize=fontsize)


class MontagePlots(ArgSchemaParser):
    default_schema = MontagePlotsSchema

    def run(self):
        matches = jsongz.load(self.args['collection_path'])
        resolved = renderapi.resolvedtiles.ResolvedTiles(
                json=jsongz.load(self.args['resolved_path']))

        xy, res, mxy, mres = make_xyres(matches, resolved)
        if self.args['save_json_path']:
            with open(self.args['save_json_path'], 'w') as f:
                json.dump(
                        {
                            'xy': xy.tolist(),
                            'res': res.tolist(),
                            'filtered_xy': mxy.tolist(),
                            'filtered_res': mres.tolist()
                            }, f, indent=2)

        pdf = None 
        if self.args['pdf_out']:
            pdf = PdfPages(self.args['pdf_out'])
        if self.args['make_plot']:
            f, a = plt.subplots(
                    2, 2, clear=True, num=1,
                    sharex=True, sharey=True,
                    figsize=(12, 12))
            vmnmx = 5
            title = self.args['collection_path']
            title += '\n' + self.args['resolved_path'] + '\n'
            one_plot(
                    f, a[0, 0], xy, res[:, 0],
                    vmin=-vmnmx, vmax=vmnmx,
                    title=title+'x res [px]', fontsize=8)
            one_plot(
                    f, a[0, 1], xy, res[:, 1],
                    vmin=-vmnmx, vmax=vmnmx, title='y res [px]')
            one_plot(
                    f, a[1, 0], xy, np.linalg.norm(res, axis=1),
                    vmin=0, vmax=vmnmx*np.sqrt(2), title='mag res [px]')
            one_plot(
                    f, a[1, 1], mxy, np.linalg.norm(mres, axis=1),
                    vmin=0, vmax=vmnmx*np.sqrt(2), title='filtered')
            a[0, 0].invert_yaxis()
        if self.args['show']:
            plt.show()
        if pdf:
            pdf.savefig(f)
            pdf.close()
            print('wrote %s' % os.path.abspath(self.args['pdf_out']))


if __name__ == "__main__":
    mmod = MontagePlots(input_data=example)
    mmod.run()
