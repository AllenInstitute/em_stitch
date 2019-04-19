import renderapi
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import os
import logging
import re
import warnings
import datetime
from .utils import estimate_stage_affine
from ..utils.utils import src_from_xy
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def plot_residual_histograms(monbase, zr, axes_shape, num=1):
    fig, axes = plt.subplots(
            axes_shape[0], axes_shape[1], clear=True,
            num=num, sharex=True, sharey=True)
    bins = np.arange(12)
    iplot = 0
    counts = []
    zr = np.array(zr)
    for z in zr:
        mondir = os.path.join(monbase, "%06d/0" % z)
        res_path = os.path.join(mondir, 'residuals.json')
        print(res_path)
        if os.path.isfile(res_path):
            print(res_path)
            with open(res_path, 'r') as f:
                j = json.load(f)
            resmag = np.linalg.norm(np.array(j['res']), axis=1)

            counts.append({
                'z': z,
                'n_good': len(j['res']),
                'n_bad': len(j['filtered_res'])
                })

            print(z)
            print(np.mod(np.argwhere(zr == z), 9))
            if np.mod(np.argwhere(zr == z), 9) == 0:
                resmag0 = resmag
                zr[0] = z
                print(zr[0])
            a = axes.flatten()[iplot]
            a.hist(resmag, bins=bins, color='g', label=('%d' % z))
            a.hist(
                    resmag0, bins=bins, histtype='step',
                    color='k', label=('%d' % zr[0]))
            a.legend(fontsize=6)
            a.set_title('mean = %0.2f px' % resmag.mean())
            iplot += 1
    last_row = axes.shape[0] - 1

    if len(axes.shape) == 1:
        axes[0].set_xlabel('residual [px]')
        axes[0].set_ylabel('match count')
    else:
        axes[last_row][0].set_xlabel('residual [px]')
        axes[last_row][0].set_ylabel('match count')

    f, a = plt.subplots(2, 1, clear=True, num=2, sharex=True)
    zp = [i['z'] for i in counts]
    g = [i['n_good'] for i in counts]
    b = [i['n_bad'] for i in counts]
    a[0].plot(zp, g, '-og', label='n matches kept')
    a[1].plot(zp, b, '-or', label='n matches discarded')
    a[1].set_xlabel('z')
    a[0].set_title(monbase)
    [ia.set_ylabel('n') for ia in a]
    [ia.legend() for ia in a]


def plot_filter_results(p, q, w, labels, num=1):
    f, a = plt.subplots(1, 2, clear=True, num=num)
    ulab = np.unique(labels)
    color_id = plt.cm.tab20b(np.linspace(0, 1, ulab.size))
    for u in ulab:
        ind = np.argwhere(labels == u).flatten()
        for iw, marker in zip([1.0, 0.0], ['o', 'X']):
            j = np.isclose(w[ind], iw)
            a[0].scatter(
                    p[ind[j], 0],
                    p[ind[j], 1],
                    marker=marker,
                    edgecolor='k',
                    color=color_id[ulab == u][0])
            a[1].scatter(
                    q[ind[j], 0],
                    q[ind[j], 1],
                    marker=marker,
                    edgecolor='k',
                    color=color_id[ulab == u][0])
    [ia.patch.set_color((0.85, 0.85, 0.85)) for ia in a]
    [ia.set_aspect('equal') for ia in a]


class LensCorrectionPlots():
    def __init__(self, datadir, outputdir):
        self.set_dir(datadir, outputdir)

    def set_dir(self, fdir, outdir):
        self.datadir = fdir
        self.outputdir = outdir
        self.lens_corr_file = self.get_file(outdir, 'lens_corr*.json')
        self.metafile = self.get_file(fdir, '_metadata*')
        self.collectionfile = self.get_file(outdir, 'collection.json')
        self.templatefile = self.get_file(fdir, '_template*')
        self.raw_tilespecs = self.get_file(outdir, 'raw_tilespecs.json')
        self.solved_tilespecs = self.get_file(outdir, 'solved_tilespecs.json')

    def make_all_plots(self, show=True, pdfdir=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        if pdfdir == 'from_files':
            pdfdir = self.outputdir

        figs = []
        figs.append(self.quiver(arrow_scale=10))
        figs += self.data_coverage()
        figs.append(self.tile_positions())
        figs += self.data_coverage(show_residuals=True)

        if pdfdir is not None:
            self.pltfilename = os.path.join(
                    pdfdir,
                    "lens_corr_plots.pdf")
            pdf = PdfPages(self.pltfilename)
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
            pdf.close()
            self.logger.info('wrote \n  %s' % self.pltfilename)

    def get_file(self, fdir, fstr, return_dir=False):
        fname = os.path.join(
                    fdir,
                    fstr)
        files = glob.glob(fname)
        if len(files) != 1:
            print('expected 1 file, found %d for %s' % (
                len(files), fname))
            return None
        if return_dir:
            return os.path.dirname(files[0])

        with open(files[0], 'r') as f:
            j = json.load(f)
        return j

    def add_residuals(self, matches, tspecs, reftf):
        tids = np.array([t.tileId for t in tspecs])

        for m in matches:
            pind = np.argwhere(tids == m['pId']).flatten()[0]
            qind = np.argwhere(tids == m['qId']).flatten()[0]
            p = np.array(m['matches']['p']).transpose()
            q = np.array(m['matches']['q']).transpose()

            for tf in tspecs[pind].tforms:
                if isinstance(tf, renderapi.transform.ReferenceTransform):
                    p = reftf.tform(p)
                else:
                    p = tf.tform(p)
            for tf in tspecs[qind].tforms:
                if isinstance(tf, renderapi.transform.ReferenceTransform):
                    q = reftf.tform(q)
                else:
                    q = tf.tform(q)
            m['matches']['res'] = np.linalg.norm(p - q, axis=1).tolist()
        return matches

    def add_discarded(self, orig, used):
        ups = np.array([u['pId'] for u in used])
        uqs = np.array([u['qId'] for u in used])

        for m in orig:
            discarded = []
            uind = np.argwhere((ups == m['pId']) & (uqs == m['qId'])).flatten()
            if uind.size > 0:
                uind = uind[0]
                um = used[uind]
                up = np.array(um['matches']['p']).transpose()
                for op in np.array(m['matches']['p']).transpose():
                    tf = (up == op)
                    if np.any(tf[:, 0] & tf[:, 1]):
                        discarded.append(0)
                    else:
                        discarded.append(1)
            m['matches']['discarded'] = list(discarded)
        return orig

    def tile_positions(self, fignum=None):
        if self.raw_tilespecs is None:
            print('no raw tilespecs file')
            return
        if self.solved_tilespecs is None:
            print('no solved tilespecs file')
            return
        if self.lens_corr_file is None:
            print('no lens correction file')
            return

        ref = renderapi.transform.ThinPlateSplineTransform(
                json=self.lens_corr_file)
        raw = [renderapi.tilespec.TileSpec(json=j)
               for j in self.raw_tilespecs]
        slv = [renderapi.tilespec.TileSpec(json=j)
               for j in self.solved_tilespecs]
        rtb = [t.bbox_transformed(reference_tforms=[ref])
               for t in raw]
        stb = [t.bbox_transformed(reference_tforms=[ref])
               for t in slv]

        rcorner = np.array([r[0, :2] for r in rtb])
        scorner = np.array([r[0, :2] for r in stb])

        fig, ax = plt.subplots(1, 1, num=fignum, clear=True, figsize=(11, 8))
        ax.plot(rcorner[:, 0], rcorner[:, 1], 'ok', alpha=0.5)
        ax.plot(scorner[:, 0], scorner[:, 1], 'og', alpha=0.5)
        ax.set_aspect('equal')
        ax.legend([
            'original tile corners',
            'translation solved tile corners'], loc=(0.25, 0.25))

        self.aff = estimate_stage_affine(raw, slv)
        astr = ""
        for a in self.aff.M:
            for ia in a:
                astr += "%10.4f " % ia
            astr += '\n'

        ax.set_title(astr)
        ax.invert_yaxis()

        return fig

    def data_coverage(
            self, fignum=None, ax_shape=(4, 5),
            show_residuals=False, clim=5):
        if self.raw_tilespecs is None:
            print('no raw tilespecs file')
            return
        if self.templatefile is None:
            print('no template file')
            return
        if self.collectionfile is None:
            print('no collection.json')
            return

        all_matches = self.templatefile['collection']
        matches = self.add_discarded(all_matches, self.collectionfile)
        ckey = 'discarded'
        cmap = 'Set1_r'

        if show_residuals:
            if self.solved_tilespecs is None:
                print('no solved tilespecs file')
                return
            if self.lens_corr_file is None:
                print('no lens transform file')
                return
            matches = self.add_residuals(
                    self.collectionfile,
                    [renderapi.tilespec.TileSpec(json=t)
                     for t in self.solved_tilespecs],
                    renderapi.transform.ThinPlateSplineTransform(
                        json=self.lens_corr_file))
            ckey = 'res'
            cmap = 'magma'

        tspecs = np.array([renderapi.tilespec.TileSpec(
            json=ij) for ij in self.raw_tilespecs])
        tids = np.array([t.tileId for t in tspecs])

        nmatches = len(all_matches)
        nplts = ax_shape[0] * ax_shape[1]
        nfigs = np.ceil(float(nmatches) / nplts).astype('int')
        figs = []
        axes = []

        for i in range(nfigs):
            f, a = plt.subplots(
                    ax_shape[0], ax_shape[1], num=fignum,
                    clear=True, figsize=(11, 8))
            figs.append(f)
            fignum = f.number + 1
            axes.append(a.flatten())
        axes = np.concatenate(axes)

        [a.set_axis_off() for a in axes]

        bboxes = np.array([t.bbox_transformed() for t in tspecs])

        for i in range(nmatches):
            m = matches[i]

            pind = np.argwhere(tids == m['pId']).flatten()[0]
            qind = np.argwhere(tids == m['qId']).flatten()[0]
            pspec = tspecs[pind]
            [axes[i].plot(b[:, 0], b[:, 1], '-k', alpha=0.05)
             for b in bboxes]
            axes[i].plot(
                    bboxes[pind][:, 0],
                    bboxes[pind][:, 1],
                    color='g',
                    alpha=0.2)
            axes[i].plot(
                    bboxes[qind][:, 0],
                    bboxes[qind][:, 1],
                    color='k',
                    alpha=0.2)
            axes[i].set_aspect('equal')

            pxy = pspec.tforms[0].tform(
                    np.array(m['matches']['p']).transpose())

            c = m['matches'][ckey]

            a1 = axes[i].scatter(
                    pxy[:, 0],
                    pxy[:, 1],
                    marker='s',
                    c=c,
                    s=7.3,
                    cmap=cmap)
            if show_residuals:
                a1.set_clim(0, clim)

            title = tids[pind][-23:] + '\n' + tids[qind][-23:]
            axes[i].set_title(title, fontsize=8)

        if show_residuals:
            figs[-1].colorbar(a1, ax=axes[-1])

        f, axes2 = plt.subplots(
                1, 2, num=fignum, clear=True,
                sharex=True, sharey=True, figsize=(11, 8))
        bbox = tspecs[0].bbox_transformed(tf_limit=0)
        for a in axes2:
            a.plot(bbox[:, 0], bbox[:, 1], color='k', alpha=0.2)
        p = np.concatenate([m['matches']['p'] for m in matches], axis=1)
        q = np.concatenate([m['matches']['q'] for m in matches], axis=1)
        c = np.concatenate([m['matches'][ckey] for m in matches])
        scatts = []
        scatts.append(axes2[0].scatter(
            p[0, :], p[1, :], marker='s', c=c, s=7.3, alpha=0.3, cmap=cmap))
        scatts.append(axes2[1].scatter(
            q[0, :], q[1, :], marker='s', c=c, s=7.3, alpha=0.3, cmap=cmap))
        axes2[0].set_title('matches \"p\"')
        axes2[1].set_title('matches \"q\"')
        [ax.set_aspect('equal') for ax in axes2.flatten()]
        figs.append(f)

        if show_residuals:
            for s in scatts:
                s.set_clim(0, clim)

        return figs

    def quiver(
            self,
            quiver_grid_size=20,
            arrow_scale=1.0,
            fignum=None):

        if self.lens_corr_file is None:
            print('no lens correction file')
            return

        tform = renderapi.transform.ThinPlateSplineTransform(
                json=self.lens_corr_file)

        xmin = tform.srcPts[0, :].min()
        xmax = tform.srcPts[0, :].max()
        ymin = tform.srcPts[1, :].min()
        ymax = tform.srcPts[1, :].max()

        x = np.linspace(xmin, xmax, quiver_grid_size)
        y = np.linspace(ymin, ymax, quiver_grid_size)
        xt, yt = np.meshgrid(x, y)
        src = np.vstack((xt.flatten(), yt.flatten())).transpose()
        dst = tform.tform(src)
        delta = dst - src

        rmax = np.linalg.norm(delta, axis=1).max()

        scale1 = 1.0 / arrow_scale

        fig, axes = plt.subplots(1, 1, num=fignum, clear=True, figsize=(11, 8))
        axes.quiver(
                src[:, 0],
                src[:, 1],
                delta[:, 0],
                delta[:, 1],
                angles='xy',
                scale=scale1,
                scale_units='xy')
        axes.invert_yaxis()
        axes.set_aspect('equal')
        axes.set_title("full transform max: %0.1f "
                       "pixels\narrow scale = %0.1f" % (rmax, 1/scale1))

        return fig


if __name__ == "__main__":
    ddir = "/data/em-131fs3/lctest/T4_6/20190306145208_reference/0"
    pmod = LensCorrectionPlots(ddir, ddir)
    pmod.make_all_plots(pdfdir=ddir)
