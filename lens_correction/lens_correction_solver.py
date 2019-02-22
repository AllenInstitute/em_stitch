from argschema import ArgSchemaParser
from .schemas import LensCorrectionSchema
from .generate_EM_tilespecs_from_metafile import GenerateEMTileSpecsModule
from .mesh_and_solve_transform import MeshAndSolveTransform
from .plots import LensCorrectionPlots
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from scipy import ndimage
import logging
import os
import glob
import json
import numpy as np
import renderapi
import cv2
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


example = {
        "data_dir": "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/20190221123543_reference/0",
        "output_dir": "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/20190221123543_reference/0",
        "mask_file": None,
        "ransac_thresh": 5,
        "nvertex": 1000,
        "regularization": {
            "default_lambda": 1.0,
            "lens_lambda": 1.0,
            "translation_factor": 1e-3
            },
        "log_level": "INFO",
        "write_pdf": True
        }


class LensCorrectionException(Exception):
    pass


def one_file(fdir, fstub):
    fullstub = os.path.join(fdir, fstub)
    files = glob.glob(fullstub)
    lf = len(files)
    if lf != 1:
        raise LensCorrectionException(
            "expected 1 file, found %d for %s" % (
                lf,
                fullstub))
    return files[0]


def tilespec_input_from_metafile(
        metafile, mask_file, template_file, output_dir, log_level):
    result = {}
    result['metafile'] = metafile

    with open(metafile, 'r') as f:
        jmeta = json.load(f)

    with open(template_file, 'r') as f:
        jtemp = json.load(f)

    result['z'] = np.random.randint(0, 1000)
    result['minimum_intensity'] = 0
    bpp = jmeta[0]['metadata']['camera_info']['camera_bpp']
    result['maximum_intensity'] = int(np.power(2, bpp) - 1)
    result['sectionId'] = jmeta[0]['metadata']['grid']
    result['maskUrl'] = mask_file
    result['output_path'] = os.path.join(output_dir, 'raw_tilespecs.json')
    result['log_level'] = log_level
    return result


def local_RANSAC(match, ncluster=14, thresh=10, nmin=10):
    p = np.array(match['matches']['p']).transpose()
    q = np.array(match['matches']['q']).transpose()

    # find some clusters with a minimum number of points
    while True:
        kmeans = KMeans(n_clusters=ncluster, random_state=0)
        lab = kmeans.fit_predict(p)
        ulab, cnts = np.unique(lab, return_counts=True)
        if np.all(cnts >= nmin) | (ncluster == 1):
            break
        ncluster -= 1

    # run RANSAC on each cluster and track inliers
    ind = [] 
    for u in ulab:
        cind = np.argwhere(lab == u).flatten()
        r = RANSACRegressor(residual_threshold=thresh)
        r.fit(p[cind], q[cind])
        ind += cind[np.argwhere(r.inlier_mask_).flatten()].tolist()
    
    return p, q, lab, np.array(ind)


def make_collection_json(template_file, output_dir, thresh, match_filter='local_RANSAC'):
#def make_collection_json(template_file, output_dir, thresh, match_filter='n_stdev_peak'):
    with open(template_file, 'r') as f:
        matches = json.load(f)
    collection_file = os.path.join(
            output_dir,
            "collection.json")
    new_matches = []
    counts = []
    for m in matches['collection']:
        counts.append({})
        ind = np.arange(len(m['matches']['p'][0]))
        counts[-1]['n_from_gpu'] = ind.size
        
        if match_filter == 'local_RANSAC':
            _, _, _, ind = local_RANSAC(m, thresh=thresh)

        counts[-1]['n_after_filter'] = ind.size
        if ind.size > 1:
            new_matches.append({})
            for k in ['pGroupId', 'qGroupId', 'pId', 'qId']:
                new_matches[-1][k] = m[k]
            new_matches[-1]['matches'] = {}
            new_matches[-1]['matches']['p'] = \
                np.array(m['matches']['p'])[:, ind].tolist()
            new_matches[-1]['matches']['q'] = \
                np.array(m['matches']['q'])[:, ind].tolist()
            new_matches[-1]['matches']['w'] = [1.0] * ind.size

    with open(collection_file, 'w') as f:
        json.dump(new_matches, f)
    return collection_file, counts


def src_from_xy(x, y, transpose=True):
    xt, yt = np.meshgrid(x, y)
    src = np.vstack((xt.flatten(), yt.flatten())).astype('float')
    if not transpose:
        return src
    return src.transpose()


def split_inverse_tform(tform, src, block_size):
    nsplit = np.ceil(float(src.shape[0]) / float(block_size))
    split_src = np.array_split(src, nsplit, axis=0)
    dst = []
    for s in split_src:
        dst.append(tform.inverse_tform(s))
    dst = np.concatenate(dst)
    return dst


def maps_from_tform(tform, width, height, logger, block_size=10000, res=32):
    t0 = time.time()

    x = np.arange(0, width + res, res)
    y = np.arange(0, height + res, res)
    src = src_from_xy(x, y)
    idst = split_inverse_tform(tform, src, block_size)
    ix = idst[:, 0].reshape(y.size, x.size)
    iy = idst[:, 1].reshape(y.size, x.size)

    fx = np.arange(0, width)
    fy = np.arange(0, height)
    src = np.flipud(src_from_xy(fx, fy, transpose=False).astype('float32'))
    src[0, :] *= ((ix.shape[0] - 1) / y.max())
    src[1, :] *= ((ix.shape[1] - 1) / x.max())

    dx = ndimage.map_coordinates(ix, src, order=1)
    dy = ndimage.map_coordinates(iy, src, order=1)

    map1 = dx.reshape((fy.size, fx.size)).astype('float32')
    map2 = dy.reshape((fy.size, fx.size)).astype('float32')

    # actually do it, to find a mask
    mask = np.ones_like(map1)
    mask = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST)
    mask = np.uint8(mask * 255)
    t1 = time.time()
    logger.info(" created maps for remap:\n  took %0.1f seconds" % (t1 - t0))
    return map1, map2, mask


class LensCorrectionSolver(ArgSchemaParser):
    default_schema = LensCorrectionSchema

    def run(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.check_for_files()
        self.output_dir = self.args.get('output_dir', self.args['data_dir'])
        self.logger.info("destination directory:\n  %s" % self.output_dir)

        tspecin = tilespec_input_from_metafile(
                self.metafile,
                self.args['mask_file'],
                self.matchfile,
                self.output_dir,
                self.args['log_level'])
        gentspecs = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
        gentspecs.run()

        assert os.path.isfile(tspecin['output_path'])
        self.logger.info("raw tilespecs written:\n  %s" % tspecin['output_path'])

        collection_path, self.filter_counts = make_collection_json(
                self.matchfile,
                self.output_dir,
                self.args['ransac_thresh'])

        self.n_from_gpu = np.array([i['n_from_gpu'] for i in self.filter_counts]).sum()
        self.n_after_filter = np.array([i['n_after_filter'] for i in self.filter_counts]).sum()
        self.logger.info("filter counts: %0.2f %% kept" % (100 * float(self.n_after_filter) / self.n_from_gpu))

        assert os.path.isfile(collection_path)
        self.logger.info("filtered collection written:\n  %s" % collection_path)

        solver_args = {
                'nvertex': self.args['nvertex'],
                'regularization': self.args['regularization'],
                'good_solve': self.args['good_solve'],
                'tilespec_file': tspecin['output_path'],
                'match_file': collection_path,
                'output_dir': self.output_dir,
                'outfile': 'lens_correction_transform.json',
                'log_level': self.args['log_level']}

        self.solver = MeshAndSolveTransform(input_data=solver_args, args=[])
        self.solver.run()

        with open(tspecin['output_path'], 'r') as f:
            jtspecs = json.load(f)

        with open(
                os.path.join(
                    self.output_dir,
                    solver_args['outfile']), 'r') as f:
            jtform = json.load(f)

        rspec = renderapi.tilespec.TileSpec(json=jtspecs[0])

        self.map1, self.map2, self.mask = maps_from_tform(
                renderapi.transform.ThinPlateSplineTransform(
                    json=jtform),
                rspec.width,
                rspec.height,
                self.logger,
                res=32)

        maskname = os.path.join(self.output_dir, 'mask.png')
        cv2.imwrite(maskname, self.mask)
        self.logger.info("wrote:\n  %s" % maskname)

        if self.args['write_pdf']:
            plts = LensCorrectionPlots(self.args['data_dir'], self.output_dir)
            plts.make_all_plots(pdfdir='from_files', show=False)

    def check_for_files(self):
        self.metafile = one_file(self.args['data_dir'], '_metadata*')
        self.matchfile = one_file(self.args['data_dir'], '_template_matches*')
        self.logger.info("Using files: \n  %s\n  %s" % (
            self.metafile, self.matchfile))


if __name__ == '__main__':
    lcmod = LensCorrectionSolver(input_data=example)
    lcmod.run()
