from argschema import ArgSchemaParser
from .schemas import LensCorrectionSchema
from .generate_EM_tilespecs_from_metafile import GenerateEMTileSpecsModule
from .mesh_and_solve_transform import MeshAndSolveTransform
from .plots import LensCorrectionPlots
import utils
import logging
import os
import glob
import json
import numpy as np
import renderapi
import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


example = {
        "data_dir": "/data/em-131fs3/lctest/T4_07/20190315142205_reference/0",
        "output_dir": "/data/em-131fs3/lctest/T4_07/20190315142205_reference/0",
        "mask_file": None,
        "ransac_thresh": 10,
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


# trivial change


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
        metafile, mask_file, output_dir, log_level):
    result = {}
    result['metafile'] = metafile

    with open(metafile, 'r') as f:
        jmeta = json.load(f)

    result['z'] = np.random.randint(0, 1000)
    result['minimum_intensity'] = 0
    bpp = jmeta[0]['metadata']['camera_info']['camera_bpp']
    result['maximum_intensity'] = int(np.power(2, bpp) - 1)
    result['sectionId'] = jmeta[0]['metadata']['grid']
    result['maskUrl'] = mask_file
    result['output_path'] = os.path.join(output_dir, 'raw_tilespecs.json')
    result['log_level'] = log_level
    return result


def make_collection_json(template_file, output_dir, thresh):
    with open(template_file, 'r') as f:
        matches = json.load(f)
    collection_file = os.path.join(
            output_dir,
            "collection.json")
    counts = []
    for m in matches['collection']:
        counts.append({})
        ind = np.arange(len(m['matches']['p'][0]))
        counts[-1]['n_from_gpu'] = ind.size

        _, _, w, _ = utils.pointmatch_filter(
                m,
                n_clusters=None,
                n_cluster_pts=20,
                ransacReprojThreshold=40.0,
                model='Similarity')

        m['matches']['w'] = w.tolist()

        counts[-1]['n_after_filter'] = np.count_nonzero(w)

    with open(collection_file, 'w') as f:
        json.dump(matches['collection'], f)
    return collection_file, counts


class LensCorrectionSolver(ArgSchemaParser):
    default_schema = LensCorrectionSchema

    def __init__(self, *args, **kwargs):
        super(LensCorrectionSolver, self).__init__(*args, **kwargs)
        self.jtform = None

    def run(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.args['log_level'])
        utils.logger.setLevel(self.args['log_level'])
        self.check_for_files()
        self.output_dir = self.args.get('output_dir', self.args['data_dir'])
        self.logger.info("destination directory:\n  %s" % self.output_dir)

        tspecin = tilespec_input_from_metafile(
                self.metafile,
                self.args['mask_file'],
                self.output_dir,
                self.args['log_level'])
        gentspecs = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
        gentspecs.run()

        assert os.path.isfile(tspecin['output_path'])
        self.logger.info(
                "raw tilespecs written:\n  %s" % tspecin['output_path'])

        collection_path, self.filter_counts = make_collection_json(
                self.matchfile,
                self.output_dir,
                self.args['ransac_thresh'])

        self.n_from_gpu = np.array(
                [i['n_from_gpu'] for i in self.filter_counts]).sum()
        self.n_after_filter = np.array(
                [i['n_after_filter'] for i in self.filter_counts]).sum()
        self.logger.info(
                "filter counts: %0.2f %% kept" %
                (100 * float(self.n_after_filter) / self.n_from_gpu))

        assert os.path.isfile(collection_path)
        self.logger.info(
                "filtered collection written:\n  %s" % collection_path)

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
            self.jtform = json.load(f)

        rspec = renderapi.tilespec.TileSpec(json=jtspecs[0])

        self.map1, self.map2, self.mask = utils.maps_from_tform(
                renderapi.transform.ThinPlateSplineTransform(
                    json=self.jtform),
                rspec.width,
                rspec.height,
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
