import pytest
from lens_correction.lens_correction_solver import (
        LensCorrectionSolver,
        tilespec_input_from_metafile,
        LensCorrectionException)
from lens_correction.generate_EM_tilespecs_from_metafile import (
        GenerateEMTileSpecsModule,
        RenderModuleException)
import copy
import os
import glob
import shutil
import json

DATADIR = './tests/test_data'
FAILDATADIR = './tests/test_data2'

@pytest.fixture(scope='module')
def lens_args():
    example = {
            "data_dir": DATADIR,
            "mask_file": None,
            "n_stdev_thresh": 5,
            "nvertex": 300,
            }
    yield example


def test_gen_tilespecs(lens_args, tmpdir_factory):
    output_dir = str(tmpdir_factory.mktemp('gen_em_out'))
    metafile = glob.glob(os.path.join(
        lens_args['data_dir'], '_metadata*'))[0]
    templatefile = glob.glob(os.path.join(
        lens_args['data_dir'], '_template*'))[0]
    tspecin = tilespec_input_from_metafile(
            metafile, None, templatefile, output_dir)
    g = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
    g.run()


def test_solver(lens_args, tmpdir_factory):
    params = copy.deepcopy(lens_args)
    params['output_dir'] = str(tmpdir_factory.mktemp('lens_corr_out'))

    lcmod = LensCorrectionSolver(input_data=params, args=[])
    lcmod.run()

    params['mask_file'] = os.path.join(
            os.path.abspath(params['data_dir']),
            'test_mask.png')

    lcmod = LensCorrectionSolver(input_data=params, args=[])
    lcmod.run()

    params['data_dir'] = FAILDATADIR
    with pytest.raises(LensCorrectionException):
        lcmod = LensCorrectionSolver(input_data=params, args=[])
        lcmod.run()
