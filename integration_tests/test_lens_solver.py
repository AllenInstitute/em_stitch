from jinja2 import Environment, FileSystemLoader
import json
import pytest
import os
import copy
from bigfeta import jsongz
from em_stitch.lens_correction.lens_correction_solver import (
        LensCorrectionSolver, make_collection_json, one_file,
        LensCorrectionException, tilespec_input_from_metafile)
from em_stitch.lens_correction.mesh_and_solve_transform import \
        MeshAndSolveTransform
from em_stitch.utils.generate_EM_tilespecs_from_metafile import \
        GenerateEMTileSpecsModule
from tempfile import TemporaryDirectory
from marshmallow import ValidationError
import renderapi
import glob
import shutil

test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
example_env = Environment(loader=FileSystemLoader(test_files_dir))


def json_template(env, template_file, **kwargs):
    template = env.get_template(template_file)
    d = json.loads(template.render(**kwargs))
    return d


@pytest.fixture(scope='module')
def solver_input_args():
    data_dir = os.path.join(test_files_dir, "lens_example")
    with TemporaryDirectory() as output_dir:
        yield json_template(
                example_env,
                "lens_solver_example.json",
                data_dir=data_dir,
                output_dir=output_dir)


@pytest.mark.parametrize('source', ['file', 'memory', 'fail'])
def test_solve_from_file_and_memory(solver_input_args, source):
    local_args = copy.deepcopy(solver_input_args)
    metafile = one_file(local_args['data_dir'], '_metadata*.json')
    templatefile = one_file(local_args['data_dir'], '_template*.json')
    compress = True
    with TemporaryDirectory() as output_dir:
        tspecin = tilespec_input_from_metafile(
                metafile,
                local_args['mask_file'],
                output_dir,
                local_args['log_level'],
                compress)
        gentspecs = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
        gentspecs.run()
        cfile, counts = make_collection_json(
                templatefile,
                output_dir,
                compress,
                local_args['ransac_thresh'])

        solver_args = {
                'nvertex': local_args['nvertex'],
                'regularization': local_args['regularization'],
                'output_dir': output_dir,
                'compress_output': compress,
                'log_level': local_args['log_level']}

        if source == 'file':
            solver_args['tilespec_file'] = gentspecs.args['output_path']
            solver_args['match_file'] = cfile
            solver_args['outfile'] = 'resolvedtiles.json.gz'
            solver = MeshAndSolveTransform(input_data=solver_args, args=[])
            solver.run()
            with open(solver.args['output_json'], 'r') as f:
                tspec_path = json.load(f)['resolved_tiles']
            tfile = renderapi.resolvedtiles.ResolvedTiles(
                    json=jsongz.load(tspec_path))
            assert (
                    len(tfile.tilespecs) ==
                    len(gentspecs.tilespecs) ==
                    len(solver.resolved.tilespecs))
        if source == 'memory':
            solver_args['tilespecs'] = gentspecs.tilespecs
            solver_args['matches'] = jsongz.load(cfile)
            solver = MeshAndSolveTransform(input_data=solver_args, args=[])
            solver.run()
            assert (
                    len(gentspecs.tilespecs) ==
                    len(solver.resolved.tilespecs))
        if source == 'fail':
            solver_args['tilespec_file'] = gentspecs.args['output_path']
            solver_args['tilespecs'] = gentspecs.tilespecs
            solver_args['matches'] = jsongz.load(cfile)
            with pytest.raises(ValidationError):
                MeshAndSolveTransform(input_data=solver_args, args=[])
            solver_args.pop('tilespecs')
            solver_args['tilespec_file'] = gentspecs.args['output_path']
            solver_args['matches'] = jsongz.load(cfile)
            solver_args['match_file'] = cfile
            with pytest.raises(ValidationError):
                MeshAndSolveTransform(input_data=solver_args, args=[])


@pytest.mark.parametrize("timestamp", [True, False])
def test_solver(solver_input_args, timestamp):
    local_args = copy.deepcopy(solver_input_args)
    local_args['timestamp'] = timestamp
    with TemporaryDirectory() as output_dir:
        local_args['output_dir'] = output_dir
        lcs = LensCorrectionSolver(input_data=local_args, args=[])
        lcs.run()
        assert os.path.isfile(lcs.args['output_json'])
        with open(lcs.args['output_json'], 'r') as f:
            j = json.load(f)
        for f in j['output'].values():
            assert os.path.isfile(f)


def test_multifile_exception(solver_input_args):
    # this happened once, now this is here
    local_args = copy.deepcopy(solver_input_args)
    with TemporaryDirectory() as inout_dir:
        local_args['data_dir'] = inout_dir
        local_args['output_dir'] = inout_dir

        for f in glob.glob(
                os.path.join(
                    solver_input_args['data_dir'],
                    "*.json")):
            shutil.copy(
                    f,
                    os.path.join(
                        local_args['data_dir']))
        meta = glob.glob(
                os.path.join(
                    local_args['data_dir'], "_meta*.json"))[0]
        msp = os.path.splitext(
                os.path.basename(meta))
        meta2 = os.path.join(
                local_args['data_dir'],
                msp[0] + 'xxx' + '.' + msp[1])
        shutil.copy(meta, meta2)

        with pytest.raises(LensCorrectionException):
            lcs = LensCorrectionSolver(input_data=local_args, args=[])
            lcs.run()


@pytest.mark.parametrize("compress", [True, False])
def test_make_collection(solver_input_args, compress):
    ftemplate = glob.glob(
            os.path.join(
                solver_input_args['data_dir'],
                "_template_matches_*.json"))[0]
    with TemporaryDirectory() as output_dir:
        cfile, counts = make_collection_json(
                ftemplate,
                output_dir,
                compress,
                solver_input_args['ransac_thresh'])
        # for debug purposes, sometimes ignore some matches to see
        # how the solve does without them
        m = jsongz.load(cfile)
        n0 = len(m)
        cfile, counts = make_collection_json(
                ftemplate,
                output_dir,
                solver_input_args['ransac_thresh'],
                compress,
                ignore_match_indices=[0, 1])
        m = jsongz.load(cfile)
        n1 = len(m)
        assert n0 == (n1 + 2)
