from jinja2 import Environment, FileSystemLoader
import json
import pytest
import os
import copy
from EMaligner import jsongz
from em_stitch.lens_correction.lens_correction_solver import (
        LensCorrectionSolver, make_collection_json,
        LensCorrectionException)
from tempfile import TemporaryDirectory
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


def test_solver(solver_input_args):
    local_args = copy.deepcopy(solver_input_args)
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
