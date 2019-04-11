from jinja2 import Environment, FileSystemLoader
import json
import pytest
import os
import copy
from lens_correction.lens_correction_solver import (
        LensCorrectionSolver)
from tempfile import TemporaryDirectory

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
        # check for files
        files = ['raw_tilespecs.json',
                 'collection.json',
                 'solver_summary.json',
                 'lens_correction_transform.json',
                 'solved_tilespecs.json',
                 'mask.png',
                 'lens_corr_plots.pdf']
        for f in files:
            assert os.path.isfile(
                    os.path.join(local_args['output_dir'], f))
