from jinja2 import Environment, FileSystemLoader
import json
import pytest
import os
import copy
from EMaligner import jsongz
from on_scope.montage.montage_solver import MontageSolver
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
    data_dir = os.path.join(test_files_dir, "montage_example")
    with TemporaryDirectory() as output_dir:
        yield json_template(
                example_env,
                "montage_solver_example.json",
                data_dir=data_dir,
                output_dir=output_dir,
                template_dir=test_files_dir)


def test_solver(solver_input_args):
    local_args = copy.deepcopy(solver_input_args)
    with TemporaryDirectory() as output_dir:
        local_args['output_dir'] = output_dir
        ms = MontageSolver(input_data=local_args, args=[])
        ms.run()
        assert os.path.isfile(ms.args['output_json'])
        with open(ms.args['output_json'], 'r') as f:
            j = json.load(f)
        assert len(j) == 2
        for ij in j:
            assert os.path.isfile(ij['output'])
            assert os.path.isfile(ij['collection'])
            for k in ['x', 'y', 'mag']:
                assert ij[k]['mean'] < 2.0
                assert ij[k]['stdev'] < 2.0
