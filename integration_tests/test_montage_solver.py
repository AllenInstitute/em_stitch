from jinja2 import Environment, FileSystemLoader
import json
import pytest
import os
import copy
from em_stitch.montage.montage_solver import (
        MontageSolver, get_transform)
from tempfile import TemporaryDirectory
import glob
import shutil
from marshmallow import ValidationError

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


def test_read_from(solver_input_args):
    meta = glob.glob(os.path.join(
        solver_input_args['data_dir'],
        '_metadata*.json'))[0]
    tf0 = get_transform(meta, '', {}, 'metafile')
    with TemporaryDirectory() as output_dir:
        tfp = os.path.join(output_dir, 'ref.json')
        with open(tfp, 'w') as f:
            json.dump(tf0.to_dict(), f)
        tf1 = get_transform('', tfp, {}, 'reffile')
        tf2 = get_transform('', '', tf0.to_dict(), 'dict')
        assert tf0 == tf1 == tf2


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
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['output']))
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['collection']))
            for k in ['x', 'y', 'mag']:
                assert ij[k]['mean'] < 2.0
                assert ij[k]['stdev'] < 2.0


def test_solver_no_output_dir(solver_input_args):
    local_args = copy.deepcopy(solver_input_args)
    with TemporaryDirectory() as output_dir:
        meta = glob.glob(os.path.join(
            local_args['data_dir'],
            '_metadata*.json'))[0]
        newmeta = os.path.join(
                output_dir,
                os.path.basename(meta))
        shutil.copy(meta, newmeta)
        local_args['data_dir'] = output_dir
        local_args.pop('output_dir')
        ms = MontageSolver(input_data=local_args, args=[])
        ms.run()
        assert os.path.isfile(ms.args['output_json'])
        with open(ms.args['output_json'], 'r') as f:
            j = json.load(f)
        assert len(j) == 2
        for ij in j:
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['output']))
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['collection']))
            for k in ['x', 'y', 'mag']:
                assert ij[k]['mean'] < 2.0
                assert ij[k]['stdev'] < 2.0


def test_solver_metafile_specify(solver_input_args):
    local_args = copy.deepcopy(solver_input_args)
    with TemporaryDirectory() as output_dir:
        local_args['output_dir'] = output_dir
        local_args['metafile'] = glob.glob(
                os.path.join(
                    local_args['data_dir'],
                    '_metadata*.json'))[0]
        local_args.pop('data_dir')
        ms = MontageSolver(input_data=local_args, args=[])
        ms.run()
        assert os.path.isfile(ms.args['output_json'])
        with open(ms.args['output_json'], 'r') as f:
            j = json.load(f)
        assert len(j) == 2
        for ij in j:
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['output']))
            assert os.path.isfile(
                    os.path.join(
                        ms.args['output_dir'],
                        ij['collection']))
            for k in ['x', 'y', 'mag']:
                assert ij[k]['mean'] < 2.0
                assert ij[k]['stdev'] < 2.0


def test_solver_schema_errors(solver_input_args):
    local_args = copy.deepcopy(solver_input_args)
    with TemporaryDirectory() as output_dir:
        local_args['output_dir'] = output_dir
        local_args['solver_templates'][0] = os.path.join(
                os.path.dirname(local_args['solver_templates'][0]),
                'file_does_not_exist.json')
        with pytest.raises(ValidationError):
            MontageSolver(input_data=local_args, args=[])

        local_args = copy.deepcopy(solver_input_args)
        local_args['output_dir'] = output_dir
        local_args.pop('data_dir')
        with pytest.raises(ValidationError):
            MontageSolver(input_data=local_args, args=[])
