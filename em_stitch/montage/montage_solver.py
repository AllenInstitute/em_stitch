from argschema import ArgSchemaParser
import renderapi
from em_stitch.montage import meta_to_collection
from em_stitch.montage.schemas import MontageSolverSchema
from em_stitch.utils.generate_EM_tilespecs_from_metafile import \
        GenerateEMTileSpecsModule
from em_stitch.utils.utils import pointmatch_filter, get_z_from_metafile
from EMaligner import jsongz
import EMaligner.EMaligner as ema
import json
import os
import glob

dname = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../", "../", "integration_tests", "test_files")

example = {
        "data_dir": "/data/em-131fs3/lctest/T4_6/001844/0",
        "output_dir": "/data/em-131fs3/lctest/T4_6/001844/0",
        "ref_transform": None,
        "ransacReprojThreshold": 10,
        "solver_template_dir": dname,
        "solver_templates": [
            "affine_template.json",
            "polynomial_template.json",
            ]
        }


class MontageSolverException(Exception):
    pass


def do_solve(template_path, args, index):
    with open(template_path, 'r') as f:
        template = json.load(f)
    template['input_stack']['input_file'] = \
        args['input_stack']['input_file']
    template['pointmatch']['input_file'] = \
        args['pointmatch']['input_file']
    template['output_stack']['compress_output'] = \
        args['output_stack']['compress_output']
    template['first_section'] = template['last_section'] = \
        args['first_section']
    fname = os.path.join(
            os.path.dirname(args['input_stack']['input_file']),
            'resolvedtiles_%s_%d.json' % (template['transformation'], index))
    template['output_stack']['output_file'] = fname
    template['fullsize_transform'] = False
    aligner = ema.EMaligner(input_data=template, args=[])
    aligner.run()
    # these numbers only meaningful for fullsize_transform = False
    # to get results already in memory, on-scope, let's keep it that way
    # otherwise, we'll need a separate calculation that loads tilespecs
    # and matches to calculate residuals, costing more time
    res = {
            'output': os.path.basename(
                aligner.args['output_stack']['output_file']),
            'collection': os.path.basename(
                aligner.args['pointmatch']['input_file']),
            'x': {
                'mean': aligner.results['err'][0][0],
                'stdev': aligner.results['err'][0][1]
                },
            'y': {
                'mean': aligner.results['err'][1][0],
                'stdev': aligner.results['err'][1][1]
                },
            'mag': {
                'mean': aligner.results['mag'][0],
                'stdev': aligner.results['mag'][1]
                }
            }
    return res


def do_solves(collection, input_stack, z, compress, solver_args):
    args = {'input_stack': {}, 'output_stack': {}, 'pointmatch': {}}
    args['input_stack']['input_file'] = input_stack
    args['pointmatch']['input_file'] = collection
    args['output_stack']['compress_output'] = compress
    args['first_section'] = args['last_section'] = z

    results = []
    for index, template in enumerate(solver_args):
        results.append(do_solve(template, args, index))

    return results


def montage_filter_matches(matches, thresh, model='Similarity'):
    for match in matches:
        _, _, w, _ = pointmatch_filter(
                match,
                n_clusters=1,
                n_cluster_pts=6,
                ransacReprojThreshold=thresh,
                model=model)
        match['matches']['w'] = w.tolist()


def get_metafile_path(datadir):
    return glob.glob(os.path.join(datadir, '_metadata*.json'))[0]


def make_raw_tilespecs(metafile, outputdir, groupId, compress):
    z = get_z_from_metafile(metafile)
    tspecin = {
            "metafile": metafile,
            "z": z,
            "sectionId": groupId,
            "output_path": os.path.join(outputdir, 'raw_tilespecs.json'),
            "compress_output": compress
            }
    gmod = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
    gmod.run()
    return gmod.args['output_path'], z


def get_transform(metafile, tfpath, refdict, read_from):
    if read_from == 'metafile':
        with open(metafile, 'r') as f:
            j = json.load(f)
        tfj = j[2]['sharedTransform']
    elif read_from == 'reffile':
        with open(tfpath, 'r') as f:
            tfj = json.load(f)
    elif read_from == 'dict':
        tfj = refdict
    return renderapi.transform.Transform(json=tfj)


def make_resolved(rawspecpath, tform, outputdir, compress):
    # read in the tilespecs
    rtj = jsongz.load(rawspecpath)
    tspecs = [renderapi.tilespec.TileSpec(json=t) for t in rtj]

    # do not need this anymore
    os.remove(rawspecpath)

    # add the reference transform
    ref = renderapi.transform.ReferenceTransform()
    ref.refId = tform.transformId
    for t in tspecs:
        t.tforms.insert(0, ref)

    # make a resolved tile object
    resolved = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=tspecs,
            transformList=[tform])

    # write it to file and return the path
    rpath = os.path.join(outputdir, 'resolvedtiles_input.json')

    return jsongz.dump(resolved.to_dict(), rpath, compress)


class MontageSolver(ArgSchemaParser):
    default_schema = MontageSolverSchema

    def run(self):
        if 'metafile' not in self.args:
            self.args['metafile'] = get_metafile_path(self.args['data_dir'])
        else:
            self.args['data_dir'] = os.path.dirname(self.args['metafile'])

        if not self.args['output_dir']:
            self.args['output_dir'] = self.args['data_dir']

        # read the matches from the metafile
        matches = meta_to_collection.main([self.args['data_dir']])

        montage_filter_matches(
                matches,
                self.args['ransacReprojThreshold'])

        # write to file
        collection = os.path.join(self.args['output_dir'], "collection.json")
        collection = jsongz.dump(
                matches,
                collection,
                compress=self.args['compress_output'])

        # make raw tilespec json
        rawspecpath, z = make_raw_tilespecs(
                self.args['metafile'],
                self.args['output_dir'],
                matches[0]['pGroupId'],
                self.args['compress_output'])

        # get the ref transform
        tform = get_transform(
                self.args['metafile'],
                self.args['ref_transform'],
                self.args['ref_transform_dict'],
                self.args['read_transform_from'])

        # make a resolved tile object
        input_stack_path = make_resolved(
                rawspecpath,
                tform,
                self.args['output_dir'],
                self.args['compress_output'])

        templates = [os.path.join(self.args['solver_template_dir'], t)
                     for t in self.args['solver_templates']]
        self.results = do_solves(
                collection,
                input_stack_path,
                z,
                self.args['compress_output'],
                templates)

        self.args['output_json'] = os.path.join(
                self.args['output_dir'], 'montage_results.json')
        with open(self.args['output_json'], 'w') as f:
            json.dump(self.results, f, indent=2)


if __name__ == "__main__":
    mm = MontageSolver(input_data=example)
    mm.run()
