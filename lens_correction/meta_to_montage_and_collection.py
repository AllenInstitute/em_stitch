from argschema import ArgSchemaParser
import renderapi
from .meta_to_collection import main as mtc_main
from .schemas import MetaToMontageAndCollectionSchema
from .lens_correction_solver import tilespec_input_from_metafile
from .generate_EM_tilespecs_from_metafile import \
        GenerateEMTileSpecsModule
import json
import os
import glob
import numpy as np
import shutil


example = {
        "data_dir": "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/000000/0",
        "output_dir": "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/000000/0",
        "ref_transform": "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/20190221123543_reference/0/lens_correction_transform.json"
        }

def get_z_from_metafile(metafile):
    offsets = [
            {
              "load": "Tape147",
              "offset": 100000
            },
            {
              "load": "Tape148",
              "offset": 110000
            },
            {
              "load": "Tape148B",
              "offset": 110000
            },
            {
              "load": "Tape148A",
              "offset": 110000
            },
            {
              "load": "Tape149",
              "offset": 120000
            },
            {
              "load": "Tape151",
              "offset": 130000
            },
            {
              "load": "Tape162",
              "offset": 140000
            },
            {
              "load": "Tape127",
              "offset": 150000
            }]

    loads = np.array([i['load'] for i in offsets])

    with open(metafile, 'r') as f:
        j = json.load(f)
    tape = int(j[0]['metadata']['media_id'])
    offset = offsets[np.argwhere(loads == 'Tape%d' % tape).flatten()[0]]['offset']
    grid = int(j[0]['metadata']['grid'])
    return offset + grid


class MetaToMontageAndCollection(ArgSchemaParser):
    default_schema = MetaToMontageAndCollectionSchema

    def run(self):
        # make a collection json
        collection = os.path.join(
                self.args['output_dir'],
                "montage_collection.json")
        mtc_main([self.args['data_dir'], '-o', collection])
        with open(collection, 'r') as f:
            j = json.load(f)
        groupId = j[0]['pGroupId']

        # make raw tilespec json
        metafile = glob.glob(
                os.path.join(
                    self.args['data_dir'],
                    '_metadata*.json'))[0]
        z = get_z_from_metafile(metafile)
        tspecin = {
                "metafile": metafile,
                "z": z,
                "sectionId": groupId,
                "output_path": os.path.join(
                    self.args['data_dir'],
                    "raw_tilespecs.json")
                }
        gmod = GenerateEMTileSpecsModule(input_data=tspecin, args=[])
        gmod.run()

        # read in raw tilespecs
        with open(tspecin['output_path'], 'r') as f:
            rtj = json.load(f)

        # copy and read in the transform
        tfpath = os.path.join(
                self.args['output_dir'],
                'lens_corr_transform.json')
        if self.args['ref_transform'] != tfpath:
            shutil.copy(self.args['ref_transform'], tfpath)
        with open(tfpath, 'r') as f:
            tfj = json.load(f)

        ref = renderapi.transform.ReferenceTransform()                                                                                     
        ref.refId = tfj['id']

        for t in rtj:
            t['transforms']['specList'].insert(0, ref.to_dict())

        apply_lens_path = os.path.join(
                self.args['output_dir'],
                "apply_lens_tilespecs.json")

        with open(apply_lens_path, 'w') as f:
            json.dump(rtj, f, indent=2)


if __name__ == "__main__":
    mm = MetaToMontageAndCollection(input_data=example)
    mm.run()
