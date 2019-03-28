from argschema import ArgSchemaParser
import renderapi
from .meta_to_collection import main as mtc_main
from .schemas import MetaToMontageAndCollectionSchema
from .generate_EM_tilespecs_from_metafile import \
        GenerateEMTileSpecsModule
from .utils import pointmatch_filter, get_z_from_metafile
import json
import os
import glob
import shutil


example = {
        "data_dir": "/data/em-131fs3/lctest/T4_6/001844/0",
        "output_dir": "/data/em-131fs3/lctest/T4_6/001844/0",
        "ref_transform": "/data/em-131fs3/lctest/T4_6/20190306145208_reference/0/lens_correction_transform.json",
        "ransacReprojThreshold": 10
        }


def check_failed_from_metafile(metafile):
    with open(metafile, 'r') as f:
        j = json.load(f)
    return j[2]['tile_qc']['failed']


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

        print(groupId)

        # filter the collection
        for match in j:
            _, _, w, _ = pointmatch_filter(
                    match,
                    n_clusters=1,
                    n_cluster_pts=6,
                    ransacReprojThreshold=self.args['ransacReprojThreshold'],
                    model='Similarity')

            match['matches']['w'] = w.tolist()
        with open(collection, 'w') as f:
            json.dump(j, f, indent=2)

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

        if self.args['read_transform_from_meta']:
            with open(metafile, 'r') as f:
                j = json.load(f)
            tfj = j[2]['sharedTransform']
        else:
            # read in the transform from another folder
            tfpath = os.path.join(
                    self.args['output_dir'],
                    'lens_corr_transform.json')
            with open(tfpath, 'r') as f:
                tfj = json.load(f)

        ref = renderapi.transform.ReferenceTransform()
        ref.refId = tfj['id']
        for t in rtj:
            t['transforms']['specList'].insert(0, ref.to_dict())

        tspecs = [renderapi.tilespec.TileSpec(json=t) for t in rtj]
        tform = renderapi.transform.Transform(json=tfj)

        resolved = renderapi.resolvedtiles.ResolvedTiles(
                tilespecs=tspecs,
                transformList=[tform])

        apply_lens_path = os.path.join(
                self.args['output_dir'],
                "ResolvedTiles.json")

        with open(apply_lens_path, 'w') as f:
            json.dump(resolved.to_dict(), f, indent=2)


if __name__ == "__main__":
    mm = MetaToMontageAndCollection(input_data=example)
    mm.run()
