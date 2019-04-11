from argschema import ArgSchemaParser
import renderapi
import os
import json
from .schemas import UploadToRenderSchema

example = {
        "render": {
                "host": "em-131fs",
                "port": 8987,
                "owner": "danielk",
                "project": "montage_test",
                "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts"
              },
        "data_dir": "/data/em-131fs3/lctest/T4_6/001844/0",
        "stack": "T4_6",
        "collection": "T4_6"
        }


class UploadToRender(ArgSchemaParser):
    default_schema = UploadToRenderSchema

    def run(self):
        render = renderapi.connect(**self.args['render'])

        def topath(basename):
            return os.path.join(self.args['data_dir'], basename)

        with open(topath("ResolvedTiles.json"), 'r') as f:
            res = renderapi.resolvedtiles.ResolvedTiles(
                    json=json.load(f))

        renderapi.stack.create_stack(self.args['stack'], render=render)
        renderapi.client.import_tilespecs_parallel(
                self.args['stack'],
                res.tilespecs,
                sharedTransforms=res.transforms,
                render=render)
        renderapi.stack.set_stack_state(
                self.args['stack'],
                state='COMPLETE',
                render=render)

        with open(topath("montage_collection.json"), 'r') as f:
            coll = json.load(f)

        renderapi.pointmatch.import_matches(
                self.args['collection'],
                coll,
                render=render)


if __name__ == "__main__":
    umod = UploadToRender(input_data=example)
    umod.run()
