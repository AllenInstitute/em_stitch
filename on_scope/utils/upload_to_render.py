from argschema import ArgSchemaParser
import renderapi
from .schemas import UploadToRenderSchema
from EMaligner import jsongz


example = {
        "render": {
                "host": "em-131fs",
                "port": 8987,
                "owner": "danielk",
                "project": "montage_test",
                "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts"
              },
        "stack": "T4_test",
        "collection": "T4_test",
        "collection_file": "/data/em-131fs3/lctest/T4.2019.04.18.1650-1660/001650/0/collection.json.gz",
        "resolved_file": "/data/em-131fs3/lctest/T4.2019.04.18.1650-1660/001650/0/resolvedtiles_AffineModel_0.json.gz"
        }


def upload_resolved_file(render_params, stack, resolved_file, close_stack):
    if resolved_file is None:
        return

    resolved = renderapi.resolvedtiles.ResolvedTiles(
            json=jsongz.load(resolved_file))

    render = renderapi.connect(**render_params)

    renderapi.stack.create_stack(stack, render=render)
    renderapi.client.import_tilespecs_parallel(
            stack,
            resolved.tilespecs,
            sharedTransforms=resolved.transforms,
            render=render)
    if close_stack:
        renderapi.stack.set_stack_state(
                stack,
                state='COMPLETE',
                render=render)

    return


def upload_collection_file(render_params, collection, collection_file):
    if collection is None:
        return

    render = renderapi.connect(**render_params)

    renderapi.pointmatch.import_matches(
            collection,
            jsongz.load(collection_file),
            render=render)


class UploadToRender(ArgSchemaParser):
    default_schema = UploadToRenderSchema

    def run(self):
        upload_resolved_file(
                self.args['render'],
                self.args['stack'],
                self.args['resolved_file'],
                self.args['close_stack'])

        upload_collection_file(
                self.args['render'],
                self.args['collection'],
                self.args['collection_file'])


if __name__ == "__main__":
    umod = UploadToRender(input_data=example)
    umod.run()
