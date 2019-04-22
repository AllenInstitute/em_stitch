from argschema import ArgSchemaParser
import renderapi
from .schemas import UploadToRenderSchema
from EMaligner import jsongz
import os
import logging

logger = logging.getLogger(__name__)

example = {
        "client_mount_or_map": "/data/em-131fs3",  # could also be "Q:" e.g
        "fdir": 'lctest/T6.2019.04.18.110-120/000116/0/',
        "resolved_file": "resolvedtiles_AffineModel_0.json.gz",
        "collection_file": "collection.json.gz",
        "render": {
                "host": "em-131fs",
                "port": 8987,
                "owner": "danielk",
                "project": "montage_test",
                "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts"
              },
        "stack": "T4_test",
        "collection": "T4_test",
        "log_level": "INFO"
        }


def upload_resolved_file(render_params, stack, resolved_file, close_stack):
    if resolved_file is None:
        return

    resolved = renderapi.resolvedtiles.ResolvedTiles(
            json=jsongz.load(resolved_file))

    render = renderapi.connect(**render_params)

    renderapi.stack.create_stack(stack, render=render)
    renderapi.client.import_tilespecs(
            stack,
            resolved.tilespecs,
            sharedTransforms=resolved.transforms,
            use_rest=True,
            render=render)
    if close_stack:
        renderapi.stack.set_stack_state(
                stack,
                state='COMPLETE',
                render=render)

    logger.info("imported %d tilespecs to render from %s" % (
        len(resolved.tilespecs), resolved_file))

    url = ("\nhttp://" +
           "%s:%d" % (render_params['host'], render_params['port']) +
           "/render-ws/view/stacks.html?ndvizHost=em-131fs%3A8001" +
           "&renderStack=%s" % stack +
           "&renderStackOwner=%s" % render_params['owner'] +
           "&renderStackProject=%s" % render_params['project'])
    logger.info(url)

    return


def upload_collection_file(render_params, collection, collection_file):
    if collection is None:
        return

    render = renderapi.connect(**render_params)

    matches = jsongz.load(collection_file)

    renderapi.pointmatch.import_matches(
            collection,
            matches,
            render=render)

    logger.info("imported %d pointmatches to render from %s" % (
        len(matches), collection_file))


class UploadToRender(ArgSchemaParser):
    default_schema = UploadToRenderSchema

    def run(self):
        logger.setLevel(self.args['log_level'])

        resolved_file = os.path.join(
                self.args['client_mount_or_map'],
                self.args['fdir'],
                self.args['resolved_file'])

        collection_file = os.path.join(
                self.args['client_mount_or_map'],
                self.args['fdir'],
                self.args['collection_file'])

        upload_resolved_file(
                self.args['render'],
                self.args['stack'],
                resolved_file,
                self.args['close_stack'])

        upload_collection_file(
                self.args['render'],
                self.args['collection'],
                collection_file)


if __name__ == "__main__":
    umod = UploadToRender(input_data=example)
    umod.run()
