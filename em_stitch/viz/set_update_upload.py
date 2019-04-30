from argschema import ArgSchemaParser
from em_stitch.viz.schemas import SetUpdateUploadSchema
from em_stitch.viz.set_permissions import SetPermissions
from em_stitch.viz.update_urls import UpdateUrls
from em_stitch.viz.upload_to_render import UploadToRender
import os
import logging

logger = logging.getLogger(__name__)


example = {
        # these 2 are how the local client (windows or posix) sees these files
        "fdir": "lctest/T3_OL5pct/005270/0",
        "collection_file": "collection.json.gz",
        "resolved_file": "resolvedtiles_AffineModel_0.json.gz",
        "server_mount": "/data/em-131fs3",  # leave as posix
        # "client_mount_or_map": "Q:/", # windows example
        "client_mount_or_map": "/data/em-131fs3",  # posix example
        "dir_setting": '777',
        'file_exts': ['.json', 'json.gz'],
        'file_setting': '777',
        "backup_copy": True,
        "render": {
                "host": "em-131fs",
                "port": 8987,
                "owner": "the_temcas",
                "project": "montage_test",
                "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts",
                "validate_client": False
              },
        "stack": "T3_example",
        "collection": "T3_example",
        "log_level": "INFO"
        }


def set_args(args):
    new_args = {k: args[k] for k in
                ['fdir', 'client_mount_or_map', 'log_level',
                 'dir_setting', 'file_exts', 'file_setting']}
    return dict(new_args)


def update_args(args):
    new_args = {k: args[k] for k in
                ['fdir', 'client_mount_or_map', 'log_level',
                 'backup_copy', 'resolved_file', 'image_directory']}
    return dict(new_args)


def upload_args(args):
    new_args = {k: args[k] for k in
                ['fdir', 'client_mount_or_map',
                 'render', 'stack', 'collection',
                 'collection_file', 'resolved_file', 'log_level']}
    return dict(new_args)


class SetUpdateUpload(ArgSchemaParser):
    default_schema = SetUpdateUploadSchema

    def run(self):
        logger.setLevel(self.args['log_level'])

        if os.name == 'posix':
            pset = SetPermissions(input_data=set_args(self.args), args=[])
            pset.run()

        fpaths = UpdateUrls(input_data=update_args(self.args), args=[])
        fpaths.run()

        up = UploadToRender(input_data=upload_args(self.args), args=[])
        up.run()


if __name__ == "__main__":
    s = SetUpdateUpload(input_data=example)
    s.run()
