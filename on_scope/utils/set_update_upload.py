from argschema import ArgSchemaParser
from .schemas import SetUpdateUploadSchema
from .set_permissions import SetPermissions
from .update_urls import UpdateUrls
from .upload_to_render import UploadToRender
import os


example = {
        "collection_file": "/data/em-131fs3/lctest/T6.2019.04.18.110-120/000111/0/collection.json.gz",
        "resolved_file": "/data/em-131fs3/lctest/T6.2019.04.18.110-120/000111/0/resolvedtiles_AffineModel_0.json.gz",
        "dir_setting": '777',
        'file_exts': ['.json', 'json.gz'],
        'file_setting': '777',
        "backup_copy": True,
        "image_directory": None,
        "render": {
                "host": "em-131fs",
                "port": 8987,
                "owner": "danielk",
                "project": "montage_test",
                "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts"
              },
        "stack": "T4_test2",
        "collection": "T4_test2",
        }


def set_args(args):
    new_args = {
            'data_dir': os.path.dirname(args['resolved_file'])
            }
    for k in ['dir_setting', 'file_exts', 'file_setting']:
        new_args[k] = args[k]
    return dict(new_args)


def update_args(args):
    new_args = {k: args[k] for k in
                ['backup_copy', 'resolved_file', 'image_directory']}
    return dict(new_args)


def upload_args(args):
    new_args = {k: args[k] for k in
                ['render', 'stack', 'collection',
                 'collection_file', 'resolved_file']}
    return dict(new_args)


class SetUpdateUpload(ArgSchemaParser):
    default_schema = SetUpdateUploadSchema

    def run(self):
        if os.name == 'posix':
            pset = SetPermissions(input_data=set_args(self.args), args=[])
            pset.run()

        fpaths = UpdateUrls(input_data=update_args(self.args), args=[])
        fpaths.run()

        up = UploadToRender(input_data=upload_args(self.args), args=[])
        up.run()


if __name__ == "__main__":
    s = SetUpdateUpload(input_data=example, args=[])
    s.run()
