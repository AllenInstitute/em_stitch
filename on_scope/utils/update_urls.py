import os
import shutil
from six.moves import urllib
import pathlib
from EMaligner import jsongz
from argschema import ArgSchemaParser
from .schemas import UpdateUrlSchema
import renderapi

example = {
        "backup_copy": True,
        "resolved_file": "/data/em-131fs3/lctest/T4.2019.04.18.1650-1660/001650/0/resolvedtiles_AffineModel_0.json.gz",
        "image_directory": None
        }


def backup(f):
    for ext in ['.json', '.json.gz']:
        if f.endswith(ext):
            b = f.split(ext)[0]
            break
    newf = b + '_backup_' + ext
    shutil.copy(f, newf)


class UpdateUrls(ArgSchemaParser):
    default_schema = UpdateUrlSchema

    def run(self):
        if self.args['backup_copy']:
            backup(self.args['resolved_file'])

        resolved = renderapi.resolvedtiles.ResolvedTiles(
                json=jsongz.load(self.args['resolved_file']))

        if self.args['image_directory'] is None:
            self.args['image_directory'] = os.path.dirname(
                    self.args['resolved_file'])

        for t in resolved.tilespecs:
            for k in ['imageUrl', 'maskUrl']:
                s = t.ip['0'][k]
                if s:
                    orig = urllib.parse.unquote(urllib.parse.urlparse(s).path)
                    t.ip['0'][k] = pathlib.PurePosixPath(
                            self.args['image_directory'],
                            os.path.basename(orig)).as_uri()

        self.args['resolved_file'] = jsongz.dump(
                resolved.to_dict(), self.args['resolved_file'], compress=None)


if __name__ == '__main__':
    u = UpdateUrls(input_data=example, args=[])
    u.run()
