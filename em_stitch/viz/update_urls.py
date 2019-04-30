import os
import shutil
from six.moves import urllib
import pathlib
from EMaligner import jsongz
from argschema import ArgSchemaParser
from .schemas import UpdateUrlSchema
import renderapi
import logging

logger = logging.getLogger(__name__)

example = {
        "backup_copy": True,
        "client_mount_or_map": "/data/em-131fs3",  # could also be "Q:" e.g
        "fdir": 'lctest/T6.2019.04.18.110-120/000116/0/',
        "resolved_file": "resolvedtiles_AffineModel_0.json.gz",
        "server_mount": "/data/em-131fs3",  # leave as posix
        "image_directory": None,
        "log_level": "INFO"
        }


def backup(f):
    for ext in ['.json', '.json.gz']:
        if f.endswith(ext):
            b = f.split(ext)[0]
            break
    i = 0
    newf = f
    while os.path.isfile(newf):
        # were having some permission errors
        # from multiple users writing this backup
        # file.
        newf = b + '_backup_%d' % i + ext
        i += 1
    shutil.copy(f, newf)
    logger.info("copied %s to %s" % (f, newf))


class UpdateUrls(ArgSchemaParser):
    default_schema = UpdateUrlSchema

    def run(self):
        logger.setLevel(self.args['log_level'])

        resolved_path = os.path.join(
                self.args['client_mount_or_map'],
                self.args['fdir'],
                self.args['resolved_file'])

        if self.args['backup_copy']:
            backup(resolved_path)

        resolved = renderapi.resolvedtiles.ResolvedTiles(
                json=jsongz.load(resolved_path))

        if self.args['image_directory'] is None:
            self.args['image_directory'] = pathlib.PurePosixPath(
                    self.args['server_mount'],
                    self.args['fdir'])

        for t in resolved.tilespecs:
            for k in ['imageUrl', 'maskUrl']:
                s = t.ip['0'][k]
                if s:
                    orig = urllib.parse.unquote(urllib.parse.urlparse(s).path)
                    t.ip['0'][k] = pathlib.PurePosixPath(
                            self.args['image_directory'],
                            os.path.basename(orig)).as_uri()

        self.args['resolved_file'] = jsongz.dump(
                resolved.to_dict(), resolved_path, compress=None)

        logger.info("updated tilespec urls in %s" % resolved_path)


if __name__ == '__main__':
    u = UpdateUrls(input_data=example)
    u.run()
