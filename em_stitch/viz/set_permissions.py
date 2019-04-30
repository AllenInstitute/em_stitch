from argschema import ArgSchemaParser
from .schemas import SetPermissionsSchema
import subprocess
import sys
import os
import logging

logger = logging.getLogger(__name__)

example = {
        "client_mount_or_map": "/data/em-131fs3",  # could also be "Q:" e.g
        "fdir": 'lctest/T6.2019.04.18.110-120/000116/0/',
        "dir_setting": '777',
        'file_exts': ['.json', 'json.gz'],
        'file_setting': '777',
        "log_level": "INFO"
        }


def run_cmd(cmd):
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            logger.info("command <%s> was terminated "
                        "by signal %d" % (cmd, -retcode))
        else:
            logger.info("command <%s> returned %d" % (cmd, retcode))
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


def set_dirs(mydir, setting):
    if not setting:
        return
    cmd = 'sudo find %s -type d -exec chmod %s {} +' % (mydir, setting)
    run_cmd(cmd)


def set_files(mydir, exts, setting):
    if not setting:
        return
    cmd = 'sudo find %s -type f \( ' % mydir
    for i, ext in enumerate(exts):
        if i != 0:
            cmd += '-o '
        cmd += '-iname \*%s ' % ext
    cmd += '\) -exec chmod %s {} +' % setting
    run_cmd(cmd)


class SetPermissions(ArgSchemaParser):
    default_schema = SetPermissionsSchema

    def run(self):
        logger.setLevel(self.args['log_level'])

        mydir = os.path.join(
                self.args['client_mount_or_map'],
                self.args['fdir'])

        set_dirs(
                mydir,
                self.args['dir_setting'])

        set_files(
                mydir,
                self.args['file_exts'],
                self.args['file_setting'])


if __name__ == "__main__":
    p = SetPermissions(input_data=example)
    p.run()
