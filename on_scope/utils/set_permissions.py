from argschema import ArgSchemaParser
from .schemas import SetPermissionsSchema
import subprocess
import sys


example = {
        "data_dir": '/data/em-131fs3/lctest/T6.2019.04.18.110-120',
        "dir_setting": None,
        'file_exts': ['.json', 'json.gz'],
        'file_setting': '777'
        }


def run_cmd(cmd):
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
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
        set_dirs(
                self.args['data_dir'],
                self.args['dir_setting'])

        set_files(
                self.args['data_dir'],
                self.args['file_exts'],
                self.args['file_setting'])


if __name__ == "__main__":
    p = SetPermissions(input_data=example, args=[])
    p.run()
