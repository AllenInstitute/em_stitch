import json
import os
import glob
import shutil
from six.moves import urllib
import pathlib


data_dir = "/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_lens_correction/data/20190205162310_reference/0"
data_dir = "/data/em-131fs3/lctest/lctest8/20190305084756_reference/0"


def fix_image_paths(solved_path):
    backup = solved_path + '.backup.json'
    shutil.copy(solved_path, backup)

    with open(solved_path, 'r') as f:
        stj = json.load(f)
    for s in stj:
        orig = s['mipmapLevels']['0']['imageUrl']
        orig = urllib.parse.unquote(urllib.parse.urlparse(orig).path)
        print(orig)
        s['mipmapLevels']['0']['imageUrl'] = pathlib.Path(
                os.path.dirname(solved_path),
                os.path.basename(orig)).as_uri()
        print(s['mipmapLevels']['0']['imageUrl'])
    with open(solved_path, 'w') as f:
        json.dump(stj, f, indent=2)

    return


solved_path = glob.glob(os.path.join(data_dir, "solved_tilespecs.json"))[0]
fix_image_paths(solved_path)
