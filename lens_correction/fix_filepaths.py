import json
import os
import glob
import shutil

data_dir = "/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_lens_correction/data/20190205162310_reference/0"

def fix_image_paths(solved_path):
    backup = solved_path + '.backup.json'
    shutil.copy(solved_path, backup)

    with open(solved_path, 'r') as f:
        stj = json.load(f)
    for s in stj:
        orig = s['mipmapLevels']['0']['imageUrl']
        print(orig)
        orig.replace(
                os.path.dirname(orig),
                os.path.dirname(solved_path))
        #print(orig)
    with open(solved_path, 'w') as f:
        json.dump(stj, f, indent=2)

    return

solved_path = glob.glob(os.path.join(data_dir, "solved_tilespecs.json"))[0]
fix_image_paths(solved_path)

