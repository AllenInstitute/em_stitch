import renderapi
import os
import sys
import json
import shutil
import numpy as np


def fix_image_paths(solved_path):
    backup = solved_path + '.backup.json'
    with open(solved_path, 'r') as f:
        stj = json.load(f)

    iUrls = np.array([
        os.path.dirname(
                s['mipmapLevels']['0']['imageUrl']) for s in stj])

    if not np.all(iUrls == os.path.dirname(solved_path)):
        shutil.copy(solved_path, backup)
        for s in stj:
            orig = s['mipmapLevels']['0']['imageUrl']
            orig = orig.replace(
                    os.path.dirname(orig),
                    os.path.dirname(solved_path))
            s['mipmapLevels']['0']['imageUrl'] = orig
        with open(solved_path, 'w') as f:
            json.dump(stj, f, indent=2)

    return

rp = {
      "host": "em-131fs",
      "port": 8987,
      "owner": "danielk",
      "project": "lens_corr",
      "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts",
      }
#rp = {
#      "host": "localhost",
#      "port": 9000,
#      "owner": "danielk",
#      "project": "lens_corr",
#      "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts",
#      "validate_scripts": False
#      }
render = renderapi.connect(**rp)

stack = "vis_some_results"

js = []
for fname in sys.argv[1:]:
    #print(os.path.abspath(fname))
    if 'tilespecs.json' in fname:
        fix_image_paths(fname)
    with open(fname, 'r') as f:
        js.append(json.load(f))

tform = None
for j in js:
    if isinstance(j, dict):
        tform = j
    if isinstance(j, list):
        tilespecs = j

r_tilespecs = [renderapi.tilespec.TileSpec(json=ij) for ij in tilespecs]
print(len(r_tilespecs))
r_tform = []
if tform is not None:
    r_tform = [renderapi.transform.ThinPlateSplineTransform(json=tform)]

try:
    renderapi.stack.create_stack(
            stack,
            render=render)
except renderapi.errors.RenderError:
    pass
renderapi.client.import_tilespecs(
        stack,
        r_tilespecs,
        sharedTransforms=r_tform,
        render=render,
        use_rest=True)
renderapi.stack.set_stack_state(
        stack,
        state='COMPLETE',
        render=render)
