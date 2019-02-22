import renderapi
import glob
import os
import json

rp = {
            "host": "em-131fs",
            "port": 8987,
            "owner": "danielk",
            "project": "montage_test",
            "client_scripts": "/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/scripts"
          }
render = renderapi.connect(**rp)

data_dir = "/allen/programs/celltypes/workgroups/em-connectomics/danielk/lcdata/lens_correction16/000000/0"
stack = "first_test_apply_lens"
collection = "first_test_montage"

tfp = os.path.join(
        data_dir,
        "lens_corr_transform.json")
with open(tfp, 'r') as f:
    tfj = json.load(f)
tf = renderapi.transform.ThinPlateSplineTransform(
        json=tfj)

altsp = os.path.join(
        data_dir,
        "apply_lens_tilespecs.json")
with open(altsp, 'r') as f:
    altsj = json.load(f)
alts = [renderapi.tilespec.TileSpec(json=t) for t in altsj]


try:
    renderapi.stack.create_stack(stack, render=render)
except renderapi.errors.RenderError:
    pass
renderapi.client.import_tilespecs(
        stack,
        alts,
        sharedTransforms=[tf],
        render=render)
renderapi.stack.set_stack_state(
        stack,
        state='COMPLETE',
        render=render)


collp = os.path.join(
        data_dir,
        "montage_collection.json")
with open(collp, 'r') as f:
    coll = json.load(f)

renderapi.pointmatch.import_matches(
        collection,
        coll,
        render=render)
