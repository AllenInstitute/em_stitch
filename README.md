# em-lens-correction

Stand-alone lens correction solver for running at acquisition time.

This repo is very similar to mesh_lens_correction in 
https://github.com/AllenInstitute/render-modules

It is meant to be independent of render-modules and a running render server.

Jay:
you'll likely need some render guru to help you if you want to visualize a montage... but, here you go:

the following makes a collection out of the montage metadata file and algo makes an apply lens correction tilespec json
```
(em_lens_correction) danielk@ibs-danielk-ux1:/allen/programs/celltypes/workgroups/em-connectomics/danielk/em-lens-correction$ python -m lens_correction.meta_to_montage_and_collection
```

I set some environment variables:
```
$source ../EM_aligner_python/sourceme.sh
```

and btw, :

```
export RENDER_CLIENT_JAR=/allen/aibs/pipeline/image_processing/volume_assembly/render-jars/production/render-ws-java-client-standalone.jar
export RENDER_JAVA_HOME=/usr/
export RENDER_HOST='127.0.0.1'
```

then upload:
```
python lens_correction/upload_montage.py
```

and now solve...
`pip install EMaligner`

this worked well for me:
```
(em_lens_correction) danielk@ibs-danielk-ux1:/allen/programs/celltypes/workgroups/em-connectomics/danielk/em-lens-correction$ python -m EMaligner.EMaligner --input_json lens_correction/montage_test_input.json --output_mode stack --regularization.default_lambda 1e4 --regularization.translation_factor 1e-10
```

If things are the way they are as committed here, this will give you this:

`http://em-131db:8001/#!{'layers':{'first_test_solved':{'type':'image'_'source':'render://http://em-131fs:8987/danielk/montage_test/first_test_solved'}}_'navigation':{'pose':{'position':{'voxelSize':[1_1_1]_'voxelCoordinates':[49524.8046875_35542.5390625_150000.5]}}_'zoomFactor':1.8221188003905036}_'showAxisLines':false}`

render basics:
the server we're talking to. put this in a browser:
`em-131fs:8987`

before you do anything else, put this in the NdViz Host box, deleting what's there:
`em-131db:8001`

you can use the drop downs and navigate to what we just wrote:

`http://em-131fs:8987/render-ws/view/index.html?catmaidHost=&dynamicRenderHost=&matchCollection=first_test_montage&matchOwner=danielk&ndvizHost=em-131db%3A8001&renderStack=first_test_solved&renderStackOwner=danielk&renderStackProject=montage_test`

from there, clicking on RenderDashboard will take you to:

`http://em-131fs:8987/render-ws/view/stacks.html?catmaidHost=&dynamicRenderHost=&matchCollection=first_test_montage&matchOwner=danielk&ndvizHost=em-131db%3A8001&renderStack=first_test_solved&renderStackOwner=danielk&renderStackProject=montage_test`

hover over `view` for `first_test_solved` and click ndviz will take you to:

`http://em-131db:8001/#!{'layers':{'first_test_solved':{'type':'image'_'source':'render://http://em-131fs:8987/danielk/montage_test/first_test_solved'}}_'navigation':{'pose':{'position':{'voxelSize':[1_1_1]_'voxelCoordinates':[29591.5_37419_150000.5]}}_'zoomFactor':1}}`

this is the solved montage from lenscorrection16.

