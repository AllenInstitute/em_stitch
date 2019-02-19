# em_lens_correction

note: the maps for opencv.remap are not written to disk currently, they are just part of the `LensCorrectionSolver` object. 
```
        self.map1, self.map2 = maps_from_tform(
                renderapi.transform.ThinPlateSplineTransform(
                    json=jtform),
                renderapi.tilespec.TileSpec(
                    json=jtspecs[0]))
```

Stand-alone lens correction solver for running at acquisition time.

This repo is very similar to mesh_lens_correction in 
https://github.com/AllenInstitute/render-modules

It is meant to be independent of render-modules and a running render server.
