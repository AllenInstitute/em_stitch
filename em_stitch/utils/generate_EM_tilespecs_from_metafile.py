import renderapi
import os
import json
import numpy
import pathlib
from bigfeta import jsongz
from argschema import ArgSchemaParser
from .schemas import GenerateEMTileSpecsParameters

# this is a modification of https://github.com/AllenInstitute/
# render-modules/blob/master/rendermodules/dataimport/
# generate_EM_tilespecs_from_metafile.py
# that does not depend on render-modules nor
# on a running render server


class RenderModuleException(Exception):
    pass


class GenerateEMTileSpecsModule(ArgSchemaParser):
    default_schema = GenerateEMTileSpecsParameters

    @staticmethod
    def image_coords_from_stage(stage_coords, resX, resY, rotation):
        cr = numpy.cos(rotation)
        sr = numpy.sin(rotation)
        x = stage_coords[0] / resX
        y = stage_coords[1] / resY
        return (int(x * cr + y * sr),
                int(-x * sr + y * cr))

    @staticmethod
    def tileId_from_basename(fname):
        return os.path.splitext(os.path.basename(fname))[0]

    @staticmethod
    def ts_from_imgdata_tileId(imgdata, imgdir, x, y, tileId, 
                               minint=0, maxint=255, maskUrl=None,
                               width=3840, height=3840, z=None, sectionId=None,
                               scopeId=None, cameraId=None, pixelsize=None):
        raw_tforms = [renderapi.transform.AffineModel(B0=x, B1=y)]
        imageUrl = pathlib.Path(
            os.path.abspath(os.path.join(
                imgdir, imgdata['img_path']))).as_uri()
        if maskUrl is not None:
                maskUrl = pathlib.Path(maskUrl).as_uri()

        ip = renderapi.image_pyramid.ImagePyramid()
        ip[0] = renderapi.image_pyramid.MipMap(imageUrl=imageUrl,
                                               maskUrl=maskUrl)
        return renderapi.tilespec.TileSpec(
            tileId=tileId, z=z,
            width=width, height=height,
            minint=minint, maxint=maxint,
            tforms=raw_tforms,
            imagePyramid=ip,
            sectionId=sectionId, scopeId=scopeId, cameraId=cameraId,
            imageCol=imgdata['img_meta']['raster_pos'][0],
            imageRow=imgdata['img_meta']['raster_pos'][1],
            stageX=imgdata['img_meta']['stage_pos'][0],
            stageY=imgdata['img_meta']['stage_pos'][1],
            rotation=imgdata['img_meta']['angle'], pixelsize=pixelsize)

    def ts_from_imgdata(self, imgdata, imgdir, x, y,
                        minint=0, maxint=255, maskUrl=None,
                        width=3840, height=3840, z=None, sectionId=None,
                        scopeId=None, cameraId=None, pixelsize=None):
        tileId = self.tileId_from_basename(imgdata['img_path'])
        sectionId = (self.sectionId_from_z(z) if sectionId is None
                     else sectionId)
        return self.ts_from_imgdata_tileId(
            imgdata, imgdir, x, y, tileId, 
            minint, maxint, maskUrl,
            width, height, z, sectionId,
            scopeId, cameraId, pixelsize)

    @classmethod
    def ts_from_metadata(
            cls, md, image_directory, z, sectionId=None,
            minimum_intensity=0, maximum_intensity=255, maskUrl=None):
        roidata = md[0]['metadata']
        imgdata = md[1]['data']
        img_coords = {img['img_path']: cls.image_coords_from_stage(
            img['img_meta']['stage_pos'],
            img['img_meta']['pixel_size_x_move'],
            img['img_meta']['pixel_size_y_move'],
            numpy.radians(img['img_meta']['angle'])) for img in imgdata}

        minX, minY = numpy.min(numpy.array(list(img_coords.values())), axis=0)
        # assume isotropic pixels
        pixelsize = roidata['calibration']['highmag']['x_nm_per_pix']
        
        inputs = {
            "minint": minimum_intensity,
            "maxint": maximum_intensity,
            "z": z,
            "sectionId": sectionId,
            "maskUrl": maskUrl
        }

        tspecs = [
                cls.ts_from_imgdata_tileId(
                    img, image_directory,
                    img_coords[img['img_path']][0] - minX,
                    img_coords[img['img_path']][1] - minY,
                    cls.tileId_from_basename(img["img_path"]),
                    width=roidata['camera_info']['width'],
                    height=roidata['camera_info']['height'],
                    scopeId=roidata['temca_id'],
                    cameraId=roidata['camera_info']['camera_id'],
                    pixelsize=pixelsize, **inputs) for img in imgdata]
        return tspecs

    def run(self):
        with open(self.args['metafile'], 'r') as f:
            meta = json.load(f)
        roidata = meta[0]['metadata']
        imgdata = meta[1]['data']
        img_coords = {img['img_path']: self.image_coords_from_stage(
            img['img_meta']['stage_pos'],
            img['img_meta']['pixel_size_x_move'],
            img['img_meta']['pixel_size_y_move'],
            numpy.radians(img['img_meta']['angle'])) for img in imgdata}

        # if not imgdata:
        #     raise RenderModuleException(
        #         "No relevant image metadata found for metafile {}".format(
        #             self.args['metafile']))

        minX, minY = numpy.min(numpy.array(list(img_coords.values())), axis=0)
        # assume isotropic pixels
        pixelsize = roidata['calibration']['highmag']['x_nm_per_pix']

        imgdir = self.args.get(
            'image_directory', os.path.dirname(self.args['metafile']))

        self.render_tspecs = [
                self.ts_from_imgdata(
                    img, imgdir,
                    img_coords[img['img_path']][0] - minX,
                    img_coords[img['img_path']][1] - minY,
                    minint=self.args['minimum_intensity'],
                    maxint=self.args['maximum_intensity'],
                    width=roidata['camera_info']['width'],
                    height=roidata['camera_info']['height'],
                    z=self.args['z'],
                    sectionId=self.args.get('sectionId'),
                    scopeId=roidata['temca_id'],
                    cameraId=roidata['camera_info']['camera_id'],
                    pixelsize=pixelsize,
                    maskUrl=self.args['maskUrl']) for img in imgdata]

        if 'output_path' in self.args:
            self.args['output_path'] = jsongz.dump(
                self.tilespecs,
                self.args['output_path'],
                self.args['compress_output'])

    @property
    def tilespecs(self):
        tjs = [t.to_dict() for t in self.render_tspecs]
        return tjs


if __name__ == '__main__':
    gmod = GenerateEMTileSpecsModule()
    gmod.run()
