import os
import pathlib

import numpy
import renderapi

from .base import BaseGenerateEMTilespecsModule


class GenerateEMTilespecsModuleLegacy(
    BaseGenerateEMTilespecsModule
):
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
    def resolvedtiles_from_metadata(
        cls, md, image_directory, z,
        sectionId=None,
        minimum_intensity=0, maximum_intensity=255,
        maskUrl=None, width=None, height=None
    ):
        # TODO change between temcadb and pytemca format

        roidata = md[0]["metadata"]
        imgdata = md[1]["data"]
        
        img_coords = [
            cls.image_coords_from_stage(
                img['img_meta']['stage_pos'],
                img['img_meta']['pixel_size_x_move'],
                img['img_meta']['pixel_size_y_move'],
                numpy.radians(img['img_meta']['angle'])) for img in imgdata
        ]
        
        x_min, y_min = numpy.min(numpy.array(img_coords), axis=0)

        if width is None:
            width = roidata["camera_info"]["width"]
        if height is None:
            height = roidata["camera_info"]["height"]

        pixelsize = roidata['calibration']['highmag']['x_nm_per_pix']

        tspecs = [
            cls.ts_from_imgdata_tileId(
                img, image_directory,
                x_pos - x_min,
                y_pos - y_min,
                cls.tileId_from_basename(img["img_path"]),
                width=width, height=height,
                scopeId=roidata["temca_id"],
                cameraId=roidata["camera_info"]["camera_id"],
                pixelsize=pixelsize,
                minint=minimum_intensity,
                maxint=maximum_intensity,
                z=z, sectionId=sectionId,
                maskUrl=maskUrl
            ) for (img, (x_pos, y_pos))
            in zip(imgdata, img_coords)
        ]

        rts = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=tspecs,
            transformList=[]
        )

        return rts


# TODO load_tspec_metadata
