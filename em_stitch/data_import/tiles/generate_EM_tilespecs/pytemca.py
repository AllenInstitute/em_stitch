# TODO inhert from legacy, but with uri and optional lens correction
import pathlib

import renderapi
import uri_handler.uri_functions

from .legacy import GenerateEMTilespecsModuleLegacy


# TODO include md conversion to temcadb format
class GenerateEMTilespecsModule_URIPrefix(GenerateEMTilespecsModuleLegacy):
    @staticmethod
    def ts_from_imgdata_tileId(imgdata, img_prefix, x, y, tileId, 
                               minint=0, maxint=255, maskUrl=None,
                               width=3840, height=3840, z=None, sectionId=None,
                               scopeId=None, cameraId=None, pixelsize=None):
        raw_tforms = [renderapi.transform.AffineModel(B0=x, B1=y)]
        imageUrl = uri_handler.uri_functions.uri_join(
            img_prefix, imgdata["img_path"])

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


# TODO load_tspec_metadata
