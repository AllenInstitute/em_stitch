# TODO tilEM interchange format
import os
import pathlib

import numpy

import renderapi
import uri_handler.uri_functions

from .base import (
    BaseGenerateEMTilespecsModule,
    BaseGenerateEMTilespecsParameters
)


class GenerateEMTilespecsParamsTilEMTiltSeries(
        BaseGenerateEMTilespecsParameters):
    pass


def _get_value_or_error(
        d, key, override_value=None,
        error_report_param=None, error_msg_override=None):
    if override_value is not None:
        return override_value
    try:
        value = d[key]
        return value
    except KeyError:
        err_report_param = error_report_param or key
        err_msg = error_msg_override or f"{err_report_param} not available"
        raise ValueError(err_msg)


class GenerateEMTilespecsModuleTilEMTiltSeries(BaseGenerateEMTilespecsModule):
    default_schema = GenerateEMTilespecsParamsTilEMTiltSeries

    @staticmethod
    def get_image_width_and_height(img_uri):
        raise NotImplementedError

    @classmethod
    def resolvedtiles_from_metadata(
        cls, md, img_prefix, z,
        sectionId=None,
        minimum_intensity=0, maximum_intensity=255,
        maskUrl=None, width=None, height=None,
        rotation=None, resX=None, resY=None,
        img_ext=".tiff", y_pos=True, x_pos=False
    ):
        if maskUrl is not None:
            raise NotImplementedError("masking not available")

        rotation = _get_value_or_error(md, "rotation_angle", rotation, "rotation")
        resX = _get_value_or_error(md, "pixel_size", resX, "resX")
        resY = _get_value_or_error(md, "pixel_size", resY, "resY")

        img_coords = [
            cls.image_coords_from_stage(
                img_d["stage_position"],
                resX, resY,
                numpy.radians(rotation)
            ) for img_d in md["tiles"].values()
        ]
        minX, minY = numpy.min(numpy.array(img_coords), axis=0)
        maxX, maxY = numpy.max(numpy.array(img_coords), axis=0)

        img_names = [f"{img_name}{img_ext}" for img_name in md["tiles"].keys()]

        img_uris = [
            uri_handler.uri_functions.uri_join(img_prefix, img_name)
            for img_name in img_names
        ]

        # this metadata format doesn't include width or height
        if width is None or height is None:
            width, height = cls.get_image_width_and_height(img_uris[0])

        pixelsize = float(numpy.linalg.norm([resX, resY]) / numpy.sqrt(2))

        tspecs = []

        for img_uri, img_coord, img_name, img_d in zip(img_uris, img_coords,
                                                       img_names,
                                                       md["tiles"].values()):
            y_coord = (
                maxY - img_coord[1] if y_pos else img_coord[1] - minY
            )
            x_coord = (
                maxX - img_coord[0] if x_pos else img_coord[0] - minX
            )
            raw_tforms = [
                renderapi.transform.AffineModel(
                    B0=x_coord,
                    B1=y_coord  # - 2 * minY   # flip y for this stage
                )
            ]
            ip = renderapi.image_pyramid.ImagePyramid()
            ip[0] = renderapi.image_pyramid.MipMap(
                imageUrl=img_uri, maskUrl=maskUrl)

            ts = renderapi.tilespec.TileSpec(
                tileId=img_d["tile_id"], z=z,
                width=width, height=height,
                minint=minimum_intensity,
                maxint=maximum_intensity,
                tforms=raw_tforms,
                imagePyramid=ip,
                sectionId=sectionId,
                imageRow=img_d["raster_position"][0],
                imageCol=img_d["raster_position"][1],
                stageX=img_d["stage_position"][0],
                stageY=img_d["stage_position"][1],
                rotation=rotation,
                pixelsize=pixelsize
            )
            tspecs.append(ts)

        rts = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=tspecs,
            transformList=[]
        )

        return rts


class GenerateEMTilespecsModuleNewTilEMTiltSeries(BaseGenerateEMTilespecsModule):
    default_schema = GenerateEMTilespecsParamsTilEMTiltSeries

    @staticmethod
    def get_image_width_and_height(img_uri):
        raise NotImplementedError

    @classmethod
    def resolvedtiles_from_metadata(
        cls, md, img_prefix, z,
        sectionId=None,
        minimum_intensity=0, maximum_intensity=255,
        maskUrl=None, width=None, height=None,
        rotation=None, resX=None, resY=None,
        img_ext=".tiff", y_pos=True, x_pos=False
    ):
        if maskUrl is not None:
            raise NotImplementedError("masking not available")

        calibration_md = md.get("calibration", {})
        rotation = _get_value_or_error(calibration_md, "rotation_angle", rotation, "rotation")
        resX = _get_value_or_error(calibration_md, "pixel_size", resX, "resX")
        resY = _get_value_or_error(calibration_md, "pixel_size", resY, "resY")

        # rotation = _get_value_or_error(md, "rotation_angle", rotation, "rotation")
        # resX = _get_value_or_error(md, "pixel_size", resX, "resX")
        # resY = _get_value_or_error(md, "pixel_size", resY, "resY")


        img_coords = [
            cls.image_coords_from_stage(
                img_d["stage_position"],
                resX, resY,
                numpy.radians(rotation)
            ) for img_d in md["tiles"]
                # ) for img_d in md["tiles"].values()  # img_d
        ]
        minX, minY = numpy.min(numpy.array(img_coords), axis=0)
        maxX, maxY = numpy.max(numpy.array(img_coords), axis=0)

        # they changed the tiles field AGAIN!!!!
        # tile_ids = [*md["tiles"].keys()]
        tile_ids = [t["tile_id"] for t in md["tiles"]]

        # img_names = [f"{img_name}{img_ext}" for img_name in md["tiles"].keys()]
        img_names = [t["filename"] for t in md["tiles"]]

        img_uris = [
            uri_handler.uri_functions.uri_join(img_prefix, img_name)
            for img_name in img_names
        ]

        reported_width = None
        reported_height = None
        try:
            reported_width, reported_height = md.get("acquisition_params", {})["tile_size"]
        except KeyError:
            # this metadata format doesn't include width or height
            if width is None or height is None:
                reported_width, reported_height = cls.get_image_width_and_height(img_uris[0])
        width = width or reported_width
        height = height or reported_height

        # pixelsize = float(numpy.linalg.norm([resX, resY]) / numpy.sqrt(2))
        pixelsize = resX

        tspecs = []

        for tile_id, img_uri, img_coord, img_name, img_d in zip(
                                                       tile_ids, img_uris,
                                                       img_coords,
                                                       img_names,
                                                       md["tiles"]):
                                                       # md["tiles"].values()):
            y_coord = (
                maxY - img_coord[1] if y_pos else img_coord[1] - minY
            )
            x_coord = (
                maxX - img_coord[0] if x_pos else img_coord[0] - minX
            )
            raw_tforms = [
                renderapi.transform.AffineModel(
                    B0=x_coord,
                    B1=y_coord
                )
            ]
            ip = renderapi.image_pyramid.ImagePyramid()
            ip[0] = renderapi.image_pyramid.MipMap(
                imageUrl=img_uri, maskUrl=maskUrl)

            # this changed in metadata format:
            # row, col = img_d["raster_position"]
            row = img_d["row"]
            col = img_d["column"]
            ts = renderapi.tilespec.TileSpec(
                tileId=tile_id, z=z,
                width=width, height=height,
                minint=minimum_intensity,
                maxint=maximum_intensity,
                tforms=raw_tforms,
                imagePyramid=ip,
                sectionId=sectionId,
                imageRow=row,
                imageCol=col,
                stageX=img_d["stage_position"][0],
                stageY=img_d["stage_position"][1],
                rotation=rotation,
                pixelsize=pixelsize
            )
            tspecs.append(ts)

        rts = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=tspecs,
            transformList=[]
        )

        return rts

# TODO load_tspec_metadata

# TODO additional inputs
