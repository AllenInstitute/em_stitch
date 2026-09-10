import pathlib

import numpy
import renderapi
import uri_handler.uri_functions

from .base import (
    BaseGenerateEMTilespecsModule,
    BaseGenerateEMTilespecsParameters
)


def removeprefix(s, prefix):
    try:
        # python 3.9+ has this
        return s.removeprefix(prefix)
    except AttributeError:
        if s.startswith(prefix):
            return s[len(prefix):]
        return s


class GenerateEMTilespecsParamsRowColumn(BaseGenerateEMTilespecsParameters):
    pass


class GenerateEMTilespecsModuleRowColumn(BaseGenerateEMTilespecsModule):
    default_schema = GenerateEMTilespecsParamsRowColumn

    @staticmethod
    def get_image_width_and_height(img_uri):
        raise NotImplementedError

    @staticmethod
    def metadata_from_basenames(basenames):
        md = []
        for name in basenames:
            row_str, col_str = pathlib.Path(name).stem.split("_")[-2:]
            row = int(removeprefix(row_str, "r"))
            col = int(removeprefix(col_str, "c"))
            md.append([name, row, col])
        return md

    @staticmethod
    def image_coords_from_row_col(
            row_int, column_int,
            width, height,
            overlap_factor_row,
            overlap_factor_column):
        x = width * column_int * (1 - overlap_factor_column)
        y = height * row_int * (1 - overlap_factor_row)
        return x, y

    @classmethod
    def resolvedtiles_from_metadata(
        cls, md, img_prefix, z,
        sectionId=None,
        minimum_intensity=0, maximum_intensity=255,
        maskUrl=None, width=None, height=None,
        rotation=None, resX=None, resY=None,
        overlap_row=0.1, overlap_col=0.1,
        flip_y=False, flip_x=False
    ):
        if maskUrl is not None:
            raise NotImplementedError("masking not available")

        img_names = [i[0] for i in md]

        img_uris = [
            uri_handler.uri_functions.uri_join(img_prefix, img_name)
            for img_name in img_names
        ]

        # this metadata format doesn't include width or height
        if width is None or height is None:
            width, height = cls.get_image_width_and_height(img_uris[0])

        rows_columns = [(i[1], i[2]) for i in md]

        img_coords = [
            cls.image_coords_from_row_col(
                r, c,
                width, height,
                overlap_row, overlap_col
            ) for r, c in rows_columns
        ]
        minX, minY = numpy.min(numpy.array(img_coords), axis=0)
        maxX, maxY = numpy.max(numpy.array(img_coords), axis=0)

        pixelsize = float(numpy.linalg.norm([resX, resY]) / numpy.sqrt(2))

        tspecs = []

        for img_uri, img_coord, img_name, (row, col) in zip(
                img_uris, img_coords, img_names, rows_columns):
            new_y = ((maxY - img_coord[1]) if flip_y else img_coord[1] - minY)
            new_x = ((maxX - img_coord[0]) if flip_x else img_coord[0] - minX)
            raw_tforms = [
                renderapi.transform.AffineModel(
                    B0=new_x,
                    B1=new_y
                )
            ]
            ip = renderapi.image_pyramid.ImagePyramid()
            ip[0] = renderapi.image_pyramid.MipMap(
                imageUrl=img_uri, maskUrl=maskUrl)

            ts = renderapi.tilespec.TileSpec(
                tileId=pathlib.Path(img_name).stem, z=z,
                width=width, height=height,
                minint=minimum_intensity,
                maxint=maximum_intensity,
                tforms=raw_tforms,
                imagePyramid=ip,
                sectionId=sectionId,
                imageRow=row,
                imageCol=col,
                stageX=img_coord[0],
                stageY=img_coord[1],
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
