import os
from argschema import (
    ArgSchemaParser, ArgSchema
)
import numpy

from bigfeta import jsongz


class BaseGenerateEMTilespecsParameters(ArgSchema):
    pass


class BaseGenerateEMTilespecsModule(ArgSchemaParser):
    default_schema = BaseGenerateEMTilespecsParameters

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
    def sectionId_from_z(z):
        return str(float(z))

    @classmethod
    def ts_from_metadata(cls, *args, **kwargs):
        rts = cls.resolvedtiles_from_metadata(*args, **kwargs)
        tspecs = rts.tilespecs
        return tspecs

    def load_tspec_metadata(self, *args, **kwargs):
        raise NotImplementedError

    def get_coordinates_from_metadata(self, md, *args, **kwargs):
        # TODO get coordinates, include option for relative coords
        raise NotImplementedError

    def get_tilespecs_from_metadata(self, md, *args, **kwargs):
        raise NotImplementedError

    # TODO include resolvedtiles?  include shared transform?
    def get_resolvedtiles_from_metadata(self, md, *args, **kwargs):
        raise NotImplementedError

    def run(self):
        md = self.load_tspec_metadata()
        tspecs = self.tilespecs_from_metadata(md)
        self.render_tspecs = tspecs

        if self.args.get("output_path"):
            _ = jsongz.dump(
                self.tilespecs,
                self.args["output_path"],
                self.args["compress_output"]
            )
                            
    @property
    def tilespecs(self):
        tjs = [t.to_dict() for t in self.render_tspecs]
        return tjs
