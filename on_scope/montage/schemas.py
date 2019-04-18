import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float, OutputDir)
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


class MontageSolverSchema(ArgSchema):
    data_dir = InputDir(
        required=True,
        description="directory containing metafile, images, and matches")
    output_dir = OutputDir(
        required=False,
        missing=None,
        default=None,
        description="directory for output files")
    read_transform_from_meta = Boolean(
        required=False,
        missing=True,
        default=True,
        description="read lens correction transform from metafile")
    ref_transform = InputFile(
        required=False,
        missing=None,
        default=None,
        description="transform json")
    ransacReprojThreshold = Float(
        required=False,
        missing=10.0,
        default=10.0,
        description=("passed into cv2.estimateAffinePartial2D()"
                     "for RANSAC filtering of montage template matches"))
    compress_output = Boolean(
        required=False,
        missing=True,
        default=True,
        description=("tilespecs will be .json or .json.gz"))
