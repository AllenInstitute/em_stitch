import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from marshmallow import post_load, ValidationError
from argschema import ArgSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float, OutputDir, List, Str)
import os
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
    solver_templates = List(
        Str,
        required=True,
        description="input json basenames for the solver args")
    solver_template_dir = InputDir(
        required=True,
        description="location of the templates for the solver")

    @post_load
    def check_solver_inputs(self, data):
        for args in data['solver_templates']:
            argpath = os.path.join(data['solver_template_dir'], args)
            if not os.path.isfile(argpath):
                raise ValidationError(
                        "solver arg file doesn't exist: %s" % argpath)
