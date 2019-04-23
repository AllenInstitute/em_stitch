import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
import marshmallow as mm
from argschema import ArgSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float, OutputDir, List, Str, Dict)
import os
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


class MontageSolverSchema(ArgSchema):
    data_dir = InputDir(
        required=False,
        description="directory containing metafile, images, and matches")
    metafile = InputFile(
        required=False,
        description=("fullpath to metafile. Helps in the case of multiple"
                     " metafiles in one directory. data_dir will take "
                     " os.path.dirname(metafile)"))
    output_dir = OutputDir(
        required=False,
        missing=None,
        default=None,
        description="directory for output files")
    read_transform_from = Str(
        required=False,
        missing='metafile',
        default='metafile',
        validator=mm.validate.OneOf(['metafile', 'reffile', 'dict']),
        description="3 possible ways to read in the reference transform")
    ref_transform = InputFile(
        required=False,
        missing=None,
        default=None,
        description="transform json")
    ref_transform_dict = Dict(
        require=False,
        missing=None,
        description="transform in from memory")
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

    @mm.post_load
    def check_solver_inputs(self, data):
        for args in data['solver_templates']:
            argpath = os.path.join(data['solver_template_dir'], args)
            if not os.path.isfile(argpath):
                raise mm.ValidationError(
                        "solver arg file doesn't exist: %s" % argpath)

    @mm.post_load
    def check_metafile(self, data):
        if ('data_dir' not in data) & ('metafile' not in data):
            raise mm.ValidationError(" must specify either data_dir"
                                     " or metafile")
