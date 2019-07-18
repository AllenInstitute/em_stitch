import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
import marshmallow as mm
from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float, List,
        Int, OutputDir, Nested, Str, Dict)
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


class regularization(DefaultSchema):
    default_lambda = Float(
        required=False,
        default=0.5,
        missing=0.5,
        description="regularization factor")
    translation_factor = Float(
        required=False,
        default=0.005,
        missing=0.005,
        description="transaltion factor")
    lens_lambda = Float(
        required=False,
        default=0.005,
        missing=0.005,
        description="regularization for lens parameters")


class good_solve_criteria(DefaultSchema):
    error_mean = Float(
        required=False,
        default=0.2,
        missing=0.2,
        description="maximum error mean [pixels]")
    error_std = Float(
        required=False,
        default=2.0,
        missing=2.0,
        description="maximum error std [pixels]")
    scale_dev = Float(
        required=False,
        default=0.1,
        missing=0.1,
        description="maximum allowed scale deviation from 1.0")


class MeshLensCorrectionSchema(ArgSchema):
    nvertex = Int(
        required=False,
        default=1000,
        missinf=1000,
        description="maximum number of vertices to attempt")
    tilespec_file = InputFile(
        required=False,
        description="path to json of tilespecs")
    tilespecs = List(
        Dict,
        required=False,
        description="list of dict of tilespecs")
    match_file = InputFile(
        required=False,
        description="path to json of matches")
    matches = List(
        Dict,
        required=False,
        description="list of dict of matches")
    regularization = Nested(regularization, missing={})
    good_solve = Nested(good_solve_criteria, missing={})
    output_dir = OutputDir(
        required=False,
        description="directory for output files")
    outfile = Str(
        required=False,
        description=("Basename to which resolved json output of "
                     "lens correction is written"))
    compress_output = Boolean(
        required=False,
        missing=True,
        default=True,
        description=("tilespecs will be .json or .json.gz"))
    timestamp = Boolean(
        required=False,
        missing=False,
        default=False,
        description="add a timestamp to basename output")

    @mm.post_load
    def one_of_two(self, data):
        for a, b in [
                ['tilespecs', 'tilespec_file'],
                ['matches', 'match_file']]:
            if (a in data) == (b in data):  # xor
                raise mm.ValidationError(
                        'must specify one and only one of %s or %s' % (a, b))


class LensCorrectionSchema(ArgSchema):
    data_dir = InputDir(
        required=True,
        description="directory containing metafile, images, and matches")
    output_dir = OutputDir(
        required=False,
        description="directory for output files")
    mask_file = InputFile(
        required=False,
        default=None,
        missing=None,
        description="mask to apply to each tile")
    nvertex = Int(
        required=False,
        default=1000,
        missinf=1000,
        description="maximum number of vertices to attempt")
    ransac_thresh = Float(
        required=False,
        default=5.0,
        missing=5.0,
        description="ransac outlier threshold")
    regularization = Nested(regularization, missing={})
    good_solve = Nested(good_solve_criteria, missing={})
    ignore_match_indices = List(
        Int,
        required=False,
        default=None,
        missing=None,
        description=("debug feature for ignoring certain indices"
                     " of the match collection"))
    compress_output = Boolean(
        required=False,
        missing=True,
        default=True,
        description=("tilespecs will be .json or .json.gz"))
    timestamp = Boolean(
        required=False,
        missing=False,
        default=False,
        description="add a timestamp to basename output")
