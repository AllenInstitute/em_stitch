import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float, Int, OutputFile, OutputDir, Str, Nested)
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


class GenerateEMTileSpecsParameters(ArgSchema):
    metafile = InputFile(
        required=True,
        description="metadata file containing TEMCA acquisition data")
    maskUrl = InputFile(
        required=False,
        default=None,
        missing=None,
        description="absolute path to image mask to apply")
    image_directory = InputDir(
        required=False,
        description=("directory used in determining absolute paths to images. "
                     "Defaults to parent directory containing metafile "
                     "if omitted."))
    maximum_intensity = Int(
        required=False, default=255,
        description=("intensity value to interpret as white"))
    minimum_intensity = Int(
        required=False, default=0,
        description=("intensity value to interpret as black"))
    z = Float(
        required=False,
        default=0,
        description=("z value"))
    sectionId = Str(
        required=False,
        description=("sectionId to apply to tiles during ingest.  "
                     "If unspecified will default to a string "
                     "representation of the float value of z_index."))
    output_path = OutputFile(
        required=False,
        description="directory for output files")



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
        required=True,
        missing="",
        description="json of tilespecs")
    match_file = InputFile(
        required=True,
        missing="",
        description="json of matches")
    outfile = Str(
        required=True,
        missing="tmp_transform_out.json",
        description=("File to which json output of lens correction "
                     "(leaf TransformSpec) is written"))
    regularization = Nested(regularization, missing={})
    good_solve = Nested(good_solve_criteria, missing={})
    output_dir = OutputDir(
        required=False,
        description="directory for output files")


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
    write_pdf = Boolean(
        required=True,
        missing=False,
        default=False,
        description="output pdf plots (was slow on windows)")

