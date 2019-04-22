import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.fields import (
        InputDir, InputFile, Float,
        Int, OutputFile, Str, Boolean)
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
    compress_output = Boolean(
        required=False,
        missing=True,
        default=True,
        escription=("tilespecs will be .json or .json.gz"))
