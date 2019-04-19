import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        InputDir, InputFile, Float, Nested,
        Int, OutputFile, Str, Boolean, List)
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
        description=("tilespecs will be .json or .json.gz"))


class RenderClientParameters(DefaultSchema):
    host = Str(
        required=True, description='render host')
    port = Int(
        required=True, description='render post integer')
    owner = Str(
        required=True, description='render default owner')
    project = Str(
        required=True, description='render default project')
    client_scripts = Str(
        required=True, description='path to render client scripts')
    memGB = Str(
        required=False, default='5G',
        description='string describing java heap memory (default 5G)')


class UploadToRenderSchema(ArgSchema):
    render = Nested(
        RenderClientParameters,
        required=True,
        description="parameters to connect to render server")
    stack = Str(
        required=False,
        description="name of destination stack in render")
    collection = Str(
        required=False,
        description="name of destination collection in render")
    resolved_file = InputFile(
        required=False,
        missing=None,
        description="stack name")
    collection_file = InputFile(
        required=False,
        missing=None,
        description="collection name")
    close_stack = Boolean(
        required=False,
        default=True,
        missing=True,
        description="close stack or not after upload")


class UpdateFilepathSchema(ArgSchema):
    backup_copy = Boolean(
        required=False,
        default=True,
        description="backup the resolved tilespecs file before overwriting")
    resolved_file = InputFile(
        required=True,
        missing=None,
        description="stack name")
    image_directory = InputDir(
        required=False,
        missing=None,
        description=("directory where images and masks are now"
                     " defaults to dirname or resolved_file"))


class ChangePermissionsSchema(ArgSchema):
    data_dir = InputDir(
        required=True,
        description="directory for changing permissions")
    dir_setting = Str(
        required=True,
        default=None,
        missing=None,
        description='setting to recursively apply to dirs')
    file_exts = List(
        Str,
        required=True,
        missing=None,
        default=None,
        description="file extensions to change")
    file_setting = Str(
        required=True,
        default='777',
        missing='777',
        description='setting to apply to files')
