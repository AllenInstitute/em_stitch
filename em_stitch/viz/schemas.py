import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        Nested, Int, Str, Boolean, List)
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


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
        required=False,
        default='5G',
        description='string describing java heap memory (default 5G)')
    validate_client = Boolean(
        required=False,
        default=False,
        description="will avoid problems on windows if we use use_rest")


class common_schema(ArgSchema):
    client_mount_or_map = Str(
        required=True,
        default="/data/em-131fs3",
        missing="/data/em-131fs3",
        description=("where the client sees the robocopied destination"
                     " windows example 'Q:'"))
    fdir = Str(
        required=True,
        description="appended to client_mount to find files")


class SetPermissionsSchema(common_schema):
    dir_setting = Str(
        required=True,
        default=None,
        missing=None,
        description=('setting to recursively apply to dirs. '
                     'robocopy writes at 755 and we want at least 775.'))
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
        description=('setting to apply to files'
                     'robocopy writes at 744 and we want 777.'))


class resolved_schema(ArgSchema):
    resolved_file = Str(
        required=True,
        missing=None,
        description="basename of resolved_file")


class UpdateUrlSchema(common_schema, resolved_schema):
    backup_copy = Boolean(
        required=False,
        default=True,
        description="backup the resolved tilespecs file before overwriting")
    server_mount = Str(
        required=True,
        default="/data/em-131fs3",
        missing="/data/em-131fs3",
        description="where the render server sees the image files")
    image_directory = Str(
        required=False,
        missing=None,
        default=None,
        description=(" if missing, imageUrls are at server_mount/fdir/"
                     " if not missing, this."))


class UploadToRenderSchema(common_schema, resolved_schema):
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
    collection_file = Str(
        required=False,
        missing=None,
        description="collection file basename")
    close_stack = Boolean(
        required=False,
        default=True,
        missing=True,
        description="close stack or not after upload")


class SetUpdateUploadSchema(
        SetPermissionsSchema,
        UpdateUrlSchema,
        UploadToRenderSchema):
    pass
