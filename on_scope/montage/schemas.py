import warnings
from marshmallow.warnings import ChangedInMarshmallow3Warning
from argschema import ArgSchema
from argschema.fields import (
        Boolean, InputDir, InputFile, Float,
        Int, OutputDir, Str)
warnings.simplefilter(
        action='ignore',
        category=ChangedInMarshmallow3Warning)


class MetaToMontageAndCollectionSchema(ArgSchema):
    data_dir = InputDir(
        required=True,
        description="directory containing metafile, images, and matches")
    output_dir = OutputDir(
        required=False,
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


class MontagePlotsSchema(ArgSchema):
    output_dir = OutputDir(
        required=True,
        description="directory containing metafile, images, and matches")
    stack = Str(
        required=True,
        description="stack name")
    collection = Str(
        required=True,
        description="collection name")
    sectionId = Str(
        required=True,
        description='sectionId/groupId')
    z = Int(
        required=True,
        description='z value in stack')
    make_plot = Boolean(
        required=True,
        default=True,
        missing=True,
        description="make the plot and save it")
    save_json = Boolean(
        required=True,
        default=True,
        missing=True,
        description="save the json of the residuals")
