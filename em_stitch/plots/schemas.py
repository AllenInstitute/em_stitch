from argschema import ArgSchema
from argschema.fields import (
        InputDir, InputFile, Str, List, Boolean, Int,
        OutputFile, Float)


class LensQuiverSchema(ArgSchema):
    transform_list = List(
        InputFile,
        required=True,
        description=("list of paths to transforms "
                     " or resolved tiles"))
    subplot_shape = List(
        Int,
        required=True,
        missing=[1, 1],
        default=[1, 1],
        description="sets the subplots for multiple plots")
    n_grid_pts = Int(
        required=True,
        missing=20,
        default=20,
        description="number of pts per axis for quiver grid")
    fignum = Int(
        required=True,
        missing=None,
        default=None,
        description="passed to plt.subplots to number the figure")
    arrow_scale = Float(
        required=True,
        missing=1.0,
        default=1.0,
        description="relative scale of arrows to axes")
    show = Boolean(
        required=True,
        missing=True,
        default=True,
        description=("show on screen?"))
    pdf_out = OutputFile(
        required=True,
        missing='./lens_corr_plots.pdf',
        default='./lens_corr_plots.pdf',
        description="where to write the pdf output")


class MontagePlotsSchema(ArgSchema):
    collection_path = InputFile(
        required=True,
        description="point matches from here")
    resolved_path = InputFile(
        required=True,
        description="resolved tiles from here")
    save_json_path = OutputFile(
        required=True,
        missing=None,
        default=None,
        description=("save residuals to this path if not None"))
    save_plot_path = OutputFile(
        required=True,
        missing=None,
        default=None,
        description=("save plot to this path if not None"))
    make_plot = Boolean(
        required=True,
        missing=True,
        default=True,
        description=("make a plot?"))
    show = Boolean(
        required=True,
        missing=True,
        default=True,
        description=("show on screen?"))
    pdf_out = OutputFile(
        required=True,
        missing=None,
        default=None,
        description="where to write the pdf output")


class ViewMatchesSchema(ArgSchema):
    collection_path = InputFile(
        required=False,
        description="if specified, will read collection from here")
    collection_basename = Str(
        required=True,
        missing="collection.json",
        default="collection.json",
        description=("basename for collection file if collection_path"
                     " not specified. will also check for .json.gz"))
    data_dir = InputDir(
        required=True,
        description=("directory containing image files. Will also be dir"
                     " dir for collection path, if not otherwise specified"))
    resolved_tiles = List(
        Str,
        required=True,
        missing=[
            "resolvedtiles.json.gz",
            "resolvedtiles_input.json.gz"],
        description=("will take the transform from the first file"
                     " matching this list, if possible"))
    transform_file = InputFile(
        required=False,
        description=("if provided, will get lens correction transform "
                     " from here"))
    view_all = Boolean(
        required=True,
        missing=False,
        default=False,
        description=("will plot all the pair matches. can be useful "
                     "for lens correction to file. probably not desirable "
                     "for montage"))
    show = Boolean(
        required=True,
        missing=True,
        default=True,
        description=("show on screen?"))
    match_index = Int(
        required=True,
        missing=0,
        default=0,
        description=("which index of self.matches to plot"))
    pdf_out = OutputFile(
        required=True,
        missing='./view_matches_output.pdf',
        default='./view_matches_output.pdf',
        description="where to write the pdf output")
