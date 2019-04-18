import numpy as np
import triangle
import scipy.optimize
from scipy.spatial import Delaunay
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
import renderapi
import copy
import os
import datetime
import cv2
from six.moves import urllib
from argschema import ArgSchemaParser
from .schemas import MeshLensCorrectionSchema
from .utils import remove_weighted_matches
from EMaligner import jsongz
import logging

# this is a modification of https://github.com/
# AllenInstitute/render-modules/blob/master/
# rendermodules/mesh_lens_correction/MeshAndSolveTransform.py
# that does not depend on render-modules nor
# on a running render server

logger = logging.getLogger()


class MeshLensCorrectionException(Exception):
    """Exception raised when there is a \
            problem creating a mesh lens correction"""
    pass


def condense_coords(matches):
    # condense point match structure into Nx2
    x = []
    y = []
    for m in matches:
        x += m['matches']['p'][0]
        x += m['matches']['q'][0]
        y += m['matches']['p'][1]
        y += m['matches']['q'][1]
    coords = np.transpose(np.vstack((np.array(x), np.array(y))))
    return coords


def smooth_density(coords, tile_width, tile_height, n):
    # n: area divided into nxn
    min_count = np.Inf
    for i in range(n):
        r = np.arange(
                (i*tile_width/n),
                ((i+1)*tile_width/n))
        for j in range(n):
            c = np.arange(
                    (j*tile_height/n),
                    ((j+1)*tile_height/n))
            ind = np.argwhere(
                    (coords[:, 0] >= r.min()) &
                    (coords[:, 0] <= r.max()) &
                    (coords[:, 1] >= c.min()) &
                    (coords[:, 1] <= c.max())).flatten()
            if ind.size < min_count:
                min_count = ind.size
    new_coords = []
    for i in range(n):
        r = np.arange(
                (i*tile_width/n),
                ((i+1)*tile_width/n))
        for j in range(n):
            c = np.arange(
                    (j*tile_height/n),
                    ((j+1)*tile_height/n))
            ind = np.argwhere(
                    (coords[:, 0] >= r.min()) &
                    (coords[:, 0] <= r.max()) &
                    (coords[:, 1] >= c.min()) &
                    (coords[:, 1] <= c.max())).flatten()
            a = np.arange(ind.size)
            np.random.shuffle(a)
            ind = ind[a[0:min_count]]
            new_coords.append(coords[ind])
    return np.concatenate(new_coords)


def approx_snap_contour(contour, width, height, epsilon=20, snap_dist=5):
    # approximate contour within epsilon pixels,
    # so it isn't too fine in the corner
    # and snap to edges
    approx = cv2.approxPolyDP(contour, epsilon, True)
    for i in range(approx.shape[0]):
        for j in [0, width]:
            if np.abs(approx[i, 0, 0] - j) <= snap_dist:
                approx[i, 0, 0] = j
        for j in [0, height]:
            if np.abs(approx[i, 0, 1] - j) <= snap_dist:
                approx[i, 0, 1] = j
    return approx


def create_PSLG(tile_width, tile_height, maskUrl):
    # define a PSLG for triangle
    # http://dzhelil.info/triangle/definitions.html
    # https://www.cs.cmu.edu/~quake/triangle.defs.html#pslg
    if maskUrl is None:
        vertices = np.array([
                [0, 0],
                [0, tile_height],
                [tile_width, tile_height],
                [tile_width, 0]])
        segments = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0]])
    else:
        mpath = urllib.parse.unquote(
                    urllib.parse.urlparse(maskUrl).path)
        im = cv2.imread(mpath, 0)
        contours, h = cv2.findContours(
                im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx = approx_snap_contour(contours[0], tile_width, tile_height)
        vertices = np.array(approx).squeeze()
        segments = np.zeros((vertices.shape[0], 2))
        segments[:, 0] = np.arange(vertices.shape[0])
        segments[:, 1] = segments[:, 0] + 1
        segments[-1, 1] = 0
    bbox = {}
    bbox['vertices'] = vertices
    bbox['segments'] = segments

    return bbox


def calculate_mesh(a, bbox, target, get_t=False):
    t = triangle.triangulate(bbox, 'pqa%0.1f' % a)
    if get_t:
        # scipy.Delaunay has nice find_simplex method,
        # but, no obvious way to iteratively refine meshes, like triangle
        # numbering is different, so, switch here
        return Delaunay(t['vertices'])
    return target - len(t['vertices'])


def force_vertices_with_npoints(area_par, bbox, coords, npts):
    fac = 1.02
    count = 0
    max_iter = 20
    while True:
        t = calculate_mesh(
                area_par,
                bbox,
                None,
                get_t=True)
        pt_count = count_points_near_vertices(t, coords)
        if pt_count.min() >= npts:
            break
        area_par *= fac
        count += 1
        if np.mod(count, 2) == 0:
            fac += 0.5
        if np.mod(count, max_iter) == 0:
            e = ("did not meet vertex requirement "
                 "after %d iterations" % max_iter)
            raise MeshLensCorrectionException(e)
    return t, area_par


def find_delaunay_with_max_vertices(bbox, nvertex):
    # find bracketing values
    a1 = a2 = 1e6
    t1 = calculate_mesh(a1, bbox, nvertex)
    afac = np.power(10., -np.sign(t1))
    while (
            np.sign(t1) ==
            np.sign(calculate_mesh(a2, bbox, nvertex))
          ):
        a2 *= afac
    val_at_root = -1
    nvtweak = nvertex
    while val_at_root < 0:
        a = scipy.optimize.brentq(
                calculate_mesh,
                a1,
                a2,
                args=(bbox, nvtweak, ))
        val_at_root = calculate_mesh(a, bbox, nvertex)
        a1 = a * 2
        a2 = a * 0.5
        nvtweak -= 1
    mesh = calculate_mesh(a, bbox, None, get_t=True)
    return mesh, a


def compute_barycentrics(coords, mesh):
    # https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates
    triangle_indices = mesh.find_simplex(coords)
    vt = np.vstack((
        np.transpose(mesh.points),
        np.ones(mesh.points.shape[0])))
    mt = np.vstack((np.transpose(coords), np.ones(coords.shape[0])))
    bary = np.zeros((3, coords.shape[0]))
    Rinv = []
    for tri in mesh.simplices:
        Rinv.append(np.linalg.inv(vt[:, tri]))
    for i in range(mesh.nsimplex):
        ind = np.argwhere(triangle_indices == i).flatten()
        bary[:, ind] = Rinv[i].dot(mt[:, ind])
    return np.transpose(bary), triangle_indices


def count_points_near_vertices(t, coords):
    flat_tri = t.simplices.flatten()
    flat_ind = np.repeat(np.arange(t.nsimplex), 3)
    v_touches = []
    for i in range(t.npoints):
        v_touches.append(flat_ind[np.argwhere(flat_tri == i)])
    pt_count = np.zeros(t.npoints)
    found = t.find_simplex(coords, bruteforce=True)
    for i in range(t.npoints):
        for j in v_touches[i]:
            pt_count[i] += np.count_nonzero(found == j)
    return pt_count


def create_regularization(ncols, ntiles, defaultL, transL, lensL):
    # regularizations: [affine matrix, translation, lens]
    reg = np.ones(ncols).astype('float64') * defaultL
    reg[0: (ntiles * 3)] *= transL
    reg[(ntiles * 3):] *= lensL
    rmat = sparse.eye(reg.size, dtype='float64', format='csr')
    rmat.data = reg
    return rmat


def create_thinplatespline_tf(
        args, mesh, solution,
        lens_dof_start, common_transform,
        logger,
        compute_affine=False):

    dst = np.zeros_like(mesh.points)
    dst[:, 0] = mesh.points[:, 0] + solution[0][lens_dof_start:]
    dst[:, 1] = mesh.points[:, 1] + solution[1][lens_dof_start:]

    if common_transform is not None:
        # average affine result
        dst = common_transform.tform(dst)

    transform = renderapi.transform.ThinPlateSplineTransform()
    transform.estimate(mesh.points, dst, computeAffine=compute_affine)
    npts0 = transform.srcPts.shape[1]
    transform = transform.adaptive_mesh_estimate(max_iter=1000)
    npts1 = transform.srcPts.shape[1]

    logger.info(
            "adaptive_mesh_estimate reduced control points from %d to %d" %
            (npts0, npts1))

    transform.transformId = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    transform.labels = None

    return transform


def new_specs_with_tf(ref_transform, tilespecs, transforms):
    newspecs = []
    for i in range(len(tilespecs)):
        newspecs.append(copy.deepcopy(tilespecs[i]))
        newspecs[-1].tforms.insert(0,
                                   renderapi.transform.ReferenceTransform(
                                    refId=ref_transform.transformId))
        newspecs[-1].tforms[1] = transforms[i]
    return newspecs


def solve(A, weights, reg, x0, b):
    ATW = A.transpose().dot(weights)
    K = ATW.dot(A) + reg
    K_factorized = factorized(K)
    solution = []

    i = 0
    for x in x0:
        Lm = reg.dot(x) + ATW.dot(b[:, i])
        i += 1
        solution.append(K_factorized(Lm))

    errx = A.dot(solution[0]) - b[:, 0]
    erry = A.dot(solution[1]) - b[:, 1]

    return solution, errx, erry


def report_solution(errx, erry, transforms, criteria):
    translation = np.array([tf.translation for tf in transforms])

    jresult = {}
    jresult['x_res_min'] = errx.min()
    jresult['x_res_max'] = errx.max()
    jresult['x_res_mean'] = errx.mean()
    jresult['x_res_std'] = errx.std()
    jresult['y_res_min'] = erry.min()
    jresult['y_res_max'] = erry.max()
    jresult['y_res_mean'] = erry.mean()
    jresult['y_res_std'] = erry.std()

    for k in jresult.keys():
        jresult[k] = np.round(jresult[k], 3)

    message = 'lens solver results [px]'
    for val in ['res']:
        for xy in ['x', 'y']:
            d = '\n%8s' % (xy + '_' + val + ' ')
            v = ''
            for calc in ['min', 'max', 'mean', 'std']:
                d += calc + ','
                k = xy + '_' + val + '_' + calc
                v += '%8.2f' % jresult[k]
            message += d + v

    return translation, jresult, message


def create_x0(nrows, tilespecs):
    ntiles = len(tilespecs)
    x0 = []
    x0.append(np.zeros(nrows).astype('float64'))
    x0.append(np.zeros(nrows).astype('float64'))
    x0[0][0:ntiles] = np.zeros(ntiles)
    x0[1][0:ntiles] = np.zeros(ntiles)
    for i in range(ntiles):
        x0[0][i] = tilespecs[i].tforms[0].B0
        x0[1][i] = tilespecs[i].tforms[0].B1
    return x0


def create_A(matches, tilespecs, mesh):
    # let's assume translation halfsize
    dof_per_tile = 1
    dof_per_vertex = 1
    vertex_per_patch = 3
    nnz_per_row = 2*(dof_per_tile + vertex_per_patch * dof_per_vertex)
    nrows = sum([len(m['matches']['p'][0]) for m in matches])
    nd = nnz_per_row*nrows
    lens_dof_start = dof_per_tile*len(tilespecs)

    data = np.zeros(nd).astype('float64')
    b = np.zeros((nrows, 2)).astype('float64')
    indices = np.zeros(nd).astype('int64')
    indptr = np.zeros(nrows+1).astype('int64')
    indptr[1:] = np.arange(1, nrows+1)*nnz_per_row
    weights = np.ones(nrows).astype('float64')

    unique_ids = np.array(
            [t.tileId for t in tilespecs])

    # nothing fancy here, row-by-row
    offset = 0
    rows = 0
    for mi in range(len(matches)):
        m = matches[mi]
        pindex = np.argwhere(unique_ids == m['pId'])
        qindex = np.argwhere(unique_ids == m['qId'])

        npoint_pairs = len(m['matches']['q'][0])
        # get barycentric coordinates ready
        pcoords = np.transpose(
                np.vstack(
                    (m['matches']['p'][0],
                     m['matches']['p'][1])
                    )).astype('float64')
        qcoords = np.transpose(
                np.vstack(
                    (m['matches']['q'][0],
                     m['matches']['q'][1])
                    )).astype('float64')

        b[rows: (rows + pcoords.shape[0])] = qcoords - pcoords
        rows += pcoords.shape[0]
        pbary = compute_barycentrics(pcoords, mesh)
        qbary = compute_barycentrics(qcoords, mesh)

        mstep = np.arange(npoint_pairs) * nnz_per_row + offset

        data[mstep + 0] = 1.0
        data[mstep + 1] = -1.0
        data[mstep + 2] = pbary[0][:, 0]
        data[mstep + 3] = pbary[0][:, 1]
        data[mstep + 4] = pbary[0][:, 2]
        data[mstep + 5] = -qbary[0][:, 0]
        data[mstep + 6] = -qbary[0][:, 1]
        data[mstep + 7] = -qbary[0][:, 2]

        indices[mstep + 0] = pindex
        indices[mstep + 1] = qindex
        indices[mstep + 2] = (lens_dof_start +
                              mesh.simplices[pbary[1][:]][:, 0])
        indices[mstep + 3] = (lens_dof_start +
                              mesh.simplices[pbary[1][:]][:, 1])
        indices[mstep + 4] = (lens_dof_start +
                              mesh.simplices[pbary[1][:]][:, 2])
        indices[mstep + 5] = (lens_dof_start +
                              mesh.simplices[qbary[1][:]][:, 0])
        indices[mstep + 6] = (lens_dof_start +
                              mesh.simplices[qbary[1][:]][:, 1])
        indices[mstep + 7] = (lens_dof_start +
                              mesh.simplices[qbary[1][:]][:, 2])

        offset += npoint_pairs*nnz_per_row

    A = csr_matrix((data, indices, indptr), dtype='float64')

    wts = sparse.eye(weights.size, format='csr', dtype='float64')
    wts.data = weights
    return A, wts, b, lens_dof_start


def create_transforms(ntiles, solution, get_common=False):
    rtransforms = []
    for i in range(ntiles):
        rtransforms.append(renderapi.transform.AffineModel(
                           B0=solution[0][i],
                           B1=solution[1][i]))

    if get_common:
        transforms = []
        # average without translations
        common = np.array([t.M for t in rtransforms]).mean(0)
        common[0, 2] = 0.0
        common[1, 2] = 0.0
        for r in rtransforms:
            transforms.append(renderapi.transform.AffineModel())
            transforms[-1].M = r.M.dot(np.linalg.inv(common))
        ctform = renderapi.transform.AffineModel()
        ctform.M = common
    else:
        ctform = None
        transforms = rtransforms

    return ctform, transforms


def estimate_stage_affine(t0, t1):
    src = np.array([t.tforms[0].translation for t in t0])
    dst = np.array([t.tforms[1].translation for t in t1])
    aff = renderapi.transform.AffineModel()
    aff.estimate(src, dst)
    return aff


class MeshAndSolveTransform(ArgSchemaParser):
    default_schema = MeshLensCorrectionSchema

    def run(self):
        jspecs = jsongz.load(self.args['tilespec_file'])
        self.tilespecs = np.array([
                renderapi.tilespec.TileSpec(json=j) for j in jspecs])

        self.matches = jsongz.load(self.args['match_file'])

        remove_weighted_matches(self.matches, weight=0.0)

        self.tile_width = self.tilespecs[0].width
        self.tile_height = self.tilespecs[0].height
        maskUrl = self.tilespecs[0].ip[0].maskUrl

        # condense coordinates
        self.coords = condense_coords(self.matches)
        nc0 = self.coords.shape[0]
        self.coords = smooth_density(
            self.coords,
            self.tile_width,
            self.tile_height,
            10)
        nc1 = self.coords.shape[0]
        self.logger.info(
                "\n  smoothing point density reduced points from %d to %d" %
                (nc0, nc1))
        if self.coords.shape[0] == 0:
            raise MeshLensCorrectionException(
                    "no point matches left after smoothing density, \
                     probably some sparse areas of matching")

        # create PSLG
        self.bbox = create_PSLG(
                self.tile_width,
                self.tile_height,
                maskUrl)

        # find delaunay with max vertices
        self.mesh, self.area_triangle_par = \
            find_delaunay_with_max_vertices(
                self.bbox,
                self.args['nvertex'])

        # and enforce neighboring matches to vertices
        self.mesh, self.area_triangle_par = \
            force_vertices_with_npoints(
                self.area_triangle_par,
                self.bbox,
                self.coords,
                3)

        nend = self.mesh.points.shape[0]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
                "\n  aimed for %d mesh points, got %d" %
                (self.args['nvertex'], nend))

        if self.mesh.points.shape[0] < 0.5*self.args['nvertex']:
            raise MeshLensCorrectionException(
                    "mesh coarser than intended")

        # prepare the linear algebra and solve
        self.A, self.weights, self.b, self.lens_dof_start = \
            create_A(
                self.matches,
                self.tilespecs,
                self.mesh)
        self.x0 = create_x0(
                self.A.shape[1],
                self.tilespecs)

        self.reg = create_regularization(
                self.A.shape[1],
                len(self.tilespecs),
                self.args['regularization']['default_lambda'],
                self.args['regularization']['translation_factor'],
                self.args['regularization']['lens_lambda'])
        self.solution, self.errx, self.erry = solve(
                self.A,
                self.weights,
                self.reg,
                self.x0,
                self.b)

        self.common_transform, self.transforms = create_transforms(
                len(self.tilespecs), self.solution)

        tf_trans, jresult, self.solve_message = report_solution(
                self.errx,
                self.erry,
                self.transforms,
                self.args['good_solve'])

        self.logger.info(self.solve_message)

        # check quality of solution
        if not all([
                    self.errx.mean() <
                    self.args['good_solve']['error_mean'],

                    self.erry.mean() <
                    self.args['good_solve']['error_mean'],

                    self.errx.std() <
                    self.args['good_solve']['error_std'],

                    self.erry.std() <
                    self.args['good_solve']['error_std']]):

            raise MeshLensCorrectionException(
                    "Solve not good: %s" % self.solve_message)

        self.logger.debug(self.solve_message)

        self.new_ref_transform = create_thinplatespline_tf(
                self.args,
                self.mesh,
                self.solution,
                self.lens_dof_start,
                self.common_transform,
                self.logger)

        bbox = self.tilespecs[0].bbox_transformed(tf_limit=0)
        tbbox = self.new_ref_transform.tform(bbox)
        bstr = 'new transform corners:\n'
        for i in range(bbox.shape[0]-1):
            bstr += "  (%0.1f, %0.1f) -> (%0.1f, %0.1f)\n" % (
                    bbox[i, 0], bbox[i, 1],
                    tbbox[i, 0], tbbox[i, 1])
        self.logger.info(bstr)

        self.tfpath = os.path.join(
                self.args['output_dir'], self.args['outfile'])
        jsongz.dump(
                self.new_ref_transform.to_dict(), self.tfpath, compress=False)

        new_tilespecs = new_specs_with_tf(
            self.new_ref_transform,
            self.tilespecs,
            self.transforms)

        stage_affine = estimate_stage_affine(self.tilespecs, new_tilespecs)
        sastr = "affine estimate of tile translations:\n"
        sastr += "  scale: {}\n".format(stage_affine.scale)
        sastr += "  translation: {}\n".format(stage_affine.translation)
        sastr += "  shear: {}\n".format(stage_affine.shear)
        sastr += "  rotation: {}\n".format(np.degrees(stage_affine.rotation))
        self.logger.info(sastr)

        new_path = os.path.join(
                self.args['output_dir'],
                'resolvedtiles.json')

        resolved = renderapi.resolvedtiles.ResolvedTiles(
                tilespecs=new_tilespecs,
                transformList=[self.new_ref_transform])

        new_path = jsongz.dump(
                resolved.to_dict(),
                new_path,
                compress=self.args['compress_output'])

        self.args['output_json'] = os.path.join(
                self.args['output_dir'],
                'output.json')

        jresult['resolved_tiles'] = os.path.abspath(new_path)

        self.output(jresult, indent=2)

        self.logger.info(" wrote solved tilespecs:\n  %s" %
                         self.args['output_json'])

        return
