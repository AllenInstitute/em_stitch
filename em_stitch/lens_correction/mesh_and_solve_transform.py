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
from bigfeta import jsongz
import logging

try:
    # pandas unique is faster than numpy, use where appropriate
    import pandas
    _uniq = pandas.unique
except ImportError:
    import numpy
    _uniq = numpy.unique

# this is a modification of https://github.com/
# AllenInstitute/render-modules/blob/master/
# rendermodules/mesh_lens_correction/MeshAndSolveTransform.py
# that does not depend on render-modules nor
# on a running render server

logger = logging.getLogger()
default_logger = logger


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


def smooth_density_legacy(coords, tile_width, tile_height, n):
    """legacy function to homogenize distribution of points within a
        rectangular area by reducing the number of points within
        n**2 equally-sized bounding boxes to
        the minimum number of points in one of those boxes.


    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 numpy array of coordinates to consider
    tile_width : int
        width of rectangular area containing coords
    tile_height : int
        height of rectangular area containing coords
    n : int
        number of subdivisions into which tile_width and tile_height
        should be divided

    Returns
    -------
    smoothed_coords : numpy.ndarray
        Nx2 numpy array of smoothed subset of input coords
    """
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


def get_bboxes(tile_width, tile_height, n):
    """get list of bounds for n**2 equally-sized bounding boxes within a
        rectangular bounding box


    Parameters
    ----------
    tile_width : int
        width of rectangular area to divide
    tile_height : int
        height of rectangular area to divide
    n : int
        number of subdivisions into which tile_width and tile_height
        should be divided

    Returns
    -------
    vtxs : list of tuple of numpy.ndarray
        list of min/max tuples of vertices representing bounding boxes
    """
    numX = n
    numY = n
    diffX = (tile_width-1) / numX
    diffY = (tile_height-1) / numY

    squaremesh = np.mgrid[
        0:tile_width-1:numX*1j, 0:tile_height-1:numY*1j].reshape(2, -1).T
    maxpt = squaremesh.max(axis=0)

    vtxs = []
    for pt in squaremesh:
        if np.any(pt == maxpt):
            continue
        vtxs.append((pt, pt + np.array([diffX, diffY])))
    return vtxs


def smooth_density_bbox(coords, tile_width, tile_height, n):
    """homogenize distribution of points within a rectangular area by reducing
        the number of points within n**2 equally-sized bounding boxes to
        the minimum number of points in one of those boxes.


    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 numpy array of coordinates to consider
    tile_width : int
        width of rectangular area containing coords
    tile_height : int
        height of rectangular area containing coords
    n : int
        number of subdivisions into which tile_width and tile_height
        should be divided

    Returns
    -------
    smoothed_coords : numpy.ndarray
        Nx2 numpy array of smoothed subset of input coords
    """
    vtxs = get_bboxes(tile_width, tile_height, n)

    index_arr = np.zeros(coords.shape[0], dtype="int64")
    for ei, (ll, ur) in enumerate(vtxs):
        vi = ei + 1  # 0 is prohibited
        index_arr[np.all((ll <= coords) & (ur >= coords), axis=1)] = vi

    bc = np.bincount(index_arr)
    mincount = bc[1:].min()  # 0 is prohibited

    idxs = _uniq(index_arr)
    new_coords = np.concatenate([
        coords[np.random.choice(np.argwhere(index_arr == idx).flatten(),
                                mincount)] for idx in idxs])

    return new_coords


def smooth_density(coords, tile_width, tile_height, n,
                   legacy_smooth_density=False, **kwargs):
    """homogenize distribution of points within a rectangular area by reducing
        the number of points within n**2 equally-sized bounding boxes to
        the minimum number of points in one of those boxes.


    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 numpy array of coordinates to consider
    tile_width : int
        width of rectangular area containing coords
    tile_height : int
        height of rectangular area containing coords
    n : int
        number of subdivisions into which tile_width and tile_height
        should be divided
    legacy_smooth_density : boolean
        whether to use (slower) legacy code.  Not recommended.

    Returns
    -------
    smoothed_coords : numpy.ndarray
        Nx2 numpy array of smoothed subset of input coords
    """
    if legacy_smooth_density:
        return smooth_density_legacy(coords, tile_width, tile_height, n)
    else:
        return smooth_density_bbox(coords, tile_width, tile_height, n)


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
        _, contours, _ = cv2.findContours(
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


def force_vertices_with_npoints(area_par, bbox, coords, npts, **kwargs):
    """create a triangular mesh which iteratively attempts to conform to a
        minimum number of points per vertex by adjusting the maximum
        triangle area

    Parameters
    ----------
    area_par : float
        initial maximum triangle area constraint for triangle.triangulate
    bbox : dict
        PSLG bounding box dictionary from :func:create_PSLG
    coords : numpy.ndarray
        Nx2 points
    npts : int
        minimum number of points near each vertex

    Returns
    -------
    t : scipy.spatial.qhull.Delaunay
        triangle mesh with minimum point count near vertices
    area_par : float
        area parameter used to calculate result t
    """
    fac = 1.02
    count = 0
    max_iter = 20
    while True:
        t = calculate_mesh(
                area_par,
                bbox,
                None,
                get_t=True)
        pt_count = count_points_near_vertices(t, coords, **kwargs)
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


def compute_barycentrics_legacy(coords, mesh):
    """legacy function to compute barycentric coordinates on mesh

    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 array of points
    mesh : scipy.spatial.qhull.Delaunay
        triangular mesh

    Returns
    -------
    bcoords : numpy.ndarray
        Nx2 array of barycentric coordinates
    triangle_indices : numpy.ndarray
        simplex indices of barycentric coordinates
    """
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


def compute_barycentrics_native(coords, mesh):
    """convert coordinates to barycentric coordinates on mesh

    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 array of points
    mesh : scipy.spatial.qhull.Delaunay
        triangular mesh

    Returns
    -------
    bcoords : numpy.ndarray
        Nx2 array of barycentric coordinates
    triangle_indices : numpy.ndarray
        simplex indices of barycentric coordinates
    """
    triangle_indices = mesh.find_simplex(coords)
    X = mesh.transform[triangle_indices, :2]
    Y = coords - mesh.transform[triangle_indices, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    bcoords = np.c_[b, 1 - b.sum(axis=1)]
    return bcoords, triangle_indices


def compute_barycentrics(coords, mesh, legacy_barycentrics=False, **kwargs):
    """convert coordinates to barycentric coordinates on mesh

    Parameters
    ----------
    coords : numpy.ndarray
        Nx2 array of points
    mesh : scipy.spatial.qhull.Delaunay
        triangular mesh
    legacy_barycentrics : boolean
        whether to use (slower) legacy method to find barycentrics.

    Returns
    -------
    bcoords : numpy.ndarray
        Nx2 array of barycentric coordinates
    triangle_indices : numpy.ndarray
        simplex indices of barycentric coordinates
    """
    if legacy_barycentrics:
        return compute_barycentrics_legacy(coords, mesh)
    else:
        return compute_barycentrics_native(coords, mesh)


def count_points_near_vertices(
        t, coords, bruteforce_simplex_counts=False,
        count_bincount=True, **kwargs):
    """enumerate coordinates closest to the vertices in a mesh

    Parameters
    ----------
    t : scipy.spatial.qhull.Delaunay
        triangular mesh
    coords : numpy.ndarray
        Nx2 array of points to assign to vertices on t
    bruteforce_simplex_counts : boolean
        whether to do a bruteforce simplex finding
    count_bincount : boolean
       use numpy.bincount based counting rather than legacy counting

    Returns
    -------
    pt_count : numpy.ndarray
        array with counts of points corresponding to indices of vertices in t
    """
    flat_tri = t.simplices.flatten()
    flat_ind = np.repeat(np.arange(t.nsimplex), 3)
    v_touches = []
    for i in range(t.npoints):
        v_touches.append(flat_ind[np.argwhere(flat_tri == i)])
    found = t.find_simplex(coords, bruteforce=bruteforce_simplex_counts)
    if count_bincount:
        bc = np.bincount(found)
        pt_count = np.array([
            bc[v_touches[i]].sum() for i in range(t.npoints)
        ])
    else:
        pt_count = np.zeros(t.npoints)
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
        mesh, solution,
        lens_dof_start,
        logger=default_logger,
        compute_affine=False):

    dst = np.zeros_like(mesh.points)
    dst[:, 0] = mesh.points[:, 0] + solution[0][lens_dof_start:]
    dst[:, 1] = mesh.points[:, 1] + solution[1][lens_dof_start:]

    transform = renderapi.transform.ThinPlateSplineTransform()
    transform.estimate(mesh.points, dst, computeAffine=compute_affine)
    npts0 = transform.srcPts.shape[1]
    transform = transform.adaptive_mesh_estimate(max_iter=1000)
    npts1 = transform.srcPts.shape[1]

    logger.info(
            "adaptive_mesh_estimate reduced control points from %d to %d" %
            (npts0, npts1))

    transform.transformId = (
            datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3])
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


def solve(A, weights, reg, x0, b, precomputed_ATW=None, precomputed_ATWA=None,
          precomputed_K_factorized=None):
    """regularized weighted solve

    Parameters
    ----------
    A : :class:`scipy.sparse.csr`
        the matrix, N (equations) x M (degrees of freedom)
    weights : :class:`scipy.sparse.csr_matrix`
        N x N diagonal matrix containing weights
    reg : :class:`scipy.sparse.csr_matrix`
        M x M diagonal matrix containing regularizations
    x0 : :class:`numpy.ndarray`
        M x nsolve float constraint values for the DOFs
    b : :class:`numpy.ndarray`:
        N x nsolve float right-hand-side(s)
    precomputed_ATW : :class:`scipy.sparse.csc_matrix`
        value to use rather than computing A.T.dot(weights)
    precomputed_ATWA : :class:`scipy.sparse.csc_matrix`
        value to use rather than computing A.T.dot(weights).dot(A)
    precomputed_K_factorized : func
        factorized solve function to use rather than computing
        scipy.sparse.linalg.factorized(A.T.dot(weights).dot(A) + reg)

    Returns
    -------
    solution : list of numpy.ndarray
        list of numpy arrays of x and y vertex positions of solution
    errx : numpy.ndarray
        numpy array of x residuals
    erry : numpy.ndarray
        numpy array of y residuals

    """
    ATW = (A.transpose().dot(weights)
           if precomputed_ATW is None else precomputed_ATW)
    if precomputed_K_factorized is None:
        K = (ATW.dot(A) if precomputed_ATWA is None
             else precomputed_ATWA) + reg
        K_factorized = factorized(K)
    else:
        K_factorized = precomputed_K_factorized
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


def create_A(matches, tilespecs, mesh, **kwargs):
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
        pbary = compute_barycentrics(pcoords, mesh, **kwargs)
        qbary = compute_barycentrics(qcoords, mesh, **kwargs)

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


def create_transforms(ntiles, solution):
    rtransforms = []
    for i in range(ntiles):
        rtransforms.append(renderapi.transform.AffineModel(
                           B0=solution[0][i],
                           B1=solution[1][i]))
    return rtransforms


def estimate_stage_affine(t0, t1):
    src = np.array([t.tforms[0].translation for t in t0])
    dst = np.array([t.tforms[1].translation for t in t1])
    aff = renderapi.transform.AffineModel()
    aff.estimate(src, dst)
    return aff


def _create_mesh(resolvedtiles, matches, nvertex,
                 return_area_triangle_par=False, **kwargs):
    """create mesh with a given number of vertices based on example tiles
        and pointmatches

    Parameters
    ----------
    resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles containing a tilespec with mask, width, and height
        properties to use as a template for the mesh
    matches : list of dict
        list of point correspondences in render pointmatch format
    nvertex : int
        number of vertices for mesh
    return_area_triangle_par : boolean
        whether to return the area parameter used to generate the
        triangular mesh

    Returns
    -------
    mesh : scipy.spatial.qhull.Delaunay
        triangular mesh
    area_triangle_par : float
        max area constraint used in generating mesh
    """

    remove_weighted_matches(matches, weight=0.0)

    tilespecs = resolvedtiles.tilespecs
    example_tspec = tilespecs[0]

    tile_width = example_tspec.width
    tile_height = example_tspec.height
    maskUrl = example_tspec.ip[0].maskUrl

    coords = condense_coords(matches)
    nc0 = coords.shape[0]
    coords = smooth_density(
        coords,
        tile_width,
        tile_height,
        10, **kwargs)

    nc1 = coords.shape[0]
    logger.info(
        "\n  smoothing point density reduced points from %d to %d" %
        (nc0, nc1))
    if coords.shape[0] == 0:
        raise MeshLensCorrectionException(
            "no point matches left after smoothing density, \
            probably some sparse areas of matching")

    # create PSLG
    bbox = create_PSLG(
            tile_width,
            tile_height,
            maskUrl)

    # find delaunay with max vertices
    mesh, area_triangle_par = find_delaunay_with_max_vertices(
        bbox, nvertex)

    # and enforce neighboring matches to vertices
    mesh, area_triangle_par = force_vertices_with_npoints(
        area_triangle_par, bbox, coords, 3, **kwargs)

    return ((mesh, area_triangle_par) if return_area_triangle_par else mesh)


def _solve_resolvedtiles(
        resolvedtiles, matches, nvertex, regularization_lambda,
        regularization_translation_factor, regularization_lens_lambda,
        good_solve_dict,
        logger=default_logger, **kwargs):
    """generate lens correction from resolvedtiles and pointmatches

    Parameters
    ----------
    resolvedtiles : renderapi.resolvedtiles.ResolvedTiles
        resolvedtiles object on which transformation will be computed
    matches : list of dict
         point correspondences to consider in render pointmatch format
    nvertex :
        number of vertices in mesh
    regularization_lambda :  float
        lambda value for affine regularization
    regularization_translation_factor :  float
        translation factor of regularization
    regularization_lens_lambda :  float
        lambda value for lens regularization
    good_solve_dict :
        dictionary to define when a solve fails
    logger : logging.Logger
        logger to use in reporting
    Returns
    -------
    resolved : renderapi.resolvedtiles.ResolvedTiles
        new resolvedtiles object with derived lens correction applied
    new_ref_transform : renderapi.transform.leaf.ThinPlateSplineTransform
        derived lens correction transform
    jresult : dict
        dictionary of solve information
    """

    # FIXME this is done twice -- think through
    tilespecs = resolvedtiles.tilespecs
    example_tspec = tilespecs[0]

    mesh = _create_mesh(resolvedtiles, matches, nvertex, **kwargs)

    nend = mesh.points.shape[0]

    # logger = logging.getLogger(self.__class__.__name__)
    logger.info(
        "\n  aimed for %d mesh points, got %d" %
        (nvertex, nend))

    if mesh.points.shape[0] < 0.5*nvertex:
        raise MeshLensCorrectionException(
                "mesh coarser than intended")

    # prepare the linear algebra and solve
    A, weights, b, lens_dof_start = create_A(
        matches, tilespecs, mesh)

    x0 = create_x0(
        A.shape[1], tilespecs)

    reg = create_regularization(
        A.shape[1],
        len(tilespecs),
        regularization_lambda,
        regularization_translation_factor,
        regularization_lens_lambda)

    solution, errx, erry = solve(
        A, weights, reg, x0, b)

    transforms = create_transforms(
        len(tilespecs), solution)

    tf_trans, jresult, solve_message = report_solution(
            errx, erry, transforms, good_solve_dict)

    logger.info(solve_message)

    # check quality of solution
    if not all([
            errx.mean() < good_solve_dict['error_mean'],
            erry.mean() < good_solve_dict['error_mean'],
            errx.std() < good_solve_dict['error_std'],
            erry.std() < good_solve_dict['error_std']]):
        raise MeshLensCorrectionException(
                "Solve not good: %s" % solve_message)

    logger.debug(solve_message)

    new_ref_transform = create_thinplatespline_tf(
        mesh, solution, lens_dof_start, logger)

    bbox = example_tspec.bbox_transformed(tf_limit=0)
    tbbox = new_ref_transform.tform(bbox)
    bstr = 'new transform corners:\n'
    for i in range(bbox.shape[0]-1):
        bstr += "  (%0.1f, %0.1f) -> (%0.1f, %0.1f)\n" % (
                bbox[i, 0], bbox[i, 1],
                tbbox[i, 0], tbbox[i, 1])
        logger.info(bstr)

    new_tilespecs = new_specs_with_tf(
        new_ref_transform, tilespecs, transforms)

    stage_affine = estimate_stage_affine(tilespecs, new_tilespecs)
    sastr = (
        "affine estimate of tile translations:\n" +
        "  scale: {}\n".format(stage_affine.scale) +
        "  translation: {}\n".format(stage_affine.translation) +
        "  shear: {}\n".format(stage_affine.shear) +
        "  rotation: {}\n".format(np.degrees(stage_affine.rotation)))
    logger.info(sastr)

    resolved = renderapi.resolvedtiles.ResolvedTiles(
            tilespecs=new_tilespecs,
            transformList=[new_ref_transform])
    return resolved, new_ref_transform, jresult


class MeshAndSolveTransform(ArgSchemaParser):
    default_schema = MeshLensCorrectionSchema

    def solve_resolvedtiles_from_args(self):
        """use arguments to run lens correction

        Returns
        -------
        resolved : renderapi.resolvedtiles.ResolvedTiles
            new resolvedtiles object with derived lens correction applied
        new_ref_transform : renderapi.transform.leaf.ThinPlateSplineTransform
            derived lens correction transform
        jresult : dict
            dictionary of solve information
        """
        if 'tilespecs' in self.args:
            jspecs = self.args['tilespecs']
        else:
            jspecs = jsongz.load(self.args['tilespec_file'])

        self.tilespecs = np.array([
                renderapi.tilespec.TileSpec(json=j) for j in jspecs])

        if 'matches' in self.args:
            self.matches = self.args['matches']
        else:
            self.matches = jsongz.load(self.args['match_file'])

        return _solve_resolvedtiles(
            renderapi.resolvedtiles.ResolvedTiles(
                tilespecs=self.tilespecs, transformList=[]),
            self.matches, self.args["nvertex"],
            self.args["regularization"]["default_lambda"],
            self.args["regularization"]["translation_factor"],
            self.args["regularization"]["lens_lambda"],
            self.args["good_solve"],
            logger=self.logger
            )

    def run(self):
        self.resolved, self.new_ref_transform, jresult = (
            self.solve_resolvedtiles_from_args())

        new_path = None
        if 'outfile' in self.args:
            fname = self.args['outfile']
            if self.args['timestamp']:
                spf = fname.split(os.extsep, 1)
                spf[0] += '_%s' % self.new_ref_transform.transformId
                fname = os.extsep.join(spf)
            new_path = jsongz.dump(
                    self.resolved.to_dict(),
                    os.path.join(
                        self.args['output_dir'],
                        fname),
                    compress=self.args['compress_output'])
            new_path = os.path.abspath(new_path)

        fname = 'output.json'
        if self.args['timestamp']:
            fname = 'output_%s.json' % self.new_ref_transform.transformId

        self.args['output_json'] = os.path.join(
                self.args['output_dir'],
                fname)

        jresult['resolved_tiles'] = new_path

        self.output(jresult, indent=2)

        self.logger.info(" wrote solved tilespecs:\n  %s" % new_path)

        return
