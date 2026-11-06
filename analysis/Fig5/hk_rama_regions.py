"""Ramachandran regions of Hollingsworth & Karplus 2010 (BioMol Concepts 1:271-283), digitized from their Fig. 6A.

Fig. 6A is the "wrapped" plot (phi in [0, 360), psi in [-120, 240)), where every region is one intact ellipse.
The boundary of each region (beta, PII, zeta, gamma', delta, alpha, delta', PII') was clicked in WebPlotDigitizer
(hk2010_fig6a_wpd_datasets.csv) and an ellipse is
least-squares fitted to those points. All geometry (fitting, inside test, distance) is done in wrapped coordinates.

Assignment of one (phi, psi) point:
  1. inside = normalized radius q <= 1 for that ellipse;
  2. alpha is nested inside delta -> regions in PRIORITY win; other overlaps go to the smallest q;
  3. inside no ellipse -> "other", or (outside="nearest") the region whose ellipse boundary is closest
     (Euclidean distance in degrees, wrapped coordinates), optionally only within max_dist degrees.
"""
import numpy as np
import pandas as pd
from skimage.measure import EllipseModel

PRIORITY = ["alpha"]                 # nested regions that take precedence over the region around them
N_BOUNDARY = 180                     # boundary samples per ellipse for the point-to-boundary distance


def wrap(phi, psi):
    """Standard (-180, 180] angles -> H&K wrapped axes: phi in [0, 360), psi in [-120, 240)."""
    phi, psi = np.asarray(phi, float), np.asarray(psi, float)
    return np.where(phi < 0, phi + 360, phi), np.where(psi < -120, psi + 360, psi)


def unwrap(phi_w, psi_w):
    """Wrapped axes -> standard (-180, 180] angles."""
    phi_w, psi_w = np.asarray(phi_w, float), np.asarray(psi_w, float)
    return np.where(phi_w > 180, phi_w - 360, phi_w), np.where(psi_w > 180, psi_w - 360, psi_w)


def load_wpd(path):
    """WebPlotDigitizer CSV (row 1 = dataset names, row 2 = X/Y) -> {name: (n, 2) array of wrapped (phi, psi)}."""
    raw = pd.read_csv(path, header=None)
    names = raw.iloc[0].tolist()
    out = {}
    for j in range(0, raw.shape[1], 2):
        xy = raw.iloc[2:, j:j + 2].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
        out[str(names[j]).strip().replace("gama", "gamma")] = xy
    return out


def fit_ellipses(points):
    """{name: points} -> DataFrame with centre (wrapped and standard), semi-axes a >= b, angle and fit residual."""
    rows = []
    for name, xy in points.items():
        m = EllipseModel()
        if not m.estimate(xy):
            raise RuntimeError(f"ellipse fit failed for {name}")
        xc, yc, a, b, th = m.params
        if b > a:                                        # keep a as the long semi-axis
            a, b, th = b, a, th + np.pi / 2
        th = (th + np.pi / 2) % np.pi - np.pi / 2        # angle of the long axis, in (-90, 90] deg
        phi_c, psi_c = unwrap(xc, yc)
        rows.append(dict(region=name, phi_w=xc, psi_w=yc, phi=float(phi_c), psi=float(psi_c), a=a, b=b,
                         angle_deg=np.degrees(th), n_points=len(xy),
                         rms_resid_deg=float(np.sqrt(np.mean(m.residuals(xy) ** 2)))))
    return pd.DataFrame(rows).set_index("region")


def boundary(e, n=N_BOUNDARY):
    """Points on the ellipse boundary (wrapped coordinates), shape (n, 2)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(np.radians(e.angle_deg)), np.sin(np.radians(e.angle_deg))
    x = e.phi_w + e.a * np.cos(t) * c - e.b * np.sin(t) * s
    y = e.psi_w + e.a * np.cos(t) * s + e.b * np.sin(t) * c
    return np.column_stack([x, y])


def norm_radius2(phi_w, psi_w, e):
    """q = ((u / a)^2 + (v / b)^2) in the ellipse frame; q <= 1 means inside."""
    c, s = np.cos(np.radians(e.angle_deg)), np.sin(np.radians(e.angle_deg))
    dx, dy = phi_w - e.phi_w, psi_w - e.psi_w
    return ((dx * c + dy * s) / e.a) ** 2 + ((-dx * s + dy * c) / e.b) ** 2


def boundary_distance(phi_w, psi_w, e, chunk=50_000):
    """Distance (deg) from each point to the ellipse boundary on the torus (0 if inside).

    Euclidean in wrapped coordinates, taking the nearest periodic image of the boundary (shifts of 0 / +-360 deg
    in phi and psi), so points just across a wrapped-axis edge (e.g. psi = -120 vs P_II' near psi_w = 240) are
    measured correctly.
    """
    bd = boundary(e)
    shifts = [(dx, dy) for dx in (-360, 0, 360) for dy in (-360, 0, 360)]
    out = np.full(len(phi_w), np.inf)
    for i in range(0, len(phi_w), chunk):
        p = np.column_stack([phi_w[i:i + chunk], psi_w[i:i + chunk]])
        for dx, dy in shifts:
            d = np.sqrt(((p[:, None, :] - (bd + [dx, dy])[None]) ** 2).sum(-1)).min(1)
            out[i:i + chunk] = np.minimum(out[i:i + chunk], d)
    out[norm_radius2(phi_w, psi_w, e) <= 1] = 0.0
    return out


def assign(phi, psi, ell, outside="other", max_dist=None, priority=PRIORITY):
    """Standard-angle phi/psi arrays -> (labels, q_min, dist_to_nearest_boundary).

    outside = "other": points in no ellipse get "other".
    outside = "nearest": they get the region with the closest boundary, if that distance <= max_dist
    (max_dist=None: always); farther points stay "other".
    """
    pw, sw = wrap(np.ravel(phi), np.ravel(psi))
    names = list(ell.index)
    q = np.column_stack([norm_radius2(pw, sw, ell.loc[n]) for n in names])
    inside = q <= 1
    lab = np.full(len(pw), "other", dtype=object)
    q_ord = np.where(inside, q, np.inf)
    has = inside.any(1)
    lab[has] = np.array(names)[q_ord[has].argmin(1)]
    for n in priority[::-1]:                              # nested regions win over the region around them
        if n in names:
            lab[inside[:, names.index(n)]] = n
    out_idx = np.where(~has)[0]
    dist = np.zeros(len(pw))
    if len(out_idx):
        d = np.column_stack([boundary_distance(pw[out_idx], sw[out_idx], ell.loc[n]) for n in names])
        dist[out_idx] = d.min(1)
        if outside == "nearest":
            near = np.array(names)[d.argmin(1)]
            ok = np.ones(len(out_idx), bool) if max_dist is None else d.min(1) <= max_dist
            lab[out_idx[ok]] = near[ok]
    shape = np.shape(phi)
    return lab.reshape(shape), q.min(1).reshape(shape), dist.reshape(shape)
