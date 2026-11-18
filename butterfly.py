#!/usr/bin/env python3
"""4 Ways to Draw a Butterfly - one monarch, same pose, drawn four times with rising skill.

Static overhead camera like the reference reel; a real hand cut-out (horizontal pencil from the
right edge) does all the drawing; strokes are generated to move like a person's:
  * contours are pulled in short overlapping strokes, never one perfect curve
  * a light sketch first, then firmer lines that do not sit exactly on the sketch
  * speed follows the 2/3 power law (slower on bends) inside a minimum-jerk envelope
  * colour goes on in small back-and-forth patches whose direction follows the wing
  * the fingers move the pencil for small strokes while the hand itself lags behind

    python butterfly.py --plan
    python butterfly.py --still 2,10,30
    python butterfly.py --render out.mp4 [--chunk i/n]
    python butterfly.py --audio pencil.wav
"""
import argparse, json, math, os, subprocess, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET = lambda *p: os.path.join(HERE, 'assets', *p)

OW, OH, FPS = 1080, 1920, 30
WS = 1.5                                         # world px per unit (unit = output px)
UX0, UY0, UX1, UY1 = -30.0, -30.0, 1110.0, 1950.0
WW, WH = int((UX1 - UX0) * WS), int((UY1 - UY0) * WS)
ORG = np.array([UX0, UY0])
SPEED = 1.9                                      # drawing pace vs real time

CELLS = dict(simple=(276, 625), normal=(752, 625), better=(276, 1290), real=(752, 1290))
LABELS = dict(simple=(276, 380), normal=(752, 380), better=(276, 1050), real=(752, 1050))
LABEL_TEXT = dict(simple='Simple', normal='Normal', better='Better', real='Realistic')


def to_w(p):
    return (np.asarray(p, float) - ORG) * WS


# ============================================================================ geometry helpers
def crom(pts, closed=False, step=1.2):
    P = np.asarray(pts, float)
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)


def ell(c, rx, ry, a0=-90.0, sweep=-360.0, rot=0.0):
    L = math.pi * (rx + ry) * abs(sweep) / 360
    t = np.radians(np.linspace(a0, a0 + sweep, max(16, int(L / 1.2))))
    x, y = rx * np.cos(t), ry * np.sin(t)
    r = math.radians(rot)
    return np.stack([c[0] + x * math.cos(r) - y * math.sin(r), c[1] + x * math.sin(r) + y * math.cos(r)], -1)


def arclen(p):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def resample(p, step=1.2):
    s = arclen(p)
    if s[-1] < step:
        return p
    n = max(2, int(s[-1] / step) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], -1)


def normals(p):
    d = np.gradient(p, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], -1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


def inset(poly, d):
    """offset a closed path by d toward its inside"""
    P = resample(np.asarray(poly, float), 1.2)
    n = normals(P)
    cnt = P.astype(np.float32).reshape(-1, 1, 2)
    test = P[::max(1, len(P) // 12)] + n[::max(1, len(P) // 12)] * 2.0
    ins = sum(cv2.pointPolygonTest(cnt, (float(q[0]), float(q[1])), False) > 0 for q in test)
    sgn = 1 if ins > len(test) / 2 else -1
    return P + n * d * sgn


def mirror(p):
    q = np.asarray(p, float).copy()
    q[..., 0] *= -1
    return q[::-1]


class Xf:
    def __init__(self, c, s=1.0):
        self.c, self.s = np.array(c, float), s

    def __call__(self, p):
        return np.asarray(p, float) * self.s + self.c


class Region:
    """raster mask in units (R px per unit) for colouring regions"""
    R = 2.0

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.m = np.zeros((int((y1 - y0) * self.R) + 2, int((x1 - x0) * self.R) + 2), np.uint8)

    def _pts(self, p):
        return np.round((np.asarray(p) - [self.x0, self.y0]) * self.R).astype(np.int32)

    def add(self, poly, v=1):
        cv2.fillPoly(self.m, [self._pts(poly)], v)
        return self

    def sub(self, poly):
        return self.add(poly, 0)

    def copy(self):
        r = Region.__new__(Region)
        r.x0, r.y0, r.m = self.x0, self.y0, self.m.copy()
        return r

    def erode(self, units):
        k = int(2 * units * self.R) | 1
        r = self.copy()
        r.m = cv2.erode(self.m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return r

    def minus(self, other):
        r = self.copy(); r.m = self.m & (1 - other.m); return r

    def inter(self, other):
        r = self.copy(); r.m = self.m & other.m; return r

    def near(self, c, rad):
        """keep only pixels within rad of point c"""
        r = self.copy()
        yy, xx = np.mgrid[0:self.m.shape[0], 0:self.m.shape[1]]
        d = np.hypot(xx / self.R + self.x0 - c[0], yy / self.R + self.y0 - c[1])
        r.m = self.m & (d < rad).astype(np.uint8)
        return r

    def inside(self, p):
        q = self._pts(p)
        ok = (q[:, 0] >= 0) & (q[:, 0] < self.m.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < self.m.shape[0])
        out = np.zeros(len(q), bool)
        out[ok] = self.m[q[ok, 1], q[ok, 0]] > 0
        return out

    def area(self):
        return self.m.sum() / self.R ** 2


# ============================================================================ human strokes
def stroke(p, pig='g', w=2.4, k=0.6, v=400.0, kind='line'):
    return dict(p=np.asarray(p, float), pig=pig, w=w, k=k, v=v, kind=kind)


def lowfreq(n, s, amp, r):
    f = np.zeros(n)
    for _ in range(3):
        f += r.normal() * np.sin(s / r.uniform(14, 45) + r.uniform(0, 6.3))
    return f / 3 * amp


def human_contour(path, r, mode='line', closed=False, pig='g', w=2.2, k=0.6, v=None, pull=True):
    """split a path into short, overlapping, slightly-off strokes like a person draws it"""
    P = resample(np.asarray(path, float), 1.2)
    if closed:
        i0 = int(np.argmin(P[:, 1] + r.uniform(-20, 20, len(P)) * 0))     # start near the top
        i0 = (i0 + r.integers(-len(P) // 8, len(P) // 8 + 1)) % len(P)
        P = np.vstack([P[i0:], P[:i0 + 1]])
    s = arclen(P)
    L = s[-1]
    cfg = dict(sketch=(30, 75, 1.8, 0.28), line=(60, 150, 0.7, 0.0), detail=(40, 110, 0.5, 0.0))[mode]
    lo, hi, amp, dbl = cfg
    if v is None:
        v = dict(sketch=520, line=360, detail=300)[mode]
    out = []
    a = 0.0
    while a < L - 2:
        b = min(L, a + r.uniform(lo, hi))
        if L - b < lo * 0.4:
            b = L
        a0 = max(0.0, a - r.uniform(1.5, 5.0)) if a > 0 else 0.0
        idx = (s >= a0) & (s <= b)
        seg = P[idx]
        if len(seg) >= 3:
            ss = arclen(seg)
            nrm = normals(seg)
            off = lowfreq(len(seg), ss, amp, r) + r.normal() * amp * 0.35
            seg = seg + nrm * off[:, None]
            d = seg[-1] - seg[0]
            if pull and ((d[1] < -abs(d[0]) * 0.6) or (d[0] < -abs(d[1]) * 2.2)) and r.random() < 0.75:
                seg = seg[::-1]                                  # pull downward / rightward
            out.append(stroke(seg, pig, w * r.uniform(0.9, 1.1), k * r.uniform(0.85, 1.1), v, 'line'))
            if mode == 'sketch' and r.random() < dbl:            # searching double line
                seg2 = seg + nrm * (r.uniform(0.8, 1.8) * r.choice([-1, 1]))
                out.append(stroke(seg2[::-1] if r.random() < 0.5 else seg2, pig, w * 0.9, k * 0.7, v * 1.1, 'line'))
        a = b
    return out


def dot(c, rad, r, pig='k', w=3.0, k=1.0):
    turns = r.uniform(1.4, 2.0)
    t = np.linspace(0, turns * 2 * math.pi, max(12, int(turns * rad * 6)))
    rr = rad * (1 - 0.55 * t / t[-1])
    a0 = r.uniform(0, 6.3)
    p = np.stack([c[0] + rr * np.cos(t + a0), c[1] + rr * np.sin(t + a0)], -1)
    return stroke(p, pig, w, k, 140 + rad * 12, 'dot')


def patches(region, ang_fn, r, pig='o', patch=30, spacing=3.6, length=(16, 32), w=6.0, k=0.8, v=430,
            order=(0.3, 1.0), shrink=1.0):
    """colour a region in small back-and-forth patches (how people actually colour)"""
    ys, xs = np.nonzero(region.m)
    if len(xs) == 0:
        return []
    x0, y0 = xs.min() / region.R + region.x0, ys.min() / region.R + region.y0
    x1, y1 = xs.max() / region.R + region.x0, ys.max() / region.R + region.y0
    step = patch * 0.6
    cs = []
    for gy in np.arange(y0, y1 + step, step):
        for gx in np.arange(x0, x1 + step, step):
            c = np.array([gx + r.uniform(-0.35, 0.35) * step, gy + r.uniform(-0.35, 0.35) * step])
            cs.append(c)
    cs = np.array(cs)
    cs = cs[region.inside(cs)]
    if len(cs) == 0:
        cs = np.array([[xs.mean() / region.R + region.x0, ys.mean() / region.R + region.y0]])
    od = np.array(order) / np.linalg.norm(order)
    key = cs @ od
    band = np.round((cs @ np.array([-od[1], od[0]])) / (step * 1.4))
    srt = np.lexsort((key * np.where(band % 2 == 0, 1, -1), band))
    out = []
    for c in cs[srt]:
        th = ang_fn(c) + math.radians(r.uniform(-14, 14))
        d = np.array([math.cos(th), math.sin(th)])
        n = np.array([-d[1], d[0]])
        nl = max(2, int(patch * shrink / spacing * r.uniform(0.75, 1.15)))
        lines = []
        for j in range(nl):
            o = (j - nl / 2) * spacing * r.uniform(0.85, 1.15)
            half = r.uniform(*length) / 2
            cc = c + n * o + d * r.uniform(-3, 3)
            ts = np.linspace(-half, half, max(4, int(2 * half / 1.5)))
            pts = cc + ts[:, None] * d
            ins = region.inside(pts)
            if ins.sum() < 4:
                continue
            idx = np.nonzero(ins)[0]
            # longest contiguous inside run
            br = np.nonzero(np.diff(idx) > 1)[0]
            runs = np.split(idx, br + 1)
            run = max(runs, key=len)
            seg = pts[run[0]:run[-1] + 1]
            if len(lines) % 2 == 1:
                seg = seg[::-1]
            lines.append(seg)
        if not lines:
            continue
        path = [lines[0]]
        for a_, b_ in zip(lines[:-1], lines[1:]):
            # rounded turn-around between back-and-forth strokes
            m_ = (a_[-1] + b_[0]) / 2 + d * (1.5 if np.dot(a_[-1] - a_[0], d) > 0 else -1.5)
            path.append(np.array([m_]))
            path.append(b_)
        P = np.vstack(path)
        P = crom(P[::3], step=1.4) if len(P) > 8 else P
        P = P + normals(P) * lowfreq(len(P), arclen(P), 0.4, r)[:, None]
        out.append(stroke(P, pig, w * r.uniform(0.9, 1.1), k * r.uniform(0.8, 1.1), v * r.uniform(0.85, 1.15), 'fill'))
    return out


# ============================================================================ the monarch
FORE = [(6, -30), (40, -72), (95, -110), (150, -129), (190, -135), (212, -125), (215, -104), (205, -74),
        (190, -42), (172, -14), (146, 4), (100, 5), (50, -5), (14, -18)]
HIND = [(8, -14), (60, -8), (118, 4), (160, 26), (179, 58), (175, 96), (151, 128), (112, 147), (70, 143),
        (36, 119), (16, 80), (6, 36)]
FCELL = [(16, -24), (70, -60), (118, -62), (126, -42), (82, -22), (22, -16)]
HCELL = [(12, -6), (60, 3), (98, 30), (82, 58), (40, 48), (14, 24)]
FVEINS = [[(70, -60), (120, -112), (162, -127)], [(96, -61), (150, -101), (196, -121)],
          [(118, -62), (166, -92), (208, -103)], [(124, -51), (172, -64), (205, -74)],
          [(126, -42), (167, -38), (191, -42)], [(104, -32), (150, -16), (174, -13)],
          [(60, -20), (110, -5), (146, 3)]]
HVEINS = [[(60, 3), (112, 10), (157, 26)], [(98, 30), (142, 41), (176, 56)], [(92, 44), (140, 80), (174, 96)],
          [(82, 58), (122, 104), (151, 127)], [(64, 55), (92, 108), (112, 146)], [(44, 49), (58, 100), (70, 142)],
          [(24, 40), (30, 84), (38, 117)]]


def wing_paths(X, smooth=False):
    F = crom(FORE, closed=True)
    Hh = crom(HIND, closed=True)
    if smooth:  # simplified silhouette for 'Normal'
        F = crom(FORE[::2], closed=True)
        Hh = crom(HIND[::2], closed=True)
    return dict(FR=X(F), HR=X(Hh), FL=X(mirror(F)), HL=X(mirror(Hh)))


def body_paths(X, rich=True):
    head = X(ell((0, -40), 9, 9, a0=-90))
    thor = X(ell((0, -15), 11, 22, a0=-90))
    abd = X(crom([(0, 6), (8, 18), (9, 40), (7, 62), (3, 80), (0, 84), (-3, 80), (-7, 62), (-9, 40), (-8, 18)], closed=True))
    antR = X(crom([(3, -47), (12, -78), (24, -110), (34, -132)]))
    antL = X(mirror(crom([(3, -47), (12, -78), (24, -110), (34, -132)])))
    return dict(head=head, thor=thor, abd=abd, antR=antR, antL=antL,
                clubR=X(np.array([[34, -134]])), clubL=X(np.array([[-34, -134]])))


def radial(c0):
    """colouring direction: along the veins, outward from the wing base"""
    def f(p):
        d = p - c0
        return math.atan2(d[1], d[0])
    return f


def region_for(polys, pad=40):
    allp = np.vstack(polys)
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad
    R_ = Region(x0, y0, x1, y1)
    for p in polys:
        R_.add(p)
    return R_


def bf_simple(r):
    X = Xf(CELLS['simple'], 0.98)
    fR = X(ell((88, -58), 82, 64, a0=200, rot=-28)); hR = X(ell((72, 58), 58, 52, a0=200, rot=24))
    fL = X(mirror(ell((88, -58), 82, 64, a0=200, rot=-28))); hL = X(mirror(ell((72, 58), 58, 52, a0=200, rot=24)))
    body = X(ell((0, 6), 13, 66, a0=-90))
    head = X(ell((0, -76), 16, 15, a0=-90))
    antR = X(crom([(5, -90), (18, -124), (36, -142), (46, -136), (40, -128)]))
    antL = X(mirror(crom([(5, -90), (18, -124), (36, -142), (46, -136), (40, -128)])))
    smile = X(crom([(-6, -71), (0, -67), (6, -71)]))
    out = []
    for pth, cl in [(head, True), (body, True), (fR, True), (hR, True), (fL, True), (hL, True), (antR, False), (antL, False), (smile, False)]:
        out += human_contour(pth, r, 'line', closed=cl, w=2.6, k=0.75, v=420)
    out += [dot(X(c), 2.6, r, 'g', 3.2) for c in [(-6, -79), (6, -79)]]
    ph1 = dict(name='simple-draw', ant='simple', pig='g', strokes=out)
    col = []
    for poly, base in [(fR, (10, -10)), (hR, (10, 10)), (fL, (-10, -10)), (hL, (-10, 10))]:
        reg = region_for([poly]).erode(2.0)
        col += patches(reg, radial(X(base)), r, 'o', patch=34, spacing=4.4, length=(22, 40), w=7.5, k=0.8, v=520)
    ph2 = dict(name='simple-colour', ant='simple', pig='o', strokes=col)
    dots = []
    for c, rad in [((92, -62), 15), ((74, 60), 11), ((-92, -62), 15), ((-74, 60), 11)]:
        reg = region_for([X(ell(c, rad, rad))])
        dots += patches(reg, lambda p: 0.6, r, 'k', patch=24, spacing=3.2, length=(8, 26), w=5.0, k=1.0, v=300)
    breg = region_for([body, head]).erode(1.5)
    dots += patches(breg, lambda p: 1.57, r, 'k', patch=18, spacing=3.4, length=(10, 26), w=5.0, k=0.95, v=320)
    ph3 = dict(name='simple-spots', ant='simple', pig='k', strokes=dots)
    return [ph1, ph2, ph3]


def bf_normal(r):
    X = Xf(CELLS['normal'], 1.0)
    W_ = wing_paths(X, smooth=True)
    B = body_paths(X)
    sk = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        sk += human_contour(W_[nm], r, 'sketch', closed=True, w=1.8, k=0.38)
    for nm in ['thor', 'abd', 'head']:
        sk += human_contour(B[nm], r, 'sketch', closed=True, w=1.8, k=0.38)
    ln = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        ln += human_contour(W_[nm], r, 'line', closed=True, w=2.4, k=0.75)
    for nm in ['head', 'thor', 'abd']:
        ln += human_contour(B[nm], r, 'line', closed=True, w=2.4, k=0.8)
    for nm in ['antR', 'antL']:
        ln += human_contour(B[nm], r, 'detail', w=2.0, k=0.75)
    ln += [dot(B['clubR'][0], 3.2, r, 'g', 3.4), dot(B['clubL'][0], 3.2, r, 'g', 3.4)]
    # two simple veins per wing
    for v_ in [FVEINS[1], FVEINS[4], HVEINS[2], HVEINS[4]]:
        p = crom(v_)
        ln += human_contour(X(p), r, 'detail', w=1.9, k=0.6)
        ln += human_contour(X(mirror(p)[::-1]), r, 'detail', w=1.9, k=0.6)
    col = []
    for nm, base in [('FR', (10, -16)), ('HR', (10, 6)), ('FL', (-10, -16)), ('HL', (-10, 6))]:
        reg = region_for([W_[nm]]).erode(2.0)
        col += patches(reg, radial(X(base)), r, 'o', patch=32, spacing=4.0, length=(20, 36), w=7.0, k=0.85, v=500)
    blk = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        inner = inset(W_[nm], 5.0)
        blk += human_contour(inner, r, 'line', closed=True, pig='k', w=8.5, k=0.95, v=320)
    breg = region_for([B['thor'], B['abd'], B['head']]).erode(1.2)
    blk += patches(breg, lambda p: 1.57, r, 'k', patch=16, spacing=3.0, length=(10, 30), w=4.5, k=1.0, v=320)
    return [dict(name='normal-sketch', ant='normal', pig='g', strokes=sk + ln),
            dict(name='normal-colour', ant='normal', pig='o', strokes=col),
            dict(name='normal-border', ant='normal', pig='k', strokes=blk)]


def monarch_parts(X):
    W_ = wing_paths(X)
    B = body_paths(X)
    veins = []
    for v_ in FVEINS + HVEINS:
        p = crom(v_)
        veins.append(X(p)); veins.append(X(mirror(p)[::-1]))
    cells = [X(crom(FCELL, closed=True)), X(crom(HCELL, closed=True)),
             X(mirror(crom(FCELL, closed=True))), X(mirror(crom(HCELL, closed=True)))]
    return W_, B, veins, cells


def band_fill(poly, bw, dots, drad, r, pig='k', step=2.3, w=3.6, k=1.0, v=420):
    """fill a border band with strokes that follow the edge, stepping around the dots"""
    out = []
    for o in np.arange(1.2, bw - 0.4, step):
        Q = inset(poly, o + r.uniform(-0.4, 0.4))
        keep = np.ones(len(Q), bool)
        for d in dots:
            keep &= np.hypot(Q[:, 0] - d[0], Q[:, 1] - d[1]) > drad + 1.6
        idx = np.nonzero(keep)[0]
        if len(idx) < 4:
            continue
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(run) > 4:
                out += human_contour(Q[run], r, 'detail', pig=pig, w=w, k=k, v=v)
    return out


def white_dot_centres(poly, cx, cy, bw, r, fore, s=1.0, rows=(0.36, 0.74), gap=17.0):
    """monarch dots: two rows along the OUTER margin only (not along the body side or where wings overlap)"""
    out = []
    for fr in rows:
        Q = inset(poly, bw * fr)
        s_ = arclen(Q)
        for t in np.arange(r.uniform(0, gap), s_[-1], gap * r.uniform(0.9, 1.1)):
            q = Q[min(int(np.searchsorted(s_, t)), len(Q) - 1)]
            dx, dy = abs(q[0] - cx), q[1] - cy
            if dx < 70 * s:
                continue
            if fore and dy > -22 * s:
                continue                                       # forewing inner margin
            if not fore and dy < 30 * s and dx < 150 * s:
                continue                                       # hindwing top edge under the forewing
            out.append(q)
    return out


def bf_better(r, key='better', real=False):
    s = 1.03
    X = Xf(CELLS[key], s)
    W_, B, veins, cells = monarch_parts(X)
    sk = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        sk += human_contour(W_[nm], r, 'sketch', closed=True, w=1.7, k=0.34)
    for nm in ['thor', 'abd', 'head']:
        sk += human_contour(B[nm], r, 'sketch', closed=True, w=1.7, k=0.34)
    for nm in ['antR', 'antL']:
        sk += human_contour(B[nm], r, 'sketch', w=1.6, k=0.32)
    ln = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        ln += human_contour(W_[nm], r, 'line', closed=True, w=2.2, k=0.7)
    for c_ in cells:
        ln += human_contour(c_, r, 'detail', closed=True, w=1.8, k=0.55)
    for v_ in veins:
        ln += human_contour(v_, r, 'detail', w=1.8, k=0.55)
    for nm in ['head', 'thor', 'abd']:
        ln += human_contour(B[nm], r, 'line', closed=True, w=2.2, k=0.75)
    for nm in ['antR', 'antL']:
        ln += human_contour(B[nm], r, 'detail', w=1.8, k=0.7)
    phases = [dict(name=f'{key}-sketch', ant=key, pig='g', strokes=sk + ln)]

    bw = 15 * s
    wing_regs = {nm: region_for([W_[nm]]) for nm in ['FR', 'HR', 'FL', 'HL']}
    bases = dict(FR=X((8, -18)), HR=X((8, 4)), FL=X((-8, -18)), HL=X((-8, 4)))
    inner = {nm: wing_regs[nm].erode(bw - 2.5) for nm in wing_regs}
    if real:
        yel = []
        for nm in ['FR', 'FL', 'HR', 'HL']:
            reg = inner[nm].erode(1.0)
            yel += patches(reg, radial(bases[nm]), r, 'y', patch=40, spacing=5.0, length=(24, 44), w=8.0, k=0.55, v=560)
        phases.append(dict(name=f'{key}-yellow', ant=key, pig='y', strokes=yel))
    org = []
    for nm in ['FR', 'FL', 'HR', 'HL']:
        reg = inner[nm].erode(0.5)
        if real:   # several light layers at different angles build a smooth, saturated orange
            for j, (da, kk, sp) in enumerate([(0.0, 0.55, 3.4), (0.45, 0.45, 3.2), (-0.4, 0.35, 3.6)]):
                new = patches(reg, lambda p, b=bases[nm], da=da: radial(b)(p) + da, r, 'o', patch=32, spacing=sp,
                              length=(18, 36), w=7.0, k=kk, v=560)
                if j == 2:
                    for st_ in new:
                        st_['burnish'] = True                      # heavy final layer pushes into the tooth
                org += new
        else:
            org += patches(reg, radial(bases[nm]), r, 'o', patch=34, spacing=3.8, length=(20, 38), w=7.0, k=0.9, v=520)
    phases.append(dict(name=f'{key}-orange', ant=key, pig='o', strokes=org))
    if real:
        brn = []
        for nm in ['FR', 'FL', 'HR', 'HL']:
            near_base = inner[nm].near(bases[nm], 70 * s)
            edge = inner[nm].minus(inner[nm].erode(12))
            for reg, kk in [(near_base, 0.42), (edge, 0.38), (inner[nm].near(bases[nm], 40 * s), 0.35)]:
                brn += patches(reg, radial(bases[nm]), r, 'b', patch=24, spacing=2.9, length=(12, 26), w=6.0, k=kk, v=480)
        for v_ in veins:   # burnt orange hugging every vein
            brn.append(stroke(v_, 'b', (9.0, 5.0), 0.38, 360, 'line'))
        phases.append(dict(name=f'{key}-shade', ant=key, pig='b', strokes=brn))
    # black: veins, border band (leaving the white dots), body
    blk = []
    for v_ in veins:
        blk += human_contour(v_, r, 'detail', pig='k', w=3.4 if real else 3.0, k=1.0, v=300)
    for c_ in cells:
        blk += human_contour(c_, r, 'detail', closed=True, pig='k', w=3.0, k=0.95, v=300)
    dots_all = []
    for nm in ['FR', 'HR', 'FL', 'HL']:
        side = 1 if nm.endswith('R') else -1
        band = wing_regs[nm].minus(inner[nm])
        dots = white_dot_centres(W_[nm], CELLS[key][0], CELLS[key][1], bw, r, nm.startswith('F'), s) if real else []
        dots_all += dots
        drad = 2.2 * s
        blk += human_contour(W_[nm], r, 'line', closed=True, pig='k', w=4.0, k=1.0, v=320)
        blk += band_fill(W_[nm], bw + 2.0, dots, drad, r, 'k', step=2.1, w=3.8, k=1.0, v=440)
    breg = region_for([B['thor'], B['abd'], B['head']]).erode(1.0)
    blk += patches(breg, lambda p: 1.57, r, 'k', patch=14, spacing=2.8, length=(8, 24), w=4.0, k=1.0, v=300)
    for nm in ['antR', 'antL']:
        blk += human_contour(B[nm], r, 'detail', pig='k', w=2.4, k=1.0, v=320)
    blk += [dot(B['clubR'][0], 3.6, r, 'k', 3.6), dot(B['clubL'][0], 3.6, r, 'k', 3.6)]
    phases.append(dict(name=f'{key}-black', ant=key, pig='k', strokes=blk))
    if real:
        wht = [dot(d, 1.6 * s, r, 'w', 2.6, 1.0) for d in dots_all]
        # apical white spots, thorax spots and a little sheen
        for c, rad in [((172, -108), 5), ((186, -98), 4.5), ((160, -114), 4), ((196, -86), 3.6)]:
            wht.append(dot(X(c), rad * 0.7, r, 'w', 3.4, 0.9))
            wht.append(dot(X((-c[0], c[1])), rad * 0.7, r, 'w', 3.4, 0.9))
        for c in [(-4, -26), (4, -22), (-4, -14), (4, -8), (0, -42)]:
            wht.append(dot(X(c), 1.4, r, 'w', 2.6, 0.9))
        for nm in ['FR', 'FL']:
            reg = inner[nm].near(X((120 if nm == 'FR' else -120, -80)), 30 * s)
            wht += patches(reg, radial(bases[nm]), r, 'w', patch=22, spacing=4.0, length=(10, 22), w=5.0, k=0.09, v=520)
        phases.append(dict(name=f'{key}-white', ant=key, pig='w', strokes=wht))
        # cast shadow on the paper (down-right) and crisp final edges
        shd = []
        allw = region_for([W_['FR'], W_['HR'], W_['FL'], W_['HL']], pad=60)
        sh = allw.copy(); sh.m[:] = 0
        for nm in ['FR', 'HR', 'FL', 'HL']:
            sh.add(W_[nm] + np.array([11, 16]) * s)
        sh = sh.minus(allw)
        for da, kk in [(0.35, 0.3), (-0.5, 0.22)]:
            shd += patches(sh, lambda p, da=da: da, r, 'g', patch=26, spacing=2.8, length=(12, 26), w=8.0, k=kk, v=520)
        fin = []
        for nm in ['FR', 'HR', 'FL', 'HL']:
            fin += human_contour(W_[nm], r, 'line', closed=True, pig='k', w=2.4, k=1.0, v=380)
        for nm in ['abd', 'thor']:   # body hairs
            P = resample(B[nm], 6.0)
            c0 = P.mean(0)
            for q in P[::2]:
                dd = q - c0; dd /= np.linalg.norm(dd) + 1e-9
                fin.append(stroke(np.array([q - dd * 2, q + dd * r.uniform(3, 6)]), 'g', 1.2, 0.8, 200, 'line'))
        phases.append(dict(name=f'{key}-shadow', ant=key, pig='g', strokes=shd + fin))
    return phases


def build_phases():
    r = np.random.default_rng(21)
    ph = []
    ph += bf_simple(r)
    ph += bf_normal(r)
    ph += bf_better(r, 'better', real=False)
    ph += bf_better(r, 'real', real=True)
    return ph


# ============================================================================ timeline
def mj_inv(tau):
    u = np.linspace(0, 1, 400)
    s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    return np.interp(tau, s, u)


def stroke_times(st, r):
    p = st['p']
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    ds = np.maximum(ds, 1e-3)
    L = ds.sum()
    v0 = st['v'] * SPEED * r.uniform(0.85, 1.15)
    if st['kind'] == 'dot':
        return np.r_[0, np.cumsum(ds / v0)]
    if len(p) < 4:                                    # quick flick
        return mj_inv(np.r_[0, np.cumsum(ds)] / L) * (L / v0 * 1.6 + 0.03)
    d = np.diff(p, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kap = np.abs(np.gradient(ang)) / ds
    if len(kap) >= 7:
        kap = np.convolve(kap, np.ones(7) / 7, mode='same')
    kap = np.clip(kap, 1 / 500, 1 / 2.5)
    vloc = (kap * 60) ** (-1 / 3)                                  # 2/3 power law, ~1 at radius 60
    vloc = np.clip(vloc, 0.35, 1.6)
    tnat = np.r_[0, np.cumsum(ds / vloc)]
    tau = tnat / tnat[-1]
    T = (tnat[-1] / v0) * 1.45 + 0.04 + 0.012 * math.sqrt(L)
    return mj_inv(tau) * T


def build_timeline(phases):
    r = np.random.default_rng(5)
    strokes, segs = [], []
    t = 0.0
    for pi_, ph in enumerate(phases):
        ph['t0'] = t
        prev = None
        for st in ph['strokes']:
            st['tt'] = stroke_times(st, r)
            if prev is not None:
                a, b = prev, st['p'][0]
                dist = np.linalg.norm(b - a)
                dur = 0.07 + 0.035 * math.sqrt(dist) + r.uniform(0, 0.05)
                if st['kind'] == 'fill' and dist < 30:
                    dur *= 0.6
                if r.random() < 0.06 and st['kind'] != 'fill':
                    dur += r.uniform(0.15, 0.45)                   # a look at the drawing
                segs.append(dict(kind='air', t0=t, t1=t + dur, a=a, b=b, amp=min(1.0, 0.25 + dist / 250)))
                t += dur
            st['t0'] = t
            st['t1'] = t + st['tt'][-1]
            st['ph'] = pi_
            segs.append(dict(kind='ink', t0=t, t1=st['t1'], s=len(strokes)))
            strokes.append(st)
            t = st['t1']
            prev = st['p'][-1]
        ph['t1'] = t
        t += 0.6
    return strokes, segs, t


# ============================================================================ hand
HAND = json.load(open(ASSET('hand2.json')))
HAND_IMG = {}
PIG_HAND = dict(g='graphite', o='orange', y='yellow', b='brown', k='black', w='white')
HSU = 1.5                     # hand units per source px = the reference reel's own scale
BASE_PHI = dict(simple=math.radians(4), normal=math.radians(24), better=math.radians(10), real=math.radians(26))


def hand_img(pig):
    nm = PIG_HAND[pig]
    if nm not in HAND_IMG:
        a = cv2.imread(ASSET(f'hand2_{nm}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        HAND_IMG[nm] = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    return HAND_IMG[nm]


def hand_matrix(tip_s, scale, phi):
    c, s = math.cos(phi), math.sin(phi)
    A = np.array([[c, -s], [s, c]]) * scale
    t = np.asarray(tip_s) - A @ np.array(HAND['tip'])
    return np.hstack([A, t[:, None]])


class Session:
    def __init__(self):
        self.phases = build_phases()
        self.strokes, self.segs, self.T = build_timeline(self.phases)
        self.seg_t0 = np.array([s['t0'] for s in self.segs])
        self.st_t0 = np.array([s['t0'] for s in self.strokes])
        self.st_t1 = np.array([s['t1'] for s in self.strokes])
        self.hz = 120
        n = int(self.T * self.hz) + 2
        tips = np.array([self.tip(i / self.hz)[0] for i in range(n)])
        self.tips = tips
        self.anchor = self._spring(tips)

    def seg_at(self, t):
        i = int(np.searchsorted(self.seg_t0, t, side='right')) - 1
        return self.segs[max(0, min(i, len(self.segs) - 1))]

    def phase_at(self, t):
        for ph in self.phases:
            if ph['t0'] - 1e-6 <= t <= ph['t1'] + 0.7:
                return ph
        return self.phases[-1]

    def tip(self, t):
        sg = self.seg_at(t)
        if sg['kind'] == 'ink':
            st = self.strokes[sg['s']]
            u = min(max(t - st['t0'], 0), st['tt'][-1])
            j = int(np.searchsorted(st['tt'], u, side='right')) - 1
            j = min(max(j, 0), len(st['p']) - 2)
            f = (u - st['tt'][j]) / max(st['tt'][j + 1] - st['tt'][j], 1e-6)
            f = min(max(f, 0), 1)
            return st['p'][j] * (1 - f) + st['p'][j + 1] * f, True, 0.0
        u = (t - sg['t0']) / (sg['t1'] - sg['t0'])
        u = min(max(u, 0), 1)
        e = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        a, b = sg['a'], sg['b']
        d = b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        pos = a + d * e + nrm * math.sin(math.pi * u) * 0.05 * np.linalg.norm(d)
        return pos, False, math.sin(math.pi * u) * sg['amp']

    def _spring(self, tips):
        """the hand itself follows the tip with a lag; the fingers make up the difference"""
        w = np.zeros_like(tips)
        vel = np.zeros(2)
        om = 2 * math.pi * 1.6
        dt = 1 / self.hz
        starts = {int(ph['t0'] * self.hz) for ph in self.phases}
        cur = tips[0].copy()
        for i, p in enumerate(tips):
            if i in starts:
                cur, vel = p.copy(), np.zeros(2)
            acc = om * om * (p - cur) - 2 * om * vel
            vel += acc * dt
            cur = cur + vel * dt
            w[i] = cur
        return w

    def hand_pose(self, t, ant):
        """-> (phi, scale factor) so the pencil tip sits on the tip while the hand lags"""
        i = min(int(t * self.hz), len(self.tips) - 1)
        d = self.tips[i] - self.anchor[i]
        nd = np.linalg.norm(d)
        if nd > 16:
            d = d / nd * 16
        ax_ = self.anchor[i]
        reach = min(max((ax_[0] - 380) / 520, 0), 1)
        reach = reach * reach * (3 - 2 * reach)
        phi0 = BASE_PHI[ant] * 0.4 + math.radians(4 + (36 if ax_[1] < 950 else 44) * reach)
        ax = np.array([math.cos(phi0), math.sin(phi0)])            # tip -> hand direction
        lat = ax[0] * d[1] - ax[1] * d[0]
        along = np.dot(ax, d)
        Lg = 300.0
        tip = self.tips[i]
        c = np.array(CELLS[ant])
        slow = 0.05 * (tip[1] - c[1]) / 150 - 0.03 * (tip[0] - c[0]) / 200
        phi = phi0 + lat / Lg * 0.9 + slow
        sc = 1 - 0.5 * along / Lg
        return phi, sc


# ============================================================================ canvas
PIGS = ['y', 'o', 'b', 'k', 'g', 'w']
PCOL = dict(y=(0.97, 0.78, 0.22), o=(0.93, 0.43, 0.07), b=(0.52, 0.15, 0.04), k=(0.07, 0.06, 0.06),
            w=(0.97, 0.96, 0.93))
POPA = dict(y=0.80, o=0.92, b=0.85, k=0.96, w=0.92)
PDEN = dict(y=1.3, o=1.6, b=1.5, k=1.9, g=1.9, w=1.5)


class Canvas:
    def __init__(self):
        r = np.random.default_rng(7)
        h = np.zeros((WH, WW), np.float32)
        fine = np.zeros((WH, WW), np.float32)
        for sig, amp in [(1.0, 0.40), (2.4, 0.50), (5.0, 0.30), (14.0, 0.18)]:
            n = r.standard_normal((WH, WW), dtype=np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sig)
            n /= n.std() + 1e-6
            h += amp * n
            if sig < 3:
                fine += amp * n
        h /= h.std(); fine /= fine.std()
        gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
        sh = 1.0 - 0.036 * (gx * 0.55 + gy * 0.8)
        yy = np.linspace(0, 1, WH, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, WW, dtype=np.float32)[None, :]
        big = cv2.resize(r.standard_normal((10, 6)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
        light = 1.05 - 0.10 * yy + 0.025 * xx + 0.012 * big
        self.light = np.clip(sh * light, 0.6, 1.3).astype(np.float32)
        self.hn = np.clip(fine * 0.2 + 0.5 + 0.03 * h, 0, 1).astype(np.float32)
        del gx, gy, sh, h, fine
        self.D = {k: np.zeros((WH, WW), np.float32) for k in PIGS}
        self.state = {}
        lab = cv2.imread(ASSET('labels2.png'), cv2.IMREAD_UNCHANGED)
        self.labels = cv2.cvtColor(lab, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255 if lab is not None else None

    def advance(self, S, t_from, t_to):
        i0 = int(np.searchsorted(S.st_t1, t_from, side='right'))
        i1 = int(np.searchsorted(S.st_t0, t_to, side='right'))
        for si in range(i0, i1):
            st = S.strokes[si]
            u = min(t_to, st['t1']) - st['t0']
            j = int(np.searchsorted(st['tt'], u, side='right')) - 1
            j = min(j, len(st['p']) - 1)
            pts = st['p'][:j + 1]
            if j < len(st['p']) - 1:
                f = (u - st['tt'][j]) / max(st['tt'][j + 1] - st['tt'][j], 1e-6)
                pts = np.vstack([pts, st['p'][j] * (1 - f) + st['p'][j + 1] * f])
            self._deposit(si, st, pts)
            if j >= len(st['p']) - 1 and si in self.state:
                del self.state[si]                              # finished: free the buffers

    def _deposit(self, si, st, pts_u):
        w0, w1 = (st['w'], st['w']) if np.isscalar(st['w']) else st['w']
        wmax = max(w0, w1) * WS
        if si not in self.state:
            P = to_w(st['p'])
            x0 = int(max(P[:, 0].min() - wmax - 4, 0)); y0 = int(max(P[:, 1].min() - wmax - 4, 0))
            x1 = int(min(P[:, 0].max() + wmax + 4, WW)); y1 = int(min(P[:, 1].max() + wmax + 4, WH))
            D = self.D[st['pig']]
            self.state[si] = dict(bb=(x0, y0, x1, y1), D0=D[y0:y1, x0:x1].copy(),
                                  S=np.zeros((y1 - y0, x1 - x0), np.uint8), n=0)
        S_ = self.state[si]
        n0 = max(S_['n'] - 1, 0)
        if len(pts_u) - n0 < 2:
            return
        x0, y0, x1, y1 = S_['bb']
        N = len(st['p'])
        P = to_w(pts_u[n0:])
        run = 5
        for a in range(0, len(P) - 1, run):
            b = min(a + run, len(P) - 1)
            seg = P[a:b + 1]
            fr = (n0 + b) / max(N - 1, 1)
            if st['kind'] == 'line':
                press = min(1.0, 0.45 + fr / 0.12 * 0.55) * (1 - 0.6 * max(0.0, (fr - 0.8) / 0.2))
            elif st['kind'] == 'fill':
                press = 0.75 + 0.25 * math.sin(math.pi * min(fr, 1.0))
            else:
                press = 1.0
            wd = (w0 + (w1 - w0) * fr) * WS * (0.65 + 0.35 * press)
            col = int(255 * min(1.0, 0.25 + 0.75 * press))
            q = np.round((seg - [x0, y0]) * 16).astype(np.int32)
            cv2.polylines(S_['S'], [q], False, col, thickness=max(1, int(round(wd))), lineType=cv2.LINE_AA, shift=4)
        S_['n'] = len(pts_u) - 1 if len(pts_u) < N else N
        bx0 = int(max(P[:, 0].min() - wmax - 3 - x0, 0)); by0 = int(max(P[:, 1].min() - wmax - 3 - y0, 0))
        bx1 = int(min(P[:, 0].max() + wmax + 3 - x0, x1 - x0)); by1 = int(min(P[:, 1].max() + wmax + 3 - y0, y1 - y0))
        if bx1 <= bx0 or by1 <= by0:
            return
        cov = S_['S'][by0:by1, bx0:bx1].astype(np.float32) / 255
        hn = self.hn[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
        k = st['k']
        thr = 0.66 - 0.45 * k * cov
        grain = np.clip((hn - thr) / 0.30 + 0.34, 0.26, 1.0)
        if st.get('burnish'):
            grain = 0.55 + 0.45 * grain
        dep = cov * grain * k * PDEN[st['pig']]
        self.D[st['pig']][y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = S_['D0'][by0:by1, bx0:bx1] + dep

    def view(self, M):
        f = dict(dsize=(OW, OH), flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        light = cv2.warpAffine(self.light, M, **f)
        col = np.empty((OH, OW, 3), np.float32)
        col[:] = np.array([0.655, 0.630, 0.596], np.float32)
        for pg in PIGS:
            D = self.D[pg]
            Dv = cv2.warpAffine(D, M, **f)
            if Dv.max() <= 0:
                continue
            c = (1 - np.exp(-1.7 * Dv))[..., None]
            if pg == 'g':
                col = col * (1 - c * 0.86)
            else:
                a = c * POPA[pg]
                col = col * (1 - a) + np.array(PCOL[pg], np.float32) * a
        return col * light[..., None]

    def labels_view(self, M, col):
        if self.labels is None:
            return col
        lab = cv2.warpAffine(self.labels, M, (OW, OH), flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
        a = lab[..., 3:]
        return col * (1 - a) + lab[..., :3] * a


# ============================================================================ edit
CAPS = {'simple-draw': 3.8, 'simple-colour': 2.4, 'simple-spots': 1.6,
        'normal-sketch': 3.8, 'normal-colour': 2.2, 'normal-border': 2.4,
        'better-sketch': 4.0, 'better-orange': 2.2, 'better-black': 3.0,
        'real-sketch': 2.8, 'real-yellow': 1.5, 'real-orange': 2.0, 'real-shade': 1.8, 'real-black': 2.6,
        'real-white': 1.8, 'real-shadow': 2.2}
HOLD = 2.6


def shot_plan(S):
    plan = []
    for ph in S.phases:
        cap = CAPS[ph['name']]
        t0, t1 = ph['t0'], ph['t1']
        if t1 - t0 <= cap:
            plan.append(dict(t0=t0, t1=t1, ph=ph['name'], hand=True))
        else:   # beginning, jump cut, the end (cut on the last stroke)
            a = cap * 0.55
            plan.append(dict(t0=t0, t1=t0 + a, ph=ph['name'], hand=True))
            plan.append(dict(t0=t1 - (cap - a), t1=t1, ph=ph['name'] + '/end', hand=True))
    plan.append(dict(t0=S.T, t1=S.T + HOLD, ph='final-hold', hand=False))
    t = 0.0
    for sh in plan:
        sh['o0'] = t
        t += sh['t1'] - sh['t0']
        sh['o1'] = t
    return plan, t


def hand_edge_points():
    a = hand_img('g')[..., 3]
    pts = []
    for y in range(0, a.shape[0], 4):
        for x in range(0, a.shape[1], 4):
            if a[y, x] > 0.05 and x > HAND['cut_x'] + 200:
                pts.append((x, y))
    return np.array(pts, float)


# ============================================================================ frame
def world_M():
    a = 1.0 / WS
    return np.array([[a, 0, -UX0 * 1.0 - 0], [0, a, -UY0]], np.float32) * 1 + np.array([[0, 0, 0], [0, 0, 0]], np.float32)


def screen_M(to):
    # locked-off camera with a hint of tripod micro-movement
    dx = 0.35 * math.sin(to * 0.8) + 0.2 * math.sin(to * 2.1 + 1)
    dy = 0.3 * math.sin(to * 0.6 + 2) + 0.2 * math.sin(to * 1.7)
    a = 1.0 / WS
    return np.array([[a, 0, UX0 + dx], [0, a, UY0 + dy]], np.float32)


def comp_hand(col, S, t, M_ws, pig, ant):
    p, contact, lift = S.tip(t)
    p_prev = S.tip(max(t - 1 / 60, 0))[0]
    speed = np.linalg.norm(p - p_prev) * 60
    nsub = 1 if speed < 350 else (3 if speed < 1100 else 5)
    img = hand_img(pig)
    acc_rgb = np.zeros((OH, OW, 3), np.float32)
    acc_a = np.zeros((OH, OW), np.float32)
    shadow = None
    for kk in range(nsub):
        tt = t - kk * (1 / 60) / max(nsub - 1, 1) if nsub > 1 else t
        pk, ck, lk = S.tip(max(tt, 0))
        ps = M_ws[:, :2] @ to_w(pk) + M_ws[:, 2]
        phi, sc = S.hand_pose(max(tt, 0), ant)
        sc *= HSU * (1 + 0.03 * lk)
        M = hand_matrix(ps, sc, phi)
        w = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        acc_rgb += w[..., :3] * w[..., 3:]
        acc_a += w[..., 3]
        if kk == 0:
            q = 4
            Ms = M.copy()
            g = 1.05 + 0.07 * lk
            Ms[:, :2] *= g
            Ms[:, 2] = ps + g * (M[:, 2] - ps) + np.array([12, 16]) * lk
            Ms = Ms / q
            a = cv2.warpAffine(img[..., 3], Ms, (OW // q, OH // q), flags=cv2.INTER_LINEAR)
            a = cv2.GaussianBlur(a, (0, 0), 2.4 + 3 * lk)
            shadow = cv2.resize(a, (OW, OH), interpolation=cv2.INTER_LINEAR)
    acc_rgb /= nsub
    acc_a /= nsub
    col = col * (1 - 0.30 * shadow[..., None])
    return col * (1 - acc_a[..., None]) + acc_rgb


def grade(col, to):
    if not hasattr(grade, 'vig'):
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = ((xx - OW / 2) / OW) ** 2 + ((yy - OH * 0.42) / OH) ** 2
        grade.vig = (1 - 0.24 * r)[..., None].astype(np.float32)
    col = col * grade.vig
    n = np.random.default_rng(int(to * 1000)).standard_normal((OH // 2, OW // 2)).astype(np.float32)
    n = cv2.resize(n, (OW, OH), interpolation=cv2.INTER_LINEAR)[..., None] * 0.006
    return np.clip(col + n, 0, 1)


class Renderer:
    def __init__(self, S):
        self.S = S
        self.cv = Canvas()
        self.t_ink = -1.0
        self.plan, self.dur = shot_plan(S)

    def frame(self, k):
        to = (k + 0.5) / FPS
        sh = next((s for s in self.plan if s['o0'] <= to < s['o1']), self.plan[-1])
        t = sh['t0'] + (to - sh['o0'])
        tin = min(t, self.S.T)
        if tin > self.t_ink:
            self.cv.advance(self.S, self.t_ink, tin)
            self.t_ink = tin
        M = screen_M(to)
        col = self.cv.view(M)
        if sh['hand']:
            ph = self.S.phase_at(t)
            Mws = M.astype(np.float64)
            col = comp_hand(col, self.S, t, Mws, ph['pig'], ph['ant'])
        col = self.cv.labels_view(M, col)
        col = grade(col, to)
        return (col * 255 + 0.5).astype(np.uint8), sh, t

    def check_edges(self):
        E = hand_edge_points()
        bad = {}
        for sh in self.plan:
            if not sh['hand']:
                continue
            for t in np.arange(sh['t0'], sh['t1'], 0.1):
                ph = self.S.phase_at(t)
                p, _, lk = self.S.tip(t)
                M_ = screen_M(0).astype(np.float64)
                ps = M_[:, :2] @ to_w(p) + M_[:, 2]
                phi, sc = self.S.hand_pose(t, ph['ant'])
                M = hand_matrix(ps, sc * HSU, phi)
                e = E @ M[:, :2].T + M[:, 2]
                on = (e[:, 0] > -5) & (e[:, 0] < OW + 5) & (e[:, 1] > -5) & (e[:, 1] < OH + 5)
                if on.any():
                    bad[sh['ph']] = bad.get(sh['ph'], 0) + 1
        return bad

    def audio_track(self):
        sr = 48000
        n = int(self.dur * sr) + sr
        nf = int(self.dur * FPS) + 2
        env = np.zeros(nf); kind = np.zeros(nf)
        for k in range(nf):
            to = k / FPS
            sh = next((s for s in self.plan if s['o0'] <= to < s['o1']), None)
            if sh is None or not sh['hand']:
                continue
            t = sh['t0'] + (to - sh['o0'])
            p, c, _ = self.S.tip(t)
            if not c:
                continue
            pig = self.S.phase_at(t)['pig']
            p2 = self.S.tip(max(t - 1 / 120, 0))[0]
            v = np.linalg.norm(p - p2) * 120
            env[k] = min(1.0, (v / 600) ** 0.7) * (0.9 if pig == 'g' else 0.7)
            kind[k] = 0 if pig == 'g' else 1
        r = np.random.default_rng(3)
        x = np.arange(n) / sr
        e = np.interp(x, np.arange(nf) / FPS, env)
        kd = np.interp(x, np.arange(nf) / FPS, kind)
        e = np.convolve(e, np.ones(240) / 240, mode='same')
        noise = r.standard_normal(n)
        F = np.fft.rfftfreq(n, 1 / sr)
        N = np.fft.rfft(noise)

        def band(lo, hi, peak):
            g = np.exp(-0.5 * ((np.log(F + 1) - np.log(peak)) / 0.55) ** 2)
            g *= (F > lo) & (F < hi)
            return np.fft.irfft(N * g, n)
        gr = band(900, 11000, 4200); wx = band(300, 6000, 1800)
        gr /= np.abs(gr).max(); wx /= np.abs(wx).max()
        gx_ = np.arange(int(n / sr * 60) + 2) / 60
        grain = np.interp(x, gx_, 1 + 0.5 * r.standard_normal(len(gx_)))
        sig = (gr * (1 - kd) + wx * kd) * e * np.clip(grain, 0.3, 2.0)
        room = np.fft.irfft(np.fft.rfft(r.standard_normal(n)) / np.sqrt(F + 20), n)
        room /= np.abs(room).max()
        sig = sig * 0.55 + room * 0.004
        sig /= max(np.abs(sig).max(), 1e-6) / 0.6
        return (sig[:int(self.dur * sr)] * 32767).astype(np.int16), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--still', default='')
    ap.add_argument('--render', default='')
    ap.add_argument('--chunk', default='')
    ap.add_argument('--audio', default='')
    a = ap.parse_args()
    t0 = time.time()
    S = Session()
    R = Renderer(S)
    if a.plan:
        for ph in S.phases:
            print(f"{ph['name']:15s} sim {ph['t0']:6.1f}-{ph['t1']:6.1f} ({ph['t1'] - ph['t0']:5.1f}s, {len(ph['strokes'])} strokes)")
        for sh in R.plan:
            print(f"  shot {sh['ph']:18s} out {sh['o0']:5.2f}-{sh['o1']:5.2f}")
        print(f'total {R.dur:.2f}s ({int(R.dur * FPS)} frames)   setup {time.time() - t0:.1f}s')
        print('hand edge on screen (shot: samples):', R.check_edges())
    if a.still:
        os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
        for s in a.still.split(','):
            k = int(float(s) * FPS)
            img, sh, t = R.frame(k)
            cv2.imwrite(os.path.join(HERE, 'out', f'bf_{float(s):06.2f}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print('still', s, sh['ph'], f'sim {t:.2f}')
    if a.audio:
        pcm, sr = R.audio_track()
        import wave
        with wave.open(a.audio, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
        print('audio', a.audio)
    if a.render:
        nfr = int(R.dur * FPS)
        k0, k1 = 0, nfr
        if a.chunk:
            i, n = map(int, a.chunk.split('/'))
            k0, k1 = nfr * i // n, nfr * (i + 1) // n
        ff = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}',
                               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
                               '-pix_fmt', 'yuv420p', a.render], stdin=subprocess.PIPE)
        tm = time.time()
        for k in range(k0, k1):
            img, _, _ = R.frame(k)
            ff.stdin.write(img.tobytes())
            if (k - k0) % 30 == 0:
                print(f'frame {k}/{k1}  {time.time() - tm:.0f}s', flush=True)
        ff.stdin.close()
        ff.wait()


if __name__ == '__main__':
    main()
