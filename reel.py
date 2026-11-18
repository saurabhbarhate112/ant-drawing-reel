#!/usr/bin/env python3
"""4 Ways to Draw an Ant - programmatic pencil reel driven by a real hand cut-out.

Everything is simulated in one continuous "drawing session" timeline (sim time).
The edit picks shot windows out of it (hard cuts, forward in time only).

    python reel.py --plan                 # print phases, shots, durations, hand-safety check
    python reel.py --still 3.2,10,20      # write PNGs of output-time frames to out/
    python reel.py --render out/ant_reel.mp4 [--chunk i/n]
"""
import argparse, json, math, os, subprocess, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET = lambda *p: os.path.join(HERE, 'assets', *p)

OW, OH, FPS = 1080, 1920, 30
WS = 1.5                                         # world px per unit (1 unit = 1 output px at zoom 1)
UX0, UY0, UX1, UY1 = -340.0, -800.0, 1420.0, 2360.0
WW, WH = int((UX1 - UX0) * WS), int((UY1 - UY0) * WS)
ORG = np.array([UX0, UY0])

RNG = np.random.default_rng(11)
SPEED = 1.5                                      # global drawing pace (x real time)


def to_w(p):
    return (np.asarray(p, float) - ORG) * WS


# ----------------------------------------------------------------------------- geometry helpers
def crom(pts, closed=False, step=1.4):
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
    t = np.radians(np.linspace(a0, a0 + sweep, max(16, int(L / 1.4))))
    x, y = rx * np.cos(t), ry * np.sin(t)
    r = math.radians(rot)
    return np.stack([c[0] + x * math.cos(r) - y * math.sin(r), c[1] + x * math.sin(r) + y * math.cos(r)], -1)


def wobble(p, amp=0.6, seed=None):
    """tiny low-frequency hand wobble along the normal"""
    p = np.asarray(p, float)
    if len(p) < 4:
        return p
    r = np.random.default_rng(seed)
    n = len(p)
    k = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)])
    f = sum(r.normal() * np.sin(k / r.uniform(18, 60) + r.uniform(0, 6)) for _ in range(3)) / 3
    d = np.gradient(p, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], -1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    return p + nrm * (f * amp)[:, None]


def offset_line(c, hw):
    """offset a centre line by half-widths hw (array) on both sides -> (sideA, sideB)"""
    d = np.gradient(c, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], -1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return c + n * hw[:, None], c - n * hw[:, None]


class Xf:
    def __init__(self, c, s=1.0, rot=0.0):
        self.c, self.s, self.r = np.array(c, float), s, math.radians(rot)

    def __call__(self, p):
        p = np.asarray(p, float) * self.s
        cr, sr = math.cos(self.r), math.sin(self.r)
        return np.stack([p[..., 0] * cr - p[..., 1] * sr, p[..., 0] * sr + p[..., 1] * cr], -1) + self.c


def stroke(p, pig='g', w=2.4, k=0.6, v=430.0, kind='line'):
    p = np.asarray(p, float)
    return dict(p=p, pig=pig, w=w, k=k, v=v, kind=kind)


def hatch(polys, ang, sp, excl=(), inset=0.0, below=None, above=None, join=2.4, jitter=0.25, seed=0):
    """zig-zag colouring strokes that fill the polygons (units). Returns a list of polylines."""
    r = np.random.default_rng(seed)
    allp = np.vstack(polys)
    x0, y0 = allp.min(0) - 6
    x1, y1 = allp.max(0) + 6
    R = 2.0
    W_, H_ = int((x1 - x0) * R) + 2, int((y1 - y0) * R) + 2
    m = np.zeros((H_, W_), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round((p - [x0, y0]) * R).astype(np.int32)], 1)
    for p in excl:
        cv2.fillPoly(m, [np.round((p - [x0, y0]) * R).astype(np.int32)], 0)
    if inset > 0:
        k = int(2 * inset * R) | 1
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    yy = (np.arange(H_) / R + y0)[:, None]
    if below is not None:
        m &= (yy > below).astype(np.uint8)
    if above is not None:
        m &= (yy < above).astype(np.uint8)
    a = math.radians(ang)
    d, n = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    cs = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    pd, pn = cs @ d, cs @ n
    ts = np.arange(pd.min(), pd.max(), 0.5)
    out, cur, flip = [], None, False
    o = pn.min() + sp * 0.5
    while o < pn.max():
        pts = ts[:, None] * d + o * n
        ix = ((pts[:, 0] - x0) * R).astype(int)
        iy = ((pts[:, 1] - y0) * R).astype(int)
        ok = (ix >= 0) & (ix < W_) & (iy >= 0) & (iy < H_)
        ins = np.zeros(len(ts), bool)
        ins[ok] = m[iy[ok], ix[ok]] > 0
        df = np.diff(np.r_[0, ins.astype(int), 0])
        st, en = np.nonzero(df == 1)[0], np.nonzero(df == -1)[0] - 1
        runs = [(s, e) for s, e in zip(st, en) if e - s >= 3]
        if runs:
            s, e = max(runs, key=lambda q: q[1] - q[0])
            js = int(r.uniform(-jitter, jitter) * sp * 2)
            je = int(r.uniform(-jitter, jitter) * sp * 2)
            s, e = np.clip(s + max(js, 0), 0, len(ts) - 1), np.clip(e + min(je, 0), 0, len(ts) - 1)
            a_, b_ = (pts[e], pts[s]) if flip else (pts[s], pts[e])
            seg = np.linspace(a_, b_, max(2, int(np.linalg.norm(b_ - a_) / 2)))
            if cur is not None and np.linalg.norm(seg[0] - cur[-1]) < join * sp:
                cur = np.vstack([cur, seg])
            else:
                if cur is not None:
                    out.append(cur)
                cur = seg
            flip = not flip
        o += sp * r.uniform(0.85, 1.15)
    if cur is not None:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------- the four ants
CELLS = dict(simple=(270, 600), normal=(810, 590), better=(275, 1300), real=(805, 1300))
LABELS = dict(simple=(270, 340), normal=(810, 340), better=(270, 1040), real=(810, 1040))
LABEL_TEXT = dict(simple='Simple', normal='Normal', better='Better', real='Realistic')


def ant_simple():
    X = Xf(CELLS['simple'], 1.05)
    head = X(ell((-118, -22), 58, 57, a0=-60))
    eyes = [X(ell(c, 4.5, 4.5, a0=0, sweep=-500)) for c in [(-139, -32), (-104, -32)]]
    smile = X(crom([(-140, -4), (-122, 6), (-103, -3)]))
    antl = X(crom([(-140, -74), (-150, -106), (-170, -130), (-188, -128), (-189, -114)]))
    antr = X(crom([(-100, -76), (-92, -110), (-74, -132), (-58, -128), (-60, -116)]))
    thor = X(ell((-25, 2), 38, 37, a0=-70))
    abd = X(ell((92, -8), 76, 60, a0=-120))
    legs = [X(crom(q)) for q in [
        [(-48, 32), (-66, 66), (-86, 92), (-100, 94)], [(-25, 39), (-28, 80), (-40, 108), (-54, 110)],
        [(-2, 34), (10, 72), (6, 104), (-6, 108)], [(55, 44), (66, 84), (56, 110), (44, 112)],
        [(95, 50), (104, 88), (98, 112), (86, 114)], [(135, 40), (152, 76), (150, 104), (140, 108)]]]
    L1 = dict(w=2.6, k=0.62, v=430)
    outline = [head, *eyes, smile, antl, antr, thor, abd, *legs]
    p1 = [stroke(wobble(head, 0.7, 1), **L1)]
    p1 += [stroke(e, w=4.0, k=0.9, v=180, kind='dot') for e in eyes]
    p1 += [stroke(wobble(q, 0.5, i + 2), **L1) for i, q in enumerate([smile, antl, antr, thor, abd, *legs])]
    red = []
    for i, poly in enumerate([head, thor, abd]):
        for h in hatch([poly], 38 + 6 * i, 8.0, inset=2.5, seed=10 + i):
            red.append(stroke(h, pig='r', w=12.0, k=0.95, v=850, kind='fill'))
    L2 = dict(w=4.6, k=1.0, v=520)
    p2 = [stroke(wobble(q, 0.5, 40 + i), **L2) for i, q in enumerate([head, thor, abd, antl, antr, *legs, smile])]
    p2 += [stroke(e, w=6.0, k=1.0, v=170, kind='dot') for e in eyes]
    return [dict(name='simple-outline', ant='simple', pig='g', strokes=p1),
            dict(name='simple-colour', ant='simple', pig='r', strokes=red),
            dict(name='simple-ink', ant='simple', pig='g', strokes=p2)], outline


def ant_normal():
    X = Xf(CELLS['normal'], 1.02, rot=-32)
    head = X(ell((-128, 0), 33, 29, a0=-90))
    mand = [X(crom([(-156, -10), (-172, -8), (-176, 0)])), X(crom([(-156, 10), (-172, 8), (-176, 0)]))]
    xs = np.array([-98, -80, -60, -40, -22, -12.0])
    hw = np.array([12, 15, 13, 9, 8, 6.0])
    meso = X(crom(list(zip(xs, -hw)) + list(zip(xs[::-1], hw[::-1])), closed=True))
    pet = X(ell((1, 0), 9, 7.5, a0=-90))
    gas = X(ell((66, 0), 58, 42, a0=-150))
    legs = [X(crom(q)) for q in [
        [(-82, 12), (-104, 48), (-150, 74), (-182, 76)], [(-58, 12), (-52, 62), (-76, 116), (-88, 142)],
        [(-30, 9), (8, 52), (58, 96), (82, 120)], [(-82, -12), (-108, -46), (-146, -80), (-176, -90)],
        [(-58, -12), (-46, -66), (-66, -118), (-74, -146)], [(-30, -9), (14, -50), (62, -92), (88, -112)]]]
    ants = [X(crom(q)) for q in [[(-150, -10), (-168, -34), (-200, -44), (-228, -40)],
                                 [(-150, 10), (-170, 32), (-204, 40), (-230, 34)]]]
    L1 = dict(w=2.4, k=0.6, v=440)
    p1 = [stroke(wobble(q, 0.4, i), **L1) for i, q in enumerate([head, *mand, meso, pet, gas])]
    p1 += [stroke(wobble(q, 0.6, 10 + i), w=2.6, k=0.65, v=480) for i, q in enumerate(legs + ants)]
    fill = []
    for j, (ang, sp, k) in enumerate([(35, 5.0, 1.0), (-42, 6.5, 0.8)]):
        for i, poly in enumerate([gas, meso, head, pet]):
            for h in hatch([poly], ang + 4 * i, sp, inset=1.5, seed=30 + i + 10 * j):
                fill.append(stroke(h, w=7.0, k=k, v=850, kind='fill'))
    thick = [stroke(wobble(q, 0.4, 50 + i), w=(7.5, 2.6), k=1.0, v=420) for i, q in enumerate(legs)]
    thick += [stroke(q, w=(3.6, 2.2), k=0.95, v=480) for q in ants + mand]
    return [dict(name='normal-outline', ant='normal', pig='g', strokes=p1),
            dict(name='normal-fill', ant='normal', pig='g', strokes=fill + thick)], None


def side_ant(key, s):
    """side-view ant shared by 'better' (line art) and 'real' (coloured)."""
    X = Xf(CELLS[key], s)
    P = {}
    P['head'] = X(crom([(-176, -28), (-152, -54), (-120, -54), (-100, -34), (-102, -6), (-128, 10), (-162, 6), (-182, -8)], closed=True))
    P['eye'] = X(ell((-142, -30), 10, 8, a0=-90))
    P['mand'] = X(crom([(-180, -4), (-202, 4), (-198, 16), (-182, 12)]))
    P['meso'] = X(crom([(-100, -26), (-82, -46), (-52, -50), (-30, -42), (-18, -24), (-22, -4), (-52, 8), (-86, 6)], closed=True))
    P['sut'] = X(crom([(-62, -48), (-60, -20), (-64, 4)]))
    P['pet'] = X(crom([(-20, -16), (-10, -34), (2, -30), (6, -10), (-8, -2)], closed=True))
    P['gas'] = X(crom([(6, -18), (30, -48), (80, -60), (128, -48), (158, -18), (150, 14), (112, 30), (56, 28), (16, 10)], closed=True))
    P['segs'] = [X(crom(q)) for q in [[(48, -56), (42, -14), (52, 27)], [(90, -58), (86, -15), (96, 29)], [(126, -48), (124, -12), (130, 22)]]]
    near = [[(-82, 4), (-106, 36), (-140, 62), (-172, 92), (-186, 98)],
            [(-52, 6), (-36, 48), (-26, 92), (-12, 124), (-2, 130)],
            [(-26, 0), (30, 34), (96, 78), (150, 118), (166, 124)]]
    far = [[(-90, 0), (-128, 24), (-160, 44), (-176, 56)],
           [(-60, 4), (-72, 52), (-88, 96), (-96, 108)],
           [(-30, -2), (10, 48), (48, 92), (60, 104)]]
    P['legs'] = []
    for q, (h0, h1) in [(q, (5.2, 2.0)) for q in near] + [(q, (3.6, 1.6)) for q in far]:
        c = X(crom(q))
        hw = np.linspace(h0, h1, len(c)) * s
        a, b = offset_line(c, hw)
        P['legs'].append(dict(c=c, hw=hw, a=a, b=b, near=h0 > 4))
    P['ant'] = [X(crom([(-150, -52), (-166, -108), (-190, -124), (-226, -140), (-250, -142)])),
                X(crom([(-138, -54), (-142, -104), (-168, -134), (-196, -160), (-214, -164)]))]
    return P


def outline_strokes(P, w, k, v, seed):
    out = []
    order = ['head', 'eye', 'mand', 'meso', 'sut', 'pet', 'gas']
    for i, nm in enumerate(order):
        out.append(stroke(wobble(P[nm], 0.35, seed + i), w=w, k=k, v=v))
    for i, sg in enumerate(P['segs']):
        out.append(stroke(wobble(sg, 0.3, seed + 20 + i), w=w * 0.85, k=k * 0.85, v=v))
    for i, L in enumerate(P['legs']):
        out.append(stroke(wobble(L['a'], 0.3, seed + 40 + i), w=w, k=k, v=v))
        out.append(stroke(wobble(L['b'], 0.3, seed + 60 + i), w=w, k=k, v=v))
    for i, a in enumerate(P['ant']):
        out.append(stroke(wobble(a, 0.5, seed + 80 + i), w=w, k=k, v=v * 1.1))
    return out


def ant_better():
    P = side_ant('better', 1.0)
    p1 = outline_strokes(P, 2.0, 0.55, 420, 100)
    p2 = outline_strokes(P, 3.2, 0.95, 520, 200)
    p2.append(stroke(ell(np.mean(P['eye'], 0) + [2, 0], 3, 3, a0=0, sweep=-540), w=5.0, k=1.0, v=150, kind='dot'))
    return [dict(name='better-sketch', ant='better', pig='g', strokes=p1),
            dict(name='better-ink', ant='better', pig='g', strokes=p2)], P


def ant_real():
    s = 1.04
    P = side_ant('real', s)
    X = Xf(CELLS['real'], s)
    hl = dict(head=X(ell((-140, -42), 15, 6, rot=-12)), meso=X(ell((-62, -40), 20, 6, rot=-8)),
              gas=X(ell((76, -40), 38, 11, rot=-10)))
    sk = outline_strokes(P, 1.9, 0.42, 470, 300)
    red = []
    for i, nm in enumerate(['gas', 'meso', 'head', 'pet']):
        ex = [hl[nm]] if nm in hl else []
        for h in hatch([P[nm]], 30 + 7 * i, 6.5, excl=ex, inset=1.5, seed=60 + i):
            red.append(stroke(h, pig='r', w=8.5, k=0.88, v=850, kind='fill'))
    for L in P['legs']:
        red.append(stroke(L['c'], pig='r', w=(2 * L['hw'][0] + 1, 2 * L['hw'][-1] + 1), k=0.9 if L['near'] else 0.75, v=520))
    for a in P['ant']:
        red.append(stroke(a, pig='r', w=3.0, k=0.8, v=520))
    brown = []
    for i, nm in enumerate(['gas', 'meso', 'head', 'pet']):
        poly = P[nm]
        cy, h = poly[:, 1].mean(), np.ptp(poly[:, 1])
        for j, (frac, k) in enumerate([(0.05, 0.55), (0.28, 0.75)]):
            for hh in hatch([poly], -28 + 9 * i + 20 * j, 6.0, inset=1.5, below=cy + frac * h, seed=80 + i + 10 * j):
                brown.append(stroke(hh, pig='b', w=7.5, k=k, v=850, kind='fill'))
    for sg in P['segs']:
        brown.append(stroke(sg, pig='b', w=5.0, k=0.8, v=420))
    for L in P['legs']:
        under, _ = offset_line(L['c'], -L['hw'] * 0.45)
        brown.append(stroke(under, pig='b', w=(L['hw'][0] * 1.1 + 1, 1.5), k=0.8, v=520))
    ink = outline_strokes(P, 2.6, 0.72, 560, 400)
    ink.append(stroke(ell(np.mean(P['eye'], 0), 5, 4, a0=0, sweep=-700), w=4.5, k=1.0, v=150, kind='dot'))
    # cast shadow: soft graphite on the paper under the ant
    shadow = []
    feet_y = CELLS['real'][1] + 124 * s
    for hh in hatch([ell((CELLS['real'][0] - 8, feet_y + 2), 180, 19)], 6, 6.5, inset=0, seed=99):
        shadow.append(stroke(hh, w=13.0, k=0.24, v=900, kind='fill'))
    for hh in hatch([ell((CELLS['real'][0] - 12, feet_y - 2), 120, 9)], -8, 6.0, seed=98):
        shadow.append(stroke(hh, w=11.0, k=0.22, v=900, kind='fill'))
    return [dict(name='real-sketch', ant='real', pig='g', strokes=sk),
            dict(name='real-red', ant='real', pig='r', strokes=red),
            dict(name='real-shade', ant='real', pig='b', strokes=brown),
            dict(name='real-ink', ant='real', pig='g', strokes=ink + shadow)], P


def build_phases():
    ph = []
    for f in (ant_simple, ant_normal, ant_better, ant_real):
        ph += f()[0]
    return ph


# ----------------------------------------------------------------------------- timeline
def stroke_times(st, r):
    p = st['p']
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    L = ds.sum()
    s = np.r_[0, np.cumsum(ds)]
    d = np.diff(p, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    turn = np.r_[turn, 0]
    if len(turn) >= 5:
        turn = np.convolve(turn, np.ones(5) / 5, mode='same')
    turn = turn / np.maximum(ds, 0.3) * 4
    v0 = st['v'] * SPEED * r.uniform(0.88, 1.12)
    if st['kind'] == 'dot':
        return np.r_[0, np.cumsum(ds / v0)]
    v = v0 / (1 + 7 * turn)
    v = np.clip(v, 0.3 * v0, v0)
    sm = 0.5 * (s[1:] + s[:-1])
    ramp = np.clip(np.minimum(sm, L - sm) / (14 if st['kind'] == 'line' else 6) + 0.35, 0, 1)
    v = np.maximum(v * ramp, 25)
    return np.r_[0, np.cumsum(ds / v)]


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
                dur = 0.09 + dist / 1500 + r.uniform(0, 0.05)
                if st['kind'] == 'fill' and dist < 25:
                    dur *= 0.6
                segs.append(dict(kind='air', t0=t, t1=t + dur, a=a, b=b, amp=min(1.0, 0.25 + dist / 300)))
                t += dur
            st['t0'] = t
            st['t1'] = t + st['tt'][-1]
            st['ph'] = pi_
            segs.append(dict(kind='ink', t0=t, t1=st['t1'], s=len(strokes)))
            strokes.append(st)
            t = st['t1']
            prev = st['p'][-1]
        ph['t1'] = t
        t += 0.6                                       # tool change / off-camera gap, always cut out
    return strokes, segs, t


class Session:
    def __init__(self):
        self.phases = build_phases()
        self.strokes, self.segs, self.T = build_timeline(self.phases)
        self.seg_t0 = np.array([s['t0'] for s in self.segs])
        # wrist spring sampled at 240 Hz
        self.hz = 240
        n = int(self.T * self.hz) + 2
        self.wt = np.arange(n) / self.hz
        tips = np.array([self.tip(t)[0] for t in self.wt])
        self.tips = tips
        self.wrist = self._spring(tips)

    def seg_at(self, t):
        i = int(np.searchsorted(self.seg_t0, t, side='right')) - 1
        return self.segs[max(0, min(i, len(self.segs) - 1))]

    def tip(self, t):
        """-> (pos units, contact bool, lift 0..1, stroke index or None, pigment)"""
        sg = self.seg_at(t)
        if sg['kind'] == 'ink':
            st = self.strokes[sg['s']]
            u = min(max(t - st['t0'], 0), st['tt'][-1])
            j = int(np.searchsorted(st['tt'], u, side='right')) - 1
            j = min(max(j, 0), len(st['p']) - 2)
            f = (u - st['tt'][j]) / max(st['tt'][j + 1] - st['tt'][j], 1e-6)
            f = min(max(f, 0), 1)
            if t >= sg['t1'] and t < sg['t1'] + 1e-9:
                f = 1
            return st['p'][j] * (1 - f) + st['p'][j + 1] * f, True, 0.0, sg['s'], st['pig']
        if t >= sg['t1']:        # gap after a phase: hold last position
            return sg.get('b', np.zeros(2)), False, 0.0, None, None
        u = (t - sg['t0']) / (sg['t1'] - sg['t0'])
        e = u * u * (3 - 2 * u)
        a, b = sg['a'], sg['b']
        d = b - a
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        pos = a + d * e + nrm * math.sin(math.pi * u) * 0.06 * np.linalg.norm(d)
        return pos, False, math.sin(math.pi * u) * sg['amp'], None, None

    def _spring(self, tips):
        w = np.zeros_like(tips)
        vel = np.zeros(2)
        om = 2 * math.pi * 0.85
        dt = 1 / self.hz
        ph_starts = {int(ph['t0'] * self.hz) for ph in self.phases}
        cur = tips[0].copy()
        for i, p in enumerate(tips):
            if i in ph_starts:
                cur, vel = p.copy(), np.zeros(2)
            acc = om * om * (p - cur) - 2 * om * vel
            vel += acc * dt
            cur = cur + vel * dt
            w[i] = cur
        return w

    def swivel(self, t):
        """wrist swivel angle (rad): the hand lags behind fast tip moves and turns about the wrist."""
        i = min(int(t * self.hz), len(self.wt) - 1)
        lag = self.tips[i] - self.wrist[i]                 # tip ahead of the (lagging) hand
        ax_ = HAND['axis']
        base = np.array([ax_[0] * math.cos(PHI0) - ax_[1] * math.sin(PHI0), ax_[0] * math.sin(PHI0) + ax_[1] * math.cos(PHI0)])
        cross = base[0] * lag[1] - base[1] * lag[0]
        a = math.atan2(cross, 120.0)
        slow = 0.018 * math.sin(t * 0.7) + 0.012 * math.sin(t * 1.9 + 1.0)
        return max(-0.17, min(0.17, a * 0.9 + slow))


# ----------------------------------------------------------------------------- hand
HAND = json.load(open(ASSET('hand.json')))
HAND_IMG = {}


def hand_img(pig):
    nm = {'g': 'hand_graphite', 'r': 'hand_red', 'b': 'hand_brown'}[pig]
    if nm not in HAND_IMG:
        a = cv2.imread(ASSET(nm + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        HAND_IMG[nm] = a
    return HAND_IMG[nm]


HSU = 1.12                     # hand units per source px
PHI0 = math.radians(20)        # base hand turn: steeper pencil, as when drawing from above


def hand_matrix(tip_s, scale, phi):
    """2x3 affine: hand-crop px -> screen px. Pencil tip lands on tip_s."""
    c, s = math.cos(phi + PHI0), math.sin(phi + PHI0)
    A = np.array([[c, -s], [s, c]]) * scale
    t = np.asarray(tip_s) - A @ np.array(HAND['tip'])
    return np.hstack([A, t[:, None]])


def hand_edge_points():
    a = hand_img('g')[..., 3]
    H_, W_ = a.shape
    cx, cy = HAND['cut']                     # the reference frame's right / bottom border (crop px)
    pts = []
    for y in range(0, H_, 4):
        for x in range(0, W_, 4):
            if a[y, x] > 0.05 and (x > cx + 30 or y > cy + 30):
                pts.append((x, y))
    return np.array(pts, float)


EDGE_PTS = None


# ----------------------------------------------------------------------------- shots / camera
CAPS = {'simple-outline': 4.0, 'simple-colour': 2.6, 'simple-ink': 2.0, 'normal-outline': 3.6, 'normal-fill': 3.2,
        'better-sketch': 4.2, 'better-ink': 2.2, 'real-sketch': 3.0, 'real-red': 3.0, 'real-shade': 2.8, 'real-ink': 3.4}


def shot_plan(S):
    """list of shots: (sim t0, sim t1, cam, hand?)  -- one per phase, cut on the last stroke."""
    by = {ph['name']: ph for ph in S.phases}
    Z = dict(simple=1.95, normal=1.9, better=1.85, real=1.8)
    plan = []
    variants = [(0, 0, 1.0), (-30, -20, 1.07), (25, 15, 0.95)]
    for i, ph in enumerate(S.phases):
        v = variants[i % 3]
        t0, t1 = ph['t0'], ph['t1']
        cap = CAPS[ph['name']]
        z = Z[ph['ant']] * v[2]
        if t1 - t0 > cap:
            a = cap * 0.55
            b = cap - a
            plan.append(dict(t0=t0, t1=t0 + a, ant=ph['ant'], z=z, off=(v[0], v[1]), hand=True, ph=ph['name']))
            v2 = variants[(i + 1) % 3]
            plan.append(dict(t0=t1 - b, t1=t1, ant=ph['ant'], z=Z[ph['ant']] * v2[2], off=(v2[0] + 10, v2[1] - 10), hand=True, ph=ph['name'] + '/end'))
        else:
            plan.append(dict(t0=t0, t1=t1, ant=ph['ant'], z=z, off=(v[0], v[1]), hand=True, ph=ph['name']))
    plan.append(dict(t0=S.T, t1=S.T + 2.8, ant=None, z=1.0, off=(0, 0), hand=False, ph='wide-hold', cam=(540.0, 960.0)))
    return plan


def ant_bbox(S, ant):
    pts = np.vstack([st['p'] for ph in S.phases if ph['ant'] == ant for st in ph['strokes']])
    return pts.min(0), pts.max(0)


def solve_cam(S, sh):
    """place the ant low in frame; nudge until the hand's cut edges never enter the frame."""
    global EDGE_PTS
    if EDGE_PTS is None:
        EDGE_PTS = hand_edge_points()
    if sh.get('cam') is not None:
        return sh['cam'], 0
    lo, hi = ant_bbox(S, sh['ant'])
    ctr = (lo + hi) / 2
    z = sh['z']
    ts = np.arange(sh['t0'], sh['t1'], 1 / 10)
    tips = [S.tip(t) for t in ts]
    best = None
    for dy in range(-40, 700, 20):
        for dx in range(-140, 220, 20):
            anchor = np.array([540 + sh['off'][0] + dx, 1240 + sh['off'][1] + dy])
            cam = ctr - (anchor - [540, 960]) / z
            # ant must stay inside the frame (with the IG caption zone allowed at the bottom)
            l_s = (lo - cam) * z + [540, 960]
            h_s = (hi - cam) * z + [540, 960]
            if l_s[0] < 12 or h_s[0] > OW - 12 or l_s[1] < 140 or h_s[1] > OH - 40:
                continue
            bad = 0
            for t, (p, c, h, si, pig) in zip(ts, tips):
                ps = (p - cam) * z + [540, 960]
                M = hand_matrix(ps, HSU * z, S.swivel(t))
                e = EDGE_PTS @ M[:, :2].T + M[:, 2]
                on = (e[:, 0] > -5) & (e[:, 0] < OW + 5) & (e[:, 1] > -5) & (e[:, 1] < OH + 5)
                bad += int(on.sum())
            score = bad * 1000 + abs(dy) + abs(dx) * 0.5
            if best is None or score < best[0]:
                best = (score, cam, bad)
    if best is None:
        return ctr, -1
    return best[1], best[2]


# ----------------------------------------------------------------------------- paper + ink
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
                fine += amp * n                                   # paper tooth that catches the pencil
        h /= h.std()
        fine /= fine.std()
        gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
        sh = 1.0 - 0.036 * (gx * 0.55 + gy * 0.8)
        yy = np.linspace(0, 1, WH, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, WW, dtype=np.float32)[None, :]
        big = cv2.resize(r.standard_normal((12, 7)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
        light = 1.04 - 0.07 * yy + 0.03 * xx + 0.012 * big
        self.light = np.clip(sh * light, 0.6, 1.3).astype(np.float32)
        self.hn = np.clip(fine * 0.2 + 0.5 + 0.03 * h, 0, 1).astype(np.float32)   # grain height 0..1
        del fine
        del gx, gy, sh, h
        self.D = {k: np.zeros((WH, WW), np.float32) for k in 'grb'}
        self.state = {}                                                        # stroke -> (bbox, D0, S, done_idx)
        lab = cv2.imread(ASSET('labels.png'), cv2.IMREAD_UNCHANGED)
        self.labels = cv2.cvtColor(lab, cv2.COLOR_BGRA2RGBA) if lab is not None else None

    def advance(self, S, t_from, t_to):
        """deposit every stroke portion drawn in (t_from, t_to]"""
        for si, st in enumerate(S.strokes):
            if st['t1'] <= t_from or st['t0'] > t_to:
                continue
            u = min(t_to, st['t1']) - st['t0']
            j = int(np.searchsorted(st['tt'], u, side='right')) - 1
            j = min(j, len(st['p']) - 1)
            pts = st['p'][:j + 1]
            if j < len(st['p']) - 1:
                f = (u - st['tt'][j]) / max(st['tt'][j + 1] - st['tt'][j], 1e-6)
                pts = np.vstack([pts, st['p'][j] * (1 - f) + st['p'][j + 1] * f])
            self._deposit(si, st, pts)

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
        P = to_w(pts_u)
        N = len(st['p'])
        # draw the new part in short runs so width can follow the stroke (taper + pressure)
        idx = np.arange(n0, len(P))
        run = 6
        for a in range(0, len(idx) - 1, run):
            b = min(a + run, len(idx) - 1)
            seg = P[idx[a]:idx[b] + 1]
            fr = idx[b] / max(N - 1, 1)
            ends = min(fr, 1 - fr) * 10 if st['kind'] == 'line' else 1
            press = 0.72 + 0.28 * min(ends, 1)
            wd = (w0 + (w1 - w0) * fr) * WS * (0.8 + 0.2 * press)
            col = int(255 * min(1.0, press))
            q = np.round((seg - [x0, y0]) * 16).astype(np.int32)
            cv2.polylines(S_['S'], [q], False, col, thickness=max(1, int(round(wd))), lineType=cv2.LINE_AA, shift=4)
        S_['n'] = len(pts_u) - 1 if len(pts_u) < N else N
        # re-derive density over the touched box
        P2 = P[n0:]
        bx0 = int(max(P2[:, 0].min() - wmax - 3 - x0, 0)); by0 = int(max(P2[:, 1].min() - wmax - 3 - y0, 0))
        bx1 = int(min(P2[:, 0].max() + wmax + 3 - x0, x1 - x0)); by1 = int(min(P2[:, 1].max() + wmax + 3 - y0, y1 - y0))
        if bx1 <= bx0 or by1 <= by0:
            return
        cov = S_['S'][by0:by1, bx0:bx1].astype(np.float32) / 255
        hn = self.hn[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]
        k = st['k']
        thr = 0.66 - 0.45 * k * cov                      # harder press reaches into the paper valleys
        grain = np.clip((hn - thr) / 0.30 + 0.34, 0.26, 1.0)
        dep = cov * grain * k * (1.25 if st['pig'] == 'g' else 1.6)
        self.D[st['pig']][y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = S_['D0'][by0:by1, bx0:bx1] + dep

    def view(self, M):
        """composite paper + pigments in screen space. M: 2x3 world px -> screen px"""
        f = dict(dsize=(OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        light = cv2.warpAffine(self.light, M, **f)
        albedo = np.array([0.655, 0.630, 0.596], np.float32)
        col = np.empty((OH, OW, 3), np.float32)
        col[:] = albedo
        Dr = cv2.warpAffine(self.D['r'], M, **f)
        if Dr.max() > 0:
            cr = (1 - np.exp(-1.7 * Dr))[..., None] * 0.93
            col = col * (1 - cr) + np.array([0.60, 0.07, 0.09], np.float32) * cr
        Db = cv2.warpAffine(self.D['b'], M, **f)
        if Db.max() > 0:
            cb = (1 - np.exp(-1.5 * Db))[..., None] * 0.9
            col = col * (1 - cb) + np.array([0.22, 0.045, 0.035], np.float32) * cb
        Dg = cv2.warpAffine(self.D['g'], M, **f)
        if Dg.max() > 0:
            cg = (1 - np.exp(-1.9 * Dg))[..., None]
            col = col * (1 - cg * 0.86)
        col *= light[..., None]
        return col

    def labels_view(self, M, col):
        if self.labels is None:
            return col
        lab = cv2.warpAffine(self.labels, M, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)).astype(np.float32) / 255
        a = lab[..., 3:]
        return col * (1 - a) + lab[..., :3] * a


# ----------------------------------------------------------------------------- frame
def sway(t, seed):
    r = np.random.default_rng(seed)
    p = r.uniform(0, 6, 6)
    dx = 1.3 * math.sin(t * 0.9 + p[0]) + 0.6 * math.sin(t * 2.3 + p[1])
    dy = 1.1 * math.sin(t * 0.7 + p[2]) + 0.5 * math.sin(t * 2.9 + p[3])
    rot = 0.0012 * math.sin(t * 0.5 + p[4])
    return dx, dy, rot


def world_to_screen_M(cam, z, sw):
    dx, dy, rot = sw
    c, s = math.cos(rot), math.sin(rot)
    a = z / WS
    A = np.array([[c, -s], [s, c]]) * a
    # screen = A @ (wp) + t ;  wp of cam centre -> (540+dx, 960+dy)
    cw = to_w(cam)
    t = np.array([540 + dx, 960 + dy]) - A @ cw
    return np.hstack([A, t[:, None]]).astype(np.float32)


def comp_hand(col, S, t, M_ws, z, pig):
    p, contact, lift, si, _ = S.tip(t)
    # motion blur: average a few sub-positions inside a 1/60 s shutter when moving fast
    p_prev = S.tip(max(t - 1 / 60, 0))[0]
    speed = np.linalg.norm(p - p_prev) * 60 * z
    nsub = 1 if speed < 500 else (3 if speed < 1500 else 5)
    img = hand_img(pig)
    acc_rgb = np.zeros((OH, OW, 3), np.float32)
    acc_a = np.zeros((OH, OW), np.float32)
    shadow = None
    for k in range(nsub):
        tt = t - k * (1 / 60) / max(nsub - 1, 1) if nsub > 1 else t
        pk, ck, lk, _, _ = S.tip(max(tt, 0))
        ps = M_ws[:, :2] @ to_w(pk) + M_ws[:, 2]
        sc = HSU * z * (1 + 0.03 * lk)
        M = hand_matrix(ps, sc, S.swivel(tt))
        w = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        acc_rgb += w[..., :3] * w[..., 3:]
        acc_a += w[..., 3]
        if k == 0:
            # soft shadow: hand silhouette scaled away from the tip (light from the upper left)
            q = 4
            Ms = M.copy()
            g = 1.05 + 0.07 * lk
            Ms[:, :2] *= g
            Ms[:, 2] = ps + g * (M[:, 2] - ps) + np.array([10, 13]) * lk * z
            Ms = Ms / q
            a = cv2.warpAffine(img[..., 3], Ms, (OW // q, OH // q), flags=cv2.INTER_LINEAR)
            a = cv2.GaussianBlur(a, (0, 0), 2.2 + 3 * lk)
            shadow = cv2.resize(a, (OW, OH), interpolation=cv2.INTER_LINEAR)
    acc_rgb /= nsub
    acc_a /= nsub
    col = col * (1 - 0.30 * shadow[..., None])
    return col * (1 - acc_a[..., None]) + acc_rgb


def grade(col, t):
    # gentle vignette + tiny sensor noise (fixed seed per frame)
    if not hasattr(grade, 'vig'):
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = ((xx - OW / 2) / OW) ** 2 + ((yy - OH * 0.45) / OH) ** 2
        grade.vig = (1 - 0.22 * r)[..., None].astype(np.float32)
    col = col * grade.vig
    n = np.random.default_rng(int(t * 1000)).standard_normal((OH // 2, OW // 2)).astype(np.float32)
    n = cv2.resize(n, (OW, OH), interpolation=cv2.INTER_LINEAR)[..., None] * 0.006
    return np.clip(col + n, 0, 1)


class Renderer:
    def __init__(self, S):
        self.S = S
        self.cv = Canvas()
        self.t_ink = -1.0
        self.plan = shot_plan(S)
        for i, sh in enumerate(self.plan):
            sh['cam'], sh['bad'] = solve_cam(S, sh)
            sh['seed'] = i
        t = 0.0
        for sh in self.plan:
            sh['o0'] = t
            t += sh['t1'] - sh['t0']
            sh['o1'] = t
        self.dur = t

    def frame(self, k):
        to = (k + 0.5) / FPS
        sh = next((s for s in self.plan if s['o0'] <= to < s['o1']), self.plan[-1])
        t = sh['t0'] + (to - sh['o0'])
        tin = min(t, self.S.T)
        if tin > self.t_ink:
            self.cv.advance(self.S, self.t_ink, tin)
            self.t_ink = tin
        M = world_to_screen_M(sh['cam'], sh['z'], sway(to, sh['seed']))
        col = self.cv.view(M)
        if sh['hand']:
            ph = next(p for p in self.S.phases if p['t0'] - 1e-6 <= t <= p['t1'] + 0.7)
            col = comp_hand(col, self.S, t, M, sh['z'], ph['pig'])
        col = self.cv.labels_view(M, col)
        col = grade(col, to)
        return (col * 255 + 0.5).astype(np.uint8), sh, t

    def audio_track(self):
        """per-frame pencil contact + speed -> synthesised pencil-on-paper sound"""
        sr = 48000
        n = int(self.dur * sr) + sr
        env = np.zeros(int(self.dur * FPS) + 2)
        kind = np.zeros_like(env)
        for k in range(len(env)):
            to = k / FPS
            sh = next((s for s in self.plan if s['o0'] <= to < s['o1']), None)
            if sh is None or not sh['hand']:
                continue
            t = sh['t0'] + (to - sh['o0'])
            p, c, _, si, pig = self.S.tip(t)
            if not c:
                continue
            p2 = self.S.tip(max(t - 1 / 120, 0))[0]
            v = np.linalg.norm(p - p2) * 120
            env[k] = min(1.0, (v / 650) ** 0.7) * (0.9 if pig == 'g' else 0.65)
            kind[k] = 0 if pig == 'g' else 1
        r = np.random.default_rng(3)
        x = np.arange(n) / sr
        e = np.interp(x, np.arange(len(env)) / FPS, env)
        kd = np.interp(x, np.arange(len(env)) / FPS, kind)
        e = np.convolve(e, np.ones(240) / 240, mode='same')
        noise = r.standard_normal(n)
        F = np.fft.rfftfreq(n, 1 / sr)
        N = np.fft.rfft(noise)

        def band(lo, hi, peak):
            g = np.exp(-0.5 * ((np.log(F + 1) - np.log(peak)) / 0.55) ** 2)
            g *= (F > lo) & (F < hi)
            return np.fft.irfft(N * g, n)
        gr = band(900, 11000, 4200)
        wx = band(300, 6000, 1800)
        gr /= np.abs(gr).max(); wx /= np.abs(wx).max()
        gx_ = np.arange(int(n / sr * 60) + 2) / 60
        grain = np.interp(x, gx_, 1 + 0.5 * r.standard_normal(len(gx_)))
        sig = (gr * (1 - kd) + wx * kd) * e * np.clip(grain, 0.3, 2.0)
        room = np.fft.irfft(np.fft.rfft(r.standard_normal(n)) / np.sqrt(F + 20), n)
        room /= np.abs(room).max()
        sig = sig * 0.55 + room * 0.004
        sig /= max(np.abs(sig).max(), 1e-6) / 0.6
        return (sig[:int(self.dur * sr)] * 32767).astype(np.int16), sr


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--still', default='')
    ap.add_argument('--render', default='')
    ap.add_argument('--chunk', default='')
    ap.add_argument('--audio', default='')
    a = ap.parse_args()
    S = Session()
    R = Renderer(S)
    if a.plan:
        for ph in S.phases:
            print(f"{ph['name']:16s} sim {ph['t0']:6.2f}-{ph['t1']:6.2f}  ({ph['t1'] - ph['t0']:5.2f}s, {len(ph['strokes'])} strokes)")
        for sh in R.plan:
            print(f"shot {sh['ph']:20s} out {sh['o0']:6.2f}-{sh['o1']:6.2f} z={sh['z']:.2f} cam=({sh['cam'][0]:.0f},{sh['cam'][1]:.0f}) hand-edge-violations={sh['bad']}")
        print(f'total {R.dur:.2f}s  ({int(R.dur * FPS)} frames)')
    if a.still:
        os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
        for s in a.still.split(','):
            k = int(float(s) * FPS)
            img, sh, t = R.frame(k)
            cv2.imwrite(os.path.join(HERE, 'out', f'still_{float(s):06.2f}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
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
        import time
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
