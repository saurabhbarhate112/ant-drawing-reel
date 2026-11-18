"""Horizontal right-edge hand pose (reference frame 1, t=0): colour-key against the grey
paper, pale the skin, extend the forearm off the right edge, make coloured-pencil variants.

Writes assets/hand2_<name>.png (RGBA) and assets/hand2.json.
"""
import json, cv2, numpy as np

SRC = 'work/allf/0001.png'
TIP = np.array([185.8, 386.7])            # graphite point (full-frame px), checked on a 4x zoom
BODY = np.array([560.0, 380.0])           # a point on the pencil centre line under the fingers
EXT = 300                                 # px of forearm continued past the right edge

im = cv2.imread(SRC)
H, W = im.shape[:2]
lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)

# key: skin is colourful, pencil/shadow is darker than the local paper, paper is neither
Cr = np.sqrt((lab[..., 1] - 128) ** 2 + (lab[..., 2] - 128) ** 2)
Lc = lab[..., 0]
cand = (Cr < 9) & (Lc > 100)
yy0, xx0 = np.nonzero(cand[::4, ::4]); yy0 *= 4; xx0 *= 4
X = np.stack([np.ones_like(xx0), xx0, yy0, xx0 * xx0, yy0 * yy0, xx0 * yy0], 1).astype(np.float64)
coef = np.linalg.lstsq(X, Lc[yy0, xx0].astype(np.float64), rcond=None)[0]
gy, gx = np.mgrid[0:H, 0:W].astype(np.float64)
Lfit = (coef[0] + coef[1] * gx + coef[2] * gy + coef[3] * gx * gx + coef[4] * gy * gy + coef[5] * gx * gy).astype(np.float32)
z = np.maximum((Cr - 9) / 1.6, (Lfit - Lc - 22) / 4.0)
z = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.2)

ax = BODY - TIP; ax /= np.linalg.norm(ax)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
proj = (xx - TIP[0]) * ax[0] + (yy - TIP[1]) * ax[1]
perp = -(xx - TIP[0]) * ax[1] + (yy - TIP[1]) * ax[0]

m = ((z > 4.5) & (proj > -0.5)).astype(np.uint8)
# the 'Realistic' label sits under the knuckles: drop its cream letters + dark outline
txt = ((Lc > Lfit + 22) & (lab[..., 2] > 140)).astype(np.uint8)
txt = cv2.dilate(txt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
box = np.zeros_like(m); box[600:690, 370:610] = 1
m[(txt > 0) & (box > 0)] = 0
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
n, lb, st, _ = cv2.connectedComponentsWithStats(m)
touch = [i for i in range(1, n) if st[i, 0] + st[i, 2] >= W - 1]
best = max(touch, key=lambda i: st[i, 4])
m = (lb == best).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
ff = m.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
m = m | (1 - ff)
# the pencil as a solid shape (wood and highlights are as pale as the paper, so key them geometrically)
pc = 2 + 6 * np.clip((proj - 45) / 100, 0, 1)
hw = np.where(proj <= 61, 0.9 + proj * 0.27, np.clip(17.4 + (proj - 61) * 0.22, 0, 28.5))
cone = (proj >= 0) & (proj <= 450) & (np.abs(perp - pc) <= hw)
m[(proj <= 120) & ~cone] = 0
m[cone] = 1
soft = np.clip((z - 2.5) / 5.0, 0, 1)
soft[cv2.erode(cone.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 1
core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(np.float32)
alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.9) * np.maximum(soft, core)

# pale skin (lit skin only, shadows stay deep)
L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
skin = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 35)
pencil_zone = (np.abs(perp) < 34) & (proj < 380)
skin = (skin & ~pencil_zone).astype(np.float32)
skin = cv2.GaussianBlur(skin, (0, 0), 3) * alpha * np.clip((L - 35) / 50.0, 0, 1)
lw = np.clip((L - 60) / 70.0, 0, 1)
Lp = L + ((255 - L) * 0.15 + 5) * lw
labp = np.stack([L + (Lp - L) * skin, 128 + A * (1 - 0.2 * skin), 128 + B * (1 - 0.2 * skin)], -1)
pale = cv2.cvtColor(np.clip(labp, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

# forearm: continue the right-edge column outward, defocused, tapering toward the wrist centre, darker
P = EXT
big = cv2.copyMakeBorder(pale, 0, 0, 0, P, cv2.BORDER_REPLICATE)
abig = cv2.copyMakeBorder(alpha, 0, 0, 0, P, cv2.BORDER_REPLICATE)
col = alpha[:, W - 1] > 0.5
ys = np.nonzero(col)[0]
top, bot = ys.min(), ys.max()
yy2, xx2 = np.mgrid[0:H, 0:W + P].astype(np.float32)
out = np.clip(xx2 - (W - 1), 0, None)
cy = (top + bot) / 2 + 30
half0 = (bot - top) / 2
half = half0 * (1 - 0.30 * np.clip(out / P, 0, 1))          # forearm narrows to the wrist
cyx = cy + 25 * np.clip(out / P, 0, 1)
taper = np.clip((half - np.abs(yy2 - cyx)) / 40.0, 0, 1)
abig = np.where(out > 0, np.minimum(abig, taper), abig)
b1 = cv2.GaussianBlur(big, (0, 0), 4)
b2 = cv2.GaussianBlur(big, (0, 0), 16)
k1 = np.clip(out / 60.0, 0, 1)[..., None]
k2 = np.clip((out - 20) / 160.0, 0, 1)[..., None]
big = big * (1 - k1) + (b1 * (1 - k2) + b2 * k2) * k1
big = big * (1 - 0.18 * np.clip(out / P, 0, 1) ** 1.5)[..., None]
a1 = cv2.GaussianBlur(abig, (0, 0), 8)
abig = abig * (1 - k1[..., 0]) + a1 * k1[..., 0]
big = np.clip(big, 0, 255).astype(np.uint8)

ys_, xs_ = np.nonzero(abig > 0.01)
x0, y0, y1 = int(xs_.min()), int(ys_.min()), int(ys_.max()) + 1


def save(name, rgb):
    rgba = np.dstack([rgb, (abig * 255).astype(np.uint8)])[y0:y1, x0:]
    cv2.imwrite(f'assets/hand2_{name}.png', rgba)


save('graphite', big)

# coloured pencils: map the luminance of the navy body + graphite point onto a colour ramp
hsvb = cv2.cvtColor(big, cv2.COLOR_BGR2HSV).astype(np.float32)
projb = (xx2 - TIP[0]) * ax[0] + (yy2 - TIP[1]) * ax[1]
perpb = -(xx2 - TIP[0]) * ax[1] + (yy2 - TIP[1]) * ax[0]
hb, sb, vb = hsvb[..., 0], hsvb[..., 1], hsvb[..., 2]
wood = (hb < 28) & (sb > 25) & (vb > 120)
skinpx = ((hb < 25) | (hb > 160)) & (sb > 45) & (vb > 70)
pcb = 2 + 6 * np.clip((projb - 45) / 100, 0, 1)
body = (np.abs(perpb - pcb) < 29) & (projb > -1) & (projb < 400) & ~wood & ~skinpx
body = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 0.8)[..., None]
lum = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255.0
COLS = {  # BGR dark, light
    'orange': ((10, 60, 150), (90, 180, 255)),
    'yellow': ((20, 120, 160), (140, 235, 255)),
    'brown': ((15, 30, 60), (110, 140, 190)),
    'white': ((150, 155, 160), (250, 250, 250)),
    'black': ((18, 18, 20), (95, 95, 100)),
}
for name, (dark, light) in COLS.items():
    dark, light = np.array(dark, np.float32), np.array(light, np.float32)
    t = np.clip((lum - 0.06) / 0.55, 0, 1)
    c = dark + (light - dark) * t
    o = big * (1 - body) + c * body
    save(name, np.clip(o, 0, 255).astype(np.uint8))

meta = dict(tip=[float(TIP[0] - x0), float(TIP[1] - y0)], cut_x=float(W - 1 - x0), axis=[float(ax[0]), float(ax[1])],
            src_frame=1, note='crop px; pencil points from tip along -axis; cut_x = reference frame right border')
json.dump(meta, open('assets/hand2.json', 'w'), indent=1)
print(meta, 'size', big.shape, 'crop', x0, y0, y1)

prev = []
for nm in ['graphite', 'orange', 'white']:
    a = cv2.imread(f'assets/hand2_{nm}.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    al = a[..., 3:] / 255
    prev.append((a[..., :3] * al + np.array((150, 158, 165)) * (1 - al)).astype(np.uint8))
cv2.imwrite('work/hand2_preview.jpg', np.vstack(prev))
