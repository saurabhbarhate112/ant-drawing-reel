"""Cut the pencil hand out of the reference reel (frame 523 = 17.4 s) by
difference-keying against a hand-free frame (301 = 10.0 s), then pale the skin.

Writes assets/hand_graphite.png, hand_red.png, hand_brown.png (RGBA) and
assets/hand.json (tip point, pencil axis, cut edges) in source pixels.
"""
import json, sys, cv2, numpy as np

SRC, BG = 'work/allf/0523.png', 'work/allf/0301.png'
TIP = np.array([253.8, 1014.3])          # graphite point, measured by eye on a 4x zoom
WOOD = np.array([312.0, 1041.0])          # graphite -> wood boundary on the axis
BODY = np.array([400.0, 1093.0])          # a point on the painted body centre line
EXT = 90                                  # px of smeared continuation past the cut edges

im = cv2.imread(SRC)
bg = cv2.imread(BG)
H, W = im.shape[:2]
f = lambda a: cv2.GaussianBlur(a.astype(np.float32), (3, 3), 0)
d = np.abs(f(im) - f(bg)).max(-1)

ax = BODY - TIP; ax /= np.linalg.norm(ax)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
proj = (xx - TIP[0]) * ax[0] + (yy - TIP[1]) * ax[1]
perp = -(xx - TIP[0]) * ax[1] + (yy - TIP[1]) * ax[0]

m = ((d > 26) & (proj > -0.5)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
best = max(range(1, n), key=lambda i: st[i, 4])
m = (lab == best).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
# fill holes
ff = m.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 1)
m = m | (1 - ff)
m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   # drop the paper halo round the soft finger edge
# the graphite cone: keep it a clean sharp triangle (the key matte rounds the point)
cone_len = float(np.dot(WOOD - TIP, ax))
cone = (proj >= 0) & (proj <= cone_len + 2) & (np.abs(perp) <= 1.2 + proj * 0.30)
m[(proj <= cone_len + 2)] = 0
m[cone] = 1
soft = np.clip((d - 18) / 40.0, 0, 1)
soft[cone] = 1
soft = np.maximum(soft, cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(np.float32))
alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.9) * soft
alpha = np.clip((alpha - 0.08) / 0.84, 0, 1)

# ---- pale the skin (not the pencil): lift L, lower chroma
labc = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
skin = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 35)
pencil_zone = (np.abs(perp) < 26) & (proj < 330)
skin = (skin & ~pencil_zone).astype(np.float32)
skin = cv2.GaussianBlur(skin, (0, 0), 3) * alpha
L, A, B = labc[..., 0], labc[..., 1] - 128, labc[..., 2] - 128
lw = np.clip((L - 60) / 70.0, 0, 1)            # lift only lit skin, keep shadows deep
Lp = L + ((255 - L) * 0.15 + 5) * lw
cp = 0.80
skin = skin * np.clip((L - 35) / 50.0, 0, 1)
labp = np.stack([L + (Lp - L) * skin, 128 + A * (1 - (1 - cp) * skin), 128 + B * (1 - (1 - cp) * skin)], -1)
pale = cv2.cvtColor(np.clip(labp, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

# ---- extend past the right / bottom cut edges with a smeared, defocused continuation
P = EXT
# 1) take the pencil out of the strip near the cut edges (it is put back as a straight band below)
PC = 7.0                                  # pencil centre line is this far below the tip->body axis
pen_far = (np.abs(perp - PC) < 34) & (proj > 300)
edge = (xx > W - 40) | (yy > H - 40)
noskin = pale.copy()
noskin = cv2.inpaint(noskin, ((pen_far & edge) * 255).astype(np.uint8), 12, cv2.INPAINT_TELEA)
src = np.where((pen_far & edge)[..., None], noskin, pale)
# 2) skin continues past the edges, increasingly defocused and darker (out of focus near the lens)
big = cv2.copyMakeBorder(src, 0, P, 0, P, cv2.BORDER_REPLICATE).astype(np.float32)
abig = cv2.copyMakeBorder(alpha, 0, P, 0, P, cv2.BORDER_REPLICATE)
yy2, xx2 = np.mgrid[0:H + P, 0:W + P].astype(np.float32)
outside = np.maximum(xx2 - (W - 1), yy2 - (H - 1)).clip(0, None)
# 3) the pencil carries on straight under the hand: a dark band along its axis
proj2 = (xx2 - TIP[0]) * ax[0] + (yy2 - TIP[1]) * ax[1]
perp2 = -(xx2 - TIP[0]) * ax[1] + (yy2 - TIP[1]) * ax[0]
band = (np.abs(perp2 - PC) < 27) & (proj2 > 300) & (outside > 0)
band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 3)
shade_band = 0.75 + 0.5 * np.clip(1 - np.abs(perp2 - PC + 8) / 22, 0, 1)    # soft highlight on the top facet
navy = np.array([62, 44, 32], np.float32)
big = big * (1 - band[..., None]) + (navy * shade_band[..., None]) * band[..., None]
BAND = band
b1 = cv2.GaussianBlur(big, (0, 0), 7)
b2 = cv2.GaussianBlur(big, (0, 0), 26)
k1 = np.clip(outside / 25.0, 0, 1)[..., None]
k2 = np.clip((outside - 15) / 70.0, 0, 1)[..., None]
big = big * (1 - k1) + (b1 * (1 - k2) + b2 * k2) * k1
big = big * (1 - 0.22 * np.clip(outside / P, 0, 1))[..., None]   # falls off into shade
a1 = cv2.GaussianBlur(abig, (0, 0), 6)
a2 = cv2.GaussianBlur(abig, (0, 0), 22)
abig = abig * (1 - k1[..., 0]) + (a1 * (1 - k2[..., 0]) + a2 * k2[..., 0]) * k1[..., 0]
abig = np.where(outside > 0, np.clip((abig - 0.15) / 0.7, 0, 1), abig)
big = np.clip(big, 0, 255).astype(np.uint8)

def save(name, rgb):
    rgba = np.dstack([rgb, (abig * 255).astype(np.uint8)])
    ys, xs = np.nonzero(abig > 0.01)
    x0, y0 = xs.min(), ys.min()
    cv2.imwrite(f'assets/{name}.png', rgba[y0:, x0:])
    return int(x0), int(y0)

x0, y0 = save('hand_graphite', big)

# recoloured pencils: map the luminance of the navy body + graphite point onto a new colour ramp
hsvb = cv2.cvtColor(big, cv2.COLOR_BGR2HSV).astype(np.float32)
projb = np.pad(proj, ((0, P), (0, P)), mode='edge')
perpb = np.pad(perp, ((0, P), (0, P)), mode='edge')
skinb = np.pad(skin, ((0, P), (0, P)), mode='edge')
hb, sb, vb = hsvb[..., 0], hsvb[..., 1], hsvb[..., 2]
wood = (hb < 28) & (sb > 25) & (vb > 120)
skinpx = ((hb < 25) | (hb > 160)) & (sb > 45) & (vb > 70)
body = ((perpb > -23) & (perpb < 38) & (projb > -1) & (projb < 330) & ~wood & ~skinpx & (skinb < 0.1)) | (BAND > 0.5)
body = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 0.8)[..., None]
lum = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255.0
for name, dark, light in [('hand_red', (25, 20, 95), (150, 150, 235)), ('hand_brown', (18, 30, 55), (120, 150, 190))]:
    dark, light = np.array(dark, np.float32), np.array(light, np.float32)
    t = np.clip((lum - 0.08) / 0.55, 0, 1)
    col = dark + (light - dark) * t
    out = big * (1 - body) + col * body
    save(name, np.clip(out, 0, 255).astype(np.uint8))

meta = dict(tip=[float(TIP[0] - x0), float(TIP[1] - y0)], cut=[float(W - 1 - x0), float(H - 1 - y0)], axis=[float(ax[0]), float(ax[1])],
            src_frame=523, note='pixels relative to the cropped PNG; pencil points from tip along -axis')
json.dump(meta, open('assets/hand.json', 'w'), indent=1)
print(meta)

# preview: before/after on paper colour
prev = []
for nm in ['hand_graphite', 'hand_red', 'hand_brown']:
    a = cv2.imread(f'assets/{nm}.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    bgc = np.full(a.shape[:2] + (3,), (150, 158, 165), np.float32)
    al = a[..., 3:] / 255
    prev.append((a[..., :3] * al + bgc * (1 - al)).astype(np.uint8))
orig = im[y0:, x0:]
orig = cv2.copyMakeBorder(orig, 0, prev[0].shape[0] - orig.shape[0], 0, prev[0].shape[1] - orig.shape[1], cv2.BORDER_CONSTANT)
cv2.imwrite('work/hand_preview.jpg', np.hstack([orig] + prev))
