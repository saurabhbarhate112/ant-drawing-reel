"""Render the four level labels into a world-sized RGBA overlay (assets/labels.png)."""
import os, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import butterfly as reel

font = ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf', int(60 * reel.WS))
img = Image.new('RGBA', (reel.WW, reel.WH), (0, 0, 0, 0))
shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shadow)
for k, txt in reel.LABEL_TEXT.items():
    x, y = reel.to_w(reel.LABELS[k])
    bb = d.textbbox((0, 0), txt, font=font)
    tx, ty = x - (bb[2] - bb[0]) / 2 - bb[0], y - (bb[3] - bb[1]) / 2 - bb[1]
    ds.text((tx + 5, ty + 8), txt, font=font, fill=(20, 14, 10, 210), stroke_width=5, stroke_fill=(20, 14, 10, 210))
    d.text((tx, ty), txt, font=font, fill=(252, 234, 204, 255), stroke_width=3, stroke_fill=(38, 26, 18, 255))
shadow = shadow.filter(ImageFilter.GaussianBlur(9))
out = Image.alpha_composite(shadow, img)
out.save(os.path.join(reel.HERE, 'assets', 'labels2.png'), optimize=True)
print('labels', out.size)
