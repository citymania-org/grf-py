import sys
import math
from collections import defaultdict
from pprint import pprint

from PIL import Image
import numpy as np
import spectra

import grf

def mtorgb(c):
    return grf.PALETTE[c]

to_spectra = lambda r, g, b: spectra.rgb(float(r) / 255., float(g) / 255., float(b) / 255.)

def mtospectra(c):
    return to_spectra(*mtorgb(c))

def spectratorgb(c):
    r, g, b = c.clamped_rgb
    return int(255 * r + .5), int(255 * g + .5), int(255 * b + .5)

def blend(a, b, ratio):
    return spectratorgb(a.blend(b, ratio=r))


r = [0] * 8
for i, cl in enumerate(grf.CC_COLOURS):
    for j, c in enumerate(cl):
        rgb = mtorgb(c)
        r[j] += max(rgb) + min(rgb)
value_mean = [x / len(grf.CC_COLOURS) for x in r]
print(value_mean)
print([(value_mean[i] + value_mean[i-1]) / 2 for i in range(1, len(value_mean))])

# VALUE_BOUNDARIES = [111, 141, 173, 208, 244, 285, 333, 382, 430]
VALUE_BOUNDARIES = [0, 110, 160, 210, 260, 310, 360, 410, 511]
VALUE_IDX = {}
for i in range(1, len(VALUE_BOUNDARIES)):
    for j in range(VALUE_BOUNDARIES[i - 1], VALUE_BOUNDARIES[i]):
        VALUE_IDX[j] = i - 1


def adjust_brightness(c, brightness):
    if brightness == 128:
        return c

    r, g, b = c
    combined = (r << 32) | (g << 16) | b
    combined *= brightness

    r = (combined >> 39) & 0x1ff
    g = (combined >> 23) & 0x1ff
    b = (combined >> 7) & 0x1ff

    if (combined & 0x800080008000) == 0:
        return (r, g, b)

    ob = 0
    # Sum overbright
    if r > 255: ob += r - 255
    if g > 255: ob += g - 255
    if b > 255: ob += b - 255

    # Reduce overbright strength
    ob //= 2
    return (
        255 if r >= 255 else min(r + ob * (255 - r) // 256, 255),
        255 if g >= 255 else min(g + ob * (255 - g) // 256, 255),
        255 if b >= 255 else min(b + ob * (255 - b) // 256, 255),
    )

target = []

BLACK = to_spectra(0, 0, 0)
WHITE = to_spectra(255, 255, 255)
for cl in grf.CC_COLOURS:
    res = []
    l, c, h = mtospectra(cl[4]).to('lch').values
    for v in range(511):
        res.append(spectratorgb(spectra.lch(l, c * v / 510, h)))
    target.append(res)


def l(c):
    return max(c) + min(c)


def magenta(v):
    v = int(v + .5)
    if v <= 255: return (v, 0, v)
    else: return (255, v - 255, 255)


def color_distance(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    rmean = r1 + r2
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    return (1020 + rmean) * dr * dr + 2040 * dg * dg + (1530 - rmean) * db * db


npimg = np.zeros((100 + 50 * len(grf.CC_COLOURS), 512 * 3, 3), dtype=np.uint8)

vtb = []
vti = []
for v in range(511):
    bi, bj, bd = 0, 0, 100000000000000

    s, e = 0, 8
    for i in range(s, e):
        for j in range(256):
            d = 0
            for cl, ct in zip(grf.CC_COLOURS, target):
                rgb = adjust_brightness(mtorgb(cl[i]), j)
                # d += abs(max(rgb) + min(rgb) - max(ct[v]) - min(ct[v]))
                d += color_distance(rgb, ct[v])
            if d < bd:
                bi, bj, bd = i, j, d

    vtb.append(bj)
    vti.append(bi)
    npimg[50:99, v] = (bj, bj, bj)
    print(end='.', flush=True)
print()

# for v in range(511):
#     npimg[50:99, v + 512] = magenta(v)

y = 100
for i, cl in enumerate(grf.CC_COLOURS):
    for v in range(511):
        rgb = mtorgb(cl[vti[v]])
        npimg[y: y + 49, v] = rgb
        npimg[y: y + 49, v + 512 * 2] = adjust_brightness(rgb, vtb[v])
        npimg[y: y + 49, v + 512] = target[i][v]
    y += 50

img = Image.fromarray(npimg)
img.save("magenta.png")

for i in range(8):
    print(VALUE_BOUNDARIES[i + 1] - VALUE_BOUNDARIES[i])

print(vtb)
print(vti)

# CC = grf.CC_COLOURS[0]
# colors = defaultdict(list)
# for c in CC:
#   rgb = grf.PALETTE[c * 3: c * 3 + 3]
#   for i in range(256):
#       ab = adjust_brightness(rgb, i)
#       colors[l(ab)].append((c, i))


# def distance(c1, c2):
#   r1, g1, b1 = c1
#   r2, g2, b2 = c2
#   r = r1 - r2;
#   g = g1 - g2;
#   b = b1 - b2;

#   # if r1 + r2 < 256:
#   #   return ((2 * r * r) + (4 * g * g) + (3 * b * b))
#   return ((3 * r * r) + (4 * g * g) + (2 * b * b))


# def pdistance(c1, c2):
#   d = distance(c1, c2)
#   print(f'{c1} - {c2} : {d}')

# # pdistance((128, 0, 0), (129, 0, 127))
# pdistance((128, 0, 0), (129, 0, 255))
# pdistance((128, 0, 0), (127, 0, 209))