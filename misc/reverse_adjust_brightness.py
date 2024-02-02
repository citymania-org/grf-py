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
        r[j] += (max(rgb) + min(rgb)) / 2.0
        # l, c, h = mtospectra(c).to('lch').values
        # r[j] += l
value_mean = [x / len(grf.CC_COLOURS) for x in r]
print(value_mean)
print([(value_mean[i] + value_mean[i-1]) / 2 for i in range(1, len(value_mean))])

vtb2 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 55, 57, 58, 59, 59, 61, 61, 63, 64, 64, 65, 67, 68, 69, 69, 70, 72, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 96, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 125, 126, 127, 128, 128, 130, 131, 132, 133, 134, 134, 136, 136, 136, 138, 138, 141, 117, 117, 119, 119, 119, 119, 122, 122, 123, 123, 124, 125, 125, 127, 127, 128, 128, 129, 130, 131, 131, 132, 133, 134, 134, 134, 136, 136, 137, 138, 138, 140, 117, 118, 118, 119, 119, 120, 121, 121, 122, 122, 123, 123, 124, 125, 125, 126, 127, 127, 128, 128, 129, 130, 130, 131, 131, 132, 132, 134, 134, 134, 135, 136, 136, 136, 137, 119, 119, 120, 120, 121, 122, 122, 123, 123, 124, 124, 125, 126, 126, 126, 127, 127, 128, 128, 129, 130, 130, 130, 131, 132, 132, 132, 133, 134, 134, 134, 136, 136, 136, 136, 137, 138, 121, 121, 121, 122, 122, 123, 123, 123, 123, 125, 125, 125, 125, 126, 126, 127, 127, 127, 128, 128, 128, 129, 130, 130, 131, 131, 131, 132, 132, 133, 133, 133, 134, 135, 135, 135, 135, 136, 136, 137, 138, 120, 120, 120, 121, 122, 122, 123, 123, 123, 123, 124, 124, 125, 125, 125, 126, 126, 127, 127, 127, 127, 128, 128, 128, 129, 129, 130, 130, 131, 131, 132, 132, 132, 133, 134, 134, 134, 134, 135, 135, 136, 136, 137, 138, 138, 138, 138, 139, 139, 121, 121, 121, 122, 122, 122, 122, 123, 123, 124, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 127, 128, 128, 129, 129, 129, 130, 130, 131, 131, 131, 132, 132, 133, 133, 134, 134, 134, 135, 135, 135, 136, 137, 137, 137, 137, 137, 139, 139, 122, 123, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 126, 127, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 131, 131, 131, 132, 132, 133, 133, 133, 133, 134, 135, 135, 135, 136, 137, 137, 137, 137, 138, 138, 139, 139, 141, 141, 141, 141, 141, 141, 141, 142, 143, 144, 144, 144, 145, 145, 149, 151, 152, 153, 153, 153, 155, 155, 155, 156, 158, 158, 159, 159, 159, 163, 163, 164, 164, 164, 166, 168, 168, 168, 168, 236, 236, 236, 245, 245, 249, 250, 252, 253, 254, 254, 215, 219, 238, 238, 242, 245, 248, 248, 248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
vti2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

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
    for v in range(511):
        if v <= value_mean[0]:
            a, b = BLACK, mtospectra(cl[0])
            r = v / value_mean[0]
            c = blend(a, b, ratio=r)
        elif v >= value_mean[7]:
            a, b = mtospectra(cl[7]), WHITE
            # r = (v - value_mean[7]) / (510 - value_mean[7])
            r = (v - value_mean[7]) / 2. + .5
            # c = blend(a, b, ratio=r)
            ar, ag, ab = a.clamped_rgb
            c = (min(int(255 * ar + r), 255), min(int(255 * ag + r), 255), min(int(255 * ab + r), 255))
            # c = adjust_brightness(mtorgb(cl[7]), 128 + int(127 * (v - value_mean[7]) / (510 - value_mean[7]) + .5))
        else:
            for i in range(7):
                if v > value_mean[i + 1]:
                    continue
                a, b = mtospectra(cl[i]), mtospectra(cl[i + 1])
                r = (v - value_mean[i]) / (value_mean[i + 1] - value_mean[i])
                break
            c = blend(a, b, ratio=r)
        res.append(c)
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


npimg = np.zeros((100 + 50 * len(grf.CC_COLOURS), 512 * 5, 3), dtype=np.uint8)

vtb = []
vti = []
for i in range(8):
    cc = magenta(value_mean[i])

    for v in range(VALUE_BOUNDARIES[i], VALUE_BOUNDARIES[i + 1]):
        c = magenta(v)
        bj, bd = 0, 1000000000000
        for j in range(255):
            rgb = adjust_brightness(cc, j)
            d = color_distance(c, rgb)
            if d < bd:
                bj, bd = j, d

        vtb.append(bj)
        vti.append(i)

vtb1, vti1 = vtb, vti

vtb, vti = vtb1, vti1
vtb = []
vti = []
for v in range(511):
    bi, bj, bd = 0, 0, 100000000000000

    s, e = 0, 8
    if v < 100: s, e = 0, 1
    # elif v > 450: s, e = 7, 8

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

for v in range(511):
    npimg[50:99, v + 512] = magenta(v)

y = 100
for i, cl in enumerate(grf.CC_COLOURS):
    for v in range(511):
        rgb = mtorgb(cl[vti[v]])
        npimg[y: y + 49, v] = rgb
        npimg[y: y + 49, v + 512 * 4] = adjust_brightness(rgb, vtb[v])
        npimg[y: y + 49, v + 512] = target[i][v]
        rgb = mtorgb(cl[vti1[v]])
        npimg[y: y + 49, v + 512 * 2] = adjust_brightness(rgb, vtb1[v])
        rgb = mtorgb(cl[vti2[v]])
        npimg[y: y + 49, v + 512 * 3] = adjust_brightness(rgb, vtb2[v])
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
