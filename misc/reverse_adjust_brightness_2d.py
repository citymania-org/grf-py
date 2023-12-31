import sys
import math
from collections import defaultdict
from pprint import pprint

from PIL import Image
import numpy as np
import spectra

import grf

SATSTEPS = 16


def mtorgb(c):
    return grf.PALETTE[3 * c:3 * c + 3]

def mtospectra(c):
    return grf.to_spectra(*mtorgb(c))

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

cc_colours = grf.CC_COLOURS[:2]
npimg = np.zeros((2 * (SATSTEPS + 1) * len(cc_colours), 512 * 4, 3), dtype=np.uint8)

BLACK = grf.to_spectra(0, 0, 0)
WHITE = grf.to_spectra(255, 255, 255)
for ci, cl in enumerate(cc_colours):
    print(ci, flush=True)
    for v in range(511):
        if v <= value_mean[0]:
            a, b = BLACK, mtospectra(cl[0])
            r = v / value_mean[0]
            # c = blend(a, b, ratio=r)
            cc = a.blend(b, ratio=r)
        elif v >= value_mean[7]:
            a, b = mtospectra(cl[7]), WHITE
            # r = (v - value_mean[7]) / (510 - value_mean[7])
            r = (v - value_mean[7]) / 2. + .5
            # c = blend(a, b, ratio=r)
            ar, ag, ab = a.clamped_rgb
            cc = (min(int(255 * ar + r), 255), min(int(255 * ag + r), 255), min(int(255 * ab + r), 255))
            cc = grf.to_spectra(*cc)
            # c = adjust_brightness(mtorgb(cl[7]), 128 + int(127 * (v - value_mean[7]) / (510 - value_mean[7]) + .5))
        else:
            for i in range(7):
                if v > value_mean[i + 1]:
                    continue
                a, b = mtospectra(cl[i]), mtospectra(cl[i + 1])
                r = (v - value_mean[i]) / (value_mean[i + 1] - value_mean[i])
                break
            # c = blend(a, b, ratio=r)
            cc = a.blend(b, ratio=r)

        l, c, h = cc.to('lch').values
        for j in range(1, SATSTEPS + 1):
            if j == SATSTEPS:
                ct = spectratorgb(cc)
                npimg[satofs:satofs + SATSTEPS - 1, v] = ct
            else:
                ct = spectratorgb(spectra.lch(l, c * j / SATSTEPS, h))
            satofs = j + SATSTEPS * 2 * ci - 1
            npimg[satofs, v] = ct
        print(end='.', flush=True)


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



color_idx = []
for cl in grf.CC_COLOURS:
    idx = {}
    for i in range(8):
        for j in range(256):
            c = adjust_brightness(mtorgb(cl[i]), j)
            idx[c] = (i, j)
    color_idx.append(idx)



for v in range(511):
    print(v)
    for s in range(SATSTEPS):
        bi, bj, bd = None, None, 100000000000000
        for i in range(8):
            tbj, tbd = None, 100000000000000
            for j in range(16):
                d = 0
                jj = j * 16 + 8
                for ci, cl in enumerate(cc_colours):
                    t = tuple(npimg[s + SATSTEPS * 2 * ci, v])
                    c = adjust_brightness(mtorgb(cl[i]), jj)
                    d += color_distance(c, t)
                if d < tbd:
                    tbj, tbd = jj, d
                if d < bd:
                    bi, bj, bd = i, jj, d

            for j in range (tbj - 8, tbj + 8):
                if j == tbj:
                    continue
                d = 0
                for ci, cl in enumerate(cc_colours):
                    t = tuple(npimg[s + SATSTEPS * 2 * ci, v])
                    c = adjust_brightness(mtorgb(cl[i]), j * 16 + 8)
                    d += color_distance(c, t)
                if d < bd:
                    bi, bj, bd = i, j, d

        for ci, cl in enumerate(cc_colours):
            cc = mtorgb(cl[bi])
            c = adjust_brightness(cc, bj)
            satofs = SATSTEPS * 2 * ci + s
            npimg[satofs, v + 512 * 2] = cc
            npimg[satofs, v + 512 * 3] = (bj, bj, bj)
            npimg[satofs, v + 512] = c
            if s == SATSTEPS - 1:
                npimg[satofs:satofs + SATSTEPS - 1, v + 512] = c

        print(end='.', flush=True)


img = Image.fromarray(npimg)
img.save("magenta.png")

