import colour
import spectra
import numpy as np
from PIL import Image

import grf


# def srgb_to_linear(rgb):
#     """Convert sRGB to linear RGB"""
#     rgb = np.array(rgb)
#     mask = rgb <= 0.04045
#     rgb[mask] /= 12.92
#     rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
#     return rgb


# def linear_to_srgb(rgb_linear):
#     """Convert linear RGB to sRGB"""
#     rgb_linear = np.array(rgb_linear)
#     mask = rgb_linear <= 0.0031308
#     rgb_linear[mask] *= 12.92
#     rgb_linear[~mask] = 1.055 * (rgb_linear[~mask] ** (1.0 / 2.4)) - 0.055
#     return rgb_linear.clip(0, 1)


# def srgb_to_oklab(rgb):
#     """Convert sRGB to Oklab"""
#     rgb_linear = srgb_to_linear(np.array(rgb) / 255.)
#     xyz = np.dot(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
#                             [0.2119034982, 0.6806995451, 0.1073969566],
#                             [0.0883024619, 0.2817188376, 0.6299787005]]), rgb_linear)

#     xyzn = np.array([0.9504559271, 1.0, 1.0890577508])
#     xyz /= xyzn

#     l = 0.2104542553 * np.log10(xyz[1])
#     l_denom = np.linalg.norm(xyz)
#     a = 0.9373114733 * np.arctan2(xyz[2], xyz[0])
#     b = 0.0585503811 * np.arcsin(xyz[1] / l_denom)

#     return [l, a, b]


# def oklab_to_srgb(lab):
#     """Convert Oklab to sRGB"""
#     l, a, b = lab

#     fy = (l + 16) / 116
#     fx = a / 500 + fy
#     fz = fy - b / 200

#     x = 0.9504559271 * (fx ** 3 if fx ** 3 > 0.008856 else (fx - 16 / 116) / 7.787)
#     y = (l ** 3 if l > 0.008856 else l / 903.3)
#     z = 1.0890577508 * (fz ** 3 if fz ** 3 > 0.008856 else (fz - 16 / 116) / 7.787)

#     xyz = np.array([x, y, z])

#     rgb_linear = np.dot(np.array([[3.2404542, -1.5371385, -0.4985314],
#                                     [-0.969266, 1.8760108, 0.041556],
#                                     [0.0556434, -0.2040259, 1.0572252]]), xyz)

#     return linear_to_srgb(rgb_linear) * 255


def srgb_to_linear(rgb):
    mask = rgb <= 0.04045
    rgb[mask] /= 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    return rgb


def linear_to_srgb(rgb_linear):
    rgb_linear = np.array(rgb_linear)
    mask = rgb_linear <= 0.0031308
    rgb_linear[mask] *= 12.92
    rgb_linear[~mask] = 1.055 * (rgb_linear[~mask] ** (1 / 2.4)) - 0.055
    return rgb_linear


def srgb_to_oklab(rgb):
    rgb_linear = srgb_to_linear(np.array(rgb) / 255.)

    # Convert to linear RGB in the range [0, 1]
    x = np.array((
        (0.4122214708, 0.5363325363, 0.0514459929),
        (0.2119034982, 0.6806995451, 0.1073969566),
        (0.0883029595, 0.2817188376, 0.6299787005),
    )).dot(rgb_linear)

    x_ = np.cbrt(x)

    return np.array((
        (0.2104542553, +0.7936177850, -0.0040720468),
        (1.9779984951, -2.4285922050, +0.4505937099),
        (0.0259040371, +0.7827717662, -0.8086757660),
    )).dot(x_)


def oklab_to_srgb(lab):
    x_ = np.array((
        (1, +0.3963377774, +0.2158037573),
        (1, -0.1055613458, -0.0638541728),
        (1, -0.0894841775, -1.2914855480),
    )).dot(lab)

    x = x_ ** 3

    rgb = np.array((
        (+4.0767416621, -3.3077115913, +0.2309699292),
        (-1.2684380046, +2.6097574011, -0.3413193965),
        (-0.0041960863, -0.7034186147, +1.7076147010),
    )).dot(x)

    return linear_to_srgb(rgb) * 255



# def srgb_to_oklab(rgb):
#     """
#     Convert RGB color to Oklab color space.

#     Args:
#     - rgb (tuple): Tuple representing RGB color values in the range [0, 1].

#     Returns:
#     - tuple: Oklab color space values.
#     """
#     rgb_clamped = tuple(max(0, min(1, c)) for c in rgb)  # Ensure RGB values are within [0, 1]
#     rgb_linear = colour.models.eotf_inverse(rgb_clamped, function='sRGB')  # Convert to linear RGB
#     oklab = colour.RGB_to_Oklab(rgb_linear)  # Convert linear RGB to Oklab
#     return oklab


# def oklab_to_srgb(oklab):
#     """
#     Convert Oklab color to RGB color space.

#     Args:
#     - oklab (tuple): Tuple representing Oklab color space values.

#     Returns:
#     - tuple: RGB color values in the range [0, 1].
#     """
#     rgb_linear = colour.Oklab_to_RGB(oklab)  # Convert Oklab to linear RGB
#     rgb = colour.models.eotf(rgb_linear, function='sRGB')  # Apply the sRGB EOTF
#     return rgb


class HueShiftCCReplaceSprite(grf.Sprite):
    def __init__(self, sprite, c1, c2):
        self.sprite = sprite
        self.c1 = c1
        self.c2 = c2

    def get_fingerprint(self):
        return {
            'class': self.__class__.__name__,
            'sprite': self.sprite.get_fingerprint(),
        }


    def get_data_layers(self, encoder=None, crop=None):
        remap = np.full((256, 3), 128, dtype=np.uint8)
        remap[0] = (0, 0, 255)

        for i in grf.SAFE_COLOURS:
            remap[i] = grf.PALETTE[i * 3: i * 3 + 3]
        remap[0xC6: 0xCE] = palette

        w, h, xofs, yofs, npimg, npalpha, npmask = self.sprite.get_data_layers(encoder, crop)
        assert npimg is None

        npimg = remap[npmask]
        np.set_printoptions(threshold=10000)
        return w, h, xofs, yofs, npimg, None, None


    def get_debug_image(self):
        _, _, _, _, npimg, _, _ = self.get_data_layers()
        im = Image.fromarray(npimg, mode='RGB')
        return im


def make_hue_shift_palette(c1, c2):
    CCL = [27.07510258707097, 34.38502265503816, 42.02402843768756, 49.50112987272844, 57.173277387157796, 65.74123652910247, 74.82268281895817, 83.32566595929231]
    l1, a1, b1 = srgb_to_oklab(c1)
    l2, a2, b2 = srgb_to_oklab(c2)
    # l1, a1, b1 = grf.to_spectra(*c1).to('lab').values
    # l2, a2, b2 = grf.to_spectra(*c2).to('lab').values
    lscale = (l2 - l1) / (CCL[-1] - CCL[0])
    palette = []
    ndi = np.zeros((50, 400, 3), dtype=np.uint8)
    mcol = lambda x:  max(min(int(x + .5), 255), 0)
    for i in range(8):
        f = i / 7.
        c = (
            l1 + (CCL[i] - CCL[0]) * lscale,
            a2 * f + a1 * (1 - f),
            b2 * f + b1 * (1 - f),
        )
        # r, g, b = c.to('rgb').values
        r, g, b = oklab_to_srgb(c)
        rgb = (mcol(r), mcol(g), mcol(b))
        palette.append(rgb)
        ndi[:, i * 50:i * 50 + 50] = rgb

    img = Image.fromarray(ndi, mode='RGB')
    img.show()
    return palette


dc = lambda rgb: oklab_to_srgb(srgb_to_oklab(rgb))
print(dc((111, 222, 121)))
print(dc((111, 222, 0)))
print(dc((111, 0, 0)))
# print(linear_to_srgb(srgb_to_linear(np.array((.1, .2, .3)))))


# ase = lib.AseImageFile('sprites/vehicles/road_lorries_firstgeneration_32bpp.ase')
# s = grf.FileSprite(ase, 0, 0, 523, 337, bpp=grf.BPP_8)
# s.get_image()[0].show()
print(make_hue_shift_palette((97, 40, 0), (171, 255, 97)))
# print(make_hue_shift_palette((1, 1, 0), (171, 255, 97)))
# print(make_hue_shift_palette((30, 0, 0), (192, 90, 255)))
print(make_hue_shift_palette((50, 40, 245), (110, 56, 245)))
print(make_hue_shift_palette((50, 40, 245), (220, 203, 56)))
print(make_hue_shift_palette((255, 36, 244), (172, 5, 255)))
# print(make_hue_shift_palette((69, 6, 106), (253, 96, 23)))
print(make_hue_shift_palette((0x03,0x11,0x6F), (0xF9,0xDE,0x2D)))
print(make_hue_shift_palette((220, 203, 56), (220, 203, 56)))



def debug_cc_recolour(sprites, horizontal=False):
    PADDING = 10

    slayers = []
    for i, s in enumerate(sprites):
        slayers.append(s.get_data_layers())

    cc_colours = grf.CC_COLOURS[:5]

    # cc_colours.append([(95, 40, 0), (106, 68, 9), (117, 97, 20), (126, 125, 31), (134, 154, 43), (143, 187, 59), (152, 222, 77), (158, 255, 93)])
    # cc_colours.append([(99, 18, 246), (104, 22, 246), (110, 26, 246), (115, 29, 245), (120, 33, 245), (125, 36, 245), (130, 40, 246), (135, 44, 246)])
    # cc_colours.append([(99, 18, 246), (131, 54, 221), (154, 81, 197), (170, 105, 172), (183, 129, 147), (194, 154, 122), (204, 180, 94), (211, 204, 52)])
    # cc_colours.append([(255, 4, 245), (251, 0, 247), (242, 0, 249), (232, 0, 251), (222, 0, 253), (211, 0, 254), (199, 0, 255), (187, 0, 255)])
    # cc_colours.append([(35, 10, 111), (74, 37, 107), (104, 64, 102), (130, 91, 96), (155, 120, 87), (182, 153, 79), (212, 189, 66), (239, 224, 39)])

    cc_colours.append([(97, 40, 0), (110, 68, 8), (123, 96, 21), (133, 124, 33), (143, 153, 46), (153, 185, 62), (164, 221, 80), (171, 255, 97)])
    cc_colours.append([(50, 40, 245), (60, 43, 245), (69, 46, 245), (77, 48, 244), (86, 50, 244), (94, 52, 244), (102, 54, 245), (110, 56, 245)])
    cc_colours.append([(50, 40, 245), (59, 84, 228), (78, 112, 210), (102, 133, 190), (128, 152, 169), (158, 170, 144), (189, 187, 113), (220, 203, 56)])
    cc_colours.append([(255, 36, 244), (243, 38, 246), (231, 38, 248), (220, 37, 250), (208, 34, 252), (196, 29, 253), (184, 20, 254), (172, 5, 255)])
    cc_colours.append([(3, 17, 111), (23, 54, 114), (55, 82, 115), (88, 108, 113), (123, 134, 107), (163, 162, 99), (206, 193, 84), (249, 222, 45)])

    pos = []
    if horizontal:
        maxw = max(x[0] for x in slayers)
        x = PADDING
        for i in range(len(cc_colours)):
            y = PADDING
            row = []
            for s in slayers:
                row.append(((x, y)))
                y += s[1] + PADDING
            pos.append(row)
            x += maxw + PADDING
    else:
        # s = (sumh + (len(slayers) + 1) * PADDING, maxw + 2 * PADDING)
        y = PADDING
        maxh = max(x[1] for x in slayers)
        for i in range(len(cc_colours)):
            x = PADDING
            row = []
            for s in slayers:
                row.append(((x, y)))
                x += s[0] + PADDING
            pos.append(row)
            y += maxh + PADDING

    npres = np.zeros((y, x, 4), dtype=np.uint8)
    for jj, s, (w, h, xofs, yofs, npimg, npalpha, npmask) in zip(range(len(sprites)), sprites, slayers):
        if npmask is None:
            raise ValueError(f'Sprite {s.name} has no mask!')

        for ii, cl in enumerate(cc_colours):
            rgb_palette = not isinstance(cl[0], int)
            r = grf.PaletteRemap()
            if not rgb_palette:
                r.set_ranges(((0xC6, 0xCD, cl),))
            ni = r.remap_array(npmask)

            x, y = pos[ii][jj]
            for i in range(h):
                for j in range(w):
                    m = ni[i, j]
                    if m == 0:
                        if npimg is not None:
                            npres[y + i, x + j][:3] = npimg[i, j][:3]
                    else:
                        if rgb_palette and 0xC6 <= m <= 0xCD:
                            rgb = cl[m - 0xC6]
                        else:
                            rgb = grf.PALETTE[m * 3:m * 3 + 3]
                        b = max(npimg[i, j][:3]) if npimg is not None else 128
                        npres[y + i, x + j][:3] = adjust_brightness(rgb, b)
                    npres[y + i, x + j, 3] = npimg[i, j][3] if npimg is not None else 255

    im = Image.fromarray(npres, mode='RGBA')
    im.show()


def cmd_debugcc_handler(g, grf_file, args):
    path = args.image_file
    if path.endswith('.ase'):
        imgfile = lib.AseImageFile(path, layer=args.layer)
        sprite = lib.CCReplacingFileSprite(imgfile, 0, 0, None, None, name=path)
    else:
        imgfile = grf.ImageFile(path)
        sprite = grf.FileSprite(imgfile, 0, 0, None, None, name=path)
    lib.debug_cc_recolour([sprite], horizontal=args.horizontal)

