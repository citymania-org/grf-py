import math
import os
import struct
import time

import numpy as np
from PIL import Image

from .common import ZOOM_4X, BPP_8, BPP_24, BPP_32, PALETTE, ALL_COLOURS, SAFE_COLOURS, \
    to_spectra, SPECTRA_PALETTE, WIN_TO_DOS


def color_distance(c1, c2):
    rmean = (c1.rgb[0] + c2.rgb[0]) / 2.
    r = c1.rgb[0] - c2.rgb[0]
    g = c1.rgb[1] - c2.rgb[1]
    b = c1.rgb[2] - c2.rgb[2]
    return math.sqrt(
        ((2 + rmean) * r * r) +
        4 * g * g +
        (3 - rmean) * b * b)


def find_best_color(x, in_range=SAFE_COLOURS):
    mj, md = 0, 1e100
    for j in in_range:
        c = SPECTRA_PALETTE[j]
        d = color_distance(x, c)
        if d < md:
            mj, md = j, d
    return mj


def fix_palette(img, sprite_name):
    assert (img.mode == 'P')  # TODO
    pal = tuple(img.getpalette())
    if pal == PALETTE: return img
    print(f'Custom palette in sprite {sprite_name}, converting...')
    # for i in range(256):
    #     if tuple(pal[i * 3: i*3 + 3]) != PALETTE[i * 3: i*3 + 3]:
    #         print(i, pal[i * 3: i*3 + 3], PALETTE[i * 3: i*3 + 3])
    remap = PaletteRemap()
    for i in ALL_COLOURS:
        remap.remap[i] = find_best_color(to_spectra(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]), in_range=ALL_COLOURS)
    return remap.remap_image(img)


def open_image(filename, *args, **kw):
    return fix_palette(Image.open(filename, *args, **kw))


def convert_image(image):
    if image.mode == 'P':
        return image, BPP_8
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return img, BPP_24


class BaseSprite:
    def get_data(self):
        raise NotImplemented

    def get_data_size(self):
        raise NotImplemented


class LazyBaseSprite(BaseSprite):
    def __init__(self):
        super().__init__()
        self._data = None

    def _encode(self):
        raise NotImplemented

    def get_data(self):
        if self._data is None:
            self._data = self._encode()
        return self._data

    def get_data_size(self):
        return len(self.get_data())


class IntermediateSprite:
    pass


class RealSprite(BaseSprite, IntermediateSprite):
    def __init__(self):
        self.sprite_id = None

    def get_data_size(self):
        return 4

    def get_data(self):
        return struct.pack('<I', self.sprite_id)

    def get_real_data(self, encoder):
        raise NotImplementedError


class SoundSprite(RealSprite):
    def __init__(self):
        super().__init__()
        self.id = None

    def get_hash(self):
        raise NotImplemented


class PaletteRemap(BaseSprite):
    def __init__(self, ranges=None):
        self.remap = np.arange(256, dtype=np.uint8)
        if ranges:
            self.set_ranges(ranges)

    def get_data(self):
        return b'\x00' + self.remap.tobytes()

    def get_data_size(self):
        return len(self.remap) + 1

    @classmethod
    def from_function(cls, color_func, remap_water=False):
        res = cls()
        for i in SAFE_COLOURS:
            res.remap[i] = find_best_color(color_func(SPECTRA_PALETTE[i]))
        if remap_water:
            for i in WATER_COLOURS:
                res.remap[i] = find_best_color(color_func(SPECTRA_PALETTE[i]))
        return res

    @classmethod
    def from_array(cls, array):
        assert len(array) == 256
        res = cls()
        res.remap = np.array(array, dtype=np.uint8)
        return res

    @classmethod
    def from_buffer(cls, data):
        res = cls()
        res.remap = np.frombuffer(data, dtype=np.uint8)
        return res

    def set_ranges(self, ranges):
        for r in ranges:
            f, t, v = r
            self.remap[f: t + 1] = v

    def remap_image(self, im):
        assert im.mode == 'P', im.mode
        data = np.array(im)
        data = self.remap[data]
        res = Image.fromarray(data)
        res.putpalette(PALETTE)
        return res

    def remap_array(self, a):
        return self.remap[a]


WIN_TO_DOS_REMAP = PaletteRemap.from_array(WIN_TO_DOS)


class GraphicsSprite(RealSprite):
    def __init__(self, w, h, *, xofs=0, yofs=0, zoom=ZOOM_4X, bpp=None, mask=None):
        if bpp == BPP_8 and mask is not None:
            raise ValueError("8bpp sprites can't have a mask")
        super().__init__()
        self.w = w
        self.h = h
        self.xofs = xofs
        self.yofs = yofs
        self.zoom = zoom
        self.bpp = bpp
        self.mask = mask
        self.name = f'{self.w}x{self.h}'

    def get_image(self):
        raise NotImplementedError

    def get_mask_image(self):
        raise NotImplementedError

    def draw(self, img):
        raise NotImplementedError

    def get_colourkey(self):
        return None

    # https://github.com/OpenTTD/grfcodec/blob/master/docs/grf.txt
    def get_real_data(self, encoder):
        t0 = time.time()
        img, bpp = self.get_image()

        encoder.count_loading(time.time() - t0)
        t0 = time.time()

        if self.bpp is None:
            self.bpp = bpp
        npalpha = None

        if bpp != self.bpp:
            print(f'Sprite {self.name} expected {self.bpp}bpp but file is {bpp}bpp, converting.')
            if self.bpp == BPP_24:
                img = img.convert('RGB')
            elif self.bpp == BPP_32:
                img = img.convert('RGBA')
            elif self.bpp == BPP_8:
                p_img = Image.new('P', (16, 16))
                p_img.putpalette(PALETTE)
                img = img.quantize(palette=p_img, dither=0)
        else:
            if bpp == BPP_8:
                img = fix_palette(img, self.name)

        encoder.count_conversion(time.time() - t0)
        t0 = time.time()

        npimg = np.asarray(img)
        colourkey = self.get_colourkey()
        if colourkey is not None:
            if self.bpp == BPP_24:
                npalpha = 255 * np.all(np.not_equal(npimg, colourkey), axis=2)
                npalpha = npalpha.astype(np.uint8, copy=False)
            elif self.bpp == BPP_32:
                if len(colourkey) == 3:
                    colourkey = (*colourkey, 255)
                npalpha = 255 * np.all(np.not_equal(npimg, colourkey), axis=2)
                npalpha = npalpha.astype(np.uint8, copy=False)
            else:
                print(f'Colour key on 8bpp sprites is not supported')

        info_byte = 0x40
        if self.bpp == BPP_24 or self.bpp == BPP_32:
            stack = [npimg]

            if self.bpp == BPP_32:
                npimg.shape = (self.w * self.h, 4)
                rbpp = 4
                info_byte |= 0x1 | 0x2  # rgb, alph

                if npalpha is not None:
                    npimg = npimg[:, :3]
                    npalpha.shape = (self.w * self.h, 1)
                    stack = [npimg, npalpha]
            else:
                npimg.shape = (self.w * self.h, 3)
                rbpp = 3
                info_byte = 0x1  # rgb

                if npalpha is not None:
                    npalpha.shape = (self.w * self.h, 1)
                    stack.append(npalpha)
                    rbpp += 1
                    info_byte |= 0x2  # alpha


            if self.mask is not None:
                mask_file, xofs, yofs = self.mask
                mask_img, mask_bpp = self.get_mask_image()
                if mask_bpp != BPP_8:
                    raise RuntimeError(f'Mask {mask_file.path} is not an 8bpp image')
                xofs, yofs = self.x + xofs, self.y + yofs
                mask_img = mask_img.crop((xofs, yofs, xofs + self.w, yofs + self.h))
                mask_img = fix_palette(mask_img, f'{mask_file.path} {xofs},{yofs} {self.w}x{self.h}')
                npmask = np.asarray(mask_img)
                npmask.shape = (self.w * self.h, 1)
                stack.append(npmask)
                rbpp += 1
                info_byte |= 0x4  # mask

            if len(stack) > 1:
                raw_data = np.concatenate(stack, axis=1)
            else:
                raw_data = stack[0]
            raw_data.shape = self.w * self.h * rbpp

        else:
            info_byte |= 0x4  # pal/mask
            raw_data = npimg
            raw_data.shape = self.w * self.h

        encoder.count_composing(time.time() - t0)
        data = encoder.sprite_compress(raw_data)
        return struct.pack(
            '<IIBBHHhh',
            self.sprite_id,
            len(data) + 10, info_byte,
            self.zoom,
            self.h,
            self.w,
            self.xofs,
            self.yofs,
        ) + data


class EmptyGraphicsSprite(GraphicsSprite):
    def __init__(self):
        super().__init__(1, 1)

    def draw(self, img):
        pass

    def get_real_data(self, encoder):
        return struct.pack(
            '<IIBBHHhhB',
            self.sprite_id,
            11,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
        )

EMPTY_SPRITE = EmptyGraphicsSprite()


class ImageSprite(GraphicsSprite):
    def __init__(self, image, w, h, *, mask=None, **kw):
        self._image = convert_image(image)
        if mask:
            mask_image, xofs, yofs = mask
            mask = convert_image(mask_image), xofs, yofs
        super().__init__(w, h, bpp=self._image[1], mask=mask, **kw)

    def get_image(self):
        return self._image

    def get_mask_image(self):
        return self.mask[0] if self.mask else None


class ImageFile:
    def __init__(self, path, colourkey=None):
        self.path = path
        self.colourkey = colourkey
        self._image = None

    def get_image(self):
        if self._image:
            return self._image
        img = Image.open(self.path)
        if img.mode == 'P':
            self._image = (img, BPP_8)
        elif img.mode == 'RGB':
            self._image = (img, BPP_24)
        else:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            self._image = (img, BPP_32)
        return self._image


class FileSprite(GraphicsSprite):
    def __init__(self, file, x, y, w, h, *, bpp=None, mask=None, **kw):
        assert(isinstance(file, ImageFile))
        super().__init__(w, h, bpp=bpp, mask=mask, **kw)
        self.x = x
        self.y = y
        self.file = file
        self.name = f'{self.x},{self.y} {self.w}x{self.h}'

    def get_image(self):
        img, bpp = self.file.get_image()
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        return img, bpp

    def get_mask_image(self):
        mask_file, _, _ = self.mask
        return mask_file.get_image()

    def get_colourkey(self):
        return self.file.colourkey


class RAWSound(SoundSprite):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def get_real_data(self, encoder):
        data = open(self.file, 'rb').read()
        name = os.path.basename(self.file).encode()
        if len(name) > 256:
            name = name[:251] + '_' + name[-4:]
        return struct.pack('<IIBBB', self.sprite_id, len(name) + len(data) + 4, 0xff, 0xff, len(name)) + name + b'\0' + data

    def get_hash(self):
        return self.file


class NMLFileSpriteWrapper:
    def __init__(self, sprite, file, bit_depth):
        self.sprite = sprite
        self.file = type('Filename', (object, ), {'value': file.path})
        self.bit_depth = bit_depth
        self.mask_file = type('Filename', (object, ), {'value': file.mask}) if file.mask else None
        self.mask_pos = None
        self.flags = type('Flags', (object, ), {'value': 0})

    xpos = property(lambda self: type('XSize', (object, ), {'value': self.sprite.x}))
    ypos = property(lambda self: type('YSize', (object, ), {'value': self.sprite.y}))

    xsize = property(lambda self: type('XSize', (object, ), {'value': self.sprite.w}))
    ysize = property(lambda self: type('YSize', (object, ), {'value': self.sprite.h}))

    xrel = property(lambda self: type('XSize', (object, ), {'value': self.sprite.xofs}))
    yrel = property(lambda self: type('YSize', (object, ), {'value': self.sprite.yofs}))

