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


_remap_cache = {}
_FIX_PALETTE_LOOKUP = {(PALETTE[3 * i], PALETTE[3 * i + 1], PALETTE[3 * i + 2]): i for i in ALL_COLOURS}

def fix_palette(img, sprite_name):
    assert (img.mode == 'P')  # TODO
    pal = tuple(img.getpalette())
    if pal == PALETTE: return img
    print(f'Custom palette in sprite {sprite_name}, converting...')
    # for i in range(256):
    #     if tuple(pal[i * 3: i*3 + 3]) != PALETTE[i * 3: i*3 + 3]:
    #         print(i, pal[i * 3: i*3 + 3], PALETTE[i * 3: i*3 + 3])
    pal_hash = hash(pal)
    remap = _remap_cache.get(pal_hash)
    if remap is None:
        remap = PaletteRemap()
        for i in range(len(pal) // 3):
            color = (pal[3 * i], pal[3 * i + 1], pal[3 * i + 2])
            res = _FIX_PALETTE_LOOKUP.get(color)
            if res is None:
                res = find_best_color(to_spectra(*color), in_range=ALL_COLOURS)
            remap.remap[i] = res
        _remap_cache[pal_hash] = remap
    return remap.remap_image(img)


def open_image(filename, *args, **kw):
    return fix_palette(Image.open(filename, *args, **kw), filename)


def convert_image(image):
    if image.mode == 'P':
        return image, BPP_8
    if image.mode == 'RGBA':
        return image, BPP_32
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return img, BPP_24


# Pseudo sprite in grf
class Action:
    def get_data(self):
        raise NotImplemented

    def get_data_size(self):
        raise NotImplemented


class FakeAction:
    def get_data(self):
        return b't'

    def get_data_size(self):
        return 0


# Pseudo sprite (reference) + real sprite in grf
class ResourceAction(Action):
    def __init__(self):
        self.sprite_id = None

    def get_data_size(self):
        return 4

    def get_data(self):
        return struct.pack('<I', self.sprite_id)

    def get_resources(self):
        raise NotImplementedError

    def get_resource_files(self):
        raise NotImplementedError


class SingleResourceAction(ResourceAction):
    def __init__(self, resource):
        super().__init__()
        self.resource = resource

    def get_resources(self):
        return (self.resource, )

    def get_resource_files(self):
        return self.resource.get_resource_files()


class AlternativeSprites(ResourceAction):
    def __init__(self, *sprites):
        assert all(isinstance(s, Sprite) for s in sprites), sprites
        assert len(set((s.zoom, s.bpp) for s in sprites)) == len(sprites), sprites

        super().__init__()
        self.sprites = sprites

    def get_sprite(self, *, zoom=None, bpp=None):
        for s in self.sprites:
            if (zoom is None or s.zoom == zoom) and (bpp is None or s.bpp == bpp):
                return s
        return None

    def get_resources(self):
        return self.sprites

    def get_resource_files(self):
        res = []
        for s in self.sprites:
            res.extend(s.get_resource_files())
        return res


class ResourceFile:
    def __init__(self, path):
        self.path = path


class CodeFile(ResourceFile):
    pass


class PythonFile(CodeFile):
    pass


class SoundFile(ResourceFile):
    pass


THIS_FILE = PythonFile(__file__)


class LoadedResourceFile(ResourceFile):
    def load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError


class Resource:
    def get_fingerprint(self):
        raise NotImplemented

    def get_resource_files(self):
        return NotImplemented


class Sound(Resource):
    def __init__(self):
        super().__init__()
        self.id = None


class PaletteRemap(Action):
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


def combine_fingerprint(*args, **kw):
    res = {}
    for x in args:
        if x is None:
            return None
        res.update(x)

    for k, v in kw.items():
        if callable(v):
            v = v()
            if v is None:
                return None
        res[k] = v
    return res


class Mask:
    class Mode:
        DEFAULT = 0
        OVERDRAW = 1

    def __init__(self, mode=Mode.DEFAULT):
        self.mode = mode

    def get_image(self):
        raise NotImplementedError

    def get_resource_files(self):
        raise NotImplementedError

    def get_fingerprint(self):
        return {
            'class': self.__class__.__name__,
            'mode': self.mode,
        }


class FileMask(Mask):
    def __init__(self, file, x=0, y=0, w=None, h=None, *args, **kw):
        assert(isinstance(file, ImageFile))
        super().__init__(*args, **kw)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.file = file

    def get_image(self):
        img, bpp = self.file.get_image()
        if self.w is None or self.h is None:
            self.w, self.h = img.size
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        return img, bpp

    def __str__(self):
        return str(self.file.path)

    def get_fingerprint(self):
        return combine_fingerprint(
            super().get_fingerprint(),
            x=self.x,
            y=self.y,
            w=self.w,
            h=self.h,
        )

    def get_resource_files(self):
        return (self.file, THIS_FILE)


class ImageMask(Mask):
    def __init__(self, image, *args, **kw):
        assert(isinstance(file, ImageFile))
        super.__init__(*args, **kw)
        self.image = image
        self._converted_image = None

    def get_image(self):
        if self._converted_image is None:
            self._converted_image = convert_image(self.image)
        return self._converted_image

    def __str__(self):
        w, h = self.image.size()
        return f'Image({w}*{h} {self.xofs:+},{self.yofs:+})'

    def get_resource_files(self):
        return (THIS_FILE,)

    def get_fingerprint(self):
        return None


class Sprite(Resource):
    def __init__(self, w, h, *, xofs=0, yofs=0, zoom=ZOOM_4X, bpp=None, mask=None, crop=True):
        if bpp == BPP_8 and mask is not None:
            raise ValueError("8bpp sprites can't have a mask")
        assert mask is None or isinstance(mask, Mask)
        super().__init__()
        self.w = w
        self.h = h
        self.xofs = xofs
        self.yofs = yofs
        self.zoom = zoom
        self.bpp = bpp
        self.mask = mask
        self.crop = crop

    @property
    def name(self):
        return f'{self.w}x{self.h}'

    def get_fingerprint_base(self):
        return combine_fingerprint(
            **{'class': self.__class__.__name__},
            w=self.w,
            h=self.h,
            xofs=self.xofs,
            yofs=self.yofs,
            crop=self.crop,
            mask=None if self.mask is None else self.mask.get_fingerprint,
        )

    def get_fingerprint(self):
        raise NotImplementedError

    def get_image(self):
        raise NotImplementedError

    def draw(self, img):
        raise NotImplementedError

    def get_colourkey(self):
        return None

    def get_data_layers(self, encoder=None):
        t0 = time.time()
        # TODO move into get_image
        img, bpp = self.get_image()
        w, h, xofs, yofs = self.w, self.h, self.xofs, self.yofs
        if w is None:
            w = img.size[0]
        if h is None:
            h = img.size[1]

        if encoder:
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

        if encoder is not None:
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

        crop_x = crop_y = 0
        if self.crop and w > 0 and h > 0:
            if npalpha is not None:
                npcheck = npalpha
            elif self.bpp == BPP_32:
                npcheck = npimg[:, :, 3]
            else:
                assert self.bpp == BPP_8, self.bpp
                npcheck = npimg

            cols_used = np.arange(w)[npcheck.any(0)]
            rows_used = np.arange(h)[npcheck.any(1)]
            crop_x = min(cols_used, default=0)
            crop_y = min(rows_used, default=0)
            w = max(cols_used, default=0) - crop_x + 1
            h = max(rows_used, default=0) - crop_y + 1
            xofs += crop_x
            yofs += crop_y

            if npalpha is not None:
                npalpha = npalpha[crop_y: crop_y + h, crop_x: crop_x + w]
            if self.bpp == BPP_8:
                npimg = npimg[crop_y: crop_y + h, crop_x: crop_x + w]
            else:
                npimg = npimg[crop_y: crop_y + h, crop_x: crop_x + w, :]

        if self.bpp == BPP_8:
            npmask = npimg
            npalpha = npimg = None
        else:
            assert self.bpp == BPP_24 or self.bpp == BPP_32

            if self.bpp == BPP_32 and npalpha is not None:
                # Remove alpha from npimg
                npimg = npimg[:, :, :3]

            mask = self.mask
            npmask = None
            if mask is not None:
                mask_img, mask_bpp = mask.get_image()
                if mask_bpp != BPP_8:
                    raise RuntimeError(f'Mask {mask} is not an 8bpp image')
                mask_img = fix_palette(mask_img, mask)
                if mask_img.size != (self.w, self.h):
                    print(mask)
                    raise RuntimeError(f'Mask {mask} size {mask_img.size} doesn'f't match image size {(self.w, self.h)}')
                npmask = np.asarray(mask_img)
                npmask = npmask[crop_y: crop_y + h, crop_x: crop_x + w]
                if mask.mode == Mask.Mode.OVERDRAW:
                    npimg = npimg.copy()
                    has_mask = (npmask != 0)
                    if npimg.shape[2] == 3:
                        npimg[has_mask] = (127, 127, 127)
                        if npalpha is not None:
                            npalpha[has_mask] = 255
                    else:
                        npimg[has_mask] = (127, 127, 127, 255)

        if encoder is not None:
            encoder.count_composing(time.time() - t0)

        return w, h, xofs, yofs, npimg, npalpha, npmask

    # https://github.com/OpenTTD/grfcodec/blob/master/docs/grf.txt
    def get_real_data(self, encoder):
        w, h, xofs, yofs, npimg, npalpha, npmask = self.get_data_layers(encoder)

        t0 = time.time()
        info_byte = 0x40
        stack = []
        rbpp = 0
        if npimg is not None:
            if npimg.ndim != 3:
                raise RuntimeError('Image layer is not a 3 dimensional array')
            bpp = npimg.shape[2]
            if bpp not in (3, 4):
                raise RuntimeError('Image layer should use 3 or 4 bpp')
            if bpp == 4:
                if npalpha is not None:
                    raise RuntimeError('4bpp image layer doesn''t allow separate alpha layer')
                info_byte |= 0x2
            info_byte |= 0x1
            rbpp += bpp
            stack.append(npimg.reshape(w * h, bpp))
        if npalpha is not None:
            if npalpha.ndim != 2:
                raise RuntimeError('Alpha layer is not a 2-dimensional array')
            info_byte |= 0x2
            rbpp += 1
            stack.append(npalpha.reshape((w * h, 1)))
        if npmask is not None:
            if npmask.ndim != 2:
                raise RuntimeError('Mask layer is not a 2-dimensional array')
            info_byte |= 0x4
            rbpp += 1
            stack.append(npmask.reshape((w * h, 1)))

        if len(stack) > 1:
            raw_data = np.concatenate(stack, axis=1)
        else:
            raw_data = stack[0]
        raw_data = raw_data.reshape(w * h * rbpp)

        encoder.count_composing(time.time() - t0)
        data = encoder.sprite_compress(np.ascontiguousarray(raw_data))
        return struct.pack(
            '<BBHHhh',
            info_byte,
            self.zoom,
            h,
            w,
            xofs,
            yofs,
        ) + data

    def get_image_files(self):
        raise NotImplementedError

    def get_resource_files(self):
        res = list(self.get_image_files())
        if self.mask is not None:
            res.extend(self.mask.get_resource_files())
        res.append(THIS_FILE)
        return tuple(res)

    def save_gif(self, filename: str):
        DARK_BLUE_WATER = ((32,  68, 112), (36,  72, 116), (40,  76, 120), (44,  80, 124), (48,  84, 128))
        DARK_BLUE_WATER_TOYLAND = ((28, 108, 124), (32, 112, 128), (36, 116, 132), (40, 120, 136), (44, 124, 140))
        LIGHTHOUSE_AND_STADIUM = ((240, 208, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        OIL_REFINERY_AND_FIRE = ((252, 60, 0), (252, 84, 0), (252, 108, 0), (252, 124, 0), (252, 148, 0), (252, 172, 0), (252, 196, 0))

        FIZZY_DRINKS = ((76, 24, 8), (108, 44, 24), (144, 72, 52), (176, 108,  84), (212, 148, 128))

        GLITTERY_WATER = (
            (216, 244, 252), (172, 208, 224), (132, 172, 196), (100, 132, 168),
            ( 72, 100, 144), ( 72, 100, 144), ( 72, 100, 144), ( 72, 100, 144),
            ( 72, 100, 144), ( 72, 100, 144), ( 72, 100, 144), ( 72, 100, 144),
            (100, 132, 168), (132, 172, 196), (172, 208, 224))

        GLITTERY_WATER_TOYLAND = (
            (216, 244, 252), (180, 220, 232), (148, 200, 216), (116, 180, 196),
            ( 92, 164, 184), ( 92, 164, 184), ( 92, 164, 184), ( 92, 164, 184),
            ( 92, 164, 184), ( 92, 164, 184), ( 92, 164, 184), ( 92, 164, 184),
            (116, 180, 196), (148, 200, 216), (180, 220, 232))

        RADIO_TOWER = ((255, 0, 0), (20, 0, 0))

        PAL_IDX = {tuple(PALETTE[i: i + 3]): i // 3 for i in range(0, 256 * 3, 3)}

        def extr(p, q):
            return lambda counter: (((counter * p) & 0xFFFF) * q) >> 16

        def extr2(p, q):
            return lambda counter: (((~counter * p) & 0xFFFF) * q) >> 16

        def init_cycle(pal, start, size, colours, func, cycle, *, step=1):
            cset = list(set(colours))
            assert len(cset) <= size, (len(cset), size)
            cdict = {x: i for i, x in enumerate(cset)}
            pal[start: start + len(cset)] = cset
            indexed = [cdict[x] + start for x in colours]
            res = []
            for c in range(0, cycle * 8, 8):
                j = func(c)
                res.append(indexed[j::step] + indexed[j % step:j:step])
            return start, size, res

        def init_radio_tower(pal, start, size, colours):
            pal[start: start + size] = colours  # TODO probably make fixed palette global

            def pick_color(i):
                if i < 0x3F: return 0xEF
                elif i < 0x4A or i >= 0x75: return 0xB4  # (128, 0, 0) in the regular palette
                else: return 0xF0

            res = []
            for c in range(0, 8 * 32, 8):
                i = (c >> 1) & 0x7F
                res.append((pick_color(i), pick_color(i ^ 0x40)))

            return start, res

        _, _, _, _, img, alpha, mask = self.get_data_layers()
        if img is not None or alpha is not None:
            raise NotImplementedError('Only 8bpp sprites are supported')

        pal = [tuple(PALETTE[i: i + 3]) for i in range(0, 256 * 3, 3)]
        cycles = []
        has_colors = np.isin(np.arange(0xE8, 0xFF, dtype=np.uint8), mask)
        if any(has_colors[0xEF - 0xE8: 0xEF - 0xE8 + 2]):
            cycles.append(init_radio_tower(pal, 0xEF, 2, RADIO_TOWER))

        for start, size, colours, func, cycle, step in (
                (0xE3, 5, FIZZY_DRINKS, extr2(512, 5), 16, 1),
                (0xE8, 7, OIL_REFINERY_AND_FIRE, extr2(512, 7), 16, 1),
                (0xF1, 4, LIGHTHOUSE_AND_STADIUM, extr(256, 4), 32, 1),
                (0xF5, 5, DARK_BLUE_WATER, extr(320, 5), 128, 1),
                (0xFA, 5, GLITTERY_WATER, extr(128, 15), 64, 3)):
            if any(has_colors[start - 0xE8: start + size - 0xE8]):
                cycles.append(init_cycle(pal, start, size, colours, func, cycle, step=step))

        max_cycle = max(len(l) for _, _, l in cycles)
        frames = []
        for i in range(max_cycle):
            if i == 0:
                remap = np.arange(256, dtype=np.uint8)
                for start, size, colours in cycles:
                    remap[start:start + size] = colours[i % len(colours)]
                frame_pal = sum(pal, ())
            else:
                # Optimize palette for consequtive frames by only saving colors we need
                remap = np.zeros(256, dtype=np.uint8)
                cur = 1
                frame_pal = [0,0,0]
                for start, size, colours in cycles:
                    remap[start:start + size] = range(cur, cur + size)
                    cur += size
                    for c in colours[i % len(colours)]:
                        frame_pal.extend(pal[c])
            im = Image.fromarray(remap[mask], mode='P')
            im.putpalette(frame_pal)
            frames.append(im)
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=27, loop=0, optimize=False, transparency=0, dispose=1)


class EmptySprite(Sprite):
    def __init__(self):
        super().__init__(1, 1)

    def draw(self, img):
        pass

    def get_real_data(self, encoder):
        return struct.pack(
            '<BBHHhhBB',
            0x4,
            0,
            1,
            1,
            0,
            0,
            1,
            0,
        )

    def get_resource_files(self):
        return (THIS_FILE,)

    def get_fingerprint(self):
        # Empty sprite is always the same so just return the class
        return {'class': self.__class__.__name__}


EMPTY_SPRITE = EmptySprite()


class ImageSprite(Sprite):
    def __init__(self, image, *, mask=None, **kw):
        self._image = convert_image(image)
        super().__init__(*self._image[0].size, bpp=self._image[1], mask=mask, **kw)

    def get_fingerprint(self):
        return None

    def get_image(self):
        return self._image

    def get_image_files(self):
        return ()


class ImageFile(LoadedResourceFile):
    def __init__(self, path, colourkey=None):
        self.path = path
        self.colourkey = colourkey
        self._image = None

    def load(self):
        if self._image is not None:
            return
        img = Image.open(self.path)
        if img.mode == 'P':
            self._image = (img, BPP_8)
        elif img.mode == 'RGB':
            self._image = (img, BPP_24)
        else:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            self._image = (img, BPP_32)

    def unload(self):
        if self._image is not None:
            self._image[0].close()
            self._image = None


    def get_image(self):
        self.load()
        return self._image


class FileSprite(Sprite):
    def __init__(self, file, x, y, w, h, *, bpp=None, mask=None, **kw):
        assert(isinstance(file, ImageFile))
        super().__init__(w, h, bpp=bpp, mask=mask, **kw)
        self.x = x
        self.y = y
        self.file = file

    @property
    def name(self):
        return f'{self.x},{self.y} {self.w}x{self.h}'

    def get_image(self):
        img, bpp = self.file.get_image()
        if self.w is None or self.h is None:
            self.w, self.h = img.size
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        return img, bpp

    def get_colourkey(self):
        return self.file.colourkey

    def get_image_files(self):
        return (self.file,)

    def get_fingerprint(self):
        return combine_fingerprint(
            super().get_fingerprint_base(),
            x=self.x,
            y=self.y,
        )


class RAWSound(Sound):
    def __init__(self, file):
        super().__init__()
        if not isinstance(file, SoundFile):
            file = SoundFile(file)
        self.file = file

    def get_real_data(self, encoder):
        data = open(self.file.path, 'rb').read()
        name = os.path.basename(self.file.path).encode()
        if len(name) > 256:
            name = name[:251] + '_' + name[-4:]
        return struct.pack('<BBB', 0xff, 0xff, len(name)) + name + b'\0' + data

    def get_fingerprint(self):
        return self.file

    def get_resource_files(self):
        return (self.file,)


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

    def get_resource_files(self):
        return (self.file.value,)
