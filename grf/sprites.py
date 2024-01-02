import math
import os
import struct
import time

import numpy as np
from PIL import Image

from .common import BPP_8, BPP_24, BPP_32, np_make_writable, DummyWriteContext
from .common import ZOOM_NORMAL, ZOOM_4X, ZOOM_2X, ZOOM_OUT_2X, ZOOM_OUT_4X, ZOOM_OUT_8X
from .colour import PALETTE, PIL_PALETTE, ALL_COLOURS, SAFE_COLOURS, WIN_TO_DOS, DEFAULT_BRIGHTNESS, WATER_COLOURS, NP_PALETTE
from .colour import srgb_to_oklab, oklab_blend, oklab_find_best_colour, srgb_find_best_colour, openttd_adjust_brightness
from . import colour


_remap_cache = {}
_FIX_PALETTE_LOOKUP = {PALETTE[i]: i for i in ALL_COLOURS}


def fix_palette(obj, context, img, sprite_name):
    assert (img.mode == 'P')  # TODO
    pal = tuple(img.getpalette())
    if pal == PIL_PALETTE: return img

    context.warning('sprite-palette-conversion', obj, f'Sprite required palette conversion')

    timer = context.start_timer()

    pal_hash = hash(pal)
    remap = _remap_cache.get(pal_hash)
    if remap is None:
        remap = PaletteRemap()
        for i in range(len(pal) // 3):
            colour = (pal[3 * i], pal[3 * i + 1], pal[3 * i + 2])
            res = _FIX_PALETTE_LOOKUP.get(colour)
            if res is None:
                res = srgb_find_best_colour(colour, in_range=ALL_COLOURS)
            remap.remap[i] = res
        _remap_cache[pal_hash] = remap
    res = remap.remap_image(img)
    timer.count_custom('Custom palette conversion')
    return res


def convert_image(image):
    if image.mode == 'P':
        return image, BPP_8
    if image.mode == 'RGBA':
        return image, BPP_32
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return img, BPP_24


# Tries to get the original rgba that rgb and alpha channels were split from, returns None if not possible
def restore_rgba_image(rgb, alpha):
    # We rely on base of both arrays to still be the original array
    base = rgb.base
    if base is not alpha.base:
        return None

    ii = rgb.__array_interface__
    ai = alpha.__array_interface__

    # Check that it starts on the same pixel but skips RGB components
    if ii['data'][0] + 3 != ai['data'][0]:
        return None

    # Check that they have the same strides (offsets to next element)
    if ii['strides'] is None or ii['strides'][:2] != ai['strides'] or ii['strides'][2] != 1:
        return None

    # Check that they have the same shape
    if ii['shape'][:2] != ai['shape'] or ii['shape'][2] != 3:
        return None

    # Check that all other metadata matches
    if ii['descr'] != ai['descr'] or ii['typestr'] != ai['typestr'] or ii['version'] != ai['version']:
        return None

    if base.shape[:2] == alpha.shape:
        return base

    # Sanity check for crop calculation
    if rgb.dtype != np.uint8 or alpha.dtype != np.uint8:
        return None

    sx, sy = alpha.shape
    x, y, z = base.strides
    flat_offset = ii['data'][0] - base.__array_interface__['data'][0]
    ox, oy, oz = flat_offset // x, (flat_offset % x) // y, (flat_offset % y) // z

    return base[ox:ox + sx, oy:oy + sy, oz:oz + 4]


# Pseudo sprite in grf
class Action:
    def get_data(self, context):
        raise NotImplemented


class FakeAction:
    def get_data(self, context):
        return b't'


# Pseudo sprite (reference) + real sprite in grf
class ResourceAction(Action):
    def __init__(self):
        self.sprite_id = None

    def get_data(self, context):
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

    # NOTE: may be removed in future in favor of preparing in constructor
    def prepare_files(self):
        self.resource.prepare_files()


class AlternativeSprites(ResourceAction):
    def __init__(self, *sprites):
        assert all(isinstance(s, Sprite) for s in sprites), sprites
        if len(set((s.zoom, s.bpp) for s in sprites)) != len(sprites):
            sprite_list = ", ".join(f'{s.name}<zoom={s.zoom}, bpp={s.bpp}>' for s in sprites)
            raise ValueError(f'AlternativeSprites need to have different zooms or bpp: {sprite_list}')

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

    # NOTE: may be removed in future in favor of preparing in constructor
    def prepare_files(self):
        for s in self.sprites:
            s.prepare_files()


class ResourceFile:
    def __init__(self, path):
        self.path = path

    def get_fingerprint(self):
        return {'path': str(self.path)}


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

    # NOTE: may be removed in future in favor of preparing in constructor
    def prepare_files(self):
        pass


class Sound(Resource):
    def __init__(self):
        super().__init__()
        self.id = None


class PaletteRemap(Action):
    def __init__(self, ranges=None):
        self.remap = np.arange(256, dtype=np.uint8)
        if ranges:
            self.set_ranges(ranges)

    def get_data(self, context):
        return b'\x00' + self.remap.tobytes()

    @classmethod
    def oklab_from_function(cls, colour_func, remap_water=False):
        res = cls()
        # TODO use numpy vector operations
        for i in SAFE_COLOURS:
            res.remap[i] = oklab_find_best_colour(colour_func(colour.OKLAB_PALETTE[i]))
        if remap_water:
            for i in WATER_COLOURS:
                res.remap[i] = oklab_find_best_colour(colour_func(colour.OKLAB_PALETTE[i]))
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
        res.putpalette(PIL_PALETTE)
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


class MaskMode:
    DEFAULT = 0
    OVERDRAW = 1


class Sprite(Resource):
    def __init__(self, w, h, *, xofs=0, yofs=0, zoom=ZOOM_NORMAL, bpp=None, crop=True, name=None):
        super().__init__()
        self.w = w
        self.h = h
        self.xofs = xofs
        self.yofs = yofs
        self.zoom = zoom
        self.bpp = bpp
        self.crop = crop
        self._name = name

    def __repr__(self):
        return f'{self.__class__.__name__}<{self.name}>'

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return self.default_name

    @property
    def default_name(self):
        return f'{self.w}x{self.h}'

    def get_fingerprint_base(self):
        return combine_fingerprint(
            **{'class': self.__class__.__name__},
            w=self.w,
            h=self.h,
            xofs=self.xofs,
            yofs=self.yofs,
            zoom=self.zoom,
            bpp=self.bpp,
            crop=self.crop,
        )

    def get_fingerprint(self):
        raise NotImplementedError

    def get_image(self):
        raise NotImplementedError

    def draw(self, img):
        raise NotImplementedError

    def get_colourkey(self):
        return None

    def _do_crop(self, context, w, h, rgb, alpha, mask):
        crop_x = crop_y = 0
        if self.crop:
            timer = context.start_timer()
            if alpha is not None:
                cols_bitset = alpha.any(0)
                rows_bitset = alpha.any(1)
            elif rgb is not None:
                cols_bitset = rgb.any((0, 2))
                rows_bitset = rgb.any((1, 2))
            elif mask is not None:
                cols_bitset = mask.any(0)
                rows_bitset = mask.any(1)
            else:
                raise context.log_error('sprite-no-layers', self, 'All data layers are None')

            cols_used = np.arange(w)[cols_bitset]
            rows_used = np.arange(h)[rows_bitset]

            crop_x = min(cols_used, default=0)
            crop_y = min(rows_used, default=0)
            w = max(cols_used, default=0) - crop_x + 1
            h = max(rows_used, default=0) - crop_y + 1

            if rgb is not None:
                rgb = rgb[crop_y: crop_y + h, crop_x: crop_x + w]
            if alpha is not None:
                alpha = alpha[crop_y: crop_y + h, crop_x: crop_x + w]
            if mask is not None:
                mask = mask[crop_y: crop_y + h, crop_x: crop_x + w]

            timer.count_custom('Cropping sprites')

        return crop_x, crop_y, w, h, rgb, alpha, mask

    def _do_get_image(self, context):
        timer = context.start_timer()

        img, bpp = self.get_image()
        w, h = self.w, self.h
        if w is None:
            w = img.size[0]
        if h is None:
            h = img.size[1]

        timer.count_loading()

        if self.bpp is not None and bpp != self.bpp:
            context.log_warning('sprite-bpp-mismatch', self, f'Sprite {self.name} expected {self.bpp}bpp but file is {bpp}bpp, converting.')
            if self.bpp == BPP_24:
                img = img.convert('RGB')
            elif self.bpp == BPP_32:
                img = img.convert('RGBA')
            elif self.bpp == BPP_8:
                p_img = Image.new('P', (16, 16))
                p_img.putpalette(PIL_PALETTE)
                img = img.quantize(palette=p_img, dither=0)
            bpp = self.bpp
        else:
            if bpp == BPP_8:
                img = fix_palette(self, context, img, self.name)

        timer.count_conversion()

        return w, h, img, bpp

    def get_data_layers(self, context):
        w, h, img, bpp = self._do_get_image(context)

        npimg = np.asarray(img)
        if bpp == BPP_32:
            rgb = npimg[:, :, :3]
            alpha = npimg[:, :, 3]
            mask = None
        elif bpp == BPP_24:
            rgb = npimg
            alpha = mask = None
        else:
            assert bpp == BPP_8
            rgb = alpha = None
            mask = npimg

        return w, h, rgb, alpha, mask

    # https://github.com/OpenTTD/grfcodec/blob/master/docs/grf.txt
    def get_real_data(self, context):
        w, h, rgb, alpha, mask = self.get_data_layers(context)

        # It's common to override get_data_layers so check returned layers carefully
        if rgb is not None and rgb.shape != (h, w, 3):
            raise context.log_error('sprite-rgb-shape', self, f'get_data_layers returned RGB layer with wrong shape: {rgb.shape}, expected ({w}, {h}, 3)')
        if alpha is not None and alpha.shape != (h, w):
            raise context.log_error('sprite-alpha-shape', self, f'get_data_layers returned alpha layer with wrong shape: {alpha.shape}, expected ({w}, {h})')
        if mask is not None and mask.shape != (h, w):
            raise context.log_error('sprite-mask-shape', self, f'get_data_layers returned mask layer with wrong shape: {mask.shape}, expected ({w}, {h})')
        crop_x, crop_y, w, h, rgb, alpha, mask = self._do_crop(context, w, h, rgb, alpha, mask)
        xofs, yofs = self.xofs + crop_x, self.yofs + crop_y

        timer = context.start_timer()

        info_byte = 0x40
        stack = []
        bpp = 0
        wh = w * h
        if rgb is not None:
            info_byte |= 0x1
            bpp += 3
            if alpha is None: # if both not None handle separately
                stack.append(rgb.reshape(wh, 3))
        if alpha is not None:
            info_byte |= 0x2
            bpp += 1
            if rgb is None:  # if both not None handle separately
                stack.append(alpha.reshape((wh, 1)))
        if rgb is not None and alpha is not None:
            # Try to find original RGBA array to avoid wasting time on composing it back
            rgba = restore_rgba_image(rgb, alpha)
            if rgba is not None:
                stack.append(rgba.reshape(wh, 4))
            else:
                stack.append(rgb.reshape(wh, 3))
                stack.append(alpha.reshape(wh, 1))
        if mask is not None and (bpp == 0 or np.any(mask)):
            info_byte |= 0x4
            bpp += 1
            stack.append(mask.reshape((wh, 1)))
        if len(stack) > 1:
            raw_data = np.concatenate(stack, axis=1)
        else:
            raw_data = stack[0]
        raw_data = raw_data.reshape(wh * bpp)

        timer.count_composing()

        data = context.sprite_compress(np.ascontiguousarray(raw_data))
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
        return self.get_image_files() + (THIS_FILE,)

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

        # TODO there is also FIX_PALETTE
        PAL_IDX = {PALETTE[i]: i for i in range(256)}

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

        pal = PALETTE[:]
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

    def make_rgba_image(self, context=None):
        w, h, rgb, alpha, mask = self.get_data_layers(DummyWriteContext() if context is None else context)
        img = np.zeros((h, w, 4), dtype=np.uint8)
        if alpha is not None:
            img[:, :, 3] = alpha

        if mask is None:
            img[:, :, :3] = rgb
            if alpha is None:
                img[:, :, 3] = 255
        elif rgb is None:
            has_mask = (mask > 0)
            img[has_mask, :3] = NP_PALETTE[mask[has_mask]]
            img[has_mask, 3] = 255
        else:
            has_mask = (mask > 0)
            img[not has_mask, :3] = rgb
            # TODO use animated palette
            img[has_mask] = (openttd_adjust_brightness(grf.PALETTE[m], max(c)) for c, m in zip(rgb[has_mask], mask[has_mask]))
            if alpha is None:
                img[:, :, 3] = 255
        return Image.fromarray(img, mode='RGBA')


class WithMask(Sprite):
    def __init__(self, sprite, mask, name=None, mode=MaskMode.DEFAULT):
        if mask.bpp is not None and mask.bpp != BPP_8:
            raise ValueError('Mask is not a 8bpp sprite')
        self.sprite = sprite
        self.mask = mask
        self.mode = mode
        super().__init__(
            self.sprite.w,
            self.sprite.h,
            xofs=self.sprite.xofs,
            yofs=self.sprite.yofs,
            zoom=self.sprite.zoom,
            bpp=self.sprite.bpp,
            crop=self.sprite.crop and self.mask.crop,
            name=name,
        )

    def get_data_layers(self, context):
        w, h, rgb, alpha, mask = self.sprite.get_data_layers(context)
        mw, mh, mrgb, malpha, mmask = self.mask.get_data_layers(context)

        timer = context.start_timer()

        if w != mw or h != mh:
            raise context.log_error('with-mask-dimensions-mismatch', self, f'Dimensions don''t match for sprite({w}, {h}) and mask({mw}, {mh})')
        if mrgb is not None:
            raise context.log_error('with-mask-has-rgb', self, 'Mask has an RGB layer')
        if malpha is not None:
            raise context.log_error('with-mask-has-alpha', self, 'Mask has an alpha layer')

        has_mask = (mmask != 0)
        if mask is not None:
            mask = np_make_writable(mask)
            mask[has_mask] = mmask[has_mask]
        else:
            mask = mmask

        if self.mode == MaskMode.OVERDRAW:
            if rgb is not None:
                rgb = np_make_writable(rgb)
                rgb[has_mask] = (DEFAULT_BRIGHTNESS, DEFAULT_BRIGHTNESS, DEFAULT_BRIGHTNESS)
            if alpha is not None:
                alpha = np_make_writable(alpha)
                alpha[has_mask] = 255

        timer.count_composing()

        return w, h, rgb, alpha, mask

    def get_image_files(self):
        return ()

    def get_resource_files(self):
        return super().get_resource_files() + (THIS_FILE,) + self.sprite.get_resource_files() + self.mask.get_resource_files()

    def get_fingerprint(self):
        return {
            'class': self.__class__.__name__,
            'sprite': self.sprite.get_fingerprint(),
            'mask': self.mask.get_fingerprint(),
            'mode': self.mode,
        }


class EmptySprite(Sprite):
    def __init__(self):
        super().__init__(1, 1)

    def draw(self, img):
        pass

    def get_real_data(self, context):
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
    def __init__(self, image, **kw):
        self._image = convert_image(image)
        super().__init__(*self._image[0].size, bpp=self._image[1], **kw)

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

    def prepare(self, **kw):
        pass

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
    def __init__(self, file, x, y, w, h, *, xofs=0, yofs=0, zoom=ZOOM_NORMAL, bpp=None, crop=True, name=None, **kw):
        assert(isinstance(file, ImageFile))
        super().__init__(w, h, xofs=xofs, yofs=yofs, bpp=bpp, zoom=zoom, crop=crop, name=name)
        self.x = x
        self.y = y
        self.file = file
        self.kw = kw

    @property
    def default_name(self):
        return f'{self.x},{self.y} {self.w}x{self.h}'

    def prepare_files(self):
        self.file.prepare(**self.kw)

    def get_image(self):
        img, bpp = self.file.get_image(**self.kw)
        if self.w is None or self.h is None and self.x == 0 and self.y == 0:
            self.w, self.h = img.size
        if self.x < 0 or self.y < 0 or self.x + self.w > img.size[0] or self.y + self.h > img.size[1]:
            raise RuntimeError(f"Sprite {self.name} area ({self.x}..{self.x + self.w}, {self.y}..{self.y + self.h}) is outside image borders (0..{img.size[0]}, 0..{img.size[1]})")
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        return img, bpp

    def get_data_layers(self, context):
        w, h, rgb, alpha, mask = super().get_data_layers(context)
        if self.file.colourkey is not None:
            is_key = np.all(np.equal(rgb, self.file.colourkey), axis=2)
            if np.any(is_key):
                if alpha is None:
                    alpha = np.full((h, w), 255, dtype=np.uint8)
                else:
                    alpha = np_make_writable(alpha)
                alpha[is_key] = 0
        return w, h, rgb, alpha, mask

    def get_image_files(self):
        return (self.file,)

    def get_fingerprint(self):
        return combine_fingerprint(
            super().get_fingerprint_base(),
            colourkey=None if self.file.colourkey is None else tuple(self.file.colourkey),
            x=self.x,
            y=self.y,
            kw=self.kw,
        )


class RAWSound(Sound):
    def __init__(self, file):
        super().__init__()
        if not isinstance(file, SoundFile):
            file = SoundFile(file)
        self.file = file

    def get_real_data(self, context):
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


class ZoomDebugRecolourSprite(Sprite):

    ZOOM_DEBUG_COLOURS = {
        ZOOM_4X: np.array((0, 1, 0), dtype=np.uint8),
        ZOOM_2X: np.array((0, 0, 1), dtype=np.uint8),
        ZOOM_NORMAL: np.array((1, 0, 0), dtype=np.uint8),
        ZOOM_OUT_2X: np.array((0, 1, 1), dtype=np.uint8),
        ZOOM_OUT_4X: np.array((1, 1, 0), dtype=np.uint8),
        ZOOM_OUT_8X: np.array((1, 0, 1), dtype=np.uint8),
    }

    _ZOOM_DEBUG_RECOLOURS = {}

    @classmethod
    def _get_debug_recolour(cls, zoom):
        res = cls._ZOOM_DEBUG_RECOLOURS.get(zoom)
        if res is None:
            from .colour import OKLAB_PALETTE  # lazy constant, don't import on top level
            c = cls.ZOOM_DEBUG_COLOURS[zoom]
            blended = oklab_blend(OKLAB_PALETTE, srgb_to_oklab(c * 255), ratio=.8)
            res = np.array([0] + oklab_find_best_colour(blended[1:]))
            cls._ZOOM_DEBUG_RECOLOURS[zoom] = res.astype(np.uint8)
        return res

    def __init__(self, sprite):
        self.sprite = sprite
        super().__init__(w=sprite.w, h=sprite.h, xofs=sprite.xofs, yofs=sprite.yofs, zoom=sprite.zoom, bpp=sprite.bpp)

    def get_data_layers(self, context):
        w, h, ni, na, nm = self.sprite.get_data_layers(context)

        timer = context.start_timer()

        if ni is not None:
            ni = np_make_writable(ni)
            ni *= self.ZOOM_DEBUG_COLOURS[self.sprite.zoom]
        if nm is not None:
            nm = self._get_debug_recolour(self.sprite.zoom)[nm]

        timer.count_custom('Zoom level debug recolouring')

        return w, h, ni, na, nm

    def get_image_files(self):
        return ()

    def get_resource_files(self):
        return super().get_resource_files() + (THIS_FILE, ) + self.sprite.get_resource_files()

    def get_fingerprint(self):
        return {
            'class': self.__class__.__name__,
            'sprite': self.sprite.get_fingerprint(),
        }
