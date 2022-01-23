import functools
import math
import textwrap
import pprint
import datetime
import heapq
from collections import defaultdict

from PIL import Image, ImageDraw
from nml.spriteencoder import SpriteEncoder
import spectra
import struct
import numpy as np

from .parser import Node, Expr, Value, Var, Temp, Perm, Call, parse_code, OP_INIT, SPRITE_FLAGS, GenericVar
from .common import Feature, hex_str, utoi32
from .common import TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, CANAL, BRIDGE, HOUSE, INDUSTRY_TILE, INDUSTRY, \
                    CARGO, SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE


to_spectra = lambda r, g, b: spectra.rgb(float(r) / 255., float(g) / 255., float(b) / 255.)
# working with DOS palette only
PALETTE = (0, 0, 255, 16, 16, 16, 32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80, 100, 100, 100, 116, 116, 116, 132, 132, 132, 148, 148, 148, 168, 168, 168, 184, 184, 184, 200, 200, 200, 216, 216, 216, 232, 232, 232, 252, 252, 252, 52, 60, 72, 68, 76, 92, 88, 96, 112, 108, 116, 132, 132, 140, 152, 156, 160, 172, 176, 184, 196, 204, 208, 220, 48, 44, 4, 64, 60, 12, 80, 76, 20, 96, 92, 28, 120, 120, 64, 148, 148, 100, 176, 176, 132, 204, 204, 168, 72, 44, 4, 88, 60, 20, 104, 80, 44, 124, 104, 72, 152, 132, 92, 184, 160, 120, 212, 188, 148, 244, 220, 176, 64, 0, 4, 88, 4, 16, 112, 16, 32, 136, 32, 52, 160, 56, 76, 188, 84, 108, 204, 104, 124, 220, 132, 144, 236, 156, 164, 252, 188, 192, 252, 208, 0, 252, 232, 60, 252, 252, 128, 76, 40, 0, 96, 60, 8, 116, 88, 28, 136, 116, 56, 156, 136, 80, 176, 156, 108, 196, 180, 136, 68, 24, 0, 96, 44, 4, 128, 68, 8, 156, 96, 16, 184, 120, 24, 212, 156, 32, 232, 184, 16, 252, 212, 0, 252, 248, 128, 252, 252, 192, 32, 4, 0, 64, 20, 8, 84, 28, 16, 108, 44, 28, 128, 56, 40, 148, 72, 56, 168, 92, 76, 184, 108, 88, 196, 128, 108, 212, 148, 128, 8, 52, 0, 16, 64, 0, 32, 80, 4, 48, 96, 4, 64, 112, 12, 84, 132, 20, 104, 148, 28, 128, 168, 44, 28, 52, 24, 44, 68, 32, 60, 88, 48, 80, 104, 60, 104, 124, 76, 128, 148, 92, 152, 176, 108, 180, 204, 124, 16, 52, 24, 32, 72, 44, 56, 96, 72, 76, 116, 88, 96, 136, 108, 120, 164, 136, 152, 192, 168, 184, 220, 200, 32, 24, 0, 56, 28, 0, 72, 40, 4, 88, 52, 12, 104, 64, 24, 124, 84, 44, 140, 108, 64, 160, 128, 88, 76, 40, 16, 96, 52, 24, 116, 68, 40, 136, 84, 56, 164, 96, 64, 184, 112, 80, 204, 128, 96, 212, 148, 112, 224, 168, 128, 236, 188, 148, 80, 28, 4, 100, 40, 20, 120, 56, 40, 140, 76, 64, 160, 100, 96, 184, 136, 136, 36, 40, 68, 48, 52, 84, 64, 64, 100, 80, 80, 116, 100, 100, 136, 132, 132, 164, 172, 172, 192, 212, 212, 224, 40, 20, 112, 64, 44, 144, 88, 64, 172, 104, 76, 196, 120, 88, 224, 140, 104, 252, 160, 136, 252, 188, 168, 252, 0, 24, 108, 0, 36, 132, 0, 52, 160, 0, 72, 184, 0, 96, 212, 24, 120, 220, 56, 144, 232, 88, 168, 240, 128, 196, 252, 188, 224, 252, 16, 64, 96, 24, 80, 108, 40, 96, 120, 52, 112, 132, 80, 140, 160, 116, 172, 192, 156, 204, 220, 204, 240, 252, 172, 52, 52, 212, 52, 52, 252, 52, 52, 252, 100, 88, 252, 144, 124, 252, 184, 160, 252, 216, 200, 252, 244, 236, 72, 20, 112, 92, 44, 140, 112, 68, 168, 140, 100, 196, 168, 136, 224, 200, 176, 248, 208, 184, 255, 232, 208, 252, 60, 0, 0, 92, 0, 0, 128, 0, 0, 160, 0, 0, 196, 0, 0, 224, 0, 0, 252, 0, 0, 252, 80, 0, 252, 108, 0, 252, 136, 0, 252, 164, 0, 252, 192, 0, 252, 220, 0, 252, 252, 0, 204, 136, 8, 228, 144, 4, 252, 156, 0, 252, 176, 48, 252, 196, 100, 252, 216, 152, 8, 24, 88, 12, 36, 104, 20, 52, 124, 28, 68, 140, 40, 92, 164, 56, 120, 188, 72, 152, 216, 100, 172, 224, 92, 156, 52, 108, 176, 64, 124, 200, 76, 144, 224, 92, 224, 244, 252, 200, 236, 248, 180, 220, 236, 132, 188, 216, 88, 152, 172, 244, 0, 244, 245, 0, 245, 246, 0, 246, 247, 0, 247, 248, 0, 248, 249, 0, 249, 250, 0, 250, 251, 0, 251, 252, 0, 252, 253, 0, 253, 254, 0, 254, 255, 0, 255, 76, 24, 8, 108, 44, 24, 144, 72, 52, 176, 108, 84, 210, 146, 126, 252, 60, 0, 252, 84, 0, 252, 104, 0, 252, 124, 0, 252, 148, 0, 252, 172, 0, 252, 196, 0, 64, 0, 0, 255, 0, 0, 48, 48, 0, 64, 64, 0, 80, 80, 0, 255, 255, 0, 32, 68, 112, 36, 72, 116, 40, 76, 120, 44, 80, 124, 48, 84, 128, 72, 100, 144, 100, 132, 168, 216, 244, 252, 96, 128, 164, 68, 96, 140, 255, 255, 255)
SAFE_COLORS = set(range(1, 0xD7))
ALL_COLORS = set(range(256))
SPECTRA_PALETTE = {i:to_spectra(PALETTE[i * 3], PALETTE[i * 3 + 1], PALETTE[i * 3 + 2]) for i in range(256)}
WATER_COLORS = set(range(0xF5, 0xFF))

# ZOOM_OUT_4X, ZOOM_NORMAL, ZOOM_OUT_2X, ZOOM_OUT_8X, ZOOM_OUT_16X, ZOOM_OUT_32X = range(6)
ZOOM_4X, ZOOM_NORMAL, ZOOM_2X, ZOOM_8X, ZOOM_16X, ZOOM_32X = range(6)
BPP_8, BPP_24, BPP_32 = 8, 24, 32

TEMPERATE, ARCTIC, TROPICAL, TOYLAND = 1, 2, 4, 8
ALL_CLIMATES = TEMPERATE | ARCTIC | TROPICAL | TOYLAND

af_ZA = 0x1b  # afrikaans,True,Afrikaans,Afrikaans,0,male,
ar_EG = 0x14  # arabic_egypt,True,Arabic (Egypt),Arabic (Egypt),1,,
be_BY = 0x10  # belarusian,True,Belarusian,Беларуская,6,n m f p,n nom m abl gen pre p dat acc f
bg_BG = 0x18  # bulgarian,True,Bulgarian,Български,0,n m f p,n m f p
ca_ES = 0x22  # catalan,True,Catalan,Català,0,Femenin Masculin,
cs_CZ = 0x15  # czech,True,Czech,Čeština,10,n m fp mnp map np f,small ins nom big loc gen voc dat acc
cv_RU = 0x0b  # chuvash,True,Chuvash,Чӑвашла,0,,
cy_GB = 0x0f  # welsh,True,Welsh,Cymraeg,0,,
da_DK = 0x2d  # danish,True,Danish,Dansk,0,,
de_DE = 0x02  # german,True,German,Deutsch,0,n m w p,
el_GR = 0x1e  # greek,True,Greek,Ελληνικά,2,n m f,geniki date subs
en_AU = 0x3d  # english_AU,True,English (AU),English (AU),0,,
en_GB = 0x01  # english,True,English (UK),English (UK),0,,
en_US = 0x00  # english_US,True,English (US),English (US),0,,
eo_EO = 0x05  # esperanto,True,Esperanto,Esperanto,0,,n
es_ES = 0x04  # spanish,True,Spanish,Español (ES),0,m f,
es_MX = 0x55  # spanish_MX,True,Spanish (Mexican),Español (MX),0,m f,
et_EE = 0x34  # estonian,True,Estonian,Eesti keel,0,,in sü g
eu_ES = 0x21  # basque,True,Basque,Euskara,0,,
fa_IR = 0x62  # persian,True,Persian,فارسی,0,,
fi_FI = 0x35  # finnish,True,Finnish,Suomi,0,,
fo_FO = 0x12  # faroese,True,Faroese,Føroyskt,0,n m f,
fr_FR = 0x03  # french,True,French,Français,2,m2 m f,
fy_NL = 0x32  # frisian,True,Frisian,Frysk,0,,
ga_IE = 0x08  # irish,True,Irish,Gaeilge,4,,
gd_GB = 0x13  # gaelic,True,Scottish Gaelic,Gàidhlig,13,m f,dat nom voc gen
gl_ES = 0x31  # galician,True,Galician,Galego,0,n m f,
he_IL = 0x61  # hebrew,True,Hebrew,עברית,0,m f,gen plural singular
hi_IN = 0x17  # hindi,True,Hindi,हिन्दी,0,,
hr_HR = 0x38  # croatian,True,Croatian,Hrvatski,6,female male middle,lok ins nom aku gen dat vok
hu_HU = 0x24  # hungarian,True,Hungarian,Magyar,2,,ba t
id_ID = 0x5a  # indonesian,True,Indonesian,Bahasa Indonesia,1,,
io_IO = 0x06  # ido,True,Ido,Ido,0,,
is_IS = 0x29  # icelandic,True,Icelandic,Íslenska,0,kvenkyn karlkyn hvorugkyn,
it_IT = 0x27  # italian,True,Italian,Italiano,0,m ma f,fp fs ms mp
ja_JP = 0x39  # japanese,True,Japanese,日本語,1,,
ko_KR = 0x3a  # korean,True,Korean,한국어,11,m f,
la_VA = 0x66  # latin,True,Latin,Latina,0,n m fp mp np f,dat abl acc gen
lb_LU = 0x23  # luxembourgish,True,Luxembourgish,Lëtzebuergesch,0,,
lt_LT = 0x2b  # lithuanian,True,Lithuanian,Lietuvių,5,mot vyr,kam ka ko kur kreip kas kuo
lv_LV = 0x2a  # latvian,True,Latvian,Latviešu,3,m f,kas
mk_MK = 0x26  # macedonian,True,Macedonian,Македонски,0,,
mr_IN = 0x11  # marathi,True,Marathi,मराठी,0,,
ms_MY = 0x3c  # malay,True,Malay,Melayu,0,,
mt_MT = 0x09  # maltese,True,Maltese,Malti,12,,
nb_NO = 0x2f  # norwegian_bokmal,True,Norwegian (Bokmal),Norsk (bokmål),0,masculine neuter feminine,small
nl_NL = 0x1f  # dutch,True,Dutch,Nederlands,0,,
nn_NO = 0x0e  # norwegian_nynorsk,True,Norwegian (Nynorsk),Norsk (nynorsk),0,masculine neuter feminine,small
pl_PL = 0x30  # polish,True,Polish,Polski,7,n m f,n m w c d b
pt_BR = 0x37  # brazilian_portuguese,True,Portuguese (Brazilian),Português (BR),2,m f,
pt_PT = 0x36  # portuguese,True,Portuguese,Português,0,n m fp mp f,
ro_RO = 0x28  # romanian,True,Romanian,Română,14,,
ru_RU = 0x07  # russian,True,Russian,Русский,6,n m f p,n nom m abl gen pre p dat acc f
sk_SK = 0x16  # slovak,True,Slovak,Slovenčina,10,m z s,g
sl_SI = 0x2c  # slovenian,True,Slovenian,Slovenščina,8,,t d r
sr_RS = 0x0d  # serbian,True,Serbian,Srpski,6,srednji ženski muški,lok ins nom aku big gen dat vok
sv_SE = 0x2e  # swedish,True,Swedish,Svenska,0,,
ta_IN = 0x0a  # tamil,True,Tamil,தமிழ்,0,,
th_TH = 0x42  # thai,True,Thai,Thai,1,,
tr_TR = 0x3e  # turkish,True,Turkish,Türkçe,1,,tamlanan
uk_UA = 0x33  # ukrainian,True,Ukrainian,Українська,6,m mn s f,z d r
ur_PK = 0x5c  # urdu,True,Urdu,Urdu,0,m f,
vi_VN = 0x54  # vietnamese,True,Vietnamese,Tiếng Việt,1,,
zh_CN = 0x56  # simplified_chinese,True,Chinese (Simplified),简体中文,1,,
zh_TW = 0x0c  # traditional_chinese,True,Chinese (Traditional),繁體中文,1,,
ANY_LANGUAGE = 0x7f


VEHICLE_FEATURES = (TRAIN, RV, SHIP, AIRCRAFT)

def is_leap_year(year):
    return yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)


def date_to_days(d):
    return (d - datetime.date(1, 1, 1)).days + 366


def color_distance(c1, c2):
    rmean = (c1.rgb[0] + c2.rgb[0]) / 2.
    r = c1.rgb[0] - c2.rgb[0]
    g = c1.rgb[1] - c2.rgb[1]
    b = c1.rgb[2] - c2.rgb[2]
    return math.sqrt(
        ((2 + rmean) * r * r) +
        4 * g * g +
        (3 - rmean) * b * b)


def find_best_color(x, in_range=SAFE_COLORS):
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
    for i in ALL_COLORS:
        remap.remap[i] = find_best_color(to_spectra(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]), in_range=ALL_COLORS)
    return remap.remap_image(img)


def open_image(filename, *args, **kw):
    return fix_palette(Image.open(filename, *args, **kw))


def pformat(data, indent=4, indent_first=None):
    INDENTATION = ' '
    if indent_first is None:
        indent_first = indent
    res = textwrap.indent(pprint.pformat(data, compact=True, sort_dicts=False), INDENTATION * indent)
    if indent_first != indent:
        res = res.lstrip() + INDENTATION * indent_first
    return res

# def map_rgb_image(self, im):
#     assert im.mode == 'RGB', im.mode
#     data = np.array(im)

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


class PaletteRemap(BaseSprite):
    def __init__(self, ranges=None):
        self.remap = np.arange(256, dtype=np.uint8)
        if ranges:
            self.set_ranges(ranges)

    def get_data(self):
        return b'\x00' + self.remap.tobytes()

    def get_data_size(self):
        return 257

    @classmethod
    def from_function(cls, color_func, remap_water=False):
        res = cls()
        for i in SAFE_COLORS:
            res.remap[i] = find_best_color(color_func(SPECTRA_PALETTE[i]))
        if remap_water:
            for i in WATER_COLORS:
                res.remap[i] = find_best_color(color_func(SPECTRA_PALETTE[i]))
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


class RealSprite(BaseSprite):
    def __init__(self, w, h, *, xofs=0, yofs=0, zoom=ZOOM_4X):
        self.sprite_id = None
        self.w = w
        self.h = h
        self.xofs = xofs
        self.yofs = yofs
        self.zoom = zoom

    def get_data_size(self):
        return 4

    def get_data(self):
        return struct.pack('<I', self.sprite_id)

    def get_real_data(self, encoder):
        raise NotImplementedError

    def draw(self, img):
        raise NotImplementedError


class ImageFile:
    def __init__(self, path, colourkey=None):
        self.path = path
        self.colourkey = colourkey
        self._image = None

    def get_image(self):
        if self._image:
            return self._image
        img = Image.open(self.path)
        # TODO alpha channel
        if img.mode == 'P':
            self._image = (img, BPP_8)
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            self._image = (img, BPP_24)
        return self._image


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


class FileSprite(RealSprite):
    def __init__(self, file, x, y, w, h, *, bpp=None, mask=None, **kw):
        if bpp == BPP_8 and mask is not None:
            raise ValueError("8bpp sprites can't have a mask")
        assert(isinstance(file, ImageFile))
        super().__init__(w, h, **kw)
        self.file = file
        self.bpp = bpp
        self.mask = mask
        self.x = x
        self.y = y

    # def get_real_data(self, encoder):
    #     ns = NMLFileSpriteWrapper(self, self.file, self.bit_depth)
    #     (size_x, size_y, xoffset, yoffset, compressed_data, info_byte, crop_rect, pixel_stats) = \
    #         encoder.encode_sprite(ns)
    #     return struct.pack(
    #         '<IIBBHHhh',
    #         self.sprite_id,
    #         len(compressed_data) + 10, info_byte,
    #         self.zoom,
    #         size_y,
    #         size_x,
    #         xoffset,
    #         yoffset,
    #     ) + compressed_data

    # https://github.com/OpenTTD/grfcodec/blob/master/docs/grf.txt
    def get_real_data(self, encoder):
        img, bpp = self.file.get_image()
        if self.bpp is None:
            self.bpp = bpp
        img = img.crop((self.x, self.y, self.x + self.w, self.y + self.h))
        npalpha = None

        if bpp != self.bpp:
            print(f'Sprite {self.file.path} {self.x},{self.y} {self.w}x{self.h} expected {self.bpp}bpp but file is {bpp}bpp, converting.')
            if bpp == BPP_8:
                img = img.convert('RGB')
            else:
                img = img.convert('P', palette=PALETTE)
        else:
            if bpp == BPP_8:
                img = fix_palette(img, '{self.file.path} {self.x},{self.y} {self.w}x{self.h}')

        npimg = np.asarray(img)
        if self.file.colourkey is not None:
            if self.bpp == BPP_24 and self.file.colourkey is not None:
                npalpha = 255 * np.all(np.not_equal(npimg, self.file.colourkey), axis=2)
                npalpha = npalpha.astype(np.uint8, copy=False)
            else:
                print(f'Colour key on 8bpp sprites is not supported')

        info_byte = 0x40
        if self.bpp == BPP_24:
            npimg.shape = (self.w * self.h, 3)
            stack = [npimg]
            info_byte |= 0x1  # rgb
            rbpp = 3

            if npalpha is not None:
                npalpha.shape = (self.w * self.h, 1)
                stack.append(npalpha)
                rbpp += 1
                info_byte |= 0x2  # alpha

            if self.mask is not None:
                mask_file, xofs, yofs = self.mask
                mask_img, mask_bpp = mask_file.get_image()
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


class SpriteSheet:
    def __init__(self, sprites=None):
        self._sprites = list(sprites) if sprites else []

    def make_image(self, filename, padding=5, columns=10):
        w, h = 0, padding
        lineofs = []
        for i in range(0, len(self._sprites), columns):
            w = max(w, sum(s.w for s in self._sprites[i: i + columns]))
            lineofs.append(h)
            h += padding + max((s.h for s in self._sprites[i: i + columns]), default=0)

        w += (columns + 1) * padding
        im = Image.new('L', (w, h), color=0xff)
        im.putpalette(PALETTE)

        x = 0
        for i, s in enumerate(self._sprites):
            y = lineofs[i // columns]
            if i % columns == 0:
                x = padding
            s.x = x
            s.y = y
            s.file = filename
            s.draw(im)
            x += s.w + padding

        im.save(filename)


    def write_nml(self, file):
        for s in self._sprites:
            file.write(s.get_nml())
            file.write('\n')
        file.write('\n')


class DummySprite(BaseSprite):
    def get_data(self):
        return b'\x00'

    def get_data_size(self):
        return 1


class If(LazyBaseSprite):
    def __init__(self, is_static, variable, varsize, condition, value, skip):
        self.is_static = is_static
        self.variable = variable
        self.varsize = varsize
        self.condition = condition
        self.value = value
        self.skip = skip

    def _encode(self):
        res = bytes((0x09 if self.is_static else 0x07, self.variable, self.varsize, self.condition))
        for i in range(self.varsize):
            res += bytes(((self.value >> (8 * i)) & 0xff,))
        res += bytes((self.skip,))

    def py(self):
        return f'''
            If(
                is_static={self.is_static},
                variable={self.variable},
                varsize={self.varsize},
                condition={self.condition},
                value={self.value},
                skip={self.skip},
            )'''


# Action 8
class SetDescription(BaseSprite):
    def __init__(self, version, grfid, name, description):
        assert isinstance(grfid, bytes)
        assert isinstance(name, (bytes, str))
        assert isinstance(description, (bytes, str))

        if isinstance(name, str):
            name = name.encode('utf-8')
        if isinstance(description, str):
            description = description.encode('utf-8')

        self.version = version
        self.grfid = grfid
        self.name = name
        self.description = description
        self._data = bytes((0x08, self.version))
        self._data += self.grfid + self.name + b'\x00' + self.description + b'\x00'

    def get_data(self):
        return self._data

    def get_data_size(self):
        return len(self._data)

    def py(self):
        return f'''
        SetDescription(
            version={self.version},
            grfid={self.grfid},
            name={self.name},
            description={self.description},
        )'''


# Action 0x10
class Label(LazyBaseSprite):
    def __init__(self, label, comment):
        self.label = label
        self.comment = comment

    def _encode(self):
        return bytes((0x10, self.label)) + data

    def py(self):
        return f'Label({self.label}, {self.comment!r})'


Action10 = Label


# Action 14
class InformationSprite(BaseSprite):
    def __init__(self, palette):  # TODO everything else
        # self.palette = {'D': b'\x00', 'W': b'\x01', 'A': b'\x02'}[palette]
        self.palette = palette.encode('utf-8')
        self._data = b'\x14CINFOBPALS\x01\x00' + self.palette + b'\x00\x00'

    def get_data(self):
        return self._data

    def get_data_size(self):
        return len(self._data)


# Action 14 - take 2
class SetProperties(BaseSprite):
    def __init__(self, props):
        self.palette = 'D'.encode('utf-8')
        self._data = b'\x14CINFOBPALS\x01\x00' + self.palette + b'\x00\x00'
        self.props = props

    def get_data(self):
        # raise NotImplementedError
        return self._data

    def get_data_size(self):
        # raise NotImplementedError
        return len(self._data)

    def py(self):
        return f'SetProperties(\n' + pformat(self.props) + '\n)'


# Action A
class ReplaceOldSprites(BaseSprite):
    def __init__(self, sets):
        assert isinstance(sets, (list, tuple))
        assert len(sets) <= 0xff
        for first, num in sets:
            assert isinstance(first, int)
            assert isinstance(num, int)

        super().__init__()
        self.sets = sets

    def get_data(self):
        return bytes((0xa, len(self.sets))) + b''.join(struct.pack('<BH', num, first) for first, num in self.sets)

    def get_data_size(self):
        return 2 + 3 * len(self.sets)

    def py(self):
        return f'ReplaceOldSprites(sets={self.sets!r})'


class Comment(LazyBaseSprite):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _encode(self):
        return bytes((0x0C)) + self.data

    def py(self):
        return f'Comment({self.data!r})'


ActionC = Comment


# Action 5
class ReplaceNewSprites(LazyBaseSprite):
    def __init__(self, set_type, num, *, offset=None):
        assert isinstance(set_type, int)
        assert isinstance(num, int)

        super().__init__()
        self.set_type = set_type
        self.num = num
        self.offset = offset

    def _encode(self):
        if self.offset is None:
            return bytes((0x5,)) + struct.pack('<BBH', self.set_type, 0xff, self.num)
        return bytes((0x5,)) + struct.pack('<BBHBH', self.set_type | 0xf0, 0xff, self.num, 0xff, self.offset)

    def py(self):
        return f'ReplaceNewSprites(set_type={self.set_type}, num={self.num}, offset={self.offset})'


class SpriteGenerator:
    def get_sprites(self):
        return NotImplementedError


ACTION0_COMMON_VEHICLE_PROPS = {
    0x00: ('introduction_days_since_1920', 'W', 'deprecated by introduction_date'),  # date of introduction as the amount of days since 1920
    0x02: ('reliability_decay', 'B'),  # reliability decay speed     no
    0x03: ('vehicle_life', 'B'),  # vehicle life in years   no
    0x04: ('model_life', 'B'),  # model life in years     no
    0x06: ('climates_available', 'B'),  # climate availability    should be zero
    0x07: ('loading_speed', 'B'),  # loading speed   yes
}


ACTION0_TRAIN_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x05: ('railtype', 'B'),  # Track type (see below)  should be same as front
    0x08: ('ai_flag', 'B'),  # AI special flag: set to 1 if engine is 'optimized' for passenger service (AI won't use it for other cargo), 0 otherwise     no
    0x09: ('speed', 'W'),  # Speed in mph*1.6 (see below)    no
    0x0B: ('power', 'W'),  # Power (0 for wagons)    should be zero
    0x0D: ('running_cost_factor', 'B'),  # Running cost factor (0 for wagons)  should be zero
    0x0E: ('running_cost', 'D'),  # Running cost base, see below    should be zero
    0x12: ('sprite_id', 'B'),  # Sprite ID (FD for new graphics)     yes
    0x13: ('is_dual_headed', 'B'),  # Dual-headed flag; 1 if dual-headed engine, 0 otherwise  should be zero also for front
    0x14: ('cargo_capacity', 'B'),  # Cargo capacity  yes
    0x15: ('default_cargo_type', 'B'),  # Cargo type, see CargoTypes
    0x16: ('weight', 'B'),  # Weight in tons  should be zero
    0x17: ('cost_factor', 'B'),  # Cost factor     should be zero
    0x18: ('engine_rank', 'B'),  # Engine rank for the AI (AI selects the highest-rank engine of those it can buy)     no
    0x19: ('engine_traction', 'B'),  # Engine traction type (see below)    no
    0x1A: ('sort_purchase_list', 'B*'), # Not a property, but an action: sort the purchase list.  no
    0x1B: ('wagon_power', 'W'),  # Power added by each wagon connected to this engine, see below   should be zero
    0x1C: ('refit_cost', 'B'),  # Refit cost, using 50% of the purchase price cost base   yes
    0x1D: ('refit_cargoes', 'D'),  # Bit mask of cargo types available for refitting, see column 2 (bit value) in CargoTypes     yes
    0x1E: ('cb_flags', 'B'),  # Callback flags bit mask, see below  yes
    0x1F: ('tractive_effort', 'B'),  # Coefficient of tractive effort  should be zero
    0x20: ('air_drag', 'B'),  # Coefficient of air drag     should be zero
    0x21: ('shorten_by', 'B'),  # Make vehicle shorter by this amount, see below  yes
    0x22: ('visual_effect', 'B'),  # Set visual effect type (steam/smoke/sparks) as well as position, see below  yes
    0x23: ('powered_weight', 'B'),  # Set how much weight is added by making wagons powered (i.e. weight of engine), see below    should be zero
    0x24: ('weight_hi', 'B'),  # High byte of vehicle weight, weight will be prop.24*256+prop.16     should be zero
    0x25: ('bitmask', 'B'),  # User-defined bit mask to set when checking veh. var. 42     yes
    0x26: ('retire_early', 'B'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)    no
    0x27: ('flags', 'B'),  # Miscellaneous flags     partly
    0x28: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes    yes
    0x29: ('non_refit_classes', 'W'),  # Non-refittable cargo classes    yes
    0x2A: ('non_refittable_cargo_classes', 'D'),  # Long format introduction date   no
    0x2B: ('cargo_age_period', 'W'),  # period  yes
    0x2C: ('cargo_allow_refit', 'n*B'), # refittable cargo types   yes
    0x2D: ('cargo_disallow_refit', 'n*B'),  # refittable cargo types    yes
    0x2E: ('speed_mod', 'W'),  # speed modifier    yes
}

ACTION0_RV_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x05: ('road_type', 'B'),  # Roadtype / tramtype (see below)     should be same as front
    0x08: ('precise_max_speed', 'B'),  # Speed in mph*3.2    no
    0x09: ('running_cost_factor', 'B'),  # Running cost factor     should be zero
    0x0A: ('running_cost_base', 'D'),  # Running cost base, see below    should be zero
    0x0E: ('sprite_id', 'B'),  # Sprite ID (FF for new graphics)     yes
    0x0F: ('cargo_capacity', 'B'),  # Capacity    yes
    0x10: ('default_cargo_type', 'B'),  # Cargo type, see CargoTypes
    0x11: ('cost_factor', 'B'),  # Cost factor     should be zero
    0x12: ('sound_effect', 'B'),  # Sound effect: 17/19/1A for regular, 3C/3E for toyland
    0x13: ('power', 'B'),  # Power in 10 hp, see below   should be zero
    0x14: ('weight', 'B'),  # Weight in 1/4 tons, see below   should be zero
    0x15: ('max_speed', 'B'),  # Speed in mph*0.8, see below     no
    0x16: ('refit_mask', 'D'),  # Bit mask of cargo types available for refitting (not refittable if 0 or unset), see column 2 (bit values) in CargoTypes     yes
    0x17: ('cb_flags', 'B'),  # Callback flags bit mask, see below  yes
    0x18: ('tractive_effort_coefficient', 'B'),  # Coefficient of tractive effort  should be zero
    0x19: ('air_drag_coefficient', 'B'),  # Coefficient of air drag     should be zero
    0x1A: ('refit_cost', 'B'),  # Refit cost, using 25% of the purchase price cost base   yes
    0x1B: ('retire_early', 'B'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)    no
    0x1C: ('misc_flags', 'B'),  # Miscellaneous vehicle flags     partly ("tram" should be same as front)
    0x1D: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes, see train prop. 28    yes
    0x1E: ('non_refittable_cargo_classes', 'W'),  # Non-refittable cargo classes, see train prop. 29    yes
    0x1F: ('introduction_date', 'D'),  # Long format introduction date   no
    0x20: ('sort_purchase_list', 'B*'), # Sort the purchase list  no
    0x21: ('visual_effect', 'B'),  # Visual effect   yes
    0x22: ('cargo_age_period', 'W'),  # Custom cargo ageing period  yes
    0x23: ('shorten_by', 'B'),  # Make vehicle shorter, see train property 21     yes
    0x24: ('cargo_allow_refit', 'B n*B'),  # List of always refittable cargo types, see train property 2C    yes
    0x25: ('cargo_disallow_refit', 'B n*B'),  # List of never refittable cargo types, see train property 2D     yes
}

ACTION0_SHIP_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x08: ('sprite_id', 'B'),  # Sprite (FF for new graphics)
    0x09: ('is_refittable', 'B'),  # Refittable (0 no, 1 yes)
    0x0A: ('cost_factor', 'B'),  # Cost factor
    0x0B: ('speed', 'B'),  # Speed in mph*3.2
    0x0C: ('cargo_type', 'B'),  # Cargo type, see CargoTypes
    0x0D: ('capacity', 'W'),  # Capacity
    0x0F: ('running_cost_factor', 'B'),  # Running cost factor
    0x10: ('sound', 'B'),  # Sound effect type (4=cargo ship, 5=passenger ship)
    0x11: ('refit_cargoes', 'D'),  # v≥1     Bit mask of cargo types available for refitting, see column 2 (bit values) in CargoTypes
    0x12: ('cb_flags', 'B'),  # Callback flags bit mask, see below
    0x13: ('refit_cost', 'B'),  #  Refit cost, using 1/32 of the default refit cost base
    0x14: ('ocean_speed', 'B'),  # Ocean speed fraction, sets fraction of top speed available in the ocean; e.g. 00=100%, 80=50%, FF=0.4%
    0x15: ('canal_speed', 'B'),  # Canal speed fraction, same as above but for canals and rivers
    0x16: ('retire_early', 'B'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)
    0x17: ('flags', 'B'),  # Miscellaneous vehicle flags
    0x18: ('refit_classes', 'W'),  # Refittable cargo classes, see train prop. 28
    0x19: ('non_refit_classes', 'W'),  # Non-refittable cargo classes, see train prop. 29
    0x1A: ('introduction_date', 'D'),  # Long format introduction date
    0x1B: ('sort_purchase_list', 'B*'), # Sort the purchase list
    0x1C: ('visual_effect', 'B'),  # Visual effect
    0x1D: ('cargo_age_period', 'W'),  # Custom cargo ageing period
    0x1E: ('cargo_allow_refit', 'B n*B'),  # List of always refittable cargo types, see train property 2C
    0x1F: ('cargo_disallow_refit', 'B n*B'),  # List of never refittable cargo types, see train property 2D
}

ACTION0_GLOBAL_PROPS = {
    0x08: ('basecost', 'B'),  # Cost base multipliers
    0x09: ('cargo_table', 'D'),  # Cargo translation table
    0x0A: ('currency_name', 'W'),  # Currency display names
    0x0B: ('currency_mult', 'D'),  # Currency multipliers
    0x0C: ('currency_options', 'W'),  # Currency options
    0x0D: ('currency_symbols', '0E'),  # Currency symbols
    0x0F: ('currency_euro_date', 'W'),  # Euro introduction dates
    0x10: ('snowline_table', '12*32*B'),  # Snow line height table
    0x11: ('grfid_overrides', '2*D'),  # GRFID overrides for engines
    0x12: ('railtype_table', 'D'),  # Railtype translation table
    0x13: ('gender_table1', '(BV)+'),  # Gender/case translation table
    0x14: ('gender_table2', '(BV)+'),  # Gender/case translation table
    0x15: ('plural_form', 'B'),  # Plural form
    0x16: ('roadtype_table1', 'B'),  # Road-/tramtype translation table
    0x17: ('roadtype_table2', 'D'),  # Road-/tramtype translation table
}

ACTION0_HOUSE_PROPS = {
    0x08: ('substitute', 'B'),  # Substitute building type
    0x09: ('flags', 'B'),  # Building flags
    0x0A: ('availability_years', 'W'),  # Availability years
    0x0B: ('poopulation', 'B'),  # Population
    0x0C: ('mail_mult', 'B'),  # Mail generation multiplier
    0x0D: ('passengers_acceptance', 'B'),  # Passenger acceptance
    0x0E: ('mail_acceptance', 'B'),  # Mail acceptance
    0x0F: ('goods_acceptance', 'B'),  # Goods, food or fizzy drinks acceptance
    0x10: ('authority_impact', 'W'),  # LA rating decrease on removal (should be set to the same value for every tile for multi-tile buildings
    0x11: ('removal_cost_mult', 'B'),  # Removal cost multiplier (should be set to the same value for every tile for multi-tile buildings)
    0x12: ('name', 'W'),  # Building name ID
    0x13: ('availability_mask', 'W'),  # Building availability mask
    0x14: ('cb_flags1', 'B'),  # House callback flags
    0x15: ('override', 'B'),  # House override byte
    0x16: ('refresh_multiplier', 'B'),  # Periodic refresh multiplier
    0x17: ('random_colours', '4*B'),# Four random colours to use
    0x18: ('probability', 'B'),  # Relative probability of appearing
    0x19: ('extra_flags', 'B'),  # Extra flags  (high byte of prop 0x9)
    0x1A: ('animation_frames', 'B'),  # Animation frames
    0x1B: ('animation_speed', 'B'),  # Animation speed
    0x1C: ('building_class', 'B'),  # Class of the building type
    0x1D: ('cb_flags2', 'B'),  # Callback flags 2
    0x1E: ('accepted_cargo', 'D'),  # Accepted cargo types
    0x1F: ('min_life', 'W'),  # Minimum life span in years
    0x20: ('watched_cargo_types', 'W'),  # Cargo acceptance watch list
    0x21: ('min_year', 'W'),  # Long year (zero based) of minimum appearance
    0x22: ('max_year', 'W'),  # Long year (zero based) of maximum appearance
    0x23: ('tile_acceptance', 'V'), # Tile acceptance list
}

ACTION0_INDUSTRY_TILE_PROPS = {
    0x08: ('building_type', 'B'),  # Substitute building type
    0x09: ('tile_override', 'B'),  # Industry tile override
    0x0A: ('tile_acceptance_1', 'W'),  # Tile acceptance
    0x0B: ('tile_acceptance_2', 'W'),  # Tile acceptance
    0x0C: ('tile_acceptance_3', 'W'),  # Tile acceptance
    0x0D: ('land_shape_flags', 'B'),  # Land shape flags
    0x0E: ('cb_flags', 'B'),  # Callback flags
    0x0F: ('anim_info', 'W'),  # Animation information
    0x10: ('anim_speed', 'B'),  # Animation speed.
    0x11: ('cb25_triggers', 'B'),  # Triggers for callback 25
    0x12: ('flags', 'B'),  # Special flags
    0x13: ('tile_acceptance_list', 'n*(BB)'),  # Tile acceptance list
}

ACTION0_INDUSTRY_PROPS = {
    0x08: ('substitute_type', 'B'),  # Substitute industry type
    0x09: ('override_type', 'B'),  # Industry type override
    0x0A: ('layouts', 'Layouts'),  # Set industry layout(s)
    0x0B: ('production_flags', 'B'),  # Industry production flags
    0x0C: ('closure_message', 'W'),  # Industry closure message
    0x0D: ('production_increase_message', 'W'),  # Production increase message
    0x0E: ('production_decrease_message', 'W'),  # Production decrease message
    0x0F: ('func_cost', 'B'),  # Fund cost multiplier
    0x10: ('production_cargo', 'W'),  # Production cargo types
    0x11: ('acceptance_cargo', 'D'),  # Acceptance cargo types
    0x12: ('produciton_multipliers', 'B'),  # Production multipliers
    0x13: ('acceptance_multipliers', 'B'),  # Production multipliers
    0x14: ('minimal_distributed', 'B'),  # Minimal amount of cargo distributed
    0x15: ('random_sound', 'n*B'),  # Random sound effects
    0x16: ('conflicting_indtypes', '3*B'),  # Conflicting industry types
    0x17: ('mapgen_probability', 'B'),  # Probability in random game
    0x18: ('ingame_probability', 'B'),  # Probability during gameplay
    0x19: ('map_colour', 'B'),  # Map color
    0x1A: ('special_flags', 'D'),  # Special industry flags to define special behavior
    0x1B: ('text_id', 'W'),  # New industry text ID
    0x1C: ('input_mult1', 'D'),  # Input cargo multipliers for the three input cargo types
    0x1D: ('input_mult2', 'D'),  # Input cargo multipliers for the three input cargo types
    0x1E: ('input_mult3', 'D'),  # Input cargo multipliers for the three input cargo types
    0x1F: ('name', 'W'),  # Industry name
    0x20: ('prospecting_chance', 'D'),  # Prospecting success chance
    0x21: ('cb_flags1', 'B'),  # Callback flags
    0x22: ('cb_flags2', 'B'),  # Callback flags
    0x23: ('destruction_cost', 'D'),  # Destruction cost multiplier
    0x24: ('station_text', 'W'),  # Default text for nearby station
    0x25: ('production_types', 'n*B'),  # Production cargo type list
    0x26: ('acceptance_types', 'n*B'),  # Acceptance cargo type list
    0x27: ('produciton_multipliers', 'n*B'),  # Production multiplier list
    0x28: ('input_multipliers', 'n*m*W'),  # Input cargo multiplier list
}

ACTION0_CARGO_PROPS = {
    0x08: ('bit_number', 'B'),  # Bit number for bitmasks
    0x09: ('type_text', 'W'),  # TextID for the cargo type name
    0x0A: ('unit_text', 'W'),  # TextID for the name of one unit from the cargo type
    0x0B: ('one_text', 'W'),  # TextID to be displayed for 1 unit of cargo
    0x0C: ('many_text', 'W'),  # TextID to be displayed for multiple units of cargo
    0x0D: ('abbr_text', 'W'),  # TextID for cargo type abbreviation
    0x0E: ('icon_sprite', 'W'),  # Sprite number for the icon of the cargo
    0x0F: ('weight', 'B'),  # Weight of one unit of the cargo
    0x10: ('penalty1', 'B'),  # Penalty times
    0x11: ('penalty2', 'B'),  # Penalty times
    0x12: ('base_price', 'D'),  # Base price
    0x13: ('station_colour', 'B'),  # Color for the station list window
    0x14: ('colour', 'B'),  # Color for the cargo payment list window
    0x15: ('is_freight', 'B'),  # Freight status (for freight-weight-multiplier setting); 0=not freight, 1=is freight
    0x16: ('classes', 'W'),  # Cargo classes
    0x17: ('label', 'D'),  # Cargo label
    0x18: ('town_growth_sub', 'B'),  # Substitute type for town growth
    0x19: ('town_growth_mult', 'W'),  # Multiplier for town growth
    0x1A: ('cb_flags', 'B'),  # Callback flags
    0x1B: ('units_text', 'W'),  # TextID for displaying the units of a cargo
    0x1C: ('amount_text', 'W'),  # TextID for displaying the amount of cargo
    0x1D: ('capacity_mult', 'W'),  # Capacity mulitplier
}

ACTION0_OBJECT_PROPS = {
    0x08: ('label', 'L'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Class label, see below
    0x09: ('class_name_id', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Text ID for class
    0x0A: ('name_id', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Text ID for this object
    0x0B: ('climate', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Climate availability
    0x0C: ('size', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Byte representing size, see below
    0x0D: ('build_cost_factor', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object build cost factor (sets object removal cost factor as well)
    0x0E: ('intro_date', 'D'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Introduction date, see below
    0x0F: ('eol_date', 'D'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   End of life date, see below
    0x10: ('flags', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object flags, see below
    0x11: ('anim_info', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation information
    0x12: ('anim_speed', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation speed
    0x13: ('anim_trigger', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation triggers
    0x14: ('removal_cost_factor', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object removal cost factor (set after object build cost factor)
    0x15: ('cb_flags', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Callback flags, see below
    0x16: ('building_height', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Height of the building
    0x17: ('num_views', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Number of object views
    0x18: ('num_objects', 'B'),  # Supported by OpenTTD 1.4 (r25879)1.4 Not supported by TTDPatch  Measure for number of objects placed upon map creation
}

ACTION0_RAILTYPE_PROPS = {
# common
    0x08: ('label', 'L'),  # Rail type label
    0x09: ('toolbar_caption', 'W'),  # StringID: Build rail toolbar caption[1]
    0x0A: ('menu_text', 'W'),  # StringID: Rail construction dropdown text
    0x0B: ('build_window_caption', 'W'),  # StringID: Build vehicle window caption
    0x0C: ('autoreplace_text', 'W'),  # StringID: Autoreplace text
    0x0D: ('new_engine_text', 'W'),  # StringID: New engine text
    0x13: ('construction_cost', 'W'),  # Construction costs
    0x16: ('map_colour', 'B'),  # Minimap colour
    0x17: ('introduction_date', 'D'),  # Introduction date
    0x1A: ('sort_order', 'B'),  # Sort order
    0x1B: ('name', 'W'),  # StringID: Rail type name[4]
    0x1C: ('maintenance_cost', 'W'),  # Infrastructure maintenance cost factor

# railtype
    0x0E: ('compatible_railtype_list', 'n*L'),  # Compatible rail type list[2]
    0x0F: ('powered_railtype_list', 'n*L'),  # Powered rail type list[2]
    0x10: ('railtype_flags', 'B'),  # Rail type flags
    0x11: ('curve_speed_multiplier', 'B'),  # Curve speed advantage multiplier
    0x12: ('station_graphics', 'B'),  # Station (and depot) graphics
    0x14: ('speed_limit', 'W'),  # Speed limit
    0x15: ('acceleration_model', 'B'),  # Acceleration model
    0x18: ('requires_railtype_list', 'n*L'),  # Introduction required rail type list[2]
    0x19: ('introduces_railtype_list', 'n*L'),  # Introduced rail type list[2]
    0x1D: ('alternative_railtype_list', 'n*L'),  # Alternate rail type labels that shall be "redirected" to this rail type
}

ACTION0_PROPS = {
    0x0: ACTION0_TRAIN_PROPS,
    0x1: ACTION0_RV_PROPS,
    0x2: ACTION0_SHIP_PROPS,
    0x7: ACTION0_HOUSE_PROPS,
    0x8: ACTION0_GLOBAL_PROPS,
    0x9: ACTION0_INDUSTRY_TILE_PROPS,
    0xa: ACTION0_INDUSTRY_PROPS,
    0xb: ACTION0_CARGO_PROPS,
    0xf: ACTION0_OBJECT_PROPS,
    0x10: ACTION0_RAILTYPE_PROPS,
}

ACTION0_PROP_DICT = {
    feature: {name: (id, size) for id, (name, size, *_) in fdict.items()}
    for feature, fdict in ACTION0_PROPS.items()
}


class Action0(LazyBaseSprite):  # action 0
    def __init__(self, feature, first_id, count, props):
        assert isinstance(feature, Feature)
        super().__init__()
        self.feature = feature
        self.first_id = first_id
        self.count = count
        if feature.id not in ACTION0_PROP_DICT:
            raise NotImplementedError
        self.props = props
        for x in props:
            if x not in ACTION0_PROP_DICT[feature.id]:
                raise ValueError(f'Unknown property `{x}` for a feature {feature}')

    def _encode_value(self, value, fmt):
        if fmt == 'B': return struct.pack('<B', value)
        if fmt == 'W': return struct.pack('<H', value)
        if fmt == 'D':
            if isinstance(value, datetime.date):
                value = date_to_days(value)
            return struct.pack('<I', value)
        if fmt == 'L':
            if isinstance(value, int): return struct.pack('<I', value)
            assert isinstance(value, bytes)
            assert len(value) == 4, len(value)
            return value
        if fmt == 'B*': return struct.pack('<BH', 255, value)
        if fmt == 'n*B':
            assert isinstance(value, bytes)
            assert len(value) < 256, len(value)
            return struct.pack('<B', len(value)) + value

    def _encode(self):
        res = struct.pack('<BBBBBH',
            0, self.feature.id, len(self.props), self.count, 255, self.first_id)
        pdict = ACTION0_PROP_DICT[self.feature.id]
        for prop, value in self.props.items():
            code, fmt = pdict[prop]
            res += bytes((code,)) + self._encode_value(value, fmt)
        return res

    def py(self):
        return f'''
        Action0(
            feature={self.feature},
            first_id={self.first_id},
            count={self.count},
            props=''' + pformat(self.props, indent_first=0, indent=10 + 8) + '''
        )'''


class Object(Action0):
    class Flags:
        NONE               =       0  # Just nothing.
        ONLY_IN_SCENEDIT   = 1 <<  0  # Object can only be constructed in the scenario editor.
        CANNOT_REMOVE      = 1 <<  1  # Object can not be removed.
        AUTOREMOVE         = 1 <<  2  # Object get automatically removed (like "owned land").
        BUILT_ON_WATER     = 1 <<  3  # Object can be built on water (not required).
        CLEAR_INCOME       = 1 <<  4  # When object is cleared a positive income is generated instead of a cost.
        HAS_NO_FOUNDATION  = 1 <<  5  # Do not display foundations when on a slope.
        ANIMATION          = 1 <<  6  # Object has animated tiles.
        ONLY_IN_GAME       = 1 <<  7  # Object can only be built in game.
        CC2_COLOUR         = 1 <<  8  # Object wants 2CC colour mapping.
        NOT_ON_LAND        = 1 <<  9  # Object can not be on land, implicitly sets #OBJECT_FLAG_BUILT_ON_WATER.
        DRAW_WATER         = 1 << 10  # Object wants to be drawn on water.
        ALLOW_UNDER_BRIDGE = 1 << 11  # Object can built under a bridge.
        ANIM_RANDOM_BITS   = 1 << 12  # Object wants random bits in "next animation frame" callback.
        SCALE_BY_WATER     = 1 << 13  # Object count is roughly scaled by water amount at edges.


    def __init__(self, id, **props):
        if 'size' in props:
            props['size'] = (props['size'][1] << 4) | props['size'][0]
        super().__init__(OBJECT, id, 1, props)


class Action1(LazyBaseSprite):
    def __init__(self, feature, set_count, sprite_count):
        assert isinstance(feature, Feature)
        super().__init__()
        self.feature = feature
        self.set_count = set_count
        self.sprite_count = sprite_count

    def _encode(self):
        return struct.pack('<BBBBH', 0x01,
                           self.feature.id, self.set_count, 0xFF, self.sprite_count)

    def py(self):
        return f'''
        Action1(
            feature={self.feature},
            set_count={self.set_count},
            sprite_count={self.sprite_count},
        )'''

class SpriteSet(Action1):
    def __init__(self, feature, sprite_count):
        super().__init__(feature, 1, sprite_count)


class Sprite:
    def __init__(self, id, pal=0, is_global=True, use_recolour=False, always_transparent=False, no_transparent=False):
        self.id = id
        self.pal = pal
        self.is_global = is_global
        self.use_recolour = use_recolour
        self.always_transparent = always_transparent
        self.no_transparent = no_transparent

    @classmethod
    def from_grf(cls, sprite, pal):
        return cls(
            id=sprite & 0x3fff,
            pal=pal & 0x3ffff,
            is_global=not bool(pal & 0x8000),
            use_recolour=bool(sprite & 0x8000),
            always_transparent=bool(sprite & 0x4000),
            no_transparent=bool(pal & 0x4000),
        )

    def to_grf(self):
        return (
            self.id | (self.always_transparent << 14) | (self.use_recolour << 15) |
            (self.pal | (self.no_transparent << 14) | ((not self.is_global) << 15)) << 16
        )

    def __repr__(self):
        return self.py()

    def py(self):
        return (
            f'Sprite(id={self.id}, pal={self.pal}, is_global={self.is_global}, '
            f'use_recolour={self.use_recolour}, always_transparent={self.always_transparent}, '
            f'no_transparent={self.no_transparent})'
        )


class ReferenceableAction:
    pass


def get_ref_id(ref_obj):
    if isinstance(ref_obj, Ref):
        return ref_obj.value
    if isinstance(ref_obj, int):
        return ref_obj | 0x8000
    return ref_obj.ref_id


# Action2
class GenericSpriteLayout(LazyBaseSprite, ReferenceableAction):
    def __init__(self, *, ent1, ent2, feature=None, ref_id=None):
        assert feature not in (HOUSE, INDUSTRY_TILE, OBJECT, AIRPORT_TILE, INDUSTRY), feature
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.ent1 = ent1
        self.ent2 = ent2

    def _encode(self):
        return struct.pack(
            '<BBBBB' + 'H' * (len(self.ent1) + len(self.ent2)),
            0x02,
            self.feature.id,
            self.ref_id,
            len(self.ent1),
            len(self.ent2),
            *self.ent1,
            *self.ent2,
        )

    def py(self):
        return f''' \
        GenericSpriteLayout(
            feature={self.feature},
            ref_id={self.ref_id},
            ent1={self.ent1!r},
            ent2={self.ent2!r},
        )
        '''


# Action2
class BasicSpriteLayout(LazyBaseSprite, ReferenceableAction):
    def __init__(self, *, ground, building, feature=None, ref_id=None):
        assert feature in (HOUSE, INDUSTRY_TILE, OBJECT, INDUSTRY), feature
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.ground = ground
        self.building = building

    def _encode(self):
        return struct.pack('<BBBBIIbbBBB', 0x02, self.feature.id, self.ref_id, 0,
                           self.ground['sprite'].to_grf(),
                           self.building['sprite'].to_grf(),
                           *self.building['offset'],
                           *self.building['extent'])

    def py(self):
        return f''' \
        BasicSpriteLayout(
            feature={self.feature},
            ref_id={self.ref_id},
            ground={self.ground!r},
            building={self.building!r},
        )
        '''


# Action2
class AdvancedSpriteLayout(LazyBaseSprite, ReferenceableAction):
    def __init__(self, *, ground, feature=None, ref_id=None, buildings=(), has_flags=True):
        assert feature in (HOUSE, INDUSTRY_TILE, OBJECT, INDUSTRY), feature
        assert len(buildings) < 64, len(buildings)

        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.ground = ground
        self.buildings = buildings

    def _encode_sprite(self, sprite, aux=False):
        flags = 0
        is_parent = ('extent' in sprite or 'offset' in sprite)
        for key, (parent_state, mask, _, _) in SPRITE_FLAGS.items():
            if parent_state is not None and parent_state != is_parent:
                continue
            if key in sprite:
                flags |= mask
        res = struct.pack('<IH', sprite['sprite'].to_grf(), flags)

        if aux:
            is_parent = 'extent' in sprite or 'offset' in sprite
            if is_parent:
                offset = sprite.get('offset', (0, 0, 0))
            else:
                offset = sprite.get('pixel_offset', (0, 0))
                offset = (offset[0], offset[1], 0x80)
            res += struct.pack('<BBB', *offset)
            if is_parent:
                res += struct.pack('<BBB', *sprite.get('extent', (0, 0, 0)))

        for k in ('dodraw', 'add', 'palette', 'sprite_var10', 'palette_var10'):
            # TODO deltas
            if k in sprite:
                val = sprite[k]
                if k == 'add':
                    assert isinstance(val, Temp)
                    val = val.register
                res += bytes((val, ))
        return res

    def _encode(self):
        res = struct.pack('<BBBB', 0x02, self.feature.id, self.ref_id, len(self.buildings) + 0x40)
        res += self._encode_sprite(self.ground)
        for s in self.buildings:
            res += self._encode_sprite(s, True)

        return res

    def py(self):
        return f'''
        {self.__class__.__name__}(
            feature={self.feature},
            ref_id={self.ref_id},
            ground={self.ground!r},
            buildings={self.buildings!r},
        )'''


# Action2
class ExtendedSpriteLayout(AdvancedSpriteLayout):
    # Not implemented yet
    def __init__(self, *args, **kw):
        super().__init__(*args, has_flags=False, **kw)


class Ref:
    def __init__(self, ref_id):
        self.value = ref_id
        self.is_callback = bool(ref_id & 0x8000)
        self.ref_id = ref_id & 0x7fff

    def __str__(self):
        if self.is_callback:
            return f'CB({self.ref_id})'
        return f'Ref({self.ref_id})'

    __repr__ = __str__

    def __int__(self):
        return self.value


class CB(Ref):
    def __init__(self, value):
        super().__init__(value | 0x8000)


class Range:
    def __init__(self, low, high, set):
        self.set = set
        self.low = low
        self.high = high

    def __str__(self):
        if self.low == self.high:
            return f'{self.low} -> {self.set}'
        return f'{self.low}..{self.high} -> {self.set}'

    def __repr__(self):
        return f'Range({self.low}, {self.high}, {self.set})'


class ReferencingAction:
    def get_refs(self):
        return NotImplementedError


class VarAction2(LazyBaseSprite, ReferenceableAction, ReferencingAction):
    def __init__(self, ranges, default, code, feature=None, ref_id=None, related_scope=False):
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.related_scope = related_scope
        self.ranges = ranges
        self.default = default
        self.code = code
        self._parsed_code = None

    def get_refs(self):
        yield from self.ranges.values()
        yield self.default

    @property
    def parsed_code(self):
        if self._parsed_code is None:
            # TODO related scope
            self._parsed_code = parse_code(self.feature, self.code)
        return self._parsed_code

    def __str__(self):
        return str(self.ref_id)

    def _encode(self):
        res = bytes((0x02, self.feature.id, self.ref_id, 0x8a if self.related_scope else 0x89))
        ast = self.parsed_code
        code = ast[0].compile(register=0x80)[1]
        for c in ast[1:]:
            code += bytes((OP_INIT,))
            code += c.compile(register=0x80)[1]
        res += code[:-5]
        res += bytes((code[-5] & ~0x20,))  # mark the end of a chain
        res += code[-4:]
        res += bytes((len(self.ranges),))
        ranges = self.ranges
        if isinstance(ranges, dict):
            ranges = self.ranges.items()
        for r in ranges:
            if isinstance(r, Range):
                set_obj = r.set
                low = r.low
                high = r.high
            else:
                set_obj = r[1]
                low = r[0]
                high = r[0]
            # TODO split (or validate) negative-positive ranges
            res += struct.pack('<Hii', get_ref_id(set_obj), low, high)
        res += struct.pack('<H', get_ref_id(self.default))
        return res

    def py(self):
        if '\n' not in self.code:
            code_str = repr(self.code)
        else:
            code_str = textwrap.indent("'''\n" + self.code + "'''", ' ' * 16).lstrip()
        ranges_str = pformat({
            utoi32(r.low) if r.low == r.high else (utoi32(r.low), utoi32(r.high)): r.set
            for r in self.ranges
        }, indent_first=0, indent=19)
        return f'''
        VarAction2(
            feature={self.feature},
            ref_id={self.ref_id},
            related_scope={self.related_scope},
            ranges={ranges_str},
            default={self.default},
            code={code_str},
        )'''


class IndustryProductionCallback(LazyBaseSprite):
    def __init__(self, inputs, outputs, do_again, version=2):
        assert isinstance(version, int)
        assert 0 <= version <= 2, version
        if version < 2:
            assert len(inputs) == 3, len(inputs)
            assert len(outputs) == 2, len(outputs)
        else:
            assert len(inputs) < 256, len(inputs)
            assert len(outputs) < 256, len(outputs)

        super().__init__()
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.do_again = do_again

    def _encode(self):
        inputs = self.inputs
        outputs = self.outputs
        if self.version == 0:
            return struct.pack('<BBWWWWWB', 0x0a, self.version, *inputs, *outputs, self.do_again)
        elif self.version == 1:
            return struct.pack('<BBBBBBBB', 0x0a, self.version, *inputs, *outputs, self.do_again)
        elif self.version == 2:
            fmt = 'B' + 'BB' * len(inputs)
            fmt += 'B' + 'BB' * len(outputs)
            return struct.pack(
                '<BB' + fmt + 'B',
               0x0a,
               self.version,
               len(self.inputs),
               *sum(self.inputs, []),
               len(self.outputs),
               *sum(self.outputs, []),
               self.do_again
            )

    def py(self):
        return f'''
        {self.__class__.__name__}(
            version={self.version},
            inputs={self.inputs!r},
            outputs={self.outputs!r},
            do_again={self.do_again},
        )'''


class Action3(LazyBaseSprite, ReferencingAction):
    def __init__(self, feature, ids, maps, default):
        assert isinstance(feature, Feature), feature
        super().__init__()
        self.feature = feature
        self.ids = ids
        self.maps = maps
        self.default = default

    def get_refs(self):
        yield from self.maps.values()
        yield self.default

    def _encode(self):
        idcount = len(self.ids)
        mcount = len(self.maps)
        if idcount == 0:
            return struct.pack('<BBBBH', 0x03, self.feature.id, idcount, 0, self.default)
        else:
            idlist = self.ids
            idfmt = 'B'
            if self.feature in VEHICLE_FEATURES and any(x >= 0xff for x in self.ids):
                idlist = [0xff] * (2 * idcount)
                idlist[1::2] = self.ids
                idfmt = 'BH'
            return struct.pack(
                '<BBB' + idfmt * idcount + 'B' + 'BH' * mcount + 'H',
                0x03, self.feature.id, idcount,
                *idlist, mcount, *sum(((k, get_ref_id(x)) for k, x in self.maps.items()), ()),
                get_ref_id(self.default))

    def py(self):
        return f'''
        Action3(
            feature={self.feature},
            ids={self.ids!r},
            maps={self.maps!r},
            default={self.default},
        )'''


class Map(Action3):
    def __init__(self, object, maps, default):
        super().__init__(object.feature, [object.first_id], maps, default)


class Action4(LazyBaseSprite):
    def __init__(self, feature, offset, is_generic_offset, strings, lang=None):
        assert isinstance(feature, Feature), feature
        if feature in VEHICLE_FEATURES or is_generic_offset:
            if not 0 <= offset <= 0xffff:
                raise ValueError(f'Action4 `offset` {offset} is not in range 0..0xffff (for generic offset or vehicles)')
        else:
            if not 0 <= offset <= 0xff:
                raise ValueError(f'Action4 `offset` {offset} is not in range 0..0xff (for non-generic non-vehicle offset)')

        super().__init__()
        self.feature = feature
        self.lang = lang
        self.offset = offset
        self.is_generic_offset = is_generic_offset
        self.strings = []
        for s in strings:
            if isinstance(s, bytes):
                self.strings.append(s)
            else:
                assert isinstance(s, str), type(s)
                self.strings.append(s.encode('utf-8'))

    def _encode(self):
        lang = self.lang if self.lang is not None else ANY_LANGUAGE
        str_data = b'\0'.join(self.strings) + b'\0'
        if self.is_generic_offset:
            return struct.pack('<BBBBH', 0x04, self.feature.id, lang | 0x80, len(self.strings), self.offset) + str_data
        elif self.feature in VEHICLE_FEATURES:
            return struct.pack('<BBBBBH', 0x04, self.feature.id, lang, len(self.strings), 0xff, self.offset) + str_data
        else:
            return struct.pack('<BBBBBH', 0x04, self.feature.id, lang, len(self.strings), 0xff, self.offset) + str_data

    def py(self):
        return f'''
        Action4(
            feature={self.feature},
            lang={self.lang},
            offset={self.offset},
            is_generic_offset={self.is_generic_offset},
            strings=''' + pformat(self.strings, indent_first=0, indent=10 + 8) + '''
        )'''


class Action6(LazyBaseSprite):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def _encode(self):
        return struct.pack(
            '<B' + 'BBBH' * len(self.params) + 'B',
            0x06,
            *sum(((p['num'], p['size'], 0xff, p['offset']) for p in self.param), ()),
            0xff
        )

    def py(self):
        return f'Action6(\n' + pformat(self.params) + '\n)'


EXPORT_CLASSES = [
    FileSprite, Action0, Action1, Action3, Action4, Map,
    SpriteSet, BasicSpriteLayout, AdvancedSpriteLayout, VarAction2,
    SetProperties, ReplaceOldSprites, ReplaceNewSprites, SetDescription,
    Comment, ActionC, Label, Action10, If,
]

class BaseNewGRF:
    def __init__(self):
        self.generators = []
        self._sprite_encoder = SpriteEncoder(True, False, None)

    def _add(self, l, *sprites):
        if not sprites:
            return

        # Unfold real sprite tuple if it was passed as a single arg
        if isinstance(sprites[0], tuple):
            assert len(sprites) == 1
            sprites = sprites[0]
            assert len(sprites) >= 1
            assert isinstance(sprites[0], RealSprite)

        if isinstance(sprites[0], RealSprite):
            assert(all(isinstance(s, RealSprite) for s in sprites))
            assert(len(set((s.zoom, s.bpp) for s in sprites)) == len(sprites))

            l.append(tuple(sprites))
        else:
            l.extend(sprites)

    def add(self, *sprites):
        self._add(self.generators, *sprites)

    def _write_pseudo_sprite(self, f, data, grf_type=0xff):
        f.write(struct.pack('<IB', len(data), grf_type))
        f.write(data)

    def generate_python(self):
        cn = [c.__name__ for c in EXPORT_CLASSES]
        res = (
            'import grf\n\n'
            'g = grf.BaseNewGRF()\n\n'
        )
        res += ', '.join(cn) + ' = ' + ', '.join('g.' + x for x in cn) + '\n\n'

        for s in self.generators:
            if isinstance(s, tuple):
                s = s[0]
            code = textwrap.dedent(s.py()).strip()
            res += code + '\n'
            if '\n' in code: res += '\n'

        res += '\ng.write("some.grf")'
        return res

    def generate_sprites(self):
        res = []
        for g in self.generators:
            if isinstance(g, SpriteGenerator):
                for s in g.get_sprites():
                    self._add(res, s)
            else:
                res.append(g)
        return res

    def resolve_refs(self, sprites):
        refids = {}
        actions = {}
        ordered_refs = defaultdict(list)
        unordered_refs = defaultdict(list)
        ref_count = defaultdict(int)

        def resolve(s, i):
            # Action can reference other actions, resolve all the references
            if isinstance(s, ReferencingAction):
                for r in s.get_refs():
                    if isinstance(r, Ref):
                        if r.is_callback:
                            continue
                        r = refids.get(r.ref_id)
                        if r is None:
                            raise RuntimeError(f'Unresolved direct reference {r} in action {s}')

                    if not isinstance(r, ReferenceableAction):
                        continue

                    if r is s:
                        raise RuntimeError(f'Action {s} references itself')

                    assert isinstance(r, ReferenceableAction)

                    if id(r) not in actions:
                        actions[id(r)] = (r, None)
                        unordered_refs[id(s)].append(id(r))
                    else:
                        ordered_refs[id(s)].append(id(r))
                    ref_count[id(r)] += 1
                    resolve(r, None)

            prev_i = actions.get(id(s), (None, None))[1]
            if prev_i is not None and i is not None and prev_i != i:
                raise RuntimeError(f'Action {s} was added more than once (positions {prev_i} and {i})')
            actions[id(s)] = (s, i)

        for i, s in enumerate(sprites):
            if not isinstance(s, (ReferencingAction, ReferenceableAction)):
                continue

            resolve(s, i)

            # Action can be referenced by other actions, index it
            # Must be done after processing references to allow reusing the same ref_id
            if isinstance(s, ReferenceableAction):
                if s.ref_id is not None:
                    refids[s.ref_id] = s


        ids = list(range(-255, -1))
        reserved_ids = set()
        linked_refs = {}
        visited = set()

        def dfs(aid, linked, feature):
            a, ai = actions[aid]
            if a.feature is not None and a.feature != feature:
                raise RuntimeError(f'Mixed features {a.feature} and {feature} in action {a}')
            if aid in visited:
                return
            visited.add(aid)
            if a.feature is None:
                a.feature = feature
            if isinstance(a, ReferenceableAction):
                if a.ref_id is None:
                    try:
                        while a.ref_id is None or a.ref_id in reserved_ids:
                            a.ref_id = -heapq.heappop(ids)
                    except IndexError:
                        raise RuntimeError(f'Ran out of ids while trying to reference action {a}')
                else:
                    reserved_ids.add(a.ref_id)

            if ai is not None:
                # Node is already ordered, prepend unordered nodes to it
                linked = linked_refs[aid] = []

            for x in ordered_refs[aid]:
                dfs(x, linked, feature)

            for x in unordered_refs[aid]:
                dfs(x, linked, feature)

            # Apppend node to the ordered parent node (or itself)
            linked.append(a)

            if isinstance(a, ReferenceableAction):
                ref_count[aid] -= 1
                # If this action is longer referenced return id to the pool
                if ref_count[aid] <= 0:
                    heapq.heappush(ids, a.ref_id)
                    reserved_ids.discard(a.ref_id)

        roots = [(r, v[0].feature) for r, v in actions.items() if ref_count[r] <= 0]
        # print('SPRITES', sprites)
        # print('ACTIONS', actions)
        # print('OREF', ordered_refs)
        # print('UREF', unordered_refs)
        # print('roots', roots)
        for r, f in roots:
            assert f is not None
            dfs(r, None, f)

        # print('LREF', linked_refs)

        # Construct the full list of actions with resolved ids and order
        res = []
        for s in sprites:
            if isinstance(s, ReferencingAction):
                res.extend(linked_refs.get(id(s), []))
            else:
                res.append(s)
        return res


    def write(self, filename):
        sprites = self.generate_sprites()
        sprites = self.resolve_refs(sprites)
        data_offset = 14

        next_sprite_id = 1
        for s in sprites:
            if isinstance(s, tuple):
                for x in s:
                    x.sprite_id = next_sprite_id
                next_sprite_id += 1
                s = s[0]
            data_offset += s.get_data_size() + 5

        with open(filename, 'wb') as f:
            f.write(b'\x00\x00GRF\x82\x0d\x0a\x1a\x0a')  # file header
            f.write(struct.pack('<I', data_offset))
            f.write(b'\x00')  # compression(1)
            # f.write(b'\x04\x00\x00\x00')  # num(4)
            # f.write(b'\xFF')  # grf_type(1)
            # f.write(b'\xb0\x01\x00\x00')  # num + 0xff -> recoloursprites() (257 each)
            self._write_pseudo_sprite(f, b'\x02\x00\x00\x00')

            for s in sprites:
                if isinstance(s, tuple):
                    self._write_pseudo_sprite(f, s[0].get_data(), grf_type=0xfd)
                else:
                    self._write_pseudo_sprite(f, s.get_data(), grf_type=0xff)
            f.write(b'\x00\x00\x00\x00')
            for sl in sprites:
                if not isinstance(sl, tuple):
                    continue
                for s in sl:
                    f.write(s.get_real_data(self._sprite_encoder))

            f.write(b'\x00\x00\x00\x00')

    def wrap(self, func):
        def wrapper(*args, **kw):
            obj = func(*args, **kw)
            self.add(obj)
            return obj
        return wrapper

    def bind(self, cls):
        @functools.wraps(cls)
        def wrapper(*args, **kw):
            gen = cls(*args, **kw)
            self.add(gen)
            return gen
        return wrapper


class NewGRF(BaseNewGRF):
    def __init__(self, grfid, version, name, description):
        super().__init__()
        self.generators.append(SetProperties('D'))
        self.generators.append(SetDescription(version, grfid, name, description))



for cls in EXPORT_CLASSES:

    def func(self, *args, _cls=cls, **kw):
        obj = _cls(*args, **kw)
        self.add(obj)
        return obj

    setattr(BaseNewGRF, cls.__name__, func)
