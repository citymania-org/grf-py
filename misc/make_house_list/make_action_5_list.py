import json
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
import numpy as np

import grf

WIDTH = 1020
BUILDING_HEIGHT = 120
VARIANT_HEIGHT = 32 + BUILDING_HEIGHT
PADDING = 10
FONT_SIZE = 20
LINE_HEIGHT = FONT_SIZE + 2
CLIMATE_COLOURS = ((0x74, 0xD5, 0x74), (0x68, 0x92, 0xB4), (0xff, 0xce, 0x8c), (0xff, 0x8c, 0x8c))

font = ImageFont.truetype('OpenTTD-Sans.ttf', FONT_SIZE)
font_small = ImageFont.truetype('OpenTTD-Sans.ttf', FONT_SIZE // 2)


data = json.loads(open('openttd.json', 'r').read()[9:])

dt = data['house_draw_tile_data']

TEMPERATE, ARCTIC, TROPICAL, TOYLAND = 1, 2, 4, 8
NO_CLIMATE, ALL_CLIMATES = 0, TEMPERATE | ARCTIC | TROPICAL | TOYLAND


class RecolourSprite(grf.Sprite):
    def __init__(self, sprite, recolour):
        super().__init__(sprite.w, sprite.h, xofs=sprite.xofs, yofs=sprite.yofs, bpp=sprite.bpp, zoom=sprite.zoom, crop=sprite.crop)
        self.sprite = sprite
        self.recolour = recolour
        self._image = None

    def get_resource_files(self):
        return self.sprite.get_resource_files()

    def get_fingerprint(self):
        return grf.combine_fingerprint(
            super().get_fingerprint_base(),
            sprite=self.sprite.get_fingerprint,
            recolour=True,
        )

    def get_image(self):
        if self._image is not None:
            return self._image
        im, bpp = self.sprite.get_image()
        assert bpp == 8
        data = np.array(im)
        data = self.recolour[data]
        im2 = Image.fromarray(data)
        im2.putpalette(im.getpalette())
        self._image = im2, grf.BPP_8
        return self._image


class GRFFile(grf.LoadedResourceFile):
    def __init__(self, path, *, baseset=False, real_offset=1, pseudo_offset=0):
        self.path = path
        self.real_sprites = None
        self.real_offset = real_offset
        self.pseudo_offset = pseudo_offset
        self.load(baseset)

    def load(self, baseset):
        if self.real_sprites is not None:
            return
        print(f'Decompiling {self.path}...')
        self.context = grf.decompile.ParsingContext(baseset=baseset)
        self.f = open(self.path, 'rb')
        self.g, self.container, self.real_sprites, self.pseudo_sprites = grf.decompile.read(self.f, self.context)
        cur_action = None
        cur_count = 0
        cur_offset = 0
        self.action_5_map = defaultdict(dict)
        self.action_a_map = {}
        for s in self.g.generators:
            if isinstance(s, grf.ReplaceNewSprites):
                cur_action = s
                cur_count = s.count
                cur_offset = s.offset or 0
            elif isinstance(s, grf.ReplaceOldSprites):
                assert len(s.sets) == 1
                cur_action = s
                cur_count = s.sets[0][1]
                cur_offset = s.sets[0][0]
            elif isinstance(s, grf.decompile.SpritePlaceholder) and cur_count > 0:
                if isinstance(cur_action, grf.ReplaceNewSprites):
                    self.action_5_map[cur_action.set_type][cur_offset] = s.sprite_id
                else:
                    self.action_a_map[cur_offset] = s.sprite_id
                cur_offset += 1
                cur_count -= 1

    def unload(self):
        self.context = None
        self.gen = None
        self.container = None
        self.real_sprites = None
        self.f.close()

    def get_sprite_data(self, sprite_id, num):
        s = self.real_sprites[sprite_id + self.real_offset][num]
        self.f.seek(s.offset)
        data, _ = grf.decompile.decode_sprite(self.f, s, self.container)
        return data

    def get_sprites(self, sprite_id):
        return [
            GRFSprite(self, s.type, s.bpp, sprite_id, i, w=s.width, h=s.height, xofs=s.xofs, yofs=s.yofs, zoom=s.zoom, crop=False)
            for i, s in enumerate(self.real_sprites[sprite_id + self.real_offset])
        ]

    def get_recolour_sprite(self, sprite_id):
        s = self.pseudo_sprites[sprite_id + self.pseudo_offset]
        assert isinstance(s, grf.decompile.RealRemapSprite)
        return np.frombuffer(s.data[1:], dtype=np.uint8)


class GRFSprite(grf.Sprite):

    def __init__(self, file, type, grf_bpp, sprite_id, num, *args, **kw):
        super().__init__(*args, **kw)
        self.file = file
        self.type = type
        self.grf_bpp = grf_bpp
        self.sprite_id = sprite_id
        self.num = num  # index with the same sprite_id

        bpp = grf_bpp
        if self.type & 0x04:
            bpp -= 1

        if bpp == 3:
            self.bpp = grf.BPP_24
        elif bpp == 4:
            self.bpp = grf.BPP_32
        else:
            self.bpp = grf.BPP_8

        self._image = None

    def get_resource_files(self):
        return (self.file, THIS_FILE)

    def get_fingerprint(self):
        return {
            'class': self.__class__.__name__,
            'sprite_id': self.sprite_id,
        }

    def load(self):
        if self._image is not None:
            return

        data = self.file.get_sprite_data(self.sprite_id, self.num)

        a = np.frombuffer(data, dtype=np.uint8)
        assert a.size == self.w * self.h * self.grf_bpp, (a.size, self.w, self.h, self.grf_bpp)

        bpp = self.grf_bpp
        a.shape = (self.h, self.w, bpp)

        mask = None
        if self.type & 0x04 > 0:
            bpp -= 1
            mask = Image.fromarray(a[:, :, -1], mode='P')
            mask.putpalette(grf.PIL_PALETTE)

        if bpp == 3:
            self._image = Image.fromarray(a[:, :, :bpp], mode='RGB'), grf.BPP_24
        elif bpp == 4:
            self._image = Image.fromarray(a[:, :, :bpp], mode='RGBA'), grf.BPP_32
        else:
            self._image = mask, grf.BPP_8
            mask = None

        # return self._image[0].show()

    def get_image(self):
        self.load()
        return self._image


# HZ_NOZNS             = 0x0000,  ///<       0          This is just to get rid of zeros, meaning none
# HZ_ZON1              = 1U << HZB_TOWN_EDGE,    ///< 0..4 1,2,4,8,10  which town zones the building can be built in, Zone1 been the further suburb
# HZ_ZON2              = 1U << HZB_TOWN_OUTSKIRT,
# HZ_ZON3              = 1U << HZB_TOWN_OUTER_SUBURB,
# HZ_ZON4              = 1U << HZB_TOWN_INNER_SUBURB,
# HZ_ZON5              = 1U << HZB_TOWN_CENTRE,  ///<  center of town
# HZ_ZONALL            = 0x001F,  ///<       1F         This is just to englobe all above types at once
# HZ_SUBARTC_ABOVE     = 0x0800,  ///< 11    800        can appear in sub-arctic climate above the snow line
# HZ_TEMP              = 0x1000,  ///< 12   1000        can appear in temperate climate
# HZ_SUBARTC_BELOW     = 0x2000,  ///< 13   2000        can appear in sub-arctic climate below the snow line
# HZ_SUBTROPIC         = 0x4000,  ///< 14   4000        can appear in subtropical climate
# HZ_TOYLND            = 0x8000,  ///< 15   8000        can appear in toyland climate
# HZ_CLIMALL           = 0xF800,  ///< Bitmask of all climate bits

def str_zones(availability):
    res = []
    for i in range(5):
        if availability & (1 << i) > 0:
            res.append(str(i))
    return ', '.join(res)


# g = GRFFile('', baseset=False, real_offset=0)
# # sprite_dict = g.action_5_map[0x06]
# sprite_dict = g.action_a_map
# foundations = [None] * 14
# for ofs in sorted(sprite_dict.keys()):
#     sprites = g.get_sprites(sprite_dict[ofs])
#     sprite = None
#     for s in sprites:
#         if s.bpp == grf.BPP_32 and s.zoom == grf.ZOOM_2X:
#             sprite = s
#             break
#     assert sprite is not None
#     if ofs > 1003: continue
#     foundations[ofs - 990] = sprite

# assert all(foundations)
# print('W', set(s.w for s in foundations))
# print('H', set(s.h for s in foundations))
# print('W', set(s.xofs for s in foundations))
# print('H', set(s.yofs for s in foundations))
# print([(s.xofs - (128 - s.w), s.yofs, s.w, s.h) for s in foundations])

def yoink(sprites, *, padding=0):
    sw = max(s.w for s in sprites)
    sh = max(s.h for s in sprites)
    w = padding + (sw + padding) * len(sprites)
    h = sh + padding * 2
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :] = (0, 0, 255, 255)

    x = padding
    y = padding
    for _ in sprites:
        rgba[y: y + sh, x: x + sw, :] = (0, 0, 0, 0)
        x += sw + padding

    im = Image.fromarray(rgba, mode="RGBA")
    x = padding
    y = padding
    for s in sprites:
        im.paste(s.make_rgba_image(), (x + sw - s.w, y))
        x += sw + padding

    return im


# yoink(foundations, padding=2).show()
# yoink(foundations, padding=2).save("foundations.png")
# print(foundations)
# import sys
# sys.exit(0)

# def str_availability(f):
#   if f == ALL_CLIMATES: return 'ALL'
#   res = []
#   for i, c in enumerate(('T', 'A', 'S', 'L'))

COLUMNS = 20
TABLE_X = 150
MIN_COLUMN_WIDTH = 30
MIN_ROW_HEIGHT = FONT_SIZE
padding = 5
w = 0
h = 0

grfs = [
    GRFFile('openttd.grf', baseset=False, real_offset=0),
    GRFFile('orig_extra.grf', baseset=False, real_offset=0),
]
set_types = set()
for g in grfs:
    set_types.update(g.action_5_map.keys())

full_sheet = []

for set_type in sorted(set_types):
    for g in grfs:
        if set_type in g.action_5_map:
            break

    sprite_dict = g.action_5_map[set_type]

    sheet = {}
    n_rows = 0

    for ofs in sorted(sprite_dict.keys()):
        sprites = g.get_sprites(sprite_dict[ofs])
        sprite = None
        for s in sprites:
            if s.bpp == grf.BPP_8 and s.zoom == grf.ZOOM_NORMAL:
                sprite = s
                break
        assert sprite is not None

        cx = ofs % COLUMNS
        cy = ofs // COLUMNS
        sheet[cx, cy] = sprite
        n_rows = max(cy + 1, n_rows)

    col_width = [max((MIN_COLUMN_WIDTH, *(s.w for (cx, cy), s in sheet.items() if cx == i))) for i in range(COLUMNS)]
    row_height = [max((MIN_ROW_HEIGHT, *(s.h for (cx, cy), s in sheet.items() if cy == i))) for i in range(n_rows)]

    w = max(w, sum(col_width) + (COLUMNS + 1) * padding)
    h += sum(row_height) + len(row_height) * padding + padding + FONT_SIZE + padding + padding

    full_sheet.append((set_type, sheet, col_width, row_height))

w += TABLE_X

make_check_colour = lambda c: (c // 2, c // 2, c, 255)

CHECK_1 = make_check_colour(40)
CHECK_2 = make_check_colour(80)

TEXT_COLOUR = (255, 255, 255)
CELL = 20
rgba = np.zeros((h, w, 4), dtype=np.uint8)


rgba[:, :] = CHECK_1

for y in range(2 * CELL):
    for x in range(CELL):
        ox = CELL if y >= CELL else 0
        rgba[y::2 * CELL, ox + x::2 * CELL, :] = CHECK_2

im = Image.fromarray(rgba, mode="RGBA")
# im = Image.new(mode="RGBA", size=(w, h), color=(0, 0, 255, 0))
draw = ImageDraw.Draw(im)

first = True
y = padding
for set_type, sheet, col_width, row_height in full_sheet:
    if first:
        first = False
    else:
        draw.line(((0, y), (w - 1, y)), fill=TEXT_COLOUR)
    x = padding
    y += padding
    draw.text((x, y), f'0x{set_type:02X}', font=font, fill=TEXT_COLOUR)
    ox = x + TABLE_X
    for i in range(COLUMNS):
        draw.text((ox + col_width[i] // 2, y), str(i), font=font, fill=TEXT_COLOUR, anchor='mt')
        ox += col_width[i] + padding

    y += FONT_SIZE + padding

    oy = y
    for i in range(len(row_height)):
        draw.text((x + TABLE_X - padding, oy + row_height[i] // 2), str(i * COLUMNS), font=font, fill=TEXT_COLOUR, anchor='rm')
        oy += row_height[i] + padding

    for (cx, cy), sprite in sheet.items():
        ofs = cy * COLUMNS + cx
        ox = x + TABLE_X + sum(col_width[:cx]) + padding * cx + (col_width[cx] - sprite.w) // 2
        oy = y + sum(row_height[:cy]) + padding * cy + (row_height[cy] - sprite.h) // 2
        im.alpha_composite(sprite.make_rgba_image(), (ox, oy))

    y += padding + sum(row_height) + len(row_height) * padding

im.save("action5.png")
