import json

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
OGFX_BASE_PATH = 'ogfx1_base.grf'
OGFX_BASE_PATH = 'ogfx21_base_8.grf'
# OGFX_BASE_PATH = 'trg1r.grf'
# OGFX_BASE_PATH = 'TRG1.GRF'
BASE_OPTS = {'real_offset': 0, 'pseudo_offset': 1} if OGFX_BASE_PATH[:3].upper() == 'TRG' else {}

REPLACEMENT_GRF = '../../../BonkyGFX/bonkygfx.grf'

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

# def str_availability(f):
#   if f == ALL_CLIMATES: return 'ALL'
#   res = []
#   for i, c in enumerate(('T', 'A', 'S', 'L'))

PALETTES = {
    775: 'DARK_BLUE',
    776: 'PALE_GREEN',
    777: 'PINK',
    778: 'YELLOW',
    779: 'RED',
    780: 'LIGHT_BLUE',
    781: 'GREEN',
    782: 'DARK_GREEN',
    783: 'BLUE',
    784: 'CREAM',
    785: 'MAUVE',
    786: 'PURPLE',
    787: 'ORANGE',
    788: 'BROWN',
    789: 'GREY',
    790: 'WHITE',
    791: 'BARE_LAND',
    795: 'STRUCT_BLUE',
    796: 'STRUCT_BROWN',
    797: 'STRUCT_WHITE',
    798: 'STRUCT_RED',
    799: 'STRUCT_GREEN',
    800: 'STRUCT_CONCRETE',
    801: 'STRUCT_YELLOW',
    802: 'TRANSPARENT',
    1438: 'CHURCH_RED',
    1439: 'CHURCH_CREAM',
}

houses = []
num_variations = 0
used_palettes = set()
for i, h in enumerate(data['house_specs']):
    if not h["enabled"]:
        continue
    min_year, max_year = h["min_year"], h["max_year"]
    min_year_str = str(min_year) if min_year > 0 else '*'
    max_year_str = str(max_year) if max_year < 5000000 else '*'
    years_str = min_year_str + '-' + max_year_str
    zones = str_zones(h["availability"])
    bbset = set((dt[(i << 4) + j]["subtile_x"], dt[(i << 4) + j]["subtile_y"]) for j in range(16))
    assert len(bbset) == 1
    bb = next(iter(bbset))
    print(f'#{i}: {years_str} {h["name"]} Zones: {zones} BB: {bb}')

    variations = set()
    for j in range(4):
        stages = []
        for k in range(4):
            d = dt[(i << 4) | (j << 2) | k]
            used_palettes.add(d["building_pal"])
            used_palettes.add(d["ground_pal"])
            v = tuple(d.items())
            if stages and stages[-1] == v:
                continue
            stages.append(v)
        variations.add(tuple(stages))
    houses.append((i, h, variations, years_str, zones, bb))
    num_variations += len(variations)

    for j, v in enumerate(variations):
        print(f'    Variation {j}:')
        for fd in stages:
            d = dict(fd)
            strpal = lambda p: '' if p == 0 else ':' + PALETTES[p]
            assert d["draw_proc"] in (0, 1)
            liftstr = ' LIFT' if d["draw_proc"] == 1 else ''
            print(f'        {d["ground_sprite"]}{strpal(d["ground_pal"])} {d["building_sprite"]}{strpal(d["building_pal"])} {liftstr}')

#     j.kv("min_year", hs->min_year);
#     j.kv("max_year", hs->max_year);
#     j.kv("population", hs->population);
#     j.kv("removal_cost", hs->removal_cost);
#     j.kv("name", hs->building_name);
#     j.kv("mail_generation", hs->mail_generation);
#     j.kv("flags", hs->building_flags);
#     j.kv("availability", hs->building_availability);
#     j.kv("enabled", hs->enabled);
#     j.end_dict();
# }
# j.begin_list_with_key("house_draw_tile_data");
# for (auto &d : _town_draw_tile_data) {
#     j.begin_dict();
#     j.kv("ground_sprite", d.ground.sprite);
#     j.kv("ground_pal", d.ground.pal);
#     j.kv("building_sprite", d.building.sprite);
#     j.kv("building_pal", d.building.pal);
#     j.kv("subtile_x", d.subtile_x);
#     j.kv("sublite_y", d.subtile_y);
#     j.kv("width", d.width);
#     j.kv("height", d.height);
#     j.kv("dz", d.dz);
#     j.kv("draw_proc", d.draw_proc);


ogfx = GRFFile(OGFX_BASE_PATH, baseset=True, **BASE_OPTS)
replacement = GRFFile(REPLACEMENT_GRF, baseset=False)

PALETTE_REMAPS = {}
for k in PALETTES.keys():
    PALETTE_REMAPS[k] = ogfx.get_recolour_sprite(k)

# Add fake remap that does nothing
PALETTES[0] = ''
PALETTE_REMAPS[0] = np.arange(256, dtype=np.uint8)


h = num_variations * VARIANT_HEIGHT + len(houses)
im = Image.new(mode="RGBA", size=(WIDTH, h), color=(255, 255, 255, 255))
draw = ImageDraw.Draw(im)
y = 0
for i, h, variations, years_str, zones, bb in houses:
    avl = h['availability']

    CLIMATE_SIZE = 18
    CLIMATE_OFS = font.getlength(f'#{i}') + 5
    cx, cy = PADDING + CLIMATE_OFS, y + PADDING - 3
    draw.rounded_rectangle(((cx, cy), (cx + CLIMATE_SIZE * 4 + 6, cy + CLIMATE_SIZE + 3)), outline=(0, 0, 0), radius=5)
    for j, c in enumerate(CLIMATE_COLOURS):
        ccx, ccy = cx + 2 + (CLIMATE_SIZE + 1) * j, cy + 2
        if j == 1:
            if avl & 0x2800 == 0:
                continue

            if avl & 0x800 > 0:
                draw.rounded_rectangle(((ccx, ccy), (ccx + CLIMATE_SIZE - 1, ccy + CLIMATE_SIZE // 2 + 3 - 1)), fill=c, radius=4)

            if avl & 0x2000 > 0:
                draw.rounded_rectangle(((ccx, ccy + CLIMATE_SIZE // 2 - 3), (ccx + CLIMATE_SIZE - 1, ccy + CLIMATE_SIZE - 1)), fill=c, radius=4)
        else:
            if avl & (1 << (j + 12)) == 0:
                continue
            draw.rounded_rectangle(((ccx, ccy), (ccx + CLIMATE_SIZE - 1, ccy + CLIMATE_SIZE - 1)), fill=c, radius=4)

    draw.text((PADDING, PADDING + y), f'#{i}', font=font, fill=(0, 0, 0))
    draw.text((PADDING, PADDING + LINE_HEIGHT * 1 + y), f'Years: {years_str}', font=font, fill=(0, 0, 0))
    draw.text((PADDING, PADDING + LINE_HEIGHT * 2 + y), f'Zones: {zones}', font=font, fill=(0, 0, 0))
    draw.text((PADDING, PADDING + LINE_HEIGHT * 3 + y), f'BB: {bb}', font=font, fill=(0, 0, 0))

    yb = y
    for j, stages in enumerate(variations):
        mh = 0
        for k, fd in enumerate(stages):
            d = dict(fd)
            building_id = d["building_sprite"]
            if building_id > 0:
                mh = max(mh, ogfx.get_sprites(building_id)[0].h)
        gy = y + PADDING + mh

        for k, fd in enumerate(stages):
            d = dict(fd)
            ground_id, building_id = d["ground_sprite"], d["building_sprite"]
            ground_pal_id, building_pal_id = d["ground_pal"], d["building_pal"]
            IMG_OFS = 220 + 200 * k
            if building_id > 0:
                building = ogfx.get_sprites(building_id)[0]
                building = RecolourSprite(building, PALETTE_REMAPS[building_pal_id])
                building_img = building.make_rgba_image()
                w, h = building_img.size
                im.alpha_composite(building_img, (IMG_OFS + building.xofs + 31, gy - h))
                draw.text((IMG_OFS + 64 + PADDING, gy - h),
                          f'{building_id}',
                          font=font_small, fill=(0, 0, 0))
                draw.text((IMG_OFS + 64 + PADDING, gy - h + LINE_HEIGHT // 2),
                          PALETTES[building_pal_id],
                          font=font_small, fill=(0, 0, 0))
            if d["draw_proc"] == 1:
                draw.text((IMG_OFS + 64 + PADDING, gy - h + LINE_HEIGHT),
                          'LIFT',
                          font=font_small, fill=(0, 0, 0))

            if ground_id > 0:
                ground = ogfx.get_sprites(ground_id)[0]
                ground = RecolourSprite(ground, PALETTE_REMAPS[ground_pal_id])
                ground_img = ground.make_rgba_image()
                w, h = ground_img.size
                im.alpha_composite(ground_img, (IMG_OFS, gy + 1))
                draw.text((IMG_OFS + 64 + PADDING, gy + 33 - LINE_HEIGHT // 2), str(ground_id), font=font_small, fill=(0, 0, 0))
                draw.text((IMG_OFS + 64 + PADDING, gy + 33 - LINE_HEIGHT), PALETTES[ground_pal_id], font=font_small, fill=(0, 0, 0))

        y = gy + 33
    y += PADDING
    if y - yb < 100:
        y = yb + 100
    draw.line(((0, y), (WIDTH - 1, y)), fill=(0, 0, 0))
    y += 1

im = im.crop((0, 0, WIDTH, y))
im.save('house_variations.png')

print('Palettes used:', ', '.join(v for k, v in PALETTES.items() if k in used_palettes))
# for i in 795, 796, 797, 798, 799, 800, 801:
    # remap = [PALETTE_REMAPS[i][j] for j in range(70, 80)]
    # print(f'REMAP {i}: {remap}')
for i in 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790:
    remap = [PALETTE_REMAPS[i][j] for j in range(198, 206)]
    print(f'REMAP {i}: {remap}')
# for i in PALETTES.keys():
#     print(f'REMAP {i}:')
#     for j in range(256):
#       m = PALETTE_REMAPS[i][j]
#       if m != j:
#           print(f'    {j}->{m}')

# ogfx 1&2
# 795: [144, 144, 145, 146, 147, 148, 149, 150, 151, 152]
# 796: [104, 105, 106, 107, 108, 109, 110, 111, 37, 38]
# 797: [70, 3, 5, 7, 8, 10, 11, 12, 13, 14]
# 798: [70, 178, 179, 180, 181, 162, 163, 164, 165, 166]
# 799: [70, 96, 96, 97, 98, 99, 100, 101, 102, 103]
# 800: [70, 16, 17, 18, 19, 20, 21, 22, 23, 14]
# 801: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]

# original_win
# REMAP 795: [144, 144, 145, 146, 147, 148, 149, 150, 151, 152]
# REMAP 796: [104, 105, 53, 107, 108, 109, 110, 111, 37, 38]
# REMAP 797: [70, 136, 106, 33, 40, 10, 11, 12, 13, 14]
# REMAP 798: [70, 178, 179, 180, 181, 162, 163, 164, 165, 166]
# REMAP 799: [70, 96, 96, 97, 98, 99, 100, 101, 102, 103]
# REMAP 800: [70, 16, 17, 18, 19, 20, 21, 22, 23, 14]
# REMAP 801: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
