#!/usr/bin/env python

from collections import defaultdict
from pathlib import Path
import functools
import numpy as np
import re
import sys
import struct
import traceback
import unicodedata

from PIL import Image, ImageDraw

from nml import lz77

import grf
from grf.common import hex_str, utoi8, DataReader, read_extended_byte, read_dword, read_word, PALETTE, ZOOM_NORMAL, VEHICLE_FEATURES
from grf.va2vars import VA2_VARS_INV
from grf.parser import OP_INIT, SPRITE_FLAGS
from grf.actions import ACTION0_PROPS, SpriteRef, Property, PyComment, PyCode, IntermediateSprite
from grf.sprites import WIN_TO_DOS_REMAP


RESOURCE_FOLDER = 'resources'


class ParsingContext:
    DEFAULT = 0
    GRAPHICS = 1
    SOUND = 2

    def __init__(self):
        self.type = self.DEFAULT
        self.group = None
        self._groups = {}
        self.amount = -1
        self.sprites = {}
        self.action_count = defaultdict(int)
        self.action_size = defaultdict(int)
        self.v1_sprite_id = 0

    def get_v1_sprite_id(self):
        self.v1_sprite_id += 1
        return self.v1_sprite_id - 1

    def consume(self, sprite_id=None):
        if sprite_id is not None:
            self.sprites[sprite_id] = (self.type, self.group)

        if self.type == self.DEFAULT:
            return False

        self.amount -= 1
        if self.amount <= 0:
            self.reset()

    def reset(self):
        self.type = self.DEFAULT
        self.amount = -1
        self.group = None

    def set_sound(self, amount):
        self.type = self.SOUND
        self.group = None
        self.amount = amount

    def set_graphics(self, amount, group):
        count = self._groups.get(group, 0)
        self._groups[group] = count + 1
        self.type = self.GRAPHICS
        self.amount = amount
        self.group = f'{group}_{count}'

    def get_sprite_data(self, sprite_id):
        return self.sprites.get(sprite_id, (self.DEFAULT, None))

    def count_action(self, action_id, size):
        action_type = self.type if action_id > 0xfd else self.DEFAULT
        action_key = (action_id, action_type)
        self.action_count[action_key] += 1
        self.action_size[action_key] += size

    def count_real_sprite(self, sprite_id, size):
        action_type = self.get_sprite_data(sprite_id)[0]
        action_key = (0xff, action_type)
        self.action_count[action_key] += 1
        self.action_size[action_key] += size

    def print_stats(self):
        for key in sorted(self.action_count.keys()):
            count = self.action_count.get(key, '?')
            size = self.action_size.get(key, '?')
            prefix = {
                self.DEFAULT: 'Action',
                self.GRAPHICS: 'Graphics action',
                self.SOUND: 'Sound action',
            }[key[1]]
            print(f'{prefix} 0x{key[0]:02X}: {count} times, {size} bytes total')

def str_sprite(sprite):
    sprite_id = sprite & 0x1fff
    draw = {0: 'N', 1 : 'T', 2: 'R'}[(sprite >> 14) & 3]
    color_translation = (sprite >> 16) & 0x3fff
    normal_in_transparent = bool(sprite & (1 << 30))
    sprite_type = sprite >> 31
    ntstr = ['', ' NF'][normal_in_transparent]
    return f'[{sprite_id} {draw}{ntstr} {color_translation}-{sprite_type}]'


RESOURCES = set()


def find_unique_resource_name(index, name, extension, tries=100):
    for i in range(tries):
        if i == 0:
            unique = (name + '.' + extension)
        else:
            unique = (name + f'_{i + 1}.' + extension)

        if unique not in index:
            index.add(unique)
            return unique

    raise RuntimeError(f'Exceeded the limit of {tries} tries trying to create unique resource name for "{name}"')


def gen_resource_name(index, name, extension, tries=100):
    # cut extension if it's already in the name
    if name.endswith('.' + extension):
        name = name[:-len(extension) - 1]

    # generate safe filename
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    name = re.sub(r'[-\s]+', '-', name).strip('-_')

    return find_unique_resource_name(index, name, extension, tries=tries)


def write_resource(name, extension, data, tries=100):
    folder = Path(RESOURCE_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)

    # cut extension if it's already in the name
    if name.endswith('.' + extension):
        name = name[-len(extension) - 1:]

    # generate safe filename
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    name = re.sub(r'[-\s]+', '-', name).strip('-_')

    name = find_unique_resource_name(name, extension, tries=tries)
    open(folder / name, 'wb').write(data)


def read_property(data, ofs, fmt):
    if not isinstance(fmt, str):
        if isinstance(fmt, Property):
            return fmt.read(data, ofs)
        assert callable(fmt), fmt
        return fmt(data, ofs)

    if fmt == 'B':
        return data[ofs], ofs + 1

    if fmt == 'b':
        return int.from_bytes((data[ofs],), byteorder='big', signed=True), ofs + 1

    if fmt == 'W':
        return data[ofs] | (data[ofs + 1] << 8), ofs + 2

    if fmt == 'L':
        return data[ofs: ofs + 4], ofs + 4

    if fmt == 'D':
        return read_dword(data, ofs)

    if fmt == 'B*':
        return read_extended_byte(data, ofs)

    if fmt == 'n*B':
        n = data[ofs]
        return data[ofs + 1: ofs + 1 + n], ofs + 1 + n

    if fmt == 'n*L':
        n = data[ofs]
        res = [data[ofs + 1 + i * 4: ofs + 1 + i * 4 + 4] for i in range(n)]
        return res , ofs + 1 + n * 4

    if fmt == '3*B':
        return (data[ofs], data[ofs + 1], data[ofs + 2]), ofs + 3

    if fmt == '2*L':
        return (data[ofs: ofs + 4], data[ofs + 4: ofs + 8]), ofs + 8

    if fmt == 'n*m*W':
        d = DataReader(data, ofs)
        n = d.get_byte()
        m = d.get_byte()
        res = []
        for _ in range(n):
            res.append([d.get_word() for _ in range(m)])
        return res, d.offset

    if fmt == 'n*(BB)':
        num = data[ofs]
        res = ((data[ofs + 2 * i + 1], data[ofs + 2 * i + 2]) for i in range(num))
        return res, ofs + 2 * num + 1

    if fmt == 'Layouts':
        d = DataReader(data, ofs)
        num = d.get_byte()
        size = d.get_dword()
        res = []
        for _ in range(num):
            for k in range(size):  # effectively infinite loop
                xofs = d.get_byte()
                yofs = d.get_byte()
                if xofs == 0xfe and k == 0:
                    # borrow base layout
                    raise NotImplementedError

                if xofs == 0 and yofs == 0x80:
                    break

                gfx = d.get_byte()
                if gfx == 0xfe:
                    local_tile_id = d.get_word()
                    gfx = f'todo_ref({local_tile_id})'
                elif gfx == 0xff:
                    xofs = utoi8(xofs & 0xff)
                    yofs = utoi8(yofs & 0xff)
                    gfx = None

                res.append({
                    'xofs': xofs,
                    'yofs': yofs,
                    'gfx': gfx,
                })

        return res, d.offset

    raise ValueError(f'Unknown format {fmt}')


def decode_action0(data, context):
    feature = grf.Feature(data[0])
    num_props = data[1]
    num_info = data[2]
    first_id, ofs = read_extended_byte(data, 3)
    props = {}
    for _ in range(num_props):
        prop = data[ofs]
        ofs += 1
        propdict = ACTION0_PROPS[feature]
        name, fmt, *_ = propdict[prop]
        try:
            if num_info == 1:
                value, ofs = read_property(data, ofs, fmt)
                props[name] = value
            else:
                res = []
                for _ in range(num_info):
                    value, ofs = read_property(data, ofs, fmt)
                    res.append(value)
                props[name] = res
        except Exception as e:
            raise RuntimeError(f'Error decoding property {name}: {e}')

    if num_info == 1:
        return [grf.Define(
            feature=feature,
            id=first_id,
            props=props,
        )]
    else:
        return [grf.DefineMultiple(
            feature=feature,
            first_id=first_id,
            count=num_info,
            props=props,
        )]


def decode_action1(data, context):
    sprite_count, _ = read_extended_byte(data, 2)
    feature = grf.Feature(data[0])
    set_count = data[1]
    context.set_graphics(set_count * sprite_count, feature.name)
    if set_count == 1:
        return [grf.SpriteSet(
            feature=feature,
            count=sprite_count,
        )]
    else:
        return [grf.Action1(
            feature=feature,
            set_count=set_count,
            sprite_count=sprite_count
        )]


SPRITE_GROUP_OP = [
    'ADD',   # a + b
    'SUB',   # a - b
    'SMIN',  # (signed) min(a, b)
    'SMAX',  # (signed) max(a, b)
    'UMIN',  # (unsigned) min(a, b)
    'UMAX',  # (unsigned) max(a, b)
    'SDIV',  # (signed) a / b
    'SMOD',  # (signed) a % b
    'UDIV',  # (unsigned) a / b
    'UMOD',  # (unsigned) a & b
    'MUL',   # a * b
    'AND',   # a & b
    'OR',    # a | b
    'XOR',   # a ^ b
    'STO',   # store a into temporary storage, indexed by b. return a
    'RST',   # return b
    'STOP',  # store a into persistent storage, indexed by b, return a
    'ROR',   # rotate a b positions to the right
    'SCMP',  # (signed) comparison (a < b -> 0, a == b = 1, a > b = 2)
    'UCMP',  # (unsigned) comparison (a < b -> 0, a == b = 1, a > b = 2)
    'SHL',   # a << b
    'SHR',   # (unsigned) a >> b
    'SAR',   # (signed) a >> b
]


# TLF_NOTHING           = 0x00
# TLF_DODRAW            = 0x01  # Only draw sprite if value of register TileLayoutRegisters::dodraw is non-zero.
# TLF_SPRITE            = 0x02  # Add signed offset to sprite from register TileLayoutRegisters::sprite.
# TLF_PALETTE           = 0x04  # Add signed offset to palette from register TileLayoutRegisters::palette.
# TLF_CUSTOM_PALETTE    = 0x08  # Palette is from Action 1 (moved to SPRITE_MODIFIER_CUSTOM_SPRITE in palette during loading).
# TLF_BB_XY_OFFSET      = 0x10  # Add signed offset to bounding box X and Y positions from register TileLayoutRegisters::delta.parent[0..1].
# TLF_BB_Z_OFFSET       = 0x20  # Add signed offset to bounding box Z positions from register TileLayoutRegisters::delta.parent[2].
# TLF_CHILD_X_OFFSET    = 0x10  # Add signed offset to child sprite X positions from register TileLayoutRegisters::delta.child[0].
# TLF_CHILD_Y_OFFSET    = 0x20  # Add signed offset to child sprite Y positions from register TileLayoutRegisters::delta.child[1].
# TLF_SPRITE_VAR10      = 0x40  # Resolve sprite with a specific value in variable 10.
# TLF_PALETTE_VAR10     = 0x80  # Resolve palette with a specific value in variable 10.
# TLF_KNOWN_FLAGS       = 0xFF  # Known flags. Any unknown set flag will disable the GRF.

# # /** Flags which are still required after loading the GRF. */
# TLF_DRAWING_FLAGS     = ~TLF_CUSTOM_PALETTE

# # /** Flags which do not work for the (first) ground sprite. */
# TLF_NON_GROUND_FLAGS  = TLF_BB_XY_OFFSET | TLF_BB_Z_OFFSET | TLF_CHILD_X_OFFSET | TLF_CHILD_Y_OFFSET

# # /** Flags which refer to using multiple action-1-2-3 chains. */
# TLF_VAR10_FLAGS       = TLF_SPRITE_VAR10 | TLF_PALETTE_VAR10

# # /** Flags which require resolving the action-1-2-3 chain for the sprite, even if it is no action-1 sprite. */
# TLF_SPRITE_REG_FLAGS  = TLF_DODRAW | TLF_SPRITE | TLF_BB_XY_OFFSET | TLF_BB_Z_OFFSET | TLF_CHILD_X_OFFSET | TLF_CHILD_Y_OFFSET

# # /** Flags which require resolving the action-1-2-3 chain for the palette, even if it is no action-1 palette. */
# TLF_PALETTE_REG_FLAGS = TLF_PALETTE

def read_sprite_layout_registers(d, flags, is_parent):
    # regs = {'flags': flags & TLF_DRAWING_FLAGS}
    regs = {}
    for flag, (fparent, mask, size, conversion) in SPRITE_FLAGS.items():
        if flags & mask == 0:
            continue
        if fparent is not None and fparent != is_parent:
            continue

        if size == 0:
            value = True
        elif size == 1:
            value = d.get_byte()
        elif size == 2:
            value = (d.get_byte(), d.get_byte())
        else:
            raise RuntimeError(f'Incorrect size of sprite register {flag}: {size}')

        if conversion:
            value = conversion(value)
        regs[flag] = value

    return regs


def read_sprite_layout(d, feature, ref_id, num, basic_format):
    assert feature in (grf.HOUSE, grf.INDUSTRY_TILE, grf.OBJECT, grf.INDUSTRY), feature

    # TODO a lot of duplicate code with read_sprite_layout in actions.py
    has_z_position = not basic_format
    has_flags = bool(num & 0x40)
    num &= 0x3f

    def read_sprite():
        sprite = d.get_dword()
        flags = d.get_word() if has_flags else 0
        return SpriteRef.from_grf(sprite, global_if_flagged=False), flags

    ground_sprite, ground_flags = read_sprite()
    ground = {'sprite': ground_sprite}
    ground.update(read_sprite_layout_registers(d, ground_flags, False))
    sprites = []
    for _ in range(num):
        sprite, flags = read_sprite()
        seq = {'sprite': sprite}
        offset = (d.get_byte(), d.get_byte(), d.get_byte() if has_z_position else 0)
        is_parent = (offset[2] != 0x80)
        if is_parent:
            seq['offset'] = offset
            seq['extent'] = (d.get_byte(), d.get_byte(), d.get_byte())
        else:
            seq['pixel_offset'] = (offset[0], offset[1])
        seq.update(read_sprite_layout_registers(d, flags, is_parent))
        sprites.append(seq)

    if basic_format:
        assert len(sprites) == 1, len(sprites)
        return [grf.BasicSpriteLayout(
            feature=feature,
            ref_id=ref_id,
            ground=ground,
            building=sprites[0],
        )]

    return [[grf.ExtendedSpriteLayout, grf.AdvancedSpriteLayout][has_flags](
        feature=feature,
        ref_id=ref_id,
        ground=ground,
        buildings=sprites,
    )]


VA2_GLOBALS = {
    0x00: ('date', 'W'),  # 80      W   current date (counted as days from 1920)[1]
    0x01: ('year', 'B'),  # 81      B   ￼0.6 ￼2.0   Current year (count from 1920, max. 2175 even with eternalgame)[1]
    0x02: ('month', 'B/D'),  # 82      B/D ￼0.6 ￼2.0   current month (0-11) in bits 0-7; the higher bytes contain unusable junk.[1] ￼0.7 ￼ Since OpenTTD r13594 'day of month' (0-30) is stored in bits 8-12, bit 15 is set in leapyears and 'day of year'(0-364 resp. 365) is stored in bits 16-24. All other bits are reserved and should be masked.
    0x03: ('climate', 'B'),  # 83      B   ￼0.6 ￼2.0   Current climate: 00 = temp, 01 = arctic, 02 = trop, 03 = toyland
                      # 84      D   ￼0.6 ￼2.0   GRF loading stage, see below
                      # 85      B   ￼0.6 ￼2.0   TTDPatch flags: only for bit tests
    0x06: ('drive_side', 'B'),  # 86      B   ￼0.6 ￼2.0   Road traffic side: bit 4 clear=left, set=right; other bits are reserved and must be masked. (87)    (87)    B   ￼ ￼ No longer used since TTDPatch 2.0. (was width of "€" character)
                      # 88      4*B   ￼0.6 ￼2.0   Checks specified GRFID (see condition-types)[2]
    0x09: ('date_fract', 'W'),  # 89      W   ￼0.6 ￼2.0   date fraction, incremented by 0x375 every engine tick
    0x0A: ('anim_counter', 'W'),  # 8A      W   ￼0.6 ￼2.0   animation counter, incremented every tick
    0x0B: ('ttdp_version', 'D'),  # 8B      D   ￼ ￼2.0  TTDPatch version, see below [3][4]
    0x0C: ('cur_cb_id', 'W'),  #         W   ￼0.6 ￼2.5   current callback ID (feature-specific), set to 00 when not in a callback
    0x0D: ('ttd_version', 'B'),  # 8D      B   ￼0.6 ￼2.5   TTD version, 0=DOS, 1=Windows
    0x0E: ('train_y_ofs', 'B'),  # 8E  8E  B   ￼0.6 ￼2.5   Y-Offset for train sprites
    0x0F: ('rail_cost', '3*B'),  # 8F  8F  3*B ￼0.6 ￼2.5   Rail track type cost factors
    0x10: ('cb_info1', 'D'),  #         D   ￼0.6 ￼2.5   Extra callback info 1, see below.
    0x11: ('cur_rail_tool', 'B'),  #         B   ￼ ￼2.5  current rail tool type (for station callbacks)
    0x12: ('game_mode', 'B'),  # 92      B   ￼0.6 ￼2.5   Game mode, 0 in title screen, 1 in game and 2 in editor
    0x13: ('tile_refresh_left', 'W'),  # 93  93  W   ￼ ￼2.5  Tile refresh offset to left [5]
    0x14: ('tile_refresh_right', 'W'),  # 94  94  W   ￼ ￼2.5  Tile refresh offset to right [5]
    0x15: ('tile_refresh_up', 'W'),  # 95  95  W   ￼ ￼2.5  Tile refresh offset upwards [5]
    0x16: ('tile_refresh_down', 'W'),  # 96  96  W   ￼ ￼2.5  Tile refresh offset downwards [5]
                      # 97  97  B   ￼ ￼2.5  Fixed snow line height [6][7]
    0x18: ('cb_info2', 'D'),  #         D   ￼0.6 ￼2.5   Extra callback info 2, see below.
                      # 99  99  D   ￼ ￼2.5  Global ID offset
    0x1A: ('max_uint32', 'D'),  # 9A      D   ￼0.6 ￼2.5   Has always all bits set; you can use this to make unconditional jumps
    0x1B: ('display_options', 'B'),  #         B   ￼ ￼2.5  display options; bit 0=town names, 1=station names, 2=signs, 3=animation, 4=transparency, 5=full detail
    0x1C: ('va2_ret', 'D'),  #         D   ￼0.6 ￼2.5   result from most recent VarAction2
    0x1D: ('ttd_platform', 'D'),  # 9D      D   ￼0.6 ￼2.5   TTD Platform, 0=TTDPatch, 1=OpenTTD [4]
    0x1E: ('grf_featuers', 'D'),  # 9E  9E  D   ￼0.6 ￼2.5   Misc. GRF Features
                      # 9F  D   ￼ ￼2.5  writable only: Locale-dependent settings
    0x20: ('snow_line', 'B'),  #         B   ￼0.6 ￼2.5   Current snow line height, FFh if snow isn't present at all [7]
    0x21: ('openttd_version', 'D'),  # A1      D   ￼0.6 ￼  OpenTTD version, see below. [4]
    0x22: ('difficulty_level', 'D'),  # A2      D   ￼0.7 ￼2.6   Difficulty level: 00= easy, 01=medium, 02=hard, 03=custom
    0x23: ('date_long', 'D'),  # A3      D   ￼0.7 ￼2.6   Current date long format
    0x24: ('year_zero', 'D'),  # A4      D   ￼0.7 ￼2.6   Current year zero based
    0x25: ('a3_grfid', 'D'),  #         D   ￼0.7 ￼  GRFID of the grf that contains the corresponding Action3. Useful when accessing the "related" object. Currently only supported for vehicles.
}

V2_OBJECT_VARS = {
    0x40: ('relative_pos', 'D'),  # Relative position, like Industry Tile var43
    0x41: ('tile_info', 'W'),  # Tile information, see below
    0x42: ('constructed', 'D'),  # Construction date from year 0
    0x43: ('anim_counter', 'B'),  # Animation counter, see below
    0x44: ('founder', 'B'),  # Object founder information
    0x45: ('closest_town_info', 'D'),  # Get town zone and Manhattan distance of closest town
    0x46: ('closest_town_dist_squared', 'D'),  # Get square of Euclidian distance of closest town
    0x47: ('colour', 'B'),  # Object colour
    0x48: ('views', 'B'),  # Object views
    0x60: ('type_view_ofs', 'W'),  # Get object type and view at offset
    0x61: ('random_ofs', 'B'),  # Get random bits at offset
    0x62: ('nearby_tile_info', 'D'),  # Land info of nearby tiles
    0x63: ('nearby_anim_counter', 'W'),  # Animation counter of nearby tile
    0x64: ('object_count', 'D'),  # Count of object, distance of closest instance
}

VA2_OP = {
    0x00: '+',  # \2+ result = val1 + val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x01: '-',  # \2- result = val1 - val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x02: 'min',  # \2< result = min(val1, val2)    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    val1 and val2 are both considered signed
    0x03: 'max',  # \2> result = max(val1, val2)    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x04: 'min',  # \2u<    result = min(val1, val2)    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    val1 and val2 are both considered unsigned
    0x05: 'max',  # \2u>    result = max(val1, val2)    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x06: '/',  #  \2/ result = val1 / val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    val1 and val2 are both considered signed
    0x07: '%',  #  \2% result = val1 mod val2  Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x08: 'u/',  #  \2u/    result = val1 / val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    val1 and val2 are both considered unsigned
    0x09: 'u%',  #  \2u%    result = val1 mod val2  Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5
    0x0A: '*',  #   \2* result = val1 * val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    result will be truncated to B/W/D (that makes it the same for signed/unsigned operands)
    0x0B: '&',  #   \2& result = val1 & val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    bitwise AND
    0x0C: '|',  #   \2| result = val1 | val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    bitwise OR
    0x0D: '^',  #   \2^ result = val1 ^ val2    Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.52.5    bitwise XOR
    0x0E: '(tsto)',  #  \2s or \2sto [1]    var7D[val2] = val1, result = val1   Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1246)2.6    Store result. See Temporary storage.
    0x0F: ';',  #   \2r or \2rst [1]    result = val2 [2]   Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1246)2.6
    0x10: '(psto)',  #  \2psto [3]  var7C[val2] = val1, result = val1   Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1315)2.6    Store result into persistent storage. See Persistent storage.
    0x11: '(ror)',  #  \2ror or \2rot [4]  result = val1 rotate right val2 Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1651)2.6    Always a 32-bit rotation.
    0x12: '(cmp)',  #  \2cmp [3]   see notes   Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1698)2.6    Result is 0 if val1<val2, 1 if val1=val2 and 2 if val1>val2. Both values are considered signed. [5]
    0x13: '(ucmp)',  #  \2ucmp [3]  see notes   Supported by OpenTTD 0.60.6 Supported by TTDPatch 2.6 (r1698)2.6    The same as 12, but operands are considered unsigned. [5]
    0x14: '<<',  #  \2<< [3]    result = val1 << val2   Supported by OpenTTD 1.1 (r20332)1.1 Supported by TTDPatch 2.6 (r2335)2.6   shift left; val2 should be in the range 0 to 31.
    0x15: 'u>>',  # \2u>> [3]   result = val1 >> val2   Supported by OpenTTD 1.1 (r20332)1.1 Supported by TTDPatch 2.6 (r2335)2.6   shift right (unsigned); val2 should be in the range 0 to 31.
    0x16: '>>',  #  \2>> [3]    result = val1 >> val2   Supported by OpenTTD 1.1 (r20332)1.1 Supported by TTDPatch 2.6 (r2335)2.6   shift right (signed); val2 should be in the range 0 to 31.
}


def get_va2_var(var):
    if var < 0x40:
        name, fmt = VA2_GLOBALS[var]
        return f'[{name}]', fmt
    if var == 0x5f: return '(random)', 'D'
    if var == 0x7b: return '(var_eval)', ''
    if var == 0x7c: return '(perm)', 'D'
    if var == 0x7d: return '(temp)', 'D'
    if var == 0x7e: return '(call)', 'D'
    if var == 0x7f: return '(param)', 'D'
    return V2_OBJECT_VARS[var]


def get_var_node(feature, root, var, param, shift, and_mask, node_type, add_val=None, divmod_val=None):
    if add_val is not None or divmod_val is not None:
        return grf.GenericVar(var, shift, and_mask, node_type, add_val, divmod_val, param=param)
    if var == 0x1a and shift == 0:
        return grf.Value(and_mask)

    if var == 0x7b:
        return get_var_node(feature, root, param.value, root, shift, and_mask, node_type, add_val, divmod_val)
    if (var, shift, and_mask) == (0x7c, 0, 0xffffffff):
        return grf.Perm(param)
    if (var, shift, and_mask) == (0x7d, 0, 0xffffffff):
        return grf.Temp(param)
    if (var, shift, and_mask) == (0x7e, 0, 0xffffffff):
        assert isinstance(param, grf.Value)
        return grf.Call(param.value)

    assert feature in VA2_VARS_INV, feature
    var_name = VA2_VARS_INV[feature].get((var, shift, and_mask))
    if var_name is not None:
        return grf.Var(feature, var_name, param=param)

    return grf.GenericVar(var, shift, and_mask, 0, None, None, param=param)


def read_random_switch(d, feature, atype, ref_id):
    scope = {0x80: 'self', 0x83: 'parent', 0x84: 'relative'}[atype]
    count = None
    if scope == 'relative' and feature in VEHICLE_FEATURES:
        count = d.get_byte()
    triggers_byte = d.get_byte()
    triggers = triggers_byte & 0x7f
    cmp_all = bool(triggers_byte & 0x80)
    lowest_bit = d.get_byte()
    num_groups = d.get_byte()
    groups = [grf.Ref(d.get_word()) for _ in range(num_groups)]
    return [grf.RandomSwitch(
        feature=feature,
        ref_id=ref_id,
        scope=scope,
        count=count,
        triggers=triggers,
        cmp_all=cmp_all,
        lowest_bit=lowest_bit,
        groups=groups,
    )]


def decode_action2(data, context):
    feature = grf.Feature(data[0])
    ref_id = data[1]
    atype = data[2]
    d = DataReader(data, 3)

    # VarAction2
    if atype in (0x81, 0x82, 0x85, 0x86, 0x89, 0x8a):
        group_size = (atype >> 2) & 3
        related_scope = bool(atype & 2)
        first = True
        ofs = 3
        code = []
        accumulator = None
        while True:
            op = 0 if first else d.get_byte()
            var = d.get_byte()
            param = None
            if 0x60 <= var < 0x80:
                param = grf.Value(d.get_byte())
            varadj = d.get_byte()
            shift = varadj & 0x1f
            has_more = bool(varadj & 0x20)
            node_type = varadj >> 6
            and_mask = d.get_var(group_size)
            add_val = None
            divmod_val = None
            if node_type != 0:
                # old magic, use advaction2 instead
                add_val = d.get_var(group_size)
                divmod_val = d.get_var(group_size)

            node = get_var_node(feature, accumulator, var, param, shift, and_mask, node_type, add_val, divmod_val)

            if first:
                accumulator = node
            elif op == OP_INIT:
                if var != 0x7b:
                    code.append(accumulator)
                accumulator = node
            else:
                accumulator = grf.Expr(op, accumulator, node)

            first = False
            if not has_more:
                break

        code.append(accumulator)

        # no ranges is special for "do not switch, return the switch value"
        # <frosch123> oh, also, the ranges are unsigned
        # <frosch123> so if you want to set -5..5 you have to split into two ranges -5..-1, 0..5
        n_ranges = d.get_byte()
        ranges = []
        for _ in range(n_ranges):
            group = d.get_word()
            low = d.get_var(group_size)
            high = d.get_var(group_size)
            ranges.append(grf.Range(low, high, grf.Ref(group)))

        default_group = grf.Ref(d.get_word())

        for c in code:
            c.simplify()

        return [grf.Switch(
            feature=feature,
            ref_id=ref_id,
            related_scope=related_scope,
            ranges=ranges,
            default=default_group,
            code='\n'.join(x.format() for x in code),
        )]

    # Random switch
    if atype in (0x80, 0x83, 0x84):
        return read_random_switch(d, feature, atype, ref_id)

    if feature in (grf.HOUSE, grf.INDUSTRY_TILE, grf.OBJECT, grf.AIRPORT_TILE):
        num_ent1 = atype
        # if num_ent1 == 0:
        #     ground_sprite, building_sprite, xofs, yofs, xext, yext, zext = struct.unpack_from('<IIBBBBB', data, offset=3)
        #     ground_sprite = str_sprite(ground_sprite)
        #     building_sprite = str_sprite(building_sprite)
        #     return []

        # if 0 < num_ent1 <= 0x3f:
        #     raise NotImplementedError

        return read_sprite_layout(d, feature, ref_id, max(num_ent1, 1), num_ent1 == 0)
        # num_loaded = num_ent1
        # num_loading = get_byte()
        # [get_word() for i in range(num_loaded)]
        # [get_word() for i in range(num_loading)]
        # assert False, num_ent1
        # # assert num_ent1 < 0x3f + 0x40, num_ent1
        # return

    # Special Industry production callback format
    if feature == grf.INDUSTRY:
        version = atype
        if version < 2:
            getter = [d.get_word, d.get_byte][version]
            inputs = [getter(), getter(), getter()]
            outputs = [getter(), getter()]
        elif version == 2:
            num_input = d.get_byte()
            inputs = []
            for _ in range(num_input):
                inputs.append((d.get_byte(), d.get_byte()))
            num_output = d.get_byte()
            outputs = []
            for _ in range(num_output):
                outputs.append((d.get_byte(), d.get_byte()))
        else:
            raise NotImplementedError
        do_again = d.get_byte()
        return [grf.IndustryProductionCallback(
            inputs=inputs,
            outputs=outputs,
            do_again=do_again,
            version=version,
        )]


    # print('VA2', feature)
    # raise NotImplementedError
    num_ent1 = atype
    num_ent2 = data[3]
    # print('NE', num_ent2, feature)
    ent1 = struct.unpack_from('<' + 'H' * num_ent1, data, offset=4)
    ent2 = struct.unpack_from('<' + 'H' * num_ent2, data, offset=4 + 2 * num_ent1)
    # print(f'ent1:{ent1} ent2:{ent2}')
    return [grf.GenericSpriteLayout(
        feature=feature,
        ref_id=ref_id,
        ent1=ent1,
        ent2=ent2,
    )]


def decode_action3(data, context):
    feature = grf.Feature(data[0])
    idcount = data[1]
    wagon_override = bool(idcount & 0x80)
    if wagon_override:
        idcount &= 0x7f
    objs = []
    maps = {}
    if data[1] == 0:
        _, default = struct.unpack_from('<BH', data, offset=2)
    else:
        d = DataReader(data, 2)
        objs = [d.get_extended_byte() for _ in range(idcount)]
        cidcount = d.get_byte()
        for _ in range(cidcount):
            ctype = d.get_byte()
            groupid = grf.Ref(d.get_word())
            maps[ctype] = groupid
        default = d.get_word()
        # print(f'objs:{objs} maps:{maps} default_gid:{def_gid}')
    return [grf.Action3(
        feature=feature,
        ids=objs,
        maps=maps,
        default=grf.Ref(default),
    )]


def decode_action4(data, context):
    feature = grf.Feature(data[0])
    is_generic = bool(data[1] & 0x80)
    if is_generic:  # generic word offset flag
        fmt = '<BBBH'
    elif feature in VEHICLE_FEATURES and data[3] == 0xff:  # vehicles have extended byte offset
        fmt = '<BBBxH'
    else:  # default offset is byte
        fmt = '<BBBB'
    _, lang, num, offset = struct.unpack_from(fmt, data)

    str_data = data[struct.calcsize(fmt):-1]
    strings = [s for s in str_data.split(b'\0')] if str_data else [b'']
    assert len(strings) == num, (len(strings), num,)
    return [grf.DefineStrings(
        feature=feature,
        lang=lang & 0x7f,
        offset=offset,
        is_generic_offset=is_generic,
        strings=strings,
    )]


def decode_action5(data, context):
    t = data[0]
    offset = None
    num, dataofs = read_extended_byte(data, 1)
    if t & 0xf0:
        offset, _ = read_extended_byte(data, dataofs)
        t &= ~0xf0
    context.set_graphics(num, 'action5')
    return [grf.ReplaceNewSprites(t, num, offset=offset)]


def decode_action6(data, context):
    d = DataReader(data, 0)
    params = []
    while True:
        param_num = d.get_byte()
        if param_num == 0xFF:
            break
        param_size = d.get_byte()
        offset = d.get_extended_byte()
        params.append({'num': param_num, 'size': param_size, 'offset': offset})
    return [grf.ModifySprites(params)]


def decode_action7or9(data, is_static):
    variable = data[0]
    varsize = data[1]
    condition = data[2]
    if condition < 2:
        varsize = 1
    value = 0
    for i in range(varsize):
        value += data[3 + i] << (8 * i)
    skip = data[3 + varsize]
    return [grf.If(
        is_static=is_static,
        variable=variable,
        varsize=varsize,
        condition=condition,
        value=value,
        skip=skip,
    )]


def decode_action7(data, context):
    return decode_action7or9(data, False)


def decode_action8(data, context):
    format_version = data[0]
    grfid = data[1: 5]
    pos = data.find(b'\0', 5)
    name = data[5: pos]
    description = data[pos + 1: -1]
    return [grf.SetDescription(
        grfid=grfid,
        name=name,
        description=description,
        format_version=format_version,
    )]


def decode_action9(data, context):
    return decode_action7or9(data, True)


def decode_actionA(data, context):
    num = data[0]
    sets_flipped = [struct.unpack_from('<BH', data, offset=3 * i + 1) for i in range(num)]
    sets = [(first, num) for num, first in sets_flipped]
    context.set_graphics(sum(x[1] for x in sets), 'action_a')
    return [grf.ReplaceOldSprites(sets)]


def decode_actionB(data, context):
    severity = data[0]
    lang = data[1]
    message = data[2]
    text_param = None
    res = []
    ofs = 3
    if message == 0xFF:
        message_end = data.find(b'\0', 3)
        if message_end == -1:
            res.append(PyComment('Error: Action 0x0B message has no terminal 00 byte.'))
        else:
            message = data[3: message_end]
            ofs = message_end + 1
    if ofs < len(data):
        text_end = data.find(b'\0', ofs)
        if text_end == -1:
            res.append(PyComment('Error: Action 0x0B text parameter has no terminal 00 byte.'))
        else:
            text_param = data[ofs: text_end]
            ofs = text_end + 1
        params = list(data[ofs: ])
    res.append(grf.ErrorMessage(
        severity=severity,
        lang=lang,
        message=message,
        text_param=text_param,
        params=params,
    ))
    return res


def decode_actionC(data, context):
    return [grf.Comment(data)]


OPERATIONS = {
    0x00: '{target} = {source1}', # Supported by OpenTTD Supported by TTDPatch  Assignment  target = source1
    0x01: '{target} = {source1} + {source2}', # Supported by OpenTTD Supported by TTDPatch  Addition    target = source1 + source2
    0x02: '{target} = {source1} - {source2}', # Supported by OpenTTD Supported by TTDPatch  Subtraction     target = source1 - source2
    0x03: '{target} = {source1} * {source2} (Unsigned)', # Supported by OpenTTD Supported by TTDPatch  Unsigned multiplication     target = source1 * source2, with both sources being considered to be unsigned
    0x04: '{target} = {source1} * {source2} (Signed)', # Supported by OpenTTD Supported by TTDPatch  Signed multiplication   target = source1 * source2, with both sources considered signed
    0x05: '{target} = {source1} <</>> {source2} (Unsigned)', # Supported by OpenTTD Supported by TTDPatch  Unsigned bit shift  target = source1 << source2 if source2>0, or target = source1 >> abs(source2) if source2 < 0. source1 is considered to be unsigned
    0x06: '{target} = {source1} <</>> {source2} (Signed)', # Supported by OpenTTD Supported by TTDPatch  Signed bit shift    same as 05, but source1 is considered signed)
    0x07: '{target} = {source1} & {source2}', # Supported by OpenTTD Supported by TTDPatch 2.5 (alpha 48)2.5    Bitwise AND     target = source1 AND source2
    0x08: '{target} = {source1} | {source2}', # Suported by OpenTTD Supported by TTDPatch 2.5 (alpha 48)2.5    Bitwise OR  target = source1 OR source2
    0x09: '{target} = {source1} / {source2} (Unsigned)', # Supported by OpenTTD Supported by TTDPatch 2.5 (alpha 59)2.5    Unsigned division   target = source1 / source2
    0x0A: '{target} = {source1} / {source2} (Signed)', # Supported by OpenTTD Supported by TTDPatch 2.5 (alpha 59)2.5    Signed division     target = source1 / source2
    0x0B: '{target} = {source1} % {source2} (Unsigned)', # Supported by OpenTTD Supported by TTDPatch 2.5 (alpha 59)2.5    Unsigned modulo     target = source1 % source2
    0x0C: '{target} = {source1} % {source2} (Signed)', # Supported by OpenTTD Supported by TTDPatch 2.5 (alpha 59)2.5    Signed modulo   target = source1 % source2
}


def decode_actionD(data, context):
    target = data[0]
    operation = data[1]
    if_undefined = bool(operation & 0x80)
    operation &= 0x7f
    source1 = data[2]
    source2 = data[3]
    value = None
    if source1 == 0xff or source2 == 0xff:
        value, _ = read_dword(data, 4)
    fmt = OPERATIONS[operation]
    sf = lambda x: f'[{x:02x}]' if x != 0xff else str(value)
    target_str = f'[{target:02x}]'
    op_str = fmt.format(target=target_str, source1=sf(source1), source2=sf(source2))
    return [
        PyComment(op_str),
        grf.ComputeParameters(
            target=target,
            operation=operation,
            if_undefined=if_undefined,
            source1=source1,
            source2=source2,
            value=value,
        )
    ]


def decode_action10(data, context):
    label = data[0]
    comment = data[1:]
    return [grf.Label(label, comment)]


def decode_action11(data, context):
    number, _ = read_word(data, 0)
    context.set_sound(number)
    return [grf.SoundEffects(number)]


def decode_action12(data, context):
    count = data[0]
    d = DataReader(data, 1)
    glyphs = []
    for i in range(count):
        glyphs.append({
            'font': d.get_byte(),
            'count': d.get_byte(),
            'base': d.get_word(),
        })
    return [grf.UnicodeGlyphs(glyphs)]


def decode_action13(data, context):
    grfid = data[:4]
    # TODO lang only for grf >= 8
    lang = data[4]
    count = data[5]
    offset, _ = read_word(data, 6)
    str_data = data[8:]
    strings = [s for s in str_data.split(b'\0')] if str_data else []

    return [grf.Transtations(
        grfid=grfid,
        lang=lang,
        offset=offset,
        strings=strings,
    )]


def decode_action14(data, context):
    res = {}
    ofs = 0

    def decode_chunk(res):
        nonlocal ofs
        chunk_type = data[ofs]
        ofs += 1
        if chunk_type == 0: return False
        chunk_id = data[ofs: ofs + 4]
        ofs += 4
        if chunk_type == b'C'[0]:
            res[chunk_id] = {}
            while decode_chunk(res[chunk_id]):
                pass
        elif chunk_type == b'B'[0]:
            l = data[ofs] | (data[ofs + 1] << 8)
            res[chunk_id] = data[ofs + 2: ofs + 2 + l]
            ofs += 2 + l
        elif chunk_type == b'T'[0]:
            lang = data[ofs]
            text_end = data.find(b'\0', ofs + 1)
            text = data[ofs + 1 : text_end]
            res[chunk_id] = (lang, text)
            ofs = text_end + 1
        else:
            assert False, chunk_type
        return True

    while decode_chunk(res):
        pass

    return [grf.SetProperties(res)]


def decode_sound_actionFE(data):
    grfid = data[1:5]
    number, _ = read_word(data, 5)
    return [grf.ImportSound(grfid, number)]


def decode_sound_actionFF(data):
    l = data[1]
    name = data[2: l + 2].decode()
    wav = data[l + 3:]
    return [RealSoundSprite(sprite_id)]


ACTIONS = {
    0x00: decode_action0,
    0x01: decode_action1,
    0x02: decode_action2,
    0x03: decode_action3,
    0x04: decode_action4,
    0x05: decode_action5,
    0x06: decode_action6,
    0x07: decode_action7,
    0x08: decode_action8,
    0x09: decode_action9,
    0x0a: decode_actionA,
    0x0b: decode_actionB,
    0x0c: decode_actionC,
    0x0d: decode_actionD,
    0x10: decode_action10,
    0x11: decode_action11,
    0x12: decode_action12,
    0x13: decode_action13,
    0x14: decode_action14,
}

SOUND_ACTIONS = {
    0xfe: decode_sound_actionFE,
    0xff: decode_sound_actionFF,
}

GRAPHICS_ACTIONS = {
}

def decode_pseudo_sprite(data, context):
    res = []
    action_id = data[0]
    if context.type == ParsingContext.SOUND:
        decoder = SOUND_ACTIONS.get(action_id)
        if decoder is not None:
            res.extend(decoder(data[1:]))
            context.consume()
            return res
        res.append(PyComment(f'Unexpected action 0x{action_id:02x} in sound context, reverting to global'))
        context.reset()
    elif context.type == ParsingContext.SOUND:
        decoder = GRAPHICS_ACTIONS.get(data[0])
        if decoder is not None:
            res.extend(decoder(data[1:]))
            context.consume()
            return res
        res.append(PyComment(f'Unexpected action 0x{action_id:02x} in graphics context, reverting to global'))
        context.reset()

    decoder = ACTIONS.get(data[0])
    if decoder:
        res.extend(decoder(data[1:], context))
    else:
        res.append(PyComment(f'Unexpected action 0x{data[0]:02x}'))
    return res


def read_pseudo_sprite(f, nfo_line, container, context):
    size_bytes = 2 if container < 2 else 4
    data = f.read(size_bytes)

    # if not data and container == 1:
    #     return False, []
    l = struct.unpack('<H' if container < 2 else '<I', data)[0]
    if l == 0:
        return False, [PyComment('End of pseudo sprites')], None
    grf_type = f.read(1)[0]
    grf_type_str = hex(grf_type)[2:]
    res = []

    # ofs = f.tell()
    # data = f.read(l)
    # print(f'{nfo_line}: Sprite({l}, {grf_type_str}) <{data[0]:02x}>: {hex_str(data, 100)}')
    # f.seek(ofs)

    if grf_type == 0xff:
        data = f.read(l)
        if l == 1:  # Some NewGRF files have "empty" pseudo-sprites which are 1 byte long.
            return True, [], None
        # print(f'{nfo_line}: Sprite({l}, {grf_type_str}) <{data[0]:02x}>: {hex_str(data, 100)}')
        context.count_action(data[0], l + size_bytes * 2 + 1)
        res.append(PyComment(f'{nfo_line}: Sprite({l}, {grf_type_str}) <{data[0]:02x}>: {hex_str(data, 100)}'))
        # if container == 1:
        #     sprite = RealRemapSprite(context.get_v1_sprite_id(), data)
        #     res.append(SpritePlaceholder(sprite.id))
        #     return True, res, sprite
        try:
            res.extend(decode_pseudo_sprite(data, context))
        except Exception as e:
            res.append(PyComment(f'Error decoding sprite:'))
            estr = traceback.format_exc()
            for l in estr.split('\n'):
                res.append(PyComment(l))
    elif container >= 2 and grf_type == 0xfd:
        data = f.read(l)
        context.count_action(0xfd, l + size_bytes * 2 + 1)
        sprite_id = struct.unpack('<I', data)[0]
        res.append(PyComment(f'{nfo_line}: Sprite({l}, {grf_type_str}): {hex_str(data)} ({sprite_id})'))
        res.append(SpritePlaceholder(sprite_id))
        context.consume(sprite_id)
    else:
        if container == 1:
            context.count_action(0xff, l + size_bytes * 2 + 1)
            height, width, xofs, yofs = struct.unpack('<BHhh', f.read(7))
            zoom = ZOOM_NORMAL
            if grf_type & 0x02:
                num = height * width
            else:
                num = l - 8
            sprite_type = (grf_type & 0x08) | 0x4
            sprite = RealGraphicsSprite(context.get_v1_sprite_id(), f.tell(), num, width, height, sprite_type, zoom, xofs, yofs)
            decode_sprite(f, sprite, container)
            res.append(PyComment(f'{nfo_line}: Sprite({l}, {grf_type_str}): Image: M {width}x{height} zoom={zoom} x_offs={xofs} y_offs={yofs} decomp_size={num}'))
            res.append(SpritePlaceholder(sprite.id))
            context.consume()
            return True, res, sprite
        else:
            data = f.read(l)
            context.count_action(data[0], l + size_bytes * 2 + 1)
            context.consume()
            res.append(PyComment(f'{nfo_line}: Sprite({l}, {grf_type_str}): {hex_str(data, 100)}'))
    return True, res, None


UINT16_MAX = 2**16 - 1

def decode_chunked(sprite, container, data):
    # Chunked aka tile compression
    yfmt = 'I' if container >= 2 and sprite.decomp_size > UINT16_MAX else 'H'
    offsets = struct.unpack_from('<' + yfmt * sprite.height, data)
    res = b''
    a = np.zeros((sprite.height, sprite.width, sprite.bpp), dtype=np.uint8)
    for y, ofs in enumerate(offsets):
        is_final = False
        while not is_final:
            if container >= 2 and sprite.width > 256:
                is_final = bool(data[ofs + 1] & 0x80)
                length = ((data[ofs + 1] & 0x7f) << 8) | data[ofs]
                skip = (data[ofs + 3] << 8) | data[ofs + 2]
                ofs += 4
            else:
                is_final = bool(data[ofs] & 0x80)
                length = data[ofs] & 0x7f
                skip = data[ofs + 1]
                ofs += 2
            byte_length = length * sprite.bpp
            b = np.frombuffer(data, offset=ofs, count=byte_length, dtype=np.uint8)
            b.shape = (length, sprite.bpp)
            a[y, skip:skip+length, :] = b
            ofs += byte_length
    return a


def decode_sprite(f, sprite, container):
    num = sprite.decomp_size
    data = b''
    while num > 0:
        code = f.read(1)[0]
        if code >= 128: code -= 256
        if code >= 0:
            size = 0x80 if code == 0 else code
            num -= size
            if num < 0: raise RuntimeError('Corrupt sprite')
            data += f.read(size)
        else:
            data_offset = ((code & 7) << 8) | f.read(1)[0]
            #if (dest - data_offset < dest_orig.get()) return WarnCorruptSprite(file, file_pos, __LINE__);
            size = -(code >> 3)
            num -= size
            if num < 0: raise RuntimeError('Corrupt sprite')
            if size == data_offset:
                data += data[-size:]
            else:
                data += data[-data_offset:size - data_offset]
    if num != 0: raise RuntimeError('Corrupt sprite')
    if sprite.type & 0x8:
        a = decode_chunked(sprite, container, data)
    else:
        assert len(data) == sprite.width * sprite.height * sprite.bpp, (len(data), sprite.width, sprite.height, sprite.bpp)
        a = np.frombuffer(data, dtype=np.uint8)
        a.shape = (sprite.height, sprite.width, sprite.bpp)

    # a = WIN_TO_DOS_REMAP.remap_array(a)

    return a, hash(data)


class RealSprite:
    def __init__(self, id, offset):
        self.id = id
        self.offset = offset


class RealGraphicsSprite(RealSprite):
    def __init__(self, id, offset, decomp_size, width, height, type, zoom, xofs, yofs):
        super().__init__(id, offset)
        self.decomp_size = decomp_size
        self.width = width
        self.height = height
        self.type = type
        self.zoom = zoom
        self.xofs = xofs
        self.yofs = yofs
        self._set_bpp()

    def _set_bpp(self):
        self.bpp = 0
        if self.type & 1:
            self.bpp += 3
        if self.type & 2:
            self.bpp += 1
        if self.type & 4:
            self.bpp += 1


class RealRemapSprite(RealSprite):
    def __init__(self, id, data):
        self.id = id
        self.data = data


class RealSoundSprite(RealSprite):
    def __init__(self, id, offset, size, filename):
        super().__init__(id, offset)
        self.size = size
        self.filename = filename


class SpritePlaceholder:
    def __init__(self, sprite_id):
        self.sprite_id = sprite_id


def read_real_sprite(f, nfo_line, context):
    sprite_id = struct.unpack('<I', f.read(4))[0]
    if sprite_id == 0:
        return None, [PyComment('End of pseudo sprites')]
    num, t = struct.unpack('<IB', f.read(5))
    start_pos = f.tell()
    sprite_type, sprite_group = context.get_sprite_data(sprite_id)
    context.count_real_sprite(sprite_id, 8 + num)
    sprite_type_str = {
        ParsingContext.GRAPHICS: 'image',
        ParsingContext.SOUND: 'sound',
    }.get(sprite_type, 'unknown')
    s = f'{nfo_line}: Real sprite({sprite_id}): ({num}, {t:02x}): '
    if sprite_type == ParsingContext.GRAPHICS:
        zoom, height, width, xofs, yofs = struct.unpack('<BHHhh', f.read(9))
        fmt = ''
        if t & 0x01: fmt = fmt + 'RGB'
        if t & 0x02: fmt = fmt + 'A'
        if t & 0x04: fmt = fmt + 'M'
        bpp = len(fmt)
        decomp_size = struct.unpack('<I', f.read(4))[0] if t & 0x08 else width * height * bpp
        s += f'Image {sprite_group}: {fmt} {width}x{height} zoom={zoom} x_offs={xofs} y_offs={yofs} decomp_size={decomp_size}'
        sprite = RealGraphicsSprite(sprite_id, f.tell(), decomp_size, width, height, t, zoom, xofs, yofs)
        f.seek(start_pos + num - 1, 0)
        return sprite, [PyComment(s)]

    if sprite_type == ParsingContext.SOUND:
        l = f.read(2)[1]
        filename = f.read(l + 1)[:-1].decode()
        sprite = RealSoundSprite(sprite_id, f.tell(), num - 3 - l, filename)
        f.seek(start_pos + num - 1, 0)
        return sprite, [PyComment(s + f'Sound: {sprite.filename}')]

    sprite = RealSprite(f.tell(), num - 1)
    f.seek(start_pos + num - 1, 0)
    return sprite, [PyComment(s + 'Unknown sprite type')]


class GraphicsSpriteGen(IntermediateSprite):
    def __init__(self, sprites):
        self.sprites = sprites

    # TODO
    # def get_imports():
    #     return ['ImageFile', 'ImageSprite']

    def py(self, context):
        res = ''

        def make_file_var(path):
            nonlocal res
            if path is None:
                return

            if path in context.resources:
                return context.resources[path]

            file_var = path[path.rfind('/') + 1:].replace('.', '_')
            res += f"{file_var} = ImageFile('{path}')\n"
            context.resources[path] = file_var

        for _, _, _, m_path, rgba_path in self.sprites:
            make_file_var(m_path)
            make_file_var(rgba_path)

        sprite_sl = []
        for sprite, x, y, m_path, rgba_path in self.sprites:
            if m_path is None and rgba_path is None:
                # TODO broken sprite?
                sprite_sl.append(f'EMPTY_SPRITE')
                continue
            main_file_var = context.resources[rgba_path or m_path]
            if rgba_path and m_path:
                mask_file_var = context.resources[m_path]
                mask_str = f', mask=({mask_file_var}, 0, 0)'
            else:
                mask_str = ''
            sprite_sl.append(f'FileSprite({main_file_var}, {x}, {y}, {sprite.width}, {sprite.height}, xofs={sprite.xofs}, yofs={sprite.yofs}{mask_str})')

        if len(sprite_sl) > 1:
            res += 'g.add((\n'
            for s in sprite_sl:
                res += f'    {s},\n'
            res += '))'
        else:
            res += f'g.add({sprite_sl[0]})'

        return res


def save_graphic_resources(container, f, resource_dir, resource_dir_rel, context, sprites, out_sprites):
    by_group = defaultdict(list)
    for sl in sprites.values():
        for i, s in enumerate(sl):
            if not isinstance(s, RealGraphicsSprite):
                continue
            group = None
            if s.id is not None:
                group = context.get_sprite_data(s.id)[1]
            group = group or 'misc'
            by_group[group].append((s, i))
    PADDING = 5

    sprite_count = defaultdict(int)
    sprite_size = {}
    sprite_data_hashes = set()
    by_id = defaultdict(list)
    for group, sprites in by_group.items():
        w = 0
        h = 0

        line_ofs = []
        N = 8
        for i in range(0, len(sprites), N):
            lw = sum(s.width for s, _ in sprites[i: i + N]) + PADDING * 2 * N
            lh = PADDING * 2 + max(s.height for s, _ in sprites[i: i + N])
            w = max(w, lw)
            line_ofs.append(h)
            h += lh
        formats = set(s.type for s, _ in sprites)

        m, rgba = None, None
        if any(s.type & 0x03 > 0 for s, _ in sprites):
            rgba = Image.new('RGBA', (w, h))

        if any(s.type & 0x04 > 0 for s, _ in sprites):
            m = Image.new('P', (w, h), color=0xff)
            m.putpalette(PALETTE)

        img_sprites = []
        for i, (s, si) in enumerate(sprites):
            yofs = PADDING + line_ofs[i // N]
            if i % N == 0:
                xofs = PADDING

            if s.type & 0x03 == 0x2:
                # TODO wtf is alpha without rgb
                continue

            f.seek(s.offset)
            duplicate_type = 0
            data, sprite_hash = decode_sprite(f, s, container)
            if sprite_hash in sprite_data_hashes:
                duplicate_type = 1
            else:
                sprite_data_hashes.add(sprite_hash)
            sprite_hash = hash((sprite_hash, s.xofs, s.yofs))
            s_size = f.tell() - s.offset + 18
            assert sprite_size.get(sprite_hash, s_size) == s_size
            sprite_count[sprite_hash] += 1
            sprite_size[sprite_hash] = s_size
            if sprite_count[sprite_hash] > 1:
                duplicate_type = 2

            a = np.frombuffer(data, dtype=np.uint8)
            assert a.size == s.width * s.height * s.bpp, (a.size, s.width, s.height, s.bpp)
            bpp = s.bpp
            a.shape = (s.height, s.width, bpp)

            if s.type & 0x04 > 0:
                bpp -= 1
                if duplicate_type > 0:
                    draw = ImageDraw.Draw(m)
                    draw.rectangle((xofs - 2, yofs - 2, xofs + s.width + 1, yofs + s.height + 1), outline=[0, 0xBE, 0xB8][duplicate_type])
                im = Image.fromarray(a[:, :, -1], mode='P')
                m.paste(im, (xofs, yofs))

            if bpp >= 3 and duplicate_type > 0:
                draw = ImageDraw.Draw(rgba)
                draw.rectangle((xofs - 2, yofs - 2, xofs + s.width + 1, yofs + s.height + 1), outline=[0, 0xff00ffff, 0xff0000ff][duplicate_type])
            if bpp == 3:
                im = Image.fromarray(a[:, :, :bpp], mode='RGB')
                rgba.paste(im, (xofs, yofs))
            elif bpp == 4:
                im = Image.fromarray(a[:, :, :bpp], mode='RGBA')
                rgba.paste(im, (xofs, yofs))

            img_sprites.append((s, xofs, yofs))

            xofs += s.width + 2 * PADDING

        res = []
        if m is not None:
            m_filename = f'{group}.png'
            m.save(resource_dir / m_filename, 'PNG')
            m_path = resource_dir_rel + '/' + m_filename
        else:
            m_path = None

        if rgba is not None:
            rgba_filename = f'{group}_32bpp.png'
            rgba.save(resource_dir / rgba_filename, 'PNG')
            rgba_path = resource_dir_rel + '/' + rgba_filename
        else:
            rgba_path = None

        for s, x, y in img_sprites:
            by_id[s.id].append((s, x, y, m_path, rgba_path))

    for sid, sl in by_id.items():
        out_sprites[sid] = GraphicsSpriteGen(sl)

    dup_sprites = 0
    dup_size = 0
    total_sprites = 0
    total_size = 0
    for k, v in sprite_count.items():
        total_sprites += v
        total_size += v * sprite_size[k]
        if v > 1:
            dup_sprites += v - 1
            dup_size += (v - 1) * sprite_size[k]

    dup_pc = (100 * dup_sprites / total_sprites) if total_sprites > 0 else 0.
    size_pc = (100 * dup_size / total_size) if total_sprites > 0 else 0.
    print(f'Found {dup_sprites}({dup_pc:.0f}%) duplicate sprites, {dup_size}({size_pc:.0f}%) bytes total')


def save_sound_resources(container, f, resource_dir, resource_dir_rel, context, sprites, out_sprites):
    filenames = set()
    for sl in sprites.values():
        for i, s in enumerate(sl):
            if not isinstance(s, RealSoundSprite):
                continue
            fname = gen_resource_name(filenames, s.filename, 'wav')
            f.seek(s.offset)
            wav_data = f.read(s.size)
            open(resource_dir / fname, 'wb').write(wav_data)
            out_sprites[s.id] = PyCode(f"g.add(RAWSound('{resource_dir_rel}/{fname}'))")


def decompile(in_grf_path):
    out_dir_name = in_grf_path.name
    if '.' in out_dir_name:
        out_dir_name = out_dir_name[:out_dir_name.rfind('.')]
    else:
        out_dir_name += '_out'
    out_dir = Path(out_dir_name)
    if out_dir.exists():
        print(f'ERROR: Output directory `{out_dir}` already exists')
        sys.exit(1)

    out_dir.mkdir()
    resource_dir_rel = 'resources'
    resource_dir = out_dir / resource_dir_rel
    resource_dir.mkdir()
    print(f'Started decompiling `{in_grf_path}` into directory `{out_dir}`.')

    context = ParsingContext()
    with open(in_grf_path, 'rb') as f:
        first = f.read(1)
        container = None
        gen = grf.BaseNewGRF()
        comment = lambda text: gen.add(PyComment(text))
        if first == b'\00':
            header_bytes = first + f.read(9)
            comment(f'New container header: {hex_str(header_bytes)}')
            data_offest, compression = struct.unpack('<IB', f.read(5))
            header_offset = f.tell() - 1
            comment(f'Offset: {data_offest} compresion: {compression}')
            magic_sprite_bytes = f.read(5 + 4)
            comment(f'Magic sprite: {hex_str(magic_sprite_bytes)}')
            container = 2
        else:
            f.seek(0, 0)
            comment(f'Old container, no header!')
            container = 1
            # magic_sprite_bytes = f.read(5 + 2)
            # comment(f'Magic sprite: {hex_str(magic_sprite_bytes)}')

        # print('Magic sprite:', hex_str(f.read(4)))
        nfo_line = 1
        real_sprites = defaultdict(list)
        while (res := read_pseudo_sprite(f, nfo_line, container, context))[0]:
            gen.add(*res[1])
            if (sprite := res[2]) is not None:
                real_sprites[sprite.id].append(sprite)
            nfo_line += 1

        if container == 2:
            real_data_offset = f.tell() - header_offset
            while (res := read_real_sprite(f, nfo_line, context))[0] is not None:
                sprite = res[0]
                real_sprites[sprite.id].append(sprite)
                gen.add(*res[1])
                nfo_line += 1

            if data_offest != real_data_offset:
                comment(f'[ERROR] Data offset check failed: {data_offest} {real_data_offset}')

        res_sprites = defaultdict(list)
        save_graphic_resources(container, f, resource_dir, resource_dir_rel, context, real_sprites, res_sprites)
        save_sound_resources(container, f, resource_dir, resource_dir_rel, context, real_sprites, res_sprites)

        # Save recolour sprites
        for sl in real_sprites.values():
            for i, s in enumerate(sl):
                if not isinstance(s, RealRemapSprite):
                    continue
                res_sprites[s.id] = PyCode(f"g.add(PaletteRemap.from_buffer({s.data!r}))")

        for i, g in enumerate(gen.generators):
            if not isinstance(g, SpritePlaceholder):
                continue
            new_gen = res_sprites.get(g.sprite_id)
            if new_gen is None:
                gen.generators[i] = PyComment(f'Failed to resolve sprite {g.sprite_id}')
            else:
                gen.generators[i] = new_gen

        with open(out_dir / 'generate.py', 'w') as f:
            f.write(f'# This file is generated by decompiling {in_grf_path.name} with grftopy.\n')
            f.write('# It\'s is only intended to be used as reference, don\'t expect it to actually run and produce grf.\n')
            f.write('\n')
            f.write(gen.generate_python('grfpy_' + in_grf_path.name))

    print(f'Finished decompiling `{in_grf_path}` into directory `{out_dir}`.')
    context.print_stats()


def main():
    in_grf_path = Path(sys.argv[1])
    decompile(in_grf_path)


if __name__ == "__main__":
    main()
