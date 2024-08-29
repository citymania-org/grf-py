import datetime
import textwrap
import struct
import pprint
from abc import abstractmethod
from typing import NamedTuple, Union, Optional
from collections.abc import Iterable, Sequence

from typeguard import typechecked


from .common import Feature, hex_str, utoi32, VEHICLE_FEATURES, date_to_days, ANY_LANGUAGE, \
                    DataReader, to_bytes, read_dword, read_word, days_to_date, read_extended_byte, \
                    encode_extended_byte, Date, check_sequence_range
from .common import TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, CANAL, BRIDGE, HOUSE, GLOBAL_VAR, \
                    INDUSTRY_TILE, INDUSTRY, CARGO, SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, \
                    AIRPORT_TILE, ROADTYPE, TRAMTYPE, NO_CLIMATE, ALL_CLIMATES, TEMPERATE, ARCTIC, TROPICAL, TOYLAND, TOWN

from .parser import Node, Expr, Value, Var, Temp, Perm, Call, parse_code, OP_INIT, SPRITE_FLAGS
from .sprites import Action, Sound, FakeAction
from .exceptions import FormatError


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

    def __eq__(self, ref):
        return (
            isinstance(ref, Ref) and
            self.value == ref.value and
            self.is_callback == ref.is_callback and
            self.ref_id == ref.ref_id
        )


def get_ref_id(ref_obj):
    if isinstance(ref_obj, Ref):
        return ref_obj.value
    if isinstance(ref_obj, int):
        return ref_obj | 0x8000
    if isinstance(ref_obj, Sound):
        return ref_obj.id | 0x8000
    return ref_obj.ref_id


def ref_eq(a, b):
    return get_ref_id(a) == get_ref_id(b)


def ref_dict_eq(a, b):
    if len(a) != len(b):
        return False
    for k, v in a.items():
        if k not in b:
            return False
        if get_ref_id(v) != get_ref_id(b[k]):
            return False
    return True


class CB(Ref):
    def __init__(self, value):
        super().__init__(value | 0x8000)


class Range:
    def __init__(self, low, high, ref):
        self.ref = ref
        self.low = low
        self.high = high

    def __str__(self):
        if self.low == self.high:
            return f'{self.low} -> {self.ref}'
        return f'{self.low}..{self.high} -> {self.ref}'

    def __repr__(self):
        return f'Range({self.low}, {self.high}, {self.ref})'

    def __eq__(self, r):
        return (
            isinstance(r, Range) and
            get_ref_id(self.ref) == get_ref_id(r.ref) and
            self.low == r.low and
            self.high == r.high
        )


def _py_dict(data, key_func, value_func, indent=12):
    indent_str = ' ' * indent
    res = ''
    it = data.items() if isinstance(data, dict) else data
    for k, v in it:
        kstr = key_func(k)
        vstr = value_func(k, v)
        res += f'\n{indent_str}    {kstr}: {vstr},'
    if not res:
        return '{}'
    if '\n' not in res[1:]:
        return f'{{{res[:-1].lstrip()}}}'
    return f'{{{res}\n{indent_str}}}'


def _py_ref_dict(ref_dict, value_func):
    def key_func(k):
        if not isinstance(k, Range):
            return repr(k)
        if k.low == k.high:
            return utoi32(k.low)
        return repr((utoi32(k.low), utoi32(k.high)))
    return _py_dict(ref_dict, key_func, value_func)


class ReferenceableAction:
    def __init__(self):
        self.ref_var = None

    def _py_ref_id(self, context):
        if self.ref_id is None:
            return ''
        return f'\n            ref_id={self.ref_id},'

    def py_ref(self):
        if self.ref_var is not None:
            return self.ref_var
        return f'Ref({self.ref_id})'


class ReferencingAction:
    def get_refs(self):
        raise NotImplementedError

    def set_refs(self, refs):
        raise NotImplementedError

    def _format_ref(self, context, ref):
        if isinstance(ref, ReferenceableAction):
            return ref.py_ref()
        return repr(ref)


class SpriteRef:
    def __init__(self, id, pal=0, is_global=True, use_recolour=False, always_transparent=False, no_transparent=False):
        assert id <= 0x3fff, id
        assert pal <= 0x3fff, pal
        self.id = id
        self.pal = pal
        self.is_global = is_global
        self.use_recolour = use_recolour
        self.always_transparent = always_transparent
        self.no_transparent = no_transparent

    @classmethod
    def from_grf(cls, data, *, global_if_flagged):
        pal = data >> 16
        sprite = data & 0xffff

        return cls(
            id=sprite & 0x3fff,
            pal=pal & 0x3fff,
            is_global=(bool(pal & 0x8000) == global_if_flagged),
            use_recolour=bool(sprite & 0x8000),
            always_transparent=bool(sprite & 0x4000),
            no_transparent=bool(pal & 0x4000),
        )

    def to_grf(self, *, global_if_flagged):
        return (
            self.id | (self.always_transparent << 14) | (self.use_recolour << 15) |
            (self.pal | (self.no_transparent << 14) | ((self.is_global == global_if_flagged) << 15)) << 16
        )

    def __repr__(self):
        return self.py(None)

    def py(self, context):
        return (
            f'SpriteRef(id={self.id}, pal={self.pal}, is_global={self.is_global}, '
            f'use_recolour={self.use_recolour}, always_transparent={self.always_transparent}, '
            f'no_transparent={self.no_transparent})'
        )


class Property:
    def validate(cls, value):
        raise NotImplementedError

    def read(cls, data, ofs):
        raise NotImplementedError

    def encode(cls, value):
        raise NotImplementedError

    def format(cls, value, indent):
        return repr(value)


class DateProperty(Property):
    def validate(cls, value):
        if not isinstance(value, (datetime.date, int, Date)):
            raise ValueError(f'grf.date, datetime.date or int object expected')

    def read(cls, data, ofs):
        value, ofs = read_dword(data, ofs)
        return days_to_date(value), ofs

    def encode(cls, value):
        if isinstance(value, datetime.date):
            value = date_to_days(value)
        elif isinstance(value, Date):
            value = value.to_days()
        return struct.pack('<I', value)


class IDProperty(Property):
    def __init__(self, *, extended=False):
        self.extended = extended

    def _resolve(self, value):
        if isinstance(value, int):
            return value
        if hasattr(value, 'id'):
            return value.id
        return None

    def validate(self, value):
        value = self._resolve(value)
        if not isinstance(value, int):
            raise ValueError(f'int or object with id expected')

    def read(self, data, ofs):
        if self.extended:
            value, ofs = read_extended_byte(data, ofs)
        else:
            value, ofs = read_word(data, ofs)
        return value, ofs

    def encode(self, value):
        value = self._resolve(value)
        if self.extended:
            return encode_extended_byte(value)
        else:
            return struct.pack('<H', value)


class ClimateProperty(Property):
    def validate(cls, value):
        if not (0 <= value <= ALL_CLIMATES):
            raise ValueError(f'Expected value between NO_CLIMATE(0) and ALL_CLIMATES({ALL_CLIMATES})')

    def read(cls, data, ofs):
        return data[ofs], ofs + 1

    def encode(cls, value):
        return struct.pack('<B', value)

    def format(cls, value, indent):
        if value == NO_CLIMATE: return 'grf.NO_CLIMATE'
        if value == ALL_CLIMATES: return 'grf.ALL_CLIMATES'

        # Preserve incorrect decompiled values
        if not (0 < value < ALL_CLIMATES):
            return repr(value)

        climates = {
            TEMPERATE: 'grf.TEMPERATE',
            ARCTIC: 'grf.ARCTIC',
            TROPICAL: 'grf.TROPICAL',
            TOYLAND:'grf.TOYLAND',
        }
        res = []
        for k, v in climates.items():
            if k & value > 0:
                res.append(v)
        return ' | '.join(res)


class ByteListProperty(Property):
    def validate(cls, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'list or tuple object expected')
        if not all(isinstance(x, int) and 0 <= x < 256 for x in value):
            raise ValueError(f'expected integer values in range 0..255')

    def read(cls, data, ofs):
        n = data[ofs]
        res = tuple(map(int, data[ofs + 1: ofs + 1 + n]))
        return res, ofs + 1 + n

    def encode(cls, value):
        return struct.pack('<B', len(value)) + bytes(value)


class CargoListProperty(ByteListProperty):
    pass


class PyComment(FakeAction):
    def __init__(self, text):
        self.text = text

    def py(self, context):
        return f'# {self.text}'


class PyCode(FakeAction):
    def __init__(self, code):
        self.code = code

    def py(self, context):
        return f'{self.code}'


# Action 0

ACTION0_COMMON_VEHICLE_PROPS = {
    0x00: ('introduction_days_since_1920', 'W', 'deprecated by introduction_date'),  # date of introduction as the amount of days since 1920
    0x02: ('reliability_decay', 'B'),  # reliability decay speed     no
    0x03: ('vehicle_life', 'B'),  # vehicle life in years   no
    0x04: ('model_life', 'B'),  # model life in years     no
    0x06: ('climates_available', ClimateProperty()),  # climate availability    should be zero
    0x07: ('loading_speed', 'B'),  # loading speed   yes
}


ACTION0_TRAIN_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x05: ('track_type', 'B'),  # Track type (see below)  should be same as front
    0x08: ('ai_special_flag', 'B'),  # AI special flag: set to 1 if engine is 'optimized' for passenger service (AI won't use it for other cargo), 0 otherwise     no
    0x09: ('max_speed', 'W'),  # Speed in mph*1.6 (see below)    no
    0x0B: ('power', 'W'),  # Power (0 for wagons)    should be zero
    0x0D: ('running_cost_factor', 'B'),  # Running cost factor (0 for wagons)  should be zero
    0x0E: ('running_cost_base', 'D'),  # Running cost base, see below    should be zero
    0x12: ('sprite_id', 'B'),  # Sprite ID (FD for new graphics)     yes
    0x13: ('dual_headed', 'B'),  # Dual-headed flag; 1 if dual-headed engine, 0 otherwise  should be zero also for front
    0x14: ('cargo_capacity', 'B'),  # Cargo capacity  yes
    0x15: ('default_cargo_type', 'B'),  # Cargo type, see CargoTypes
    0x16: ('weight_low', 'B'),  # Weight in tons  should be zero
    0x17: ('cost_factor', 'B'),  # Cost factor     should be zero
    0x18: ('ai_engine_rank', 'B'),  # Engine rank for the AI (AI selects the highest-rank engine of those it can buy)     no
    0x19: ('engine_class', 'B'),  # Engine traction type (see below)    no
    0x1A: ('sort_purchase_list', IDProperty(extended=True)), # Not a property, but an action: sort the purchase list.  no
    0x1B: ('extra_power_per_wagon', 'W'),  # Power added by each wagon connected to this engine, see below   should be zero
    0x1C: ('refit_cost', 'B'),  # Refit cost, using 50% of the purchase price cost base   yes
    0x1D: ('refittable_cargo_types', 'D'),  # Bit mask of cargo types available for refitting, see column 2 (bit value) in CargoTypes     yes
    0x1E: ('cb_flags', 'B'),  # Callback flags bit mask, see below  yes
    0x1F: ('tractive_effort_coefficient', 'B'),  # Coefficient of tractive effort  should be zero
    0x20: ('air_drag_coefficient', 'B'),  # Coefficient of air drag     should be zero
    0x21: ('shorten_by', 'B'),  # Make vehicle shorter by this amount, see below  yes
    0x22: ('visual_effect_and_powered', 'B'),  # Set visual effect type (steam/smoke/sparks) as well as position, see below  yes
    0x23: ('extra_weight_per_wagon', 'B'),  # Set how much weight is added by making wagons powered (i.e. weight of engine), see below    should be zero
    0x24: ('weight_high', 'B'),  # High byte of vehicle weight, weight will be prop.24*256+prop.16     should be zero
    0x25: ('bitmask_vehicle_info', 'B'),  # User-defined bit mask to set when checking veh. var. 42     yes
    0x26: ('retire_early', 'b'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)    no
    0x27: ('misc_flags', 'B'),  # Miscellaneous flags     partly
    0x28: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes    yes
    0x29: ('non_refittable_cargo_classes', 'W'),  # Non-refittable cargo classes    yes
    0x2A: ('introduction_date', DateProperty()),  # Long format introduction date   no
    0x2B: ('cargo_age_period', 'W'),  # period  yes
    0x2C: ('cargo_allow_refit', CargoListProperty()), # refittable cargo types   yes
    0x2D: ('cargo_disallow_refit', CargoListProperty()),  # refittable cargo types    yes
    0x2E: ('curve_speed_mod', 'W'),  # speed modifier    yes
    0x2F: ('variant_group', IDProperty()),  # Vehicle variant group
    0x30: ('extra_flags', 'D'),  # extra flags
    0x31: ('extra_cb_flags', 'B'),  # extra callback flags
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
    0x16: ('refittable_cargo_types', 'D'),  # Bit mask of cargo types available for refitting (not refittable if 0 or unset), see column 2 (bit values) in CargoTypes     yes
    0x17: ('cb_flags', 'B'),  # Callback flags bit mask, see below  yes
    0x18: ('tractive_effort_coefficient', 'B'),  # Coefficient of tractive effort  should be zero
    0x19: ('air_drag_coefficient', 'B'),  # Coefficient of air drag     should be zero
    0x1A: ('refit_cost', 'B'),  # Refit cost, using 25% of the purchase price cost base   yes
    0x1B: ('retire_early', 'b'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)    no
    0x1C: ('misc_flags', 'B'),  # Miscellaneous vehicle flags     partly ("tram" should be same as front)
    0x1D: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes, see train prop. 28    yes
    0x1E: ('non_refittable_cargo_classes', 'W'),  # Non-refittable cargo classes, see train prop. 29    yes
    0x1F: ('introduction_date', DateProperty()),  # Long format introduction date   no
    0x20: ('sort_purchase_list', IDProperty(extended=True)), # Sort the purchase list  no
    0x21: ('visual_effect', 'B'),  # Visual effect   yes
    0x22: ('cargo_age_period', 'W'),  # Custom cargo ageing period  yes
    0x23: ('shorten_by', 'B'),  # Make vehicle shorter, see train property 21     yes
    0x24: ('cargo_allow_refit', CargoListProperty()),  # List of always refittable cargo types, see train property 2C    yes
    0x25: ('cargo_disallow_refit', CargoListProperty()),  # List of never refittable cargo types, see train property 2D     yes
    0x26: ('variant_group', IDProperty()),  # Vehicle variant group
    0x27: ('extra_flags', 'D'),  # extra flags
    0x28: ('extra_cb_flags', 'B'),  # extra callback flags
}

ACTION0_SHIP_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x08: ('sprite_id', 'B'),  # Sprite (FF for new graphics)
    0x09: ('is_refittable', 'B'),  # Refittable (0 no, 1 yes)
    0x0A: ('cost_factor', 'B'),  # Cost factor
    0x0B: ('max_speed', 'B'),  # Speed in mph*3.2
    0x0C: ('default_cargo_type', 'B'),  # Cargo type, see CargoTypes
    0x0D: ('cargo_capacity', 'W'),  # Capacity
    0x0F: ('running_cost_factor', 'B'),  # Running cost factor
    0x10: ('sound_effect', 'B'),  # Sound effect type (4=cargo ship, 5=passenger ship)
    0x11: ('refittable_cargo_types', 'D'),  # v≥1     Bit mask of cargo types available for refitting, see column 2 (bit values) in CargoTypes
    0x12: ('cb_flags', 'B'),  # Callback flags bit mask, see below
    0x13: ('refit_cost', 'B'),  #  Refit cost, using 1/32 of the default refit cost base
    0x14: ('ocean_speed_fraction', 'B'),  # Ocean speed fraction, sets fraction of top speed available in the ocean; e.g. 00=100%, 80=50%, FF=0.4%
    0x15: ('canal_speed_fraction', 'B'),  # Canal speed fraction, same as above but for canals and rivers
    0x16: ('retire_early', 'b'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)
    0x17: ('misc_flags', 'B'),  # Miscellaneous vehicle flags
    0x18: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes, see train prop. 28
    0x19: ('non_refittable_cargo_classes', 'W'),  # Non-refittable cargo classes, see train prop. 29
    0x1A: ('introduction_date', DateProperty()),  # Long format introduction date
    0x1B: ('sort_purchase_list', IDProperty(extended=True)), # Sort the purchase list
    0x1C: ('visual_effect', 'B'),  # Visual effect
    0x1D: ('cargo_age_period', 'W'),  # Custom cargo ageing period
    0x1E: ('cargo_allow_refit', CargoListProperty()),  # List of always refittable cargo types, see train property 2C
    0x1F: ('cargo_disallow_refit', CargoListProperty()),  # List of never refittable cargo types, see train property 2D
    0x20: ('variant_group', IDProperty()),  # Vehicle variant group
    0x21: ('extra_flags', 'D'),  # extra flags
    0x22: ('extra_cb_flags', 'B'),  # extra callback flags
}

ACTION0_AIRCRAFT_PROPS = {
    **ACTION0_COMMON_VEHICLE_PROPS,
    0x08: ('sprite_id', 'B'),  # Sprite (FF for new graphics)
    0x09: ('is_helicopter', 'B'),  # Is helicopter? 2=no, 0=yes
    0x0A: ('is_large', 'B'),  # Is large? 0=no, 1=yes (i.e. can't safely land on small airports)
    0x0B: ('cost_factor', 'B'),  # Cost factor
    0x0C: ('max_speed', 'B'),  # Speed in units of 8 mph, that is: property = (speed in mph) / 8
    0x0D: ('acceleration', 'B'),  # Acceleration in units of 3/8 mph/tick, that is: property = (acceleration in mph/tick) * 8/3 [1]
    0x0E: ('running_cost_factor', 'B'),  # Running cost factor
    0x0F: ('passenger_capacity', 'W'),  # Primary cargo capacity (passenger or refitted cargo)
    0x11: ('mail_capacity', 'B'),  # Secondary cargo capacity (mail)
    0x12: ('sound_effect', 'B'),  # Sound effect
    0x13: ('refittable_cargo_types', 'D'),  # Bit mask of cargo types available for refitting, see column 2 (Bit Value) in CargoTypes
    0x14: ('cb_flags', 'B'),  # Callback flags bit mask, see below
    0x15: ('refit_cost', 'B'),  # Refit cost, using 1/32 of the default refit cost base
    0x16: ('retire_early', 'b'),  # Retire vehicle early, this many years before the end of phase 2 (see Action0General)
    0x17: ('misc_flags', 'B'),  # Miscellaneous vehicle flags
    0x18: ('refittable_cargo_classes', 'W'),  # Refittable cargo classes, see train prop. 28
    0x19: ('non_refittable_cargo_classes', 'W'),  # Non-refittable cargo classes, see train prop. 29
    0x1A: ('introduction_date', DateProperty()),  # Long format introduction date
    0x1B: ('sort_purchase_list', IDProperty(extended=True)),  # Sort the purchase list
    0x1C: ('cargo_age_period', 'W'),  # Custom cargo ageing period
    0x1D: ('cargo_allow_refit', CargoListProperty()),  # List of always refittable cargo types, see train property 2C
    0x1E: ('cargo_disallow_refit', CargoListProperty()),  # List of never refittable cargo types, see train property 2D
    0x1F: ('range', 'W'),  # Aircraft range in tiles. Distance is euclidean, a value of 0 means range is unlimited
    0x20: ('variant_group', IDProperty()),  # Vehicle variant group
    0x21: ('extra_flags', 'D'),  # extra flags
    0x22: ('extra_cb_flags', 'B'),  # extra callback flags
}


class BaseSpriteData:
    def __init__(self, sprite, *, flags=None, registers=None):
        assert isinstance(sprite, SpriteRef)
        self.sprite = sprite
        self.flags = flags
        self.registers = registers


class TopLevelSprite(BaseSpriteData):
    def __init__(self, sprite, *, children=None, **kw):
        assert isinstance(sprite, SpriteRef)
        super().__init__(sprite, **kw)
        self.children = children or []

    def add_child(self, child):
        self.children.append(child)


class GroundSprite(TopLevelSprite):
    pass


class ParentSprite(TopLevelSprite):
    def __init__(self, *, sprite, extent, offset, **kw):
        assert isinstance(offset, tuple) and len(offset) == 3
        assert isinstance(extent, tuple) and len(extent) == 3

        super().__init__(sprite, **kw)
        self.offset = offset
        self.extent = extent


class ChildSprite(BaseSpriteData):
    def __init__(self, sprite, xofs, yofs, **kw):
        assert isinstance(sprite, SpriteRef)
        assert isinstance(xofs, int) and isinstance(yofs, int)
        super().__init__(sprite, **kw)
        self.xofs = xofs
        self.yofs = yofs


class SpriteLayout:
    def __init__(self, sprites):
        self.sprites = sprites


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


def read_sprite_layout(d, num, basic_format, global_if_flagged=True):
    has_z_position = not basic_format
    has_flags = bool(num & 0x40)
    num &= 0x3f

    def read_sprite():
        sprite = d.get_dword()
        flags = d.get_word() if has_flags else 0
        return SpriteRef.from_grf(sprite, global_if_flagged=global_if_flagged), flags

    sprites = []

    #read ground sprite
    ground_sprite, ground_flags = read_sprite()
    ground_registers = read_sprite_layout_registers(d, ground_flags, False)
    cur_parent = GroundSprite(ground_sprite, flags=ground_flags, registers=ground_registers)
    sprites.append(cur_parent)

    for _ in range(num):
        sprite, flags = read_sprite()
        offset = (d.get_byte(), d.get_byte(), d.get_byte() if has_z_position else 0)
        if offset[2] == 0x80:
            # read child sprite
            registers = read_sprite_layout_registers(d, flags, False)
            cur_parent.add_child(ChildSprite(sprite, offset[0], offset[1],
                                             flags=flags, registers=registers))
        else:
            # read parent sprite
            extent = (d.get_byte(), d.get_byte(), d.get_byte())
            registers = read_sprite_layout_registers(d, flags, True)
            cur_parent = ParentSprite(sprite=sprite, extent=extent, offset=offset,
                                      flags=flags, registers=registers)
            sprites.append(cur_parent)

    return SpriteLayout(sprites)


def encode_sprite_layout_registers(registers, is_parent):
    flags = 0
    res = b''
    for register, value in (registers or {}).items():
        assert register in SPRITE_FLAGS, register
        fparent, mask, size, conversion = SPRITE_FLAGS[register]

        if fparent is not None and fparent != is_parent:
            raise ValueError(f'Unexpected sprite register `{register}` with is_parent={is_parent}')

        if conversion:
            assert conversion is Temp, conversion
            value = value.get_index()

        if size == 0:
            if not value:
                continue
        elif size == 1:
            res += bytes((value,))
        elif size == 2:
            res += bytes((value[0], value[1]))
        else:
            raise RuntimeError(f'Incorrect size of sprite register {flag}: {size}')

        flags |= mask

    return res, flags


def encode_sprite(sprite, has_z_position):
    flags = 0
    is_parent = isinstance(sprite, TopLevelSprite)
    regs_bytes, flags = encode_sprite_layout_registers(sprite.registers, is_parent)
    res = struct.pack(
        '<IH',
        sprite.sprite.to_grf(global_if_flagged=False),
        flags,
    )
    if not isinstance(sprite, GroundSprite):
        if isinstance(sprite, ParentSprite):
            assert len(sprite.offset) == 3 if has_z_position else 2
            res += bytes(sprite.offset)
            assert len(sprite.extent) == 3
            res += bytes(sprite.extent)
        elif isinstance(sprite, ChildSprite):
            # TODO basic format (no z position) has no child sprites?
            res += bytes((sprite.xofs, sprite.yofs, 0x80))

    return res + regs_bytes


def encode_sprite_layout(layout, has_flags, basic_format):
    # TODO basic_format has no flags
    has_z_position = not basic_format
    num = len(layout.sprites) - 1
    assert num < 0x40, num
    if has_flags:
        num |= 0x40

    res = bytes((num,))

    for sprite in layout.sprites:
        res += encode_sprite(sprite, has_z_position)

    return res


class SpriteLayoutList:
    def __init__(self, layouts):
        self.layouts = layouts


class OldStationLayoutProperty(Property):
    def read(self, data, ofs):
        d = DataReader(data, ofs)
        num = d.get_extended_byte()
        res = []
        for i in range(num):
            ground_data = d.get_dword()
            if ground_data == 0:
                res.append(None)
                continue

            sprites = []
            cur_parent = GroundSprite(SpriteRef.from_grf(ground_data, global_if_flagged=False))
            sprites.append(cur_parent)
            while True:
                sprite = {}
                xofs = d.get_byte()
                if xofs == 0x80:
                    break
                # TODO somewhat replicates action2 layout
                offset = (xofs, d.get_byte(), d.get_byte())
                extent = (d.get_byte(), d.get_byte(), d.get_byte())
                sprite = SpriteRef.from_grf(d.get_dword(), global_if_flagged=False)
                if offset[2] == 0x80:
                    cur_parent.add_child(ChildSprite(sprite, offset[0], offset[1]))
                else:
                    cur_parent = ParentSprite(sprite=sprite, extent=extent, offset=offset)
                    sprites.append(cur_parent)
            res.append(SpriteLayout(sprites))

        return SpriteLayoutList(res), d.offset

    def format(self, value, indent):
        return pformat(value, indent=indent, indent_first=0)


class NewStationLayoutProperty(Property):
    def read(self, data, ofs):
        d = DataReader(data, ofs)
        num = d.get_extended_byte()
        res = []
        for i in range(num):
            snum = d.get_byte()
            # var10 flags allowed
            res.append(read_sprite_layout(d, snum, False, global_if_flagged=False))

        return SpriteLayoutList(res), d.offset

    def encode(self, value):
        num = len(value.layouts)
        assert num <= 0xffff, num
        res = struct.pack('<BH', 255, num)
        res += b''.join(encode_sprite_layout(l, True, False) for l in value.layouts)
        return res

    def format(self, value, indent):
        return pformat(value, indent=indent, indent_first=0)


ACTION0_STATION_PROPS = {
    0x08: ('class_label', 'L'),  # Class ID, see below
    0x09: ('layout', OldStationLayoutProperty()),  # Sprite layout, see below
    0x0A: ('copy_layout', 'B'),  # Copy sprite layout
    0x0B: ('cb_flags', 'B'),  # Callback flags
    0x0C: ('disabled_platforms', 'B'),  # Bit mask of disabled numbers of platforms
    0x0D: ('disabled_length', 'B'),  # Bit mask of disabled platform lengths
    0x0E: ('custom_layout', 'V'),  # Define custom layout, see below
    0x0F: ('copy_custom_layout', 'B'),  # Copy custom layout from stationid given by argument
    0x10: ('cargo_threshold', 'W'),  # Little/lots threshold
    0x11: ('draw_pylon_tiles', 'B'),  # Pylon placement
    0x12: ('cargo_random_triggers', 'D'),  # Bit mask of cargo type triggers for random sprites
    0x13: ('general_flags', 'B'),  # General flags
    0x14: ('hide_wire_tiles', 'B'),  # Overhead wire placement
    0x15: ('non_traversable_tiles', 'B'),  # Can train enter tile
    0x16: ('animation_info', 'W'),  # Animation information
    0x17: ('animation_speed', 'B'),  # Animation speed
    0x18: ('animation_triggers', 'W'),  # Animation triggers
    0x19: ('road_routing', 'V'),  # Road routing (reserved for future use)
    0x1A: ('advanced_layout', NewStationLayoutProperty()),  # Advanced sprite layout with register modifiers
    0x1B: ('min_bridge_height', '8*B'),  # Advanced sprite layout with register modifiers
    0x1C: ('station_name', 'W'),  # Station name
    0x1D: ('station_class_name', 'W'),  # Station class name
}

# TODO river

class BridgeLayoutProperty(Property):
    def validate(cls, value):
        if not isinstance(value, dict):
            raise ValueError('Expected dict')
        if "table_id" not in value:
            raise ValueError('Missing key table_id')
        if "tables" not in value:
            raise ValueError('Missing key tables')
        if value["table_id"] + len(value["tables"]) > 7:
            raise ValueError('Table ID exceeds 7')
        if any(len(table) != 32 for table in value["tables"]):
            raise ValueError('Each table should have exactly 32 SpriteRefs')
        if not all(isinstance(element, SpriteRef) for table in value["tables"] for element in table):
            raise ValueError('Each table should consists of SpriteRefs')

    def read(cls, data, ofs):
        d = DataReader(data, ofs)
        table_id = d.get_byte()
        num_tables = d.get_byte()
        tables = []
        for _ in range(num_tables):
            # if tableid >= 7 invalid data
            tables.append([SpriteRef.from_grf(d.get_dword(), global_if_flagged=False) for _ in range(32)])
        return {
            'table_id': table_id,
            'tables': tables,
        }, d.offset

    def encode(cls, value):
        res = bytes((value["table_id"], len(value["tables"])))
        for table in value["tables"]:
            res += struct.pack('<' + 'I' * 32, *[sprite_ref.to_grf(global_if_flagged=False) for sprite_ref in table])
        return res

    def format(self, value, indent):
        return pformat(value, indent=indent, indent_first=0)


ACTION0_BRIDGE_PROPS = {
    0x00:  ('fallback_type', 'B'),  # Failback Type, a default TTD Bridge ID
    0x08:  ('intro_year_since_1920', 'B'),  # Year of availability, counted from 1920 (set to 1920 for first bridge if newstartyear<1930)
    0x09:  ('min_length', 'B'),  # Minimum length, not counting ramps
    0x0A:  ('max_length', 'B'),  # Maximum length, not counting ramps
    0x0B:  ('cost_factor_byte', 'B'),  # Cost factor
    0x0C:  ('max_speed', 'W'),  # Max. speed
    0x0D:  ('layout', BridgeLayoutProperty()),  # Sprite layout, see below
    0x0E:  ('flags', 'B'),  # Various flags, bitcoded
    0x0F:  ('intro_year', 'D'),  # Long format year of availability, counted from year 0
    0x10:  ('purchase_text', 'W'),  # Purchase text
    0x11:  ('description_rail', 'W'),  # Description of a rail bridge
    0x12:  ('description_road', 'W'),  # Description of a road bridge
    0x13:  ('cost_factor', 'W'),  # Cost factor word access
}


class TileAcceptanceProperty(Property):
    def validate(cls, value):
        # TODO use typing
        if not isinstance(value, (tuple, list)):
            raise ValueError(f'Expected tuple or list')
        if not all(isinstance(x, (tuple, list)) and len(x) == 2 for x in value):
            raise ValueError(f'Elements should be tuple or list of length 2')
        if not all(0<= a <=8 for _, a in value):
            raise ValueError(f'Acceptance should be in range 0..255')
        if not all(0<= a <=8 for c, _ in value):
            raise ValueError(f'Cargo ID should be in range 0..255')

    def read(cls, data, ofs):
        num = data[ofs]
        res = [(data[ofs + 2 * i + 1], data[ofs + 2 * i + 2]) for i in range(num)]
        return res, ofs + 2 * num + 1

    def encode(cls, value):
        res = bytes((len(value),))
        for cargo, acceptance in value:
            res += bytes((cargo, acceptance))
        return res


ACTION0_HOUSE_PROPS = {
    0x08: ('substitute', 'B'),  # Substitute building type
    0x09: ('flags', 'B'),  # Building flags
    0x0A: ('availability_years', 'W'),  # Availability years
    0x0B: ('population', 'B'),  # Population
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
    0x1F: ('min_life', 'B'),  # Minimum life span in years
    0x20: ('watched_cargo_types', 'W'),  # Cargo acceptance watch list
    0x21: ('min_year', 'W'),  # Long year (zero based) of minimum appearance
    0x22: ('max_year', 'W'),  # Long year (zero based) of maximum appearance
    0x23: ('tile_acceptance', TileAcceptanceProperty()), # Tile acceptance list
}


class MultiDictProperty(Property):
    def validate(cls, value):
        if isinstance(value, int) and 0 <= value < 256:
            return
        if (isinstance(value, (tuple, list)) and len(value) == 2
                and 0 <= value[0] < 16 and 0 <= value[1] < 16):
            return
        raise ValueError(f'Expected integer 0-255 or a tuple (x, y) both in range 0-15')

    def read(cls, data, ofs):
        res = {}
        while data[ofs] != 0:
            name_end = data.find(b'\0', ofs + 1)
            k, v = data[ofs], data[ofs + 1: name_end]
            if k in res:
                if isinstance(res[k], list):
                    res[k].append(v)
                else:
                    res[k] = [res[k], v]
            else:
                res[k] = v
            ofs = name_end + 1
        return res, ofs + 1

    def encode(cls, value):
        res = b''
        for k, vl in value.items():
            if isinstance(vl, (bytes, str)):
                vl = (vl,)
            for v in vl:
                res += bytes((k,))
                res += to_bytes(v)
                res += b'\0'
        res += b'\0'
        return res


ACTION0_GLOBAL_PROPS = {
    0x08: ('basecost', 'B'),  # Cost base multipliers
    0x09: ('cargo_table', 'L'),  # Cargo translation table
    0x0A: ('currency_name', 'W'),  # Currency display names
    0x0B: ('currency_mult', 'D'),  # Currency multipliers
    0x0C: ('currency_options', 'W'),  # Currency options
    0x0D: ('currency_symbols', '0E'),  # Currency symbols
    0x0F: ('currency_euro_date', 'W'),  # Euro introduction dates
    0x10: ('snowline_table', '12*32*B'),  # Snow line height table
    0x11: ('grfid_overrides', '2*L'),  # GRFID overrides for engines
    0x12: ('railtype_table', 'L'),  # Railtype translation table
    0x13: ('lang_genders', MultiDictProperty()),  # Gender/case translation table
    0x14: ('lang_cases', MultiDictProperty()),  # Gender/case translation table
    0x15: ('lang_plural', 'B'),  # Plural form
    0x16: ('roadtype_table1', 'B'),  # Road-/tramtype translation table
    0x17: ('roadtype_table2', 'D'),  # Road-/tramtype translation table
}

ACTION0_INDUSTRY_TILE_PROPS = {
    0x08: ('substitute_type', 'B'),  # Substitute building type
    0x09: ('override_type', 'B'),  # Industry tile override
    0x0A: ('tile_acceptance_1', 'W'),  # Tile acceptance
    0x0B: ('tile_acceptance_2', 'W'),  # Tile acceptance
    0x0C: ('tile_acceptance_3', 'W'),  # Tile acceptance
    0x0D: ('land_shape_flags', 'B'),  # Land shape flags
    0x0E: ('cb_flags', 'B'),  # Callback flags
    0x0F: ('anim_info', 'W'),  # Animation information
    0x10: ('anim_speed', 'B'),  # Animation speed.
    0x11: ('cb25_triggers', 'B'),  # Triggers for callback 25
    0x12: ('flags', 'B'),  # Special flags
    0x13: ('tile_acceptance_list', TileAcceptanceProperty()),  # Tile acceptance list
}


class IndustryTileRef(int):
    def __repr__(self):
        return f'IndustryTileRef({super().__repr__()})'


class OldIndustryTileRef(int):
    def __repr__(self):
        return f'OldIndustryTileRef({super().__repr__()})'


class IndustryLayout(list):
    class BaseTile:

        def __init__(self, *, xofs: int, yofs: int):
            if not 0 <= xofs < 256:
                raise ValueError('Argument `xofs` must be in range 0..255')
            if not 0 <= yofs < 256:
                raise ValueError('Argument `yofs` must be in range 0..255')
            if xofs == 0 and yofs == 0x80:
                raise ValueError('Pair `xofs` == 0 and `yofs` == 0x80 have a special meaning and can''t be used in a layout')
            self.xofs = xofs
            self.yofs = yofs

        @abstractmethod
        def encode(self):
            pass

    class NewTile(BaseTile):
        def __init__(self, *, xofs: int, yofs: int, id: int):
            if not 0 <= id <= 0xFFFF:
                raise ValueError('`id` should in range 0..0xFFFF')
            super().__init__(xofs=xofs, yofs=yofs)
            self.id = id

        def __repr__(self):
            return f'IndustryLayout.NewTile(xofs={self.xofs}, yofs={self.yofs}, id={self.id})'

        def encode(self):
            return struct.pack('<BBBH', self.xofs, self.yofs, 0xFE, self.id)


    class OldTile(BaseTile):
        def __init__(self, *, xofs: int, yofs: int, id: int):
            if not 0 <= id < 0xFE:
                raise ValueError('`id` should in range 0..0xFD')
            # TODO spec limit is <= 0xAE but format allows 0xFE
            super().__init__(xofs=xofs, yofs=yofs)
            self.id = id

        def __repr__(self):
            return f'IndustryLayout.OldTile(xofs={self.xofs}, yofs={self.yofs}, id={self.id})'

        def encode(self):
            return struct.pack('<BBB', self.xofs, self.yofs, self.id)


    class SpecialCheck:
        def __init__(self, *, xofs: int, yofs: int):
            if not -128 <= xofs < 128:
                raise ValueError('Argument `xofs` must be in range -128..127')
            if not -128 <= yofs < 128:
                raise ValueError('Argument `yofs` must be in range -128..127')
            if xofs == 0 and yofs == -128:
                raise ValueError('Pair `xofs` == 0 and `yofs` == -128 have a special meaning and can''t be used in a layout')
            self.xofs = xofs
            self.yofs = yofs

        def __repr__(self):
            return f'IndustryLayout.SpecialCheck(xofs={self.xofs}, yofs={self.yofs})'

        def encode(self):
            return struct.pack('<bbB', self.xofs, self.yofs, 0xFF)

    Item = Union[NewTile, OldTile, SpecialCheck]

    def __init__(self, values: list[Item]):
        if values and values[0].xofs == 0xFE:
            raise ValueError('First item in `IndustryLayout` can''t have xofs == 0xFE')
        super().__init__(values)


class IndustryLayoutsProperty(Property):
    def validate(cls, value):
        if len(value) > 255:
            raise ValueError(f'Layout limit per industry exceeded (255 max, {len(value)} found)')

    def read(self, data, ofs):
        d = DataReader(data, ofs)
        num = d.get_byte()
        size = d.get_dword()
        res = []
        for _ in range(num):
            layout = []
            for k in range(size):  # effectively infinite loop as size is byte size not amount
                # TODO add warning comment if size is exceeded (like openttd does)
                xofs = d.get_byte()
                if xofs == 0xfe and k == 0:
                    # borrow base layout
                    raise NotImplementedError

                yofs = d.get_byte()

                if xofs == 0 and yofs == 0x80:
                    break

                gfx = d.get_byte()
                if gfx == 0xfe:
                    local_tile_id = d.get_word()
                    tile = IndustryLayout.NewTile(xofs=xofs, yofs=yofs, id=local_tile_id)
                elif gfx == 0xff:
                    tile = IndustryLayout.SpecialCheck(xofs=utoi8(xofs & 0xff), yofs=utoi8(yofs & 0xff))
                else:
                    tile = IndustryLayout.OldTile(xofs=xofs, yofs=yofs, id=gfx)

                layout.append(tile)
            res.append(IndustryLayout(layout))

        return res, d.offset

    def encode(self, value: list[IndustryLayout]):
        res = b''
        for l in value:
            for t in l:
                res += t.encode()
            res += bytes((0, 0x80))
        return struct.pack('<BI', len(value), len(res)) + res

    def format(self, value, indent):
        return pformat(value, indent=indent, indent_first=0)


ACTION0_INDUSTRY_PROPS = {
    0x08: ('substitute_type', 'B'),  # Substitute industry type
    0x09: ('override_type', 'B'),  # Industry type override
    0x0A: ('layouts', IndustryLayoutsProperty()),  # Set industry layout(s)
    0x0B: ('production_flags', 'B'),  # Industry production flags
    0x0C: ('closure_message', 'W'),  # Industry closure message
    0x0D: ('production_increase_message', 'W'),  # Production increase message
    0x0E: ('production_decrease_message', 'W'),  # Production decrease message
    0x0F: ('fund_cost', 'B'),  # Fund cost multiplier
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
    0x21: ('cb_flags', 'B'),  # Callback flags
    0x22: ('extra_cb_flags', 'B'),  # Callback flags
    0x23: ('destruction_cost', 'D'),  # Destruction cost multiplier
    0x24: ('station_text', 'W'),  # Default text for nearby station
    0x25: ('production_types', 'n*B'),  # Production cargo type list
    0x26: ('acceptance_types', 'n*B'),  # Acceptance cargo type list
    0x27: ('produciton_multipliers', 'n*B'),  # Production multiplier list
    0x28: ('input_multipliers', 'n*m*W'),  # Input cargo multiplier list
}

class TownGrowthEffect:
    PASSENGERS = 0x00
    MAIL = 0x02
    GOODS = 0x05
    CANDYFLOSS = 0x05
    WATER = 0x09
    FOOD = 0x0B
    FIZZY_DRINKS = 0x0B
    NONE = 0xFF


ACTION0_CARGO_PROPS = {
    0x08: ('bit_number', 'B'),  # Bit number for bitmasks
    0x09: ('type_name', 'W'),  # TextID for the cargo type name
    0x0A: ('unit_name', 'W'),  # TextID for the name of one unit from the cargo type
    0x0B: ('one_text', 'W'),  # TextID to be displayed for 1 unit of cargo
    0x0C: ('many_text', 'W'),  # TextID to be displayed for multiple units of cargo
    0x0D: ('type_abbreviation', 'W'),  # TextID for cargo type abbreviation
    0x0E: ('icon_sprite', 'W'),  # Sprite number for the icon of the cargo
    0x0F: ('weight', 'B'),  # Weight of one unit of the cargo
    0x10: ('penalty_lowerbound', 'B'),  # Penalty times
    0x11: ('single_penalty_length', 'B'),  # Penalty times
    0x12: ('base_price', 'D'),  # Base price
    0x13: ('station_list_colour', 'B'),  # Color for the station list window
    0x14: ('cargo_payment_list_colour', 'B'),  # Color for the cargo payment list window
    0x15: ('is_freight', 'B'),  # Freight status (for freight-weight-multiplier setting); 0=not freight, 1=is freight
    0x16: ('cargo_classes', 'W'),  # Cargo classes
    0x17: ('label', 'L'),  # Cargo label
    0x18: ('town_growth_effect', 'B'),  # Substitute type for town growth
    0x19: ('town_growth_multiplier', 'W'),  # Multiplier for town growth
    0x1A: ('cb_flags', 'B'),  # Callback flags
    0x1B: ('units_of_cargo', 'W'),  # TextID for displaying the units of a cargo
    0x1C: ('items_of_cargo', 'W'),  # TextID for displaying the amount of cargo
    0x1D: ('capacity_multiplier', 'W'),  # Capacity mulitplier
}

ACTION0_SOUND_EFFECT_PROPS = {
    0x08: ('relative_volume', 'B'),  # Relative volume
    0x09: ('priority', 'B'),  # Priority
    0x0A: ('override', 'B'),  # Override old sound
}

# TODO signal

class SizeTupleProperty(Property):
    def validate(cls, value):
        if isinstance(value, int) and 0 <= value < 256:
            return
        if (isinstance(value, (tuple, list)) and len(value) == 2
                and 0 <= value[0] < 16 and 0 <= value[1] < 16):
            return
        raise ValueError(f'Expected integer 0-255 or a tuple (x, y) both in range 0-15')

    def read(cls, data, ofs):
        return (data[ofs] & 0xf, data[ofs] >> 4), ofs + 1

    def encode(cls, value):
        if isinstance(value, tuple):
            value = (value[1] << 4) | value[0]
        return bytes((value, ))


ACTION0_OBJECT_PROPS = {
    0x08: ('class_label', 'L'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Class label, see below
    0x09: ('class_name_id', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Text ID for class
    0x0A: ('name_id', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Text ID for this object
    0x0B: ('climates_available', ClimateProperty()),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Climate availability
    0x0C: ('size', SizeTupleProperty()),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Byte representing size, see below
    0x0D: ('build_cost_multiplier', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object build cost factor (sets object removal cost factor as well)
    0x0E: ('introduction_date', DateProperty()),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Introduction date, see below
    0x0F: ('end_of_life_date', DateProperty()),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   End of life date, see below
    0x10: ('flags', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object flags, see below
    0x11: ('animation_info', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation information
    0x12: ('animation_speed', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation speed
    0x13: ('animation_triggers', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Animation triggers
    0x14: ('remove_cost_multiplier', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Object removal cost factor (set after object build cost factor)
    0x15: ('cb_flags', 'W'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Callback flags, see below
    0x16: ('height', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Height of the building
    0x17: ('num_views', 'B'),  # Supported by OpenTTD 1.1 (r20670)1.1 Supported by TTDPatch 2.6 (r2340)2.6   Number of object views
    0x18: ('count_per_map256', 'B'),  # Supported by OpenTTD 1.4 (r25879)1.4 Not supported by TTDPatch  Measure for number of objects placed upon map creation
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
    0x17: ('introduction_date', DateProperty()),  # Introduction date
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

ACTION0_ROADTYPE_PROPS = {
    0x08: ('label', 'L'),  # Road type label
    0x09: ('toolbar_caption', 'W'),  # StringID: Build road toolbar caption
    0x0A: ('menu_text', 'W'),  # StringID: Road construction dropdown text
    0x0B: ('build_window_caption', 'W'),  # StringID: Build vehicle window caption
    0x0C: ('autoreplace_text', 'W'),  # StringID: Autoreplace text
    0x0D: ('new_engine_text', 'W'),  # StringID: New engine text
    0x0F: ('powered_roadtype_list', 'n*L'),  # Powered road type list[1]
    0x10: ('roadtype_flags', 'B'),  # Road type flags
    0x13: ('construction_cost', 'W'),  # Construction costs
    0x14: ('speed_limit', 'W'),  # Speed limit
    0x16: ('map_colour', 'B'),  # Minimap colour
    0x17: ('introduction_date', DateProperty()), # Introduction date
    0x18: ('requires_roadtype_list', 'n*L'),  # Introduction required road type list[1]
    0x19: ('introduces_roadtype_list', 'n*L'),  # Introduced road type list[1]
    0x1A: ('sort_order', 'B'),  # Sort order
    0x1B: ('name', 'W'),  # StringID: Road type name
    0x1C: ('maintenance_cost', 'W'),  # Infrastructure maintenance cost factor
    0x1D: ('alternative_roadtype_list', 'n*L'),  # Alternate road type labels that shall be "redirected" to this road type
}

# TODO airport_tile

ACTION0_PROPS = {
    TRAIN: ACTION0_TRAIN_PROPS,
    RV: ACTION0_RV_PROPS,
    SHIP: ACTION0_SHIP_PROPS,
    AIRCRAFT: ACTION0_AIRCRAFT_PROPS,
    STATION: ACTION0_STATION_PROPS,
    BRIDGE: ACTION0_BRIDGE_PROPS,
    HOUSE: ACTION0_HOUSE_PROPS,
    GLOBAL_VAR: ACTION0_GLOBAL_PROPS,
    INDUSTRY_TILE: ACTION0_INDUSTRY_TILE_PROPS,
    INDUSTRY: ACTION0_INDUSTRY_PROPS,
    CARGO: ACTION0_CARGO_PROPS,
    SOUND_EFFECT: ACTION0_SOUND_EFFECT_PROPS,
    OBJECT: ACTION0_OBJECT_PROPS,
    RAILTYPE: ACTION0_RAILTYPE_PROPS,
    ROADTYPE: ACTION0_ROADTYPE_PROPS,
    TRAMTYPE: ACTION0_ROADTYPE_PROPS,
}

ACTION0_PROP_DICT = {
    feature: {name: (id, size) for id, (name, size, *_) in fdict.items()}
    for feature, fdict in ACTION0_PROPS.items()
}

def py_property(feature, name, value, indent=4):
    prop = ACTION0_PROP_DICT[feature][name][1]
    if isinstance(prop, Property):
        return prop.format(value, indent=indent)
    return repr(value)

# Action 0

class RVFlags:
    TRAM = 0x1 # Vehicle is a tram/light rail vehicle and requires tram tracks to operate
    USE_2CC = 0x2 # Uses two company colors
    RESERVED_1 = 0x4  # reserved, do not use
    RESERVED_2 = 0x8  # reserved, do not use
    AUTOREFIT = 0x10  # Auto-refitting is enabled for refits where callback 15E allows it or prop 1A specifies zero cost.
    USE_CARGO_MULT = 0x20  # Use cargo multiplier for default cargo. See page about vehicle refitting.
    NO_BREAKDOWN_SMOKE = 0x40  # Disable breakdown smoke effect.
    USE_SPRITE_STACK = 0x80  # Compose vehicle from multiple sprites.


class TrainFlags:
    TILT = 0x1
    USE_2CC = 0x2
    MULTIPLE_UNIT = 0x4
    ALLOW_FLIPPING = 0x8
    AUTOREFIT = 0x10
    USE_CARGO_MULT = 0x20
    NO_BREAKDOWN_SMOKE = 0x40
    USE_SPRITE_STACK = 0x80


class ShipFlags:
    USE_2CC = 0x2
    AUTOREFIT = 0x10
    USE_CARGO_MULT = 0x20
    NO_BREAKDOWN_SMOKE = 0x40
    USE_SPRITE_STACK = 0x80


class AIFlags:
    CARGO = 0x00
    PASSENGER = 0x01


class TrainRunningCost:
    STEAM = 0x4C30
    DIESEL = 0x4C36
    ELECTRIC = 0x4C3C
    ROADVEH = 0x4C48
    NONE = 0x0000


class TrainVisualEffect:
    DEFAULT = 0x00
    STEAM = 0x10
    DIESEL = 0x20
    ELECTRIC = 0x30
    DISABLE = 0x40


def train_hpi(value):
    return value


def train_ton(value):
    return value


def nml_te(value):
    return int(float(value) * 255 + 0.5)

def nml_drag(value):
    return int(float(value) * 255 + 0.5)


class CargoClass:
    PASSENGERS = 0x1  # passengers, also tourists (ECS)
    MAIL = 0x2  # mail
    EXPRESS = 0x4  # express goods, also tourists (ECS)
    ARMOURED = 0x8  # valuables, diamonds, gold and alike
    BULK = 0x10  # coal, ore, grain,...
    PIECE_GOODS = 0x20  # containers, crates, livestock
    LIQUID = 0x40  # oil, milk, water, ...
    REFRIGERATED = 0x80  # food, milk, ...
    HAZARDOUS = 0x100  # chemicals?, uranium, ...
    COVERED = 0x200  # grain, cement, fruit, ...
    OVERSIZED = 0x400  # vehicles, ...
    POWDERIZED = 0x800  # cement, ...
    NON_POURABLE = 0x1000  # sugar cane, wool or straw bales, ...
    SPECIAL = 0x8000  # Special cargo, used for refit tricks. (e.g. regearing in NARS)
    NONE = 0  # S  Special value that you can used to instead of 0.
    # ALL_NORMAL_CARGO_CLASSES    Bitmask of all cargo classes except CC_SPECIAL. This is the same as bitmask(CC_PASSENGERS, CC_MAIL, ..., CC_OVERSIZED). Note: This is already a bitmask, don't use the bitmask(..) function with this.
    # ALL_CARGO_CLASSES   bitmask(CC_SPECIAL) Note: This is already a bitmask, don't use the bitmask(..) function with this.


class DefineMultiple(Action):
    def __init__(self, *, feature, first_id, props, count=None):
        assert isinstance(feature, Feature)
        super().__init__()
        self.feature = feature
        self.first_id = first_id
        if count is None:
            count = len(next(iter(props.values())))
        self.count = count
        if feature not in ACTION0_PROP_DICT:
            raise NotImplementedError
        self.props = props
        for x in props:
            if x not in ACTION0_PROP_DICT[feature]:
                raise ValueError(f'Unknown property `{x}` for a feature {feature}')

    def _encode_label(self, value):
        if isinstance(value, int): return struct.pack('<I', value)
        assert isinstance(value, bytes), (type(value), value)
        assert len(value) == 4, (len(value), value)
        return value

    def _encode_value(self, context, value, fmt):
        if isinstance(fmt, Property):
            return fmt.encode(value)
        if fmt == 'B': return struct.pack('<B', value)
        if fmt == 'b': return struct.pack('<b', value)
        if fmt == 'W': return struct.pack('<H', value)
        if fmt == 'D': return struct.pack('<I', value)
        if fmt == 'L': return self._encode_label(value)
        if fmt == 'B*': return struct.pack('<BH', 255, value)
        if fmt == 'n*B':
            if isinstance(value, bytes):
                if len(value) >= 256:
                    context.format_error(self, f'Too many bytes ({len(value)}) for property format n*B (max 255)')
                return struct.pack('<B', len(value)) + value
            elif isinstance(value, list):
                if len(value) >= 256:
                    context.format_error(self, f'Too many items ({len(value)}) for property format n*B (max 255)')
                fail = next((isinstance(x, int) and 0 <= x <= 255 for x in value), None)
                if fail is not None:
                    context.format_error(self, f'Items should be integers in range(0, 256) for property format n*B (max 255), found {fail}')
                return struct.pack('<B', len(value)) + bytes(value)
        if fmt == '8*B':
            if not isinstance(value, tuple):
                context.format_error(self, f'8*B format value needs to be a tuple')
            assert len(value) == 8, (len(value), value)
            res = bytes()
            for l in value:
                res += struct.pack('<B', l)
            return res
        if fmt == '2*L':
            if not isinstance(value, tuple):
                context.format_error(self, f'2*L format value needs to be a tuple')
            assert len(value) == 2, (len(value), value)
            return self._encode_label(value[0]) + self._encode_label(value[1])
        if fmt == 'n*L':
            res = bytes((len(value),))
            for l in value:
                res += self._encode_label(l)
            return res
        raise RuntimeError(f'Unsupported property format `{fmt}`')

    def get_data(self, context):
        res = struct.pack('<BBBBBH',
            0, self.feature.id, len(self.props), self.count, 255, self.first_id)
        pdict = ACTION0_PROP_DICT[self.feature]
        for prop, value in self.props.items():
            code, fmt = pdict[prop]
            res += bytes((code,))
            for i in range(self.count):
                try:
                    res += self._encode_value(context, value[i], fmt)
                except Exception as e:
                    raise context.log_error(self, f'Error encoding value {value[i]} for property {prop} with format `{fmt}`: {e}')
        return res

    def py(self, context):
        def value_func(k, v):
            pdata = ', '.join(py_property(self.feature, k, value, indent=16) for value in v)
            return f'[{pdata}]'

        propstr = _py_dict(self.props, repr, value_func)

        return f'''
        DefineMultiple(
            feature={self.feature},
            first_id={self.first_id},
            count={self.count},
            props={propstr}
        )
        '''

    def __eq__(self, action):
        return (
            isinstance(action, DefineMultiple) and
            self.feature == action.feature and
            self.first_id == action.first_id and
            self.count == action.count and
            self.props == action.props
        )


class Define(DefineMultiple):
    def __init__(self, *, feature, id, props):
        multi_props = {k: [v] for k, v in props.items()}
        super().__init__(feature=feature, first_id=id, count=1, props=multi_props)

    @property
    def id(self):
        return self.first_id

    def py(self, context):
        propstr = _py_dict(self.props, repr, lambda k, v: py_property(self.feature, k, v[0], indent=16))
        return f'''
        Define(
            feature={self.feature},
            id={self.first_id},
            props={propstr}
        )'''


# Action 1

class Action1(Action):
    def __init__(self, feature, set_count, sprite_count, *, first_set=None):
        assert isinstance(feature, Feature)
        super().__init__()
        self.feature = feature
        self.set_count = set_count
        self.sprite_count = sprite_count
        self.first_set = first_set

    def get_data(self, context):
        if self.set_count <= 255 and self.first_set is None:
            # Basic format
            return struct.pack('<BBBBH', 0x01,
                               self.feature.id, self.set_count, 0xFF, self.sprite_count)

        # Extended format (OpenTTD >= 1.2.2)
        return struct.pack('<BBBBHBHBH', 0x01,
                           self.feature.id,
                           0x00,
                           0xFF, self.first_set or 0,
                           0xFF, self.set_count,
                           0xFF, self.sprite_count)

    def py(self, context):
        return f'''
        Action1(
            feature={self.feature},
            set_count={self.set_count},
            sprite_count={self.sprite_count},
            first_set={self.first_set},
        )'''


class SpriteSet(Action1):
    def __init__(self, feature, count):
        super().__init__(feature, 1, count)

    def py(self, context):
        return f'''
        SpriteSet(
            feature={self.feature},
            count={self.sprite_count},
        )'''


# Action 2

class GenericSpriteLayout(Action, ReferenceableAction):
    # @typechecked
    def __init__(
        self,
        *,
        ent1: Sequence[int],
        ent2: Sequence[int],
        feature: Optional[Feature] = None,
        ref_id : Optional[int] = None,
    ):
        assert feature not in (HOUSE, INDUSTRY_TILE, OBJECT, AIRPORT_TILE, INDUSTRY), feature
        check_sequence_range(ent1, 0, 0xffff, 'ent1')
        check_sequence_range(ent2, 0, 0xffff, 'ent2')

        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.ent1 = ent1
        self.ent2 = ent2

    def get_data(self, context):
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

    def py(self, context):
        ref_id_str = '            ref_id={self.ref_id},\n'
        return f''' \
        GenericSpriteLayout(
            feature={self.feature},{self._py_ref_id(context)}
            ent1={self.ent1!r},
            ent2={self.ent2!r},
        )
        '''


class BasicSpriteLayout(Action, ReferenceableAction):
    def __init__(self, *, ground, building, feature=None, ref_id=None):
        assert feature in (HOUSE, INDUSTRY_TILE, OBJECT, INDUSTRY), feature
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.ground = ground
        self.building = building

    def get_data(self, context):
        return struct.pack('<BBBBIIbbBBB', 0x02, self.feature.id, self.ref_id, 0,
                           self.ground['sprite'].to_grf(global_if_flagged=False),
                           self.building['sprite'].to_grf(global_if_flagged=False),
                           *self.building['offset'],
                           *self.building['extent'])

    def py(self, context):
        return f''' \
        BasicSpriteLayout(
            feature={self.feature},{self._py_ref_id(context)}
            ground={self.ground!r},
            building={self.building!r},
        )
        '''


class AdvancedSpriteLayout(Action, ReferenceableAction):
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
        res = struct.pack('<IH', sprite['sprite'].to_grf(global_if_flagged=False), flags)

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

    def get_data(self, context):
        res = struct.pack('<BBBB', 0x02, self.feature.id, self.ref_id, len(self.buildings) + 0x40)
        res += self._encode_sprite(self.ground)
        for s in self.buildings:
            res += self._encode_sprite(s, True)

        return res

    def py(self, context):
        return f'''
        {self.__class__.__name__}(
            feature={self.feature},{self._py_ref_id(context)}
            ground={self.ground!r},
            buildings={self.buildings!r},
        )'''


class ExtendedSpriteLayout(AdvancedSpriteLayout):
    # Not implemented yet
    def __init__(self, *args, **kw):
        super().__init__(*args, has_flags=False, **kw)


class CodeContext:
    def __init__(self, register, subroutines):
        self.subroutines = subroutines
        self._min_register = register
        self._used_registers = set()
        self._variables = {}

    def reserve_register(self):
        register = None
        for i in range(self._min_register, 256):
            if i not in self._used_registers:
                register = i
                break
        if register is None:
            raise RuntimeError('Out of temporary registers')
        self._min_register = register + 1
        self._used_registers.add(register)
        return register

    def release_register(self, register):
        self._used_registers.remove(register)
        self._min_register = min(self._min_register, register)

    def reserve_variable(self, name):
        register = self._variables.get(name)
        if register is not None:
            return register
        register = self.reserve_register()
        self._variables[name] = register
        return register

    def get_variable_register(self, name):
        return self._variables.get(name)


class Switch(Action, ReferenceableAction, ReferencingAction):
    # TODO Change archaic term "subrotine" to "function"
    def __init__(self, code, ranges, default, *, feature=None, ref_id=None, related_scope=False, subroutines=None):
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.related_scope = related_scope

        self._ranges = []
        if isinstance(ranges, dict):
            ranges = ranges.items()
        for r in ranges:
            if isinstance(r, Range):
                self._ranges.append(r)
            else:
                set_obj = r[1]
                if isinstance(r[0], tuple):
                    low, high = r[0]
                else:
                    low = high = r[0]
                self._ranges.append(Range(low, high, set_obj))

        self.default = default
        if isinstance(code, (list, tuple)):
            code = '\n'.join(code)
        self.code = code
        self._parsed_code = None
        self.subroutines = subroutines

    def get_refs(self):
        if self.subroutines is not None:
            yield from self.subroutines.values()
        yield from (r.ref for r in self._ranges)
        yield self.default

    def set_refs(self, refs):
        it = iter(refs)
        if self.subroutines is not None:
            for k in self.subroutines.keys():
                self.subroutines[k] = next(it)
        for r in self._ranges:
            r.ref = next(it)
        self.default = next(it)

    @property
    def parsed_code(self):
        if self._parsed_code is None:
            if self.related_scope:
                feature = {
                    TRAIN: TRAIN,
                    RV: RV,
                    SHIP: SHIP,
                    AIRCRAFT: AIRCRAFT,
                    BRIDGE: TOWN,
                    HOUSE: TOWN,
                    INDUSTRY_TILE: INDUSTRY,
                    INDUSTRY: TOWN,
                    AIRPORT: TOWN,
                    OBJECT: TOWN,
                    AIRPORT_TILE: AIRPORT,
                }[self.feature]
            else:
                feature = self.feature
            try:
                self._parsed_code = parse_code(feature, self.code)
                if self._parsed_code is None:
                    raise RuntimeError('parser error')
            except Exception as e:
                print('Failed code: \n', self.code)
                raise
        return self._parsed_code

    def get_data(self, context):
        context = CodeContext(subroutines=self.subroutines, register=0x80)
        res = bytes((0x02, self.feature.id, self.ref_id, 0x8a if self.related_scope else 0x89))
        ast = self.parsed_code
        code = ast[0].compile(context)[1]
        for c in ast[1:]:
            code += bytes((OP_INIT,))
            code += c.compile(context)[1]
        res += code[:-5]
        res += bytes((code[-5] & ~0x20,))  # mark the end of a chain
        res += code[-4:]

        ranges = b''
        for r in self._ranges:
            ref_id = get_ref_id(r.ref)
            if r.low < 0 and r.high >= 0:
                # Split negative-positive range in two
                ranges += struct.pack('<Hii', ref_id, r.low, -1)
                ranges += struct.pack('<Hii', ref_id, 0, r.high)
            else:
                ranges += struct.pack('<Hii', ref_id, r.low, r.high)
        res += bytes((len(ranges) // 10,))
        res += ranges

        res += struct.pack('<H', get_ref_id(self.default))
        return res

    def py(self, context):
        if '\n' not in self.code:
            code_str = repr(self.code)
        else:
            code_str = textwrap.indent("'''\n" + self.code + "'''", ' ' * 16).lstrip()
        ranges_str = _py_ref_dict(((r, r.ref) for r in self._ranges), lambda k, r: self._format_ref(context, r))
        default_str = self._format_ref(context, self.default)
        return f'''
        Switch(
            feature={self.feature},{self._py_ref_id(context)}
            related_scope={self.related_scope},
            code={code_str},
            ranges={ranges_str},
            default={default_str},
        )'''

    def __eq__(self, action):
        return (
            isinstance(action, Switch) and
            self.feature == action.feature and
            self.ref_id == action.ref_id and
            self.related_scope == action.related_scope and
            self.code == action.code and
            self._ranges == action._ranges and
            ref_eq(self.default, action.default)
        )


# TODO separate eval-switch and ranges-switch better
class Eval(Switch):
    def __init__(self, code, default, **kw):
        super().__init__(code=code, ranges=(), default=default, **kw)


class RandomSwitch(Action, ReferenceableAction, ReferencingAction):
    def __init__(self, *, scope, triggers, cmp_all, lowest_bit, groups, count=None, feature=None, ref_id=None):
        if scope == 'relative' and feature in VEHICLE_FEATURES:
            assert count is not None
        else:
            assert count is None
        super().__init__()
        self.feature = feature
        self.ref_id = ref_id
        self.scope = scope
        self.count = count
        self.triggers = triggers
        self.cmp_all = cmp_all
        self.lowest_bit = lowest_bit
        self.groups = groups

    def get_refs(self):
        yield from self.groups

    def set_refs(self, refs):
        self.groups = refs

    def get_data(self, context):
        atype = {'self': 0x80, 'parent': 0x83, 'relative': 0x84}[self.scope]
        res = bytes((0x02, self.feature.id, self.ref_id, atype))
        if self.scope == 'relative' and self.feature not in VEHICLE_FEATURES:
            res += bytes((self.count, ))
        # TODO vars instead of refs
        res += struct.pack(
            '<BBB' + 'H' * len(self.groups),
            self.triggers | (int(self.cmp_all) * 0x80),
            self.lowest_bit,
            len(self.groups),
            *map(get_ref_id, self.groups)
        )
        return res

    def py(self, context):
        groupsdata = ', '.join(self._format_ref(context, r) for r in self.groups)
        return f'''
        RandomSwitch(
            feature={self.feature},{self._py_ref_id(context)}
            scope={self.scope!r},
            count={self.count!r},
            triggers={self.triggers},
            cmp_all={self.cmp_all},
            lowest_bit={self.lowest_bit},
            groups=[{groupsdata}],
        )'''


class IndustryProductionCallback(Action):
    def __init__(self, *, inputs, outputs, do_again, version=2):
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

    def get_data(self, context):
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

    def py(self, context):
        return f'''
        {self.__class__.__name__}(
            version={self.version},
            inputs={self.inputs!r},
            outputs={self.outputs!r},
            do_again={self.do_again},
        )'''


# Action 3

class Action3(Action, ReferencingAction):
    def __init__(self, *, feature, ids, maps, default, wagon_override=False):
        assert isinstance(feature, Feature), feature
        super().__init__()
        self.feature = feature
        self.ids = ids
        self.maps = maps
        self.default = default
        self.wagon_override = wagon_override  # TODO make a separate class for overrides

    def get_refs(self):
        yield from self.maps.values()
        yield self.default

    def set_refs(self, refs):
        self.default = refs[-1]
        for i, k in enumerate(self.maps.keys()):
            self.maps[k] = refs[i]

    def get_data(self, context):
        idcount = len(self.ids)
        mcount = len(self.maps)
        if idcount == 0:
            return struct.pack('<BBBBH', 0x03, self.feature.id, idcount, 0, self.default)
        else:
            idlist = self.ids
            idfmt = 'B'
            if (self.feature in VEHICLE_FEATURES or self.feature == STATION) and any(x >= 0xff for x in self.ids):
                idlist = [0xff] * (2 * idcount)
                idlist[1::2] = self.ids
                idfmt = 'BH'
            return struct.pack(
                '<BBB' + idfmt * idcount + 'B' + 'BH' * mcount + 'H',
                0x03, self.feature.id, idcount,

                *idlist, mcount, *sum(((k, get_ref_id(x)) for k, x in self.maps.items()), ()),
                get_ref_id(self.default))

    def py(self, context):
        value_func = lambda k, v: self._format_ref(context, v)
        return f'''
        Action3(
            feature={self.feature},
            wagon_override={self.wagon_override},
            ids={self.ids!r},
            maps={_py_ref_dict(self.maps, value_func)},
            default={self._format_ref(context, self.default)},
        )'''

    def __eq__(self, action):
        return (
            isinstance(action, Action3) and
            self.feature == action.feature and
            self.ids == action.ids and
            ref_dict_eq(self.maps, action.maps) and
            ref_eq(self.default, action.default) and
            self.wagon_override == action.wagon_override
        )


class Map(Action3):
    def __init__(self, definition, maps, default):
        assert isinstance(definition, DefineMultiple)
        super().__init__(
            feature=definition.feature,
            ids=[definition.first_id],
            maps=maps,
            default=default
        )

    def py(self, context):
        value_func = lambda k, v: self._format_ref(context, v)
        return f'''
        Action3(
            feature={self.feature},
            wagon_override={self.wagon_override},
            ids={self.ids!r},
            maps={_py_ref_dict(self.maps, value_func)},
            default={self._format_ref(context, self.default)},
        )'''


# Action 4

class DefineStrings(Action):
    def __init__(self, *, feature, offset, is_generic_offset, strings, lang=ANY_LANGUAGE):
        assert isinstance(feature, Feature), feature
        if feature in VEHICLE_FEATURES or is_generic_offset:
            if not 0 <= offset <= 0xffff:
                raise ValueError(f'{self.__class__.__name__} `offset` {offset} is not in range 0..0xffff (for generic offset or vehicles)')
        else:
            if not 0 <= offset <= 0xff:
                raise ValueError(f'{self.__class__.__name__} `offset` {offset} is not in range 0..0xff (for non-generic non-vehicle offset)')

        assert all(isinstance(s, bytes) for s in strings), strings

        super().__init__()
        self.feature = feature
        self.lang = lang
        self.offset = offset
        self.is_generic_offset = is_generic_offset
        self.strings = strings

    def get_data(self, context):
        str_data = b'\0'.join(self.strings) + b'\0'
        if self.is_generic_offset:
            return struct.pack('<BBBBH', 0x04, self.feature.id, self.lang | 0x80, len(self.strings), self.offset) + str_data
        elif self.feature in VEHICLE_FEATURES and self.offset >= 0xff:
            return struct.pack('<BBBBBH', 0x04, self.feature.id, self.lang, len(self.strings), 0xff, self.offset) + str_data
        else:
            return struct.pack('<BBBBB', 0x04, self.feature.id, self.lang, len(self.strings), self.offset) + str_data

    def py(self, context):
        offset_str = f'0x{self.offset:04x}' if self.is_generic_offset else self.offset
        return f'''
        {self.__class__.__name__}(
            feature={self.feature},
            lang={self.lang},
            offset={offset_str},
            is_generic_offset={self.is_generic_offset},
            strings=''' + pformat(self.strings, indent_first=0, indent=10 + 8) + '''
        )'''

    def __eq__(self, action):
        return (
            isinstance(action, DefineStrings) and
            self.feature == action.feature and
            self.lang == action.lang and
            self.offset == action.offset and
            self.is_generic_offset == action.is_generic_offset and
            self.strings == action.strings
        )

# Action 5

class ReplaceNewSprites(Action):
    def __init__(self, set_type, count, *, offset=None):
        assert isinstance(set_type, int)
        assert isinstance(count, int)

        super().__init__()
        self.set_type = set_type
        self.count = count
        self.offset = offset

    def get_data(self, context):
        if self.offset is None:
            return bytes((0x5,)) + struct.pack('<BBH', self.set_type, 0xff, self.count)
        return bytes((0x5,)) + struct.pack('<BBHBH', self.set_type | 0x80, 0xff, self.count, 0xff, self.offset)

    def py(self, context):
        return f'ReplaceNewSprites(set_type={self.set_type}, count={self.count}, offset={self.offset})'


# Action 6

class ModifySprites(Action):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def get_data(self, context):
        return struct.pack(
            '<B' + 'BBBH' * len(self.params) + 'B',
            0x06,
            *sum(((p['num'], p['size'], 0xff, p['offset']) for p in self.params), ()),
            0xff
        )

    def py(self, context):
        return f'ModifySprites(\n' + pformat(self.params) + '\n)'


# Action 7, Action 9

class If(Action):
    def __init__(self, *, is_static, variable, condition, value, skip, varsize=None):
        assert varsize is not None or isinstance(value, bytes)

        super().__init__()
        self.is_static = is_static
        self.variable = variable
        self.varsize = varsize
        self.condition = condition
        self.value = value
        self.skip = skip

    def get_data(self, context):
        varsize = self.varsize if self.varsize is not None else len(self.value)
        res = bytes((0x09 if self.is_static else 0x07, self.variable, varsize, self.condition))
        if isinstance(self.value, bytes):
            res += self.value
        else:
            for i in range(self.varsize):
                res += bytes(((self.value >> (8 * i)) & 0xff,))
        res += bytes((self.skip,))
        return res

    def py(self, context):
        valuestr = f'  # {repr(struct.pack("<I", self.value))[2:-1]}' if self.value is not None and self.value <= 0xFFFFFFFF else ''
        return f'''
            If(
                is_static={self.is_static},
                variable={self.variable},
                varsize={self.varsize},
                condition={self.condition},
                value={self.value},{valuestr}
                skip={self.skip},
            )'''

# Action7 = Action9 = If


# Action 8

class SetDescription(Action):
    def __init__(self, *, grfid, name, description, format_version=8):
        assert isinstance(grfid, bytes)
        assert isinstance(name, (bytes, str))
        assert isinstance(description, (bytes, str))

        if isinstance(name, str):
            name = name.encode('utf-8')
        if isinstance(description, str):
            description = description.encode('utf-8')

        self.format_version = format_version
        self.grfid = grfid
        self.name = name
        self.description = description
        self._data = bytes((0x08, self.format_version))
        self._data += self.grfid + self.name + b'\x00' + self.description + b'\x00'

    def __eq__(self, action):
        return (
            isinstance(action, SetDescription) and
            self.grfid == action.grfid and
            self.name == action.name and
            self.description == action.description and
            self.format_version == action.format_version
        )

    def get_data(self, context):
        return self._data

    def py(self, context):
        return f'''
        SetDescription(
            format_version={self.format_version},
            grfid={self.grfid},
            name={self.name},
            description={self.description},
        )'''

Action8 = SetDescription

# Action A

class ReplaceOldSprites(Action):
    def __init__(self, sets):
        assert isinstance(sets, (list, tuple))
        assert len(sets) <= 0xff
        for first, num in sets:
            assert isinstance(first, int)
            assert isinstance(num, int)

        super().__init__()
        self.sets = sets

    def get_data(self, context):
        return bytes((0xa, len(self.sets))) + b''.join(struct.pack('<BH', num, first) for first, num in self.sets)

    def py(self, context):
        return f'ReplaceOldSprites(sets={self.sets!r})'

ActionA = ReplaceOldSprites

# Action B

class ErrorMessage(Action):
    def __init__(self, *, severity, lang, message, text_param=None, params=None):
        super().__init__()
        assert 0 <= severity < 256, severity
        assert 0 <= lang <= ANY_LANGUAGE, lang
        assert all(0 <= p < 256 for p in params), params
        self.severity = severity
        self.lang = lang
        assert isinstance(message, (int, str, bytes))
        self.message = message
        self.text_param = text_param
        self.params = params or []

    def get_data(self, context):
        message_id = self.message if isinstance(self.message, int) else 0xFF
        data = bytes((0x0B, self.severity, self.lang, message_id))
        if message_id == 0xFF:
            data += to_bytes(self.message)
        data += b'\0'
        if self.text_param is not None or self.params:
            data += to_bytes(self.text_param or '')
            data += b'\0'
            data += bytes(self.params)
        return data

    def py(self, context):
        return f'''
        ErrorMessage(
            severity={self.severity},
            lang={self.lang},
            message={self.message!r},
            text_param={self.text_param!r},
            params={self.params!r},
        )'''

ActionB = ErrorMessage

# Action C

class Comment(Action):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def get_data(self, context):
        return bytes((0x0C)) + self.data

    def py(self, context):
        return f'Comment({self.data!r})'

ActionC = Comment

# Action D

class ComputeParameters(Action):
    def __init__(self, *, target, operation, if_undefined, source1, source2, value=None):
            super().__init__()
            self.target = target
            self.operation = operation
            self.if_undefined = if_undefined
            self.source1 = source1
            self.source2 = source2
            self.value = value

    def get_data(self, context):
        operation_byte = self.operation | 0x80 * self.if_undefined
        res = bytes((0x0D, self.target, operation_byte, self.source1, self.source2))
        if self.source1 == 0xff or self.source2 == 0xff:
            if isinstance(self.value, bytes):
                res += self.value
            else:
                res += struct.pack('<I', self.value)
        return res

    def py(self, context):
        # fmt = OPERATIONS[operation]
        # sf = lambda x: f'[{x:02x}]' if x != 0xff else str(value)
        # target_str = f'[{target:02x}]'
        # op_str = fmt.format(target=target_str, source1=sf(source1), source2=sf(source2))
        valuestr = f'  # {repr(struct.pack("<I", self.value))[2:-1]}' if self.value is not None else ''
        return f'''
        ComputeParameters(
            target={self.target},
            operation={self.operation},
            if_undefined={self.if_undefined},
            source1={self.source1},
            source2={self.source2},
            value={self.value},{valuestr}
        )'''


# Action 10

class Label(Action):
    def __init__(self, label, comment):
        if not(0 <= label <= 255):
            raise ValueError(f'label {label} not in range 0..255')
        super().__init__()
        self.label = label
        self.comment = comment

    def get_data(self, context):
        return bytes((0x10, self.label)) + self.comment

    def py(self, context):
        return f'Label({self.label}, {self.comment!r})'


# Action 11

class SoundEffects(Action):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def get_data(self, context):
        return struct.pack('<BH', 0x11, self.count)

    def py(self, context):
        return f'SoundEffects({self.count})'


class ImportSound(Action):
    def __init__(self, grfid, number):
        super().__init__()
        self.grfid = grfid
        self.number = number

    def get_data(self, context):
        return b'\xFE\x00' + self.grfid + struct.pack('<H', self.number)

    def py(self, context):
        return f'ImportSound({self.grfid!r}, {self.number})'


# Action 12

class UnicodeGlyphs(Action):
    def __init__(self, glyphs):
        assert len(glyphs) < 256
        super().__init__()
        self.glyphs = []

    def get_data(self, context):
        data = bytes((0x12, len(self.glyphs)))
        for g in self.glyphs:
            data += struct.pack('<BBH', g['font'], g['count'], g['base'])
        return data


# Action 13

class Translations(Action):
    def __init__(self, *, grfid, offset, is_generic_offset, lang=ANY_LANGUAGE):
        if not 0xD000 <= offset <= 0xDC00:
            raise ValueError(f'{self.__class__.__name__} `offset` {offset} is not in range 0xD000..0xDC00')

        super().__init__()
        self.grfid = grfid
        self.lang = lang
        self.offset = offset
        self.strings = []
        for s in strings:
            if isinstance(s, bytes):
                self.strings.append(s)
            else:
                assert isinstance(s, str), type(s)
                self.strings.append(s.encode('utf-8'))

    def get_data(self, context):
        res = bytes((0x13,)) + self.grfid
        str_data = b'\0'.join(self.strings) + b'\0'
        res += struct.pack('<BBH', self.lang, len(self.strings), self.offset) + str_data
        return res

    def py(self, context):
        offset_str = f'0x{self.offset:04x}' if self.is_generic_offset else self.offset
        return f'''
        {self.__class__.__name__}(
            grfid={self.grfid},
            lang={self.lang},
            offset={offset_str},
            strings=''' + pformat(self.strings, indent_first=0, indent=10 + 8) + '''
        )'''


# Action 14

class SetProperties(Action):
    def __init__(self, props):
        super().__init__()
        self.props = props

    def __eq__(self, action):
        return (
            isinstance(action, SetProperties) and
            self.props == action.props
        )

    def get_data(self, context):
        data = b'\x14'

        def rec(k, v):
            nonlocal data
            assert k is not None or isinstance(v, dict), (k, v)

            if k is not None:
                if isinstance(k, str):
                    k = k.encode('utf-8')
                elif isinstance(k, int):
                    k = struct.pack('<I', k)
            if isinstance(v, dict):
                if k is not None:
                    data += b'C' + k
                for ck, cv in v.items():
                    rec(ck, cv)
                data += b'\0'
            elif isinstance(v, tuple):
                # TOOD move checks to construction
                if len(v) != 2:
                    raise ValueError(f'Expected a pair, found: {v}')
                lang_id, text = v
                assert isinstance(lang_id, int) and isinstance(text, (bytes, str))
                if isinstance(text, str):
                    text = text.encode('utf-8')
                data += b'T' + k + bytes((lang_id,)) + text + b'\0'
            elif isinstance(v, list):
                for x in v:
                    rec(k, x)
            elif isinstance(v, str):
                data += b'T' + k + b'\0' + v.encode('utf-8') + b'\0'
            elif isinstance(v, int):
                data += b'B' + k + b'\x04\x00' + struct.pack('<I', v)
            elif isinstance(v, bytes):
                data += b'B' + k + struct.pack('<H', len(v)) + v
            else:
                # TODO reorganize code so it checks isinstance(v, StringRef):
                for lang_id, text in v.get_pairs():
                    data += b'T' + k + bytes((lang_id,)) + text + b'\0'
            # else:
            #     raise ValueError(f'Unexpected property value type {type(v)}: {v}')

        rec(None, self.props)
        return data

    def py(self, context):
        return f'''
        {self.__class__.__name__}(
            props={pformat(self.props, indent_first=0, indent=10 + 8)}
        )'''


class PrettyPrinter(pprint.PrettyPrinter):

    def __init__(self, **kw):
        self._compact_copy = kw.get('compact', False)
        super().__init__(**kw)

    # def _format_items_no_compact(self, *args, **kw):
    #     self._compact = False
    #     super()._format_items(*args, **kw)
    #     self._compact = self._compact_copy

    def _format_items_no_compact(self, items, stream, indent, allowance, context, level):
        write = stream.write
        indent += self._indent_per_level
        if self._indent_per_level > 1:
            write((self._indent_per_level - 1) * ' ')
        delimnl = ',\n' + ' ' * indent
        delim = ''
        width = max_width = self._width - indent + 1
        it = iter(items)
        try:
            next_ent = next(it)
        except StopIteration:
            return
        last = False
        while not last:
            ent = next_ent
            try:
                next_ent = next(it)
            except StopIteration:
                last = True
                max_width -= allowance
                width -= allowance
            write(delim)
            delim = delimnl
            self._format(ent, stream, indent,
                         allowance if last else 1,
                         context, level)

    def _format_list_object(self, name, items, stream, indent, allowance, context, level):
        if not len(items):
            stream.write(f'{name}([])')
            return
        stream.write(f'{name}([\n ')
        stream.write(' ' * indent)
        self._format_items_no_compact(items, stream, indent, allowance + 2, context, level)
        stream.write('\n')
        stream.write(' ' * indent)
        stream.write('])')

    def _format(self, obj, stream, indent, allowance, context, level):
        if isinstance(obj, SpriteLayoutList):
            self._format_list_object(obj.__class__.__name__, obj.layouts, stream, indent, allowance, context, level)
        elif isinstance(obj, SpriteLayout):
            self._format_list_object(obj.__class__.__name__, obj.sprites, stream, indent, allowance, context, level)
        elif isinstance(obj, BaseSpriteData):
            name = obj.__class__.__name__
            stream.write(f'{name}(\n')
            indent += self._indent_per_level
            stream.write(' ' * indent)
            stream.write(f'sprite={obj.sprite!r},\n')
            if obj.flags is not None:
                stream.write(' ' * indent)
                stream.write(f'flags={obj.flags!r},\n')
            if obj.registers is not None:
                stream.write(' ' * indent)
                stream.write(f'registers={obj.registers!r},\n')
            if isinstance(obj, TopLevelSprite) and obj.children:
                stream.write(' ' * indent)
                stream.write(f'children=[\n ')
                stream.write(' ' * indent)
                self._format_items_no_compact(obj.children, stream, indent, allowance + 1, context, level)
                stream.write(f'\n')
                stream.write(' ' * indent)
                stream.write(f']\n')

            indent -= self._indent_per_level
            stream.write(' ' * indent)
            stream.write(')')
        elif hasattr(obj, 'format'):
            obj.format(self, stream, indent, allowance, context, level)
        else:
            super()._format(obj, stream, indent, allowance, context, level)


def pformat(data, indent=4, indent_first=None):
    INDENTATION = ' '
    if indent_first is None:
        indent_first = indent
    pp = PrettyPrinter(indent=4, compact=True, sort_dicts=False, width=120)
    res = textwrap.indent(pp.pformat(data), INDENTATION * indent)
    if indent_first != indent:
        res = res.lstrip() + INDENTATION * indent_first
    return res
