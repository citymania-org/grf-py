import grf


class Callback:
    class Vehicle:
        VISUAL_EFFECT_AND_POWERED = 0x10
        WAGON_LENGTH = 0x11
        LOAD_AMOUNT = 0x12
        REFIT_CAPACITY = 0x15
        ARTICULATED_PART = 0x16
        CARGO_SUBTYPE = 0x19
        CAN_ATTACH_WAGON = 0x1d
        PURCHASE_TEXT = 0x23
        COLOUR_MAPPING = 0x2d
        START_STOP_CHECK = 0x31
        DAY_32 = 0x32
        SOUND_EFFECT = 0x33  # TODO some other also use it
        AUTOREPLACE_SELECTION = 0x34
        CHANGE_PROPERTIES = 0x36
        NAME = 0x161
        REVERSE_BUILD_PROBABILITY = 0x162

    class Train(Vehicle):
        class Properties:
            LOADING_SPEED = 0x07
            MAX_SPEED = 0x09
            POWER = 0x0B
            RUNNING_COST_FACTOR=0x0D
            CARGO_CAPACITY = 0x14
            WEIGHT = 0x16
            COST_FACTOR = 0x17
            TRACTIVE_EFFORT_COEFFICIENT = 0x1F
            SHORTEN_BY = 0x21
            BITMASK_VEHICLE_INFO = 0x25
            CARGO_AGE_PERIOD = 0x2B
            CURVE_SPEED_MOD = 0x2E

    class RoadVehicle(Vehicle):
        class Properties:
            LOADING_SPEED = 0x07
            RUNNING_COST_FACTOR = 0x09
            CARGO_CAPACITY = 0x0F
            COST_FACTOR = 0x11
            POWER = 0x13
            WEIGHT = 0x14
            MAX_SPEED = 0x15
            TRACTIVE_EFFORT_COEFFICIENT = 0x18
            CARGO_AGE_PERIOD = 0x22
            SHORTEN_BY = 0x23

    class Ship(Vehicle):
        class Properties:
            LOADING_SPEED = 0x07
            COST_FACTOR = 0x0A
            MAX_SPEED = 0x0B
            CARGO_CAPACITY = 0x0D
            RUNNING_COST_FACTOR = 0x0F
            CARGO_AGE_PERIOD = 0x1D

    class Aircraft(Vehicle):
        class Properties:
            LOADING_SPEED = 0x07
            COST_FACTOR = 0x0B
            MAX_SPEED = 0x0C
            RUNNING_COST_FACTOR = 0x0E
            PRIMARY_CARGO_CAPACITY = 0x0F
            SECONDARY_CARGO_CAPACITY = 0x11
            CARGO_AGE_PERIOD = 0x1C
            RANGE = 0x1F

    class Station:
        AVAILABILITY = 0x13
        SPRITE_LAYOUT = 0x14
        CUSTOM_LAYOUT = 0x24
        ANIMATION_CONTROL = 0x140
        NEXT_FRAME = 0x141
        FRAME_LENGTH = 0x142

    class House:
        CONSTRUCTION_CHECK = 0x17
        NEXT_FRAME = 0x1a
        ANIMATION_CONTROL = 0x1b
        CONSTRUCTION_CHANGE = 0x1c
        COLOUR = 0x1e
        CARGO_ACCEPTANCE = 0x1f
        FRAME_LENGTH = 0x20
        BUILDING_DESTRUCTION = 0x21
        ACCEPTED_CARGO = 0x2a
        PROTECT = 0x143
        WATCHED_CARGO_ACCEPTED = 0x148
        NAME = 0x14d
        DEFAULT_FOUNDATIONS = 0x14e
        AUTOSLOPING = 0x14f
        REFIT_COST = 0x15e
        ADVANCED_EFFECT = 0x160

    class IndustryTile:
        ANIMATION_CONTROL = 0x25
        NEXT_FRAME = 0x26
        FRAME_LENGTH = 0x27
        CARGO_ACCEPTANCE = 0x2b
        ACCEPTED_CARGO = 0x2c
        SHAPE_CHECK = 0x2f
        DEFAULT_FOUNDATIONS = 0x30
        AUTOSLOPING = 0x3c

    class Industry:
        AVAILABILITY = 0x22
        CHECK_LOCATION = 0x28
        RANDOM_PRODUCTION = 0x29
        MONTHLY_PRODUCTION = 0x35
        CARGO_SUBTYPE_DISPLAY = 0x37
        ADDITIONAL_FUND_TEXT = 0x38
        ADDITIONAL_TEXT = 0x3a
        SPECIAL_EFFECTS = 0x3b
        OPT_OUT_OF_ACCEPTING = 0x3d
        COLOUR = 0x14a
        INPUT_CARGO = 0x14b
        OUTPUT_CARGO = 0x14c
        INITIAL_PRODUCTION = 0x15f

    class Cargo:
        PROFIT_CALCULATION = 0x39
        STATION_RATING = 0x145

    class Sound:
        AMBIENT = 0x144

    class Signal:
        SPRITE = 0x146

    class Canal:
        SPRITE_OFFSET = 0x147

    class AirportTile:
        DEFAULT_FOUNDATIONS = 0x150
        ANIMATION_CONTROL = 0x152
        NEXT_FRAME = 0x153
        FRAME_LENGTH = 0x154

    class Airport:
        EXTRA_INFO = 0x155
        LAYOUT_NAME = 0x156

    class Object:
        SLOPE_CHECK = 0x157
        NEXT_FRAME = 0x158
        ANIMATION_CONTROL = 0x159
        FRAME_LENGTH = 0x15a
        COLOUR = 0x15b
        ADDITIONAL_TEXT = 0x15c
        AUTOSLOPING = 0x15d


class DefaultCallback(Callback):
    __slots__ = ('default',)

    def __init__(self, default):
        self.default = default

    def is_set(self):
        return self.default is not None

    def __str__(self):
        return f'DefaultCallback({self.default})'


class PurchaseCallback(Callback):
    __slots__ = ('purchase',)

    def __init__(self, purchase):
        self.purchase = purchase

    def is_set(self):
        return self.purchase is not None

    def __str__(self):
        return f'PurchaseCallback({self.purchase})'


class DualCallback(Callback):
    __slots__ = ('default', 'purchase')

    def __init__(self, default, purchase=None):
        self.default = default
        self.purchase = default if purchase is None else purchase

    def is_set(self):
        return self.default is not None or self.purchase is not None

    def __str__(self):
        return f'DualCallback({self.default}, {self.purchase})'


class GraphicsCallback(Callback):
    __slots__ = ('default', 'purchase')

    def __init__(self, default, purchase=None):
        self.default = default
        self.purchase = purchase

    def is_set(self):
        return self.default is not None or self.purchase is not None

    def __str__(self):
        return f'GraphicsCallback({self.default}, {self.purchase})'


#props
# 'loading_speed'           : {'type': 'cb', 'num': 0x36, 'var10': 0x07},
# 0x36: {'name': 'change_properties', 'flag': 0, 'class': DualCallback},

PROPERTIES = {
    grf.TRAIN: {
        0x07: {'name': 'loading_speed', 'class': DefaultCallback},
        0x09: {'name': 'max_speed', 'class': DualCallback},
        0x0b: {'name': 'power', 'class': DualCallback},
        0x0d: {'name': 'running_cost_factor', 'class': DualCallback},
        0x14: {'name': 'cargo_capacity', 'class': DualCallback},
        0x16: {'name': 'weight', 'class': DualCallback},
        0x17: {'name': 'cost_factor', 'class': DualCallback},
        0x1f: {'name': 'tractive_effort_coefficient', 'class': DualCallback},
        0x21: {'name': 'shorten_by', 'class': DefaultCallback},
        0x25: {'name': 'bitmask_vehicle_info', 'class': DefaultCallback},
        0x2b: {'name': 'cargo_age_period', 'class': DefaultCallback},
        0x2e: {'name': 'curve_speed_mod', 'class': DefaultCallback},
    },
    grf.RV: {
        0x07: {'name': 'loading_speed', 'class': DefaultCallback},
        0x09: {'name': 'running_cost_factor', 'class': DualCallback},
        0x0f: {'name': 'cargo_capacity', 'class': DualCallback},
        0x11: {'name': 'cost_factor', 'class': DualCallback},
        0x13: {'name': 'power', 'class': DualCallback},
        0x14: {'name': 'weight', 'class': DualCallback},
        0x15: {'name': 'max_speed', 'class': DualCallback},
        0x18: {'name': 'tractive_effort_coefficient', 'class': DualCallback},
        0x22: {'name': 'cargo_age_period', 'class': DefaultCallback},
        0x23: {'name': 'shorten_by', 'class': DefaultCallback},
    },
    grf.SHIP: {
        0x07: {'name': 'loading_speed', 'class': DefaultCallback},
        0x0a: {'name': 'cost_factor', 'class': DualCallback},
        0x0b: {'name': 'max_speed', 'class': DualCallback},
        0x0d: {'name': 'cargo_capacity', 'class': DualCallback},
        0x0f: {'name': 'running_cost_factor', 'class': DualCallback},
        0x1d: {'name': 'cargo_age_period', 'class': DefaultCallback},
    },
    grf.AIRCRAFT: {
        0x07: {'name': 'loading_speed', 'class': DefaultCallback},
        0x0a: {'name': 'cost_factor', 'class': DualCallback},
        0x0b: {'name': 'max_speed', 'class': DualCallback},
        0x0f: {'name': 'running_cost_factor', 'class': DualCallback},
        0x0d: {'name': 'primary_cargo_capacity', 'class': DualCallback},
        0x0d: {'name': 'secondary_cargo_capacity', 'class': DualCallback},
        0x1d: {'name': 'cargo_age_period', 'class': DefaultCallback},
        0x1d: {'name': 'range', 'class': DualCallback},
    },
}

VEHICLE_CALLBACKS = {
    0x1: {'name': 'random_trigger', 'flag': 0, 'class': DefaultCallback},
    # ver 7 0x12: {'name': 'loading_speed', 'flag': 0x4, 'class': DefaultCallback},
    0x15: {'name': 'cargo_capacity', 'flag': 0x8, 'class': DualCallback},
    0x19: {'name': 'cargo_subtype_text', 'flag': 0x20, 'class': DefaultCallback},
    0x23: {'name': 'additional_text', 'flag': 0, 'class': PurchaseCallback},
    0x2d: {'name': 'colour_mapping', 'flag': 0x40, 'class': DualCallback},
    0x31: {'name': 'start_stop', 'flag': 0, 'class': DefaultCallback},
    0x32: {'name': 'every_32_days', 'flag': 0, 'class': DefaultCallback},
    0x33: {'name': 'sound_effect', 'flag': 0x80, 'class': DefaultCallback},
    0x15e: {'name': 'refit_cost', 'flag': 0, 'class': DualCallback},
    0x161: {'name': 'name', 'flag': 0x100, 'class': PurchaseCallback},
    0x162: {'name': 'reverse_build_probability', 'flag': 0, 'class': DefaultCallback},
    # 0x34: {'name': 'autoreplace_selection', 'flag': 0, 'class': DefaultCallback},
    0x36: {'name': 'change_properties', 'flag': 0, 'class': DualCallback},
}

CALLBACKS = {
    grf.TRAIN: {
        **VEHICLE_CALLBACKS,
        # 0x07: {'name': 'loading_speed', 'flag': 0, 'class': DualCallback},
        # 0x09: {'name': 'max_speed', 'flag': 0, 'class': DualCallback},
        0x10: {'name': 'visual_effect_and_powered', 'flag': 0x1, 'class': DefaultCallback},
        0x16: {'name': 'articulated_part', 'flag': 0x10, 'class': DualCallback},
        0x1d: {'name': 'can_attach_wagon', 'flag': 0, 'class': DefaultCallback},
        0x160: {'name': 'create_effect', 'flag': 0, 'class': DefaultCallback},
        # 0x11: {'name': 'wagon_length', 'flag': 0x2, 'class': DefaultCallback},
        # 0x0B: {'name': 'power', 'flag': 0, 'class': DualCallback},
        # 0x0D: {'name': 'running_cost_factor', 'flag': 0, 'class': DualCallback},
        # 0x14: {'name': 'cargo_capacity', 'flag': 0, 'class': DualCallback},
        # 0x16: {'name': 'weight', 'flag': 0, 'class': DualCallback},
        # 0x17: {'name': 'cost_factor', 'flag': 0, 'class': DualCallback},
        # 0x1F: {'name': 'tractive_effort_coefficient', 'flag': 0, 'class': DualCallback},
        # 0x21: {'name': 'shorten_by', 'flag': 0, 'class': DualCallback},
        # 0x25: {'name': 'bitmask_vehicle_info', 'flag': 0, 'class': DualCallback},
        # 0x2B: {'name': 'cargo_age_period', 'flag': 0, 'class': DualCallback},
        # 0x2E: {'name': 'curve_speed_mod', 'flag': 0, 'class': DualCallback},
    },
    grf.RV: {
        **VEHICLE_CALLBACKS,
        0x10: {'name': 'visual_effect', 'flag': 0, 'class': DefaultCallback},
        0x16: {'name': 'articulated_part', 'flag': 0x10, 'class': DualCallback},
        0x160: {'name': 'create_effect', 'flag': 0, 'class': DefaultCallback},
    },
    grf.SHIP: {
        **VEHICLE_CALLBACKS,
        0x10: {'name': 'visual_effect', 'flag': 0, 'class': DefaultCallback},
    },
    grf.AIRCRAFT: {
        **VEHICLE_CALLBACKS,
    },
    grf.STATION: {
        0x13: {'name': 'availability', 'flag': 0x1, 'class': PurchaseCallback},
        0x14: {'name': 'select_sprite_layout', 'flag': 0x2, 'class': DualCallback},
        0x24: {'name': 'select_tile_layout', 'flag': 0, 'class': PurchaseCallback},
        0x140: {'name': 'anim_control', 'flag': 0, 'class': DefaultCallback},
        0x141: {'name': 'anim_next_frame', 'flag': 0x4, 'class': DefaultCallback},
        0x142: {'name': 'anim_speed', 'flag': 0x8, 'class': DefaultCallback},
        0x149: {'name': 'tile_check', 'flag': 0x10, 'class': PurchaseCallback},
    },
    grf.CANAL: {
        0x147: {'name': 'sprite_offset', 'flag': 0, 'class': DefaultCallback},
    },
    grf.HOUSE: {
        0x1: {'name': 'random_trigger', 'flag': 0, 'class': DefaultCallback},
        0x17: {'name': 'construction_check', 'flag': 0x1, 'class': DefaultCallback},
        0x1a: {'name': 'anim_next_frame', 'flag': 0x2, 'class': DefaultCallback},
        0x1b: {'name': 'anim_control', 'flag': 0x4, 'class': DefaultCallback},
        0x1c: {'name': 'construction_anim', 'flag': 0x8, 'class': DefaultCallback},
        0x1e: {'name': 'colour', 'flag': 0x10, 'class': DefaultCallback},
        0x1f: {'name': 'cargo_amount_accept', 'flag': 0x20, 'class': DefaultCallback},
        0x20: {'name': 'anim_speed', 'flag': 0x40, 'class': DefaultCallback},
        0x21: {'name': 'destruction', 'flag': 0x80, 'class': DefaultCallback},
        0x2a: {'name': 'cargo_type_accept', 'flag': 0x100, 'class': DefaultCallback},
        0x2e: {'name': 'cargo_production', 'flag': 0x200, 'class': DefaultCallback},
        0x143: {'name': 'protection', 'flag': 0x400, 'class': DefaultCallback},
        0x148: {'name': 'watched_cargo_accepted', 'flag': 0, 'class': DefaultCallback},
        0x14d: {'name': 'name', 'flag': 0, 'class': DefaultCallback},
        0x14e: {'name': 'foundations', 'flag': 0x800, 'class': DefaultCallback},
        0x14f: {'name': 'autoslope', 'flag': 0x1000, 'class': DefaultCallback},
    },
    grf.INDUSTRY_TILE: {
        0x1: {'name': 'random_trigger', 'flag': 0, 'class': DefaultCallback},
        0x25: {'name': 'anim_control', 'flag': 0, 'class': DefaultCallback},
        0x26: {'name': 'anim_next_frame', 'flag': 0x1, 'class': DefaultCallback},
        0x27: {'name': 'anim_speed', 'flag': 0x2, 'class': DefaultCallback},
        0x2b: {'name': 'cargo_amount_accept', 'flag': 0x4, 'class': DefaultCallback},
        0x2c: {'name': 'cargo_type_accept', 'flag': 0x8, 'class': DefaultCallback},
        0x2f: {'name': 'tile_check', 'flag': 0x10, 'class': DefaultCallback},
        0x30: {'name': 'foundations', 'flag': 0x20, 'class': DefaultCallback},
        0x3c: {'name': 'autoslope', 'flag': 0x40, 'class': DefaultCallback},
    },
    grf.INDUSTRY: {
        0x22: {'name': 'construction_probability', 'flag': 0x1, 'class': DefaultCallback},
        0x28: {'name': 'location_check', 'flag': 0x8, 'class': DefaultCallback},
        0x29: {'name': 'random_prod_change', 'flag': 0x10, 'class': DefaultCallback},
        0x35: {'name': 'monthly_prod_change', 'flag': 0x20, 'class': DefaultCallback},
        0x37: {'name': 'cargo_subtype_display', 'flag': 0x40, 'class': DefaultCallback},
        0x38: {'name': 'extra_text_fund', 'flag': 0x80, 'class': DefaultCallback},
        0x3A: {'name': 'extra_text_industry', 'flag': 0x100, 'class': DefaultCallback},
        0x3B: {'name': 'control_special', 'flag': 0x200, 'class': DefaultCallback},
        0x3D: {'name': 'stop_accept_cargo', 'flag': 0x400, 'class': DefaultCallback},
        0x14A: {'name': 'colour', 'flag': 0x800, 'class': DefaultCallback},
        0x14B: {'name': 'input_cargo', 'flag': 0x1000, 'class': DefaultCallback},
        0x14C: {'name': 'output_cargo', 'flag': 0x2000, 'class': DefaultCallback},
        0x15F: {'name': 'init_production', 'flag': 0x4000, 'class': DefaultCallback},
    },
    grf.CARGO: {
        0x39: {'name': 'profit', 'flag': 0x1, 'class': DefaultCallback},
        0x145: {'name': 'station_rating', 'flag': 0x145, 'class': DefaultCallback},
    },
    grf.AIRPORT: {
        0x155: {'name': 'additional_text', 'flag': 0, 'class': DefaultCallback},
        0x156: {'name': 'layout_name', 'flag': 0, 'class': DefaultCallback},
    },
    grf.OBJECT: {
        0x157: {'name': 'tile_check', 'flag': 0x1, 'class': PurchaseCallback},
        0x158: {'name': 'anim_next_frame', 'flag': 0x2, 'class': DefaultCallback},
        0x159: {'name': 'anim_control', 'flag': 0, 'class': DefaultCallback},
        0x15a: {'name': 'anim_speed', 'flag': 0x4, 'class': DefaultCallback},
        0x15b: {'name': 'colour', 'flag': 0x8, 'class': DefaultCallback},
        0x15c: {'name': 'additional_text', 'flag': 0x10, 'class': PurchaseCallback},
        0x15d: {'name': 'autoslope', 'flag': 0x20, 'class': DefaultCallback},
    },
    grf.RAILTYPE: {
        0x00: {'name': 'gui', 'flag': 0, 'class': DefaultCallback},
        0x01: {'name': 'track_overlay', 'flag': 0, 'class': DefaultCallback},
        0x02: {'name': 'underlay', 'flag': 0, 'class': DefaultCallback},
        0x03: {'name': 'tunnels', 'flag': 0, 'class': DefaultCallback},
        0x04: {'name': 'catenary_wire', 'flag': 0, 'class': DefaultCallback},
        0x05: {'name': 'catenary_pylons', 'flag': 0, 'class': DefaultCallback},
        0x06: {'name': 'bridge_surfaces', 'flag': 0, 'class': DefaultCallback},
        0x07: {'name': 'level_crossings', 'flag': 0, 'class': DefaultCallback},
        0x08: {'name': 'depots', 'flag': 0, 'class': DefaultCallback},
        0x09: {'name': 'fences', 'flag': 0, 'class': DefaultCallback},
        0x0A: {'name': 'tunnel_overlay', 'flag': 0, 'class': DefaultCallback},
        0x0B: {'name': 'signals', 'flag': 0, 'class': DefaultCallback},
        0x0C: {'name': 'precombined', 'flag': 0, 'class': DefaultCallback},
    },
    grf.AIRPORT_TILE: {
        0x150: {'name': 'foundations', 'flag': 0x20, 'class': DefaultCallback},
        0x152: {'name': 'anim_control', 'flag': 0, 'class': DefaultCallback},
        0x153: {'name': 'anim_next_frame', 'flag': 0x1, 'class': DefaultCallback},
        0x154: {'name': 'anim_speed', 'flag': 0x2, 'class': DefaultCallback},
    },
    grf.ROADTYPE: {
        0x00: {'name': 'gui', 'flag': 0, 'class': DefaultCallback},
        0x01: {'name': 'track_overlay', 'flag': 0, 'class': DefaultCallback},
        0x02: {'name': 'underlay', 'flag': 0, 'class': DefaultCallback},
        0x03: {'name': 'tunnels', 'flag': 0, 'class': DefaultCallback},
        0x04: {'name': 'catenary_front', 'flag': 0, 'class': DefaultCallback},
        0x05: {'name': 'catenary_back', 'flag': 0, 'class': DefaultCallback},
        0x06: {'name': 'bridge_surfaces', 'flag': 0, 'class': DefaultCallback},
        0x08: {'name': 'depots', 'flag': 0, 'class': DefaultCallback},
        0x0A: {'name': 'roadstops', 'flag': 0, 'class': DefaultCallback},
        0x0B: {'name': 'direction_markings', 'flag': 0, 'class': DefaultCallback},
    },
    grf.TRAMTYPE: {
        0x00: {'name': 'gui', 'flag': 0, 'class': DefaultCallback},
        0x01: {'name': 'track_overlay', 'flag': 0, 'class': DefaultCallback},
        0x02: {'name': 'underlay', 'flag': 0, 'class': DefaultCallback},
        0x03: {'name': 'tunnels', 'flag': 0, 'class': DefaultCallback},
        0x04: {'name': 'catenary_front', 'flag': 0, 'class': DefaultCallback},
        0x05: {'name': 'catenary_back', 'flag': 0, 'class': DefaultCallback},
        0x06: {'name': 'bridge_surfaces', 'flag': 0, 'class': DefaultCallback},
        0x08: {'name': 'depots', 'flag': 0, 'class': DefaultCallback},
    },
    grf.ROAD_STOP: {
        0x13: {'name': 'availability', 'flag': 0x1, 'class': DefaultCallback},
        0x140: {'name': 'anim_control', 'flag': 0, 'class': DefaultCallback},
        0x141: {'name': 'anim_next_frame', 'flag': 0x2, 'class': DefaultCallback},
        0x142: {'name': 'anim_speed', 'flag': 0x3, 'class': DefaultCallback},
    },
}

FEATURE_CALLBACKS = {
    grf.TRAIN: Callback.Train,
    grf.RV: Callback.RoadVehicle,
    grf.SHIP: Callback.Ship,
    # TODO aircraft
    grf.STATION: Callback.Station,
    grf.HOUSE: Callback.House,
    # TODO GLOBAL_VAR
    grf.INDUSTRY_TILE: Callback.IndustryTile,
    grf.INDUSTRY: Callback.Industry,
    grf.CARGO: Callback.Cargo,
    grf.SOUND_EFFECT: Callback.Sound,
    grf.AIRPORT: Callback.Airport,
    grf.SIGNAL: Callback.Signal,
    grf.OBJECT: Callback.Canal,
    grf.RAILTYPE: Callback.AirportTile,
    # TODO airport_tille
    # grf.ROADTYPE:
    # grf.TRAMTYPE:
    # grf.ROAD_STOP:
}

class CallbackDescriptor:

    _IDX = {}

    @classmethod
    def create(cls, class_, cbid):
        key = (class_, cbid)
        value = cls._IDX.get(class_)
        if value is None:
            value = cls._IDX[key] = cls(class_, cbid)
        return value

    def __init__(self, class_, cbid):
        assert issubclass(class_, Callback)
        self._class = class_
        self._cbid = cbid

    def __set__(self, obj, value):
        if isinstance(value, Callback):
            if value.__class__ is not self._class:
                raise ValueError(f'Expected {self._class.__name__} found {value.__class__.__name__}')
        else:
            value = self._class(value)
        obj._callbacks[self._cbid] = value

    def __get__(self, obj, objtype=None):
        res = obj._callbacks.get(self._cbid)
        if res is None:
            res = self._class(None)
            obj._callbacks[self._cbid] = res
        return res


class CallbackManager(object):
    __slots__ = ('_callbacks')

    graphics = CallbackDescriptor.create(GraphicsCallback, 0)

    class Properties:
        __slots__ = ('_callbacks',)

        def __init__(self, callbacks=None):
            self._callbacks = {}
            for k, v in (callbacks or {}).items():
                setattr(self, k, v)

        def is_set(self):
            return any(v.is_set() for v in self._callbacks.values())

        def make_switches(self, default):
            default_ranges = {}
            purchase_ranges = {}
            same_switch = True
            for k, v in self._callbacks.items():
                if not v.is_set():
                    continue
                if isinstance(v, DualCallback):
                    if v.default is not None:
                        default_ranges[k] = v.default
                    if v.purchase is not None:
                        purchase_ranges[k] = v.purchase
                    if v.default is None or v.purchase is None or grf.get_ref_id(v.default) != grf.get_ref_id(v.purchase):
                        same_switch = False
                elif isinstance(v, PurchaseCallback):
                    purchase_ranges[k] = v.purchase
                elif isinstance(v, DefaultCallback):
                    default_ranges[k] = v.default
                else:
                    assert False, type(v)

            if same_switch:
                purchase_switch = default_switch = grf.Switch(
                    code='extra_callback_info1_byte',
                    ranges={
                        **default_ranges,
                        **purchase_ranges,
                    },
                    default=default,
                )
            else:
                default_switch = purchase_switch = None
                if default_ranges:
                    default_switch = grf.Switch(
                        code='extra_callback_info1_byte',
                        ranges=default_ranges,
                        default=default,
                    )
                if purchase_ranges:
                    purchase_switch = grf.Switch(
                        code='extra_callback_info1_byte',
                        ranges=purchase_ranges,
                        default=default,
                    )
            return default_switch, purchase_switch

    def __init__(self, callbacks=None):
        callbacks = (callbacks or {}).copy()
        self._callbacks = {}
        self.graphics = callbacks.pop('graphics', None)

        if self.Properties:
            props = callbacks.pop('properties', None)
            self.properties = self.Properties(props)

        for k, v in callbacks.items():
            setattr(self, k, v)

    # def __setattr__(self, name, value):
    #     if name.startswith('_') or name in self._allowed_attributes:
    #         return super().__setattr__(name, value)
    #     raise AttributeError(f'No callback {name} for feature {self.feature}')

    def set_flag_props(self, props):
        res = 0
        for cbid, cb in self._callbacks.items():
            if cbid != 0 and cb.is_set():
                res |= CALLBACKS[self.feature][cbid]['flag']
        if res & 0xFF > 0:
            props['cb_flags'] = props.get('cb_flags', 0) | (res & 0xFF)
        if res & 0xFF00 > 0:
            props['extra_cb_flags'] = props.get('extra_cb_flags', 0) | (res >> 8)

    def make_switch(self):
        has_graphics = self.feature in grf.VEHICLE_FEATURES or self.feature in {grf.STATION, grf.HOUSE, grf.INDUSTRY_TILE}
        has_purchase = self.feature in grf.VEHICLE_FEATURES or self.feature in {grf.STATION, grf.OBJECT, grf.ROAD_STOP}
        if self.graphics.default is None and has_graphics:
            raise ValueError('No graphics')

        default_callbacks = {}
        purchase_callbacks = {}
        if hasattr(self, 'properties') and self.properties.is_set():
            if self.change_properties.is_set():
                raise RuntimeError('Can''t use change_properties callback together with individual property callbacks.')
            default, purchase = self.properties.make_switches(self.graphics.default)
            if default is not None:
                default_callbacks[Callback.Vehicle.CHANGE_PROPERTIES] = default
            if purchase is not None:
                purchase_callbacks[Callback.Vehicle.CHANGE_PROPERTIES] = purchase

        for k, c in self._callbacks.items():
            if k == 0:
                continue
            default_cb = getattr(c, 'default', None)
            if default_cb is not None:
                default_callbacks[k] = default_cb
            purchase_cb = getattr(c, 'purchase', None)
            if purchase_cb is not None:
                purchase_callbacks[k] = purchase_cb


        maps = {}

        if has_graphics:
            default = self.graphics.default
        else:
            default = None

        if has_purchase:
            if purchase_callbacks:
                # TODO purchase_switch can be the same as default
                purchase_switch = grf.Switch(
                    ranges=purchase_callbacks,
                    default=self.graphics.default if self.graphics.purchase is None else self.graphics.purchase,
                    code='current_callback',
                )
                maps = {255: purchase_switch}
            elif has_graphics and self.graphics.purchase is not None:
                maps = {255: self.graphics.purchase}

        if default_callbacks:
            default = grf.Switch(
                ranges={**default_callbacks},
                default=self.graphics.default or 0,  # TODO should it return 0?
                code='current_callback',
            )
        return default, maps

    def make_map_action(self, definition):
        assert isinstance(definition, grf.DefineMultiple)
        default, maps = self.make_switch()
        if default is None:
            return []
        return [grf.Map(
            definition,
            maps=maps,
            default=default,
        )]


def _make_property_class(feature):
    assert feature in grf.VEHICLE_FEATURES
    props = {}

    for prop_id, data in PROPERTIES[feature].items():
        props[data['name']] = CallbackDescriptor.create(data['class'], prop_id)

    return type(f'{feature.class_name}Properties', (CallbackManager.Properties, ), props)


def _make_callback_class(feature):
    props = {
        'feature': feature,
    }

    slots = []
    if feature in grf.VEHICLE_FEATURES:
        props['Properties'] = _make_property_class(feature)
        slots.append('properties')
    else:
        props['Properties'] = None

    for cb_id, data in CALLBACKS[feature].items():
        name = data['name']
        props[name] = CallbackDescriptor.create(data['class'], cb_id)

    props['__slots__'] = slots

    return type(f'{feature.class_name}CallbackManager', (CallbackManager, ), props)


def make_callback_manager(feature, callbacks=None):
    idx = make_callback_manager.CLASSES
    cls = idx.get(feature)
    if cls is None:
        cls = _make_callback_class(feature)
        idx[feature] = cls
    return cls(callbacks=callbacks)


make_callback_manager.CLASSES = {}
