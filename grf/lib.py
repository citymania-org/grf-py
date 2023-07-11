from collections import defaultdict
from collections.abc import Iterable

import grf

def fake_vehicle_info(props):
    return '{}'.join('{BLACK}' + k + ': {GOLD}' + v for k, v in props.items())

def make_cb_switches(callbacks, maps, layout):
    # TODO combine similar switches?
    out_maps = {}
    for k, cblist in maps.items():
        if not cblist: continue
        out_maps[k] = grf.Switch(
            ranges={0: layout, **cblist},
            default=layout,
            code='current_callback',
        )
    default = layout
    if callbacks:
        default = grf.Switch(
            ranges={0: layout, **callbacks},
            default=layout,
            code='current_callback',
        )
    return default, out_maps


def combine_ranges(value):
    last_id = None
    res = []
    for i, value in sorted(value):
        if last_id != i - 1:
            res.append((i, [value]))
        else:
            res[-1][1].append(value)
        last_id = i
    return res


class SoundEvent:
    START = 1 # Vehicle leaves station or depot, plane takes off
    TUNNEL = 2 #Vehicle enters tunnel
    BREAKDOWN = 3 # Vehicle breaks down (not for planes)
    RUNNING = 4 #   Once per engine tick, but no more than once per vehicle motion
    TOUCHDOWN = 5 # Aircraft touches down
    VISUAL_EFFECT = 6 # Visual effect is generated (steam plume, diesel smoke, electric spark)
    RUNNING_16 = 7 #Every 16 engine ticks if in motion
    STOPPED = 8 #   Every 16 engine ticks if stopped
    LOAD_UNLOAD = 9 #   Consist loads or unloads cargo
    BRIDGE = 10


class DefaultSound:
    CONSTRUCTION_WATER = 0x00
    FACTORY = 0x01
    DEPARTURE_STEAM = 0x02
    TRAIN_THROUGH_TUNNEL = 0x03
    DEPARTURE_CARGO_SHIP = 0x04
    DEPARTURE_FERRY = 0x05
    TAKEOFF_PROPELLER = 0x06
    TAKEOFF_JET = 0x07
    DEPARTURE_TRAIN = 0x08
    MINE = 0x09
    POWER_STATION = 0x0A
    STEAM = 0x0B
    LEVEL_CROSSING = 0x0C
    BREAKDOWN_ROADVEHICLE = 0x0D
    BREAKDOWN_TRAIN_SHIP = 0x0E
    CRASH = 0x0F
    EXPLOSION = 0x10
    TRAIN_COLLISION = 0x11
    CASHTILL = 0x12
    BEEP = 0x13
    NEWS_TICKER = 0x14
    SKID_PLANE = 0x15
    TAKEOFF_HELICOPTER = 0x16
    DEPARTURE_OLD_RV_1 = 0x17
    DEPARTURE_OLD_RV_2 = 0x18
    DEPARTURE_MODERN_BUS = 0x19
    DEPARTURE_OLD_BUS = 0x1A
    APPLAUSE = 0x1B
    NEW_ENGINE = 0x1C
    CONSTRUCTION_OTHER = 0x1D
    CONSTRUCTION_RAIL = 0x1E
    ROAD_WORKS = 0x1F
    CAR_HORN = 0x20
    CAR_HORN_2 = 0x21
    FARM_1 = 0x22
    FARM_2 = 0x23
    FARM_3 = 0x24
    CONSTRUCTION_BRIDGE = 0x25
    SAWMILL = 0x26
    GOOD_YEAR = 0x27
    BAD_YEAR = 0x28
    SUGAR_MINE_2 = 0x29
    TOY_FACTORY_3 = 0x2A
    TOY_FACTORY_2 = 0x2B
    TOY_FACTORY_1 = 0x2C
    SUGAR_MINE_1 = 0x2D
    BUBBLE_GENERATOR = 0x2E
    BUBBLE_GENERATOR_FAIL = 0x2F
    TOFFEE_QUARRY = 0x30
    BUBBLE_GENERATOR_SUCCESS = 0x31
    POP_2 = 0x32
    PLASTIC_MINE = 0x33
    ARCTIC_SNOW_1 = 0x34
    BREAKDOWN_ROADVEHICLE_TOYLAND = 0x35
    LUMBER_MILL_3 = 0x36
    LUMBER_MILL_2 = 0x37
    LUMBER_MILL_1 = 0x38
    ARCTIC_SNOW_2 = 0x39
    BREAKDOWN_TRAIN_SHIP_TOYLAND = 0x3A
    TAKEOFF_JET_FAST = 0x3B
    DEPARTURE_BUS_TOYLAND_1 = 0x3C
    TAKEOFF_JET_BIG = 0x3D
    DEPARTURE_BUS_TOYLAND_2 = 0x3E
    DEPARTURE_TRUCK_TOYLAND_1 = 0x3F
    DEPARTURE_TRUCK_TOYLAND_2 = 0x40
    DEPARTURE_MAGLEV = 0x41
    RAINFOREST_1 = 0x42
    RAINFOREST_2 = 0x43
    RAINFOREST_3 = 0x44
    TAKEOFF_PROPELLER_TOYLAND_1 = 0x45
    TAKEOFF_PROPELLER_TOYLAND_2 = 0x46
    DEPARTURE_MONORAIL = 0x47
    RAINFOREST_4 = 0x48


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

class CallbackManager:

    class Properties:
        def __init__(self, domain, callbacks=None):
            self._domain = domain
            self._callbacks = {}
            for k, v in (callbacks or {}).items():
                if v is not None:
                    setattr(self, k, v)

        def is_set(self):
            return any(v is not None for v in self._callbacks.values())

        def get_ranges(self):
            return {k: v for k, v in self._callbacks.items() if v is not None}

        def __setattr__(self, name, value):
            if name.startswith('_'):
                return super().__setattr__(name, value)

            if name.lower() != name:
                raise AttributeError(name)

            prop_id = getattr(self._domain, name.upper())
            if prop_id in self._callbacks:
                raise RuntimeError(f'Callback for property {name} is already defined')

            self._callbacks[prop_id] = value

    def __init__(self, feature, callbacks=None):
        self.feature = feature
        self._domain = FEATURE_CALLBACKS[feature]
        self._callbacks = {}
        self.graphics = None
        self.purchase_graphics = None
        callbacks = callbacks or {}
        prop_cb = callbacks.pop('properties', None)
        for k, v in callbacks.items():
            if v is not None:
                setattr(self, k, v)
        prop_domain = getattr(self._domain, 'Properties', None)
        if prop_domain is not None:
            self.properties = self.Properties(prop_domain, prop_cb)

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('graphics', 'purchase_graphics', 'properties', 'feature'):
            return super().__setattr__(name, value)
        if name.lower() != name:
            raise AttributeError(name)

        cb_id = getattr(self._domain, name.upper(), None)
        if cb_id is None:
            raise AttributeError(f'No callback {name} for feature {self.feature}')
        if cb_id in self._callbacks:
            raise RuntimeError(f'Callback {name} is already defined')
        self._callbacks[cb_id] = value

    def __delattr__(self, name):
        cb_id = getattr(self._domain, name.upper())
        if cb_id not in self._callbacks:
            raise RuntimeError(f'Callback {name} is not defined')
        del self._callbacks[cb_id]

    def set_flag_props(self, props):
        # TODO checeked only for vehicles
        FLAGS = {
            Callback.Vehicle.VISUAL_EFFECT_AND_POWERED: 0x1,
            Callback.Vehicle.WAGON_LENGTH: 0x2,
            Callback.Vehicle.LOAD_AMOUNT: 0x4,
            Callback.Vehicle.REFIT_CAPACITY: 0x8,
            Callback.Vehicle.ARTICULATED_PART: 0x10,
            Callback.Vehicle.CARGO_SUBTYPE: 0x20,
            Callback.Vehicle.COLOUR_MAPPING: 0x40,
            Callback.Vehicle.SOUND_EFFECT: 0x80,
            Callback.Vehicle.NAME: 0x100,
        }
        res = 0
        for k in self._callbacks.keys():
            res |= FLAGS.get(k, 0)
        if res & 0xFF > 0:
            props['cb_flags'] = props.get('cb_flags', 0) | (res & 0xFF)
        if res & 0xFF00 > 0:
            props['extra_cb_flags'] = props.get('extra_cb_flags', 0) | (res >> 8)

    def make_switch(self):
        if self.graphics is None:
            raise ValueError('No graphics')

        if self.properties.is_set():
            if self._callbacks.get('change_properties') is not None:
                raise RuntimeError('Can''t use change_properties callback together with individual property callbacks.')
            self.change_properties = grf.Switch(
                code='extra_callback_info1_byte',
                ranges=self.properties.get_ranges(),
                default=self.graphics,
            )
        # False - only purchase, True - general + purchase
        PURCHASE = {
            Callback.Vehicle.PURCHASE_TEXT: False,
            Callback.Vehicle.CHANGE_PROPERTIES: True,
            Callback.Vehicle.ARTICULATED_PART: True,
            Callback.Vehicle.CHANGE_PROPERTIES: True,
            Callback.Vehicle.COLOUR_MAPPING: True,
            Callback.Vehicle.NAME: False,
        }
        callbacks = {}
        purchase_callbacks = {}
        for k, c in self._callbacks.items():
            pdata = PURCHASE.get(k)
            if pdata is not False:
                callbacks[k] = c
            if pdata is not None:
                purchase_callbacks[k] = c

        maps = {}
        if purchase_callbacks:
            purchase_switch = grf.Switch(
                ranges=purchase_callbacks,
                default=self.purchase_graphics or self.graphics,
                code='current_callback',
            )
            maps = {255: purchase_switch}
        elif self.purchase_graphics:
            maps = {255: self.purchase_graphics}

        default = self.graphics
        if callbacks:
            default = grf.Switch(
                ranges={**callbacks},
                default=self.graphics,
                code='current_callback',
            )
        return default, maps

    def make_map_action(self, definition):
        assert isinstance(definition, grf.DefineMultiple)
        default, maps = self.make_switch()
        return grf.Map(
            definition,
            maps=maps,
            default=default,
        )


class SpriteTable(grf.SpriteGenerator):
    def __init__(self, feature):
        assert isinstance(feature, grf.Feature)
        self.feature = feature
        self._rows = []
        self._width = None

    def add_row(self, row):
        assert isinstance(row, (list, tuple))
        if self._width is not None and len(row) != self._width:
            raise ValueError(f"Length of row ({len(row)}) doesn't match the width of {self.__class__.__name__} ({self._width})")
        row_id = len(self._rows)
        self._rows.append(tuple(row))
        return row_id

    def get_sprites(self, g):
        if not self._rows:
            raise RuntimeError(f'Trying to get sprites for empty {self.__class__.__name__}')
        res = []
        res.append(grf.Action1(
            feature=self.feature,
            set_count=len(self._rows),
            sprite_count=self._width,
        ))
        for r in self._rows:
            res.extend(r)
        return res


class VehicleSpriteTable(SpriteTable):
    def __init__(self, feature):
        assert feature in grf.VEHICLE_FEATURES
        super().__init__(feature)
        self._width = 8

    def get_layout(self, row_id):
        return grf.GenericSpriteLayout(ent1=(row_id,), ent2=(row_id,))

    def add_purchase_graphics(self, sprite, second_head=None):
        return self.add_row((
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            second_head or grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            grf.EMPTY_SPRITE,
            sprite,
            grf.EMPTY_SPRITE
        ))


class DisableDefault(grf.SpriteGenerator):
    def __init__(self, feature, ids):
        assert feature in grf.VEHICLE_FEATURES, feature
        assert isinstance(ids, (int, Iterable)), type(ids)
        if isinstance(ids, int): ids = [ids,]
        assert all(isinstance(x, int) for x in ids)
        self.feature = feature
        self.ids = ids

    def get_sprites(self, g):

        return (
            grf.DefineMultiple(
                feature=self.feature,
                first_id=first,
                count=len(values),
                props={'climates_available': values}
            )
            for first, values in combine_ranges((x, grf.NO_CLIMATE) for x in self.ids)
        )


class Vehicle(grf.SpriteGenerator):

    class ExtraFlags:
        DISABLE_NEW_VEHICLE_MESSAGE = 0x01
        DISABLE_EXCLUSIVE_PREVIEW = 0x02
        SYNC_VARIANT_EXCLUSIVE_PREVIEW = 0x04
        SYNC_VARIANT_RELIABILITY = 0x08

    def __init__(self, feature, *, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None):
        self._validate_liveries(liveries)

        self.feature = feature
        self.liveries = liveries
        self.callbacks = CallbackManager(self.feature, callbacks)
        self.purchase_sprite = purchase_sprite
        self.additional_text = additional_text

    def _validate_liveries(self, liveries):
        for l in liveries or []:
            if 'name' not in l:
                raise ValueError(f'{self.__class__.__name__} livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'{self.__class__.__name__} livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'{self.__class__.__name__} livery expects 8 sprites, found {len(sprites)}')

    def _gen_name_sprites(self, vehicle_id):
        if isinstance(self.name, grf.StringRef):
            return self.name.get_actions(self.feature, vehicle_id)
        else:
            return [grf.DefineStrings(
                feature=self.feature,
                offset=vehicle_id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8') if isinstance(self.name, str) else self.name]
            )]

    def _gen_purchase_sprites(self):
        res = []

        if self.purchase_sprite:
            # Make sure purchase layouts go before main action 1
            # TODO make Action1 referenceable and move to _set_callbacks
            res.append(grf.SpriteSet(feature=self.feature, count=1))
            res.append(self.purchase_sprite)
            res.append(layout := grf.GenericSpriteLayout(ent1=(0,), ent2=(0,)))
            self.callbacks.purchase_graphics = layout
        return res

    def _set_callbacks(self, g):
        if self.additional_text:
            self.callbacks.purchase_text = g.strings.add(self.additional_text).get_global_id()
        return []


class RoadVehicle(Vehicle):

    class Speed:
        def __init__(self, precise_value):
            self.precise_value = precise_value
            self.value = (self.precise_value + 2) // 4

        def __int__(self):
            return self.value

        def __str__(self):
            return str(self.value)

        def __repr__(self):
            return f'RoadVehicle.Speed({self.precise_value})'

    @staticmethod
    def kmhish(speed):
        return RoadVehicle.Speed(speed * 2)

    @staticmethod
    def mph(speed):
        # TODO doesn't show exact mph in the game
        return RoadVehicle.Speed((speed * 16 + 4) // 5)

    def __init__(self, *, id, name, max_speed, length=None, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.RV, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)
        if length is not None:
            if not 1 <= length <= 8:
                raise ValueError('Invalid length (expected 1..8)')

            if 'shorten_by' in props:
                raise ValueError('Length and shorten_by properties are mutually exclusive')

        self.id = id
        self.name = name

        if isinstance(max_speed, self.Speed):
            self.max_speed = max_speed
        else:
            self.max_speed = self.Speed(max_speed * 4)
        self._props = props

        if length is not None:
            self._props['shorten_by'] = 8 - length

    def _set_callbacks(self, g):
        res = super()._set_callbacks(g)

        if self.max_speed.precise_value >= 0x400:
            # TODO only set speed property
            self.callbacks.change_properties = grf.Switch(
                ranges={
                    0x15: self.max_speed.value,
                },
                default=self.callbacks.graphics,
                code='extra_callback_info1_byte',
            )

        return res

    def get_sprites(self, g):
        res = self._gen_purchase_sprites()

        if self.liveries:
            layouts = []
            for i, l in enumerate(self.liveries):
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            if len(self.liveries) > 1:

                self.callbacks.graphics = grf.Switch(
                    related_scope=True,
                    ranges=dict(enumerate(layouts)),
                    default=layouts[0],
                    code='cargo_subtype',
                )
            else:
                self.callbacks.graphics = layouts[0]

        res.extend(self._set_callbacks(g))

        # Liveries
        if self.liveries and len(self.liveries) > 1:
            self.callbacks.cargo_subtype = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        self.callbacks.set_flag_props(self._props)

        res.extend(self._gen_name_sprites(self.id))

        res.append(definition := grf.Define(
            feature=grf.RV,
            id=self.id,
            props={
                'sprite_id': 0xff,
                'precise_max_speed': min(self.max_speed.precise_value, 0xff),
                'max_speed': min(self.max_speed.value, 0xff),
                **self._props
            }
        ))
        if self.liveries:
            res.append(grf.Action1(
                feature=grf.RV,
                set_count=len(self.liveries),
                sprite_count=8,
            ))

            for l in self.liveries:
                res.extend(l['sprites'])

        res.append(self.callbacks.make_map_action(definition))
        return res


class Train(Vehicle):
    Flags = grf.TrainFlags
    AIFlags = grf.AIFlags
    RunningCost = grf.TrainRunningCost
    VisualEffect = grf.TrainVisualEffect

    class EngineClass:
        STEAM = 0x0
        DIESEL = 0x8
        ELECTRIC = 0x28
        MONORAIL = 0x32
        MAGLEV = 0x32

    @staticmethod
    def kmhish(speed):
        return speed

    @staticmethod
    # TODO doesn't show exact mph in the game
    def mph(speed):
        return (speed * 16 + 4) // 10

    @staticmethod
    def hp(value):
        return int(value + .5)

    @staticmethod
    def ton(value):
        return int(value + .5)

    @staticmethod
    def visual_effect_and_powered(effect, *, position=0, wagon_power=True):
        return effect | position | wagon_power * 0x7f

    def __init__(self, *, id, name, max_speed, weight, length=None, sound_effects=None, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.TRAIN, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)

        REQUIRED_PROPS = ('engine_class', )
        missing_props = [p for p in REQUIRED_PROPS if p not in props]
        if missing_props:
            raise ValueError('Missing required properties for Train: {}'.format(', '.join(missing_props)))

        if length is not None and 'shorten_by' in props:
            raise ValueError('Length and shorten_by properties are mutually exclusive')

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self.sound_effects = sound_effects
        self.weight = weight
        self.length = length
        self._props = props
        self._articulated_parts = []

        mid_shorten, art_shorten, art_liveries = self._calc_length_articulation(
            length, self._props.get('shorten_by'), self.liveries)
        if art_shorten is not None:
            # TODO auto assign articulated part id
            self._head_liveries = art_liveries
            self._props['shorten_by'] = art_shorten
            self._do_add_articulated_part(f'__{self.id}_aa_mid', mid_shorten, self.liveries, self.liveries, {})
            self._do_add_articulated_part(f'__{self.id}_aa_tail', art_shorten, art_liveries, self.liveries, {})
        else:
            self._head_liveries = self.liveries
            if mid_shorten is not None:
                self._props['shorten_by'] = mid_shorten

    def _calc_length_articulation(self, length, shorten_by, liveries):
        if length is None:
            return shorten_by, None, None

        if length > 24:
            raise ValueError("Max length of Train part is 24")

        if length <= 8:
            return 8 - length, None, None

        if self.length % 2 == 1:
            central_length = 7
            articulated_length = (length - 7) // 2
        else:
            central_length = 8
            articulated_length = (length - 8) // 2
        articulated_liveries = [{
            'name': l['name'],
            'sprites': [grf.EMPTY_SPRITE] * 8,
        } for l in liveries or []]

        return 8 - central_length, 8 - articulated_length, articulated_liveries

    def _do_add_articulated_part(self, id, shorten_by, liveries, mid_liveries, props, callbacks=None):
        if shorten_by is not None:
            props['shorten_by'] = shorten_by
        self._articulated_parts.append((id, liveries, mid_liveries, callbacks, props))

    def add_articulated_part(self, *, id, liveries=None, callbacks=None, skip_props_check=False, length=None, **props):
        if not skip_props_check:
            # TODO support flipping (id + 0x4000)
            if isinstance(id, int) and id >= 0x4000:
                raise ValueError(f'Articulated part id too big ({id} >= 0x4000)')
            if self._props.get('is_dual_headed'):
                raise RuntimeError('Articulated parts are not allowed for dual-headed engines')

            if length is not None and 'shorten_by' in props:
                raise ValueError('Length and shorten_by properties are mutually exclusive')

            # REQUIRED_PROPS = ('id', 'liveries')
            # missing_props = [p for p in REQUIRED_PROPS if p not in props]
            # if missing_props:
            #     raise ValueError('Articulated part is missing required property: {}'.format(', '.join(missing_props)))

            ALLOWED_PROPS = (
                'cargo_capacity',
                'default_cargo_type',
                'refit_cost',
                'refittable_cargo_types',
                'shorten_by',
                'visual_effect_and_powered',
                'bitmask_vehicle_info',
                'misc_flags',
                'refittable_cargo_classes',
                'non_refittable_cargo_classes',
                'cargo_age_period',
                'cargo_allow_refit',
                'cargo_disallow_refit',
                'curve_speed_mod',
            )
            invalid_props = [p for p in props if p not in ALLOWED_PROPS]
            if invalid_props:
                raise ValueError('Property not allowed for articulated part: {}'.format(', '.join(invalid_props)))

        mid_shorten, art_shorten, art_liveries = self._calc_length_articulation(
            length, props.get('shorten_by'), liveries)

        if art_shorten is None:
            self._do_add_articulated_part(id, mid_shorten, liveries, liveries, props, callbacks)
        else:
            self._do_add_articulated_part(id, art_shorten, art_liveries, liveries, {})
            self._do_add_articulated_part(f'__{id}_aa_mid', mid_shorten, liveries, liveries, props, callbacks)
            self._do_add_articulated_part(f'__{id}_aa_tail', art_shorten, art_liveries, liveries, {})

        return self

    def _make_graphics(self, liveries, position):
        assert liveries
        res = [grf.Action1(
            feature=grf.TRAIN,
            set_count=len(liveries),
            sprite_count=8,
        )]
        layouts = []
        for i, l in enumerate(liveries):
            res.extend(l['sprites'])
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        if len(liveries) <= 1:
            return res, layouts[0]

        subtype_code = 'cargo_subtype'
        if position > 0:
            subtype_code=f'''
                TEMP[0x10F]=-{position}
                var(0x61, param=0xF2, shift=0, and=0xFF)
            ''',

        return res, grf.Switch(
            related_scope=False,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code=subtype_code,
        )

    def _set_callbacks(self, g):
        res = super()._set_callbacks(g)

        if self.liveries and len(self.liveries) > 1:
            # Liveries
            self.callbacks.cargo_subtype = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        if self.sound_effects:
            self.callbacks.sound_effect = grf.Switch(
                ranges=self.sound_effects,
                default=self.callbacks.graphics,
                code='extra_callback_info1_byte',
            )

        if self._articulated_parts:
            self.callbacks.articulated_part = grf.Switch(
                ranges={i + 1: g.resolve_id(self.feature, ap[0], articulated=True) for i, ap in enumerate(self._articulated_parts)},
                default=0x7fff,
                code='extra_callback_info1_byte',
            )
        return res

    def _set_articulated_part_callbacks(self, g, position, callbacks):
        pass


    def _gen_purchase_sprites(self):
        if self.purchase_sprite is None and (self.length or 8) > 8 and self.liveries is not None:
            # First part has no graphics so set purchase sprite
            self.purchase_sprite = self.liveries[0]['sprites'][6]
        return super()._gen_purchase_sprites()


    def get_sprites(self, g):
        # Check in case property was changed after add_articulated
        if self._props.get('is_dual_headed') and self._articulated_parts:
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines (vehicle id {self.id})')

        res = self._gen_purchase_sprites()

        if self._head_liveries:
            sprites, self.callbacks.graphics = self._make_graphics(self._head_liveries, 0)
            res.extend(sprites)

        res.extend(self._set_callbacks(g))

        self.callbacks.set_flag_props(self._props)

        tid = g.resolve_id(self.feature, self.id)
        res.extend(self._gen_name_sprites(tid))

        res.append(definition := grf.Define(
            feature=grf.TRAIN,
            id=tid,
            props={
                'sprite_id': 0xfd,  # magic value for newgrf sprites
                'max_speed': self.max_speed,
                'weight_low': self.weight % 256,
                'weight_high': self.weight // 256,
                **self._props
            }
        ))

        res.append(self.callbacks.make_map_action(definition))

        position = 0
        for apid, liveries, _, initial_callbacks, props in self._articulated_parts:
            apid = g.resolve_id(self.feature, apid, articulated=True)
            position += 1
            callbacks = CallbackManager(self.feature, initial_callbacks)
            self._set_articulated_part_callbacks(g, position, callbacks)

            if liveries:
                sprites, callbacks.graphics = self._make_graphics(liveries, position)
                res.extend(sprites)

            callbacks.set_flag_props(props)

            res.append(definition := grf.Define(
                feature=grf.TRAIN,
                id=apid,
                props={
                    'sprite_id': 0xfd,  # magic value for newgrf sprites
                    'engine_class': self._props.get('engine_class'),
                    **props
                }
            ))
            res.append(callbacks.make_map_action(definition))

        return res


SHIP_KMH_TO_NFO = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508]
SHIP_MPH_TO_NFO = [0, 4, 8, 10, 14, 16, 20, 24, 26, 30, 32, 36, 40, 42, 46, 48, 52, 56, 58, 62, 64, 68, 72, 74, 78, 80, 84, 88, 90, 94, 96, 100, 104, 106, 110, 112, 116, 120, 122, 126, 128, 132, 136, 138, 142, 144, 148, 152, 154, 158, 160, 164, 168, 170, 174, 176, 180, 184, 186, 190, 192, 196, 200, 202, 206, 208, 212, 216, 218, 222, 224, 228, 232, 234, 238, 240, 244, 248, 250, 254, 256, 260, 264, 266, 270, 272, 276, 280, 282, 286, 288, 292, 296, 298, 302, 304, 308, 312, 314, 318, 320, 324, 328, 330, 334, 336, 340, 344, 346, 350, 352, 356, 360, 362, 366, 368, 372, 376, 378, 382, 384, 388, 392, 394, 398, 400, 404, 408, 410, 414, 416, 420, 424, 426, 430, 432, 436, 440, 442, 446, 448, 452, 456, 458, 462, 464, 468, 472, 474, 478, 480, 484, 488, 490, 494, 496, 500, 504, 506, 510, 512, 516, 520, 522, 526, 528, 532, 536, 538, 542, 544, 548, 552, 554, 558, 560, 564, 568, 570, 574, 576, 580, 584, 586, 590, 592, 596, 600, 602, 606, 608, 612, 616, 618, 622, 624, 628, 632, 634, 638, 640, 644, 648, 650, 654, 656, 660, 664, 666, 670, 672, 676, 680, 682, 686, 688, 692, 696, 698, 702, 704, 708, 712, 714, 718, 720, 724, 728, 730, 734, 736, 740, 744, 746, 750, 752, 756, 760, 762, 766, 768, 772, 776, 778, 782, 784, 788, 792, 794, 798, 800, 804, 808, 810, 814, 816]

def _check_speed(speed, array, unit):
    if speed < 0:
        raise ValueError("Speed can't be negative.")
    if speed > len(array):
        raise ValueError(f'Speeds over {len(array)} {unit} are not supported.')
    return array[speed]


class Ship(Vehicle):

    @staticmethod
    def kmh(speed):
        return _check_speed(speed, SHIP_KMH_TO_NFO, 'km/h')

    @staticmethod
    def kmhish(speed):
        return speed // 2

    @staticmethod
    def mph(speed):
        # TODO probably doesn't show exact mph in the game
        return _check_speed(speed, SHIP_MPH_TO_NFO, 'mph')

    def __init__(self, *, id, name, max_speed, liveries=None, callbacks=None, purchase_sprite=None, additional_text=None, **props):
        super().__init__(grf.SHIP, liveries=liveries, callbacks=callbacks, purchase_sprite=purchase_sprite, additional_text=additional_text)

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self._props = props

    # def _set_callbacks(self, g):
    #     res = super()._set_callbacks(g)

    #     if self.max_speed.precise_value >= 0x400:
    #         # TODO only set speed property
    #         self.callbacks.change_properties = grf.Switch(
    #             ranges={
    #                 0x15: self.max_speed.value,
    #             },
    #             default=self.callbacks.graphics,
    #             code='extra_callback_info1_byte',
    #         )

    #     return res

    def get_sprites(self, g):
        res = self._gen_purchase_sprites()

        if self.liveries:
            layouts = []
            for i, l in enumerate(self.liveries):
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            if len(self.liveries) > 1:

                self.callbacks.graphics = grf.Switch(
                    related_scope=True,
                    ranges=dict(enumerate(layouts)),
                    default=layouts[0],
                    code='cargo_subtype',
                )
            else:
                self.callbacks.graphics = layouts[0]

        res.extend(self._set_callbacks(g))

        # Liveries
        if self.liveries and len(self.liveries) > 1:
            self.callbacks.cargo_subtype = grf.Switch(
                ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
                default=0x400,
                code='cargo_subtype',
            )

        self.callbacks.set_flag_props(self._props)

        res.extend(self._gen_name_sprites(self.id))

        res.append(definition := grf.Define(
            feature=grf.SHIP,
            id=self.id,
            props={
                'sprite_id': 0xff,
                'max_speed': self.max_speed,
                **self._props
            }
        ))
        if self.liveries:
            res.append(grf.Action1(
                feature=grf.SHIP,
                set_count=len(self.liveries),
                sprite_count=8,
            ))

            for l in self.liveries:
                res.extend(l['sprites'])

        res.append(self.callbacks.make_map_action(definition))
        return res



class Object(grf.Define):
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


class BaseCosts(grf.SpriteGenerator):
    STATION_VALUE = 0
    BUILD_RAIL = 1
    BUILD_ROAD = 2
    BUILD_SIGNALS = 3
    BUILD_BRIDGE = 4
    BUILD_DEPOT_TRAIN = 5
    BUILD_DEPOT_ROAD = 6
    BUILD_DEPOT_SHIP = 7
    BUILD_TUNNEL = 8
    BUILD_STATION_RAIL = 9
    BUILD_STATION_RAIL_LENGTH = 10
    BUILD_STATION_AIRPORT = 11
    BUILD_STATION_BUS = 12
    BUILD_STATION_TRUCK = 13
    BUILD_STATION_DOCK = 14
    BUILD_VEHICLE_TRAIN = 15
    BUILD_VEHICLE_WAGON = 16
    BUILD_VEHICLE_AIRCRAFT = 17
    BUILD_VEHICLE_ROAD = 18
    BUILD_VEHICLE_SHIP = 19
    BUILD_TREES = 20
    TERRAFORM = 21
    CLEAR_GRASS = 22
    CLEAR_ROUGH = 23
    CLEAR_ROCKS = 24
    CLEAR_FIELDS = 25
    CLEAR_TREES = 26
    CLEAR_RAIL = 27
    CLEAR_SIGNALS = 28
    CLEAR_BRIDGE = 29
    CLEAR_DEPOT_TRAIN = 30
    CLEAR_DEPOT_ROAD = 31
    CLEAR_DEPOT_SHIP = 32
    CLEAR_TUNNEL = 33
    CLEAR_WATER = 34
    CLEAR_STATION_RAIL = 35
    CLEAR_STATION_AIRPORT = 36
    CLEAR_STATION_BUS = 37
    CLEAR_STATION_TRUCK = 38
    CLEAR_STATION_DOCK = 39
    CLEAR_HOUSE = 40
    CLEAR_ROAD = 41
    RUNNING_TRAIN_STEAM = 42
    RUNNING_TRAIN_DIESEL = 43
    RUNNING_TRAIN_ELECTRIC = 44
    RUNNING_AIRCRAFT = 45
    RUNNING_ROADVEH = 46
    RUNNING_SHIP = 47
    BUILD_INDUSTRY = 48
    CLEAR_INDUSTRY = 49
    BUILD_UNMOVABLE = 50
    CLEAR_UNMOVABLE = 51
    BUILD_WAYPOINT_RAIL = 52
    CLEAR_WAYPOINT_RAIL = 53
    BUILD_WAYPOINT_BUOY = 54
    CLEAR_WAYPOINT_BUOY = 55
    TOWN_ACTION = 56
    BUILD_FOUNDATION = 57
    BUILD_INDUSTRY_RAW = 58
    BUILD_TOWN = 59
    BUILD_CANAL = 60
    CLEAR_CANAL = 61
    BUILD_AQUEDUCT = 62
    CLEAR_AQUEDUCT = 63
    BUILD_LOCK = 64
    CLEAR_LOCK = 65
    MAINTENANCE_RAIL = 66
    MAINTENANCE_ROAD = 67
    MAINTENANCE_CANAL = 68
    MAINTENANCE_STATION = 69
    MAINTENANCE_AIRPORT = 70

    def __init__(self, costs):
        self.costs = costs

    def get_sprites(self, g):
        return [
            grf.DefineMultiple(
                feature=grf.GLOBAL_VAR,
                first_id=first,
                count=len(values),
                props={'basecost': values},
            )
            for first, values in combine_ranges(self.costs.items())
        ]


class VariantGroup:
    def __init__(self, name, *items):
        self.name = name
        if len(items) < 2:
            raise ValueError('VariantGroup needs at least 2 vehicles')
        if isinstance(items[0], VariantGroup):
            raise ValueError('First element of a VariantGroup can''t be a VariantGroup')
        self.items = items

    @property
    def first(self):
        x = self.items[0]
        if isinstance(x, VariantGroup):
            return x.first
        return x


class SetPurchaseOrder(grf.SpriteGenerator):
    def __init__(self, *order):
        super().__init__()
        self.flat_order = []
        self._variant_group = {}
        self._variant_callbacks = defaultdict(dict)
        self._variant_callbacks_set = False
        self._feature = None

        def process(order, group, level, callbacks):
            for e in order:
                if isinstance(e, VariantGroup):
                    x = e.first
                    if e.name is not None:
                        cb = {
                            **callbacks,
                            level: e.name,
                        }
                    else:
                        cb = callbacks

                    process(e.items, x, level + 1, cb)
                    if group is not None:
                        self._variant_group[id(x)] = group
                    else:
                        self._variant_group.pop(id(x))
                else:
                    self.flat_order.append(e)
                    if group is not None:
                        self._variant_group[id(e)] = group
                    if callbacks:
                        self._variant_callbacks[id(e)] = (e, callbacks)
                    if self._feature is None:
                        self._feature = e.feature

        process(order, None, 0, {})
        if self._feature is None:
            raise ValueError('Could not detect feature.')

    def set_variant_callbacks(self, g):
        for _, (x, names) in self._variant_callbacks.items():
            c = x.callbacks.name = grf.Switch(
                code='extra_callback_info1',
                ranges = {
                    0x20 | (level << 8): g.strings.add(name).get_global_id()
                    for level, name in names.items()
                },
                default=0x400,
            )
        self._variant_callbacks_set = True
        return self

    def get_sprites(self, g):
        if not self._variant_callbacks_set:
            raise RuntimeError(
                'SetPurchaseOrder.set_variant_callbacks needs to be called before the'
                ' sprite generation when using custom variant group names.')
        prev = None
        res = []
        for v in reversed(self.flat_order):
            variant_props = {}
            sort_props = {}
            if prev is not None:
                sort_props = {'sort_purchase_list': g.resolve_id(self._feature, prev.id)}

            vg = self._variant_group.get(id(v))
            if vg is not None:
                variant_props = {'variant_group': g.resolve_id(self._feature, vg.id)}

            if sort_props or variant_props:
                res.append(grf.Define(
                    feature=self._feature,
                    id=g.resolve_id(self._feature, v.id),
                    props={**sort_props, **variant_props}
                ))
            prev = v
        return res


class SetGlobalTrainDepotYOffset(grf.SpriteGenerator):
    def __init__(self, offset):
        self.offset = offset

    def get_sprites(self, g):
        return [
            grf.ComputeParameters(
                target=0x8e,
                operation=0x00,
                if_undefined=False,
                source1=0xff,
                source2=0xff,
                value=self.offset,
            )
        ]


class GlobalTrainMiscFlag:
    DESERT_TREES_AND_FIELDS = 0
    DESERT_PAVEMENT_AND_LIGHTS = 1
    FIELDS_BOUNDING_BOX = 2
    DEPOT_FULL_TRAIN_WIDTH = 3
    AMBIENT_SOUND_CB = 4
    THIRD_TRACK_CATENARIES = 5
    SECOND_ROCKY_SET = 6
    TTDPATCH_FLAG = 31


class SetGlobalTrainMiscFlag(grf.SpriteGenerator):
    def __init__(self, flag):
        self.flag = flag

    def get_sprites(self, g):
        return [
            grf.ComputeParameters(
                target=0x9e,
                operation=0x08,
                if_undefined=False,
                source1=0x9e,
                source2=0xff,
                value=1 << self.flag
            )
        ]
