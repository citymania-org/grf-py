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
        POWERED_WAGONS = 0x10
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


class CallbackManager:
    def __init__(self, domain):
        self._domain = domain
        self._callbacks = {}

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super().__setattr__(name, value)
        if name.lower() != name:
            raise AttributeError(name)

        cb_id = getattr(self._domain, name.upper())
        self._callbacks[cb_id] = value

    def get_flags(self):
        # TODO checeked only for vehicles
        FLAGS = {
            Callback.Vehicle.POWERED_WAGONS: 0x1,
            Callback.Vehicle.WAGON_LENGTH: 0x2,
            Callback.Vehicle.LOAD_AMOUNT: 0x4,
            Callback.Vehicle.REFIT_CAPACITY: 0x8,
            Callback.Vehicle.ARTICULATED_PART: 0x10,
            Callback.Vehicle.CARGO_SUBTYPE: 0x20,
            Callback.Vehicle.COLOUR_MAPPING: 0x40,
            Callback.Vehicle.SOUND_EFFECT: 0x80,
        }
        res = 0
        for k in self._callbacks.keys():
            res |= FLAGS.get(k, 0)
        return res

    def make_switch(self, graphics, purchase_graphics=None):
        # False - only purchase, True - general + purchase
        PURCHASE = {
            Callback.Vehicle.PURCHASE_TEXT: False,
            Callback.Vehicle.CHANGE_PROPERTIES: True,
        }
        callbacks = {}
        purchase_callbacks = {}
        for k, c in self._callbacks.items():
            pdata = PURCHASE.get(k)
            if not pdata:
                callbacks[k] = c
            if pdata is not None:
                purchase_callbacks[k] = c

        maps = {}
        if purchase_callbacks:
            purchase_switch = grf.Switch(
                ranges=purchase_callbacks,
                default=purchase_graphics or graphics,
                code='current_callback',
            )
            maps = {255: purchase_switch}
        default = graphics
        if callbacks:
            default = grf.Switch(
                ranges={**callbacks},
                default=graphics,
                code='current_callback',
            )
        return default, maps


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


class RoadVehicle(grf.SpriteGenerator):

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
    def kmhishph(speed):
        return RoadVehicle.Speed(speed * 2)

    @staticmethod
    def mph(speed):
        # TODO doesn't show exact mph in the game
        return RoadVehicle.Speed((speed * 16 + 4) // 5)

    def __init__(self, *, id, name, liveries, max_speed, additional_text=None, livery_refits=None, **props):
        for l in liveries:
            if 'name' not in l:
                raise ValueError(f'RoadVehicle livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'RoadVehicle livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'RoadVehicle livery expects 8 sprites, found {len(sprites)}')

        self.id = id
        self.name = name

        if isinstance(max_speed, self.Speed):
            self.max_speed = max_speed
        else:
            self.max_speed = self.Speed(max_speed * 4)
        self.additional_text = additional_text
        self.liveries = liveries
        self._props = props

    def get_sprites(self, g):
        callbacks = CallbackManager(Callback.Vehicle)

        layouts = []
        for i, l in enumerate(self.liveries):
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        layout = grf.Switch(
            related_scope=True,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code='cargo_subtype',
        )

        if self.additional_text:
            callbacks.purchase_text = g.strings.add(self.additional_text).get_global_id()

        if self.max_speed.precise_value >= 0x400:
            callbacks.change_properties = grf.Switch(
                ranges={
                    0x15: self.max_speed.value,
                },
                default=layout,
                code='var(16, 0, 255)',
            )

        # Liveries
        callbacks.cargo_subtype = grf.Switch(
            ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
            default=0x400,
            code='cargo_subtype',
        )

        if callbacks.get_flags():
            self._props['cb_flags'] = self._props.get('cb_flags', 0) | callbacks.get_flags()

        res = [
            grf.DefineStrings(
                feature=grf.RV,
                offset=self.id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8')]
            ),
        ]

        res.append(grf.Define(
            feature=grf.RV,
            id=self.id,
            props={
                'sprite_id': 0xff,
                'precise_max_speed': min(self.max_speed.precise_value, 0xff),
                'max_speed': min(self.max_speed.value, 0xff),
                **self._props
            }
        ))

        res.append(grf.Action1(
            feature=grf.RV,
            set_count=len(self.liveries),
            sprite_count=8,
        ))

        for l in self.liveries:
            res.extend(l['sprites'])

        default, maps = callbacks.make_switch(layout)
        res.append(grf.Action3(
            feature=grf.RV,
            ids=[self.id],
            maps=maps,
            default=default,
        ))
        return res


class Train(grf.SpriteGenerator):
    class EngineClass:
        STEAM = 0x0
        DIESEL = 0x8
        ELECTRIC = 0x28
        MONORAIL = 0x32
        MAGLEV = 0x32

    @staticmethod
    def kmhishph(speed):
        return speed

    @staticmethod
    # TODO doesn't show exact mph in the game
    def mph(speed):
        return (speed * 16 + 4) // 10

    def __init__(self, *, id, name, liveries, max_speed, additional_text=None, sound_effects=None, **props):
        for l in liveries:
            if 'name' not in l:
                raise ValueError(f'Train livery is missing the name')
            sprites = l.get('sprites')
            if sprites is None:
                raise ValueError(f'Train livery {l["name"]} is missing sprites')
            if len(sprites) != 8:
                raise ValueError(f'Train livery expects 8 sprites, found {len(sprites)}')

        self.id = id
        self.name = name
        self.max_speed = max_speed
        self.additional_text = additional_text
        self.liveries = liveries
        self.sound_effects = sound_effects
        REQUIRED_PROPS = ('engine_class', )
        missing_props = [p for p in REQUIRED_PROPS if p not in props]
        if missing_props:
            raise ValueError('Missing required properties for Train: {}'.format(', '.join(missing_props)))
        self._props = props
        self._articulated_parts = []

    def add_articulated_part(self, *, id, liveries, **props):
        if self._props.get('is_dual_headed'):
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines')

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
            'visual_effect',
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

        self._articulated_parts.append((id, liveries, props))
        return self

    def get_sprites(self, g):
        # Check in case property was changed after add_articulated
        if self._props.get('is_dual_headed') and self._articulated_parts:
            raise RuntimeError('Articulated parts are not allowed for dual-headed engines (vehicle id {self.id})')

        callbacks = CallbackManager(Callback.Vehicle)

        layouts = []
        for i, l in enumerate(self.liveries):
            layouts.append(grf.GenericSpriteLayout(
                ent1=(i,),
                ent2=(i,),
            ))

        layout = grf.Switch(
            related_scope=True,
            ranges=dict(enumerate(layouts)),
            default=layouts[0],
            code='cargo_subtype',
        )

        if self.additional_text:
            callbacks.purchase_text = g.strings.add(self.additional_text).get_global_id()

        # Liveries
        callbacks.cargo_subtype = grf.Switch(
            ranges={i: g.strings.add(l['name']).get_global_id() for i, l in enumerate(self.liveries)},
            default=0x400,
            code='cargo_subtype',
        )

        if self.sound_effects:
            callbacks.sound_effect = grf.Switch(
                ranges=self.sound_effects,
                default=layout,
                code='extra_callback_info1 & 255',
            )

        if self._articulated_parts:
            callbacks.articulated_part = grf.Switch(
                ranges={i + 1: ap[0] for i, ap in enumerate(self._articulated_parts)},
                default=0x7fff,
                code='extra_callback_info1 & 255',
            )

        if callbacks.get_flags():
            self._props['cb_flags'] = self._props.get('cb_flags', 0) | callbacks.get_flags()

        res = [
            grf.DefineStrings(
                feature=grf.TRAIN,
                offset=self.id,
                is_generic_offset=False,
                strings=[self.name.encode('utf-8')]
            ),
        ]

        res.append(grf.Define(
            feature=grf.TRAIN,
            id=self.id,
            props={
                'sprite_id': 0xfd,  # magic value for newgrf sprites
                'max_speed': self.max_speed,
                **self._props
            }
        ))
        res.append(grf.Action1(
            feature=grf.TRAIN,
            set_count=len(self.liveries),
            sprite_count=8,
        ))

        for l in self.liveries:
            res.extend(l['sprites'])

        default, maps = callbacks.make_switch(layout)
        res.append(grf.Action3(
            feature=grf.TRAIN,
            ids=[self.id],
            maps=maps,
            default=default,
        ))

        for apid, liveries, props in self._articulated_parts:
            res.append(grf.Define(
                feature=grf.TRAIN,
                id=apid,
                props={
                    'sprite_id': 0xfd,  # magic value for newgrf sprites
                    'engine_class': self._props.get('engine_class'),
                    **props
                }
            ))

            res.append(grf.Action1(
                feature=grf.TRAIN,
                set_count=len(liveries),
                sprite_count=8,
            ))

            layouts = []
            for i, l in enumerate(liveries):
                res.extend(l['sprites'])
                layouts.append(grf.GenericSpriteLayout(
                    ent1=(i,),
                    ent2=(i,),
                ))

            layout = grf.Switch(
                related_scope=True,
                ranges=dict(enumerate(layouts)),
                default=layouts[0],
                code='cargo_subtype',
            )

            res.append(grf.Action3(
                feature=grf.TRAIN,
                ids=[apid],
                maps={},
                default=layout,
            ))
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

