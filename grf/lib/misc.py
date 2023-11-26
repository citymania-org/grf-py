from collections.abc import Iterable


import grf


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


class DisableDefault(grf.SpriteGenerator):
    # Pairs of (count, props)
    DISABLE_INFO = {
        grf.TRAIN: (
            116,
            {'climates_available': grf.NO_CLIMATE}
        ),
        grf.RV: (
            88,
            {'climates_available': grf.NO_CLIMATE}
        ),
        grf.SHIP: (
            11,
            {'climates_available': grf.NO_CLIMATE}
        ),
        grf.AIRCRAFT: (
            41,
            {'climates_available': grf.NO_CLIMATE}
        ),
        grf.HOUSE: (
            110,
           {'substitute': 0xFF},
        ),
        grf.INDUSTRY: (
            37,
           {'substitute_type': 0xFF},
        ),
        # TODO add airport action 0 props
        # grf.AIRPORT: (
        #     10,
        #    {'substitute_type': 0xFF},
        # ),
        grf.CARGO: (
            27,
            {
                'bit_number': 0xff,
                'label': 0,
            }
        ),
    }

    def __init__(self, feature, ids=None):
        if feature not in self.DISABLE_INFO:
            raise ValueError(f'Unsuppported feature `{feature}`')

        if ids is not None:
            assert isinstance(ids, (int, Iterable)), type(ids)
            if isinstance(ids, int): ids = [ids,]
            assert all(isinstance(x, int) for x in ids)
        self.feature = feature
        self.ids = ids

    def get_sprites(self, g):
        count, props = self.DISABLE_INFO[self.feature]
        ids = range(count) if self.ids is None else self.ids

        return (
            grf.DefineMultiple(
                feature=self.feature,
                first_id=first,
                count=len(values),
                props={
                    k: [v] * len(values) for k, v in props.items()
                }
            )
            for first, values in combine_ranges((x, None) for x in ids)
        )


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
