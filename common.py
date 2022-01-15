def hex_str(s, n=None):
    add = ''
    if n is not None and len(s) > n:
        s = s[:n - 3]
        add = '...'
    if isinstance(s, (bytes, memoryview)):
        return ':'.join('{:02x}'.format(b) for b in s) + add
    return ':'.join('{:02x}'.format(ord(c)) for c in s) + add


def utoi8(value):
    return value | -(value & 0x80)


def utoi32(value):
    return value | -(value & 0x80000000)


class FeatureMeta(type):
    def __call__(cls, feature):
        return cls.FEATURES[feature]


class Feature(metaclass=FeatureMeta):
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.constant = name.upper()

    def __repr__(self):
        return self.constant


FeatureMeta.FEATURES = [None] * 0x14
for k, n in ((0x00, 'train'),
             (0x01, 'rv'),
             (0x02, 'ship'),
             (0x03, 'aircraft'),
             (0x04, 'station'),
             (0x05, 'canal'),
             (0x06, 'bridge'),
             (0x07, 'house'),
             (0x08, 'global_var'),
             (0x09, 'industry_tile'),
             (0x0a, 'industry'),
             (0x0b, 'cargo'),
             (0x0c, 'sound_effect'),
             (0x0d, 'airport'),
             (0x0e, 'signal'),
             (0x0f, 'object'),
             (0x10, 'railtype'),
             (0x11, 'airport_tile'),
             (0x12, 'roadtype'),
             (0x13, 'Tramtype'),
            ):
    FeatureMeta.FEATURES[k] = type.__call__(Feature, k, n)

TRAIN, RV, SHIP, AIRCRAFT, STATION, RIVER, BRIDGE, HOUSE, GLOBAL_VAR, INDUSTRY_TILE, INDUSTRY, CARGO, \
SOUND_EFFECT, AIRPORT, SIGNAL, OBJECT, RAILTYPE, AIRPORT_TILE, ROADTYPE, TRAMTYPE = FeatureMeta.FEATURES

CANAL = RIVER
