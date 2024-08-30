from .common import Feature, FeatureMeta, VEHICLE_FEATURES
from .actions import ACTION0_PROPS, ACTION0_PROP_DICT
from .va2vars import VA2_VARS, VA2_VARS_INV, NML_VA2_GLOBALVARS


def add_feature(id, name, class_name, *, properties={}, variables={}, is_vehicle_feature=False):
    obj = type.__call__(Feature, id, name, class_name)
    FeatureMeta.FEATURES[id] = obj
    Feature._FROM_NAME[name] = obj
    if is_vehicle_feature:
        VEHICLE_FEATURES.add(obj)
    ACTION0_PROPS[obj] = properties
    ACTION0_PROP_DICT[obj] = {name: (id, size) for id, (name, size, *_) in properties.items()}
    VA2_VARS[obj] = {**NML_VA2_GLOBALVARS, **variables}
    VA2_VARS_INV[obj] = {(v['var'], v['start'], (1 << v['size']) - 1): k for k, v in VA2_VARS[obj].items()}

    return obj
