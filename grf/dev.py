"""
This module provides developer utilities for prototyping and registering new features that do not yet exist in the game.

It is intended for advanced users and developers who wish to experiment with or extend the feature set by defining custom features, properties, and variables before they are officially supported.

Usage:
    Use the `add_feature` function to register a new feature with the system. This will make the feature available for further development and testing, even if it is not present in the base game.

    Example:
        add_feature(
            id=42,
            name="myfeature",
            class_name="MyFeature",
            properties={...},
            variables={...},
            is_vehicle_feature=False
        )
"""

from .common import Feature, FeatureMeta, VEHICLE_FEATURES
from .actions import ACTION0_PROPS, ACTION0_PROP_DICT
from .va2vars import VA2_VARS, VA2_VARS_INV, NML_VA2_GLOBALVARS


def add_feature(id, name, class_name, *, properties={}, variables={}, is_vehicle_feature=False):
    """
    @brief Create and register a new NewGRF feature for development and testing purposes.

    @param id                Unique identifier for the feature. (int)
    @param name              Name of the feature. (str)
    @param class_name        Name of the class representing the feature. (str)
    @param properties        Dictionary of property definitions for the feature. (dict)
    @param variables         Dictionary of variable definitions for the feature. (dict)
    @param is_vehicle_feature Set to True if the feature is a vehicle feature. (bool)

    @return The registered feature object.

    The function will register the feature in the relevant registries, making it available for use in the rest of the system.
    """
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
