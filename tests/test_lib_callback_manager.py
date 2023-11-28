from nose.tools import raises

from grf import TRAIN, HOUSE, DualCallback, GraphicsCallback, make_callback_manager, SpriteGenerator, Action3, Switch, DefineMultiple, Ref

from .common import check_lib


class DummyDefinition(DefineMultiple):
	def __init__(self, feature, first_id):
		self.feature = feature
		self.first_id = first_id


class CBGenerator(SpriteGenerator):
	def __init__(self, manager):
		self.manager = manager

	def get_sprites(self, g):
		definition = DummyDefinition(self.manager.feature, 0)
		return self.manager.make_map_action(definition)


@raises(AttributeError)
def test_non_vehicle_properties_raises_attribute_error():
    cbm = make_callback_manager(feature=HOUSE, callbacks={})
    cbm.properties


def test_non_vehicle_can_make_switch():
    cbm = make_callback_manager(feature=HOUSE, callbacks={})
    cbm.make_switch()


@raises(ValueError)
def test_no_graphics():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={},
    )
    cbm.make_switch()


def test_assign_graphics():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={},
    )
    cbm.graphics.default = 345
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=345,
            )
        ]
    )


def test_default_graphics():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={
            'graphics': 1234,
        },
    )
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=1234,
            )
        ]
    )

    cbm.graphics.default = 345
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=345,
            )
        ]
    )


def test_split_graphics():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={
            'graphics': GraphicsCallback(
                default=1234,
                purchase=5678,
            ),
        },
    )
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: 5678},
                default=1234,
            )
        ]
    )

    cbm.graphics.default = 345
    cbm.graphics.purchase = 789
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: 789},
                default=345,
            )
        ]
    )

    cbm.graphics.purchase = None
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=345,
            )
        ]
    )

    cbm.graphics = 123
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=123,
            )
        ]
    )


def test_dual_callback():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={
            'graphics': 0xE0,
            'articulated_part': DualCallback(
                default=1234,
                purchase=5678,
            )
        },
    )
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 5678},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='current_callback',
                ranges={22: 1234},
                default=224,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(254),
            ),
        ]
    )

    cbm.articulated_part.default = 345
    cbm.articulated_part.purchase = 789
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 789},
                default=0xE0,
            ),
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='current_callback',
                ranges={22: 345},
                default=0xE0,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(254),
            ),
        ]
    )

    cbm.articulated_part.purchase = None
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 345},
                default=0xE0,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={},
                default=Ref(255),
            ),
        ]
    )

    cbm.articulated_part = 123
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 123},
                default=0xE0,
            ),
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='current_callback',
                ranges={22: 123},
                default=0xE0,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(254),
            ),
        ]
    )

    cbm.articulated_part.default = 567
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 123},
                default=0xE0,
            ),
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='current_callback',
                ranges={22: 567},
                default=0xE0,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(254),
            ),
        ]
    )

    cbm.articulated_part.default = None
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={22: 123},
                default=0xE0,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=0xE0,
            ),

        ]
    )


@raises(AttributeError)
def test_dual_callback():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={
            'graphics': 0xE0,
        },
    )
    cbm.no_such_callback = 0


def test_properties_callbacks():
    cbm = make_callback_manager(
        feature=TRAIN,
        callbacks={
            'graphics': 0xE0,
            'properties': {
                'max_speed': 123,
            }
        },
    )
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='extra_callback_info1_byte',
                ranges={9: 123},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={54: Ref(254)},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=253,
                related_scope=False,
                code='current_callback',
                ranges={54: Ref(254)},
                default=224,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(253),
            ),
        ]
    )

    cbm.properties.max_speed.default = 234
    check_lib(
        None,
        CBGenerator(cbm),
        [
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='extra_callback_info1_byte',
                ranges={9: 123},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=255,
                related_scope=False,
                code='current_callback',
                ranges={54: Ref(254)},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=253,
                related_scope=False,
                code='extra_callback_info1_byte',
                ranges={9: 234},
                default=224,
            ),
            Switch(
                feature=TRAIN,
                ref_id=254,
                related_scope=False,
                code='current_callback',
                ranges={54: Ref(253)},
                default=224,
            ),
            Action3(
                feature=TRAIN,
                wagon_override=False,
                ids=[0],
                maps={255: Ref(255)},
                default=Ref(254),
            ),
        ]
    )
