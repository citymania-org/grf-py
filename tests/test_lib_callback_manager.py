from nose.tools import raises

from grf import TRAIN, HOUSE, GraphicsCallback, make_callback_manager, SpriteGenerator, Action3, Switch, DefineMultiple

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
		return [self.manager.make_map_action(definition)]


@raises(AttributeError)
def test_non_vehicle_properties_raises_attribute_error():
    cbm = make_callback_manager(feature=HOUSE, callbacks={})
    cbm.properties


def test_non_vehicle_can_make_switch():
    cbm = make_callback_manager(feature=HOUSE, callbacks={'graphics': 0})
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

# def test_split_callback():
#     check_lib(
#         None,
#         CBGenerator(
#             TRAIN,
#             {
#                 'graphics': 0xE0,
#                 'purchase_text': Callback(
#                     default=1234,
#                     purchase=5678,
#                 ),
#             },
#         ),
#         [
#             Action3(
#                 feature=TRAIN,
#                 wagon_override=False,
#                 ids=[0],
#                 maps={},
#                 default=0xE0,
#             )

#         ]
#     )



# def test_split_property():
#     check_lib(
#         None,
#         CBGenerator(
#             TRAIN,
#             {
#                 'graphics': 0xE0,
#                 'properties': {
#                     'power': Callback(
#                         default=1234,
#                         purchase=5678,
#                     )
#                 },
#             },
#         ),
#         [
#             Action3(
#                 feature=TRAIN,
#                 wagon_override=False,
#                 ids=[0],
#                 maps={},
#                 default=0xE0,
#             )

#         ]
#     )
