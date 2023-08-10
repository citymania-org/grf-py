from nose.tools import raises

from grf import TRAIN, HOUSE, CallbackManager, SpriteGenerator, Action3, Switch, DefineMultiple

from .common import check_lib


class DummyDefinition(DefineMultiple):
	def __init__(self, feature, first_id):
		self.feature = feature
		self.first_id = first_id


class CBGenerator(SpriteGenerator):
	def __init__(self, feature, callbacks=None):
		self.callbacks = CallbackManager(feature=feature, callbacks=callbacks)

	def get_sprites(self, g):
		definition = DummyDefinition(self.callbacks.feature, 0)
		return [self.callbacks.make_map_action(definition)]



# def test_lib_rv_split_cb():
#     check_lib(
#         None,
#         CBGenerator(
#         	TRAIN,
#         	{
#         		'graphics': 0xE0,
#         		'purchase_text': {
#         			'default': 1234,
#         			'purchase': 5678,
#         		}
#         	},
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


@raises(AttributeError)
def test_lib_callback_manager_non_vehicle_properties_raises_attribute_error():
    cb = CallbackManager(feature=HOUSE, callbacks={})
    cb.properties


def test_lib_callback_manager_non_vehicle_can_make_switch():
    cb = CallbackManager(feature=HOUSE, callbacks={'graphics': 0})
    cb.make_switch()