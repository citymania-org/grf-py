from grf import Train, TRAIN, Ref, DefineStrings, Define, Action3, Switch

from .common import check_lib


def test_lib_train_basics():
	check_lib(
		None,
		Train(
			id=0x1D1D,
			name='Test Train',
			max_speed=0xAEED,
			weight=0xE1,
			engine_class=Train.EngineClass.STEAM,
			callbacks={
				'graphics': 0xE0,
			}
		),
		[
	        DefineStrings(
	            feature=TRAIN,
	            lang=127,
	            offset=7453,
	            is_generic_offset=False,
	            strings=[b'Test Train']
	        ),
	        Define(
	            feature=TRAIN,
	            id=0x1D1D,
	            props={
	                'sprite_id': 253,
	                'max_speed': 0xAEED,
	                'weight_low': 225,
	                'weight_high': 0,
	                'engine_class': 0,
	            }
	        ),
	        Action3(
	            feature=TRAIN,
	            wagon_override=False,
	            ids=[0x1D1D],
	            maps={},
	            default=0xE0,
	        )

		]
	)


def test_lib_train_properties_cb():
	check_lib(
		None,
		Train(
			id=0x1D1D,
			name='Test Train',
			max_speed=0xAEED,
			weight=0xE1,
			engine_class=Train.EngineClass.STEAM,
			callbacks={
				'graphics': 0xE0,
				'properties': {
					'loading_speed': 0x10AD,
					'max_speed': 0xAEED,
					'power': 0x50E2,
					'running_cost_factor': 0x10A0,
					'cargo_capacity': 0xCAA1,
					'tractive_effort_coefficient': 0xEEC0,
					'shorten_by': 0x5027,
					'bitmask_vehicle_info': 0x1A11,
					'cargo_age_period': 0xCAAE,
					'curve_speed_mod': 0xCEEE,
				}
			}
		),
		[
	        DefineStrings(
	            feature=TRAIN,
	            lang=127,
	            offset=7453,
	            is_generic_offset=False,
	            strings=[b'Test Train']
	        ),
	        Define(
	            feature=TRAIN,
	            id=0x1D1D,
	            props={
	                'sprite_id': 253,
	                'max_speed': 0xAEED,
	                'weight_low': 225,
	                'weight_high': 0,
	                'engine_class': 0,
	            }
	        ),
	        Switch(
	            feature=TRAIN,
	            ref_id=254,
	            related_scope=False,
	            code='extra_callback_info1_byte',
	            ranges={
	                7: 0x10AD,
	                9: 0xAEED,
	                11: 0x50E2,
	                13: 0x10A0,
	                20: 0xCAA1,
	                31: 0xEEC0,
	                33: 0x5027,
	                37: 0x1A11,
	                43: 0xCAAE,
	                46: 0xCEEE,
	            },
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
	            ids=[0x1D1D],
	            maps={255: Ref(255)},
	            default=Ref(253),
	        )
		]
	)
