from grf import RoadVehicle, RV, Ref, DefineStrings, Define, Action3, Switch

from .common import check_lib


def test_lib_rv_basics():
	check_lib(
		None,
		RoadVehicle(
			id=0x1D1D,
			name='Test RV',
			max_speed=0xEE,
			weight=0xE1,
			callbacks={
				'graphics': 0xE0,
			}
		),
		[
	        DefineStrings(
	            feature=RV,
	            lang=127,
	            offset=7453,
	            is_generic_offset=False,
	            strings=[b'Test RV']
	        ),
	        Define(
	            feature=RV,
	            id=0x1D1D,
	            props={
	                'sprite_id': 255,
	                'precise_max_speed': 255,
	                'max_speed': 0xEE,
	                'weight': 225,
	            }
	        ),
	        Action3(
	            feature=RV,
	            wagon_override=False,
	            ids=[0x1D1D],
	            maps={},
	            default=0xE0,
	        )

		]
	)


def test_lib_rv_properties_cb():
	check_lib(
		None,
		RoadVehicle(
			id=0x1D1D,
			name='Test RV',
			max_speed=0xEE,
			weight=0xE1,
			callbacks={
				'graphics': 0xE0,
				'properties': {
					'loading_speed': 0x10AD,
					'running_cost_factor': 0x10A0,
					'cargo_capacity': 0xCAA1,
					'cost_factor': 0xC057,
					'power': 0x50E2,
					'weight': 0xE167,
					'max_speed': 0xAEED,
					'tractive_effort_coefficient': 0xEEC0,
					'cargo_age_period': 0xCAAE,
					'shorten_by': 0x5027,
				}
			}
		),
		[
	        DefineStrings(
	            feature=RV,
	            lang=127,
	            offset=7453,
	            is_generic_offset=False,
	            strings=[b'Test RV']
	        ),
	        Define(
	            feature=RV,
	            id=0x1D1D,
	            props={
	                'sprite_id': 255,
	                'precise_max_speed': 255,
	                'max_speed': 0xEE,
	                'weight': 225,
	            }
	        ),
	        Switch(
	            feature=RV,
	            ref_id=254,
	            related_scope=False,
	            code='extra_callback_info1_byte',
	            ranges={
	                0x07: 0x10AD,
	                0x09: 0x10A0,
	                0x0F: 0xCAA1,
	                0x11: 0xC057,
	                0x13: 0x50E2,
	                0x14: 0xE167,
	                0x15: 0xAEED,
	                0x18: 0xEEC0,
	                0x22: 0xCAAE,
	                0x23: 0x5027,
	            },
	            default=224,
	        ),
	        Switch(
	            feature=RV,
	            ref_id=255,
	            related_scope=False,
	            code='current_callback',
	            ranges={54: Ref(254)},
	            default=224,
	        ),
	        Switch(
	            feature=RV,
	            ref_id=253,
	            related_scope=False,
	            code='current_callback',
	            ranges={54: Ref(254)},
	            default=224,
	        ),
	        Action3(
	            feature=RV,
	            wagon_override=False,
	            ids=[0x1D1D],
	            maps={255: Ref(255)},
	            default=Ref(253),
	        )
		]
	)
