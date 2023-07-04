from nose.tools import eq_

import grf


def encode_action(action, result):
	result = ''.join(result.split()).lower()
	encoded = action._encode()
	encoded_hex = grf.hex_str(encoded)
	eq_(encoded_hex, result)


def test_train_basic_properties():
	encode_action(
		grf.Define(
			feature=grf.TRAIN,
			id=0,
			props=dict(
				track_type=0x77,
				ai_special_flag=0xA1,
				max_speed=0x79ED,
				power=0x708E,
				running_cost_factor=0xC0,
				running_cost_base=0xC057BA5E,
				sprite_id=0x1D,
				dual_headed=True,
				cargo_capacity=0xCC,
				default_cargo_type=0xDC,
				weight_low=0xE1,
				weight_high=0xE2,
				cost_factor=0xCF,
				ai_engine_rank=0xAE,
				engine_class=0xEC,
				# sort_purchase_list=
				extra_power_per_wagon=0xE0EA,
				refit_cost=0x1C,
				refittable_cargo_types=0xE1C7,
				cb_flags=0xCB,
				tractive_effort_coefficient=0x7E,
				air_drag_coefficient=0xAD,
				shorten_by=0x5B,
				visual_effect_and_powered=0xE0,
				extra_weight_per_wagon=0xEE,
				bitmask_vehicle_info=0xB1,
				retire_early=0x1E,
				misc_flags=0x1F,
				refittable_cargo_classes=0xE1CC,
				non_refittable_cargo_classes=0x91CC,
				introduction_date=0xDA7E,
				cargo_age_period=0xCA6E,
				cargo_allow_refit=b'cargo_allow_refit',
				cargo_disallow_refit=b'cargo_disallow_refit',
				curve_speed_mod=0xC9ED,
				variant_group=0xA6,
				extra_flags=0xEAFA,
			)
		),
		'''
			00:00:24:01:ff:00:00:05:77:08:a1:09:ed:79:0b:8e:70:0d:c0:0e:5e:ba:57:c0:12:1d:13:01:14:
			cc:15:dc:16:e1:24:e2:17:cf:18:ae:19:ec:1b:ea:e0:1c:1c:1d:c7:e1:00:00:1e:cb:1f:7e:20:ad:
			21:5b:22:e0:23:ee:25:b1:26:1e:27:1f:28:cc:e1:29:cc:91:2a:7e:da:00:00:2b:6e:ca:2c:11:63:
			61:72:67:6f:5f:61:6c:6c:6f:77:5f:72:65:66:69:74:2d:14:63:61:72:67:6f:5f:64:69:73:61:6c:
			6c:6f:77:5f:72:65:66:69:74:2e:ed:c9:2f:a6:00:30:fa:ea:00:00
		''',
	)