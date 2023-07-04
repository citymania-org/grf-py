from nose.tools import eq_

import grf


def encode_action(action, result):
	result = ''.join(result.split()).lower()
	encoded = action._encode()
	encoded_hex = grf.hex_str(encoded)
	eq_(encoded_hex, result)


def test_switch_var_param():
	encode_action(
		grf.Switch(
			feature=grf.TRAIN,
			ref_id=0,
			ranges={1: 0},
			default=0,
			code='var(0x62, param=(0xAA), shift=0x1F, and=0xAD123456)',
		),
		'02:00:00:89:62:aa:1f:56:34:12:ad:01:00:80:01:00:00:00:01:00:00:00:00:80',
	)