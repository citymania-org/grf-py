from nose.tools import eq_

import grf
from grf.parser import parse_code


def hex_eq(value, result):
	result = ''.join(result.split()).lower()
	value_hex = grf.hex_str(value)
	eq_(value_hex, result)


def check_action(action, result):
	encoded = action._encode()
	hex_eq(encoded, result)


def test_switch_var_param():
	check_action(
		grf.Switch(
			feature=grf.TRAIN,
			ref_id=0,
			ranges={1: 0},
			default=0,
			code='var(0x62, param=(0xAA), shift=0x1F, and=0xAD123456)',
		),
		'02:00:00:89:62:aa:1f:56:34:12:ad:01:00:80:01:00:00:00:01:00:00:00:00:80',
	)


def check_code(feature, code, result):
	ast = parse_code(feature, code)
	eq_(len(ast), len(result))
	for a, r in zip(ast, result):
		is_value, data = a.compile(register=0x80)
		eq_(is_value, r[0])
		hex_eq(data, r[1])


def test_code_just_number():
	check_code(
		grf.TRAIN,
		'5',
		[(True, '1a:20:05:00:00:00')],
	)
	check_code(
		grf.TRAIN,
		'-5',
		[(True, '1a:20:fb:ff:ff:ff')],
	)


def test_code_var_param():
	check_code(
		grf.TRAIN,
		'var(0x62, param=0xAA, shift=0x1F, and=0xAD123456)',
		[(True, '62:aa:3f:56:34:12:ad')],
	)
	check_code(
		grf.TRAIN,
		'var(0x62, param=(0xAA), shift=0x1F, and=0xAD123456)',
		[(True, '62:aa:3f:56:34:12:ad')],
	)
