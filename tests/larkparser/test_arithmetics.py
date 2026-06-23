import grf


def check_parser(code, expected):
	parser = grf.larkparser.Parser()
	node = parser.parse(code)
	assert isinstance(node, grf.larkparser.Node), type(node)
	result = node.format(full_bracket=True).strip()
	expected = expected.strip()
	assert result == expected


def check_parser_with_format(code, expected, bracketed):
	parser = grf.larkparser.Parser()
	node = parser.parse(code)
	assert isinstance(node, grf.larkparser.Node), type(node)
	result = node.format(full_bracket=True).strip()
	expected = expected.strip()
	assert result == expected
	result = node.format(full_bracket=False).strip()
	assert result == bracketed


def test_add_sub():
	check_parser_with_format(
		"1+2 - 3",
		"(1 + 2) - 3",
		"1 + 2 - 3")


def test_uminus():
	# there is no uminus in grf so -x is 0 - x
	check_parser_with_format(
		"1--1*-2",
		"1 - ((0 - 1) * (0 - 2))",
		"1 - (0 - 1) * (0 - 2)")


def test_mul_div_mod():
	check_parser_with_format(
		"1 + 2 * 3 + 4 / 2 % 1",
		"(1 + (2 * 3)) + ((4 / 2) % 1)",
		"1 + 2 * 3 + 4 / 2 % 1")

	check_parser_with_format(
		"(1 + 2) * 3 + (4 / 2) % 1",
		"((1 + 2) * 3) + ((4 / 2) % 1)",
		"(1 + 2) * 3 + 4 / 2 % 1",)


def test_bitwise_ops():
	check_parser_with_format(
		"1 & 2 ^ 3 | 4",
		"((1 & 2) ^ 3) | 4",
		"1 & 2 ^ 3 | 4")

	check_parser_with_format(
		"1 | 2 & 3 ^ 4",
		"1 | ((2 & 3) ^ 4)",
		"1 | 2 & 3 ^ 4")


def test_shift_ops():
	check_parser_with_format(
		"1 << 2 >> 3 >>> 4",
		"((1 << 2) >> 3) >>> 4",
		"1 << 2 >> 3 >>> 4")

def test_assign():
	check_parser_with_format(
		"name = 5 + 6 * 7",
		"name = 5 + (6 * 7)",
		"name = 5 + 6 * 7")

	check_parser_with_format(
		"TEMP[7] = 8",
		"TEMP[0x07] = 8",
		"TEMP[0x07] = 8")


def test_storage():
	check_parser_with_format(
		"a = TEMP[8]",
		"a = TEMP[8]",
		"a = TEMP[8]")


def test_param_access():
	check_parser_with_format(
		"PARAM.test",
		"PARAM.test",
		"PARAM.test")
