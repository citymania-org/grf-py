import pathlib

import lark

from .nodes import Value, Expr, AssignVariable, Call, CallByName, MinMax, Node, Temp, Perm, GetParamByName
from .nodes import (
    OP_ADD, OP_SUB,
    OP_MINS, OP_MAXS, OP_MINU, OP_MAXU,
    OP_DIVS, OP_MODS, OP_DIVU, OP_MODU,
    OP_MUL, OP_BINAND, OP_BINOR, OP_BINXOR,
    OP_TSTO, OP_INIT, OP_PSTO, OP_ROTU,
    OP_CMPS, OP_CMPU, OP_SHL, OP_SHRU, OP_SHR,
    OPE_LT, OPE_GT, OPE_LE, OPE_GE, OPE_EQ, OPE_NE,
    OPE_HASBIT, OPE_NHASBIT
)

class SelfVar(Node):
    def __init__(self, name):
        self.name = name

    def format(self, parent_priority=0):
        return f'self.{self.name}'


class RelatedVar(Node):
    def __init__(self, name):
        self.name = name

    def format(self, parent_priority=0):
        return f'related.{self.name}'


class ParseError(Exception):
    def __init__(self, message, line=None, column=None):
        super().__init__(f"{message} (line {line}, column {column})")


@lark.v_args(inline=True)  # pass tokens directly as args
class Transformer(lark.Transformer):
    def hex(self, tok):
        return Value(int(tok.value, 16))

    def bin(self, tok):
        return Value(int(tok.value, 2))

    def dec(self, tok):
        return Value(int(tok.value))

    def _expr(self, op, left, right):
        return Expr(op, left, right)

    def name(self, value):
        return value

    # precedence 0

    def assign_var(self, name, expr):
        # right-associative
        return AssignVariable(name, expr)

    def assign_temp(self, register, expr):
        # right-associative
        assert isinstance(register, Value)
        return Expr(OP_TSTO, expr, register)

    def assign_perm(self, register, expr):
        # right-associative
        assert isinstance(register, Value)
        return Expr(OP_PSTO, expr, register)

    # precedence 1

    op_lt = lambda self, *args: self._expr(OPE_LT, *args)
    op_gt = lambda self, *args: self._expr(OPE_GT, *args)
    op_le = lambda self, *args: self._expr(OPE_LE, *args)
    op_ge = lambda self, *args: self._expr(OPE_GE, *args)
    op_eq = lambda self, *args: self._expr(OPE_EQ, *args)
    op_ne = lambda self, *args: self._expr(OPE_NE, *args)

    # precedence 2

    op_shr = lambda self, *args: self._expr(OP_SHR, *args)
    op_shru = lambda self, *args: self._expr(OP_SHRU, *args)
    op_shl = lambda self, *args: self._expr(OP_SHL, *args)

    # precedence 3

    op_binor = lambda self, *args: self._expr(OP_BINOR, *args)

    # precedence 4

    op_binxor = lambda self, *args: self._expr(OP_BINXOR, *args)

    # precedence 5

    op_binand = lambda self, *args: self._expr(OP_BINAND, *args)

    # precedence 6

    op_add = lambda self, *args: self._expr(OP_ADD, *args)
    op_sub = lambda self, *args: self._expr(OP_SUB, *args)

    # precedence 7

    op_mul = lambda self, *args: self._expr(OP_MUL, *args)
    op_divu = lambda self, *args: self._expr(OP_DIVU, *args)
    op_modu = lambda self, *args: self._expr(OP_MODU, *args)

    # precedence 8

    def uminus(self, expr):
        return Expr(OP_SUB, Value(0), expr)

    def read_var(self, name):
        return name

    def read_temp(self, register):
        return Temp(register)

    def read_perm(self, register):
        return Perm(register)

    def read_param_by_name(self, name):
        return GetParamByName(name)

    def call(self, name, *args):
        op = {
            'divs': OP_DIVS,
            'mods': OP_MODS,
            'cmpu': OP_CMPU,
            'cmps': OP_CMPS,
            'rotu': OP_ROTU,
            'shrs': OP_SHRS,
        }.get(name)
        if op is not None:
            assert len(args) == 2, args
            return self._expr(op, *args)

        op = {
            'minu': OP_MINU,
            'maxu': OP_MAXU,
            'mins': OP_MINS,
            'maxs': OP_MAXS,
        }.get(name)
        if op is not None:
            return MinMax(op, args)

        if name == "call":
            assert len(args) == 1, args
            return Call(args[0])

        assert len(args) == 1 and args[0] is None, args  # TODO arbitrary args
        return CallByName(name)

    def self_var(self, name):
        return SelfVar(name)

    def related_var(self, name):
        return RelatedVar(name)

    def start(self, *args):
        return args


class Parser:
    def __init__(self, *, debug=False):
        grammar_file = pathlib.Path(__file__).parent / "grammar.lark"
        self.parser = lark.Lark.open(grammar_file, parser="lalr", debug=debug, transformer=Transformer())

    def parse(self, code):
        try:
            return self.parser.parse(code)
        except lark.UnexpectedInput as e:
            print("Parse error!", e)
            print(e.get_context(code, span=40))
            raise


# Parse something
# code = """
# x = 3 + 4 * (2 - 1)
# y = 6 + -5
# z = minu(5, 5, 3)
# call(1)
# subroutine()
# self.test_var + related.var_var
# test.test
# """

# parser = Parser(debug=True)
# try:
#     tree = parser.parse(code)
#     for l in tree:
#         print(l)
# except lark.UnexpectedInput as e:
#     print("Parse error!", e)
#     print(e.get_context(code, span=40))


# ahyangyi_test_1 = """
#     rc = self.rail_continuation
#     TEMP[0x10] = rc[0, 1] == 0
#     TEMP[0x11] = rc[0, -1] == 0
#     TEMP[0x12] = self.get_land_info(1, 0).grf_type == GRFType::Current
#     TEMP[0x13] = self.get_land_info(-1, 0).grf_type == GRFType::Current
# """
#     Parameter(
#         "NIGHT_MODE",
#         0,
#         {0: "AUTO_DETECT", 1: "ENABLED", 2: "DISABLED"},
#         mapping=ParameterMapping(grf_parameter=0xF, first_bit=0, num_bit=3),
#     )
# ]


    # g.add(
    #     grf.ComputeParameters(
    #         target=0x40, operation=0x00, if_undefined=False, source1=0x11, source2=0xFE, value=b"\xff\xff\x00\x00"
    #     )
    # )

    # nightgfx_id = struct.unpack("<I", b"\xffOTN")[0]
    # PARAM[0x41] = 1
    # if grf_is_active(nightgfx_id) {

    # }
    # g.add(grf.ComputeParameters(target=0x41, operation=0x00, if_undefined=False, source1=0xFF, source2=0xFF, value=1))
    # g.add(grf.If(is_static=False, variable=0x88, condition=0x06, value=nightgfx_id, skip=1, varsize=4))
    # g.add(grf.ComputeParameters(target=0x41, operation=0x00, if_undefined=False, source1=0xFF, source2=0xFF, value=0))

ahyangyi_test_2 = """
rail_continuation_s = self.rail_continuation[0, 1] == 0
rail_continuation_n = self.rail_continuation[0, -1] == 0
night = PARAM[15] & 0x7
night = (night == 0) * PARAM[0x41] + (night == 1)  # 1 if enabled or autodect and nightgfx present (param 41)
snow_nightgfx = night * ((self.terrain_type & 0x4) == 0x4)
rail_continuation_s_nightgfx = night * ((self.rail_continuation[0, 1]) == 0)
rail_continuation_n_nightgfx = night * ((self.rail_continuation[0, -1]) == 0)
"""

a = """
if PARAM[0x41]:
    <layout1>
else:
    <layout2>
"""

# Action0(
#     length=grf.Eval('if PARAM.full_size { 8 } else { 7 }'),
#     length=grf.Eval('8 if PARAM.full_size else 7'),
# )

# vehicle
# 0x60 (count vehicle id): param - id
# 0x61 (var of n-th vehicle): param - variable, 0x10f - offset, 0x10e - new param
# 0x62 (curvature of n-th vehicle):  param - offset
# 0x63 (track type test): param - track type

# station
# 0x60 ()

# https://github.com/OpenTTD/OpenTTD/pull/14224
# All callbacks returning D0xx strings, have now the option to return any string id via register 0x100

# https://github.com/OpenTTD/OpenTTD/pull/14149
# Add explicit (Var)Action2 results for "callback failed" and "calculated result"

# heavy 7F usage
# https://github.com/OpenTTD-China-Set/China-Set-Stations-Wuhu/blob/nightgfx/station/lib/registers.py

# station var 68
# https://github.com/OpenTTD-China-Set/China-Set-Stations-Wuhu/blob/remove-redundant-escalators/station/lib/registers.py#L16
