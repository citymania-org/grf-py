from ..grf import SpriteGenerator
from ..actions import If, Label

from .parser import Parser
from .nodes import Value, Expr, AssignVariable, Call, CallByName, MinMax, Node, Temp, Perm, GetParamByName
from .nodes import (
    OPE_HASBIT,
    OPE_NHASBIT,
    OPE_EQ,
    OPE_NE,
    OPE_LE,
    OPE_GE,
)


class FuncSpritegen(SpriteGenerator):
    def __init__(self, func):
        self.func = func

    def get_sprites(self, g):
        return self.func(g)


class Code:
    parser = Parser()

    def __init__(self, code):
        self.root = Code.parser.parse(code)

    def compile_if(self, g, *, is_static, skip):
        assert isinstance(self.root, Expr), type(self.root)
        # NOTE: operations are inversed since it's a skip condition
        opcode = {
            OPE_NHASBIT: 0x00,
            OPE_HASBIT: 0x01,
            OPE_NE: 0x02,
            OPE_EQ: 0x03,
            OPE_GE: 0x04,
            OPE_LE: 0x05,
        }.get(self.root.op)

        if opcode is None:
            opstr = OPERATORS[self.op][0]
            raise NotImplementedError(f"Unsuported operation in Action 7/9(If): {opstr}")

        assert isinstance(self.root.a, GetParamByName)
        assert isinstance(self.root.b, Value)
        param_id = g.get_parameter_id(self.root.a.name)
        return [
            If(is_static=is_static, variable=param_id, condition=opcode, value=self.root.b.value, skip=skip, varsize=4)
        ]

    class If(SpriteGenerator):
        def __init__(self, code, body):
            self.code = Code(code)
            self.body = body

        def get_sprites(self, g):
            label_id = g.reserve_label()
            res = self.code.compile_if(g, is_static=False, skip=label_id)
            res.extend(self.body)
            # Release label when it's evaluated
            res.append(Label(label_id))
            res.append(FuncSpritegen(lambda g: [] and g.release_label(label_id)))
            return res
