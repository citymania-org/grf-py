def utoi32(value):
    """
    Convert a 32-bit unsigned integer to a signed integer using two's complement.

    Args:
        value (int): The 32-bit unsigned integer.

    Returns:
        int: The corresponding signed integer.
    """
    return value | -(value & 0x80000000)


class Node:
    def __init__(self):
        pass

    def _make_node(self, value):
        if isinstance(value, int):
            return Value(value)
        assert isinstance(value, Node)
        return value

    # def __add__(self, other):
    #     return Expr(OP_ADD, self, self._make_node(other))

    # def __sub__(self, other):
    #     return Expr(OP_SUB, self, self._make_node(other))

    # def store_temp(self, register):
    #     return Expr(OP_TSTO, self, self._make_node(register))

    # def store_perm(self, register):
    #     return Expr(OP_PSTO, self, self._make_node(register))

    def format(self, parent_priority=0, full_bracket=False):
        raise NotImplementedError

    def __str__(self):
        return self.format()

    def __repr__(self):
        return self.format()

    def compile(self, context, shift=0, and_mask=0xffffffff):
        raise NotImplementedError

    def simplify(self):
        pass

    def const_eval(self):
        return None


OP_ADD = 0x00
OP_SUB = 0x01
OP_MINS = 0x02
OP_MAXS = 0x03
OP_MINU = 0x04
OP_MAXU = 0x05
OP_DIVS = 0x06
OP_MODS = 0x07
OP_DIVU = 0x08
OP_MODU = 0x09
OP_MUL = 0x0A
OP_BINAND = 0x0B
OP_BINOR = 0x0C
OP_BINXOR = 0x0D
OP_TSTO = 0x0E
OP_INIT = 0x0F
OP_PSTO = 0x10
OP_ROTU = 0x11
OP_CMPS = 0x12
OP_CMPU = 0x13
OP_SHL = 0x14
OP_SHRU = 0x15
OP_SHR = 0x16
OPE_LT, OPE_GT, OPE_LE, OPE_GE, OPE_EQ, OPE_NE, OPE_HASBIT, OPE_NHASBIT = range(-1, -9, -1)


OPERATORS = {
    OP_ADD: ('ADD', '{a} + {b}', 8, False),
    OP_SUB: ('SUB', '{a} - {b}', 8, False),
    OP_MINS: ('MINS', 'mins({a}, {b})', 10, True),
    OP_MAXS: ('MAXS', 'maxs({a}, {b})', 10, True),
    OP_MINU: ('MINU', 'minu({a}, {b})', 10, True),
    OP_MAXU: ('MAXU', 'maxu({a}, {b})', 10, True),
    OP_DIVS: ('DIVS', 'divs({a}, {b})', 10, True),
    OP_MODS: ('MODS', 'mods({a}, {b})', 10, True),
    OP_DIVU: ('DIVU', '{a} / {b}', 9, False),
    OP_MODU: ('MODU', '{a} % {b}', 9, False),
    OP_MUL: ('MUL', '{a} * {b}', 9, False),
    OP_BINAND: ('BINAND', '{a} & {b}', 7, False),
    OP_BINOR: ('BINOR', '{a} | {b}', 5, False),
    OP_BINXOR: ('BINXOR', '{a} ^ {b}', 6, False),
    OP_TSTO: ('TSTO', 'TEMP[0x{b:02x}] = {a}', 2, True),
    OP_INIT: ('INIT', None, 1, True),
    OP_PSTO: ('PSTO', 'PERM[0x{b:02x}] = {a}', 2, True),
    OP_ROTU: ('ROTU', 'rotu({a}, {b})', 10, True),
    OP_CMPS: ('CMPS', 'cmps({a}, {b})', 10, True),
    OP_CMPU: ('CMPU', 'cmpu({a}, {b})', 10, True),
    OP_SHL: ('SHL', '{a} << {b}', 4, False),
    OP_SHRU: ('SHRU', '{a} >>> {b}', 4, False),
    OP_SHR: ('SHR', '{a} >> {b}', 4, False),

    OPE_LT: ('LT', '{a} < {b}', 3, False),
    OPE_GT: ('GT', '{a} > {b}', 3, False),
    OPE_LE: ('LE', '{a} <= {b}', 3, False),
    OPE_GE: ('GE', '{a} >= {b}', 3, False),
    OPE_EQ: ('EQ', '{a} == {b}', 3, False),
    OPE_NE: ('NE', '{a} != {b}', 3, False),
    OPE_HASBIT: ('HASBIT', 'hasbit({a}, {b})', 7, True),
    OPE_NHASBIT: ('NHASBIT', '!hasbit({a}, {b})', 7, True),
}


class Expr(Node):

    def __init__(self, op, a, b):
        self.op = op
        self.a = a
        self.b = b
        self.c = None

    def format(self, *, parent_priority=0, full_bracket=False):
        _, fmt, prio, bracket = OPERATORS[self.op]

        ares = self.a.format(parent_priority=0 if bracket else prio, full_bracket=full_bracket)
        if self.op in (OP_TSTO, OP_PSTO):
            bres = self.b.value
        else:
            bres = self.b.format(parent_priority=0 if bracket else prio, full_bracket=full_bracket)

        assert self.op != OP_INIT

        if self.c is not None:
            cres = self.c.format(prio - 1)
            res = fmt.format(a=ares, b=bres, c=cres)
        else:
            res = fmt.format(a=ares, b=bres)

        if prio < parent_priority or (full_bracket and parent_priority > 0):
        # if prio <= parent_priority or (full_bracket):
            res = f'({res})'
        return res

    def compile(self, context, shift=0, and_mask=0xffffffff):
        is_value, b_code = self.b.compile(context, shift, and_mask)
        if is_value:
            # Second arg is a simple value, do operation directly
            res = self.a.compile(context, shift, and_mask)[1]
            res += bytes((self.op,))
            res += b_code
            return False, res

        # Calculate secord arg first and store in in a temp var
        res = b_code
        register = context.reserve_register()
        res += struct.pack('<BBBIB', OP_TSTO, 0x1a, 0x20, register, OP_INIT)
        res += self.a.compile(context, shift, and_mask)[1]
        res += struct.pack('<BBBBI', self.op, 0x7d, register, 0x20, 0xffffffff)
        context.release_register(register)
        return False, res

    def simplify(self):
        # if self.op != OP_MUL or not isinstance(self.a, Expr):
        #     return

        # node = self.a
        node = self
        is_final = False
        root = NML_OPE_PARSE

        while not is_final:
            if not isinstance(node.a, Expr) or not isinstance(node.b, Value):
                break
            val = root.get((node.op, node.b.value))
            if val is None:
                break
            root = val
            node = node.a

            if node.op in root:
                self.op = root[node.op]
                # self.c = self.b
                self.a = node.a
                self.b = node.b
                break

        self.a.simplify()
        self.b.simplify()
        if self.c is not None:
            self.c.simplify()

    def const_eval(self):
        # TODO support eval on const arguments (possibly on init)
        return None


class MinMax(Node):
    def __init__(self, op, args):
        assert len(args) > 1, args
        self.op = op
        self.args = args

    def format(self, full_bracket=False):
        args_str = ", ".join(a.format(full_bracket=full_bracket) for a in self.args)
        func = {
            OP_MINS: 'mins',
            OP_MAXS: 'maxs',
            OP_MAXU: 'maxu',
            OP_MINU: 'minu',
        }.get(self.op)

        return f'{func}({args_str})'


class AssignVariable(Node):
    def __init__(self, variable, expression):
        assert isinstance(variable, str), type(variable)
        super().__init__()
        self.variable = variable
        self.expression = expression

    def format(self, parent_priority=0, full_bracket=False):
        return f'{self.variable} = {self.expression.format(full_bracket=full_bracket)}'

    def compile(self, context, shift=0, and_mask=0xffffffff):
        is_value, code = self.expression.compile(context, shift, and_mask)
        register = context.reserve_variable(self.variable)
        # TODO return something better than false to avoid double temporary usage
        return False, code + struct.pack('<BBBI', OP_TSTO, 0x1a, 0x20, register)


class Value(Node):
    def __init__(self, value):
        assert isinstance(value, int)
        super().__init__()
        self.value = value

    def format(self, parent_priority=0, full_bracket=False):
        return str(utoi32(self.value))

    def compile(self, context, shift=0, and_mask=0xffffffff):
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        valueadj = (self.value >> shift) & and_mask
        return True, struct.pack('<BBI', 0x1a, 0x20, valueadj)

    def const_eval(self):
        return self.value


class Var(Node):
    def __init__(self, feature, name, param=None):
        assert feature in VA2_VARS, feature
        super().__init__()
        self.feature = feature
        self.name = name
        self.param = param

    def format(self, parent_priority=0, full_bracket=False):
        if self.param is not None:
            if isinstance(self.param, int):
                return  f'{self.name}({self.param})'
            else:
                return f'{self.name}({self.param.format(full_bracket=full_bracket)})'
        return self.name

    def compile(self, context, shift=0, and_mask=0xffffffff):
        var_data = VA2_VARS[self.feature].get(self.name)

        if var_data is None:
            # TODO prevent name conflicts between temporary and feature vars
            register = context.get_variable_register(self.name)
            if register is None:
                raise ValueError(f'Unknown variable `{self.name}`')

            return True, struct.pack('<BBBI', 0x7d, register, 0x20 | shift, and_mask)

        var_mask = (1 << var_data['size']) - 1
        and_mask &= var_mask >> shift
        shift += var_data['start']
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        # TODO dynamic param (var 7B)
        if 'param' in var_data:
            return True, struct.pack('<BBBI', var_data['var'], var_data['param'], 0x20 | shift, and_mask)
        else:
            return True, struct.pack('<BBI', var_data['var'], 0x20 | shift, and_mask)

    def simplify(self):
        if self.param is not None:
            self.param.simplify()


class GenericVar(Node):
    def __init__(self, var, shift, and_mask, type=0, add_val=None, divmod_val=None, param=None):
        assert type == 0 or add_val is not None
        assert type == 0 or divmod_val is not None

        self.var = var
        self.shift = shift
        self.and_mask = and_mask
        self.type = type
        self.add_val = add_val
        self.divmod_val = divmod_val
        self.param = param

    def format(self, parent_priority=0, full_bracket=False):
        addstr = ''
        if self.type == 1:
            addstr = f', add={self.add_val}, div={self.divmod_val}'
        elif self.type == 2:
            addstr = f', add={self.add_val}, mod={self.divmod_val}'
        param_str = '' if self.param is None else f', param=({self.param.format(full_bracket=full_bracket)})'
        return f'var(0x{self.var:02x}{param_str}, shift={self.shift}, and=0x{self.and_mask:x}{addstr})'

    def simplify(self):
        if self.param is not None:
            self.param.simplify()

    def compile(self, context, shift=0, and_mask=0xffffffff):
        and_mask &= self.and_mask >> shift
        shift += self.shift
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        if self.add_val is not None or self.divmod_val is not None: raise NotImplementedError
        if self.param is not None:
            param = self.param.const_eval()
            if param is None:
                code = self.param.compile(context)[1]
                return True, code + struct.pack(
                    '<BBBBI',
                    OP_INIT, 0x7b, self.var,
                    0x20 | shift, and_mask
                )
            elif param > 255:
                return True, struct.pack(
                    '<BBIBBBBI',
                    0x1a, 0x20, param,  # val1 = param
                    OP_INIT, 0x7b, self.var, 0x20 | shift, and_mask,  # val1 = var(var=self.var, param=val1)
                )
            else:
                return True, struct.pack('<BBBI', self.var, param, 0x20 | shift, and_mask)
        else:
            return True, struct.pack('<BBI', self.var, 0x20 | shift, and_mask)


class Temp(Node):
    def __init__(self, register):
        assert isinstance(register, (int, Node)), type(register)
        super().__init__()
        self.register = register

    def get_index(self):
        return get_constant(self.register, 'Register number should be a simple value, not expression')

    def format(self, parent_priority=0, full_bracket=False):
        if isinstance(self.register, int):
            return  f'TEMP[0x{self.register:02x}]'
        return  f'TEMP[{self.register.format(full_bracket=full_bracket)}]'

    def compile(self, context, shift=0, and_mask=0xffffffff):
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        if isinstance(self.register, int):
            return True, struct.pack('<BBBI', 0x7d, self.register, 0x20 | shift, and_mask)
        elif isinstance(self.register, Value):
            return True, struct.pack('<BBBI', 0x7d, self.register.value, 0x20 | shift, and_mask)
        else:
            code = self.register.compile(context)[1]
            # 7b takes parameter 7d as var to access and uses stack value as it's parameter
            return False, code + struct.pack('<BBBBI', OP_INIT, 0x7b, 0x7d, 0x20 | shift, and_mask)

    def __repr__(self):
        return f'Temp({self.register!r})'

    def simplify(self):
        if self.register is not None:
            self.register.simplify()


class Perm(Node):
    def __init__(self, register):
        assert isinstance(register, Node), type(register)
        super().__init__()
        self.register = register

    def format(self, parent_priority=0, full_bracket=False):
        return  f'PERM[{self.register.format(full_bracket=full_bracket)}]'

    def compile(self, context, shift=0, and_mask=0xffffffff):
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        if isinstance(self.register, int):
            register = self.register
        elif isinstance(self.register, Value):
            register = self.register.value
        else:
            code = self.register.compile(context)[1]
            # 7b takes parameter 7c as var to access and uses stack value as it's parameter
            return False, code + struct.pack('<BBBBI', OP_INIT, 0x7b, 0x7c, 0x20 | shift, and_mask)

        return True, struct.pack('<BBBI', 0x7c, register, 0x20 | shift, and_mask)

    def __repr__(self):
        return f'Perm({self.register!r})'

    def simplify(self):
        if self.register is not None:
            self.register.simplify()


class Call(Node):
    def __init__(self, subroutine):
        # assert 0 <= subroutine < 256, subroutine
        self.subroutine = subroutine

    def format(self, parent_priority=0, full_bracket=False):
        return f'call({self.subroutine})'

    def compile(self, context, shift=0, and_mask=0xffffffff):
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        return True, struct.pack('<BBBI', 0x7e, self.subroutine, 0x20 | shift, and_mask)


class CallByName(Node):
    def __init__(self, subroutine: str):
        self.subroutine = subroutine

    def format(self, parent_priority=0, full_bracket=False):
        return f'{self.subroutine}()'

    def compile(self, context, shift=0, and_mask=0xffffffff):

        # TODO duplicate from actions.py
        def get_ref_id(ref_obj):
            # if isinstance(ref_obj, Ref):
            #     return ref_obj.value
            if isinstance(ref_obj, int):
                return ref_obj | 0x8000
            return ref_obj.ref_id

        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mas

        # TODO move to parsing stage
        if context.subroutines is None or self.subroutine not in context.subroutines:
            raise RuntimeError(f'Calling unknown subroutine `{self.subroutine}`')

        subroutine_id = get_ref_id(context.subroutines[self.subroutine])
        return True, struct.pack('<BBBI', 0x7e, subroutine_id, 0x20 | shift, and_mask)


class GetParamByName(Node):
    def __init__(self, name: str):
        self.name = name

    def format(self, parent_priority=0, full_bracket=False):
        return f'PARAM.{self.name}'
