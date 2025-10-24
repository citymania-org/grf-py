import struct
import tempfile

from .ply import lex, yacc

from .common import utoi32
from .va2vars import VA2_VARS


OP_ADD = 0x00
OP_SUB = 0x01
OP_MIN = 0x02
OP_MAX = 0x03
OP_MINU = 0x04
OP_MAXU = 0x05
OP_DIV = 0x06
OP_MOD = 0x07
OP_DIVU = 0x08
OP_MODU = 0x09
OP_MUL = 0x0A
OP_AND = 0x0B
OP_OR = 0x0C
OP_XOR = 0x0D
OP_TSTO = 0x0E
OP_INIT = 0x0F
OP_PSTO = 0x10
OP_ROT = 0x11
OP_CMP = 0x12
OP_CMPU = 0x13
OP_SHL = 0x14
OP_SHRU = 0x15
OP_SHR = 0x16

OPE_LT, OPE_GT, OPE_LE, OPE_GE, OPE_EQ, OPE_NEQ, OPE_HASBIT, OPE_NHASBIT = range(-1, -9, -1)

NML_OPE = {
    OPE_LT: (OP_CMP, OP_MIN, 1, OP_XOR, 1),
    OPE_GT: (OP_CMP, OP_SUB, 1, OP_MAX, 0),
    OPE_LE: (OP_CMP, OP_XOR, 2, OP_MIN, 1),
    OPE_GE: (OP_CMP, OP_MIN, 1, None, None),
    OPE_EQ: (OP_CMP, OP_AND, 1, None, None),
    OPE_NEQ: (OP_CMP, OP_AND, 1, OP_XOR, 1),
    OPE_HASBIT: (OP_SHRU, OP_AND, 1, None, None),
    OPE_NHASBIT: (OP_SHRU, OP_AND, 1, OP_XOR, 1),
}

NML_OPE_PARSE = {}

for k, (a, b, bv, c, cv) in NML_OPE.items():
    root = NML_OPE_PARSE
    if c is not None:
        root = root.setdefault((c, cv), {})
    root = root.setdefault((b, bv), {})
    root[a] = k


OPERATORS = {
    OP_ADD: ('ADD', '{a} + {b}', 5, False),
    OP_SUB: ('SUB', '{a} - {b}', 5, True),
    OP_MIN: ('MIN', 'min({a}, {b})', 7, False),
    OP_MAX: ('MAX', 'max({a}, {b})', 7, False),
    OP_MINU: ('MINU', 'minu({a}, {b})', 7, False),
    OP_MAXU: ('MAXU', 'maxu({a}, {b})', 7, False),
    OP_DIV: ('DIV', '{a} / {b}', 6, False),
    OP_MOD: ('MOD', '{a} % {b}', 6, False),
    OP_DIVU: ('DIVU', '{a} u/ {b}', 6, False),
    OP_MODU: ('MODU', '{a} u% {b}', 6, False),
    OP_MUL: ('MUL', '{a} * {b}', 6, False),
    OP_AND: ('AND', '{a} & {b}', 4, True),
    OP_OR: ('OR', '{a} | {b}', 4, True),
    OP_XOR: ('XOR', '{a} ^ {b}', 4, True),
    OP_TSTO: ('TSTO', 'TEMP[0x{b:02x}] = {a}', 2, True),
    OP_INIT: ('INIT', None, 1, False),
    OP_PSTO: ('PSTO', 'PERM[0x{b:02x}] = {a}', 2, True),
    OP_ROT: ('ROT', 'rot({a}, {b})', 7, False),
    OP_CMP: ('CMP', 'cmp({a}, {b})', 7, False),
    OP_CMPU: ('CMPU', 'cmpu({a}, {b})', 7, False),
    OP_SHL: ('SHL', '{a} << {b}', 4, True),
    OP_SHRU: ('SHRU', '{a} +>> {b}', 4, True),
    OP_SHR: ('SHR', '{a} >> {b}', 4, True),

    OPE_LT: ('LT', '{a} < {b}', 3, False),
    OPE_GT: ('GT', '{a} > {b}', 3, False),
    OPE_LE: ('LE', '{a} <= {b}', 3, False),
    OPE_GE: ('GE', '{a} >= {b}', 3, False),
    OPE_EQ: ('EQ', '{a} == {b}', 3, False),
    OPE_NEQ: ('NEQ', '{a} != {b}', 3, False),
    OPE_HASBIT: ('HASBIT', 'hasbit({a}, {b})', 7, False),
    OPE_NHASBIT: ('NHASBIT', '!hasbit({a}, {b})', 7, False),
}
DEFAULT_INDENT_STR = '    '


def hex_str(s):
    if isinstance(s, (bytes, memoryview)):
        return ':'.join('{:02x}'.format(b) for b in s)
    return ':'.join('{:02x}'.format(ord(c)) for c in s)


def get_constant(value, error):
    if isinstance(value, int):
        return value
    if isinstance(value, Value):
        return value.value
    raise RuntimeError(error)


class Node:
    def __init__(self):
        pass

    def _make_node(self, value):
        if isinstance(value, int):
            return Value(value)
        assert isinstance(value, Node)
        return value

    def __add__(self, other):
        return Expr(OP_ADD, self, self._make_node(other))

    def __sub__(self, other):
        return Expr(OP_SUB, self, self._make_node(other))

    def store_temp(self, register):
        return Expr(OP_TSTO, self, self._make_node(register))

    def store_perm(self, register):
        return Expr(OP_PSTO, self, self._make_node(register))

    def format(self, parent_priority=0):
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


class Expr(Node):
    def __init__(self, op, a, b):
        self.op = op
        self.a = a
        self.b = b
        self.c = None

    def format(self, parent_priority=0):
        _, fmt, prio, bracket = OPERATORS[self.op]

        ares = self.a.format(prio - 1)
        if self.op in (OP_TSTO, OP_PSTO):
            bres = self.b.value
        else:
            bres = self.b.format(prio - int(not bracket))

        assert self.op != OP_INIT

        if self.c is not None:
            cres = self.c.format(prio - 1)
            res = fmt.format(a=ares, b=bres, c=cres)
        else:
            res = fmt.format(a=ares, b=bres)

        if prio <= parent_priority:
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


class AssignVariable(Node):
    def __init__(self, variable, expression):
        assert isinstance(variable, str)
        super().__init__()
        self.variable = variable
        self.expression = expression

    def format(self, parent_priority=0):
        return f'{self.variable} = {self.expresion.format()}'

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

    def format(self, parent_priority=0):
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

    def format(self, parent_priority=0):
        if self.param is not None:
            if isinstance(self.param, int):
                return  f'{self.name}({self.param})'
            else:
                return f'{self.name}({self.param.format()})'
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

    def format(self, parent_priority=0):
        addstr = ''
        if self.type == 1:
            addstr = f', add={self.add_val}, div={self.divmod_val}'
        elif self.type == 2:
            addstr = f', add={self.add_val}, mod={self.divmod_val}'
        param_str = '' if self.param is None else f', param=({self.param.format()})'
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

    def format(self, parent_priority=0):
        if isinstance(self.register, int):
            return  f'TEMP[0x{self.register:02x}]'
        return  f'TEMP[{self.register.format()}]'

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

    def format(self, parent_priority=0):
        return  f'PERM[{self.register.format()}]'

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
        assert 0 <= subroutine < 256, subroutine
        self.subroutine = subroutine

    def format(self, parent_priority=0):
        return f'call({self.subroutine})'

    def compile(self, context, shift=0, and_mask=0xffffffff):
        assert shift < 0x20, shift
        assert and_mask <= 0xffffffff, and_mask
        return True, struct.pack('<BBBI', 0x7e, self.subroutine, 0x20 | shift, and_mask)


class CallByName(Node):
    def __init__(self, subroutine: str):
        self.subroutine = subroutine

    def format(self, parent_priority=0):
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


tokens = (
    'NAME', 'NUMBER', 'NEWLINE',
    'ADD', 'SUB', 'MUL', 'DIV', 'MOD',
    'BINAND', 'BINOR', 'BINXOR', 'SHR', 'SHL', 'SHRU',
    'ASSIGN', 'COMMA',
    'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
    'LT', 'GT', 'LE', 'GE', 'EQ', 'NEQ',
)

# Tokens

t_ADD = r'\+'
t_SUB = r'-'
t_MUL = r'\*'
t_DIV = r'/'
t_MOD = r'%'
t_BINAND = r'\&'
t_BINOR = r'\|'
t_BINXOR = r'\^'
t_SHR = r'>>'
t_SHRU = r'\+>>'
t_SHL = r'<<'
t_ASSIGN = r'='
t_COMMA = r','
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LT = r'<'
t_GT = r'>'
t_LE = r'<='
t_GE = r'>='
t_EQ = r'=='
t_NEQ = r'!='
t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'


def t_NUMBER(t):
    r'0x[0-9a-fA-F]+|0b[01]+|\d+'
    # r'0x[0-9a-fA-F]+'
    try:
        if t.value.startswith('0x'):
            t.value = int(t.value, 16)
        elif t.value.startswith('0b'):
            t.value = int(t.value, 2)
        else:
            t.value = int(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t


# Ignored characters
t_ignore = ' \t'


def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")
    return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Parsing rules

precedence = (
    ('right', 'ASSIGN'),
    ('left', 'LT', 'GT', 'LE', 'GE', 'EQ', 'NEQ'),
    ('left', 'SHRU', 'SHR', 'SHL'),
    ('left', 'BINOR'),
    ('left', 'BINXOR'),
    ('left', 'BINAND'),
    ('left', 'ADD', 'SUB'),
    ('left', 'MUL', 'DIV', 'MOD'),
    ('right', 'UMINUS'),
)


def p_lines_next(t):
    '''lines : lines NEWLINE expression'''
    t[1].append(t[3])
    t[0] = t[1]


def p_lines_expression(t):
    '''lines : expression'''
    t[0] = [t[1]]


def p_lines_empty(t):
    '''lines :'''
    t[0] = []


def p_lines_trailing_newline(t):
    '''lines : lines NEWLINE'''
    t[0] = t[1]


def p_expression_number(t):
    '''expression : NUMBER'''
    t[0] = Value(int(t[1]))


def p_expression_negative_number(t):
    '''expression : SUB NUMBER %prec UMINUS'''
    t[0] = Value(-int(t[2]))


def p_expression_binop(t):
    '''expression : expression ADD expression
                  | expression SUB expression
                  | expression MUL expression
                  | expression DIV expression
                  | expression MOD expression
                  | expression BINAND expression
                  | expression BINOR expression
                  | expression BINXOR expression
                  | expression SHRU expression
                  | expression SHR expression
                  | expression SHL expression
                  | expression LT expression
                  | expression GT expression
                  | expression LE expression
                  | expression GE expression
                  | expression EQ expression
                  | expression NEQ expression
    '''
    op = {
        '+': OP_ADD,
        '-': OP_SUB,
        '*': OP_MUL,
        '/': OP_DIV,
        '%': OP_MOD,
        '&': OP_AND,
        '|': OP_OR,
        '^': OP_XOR,
        '>>': OP_SHR,
        '+>>': OP_SHRU,
        '<<': OP_SHL,
        '<': OPE_LT,
        '>': OPE_GT,
        '<=': OPE_LE,
        '>=': OPE_GE,
        '==': OPE_EQ,
        '!=': OPE_NEQ,
    }.get(t[2])

    assert op is not None, t[2]

    if op in NML_OPE:
        a, b, bv, c, cv = NML_OPE[op]
        res = Expr(a, t[1], t[3])
        res = Expr(b, res, Value(bv))
        if c is not None:
            res = Expr(c, res, Value(cv))
        t[0] = res
    else:
        # print(op, t[1], t[3])
        t[0] = Expr(op, t[1], t[3])


def p_expression_storage(t):
    'expression : NAME LBRACKET expression RBRACKET'
    assert t[1] in ('TEMP', 'PERM'), t[1]
    cls = Temp if t[1] == 'TEMP' else Perm
    register = t[3]
    t[0] = cls(register)

def p_expression_assign_storage(t):
    'expression : NAME LBRACKET expression RBRACKET ASSIGN expression'
    assert t[1] in ('TEMP', 'PERM'), t[1]
    op = OP_TSTO if t[1] == 'TEMP' else OP_PSTO
    register = t[3].const_eval()
    if register is None:
        raise NotImplementedError('Only constant register numbers are currently supported')
    t[0] = Expr(op, t[6], Value(register))


def p_expression_assign_variable(t):
    'expression : NAME ASSIGN expression'
    t[0] = AssignVariable(t[1], t[3])

def p_expression_call_0(t):
    'expression : NAME LPAREN RPAREN'
    # TODO error if function not found
    t[0] = CallByName(t[1])


def p_expression_call_1(t):
    'expression : NAME LPAREN NUMBER RPAREN'
    assert t[1] == 'call', t[1]
    t[0] = Call(int(t[3]))


def p_expression_call_2(t):
    'expression : NAME LPAREN expression COMMA expression RPAREN'
    op = {
        'min': OP_MIN,
        'max': OP_MAX,
        'minu': OP_MINU,
        'maxu': OP_MAXU,
        'rot': OP_ROT,
        'cmp': OP_CMP,
        'cmpu': OP_CMPU,
    }.get(t[1])
    assert op is not None, t[1]
    t[0] = Expr(op, t[3], t[5])


def parse_divmod(t, ofs):
    if len(t) <= ofs:
        return 0, None, None

    assert t[ofs] == 'add'
    add_val = t[ofs + 2].const_eval()
    if add_val is None:
        raise ValueError('`add` value must be a constant expression')
    divmod_val = t[ofs + 6].const_eval()
    if t[ofs + 4] == 'div':
        op_type = 1
        if divmod_val is None:
            raise ValueError('`div` value must be a constant expression')
    elif t[ofs + 4] == 'mod':
        op_type = 2
        if divmod_val is None:
            raise ValueError('`mod` value must be a constant expression')
    else:
        raise ValueError(t[ofs + 4])

    return op_type, add_val, divmod_val


def p_expression_var_call_param_shift_and_add_divmod(t):
    #                 1     2        3       4     5     6        7       8     9    10       11       12    13   14       15       16   17    18      19       20    21    22      23        24
    '''expression : NAME LPAREN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression RPAREN
                  | NAME LPAREN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression RPAREN'''
    assert t[1] == 'var'
    var = t[3].const_eval()
    if var is None:
        raise ValueError('Variable number must be a constant expression')
    assert t[5] == 'param' and t[9] == 'shift' and t[13] == 'and'
    shift = t[11].const_eval()
    if shift is None:
        raise ValueError('`shift` value must be a constant expression')
    and_mask = t[15].const_eval()
    if and_mask is None:
        raise ValueError('`and` value must be a constant expression')
    op_type, add_val, divmod_val = parse_divmod(t, 17)
    t[0] = GenericVar(var=var, shift=shift, and_mask=and_mask, param=t[7], type=op_type, add_val=add_val, divmod_val=divmod_val)


def p_expression_var_call_shift_and_add_divmod(t):
    #                 1     2        3       4     5     6        7       8     9    10       11       12   13    14       15      16    17    18       19       20
    '''expression : NAME LPAREN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression RPAREN
                  | NAME LPAREN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression COMMA NAME ASSIGN expression RPAREN'''
    assert t[1] == 'var'
    var = t[3].const_eval()
    if var is None:
        raise ValueError('Variable number must be a constant expression')
    assert t[5] == 'shift' and t[9] == 'and'
    shift = t[7].const_eval()
    if shift is None:
        raise ValueError('`shift` value must be a constant expression')
    and_mask = t[11].const_eval()
    if and_mask is None:
        raise ValueError('`and` value must be a constant expression')
    op_type, add_val, divmod_val = parse_divmod(t, 13)
    t[0] = GenericVar(var=var, shift=shift, and_mask=and_mask,
                      type=op_type, add_val=add_val, divmod_val=divmod_val)


def p_expression_group(t):
    'expression : LPAREN expression RPAREN'
    t[0] = t[2]


def p_expression_name(t):
    'expression : NAME'
    t[0] = Var(t.parser.grf_feature, t[1])


def p_error(t):
    if t is None:
        print("Unexpected syntax error")
        return

    s = t.lexer.lexdata
    pos = t.lexpos
    st = s.rfind('\n', 0, pos)
    et = s.find('\n', pos)
    s = s[st + 1: et] if et > 0 else s[st + 1: ]

    print(f'Syntax error at `{t.value}` line {t.lineno}:')
    print(s)
    print(' ' * (pos - st - 1) + '^')
    print(f'T={t}')


lexer = lex.lex()
parser = yacc.yacc(debug=False)

def parse_code(feature, code):
    parser.grf_feature = feature
    res = parser.parse(code, lexer=lexer)
    if res is None:
        stack_state_str = ' '.join([symbol.type for symbol in parser.symstack][1:])

        print('Syntax error in input! Parser State:{} {}'
              .format(parser.state,
                      stack_state_str))
    return res


SPRITE_FLAGS = {
    'dodraw': (None, 0x01, 1, None),
    'add': (None, 0x02, 1, Temp),
    'add_palette': (None, 0x04, 1, None),
    'custom_palette': (None, 0x08, 0, None),
    'delta_parent_xy': (True, 0x10, 2, None),
    'delta_parent_z': (True, 0x20, 1, None),
    'delta_child_x': (False, 0x10, 1, None),
    'delta_child_y': (False, 0x20, 1, None),
    'sprite_var10': (None, 0x40, 1, None),
    'palette_var10': (None, 0x80, 1, None),
}


if __name__ == "__main__":
    from .common import OBJECT
    res = parse_code(OBJECT, '''
        TEMP[0x00] = var(0x61, shift=0, and=0, add=1, div=1)
        ''')
        # TEMP[0x00] = var(0x61, param=(198), shift=0, and=0, add=1)
    # res = parse_code(OBJECT, '''
    #     TEMP[0x7f] = TEMP[0x1]
    #     TEMP[0x7f] = var(0x10, shift=11, and=12)
    #     TEMP[0x7f] = var(0x12, shift=11, and=12, add=1, div=3)
    #     TEMP[0x7f] = var(0x13, shift=11, and=12, add=2, mod=4)
    #     TEMP[128] = (cmp(tile_slope, 30) & 1) * 18
    #     TEMP[129] = (cmp(tile_slope, 29) & 1) * 15
    #     TEMP[130] = (cmp(tile_slope, 27) & 1) * 17
    #     TEMP[131] = (cmp(tile_slope, 23) & 1) * 16
    #     TEMP[132] = min(cmp(tile_slope, 0), 1)
    #     TEMP[134] = (min((cmp(tile_slope, 14) ^ 2), 1) & TEMP[132]) * tile_slope + TEMP[131] + TEMP[130] + TEMP[129] + TEMP[128]
    # ''')

    for line in res:
        print(line.format())
