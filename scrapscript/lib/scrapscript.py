#!/usr/bin/env python3.10
import enum
from dataclasses import dataclass
from enum import auto
from typing import Callable, Mapping, Optional


class Lexer:
    def __init__(self, text: str):
        self.text: str = text
        self.idx: int = 0

    def has_input(self) -> bool:
        return self.idx < len(self.text)

    def read_char(self) -> str:
        c = self.peek_char()
        self.idx += 1
        return c

    def peek_char(self) -> str:
        if not self.has_input():
            raise ParseError("unexpected EOF while reading token")
        return self.text[self.idx]

    def read_one(self) -> Optional[str]:
        while self.has_input():
            c = self.read_char()
            if not c.isspace():
                break
        else:
            return None
        if c == '"':
            return self.read_string()
        if c == "-":
            if self.peek_char() == "-":
                self.read_comment()
                return self.read_one()
            if self.peek_char().isdigit():
                return self.read_number(c)
        if c.isdigit():
            return self.read_number(c)
        if c in OPER_CHARS:
            return self.read_op(c)
        if c.isidentifier():
            return self.read_var(c)
        tok = c
        while self.has_input() and not (c := self.read_char()).isspace():
            tok += c
        return tok

    def read_string(self) -> str:
        buf = ""
        while self.has_input():
            if (c := self.read_char()) == '"':
                break
            buf += c
        else:
            raise ParseError("unexpected EOF while reading string")
        return '"' + buf + '"'

    def read_comment(self) -> None:
        while self.has_input() and self.read_char() != "\n":
            pass

    def read_number(self, first_digit: str) -> str:
        buf = first_digit
        while self.has_input() and (c := self.peek_char()).isdigit():
            self.read_char()
            buf += c
        return buf

    def read_op(self, first_char: str) -> str:
        buf = first_char
        # TODO(max): To catch ill-formed operators earlier and to avoid merging
        # operators by accident, we could make a trie and do longest trie
        # match.
        while self.has_input() and (c := self.peek_char()) in OPER_CHARS:
            self.read_char()
            buf += c
        return buf

    def read_var(self, first_char: str) -> str:
        buf = first_char
        while self.has_input() and (c := self.peek_char()).isidentifier():
            self.read_char()
            buf += c
        return buf


def tokenize(x: str) -> list[str]:
    lexer = Lexer(x)
    tokens = []
    while lexer.has_input():
        token = lexer.read_one()
        if token is None:
            # EOF
            break
        tokens.append(token)
    return tokens


@dataclass(frozen=True)
class Prec:
    pl: float
    pr: float


def lp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n - 0.1)


def rp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n + 0.1)


def np(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n)


def xp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, 0)


PS = {
    "::": lp(2000),
    "": rp(1000),
    "*": lp(12),
    "/": lp(12),
    "//": lp(12),
    "%": lp(12),
    "+": lp(11),
    "-": lp(11),
    "**": rp(10),
    "==": np(9),
    "/=": np(9),
    "<": np(9),
    ">": np(9),
    "<=": np(9),
    ">=": np(9),
    ">*": rp(10),
    "++": rp(10),
    ">+": rp(10),
    "->": lp(5),
    "=": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    "]": xp(1),
}


OPER_CHARS = set("[" + "".join(PS.keys()))
assert " " not in OPER_CHARS


class ParseError(Exception):
    pass


def parse(tokens: list[str], p: float = 0) -> "Object":
    if not tokens:
        raise ParseError("unexpected end of input")
    token = tokens.pop(0)
    l: Object
    sha_prefix = "$sha1'"
    dollar_dollar_prefix = "$$"
    if token.isnumeric() or (token[0] == "-" and token[1:].isnumeric()):
        l = Int(int(token))
    elif token.isidentifier():
        l = Var(token)
    elif token.startswith(sha_prefix) and token[len(sha_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith(dollar_dollar_prefix) and token[len(dollar_dollar_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith('"') and token.endswith('"'):
        l = String(token[1:-1])
    elif token == "[":
        l = List([])
        token = tokens[0]
        if token == "]":
            tokens.pop(0)
        else:
            l.items.append(parse(tokens, 2))
            while tokens.pop(0) != "]":
                l.items.append(parse(tokens, 2))
    else:
        raise ParseError(f"unexpected token '{token}'")
    while True:
        if not tokens:
            break
        op = tokens[0]
        if op == ")" or op == "]":
            break
        if op not in PS:
            op = ""
        prec = PS[op]
        pl, pr = prec.pl, prec.pr
        if pl < p:
            break
        if op != "":
            tokens.pop(0)
        if op == "=":
            l = Assign(l, parse(tokens, pr))
        elif op == "->":
            l = Function(l, parse(tokens, pr))
        elif op == "":
            l = Apply(l, parse(tokens, pr))
        elif op == ".":
            l = Where(l, parse(tokens, pr))
        elif op == "?":
            l = Assert(l, parse(tokens, pr))
        else:
            l = Binop(BinopKind.from_str(op), l, parse(tokens, pr))
    return l


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Object:
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Int(Object):
    value: int


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class String(Object):
    value: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bool(Object):
    value: bool


Env = Mapping[str, Object]


class BinopKind(enum.Enum):
    ADD = auto()
    CONCAT = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "++": cls.CONCAT,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
            "==": cls.EQUAL,
            "/=": cls.NOT_EQUAL,
            "<": cls.LESS,
            ">": cls.GREATER,
            "<=": cls.LESS_EQUAL,
            ">=": cls.GREATER_EQUAL,
        }[x]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Binop(Object):
    op: BinopKind
    left: Object
    right: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class List(Object):
    items: list[Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assign(Object):
    name: Object
    value: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Function(Object):
    arg: Object
    body: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Apply(Object):
    func: Object
    arg: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Where(Object):
    first: Object
    second: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assert(Object):
    value: Object
    cond: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env


def eval_int(env: Env, exp: Object) -> int:
    result = eval(env, exp)
    if not isinstance(result, Int):
        raise TypeError(f"expected Int, got {type(result).__name__}")
    return result.value


def eval_str(env: Env, exp: Object) -> str:
    result = eval(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


BINOP_HANDLERS: dict[BinopKind, Callable[[Env, Object, Object], Object]] = {
    BinopKind.ADD: lambda env, x, y: Int(eval_int(env, x) + eval_int(env, y)),
    BinopKind.CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    BinopKind.SUB: lambda env, x, y: Int(eval_int(env, x) - eval_int(env, y)),
    BinopKind.MUL: lambda env, x, y: Int(eval_int(env, x) * eval_int(env, y)),
    BinopKind.DIV: lambda env, x, y: Int(eval_int(env, x) // eval_int(env, y)),
    # We have type: ignore because we haven't (re)defined eval yet.
    BinopKind.EQUAL: lambda env, x, y: Bool(eval(env, x) == eval(env, y)),  # type: ignore [arg-type]
}


# pylint: disable=redefined-builtin
def eval(env: Env, exp: Object) -> Object:
    if isinstance(exp, (Int, Bool, String, Function)):
        return exp
    if isinstance(exp, Var):
        value = env.get(exp.name)
        if value is None:
            raise NameError(f"name '{exp.name}' is not defined")
        return value
    if isinstance(exp, Binop):
        handler = BINOP_HANDLERS.get(exp.op)
        if handler is None:
            raise NotImplementedError(f"no handler for {exp.op}")
        return handler(env, exp.left, exp.right)
    if isinstance(exp, List):
        return List([eval(env, item) for item in exp.items])
    if isinstance(exp, Assign):
        # TODO(max): Rework this. There's something about matching that we need
        # to figure out and implement.
        assert isinstance(exp.name, Var)
        value = eval(env, exp.value)
        return EnvObject({**env, exp.name.name: value})
    if isinstance(exp, Where):
        res_env = eval(env, exp.second)
        assert isinstance(res_env, EnvObject)
        new_env = {**env, **res_env.env}
        return eval(new_env, exp.first)
    if isinstance(exp, Assert):
        cond = eval(env, exp.cond)
        if cond != Bool(True):
            raise AssertionError(f"condition {exp.cond} failed")
        return eval(env, exp.value)
    raise NotImplementedError(f"eval not implemented for {exp}")
