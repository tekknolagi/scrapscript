import enum
import re
from dataclasses import dataclass
from enum import auto
from typing import Callable, Mapping


def tokenize(x: str) -> list[str]:
    # TODO: Make this a proper tokenizer that handles strings with blankspace.
    stripped = re.sub(r" *--[^\n]*", "", x).strip()
    return re.split(r"[\s\n]+", stripped)


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
    ">*": rp(10),
    "++": rp(10),
    ">+": rp(10),
    "=": rp(4),
    ",": xp(1),
    "]": xp(1),
}


class ParseError(Exception):
    pass


def parse(tokens: list[str], p: float = 0) -> "Object":
    if not tokens:
        raise ParseError("unexpected end of input")
    token = tokens.pop(0)
    l: Object
    sha_prefix = "$sha1'"
    if token.isnumeric():
        l = Int(int(token))
    elif token.isidentifier():
        l = Var(token)
    elif token.startswith(sha_prefix) and token[len(sha_prefix) :].isidentifier():
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


Env = Mapping[str, Object]


class BinopKind(enum.Enum):
    ADD = auto()
    CONCAT = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "++": cls.CONCAT,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
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
}


# pylint: disable=redefined-builtin
def eval(env: Env, exp: Object) -> Object:
    if isinstance(exp, (Int, String)):
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
    raise NotImplementedError(f"eval not implemented for {exp}")
