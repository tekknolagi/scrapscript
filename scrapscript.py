#!/usr/bin/env python3.10
import argparse
import base64
import code
import dataclasses
import enum
import http.server
import json
import logging
import os
import re
import socketserver
import sys
import typing
import unittest
import urllib.request
import urllib.parse
from dataclasses import dataclass
from enum import auto
from types import ModuleType, FunctionType
from typing import Any, Callable, Dict, Mapping, Optional, Union

readline: Optional[ModuleType]
try:
    import readline
except ImportError:
    readline = None


logger = logging.getLogger(__name__)


def is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in ("$", "'", "_")


@dataclass(eq=True)
class Token:
    lineno: int = dataclasses.field(default=-1, init=False, compare=False)


@dataclass(eq=True)
class NumLit(Token):
    value: int


@dataclass(eq=True)
class StringLit(Token):
    value: str


@dataclass(eq=True)
class BytesLit(Token):
    value: str
    base: int


@dataclass(eq=True)
class Operator(Token):
    value: str


@dataclass(eq=True)
class Name(Token):
    value: str


@dataclass(eq=True)
class LeftParen(Token):
    # (
    pass


@dataclass(eq=True)
class RightParen(Token):
    # )
    pass


@dataclass(eq=True)
class LeftBrace(Token):
    # {
    pass


@dataclass(eq=True)
class RightBrace(Token):
    # }
    pass


@dataclass(eq=True)
class LeftBracket(Token):
    # [
    pass


@dataclass(eq=True)
class RightBracket(Token):
    # ]
    pass


@dataclass(eq=True)
class Juxt(Token):
    # The space between other tokens that indicates function application.
    pass


@dataclass(eq=True)
class EOF(Token):
    pass


class Lexer:
    def __init__(self, text: str):
        self.text: str = text
        self.idx: int = 0
        self.lineno: int = 1
        self.colno: int = 1
        self.line: str = ""

    def has_input(self) -> bool:
        return self.idx < len(self.text)

    def read_char(self) -> str:
        c = self.peek_char()
        if c == "\n":
            self.lineno += 1
            self.colno = 1
            self.line = ""
        else:
            self.line += c
            self.colno += 1
        self.idx += 1
        return c

    def peek_char(self) -> str:
        if not self.has_input():
            raise UnexpectedEOFError("while reading token")
        return self.text[self.idx]

    def make_token(self, cls: type, *args: Any) -> Token:
        result: Token = cls(*args)
        result.lineno = self.lineno
        return result

    def read_one(self) -> Token:
        while self.has_input():
            c = self.read_char()
            if not c.isspace():
                break
        else:
            return self.make_token(EOF)
        if c == '"':
            return self.read_string()
        if c == "-":
            if self.has_input() and self.peek_char() == "-":
                self.read_comment()
                return self.read_one()
            return self.read_op(c)
        if c == "~":
            if self.has_input() and self.peek_char() == "~":
                self.read_char()
                return self.read_bytes()
            raise ParseError(f"unexpected token {c!r}")
        if c.isdigit():
            return self.read_number(c)
        if c in "()[]{}":
            custom = {
                "(": LeftParen,
                ")": RightParen,
                "{": LeftBrace,
                "}": RightBrace,
                "[": LeftBracket,
                "]": RightBracket,
            }
            return self.make_token(custom[c])
        if c in OPER_CHARS:
            return self.read_op(c)
        if is_identifier_char(c):
            return self.read_var(c)
        raise ParseError(f"unexpected token {c!r}", ("<input>", self.lineno, self.colno, self.line))

    def read_string(self) -> Token:
        buf = ""
        while self.has_input():
            if (c := self.read_char()) == '"':
                break
            buf += c
        else:
            raise UnexpectedEOFError("while reading string")
        return self.make_token(StringLit, buf)

    def read_comment(self) -> None:
        while self.has_input() and self.read_char() != "\n":
            pass

    def read_number(self, first_digit: str) -> Token:
        buf = first_digit
        while self.has_input() and (c := self.peek_char()).isdigit():
            self.read_char()
            buf += c
        return self.make_token(NumLit, int(buf))

    def _starts_operator(self, buf: str) -> bool:
        # TODO(max): Rewrite using trie
        return any(op.startswith(buf) for op in PS.keys())

    def read_op(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input():
            c = self.peek_char()
            if not self._starts_operator(buf + c):
                break
            self.read_char()
            buf += c
        if buf in PS.keys():
            return self.make_token(Operator, buf)
        raise ParseError(f"unexpected token {buf!r}")

    def read_var(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input() and is_identifier_char(c := self.peek_char()):
            self.read_char()
            buf += c
        return self.make_token(Name, buf)

    def read_bytes(self) -> Token:
        buf = ""
        while self.has_input():
            if (c := self.read_char()).isspace():
                break
            buf += c
        base, _, value = buf.rpartition("'")
        return self.make_token(BytesLit, value, int(base) if base else 64)


def tokenize(x: str) -> typing.List[Token]:
    lexer = Lexer(x)
    tokens = []
    while (token := lexer.read_one()) and not isinstance(token, EOF):
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
    "@": rp(1001),
    "": rp(1000),
    ">>": lp(14),
    "<<": lp(14),
    "^": rp(13),
    "*": lp(12),
    "/": lp(12),
    "//": lp(12),
    "%": lp(12),
    "+": lp(11),
    "-": lp(11),
    ">*": rp(10),
    "++": rp(10),
    ">+": lp(10),
    "+<": rp(10),
    "==": np(9),
    "/=": np(9),
    "<": np(9),
    ">": np(9),
    "<=": np(9),
    ">=": np(9),
    "&&": rp(8),
    "||": rp(7),
    "|>": rp(6),
    "<|": lp(6),
    "->": lp(5),
    "|": rp(4.5),
    ":": lp(4.5),
    "=": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    # TODO: Fix precedence for spread
    "...": xp(0),
}


HIGHEST_PREC: float = max(max(p.pl, p.pr) for p in PS.values())


OPER_CHARS = set("".join(PS.keys()))
assert " " not in OPER_CHARS


class ParseError(SyntaxError):
    pass


# TODO(max): Replace with EOFError?
class UnexpectedEOFError(ParseError):
    pass


def parse_assign(tokens: typing.List[Token], p: float = 0) -> "Assign":
    assign = parse(tokens, p)
    if not isinstance(assign, Assign):
        raise ParseError("failed to parse variable assignment in record constructor")
    return assign


def parse(tokens: typing.List[Token], p: float = 0) -> "Object":
    if not tokens:
        raise UnexpectedEOFError("unexpected end of input")
    token = tokens.pop(0)
    l: Object
    if isinstance(token, NumLit):
        l = Int(token.value)
    elif isinstance(token, Name):
        # TODO: Handle kebab case vars
        l = Var(token.value)
    elif isinstance(token, BytesLit):
        base = token.base
        if base == 85:
            l = Bytes(base64.b85decode(token.value))
        elif base == 64:
            l = Bytes(base64.b64decode(token.value))
        elif base == 32:
            l = Bytes(base64.b32decode(token.value))
        elif base == 16:
            l = Bytes(base64.b16decode(token.value))
        else:
            raise ParseError(f"unexpected base {base!r} in {token!r}")
    elif isinstance(token, StringLit):
        l = String(token.value)
    elif token == Operator("|"):
        expr = parse(tokens, PS["|"].pr)  # TODO: make this work for larger arities
        if not isinstance(expr, Function):
            raise ParseError(f"expected function in match expression {expr!r}")
        cases = [MatchCase(expr.arg, expr.body)]
        while tokens and tokens[0] == Operator("|"):
            tokens.pop(0)
            expr = parse(tokens, PS["|"].pr)  # TODO: make this work for larger arities
            if not isinstance(expr, Function):
                raise ParseError(f"expected function in match expression {expr!r}")
            cases.append(MatchCase(expr.arg, expr.body))
        l = MatchFunction(cases)
    elif isinstance(token, LeftParen):
        if isinstance(tokens[0], RightParen):
            l = Hole()
        else:
            l = parse(tokens)
        tokens.pop(0)
    elif isinstance(token, LeftBracket):
        l = List([])
        token = tokens[0]
        if isinstance(token, RightBracket):
            tokens.pop(0)
        else:
            l.items.append(parse(tokens, 2))
            while not isinstance(tokens.pop(0), RightBracket):
                # TODO: Implement .. and ... operators
                l.items.append(parse(tokens, 2))
    elif isinstance(token, LeftBrace):
        l = Record({})
        token = tokens[0]
        if isinstance(token, RightBrace):
            tokens.pop(0)
        else:
            assign = parse_assign(tokens, 2)
            l.data[assign.name.name] = assign.value
            while not isinstance(tokens.pop(0), RightBrace):
                # TODO: Implement .. and ... operators
                assign = parse_assign(tokens, 2)
                l.data[assign.name.name] = assign.value
    elif token == Operator("-"):
        # Unary minus
        # Precedence was chosen to be higher than binary ops so that -a op
        # b is (-a) op b and not -(a op b).
        # Precedence was chosen to be higher than function application so that
        # -a b is (-a) b and not -(a b).
        r = parse(tokens, HIGHEST_PREC + 1)
        l = Binop(BinopKind.SUB, Int(0), r)
    else:
        raise ParseError(f"unexpected token {token!r}")

    while True:
        if not tokens:
            break
        op = tokens[0]
        if isinstance(op, (RightParen, RightBracket, RightBrace)):
            break
        if not isinstance(op, Operator):
            prec = PS[""]
            pl, pr = prec.pl, prec.pr
            if pl < p:
                break
            l = Apply(l, parse(tokens, pr))
            continue
        prec = PS[op.value]
        pl, pr = prec.pl, prec.pr
        if pl < p:
            break
        tokens.pop(0)
        if op == Operator("="):
            if not isinstance(l, Var):
                raise ParseError(f"expected variable in assignment {l!r}")
            l = Assign(l, parse(tokens, pr))
        elif op == Operator("->"):
            l = Function(l, parse(tokens, pr))
        elif op == Operator("|>"):
            l = Apply(parse(tokens, pr), l)
        elif op == Operator("<|"):
            l = Apply(l, parse(tokens, pr))
        elif op == Operator(">>"):
            l = Compose(l, parse(tokens, pr))
        elif op == Operator("<<"):
            l = Compose(parse(tokens, pr), l)
        elif op == Operator("."):
            l = Where(l, parse(tokens, pr))
        elif op == Operator("?"):
            l = Assert(l, parse(tokens, pr))
        elif op == Operator("@"):
            # TODO: revisit whether to use @ or . for field access
            l = Access(l, parse(tokens, pr))
        else:
            assert not isinstance(op, Juxt)
            assert isinstance(op, Operator)
            l = Binop(BinopKind.from_str(op.value), l, parse(tokens, pr))
    return l


OBJECT_DESERIALIZERS: Dict[str, FunctionType] = {}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Object:
    def __init_subclass__(cls, /, **kwargs: Dict[Any, Any]) -> None:
        super().__init_subclass__(**kwargs)
        deserializer = getattr(cls, "_deserialize", None)
        if deserializer:
            OBJECT_DESERIALIZERS[cls.__name__] = deserializer

    def serialize(self) -> Dict[bytes, object]:
        cls = type(self)
        result: Dict[bytes, object] = {b"type": cls.__name__.encode("utf-8")}
        for field in dataclasses.fields(cls):
            if issubclass(field.type, Object):
                value = getattr(self, field.name)
                result[field.name.encode("utf-8")] = value.serialize()
            else:
                raise NotImplementedError("serializing non-Object fields; write your own serialize function")
        return result

    def _serialize(self, **kwargs: object) -> Dict[bytes, object]:
        return {
            b"type": type(self).__name__.encode("utf-8"),
            **{key.encode("utf-8"): value for key, value in kwargs.items()},
        }

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Object":
        raise NotImplementedError(msg)

    @staticmethod
    def deserialize(msg: Dict[str, Any]) -> "Object":
        ty = msg["type"]
        assert isinstance(ty, str)
        deserializer = OBJECT_DESERIALIZERS.get(ty)
        if not deserializer:
            raise NotImplementedError(f"unknown type {ty}")
        result = deserializer(msg)
        assert isinstance(result, Object)
        return result


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Int(Object):
    value: int

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(value=self.value)

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Int":
        assert msg["type"] == "Int"
        assert isinstance(msg["value"], int)
        return Int(msg["value"])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class String(Object):
    value: str

    def serialize(self) -> Dict[bytes, object]:
        return {b"type": b"String", b"value": self.value.encode("utf-8")}

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "String":
        assert msg["type"] == "String"
        assert isinstance(msg["value"], str)
        return String(msg["value"])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bytes(Object):
    value: bytes

    def serialize(self) -> Dict[bytes, object]:
        return {b"type": b"Bytes", b"value": self.value}

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Bytes":
        assert msg["type"] == "Bytes"
        assert isinstance(msg["value"], bytes)
        return Bytes(msg["value"])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str

    def serialize(self) -> Dict[bytes, object]:
        return {b"type": b"Var", b"name": self.name.encode("utf-8")}

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Var":
        assert msg["type"] == "Var"
        assert isinstance(msg["name"], str)
        return Var(msg["name"])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bool(Object):
    value: bool

    def serialize(self) -> Dict[bytes, object]:
        return {b"type": b"Bool", b"value": self.value}

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Bool":
        assert msg["type"] == "Bool"
        assert isinstance(msg["value"], bool)
        return Bool(msg["value"])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Hole(Object):
    pass


Env = Mapping[str, Object]


class BinopKind(enum.Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EXP = auto()
    MOD = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    BOOL_AND = auto()
    BOOL_OR = auto()
    STRING_CONCAT = auto()
    LIST_CONS = auto()
    LIST_APPEND = auto()
    RIGHT_EVAL = auto()
    HASTYPE = auto()
    PIPE = auto()
    REVERSE_PIPE = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
            "^": cls.EXP,
            "%": cls.MOD,
            "==": cls.EQUAL,
            "/=": cls.NOT_EQUAL,
            "<": cls.LESS,
            ">": cls.GREATER,
            "<=": cls.LESS_EQUAL,
            ">=": cls.GREATER_EQUAL,
            "&&": cls.BOOL_AND,
            "||": cls.BOOL_OR,
            "++": cls.STRING_CONCAT,
            ">+": cls.LIST_CONS,
            "+<": cls.LIST_APPEND,
            "!": cls.RIGHT_EVAL,
            ":": cls.HASTYPE,
            "|>": cls.PIPE,
            "<|": cls.REVERSE_PIPE,
        }[x]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Binop(Object):
    op: BinopKind
    left: Object
    right: Object

    def serialize(self) -> Dict[bytes, object]:
        return {
            b"type": b"Binop",
            b"op": self.op.name.encode("utf-8"),
            b"left": self.left.serialize(),
            b"right": self.right.serialize(),
        }

    @staticmethod
    def _deserialize(msg: Dict[str, object]) -> "Binop":
        assert msg["type"] == "Binop"
        opname = msg["op"]
        assert isinstance(opname, str)
        op = BinopKind[opname]
        assert isinstance(op, BinopKind)
        left_obj = msg["left"]
        assert isinstance(left_obj, dict)
        right_obj = msg["right"]
        assert isinstance(right_obj, dict)
        left = Object.deserialize(left_obj)
        right = Object.deserialize(right_obj)
        return Binop(op, left, right)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class List(Object):
    items: typing.List[Object]

    def serialize(self) -> Dict[bytes, object]:
        return {b"type": b"List", b"items": [item.serialize() for item in self.items]}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assign(Object):
    name: Var
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
class Compose(Object):
    inner: Object
    outer: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Where(Object):
    body: Object
    binding: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assert(Object):
    value: Object
    cond: Object


def serialize_env(env: Env) -> Dict[bytes, object]:
    return {key.encode("utf-8"): value.serialize() for key, value in env.items()}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(value=serialize_env(self.env))


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchCase(Object):
    pattern: Object
    body: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchFunction(Object):
    cases: typing.List[MatchCase]

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(cases=[case.serialize() for case in self.cases])


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Relocation(Object):
    name: str

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(name=self.name.encode("utf-8"))


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunctionRelocation(Relocation):
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunction(Object):
    name: str
    func: Callable[[Object], Object]

    def serialize(self) -> Dict[bytes, object]:
        return NativeFunctionRelocation(self.name).serialize()


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Closure(Object):
    env: Env
    func: Union[Function, MatchFunction]

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(env=serialize_env(self.env), func=self.func.serialize())


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Record(Object):
    data: Dict[str, Object]

    def serialize(self) -> Dict[bytes, object]:
        return self._serialize(data={key.encode("utf-8"): value.serialize() for key, value in self.data.items()})


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Access(Object):
    obj: Object
    at: Object


def unpack_int(obj: Object) -> int:
    if not isinstance(obj, Int):
        raise TypeError(f"expected Int, got {type(obj).__name__}")
    return obj.value


def eval_int(env: Env, exp: Object) -> int:
    result = eval_exp(env, exp)
    return unpack_int(result)


def eval_str(env: Env, exp: Object) -> str:
    result = eval_exp(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


def eval_bool(env: Env, exp: Object) -> bool:
    result = eval_exp(env, exp)
    if not isinstance(result, Bool):
        raise TypeError(f"expected Bool, got {type(result).__name__}")
    return result.value


def eval_list(env: Env, exp: Object) -> typing.List[Object]:
    result = eval_exp(env, exp)
    if not isinstance(result, List):
        raise TypeError(f"expected List, got {type(result).__name__}")
    return result.items


BINOP_HANDLERS: Dict[BinopKind, Callable[[Env, Object, Object], Object]] = {
    BinopKind.ADD: lambda env, x, y: Int(eval_int(env, x) + eval_int(env, y)),
    BinopKind.SUB: lambda env, x, y: Int(eval_int(env, x) - eval_int(env, y)),
    BinopKind.MUL: lambda env, x, y: Int(eval_int(env, x) * eval_int(env, y)),
    BinopKind.DIV: lambda env, x, y: Int(eval_int(env, x) // eval_int(env, y)),
    BinopKind.EXP: lambda env, x, y: Int(eval_int(env, x) ** eval_int(env, y)),
    BinopKind.MOD: lambda env, x, y: Int(eval_int(env, x) % eval_int(env, y)),
    BinopKind.EQUAL: lambda env, x, y: Bool(eval_exp(env, x) == eval_exp(env, y)),
    BinopKind.NOT_EQUAL: lambda env, x, y: Bool(eval_exp(env, x) != eval_exp(env, y)),
    BinopKind.LESS: lambda env, x, y: Bool(eval_int(env, x) < eval_int(env, y)),
    BinopKind.GREATER: lambda env, x, y: Bool(eval_int(env, x) > eval_int(env, y)),
    BinopKind.LESS_EQUAL: lambda env, x, y: Bool(eval_int(env, x) <= eval_int(env, y)),
    BinopKind.GREATER_EQUAL: lambda env, x, y: Bool(eval_int(env, x) >= eval_int(env, y)),
    BinopKind.BOOL_AND: lambda env, x, y: Bool(eval_bool(env, x) and eval_bool(env, y)),
    BinopKind.BOOL_OR: lambda env, x, y: Bool(eval_bool(env, x) or eval_bool(env, y)),
    BinopKind.STRING_CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    BinopKind.LIST_CONS: lambda env, x, y: List([eval_exp(env, x)] + eval_list(env, y)),
    BinopKind.LIST_APPEND: lambda env, x, y: List(eval_list(env, x) + [eval_exp(env, y)]),
    BinopKind.RIGHT_EVAL: lambda env, x, y: eval_exp(env, y),
}


class MatchError(Exception):
    pass


def match(obj: Object, pattern: Object) -> Optional[Env]:
    if isinstance(pattern, Int):
        return {} if isinstance(obj, Int) and obj.value == pattern.value else None
    if isinstance(pattern, String):
        return {} if isinstance(obj, String) and obj.value == pattern.value else None
    if isinstance(pattern, Var):
        return {pattern.name: obj}
    if isinstance(pattern, Record):
        if not isinstance(obj, Record):
            return None
        result: Env = {}
        if len(pattern.data) != len(obj.data):
            # TODO: Remove this check when implementing ... operator
            return None
        for key, value in pattern.data.items():
            obj_value = obj.data.get(key)
            if obj_value is None:
                return None
            part = match(obj_value, value)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        return result
    if isinstance(pattern, List):
        if not isinstance(obj, List):
            return None
        if len(pattern.items) != len(obj.items):
            # TODO: Remove this check when implementing ... operator
            return None
        result: Env = {}  # type: ignore
        for i, pattern_item in enumerate(pattern.items):
            obj_item = obj.items[i]
            part = match(obj_item, pattern_item)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        return result
    raise NotImplementedError(f"match not implemented for {type(pattern).__name__}")


# pylint: disable=redefined-builtin
def eval_exp(env: Env, exp: Object) -> Object:
    logger.debug(exp)
    if isinstance(exp, (Int, Bool, String, Bytes, Hole, Closure, NativeFunction)):
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
        return List([eval_exp(env, item) for item in exp.items])
    if isinstance(exp, Record):
        return Record({k: eval_exp(env, exp.data[k]) for k in exp.data})
    if isinstance(exp, Assign):
        # TODO(max): Rework this. There's something about matching that we need
        # to figure out and implement.
        assert isinstance(exp.name, Var)
        value = eval_exp(env, exp.value)
        if isinstance(value, Closure):
            # We want functions to be able to call themselves without using the
            # Y combinator or similar, so we bind functions (and only
            # functions) using a letrec-like strategy. We augment their
            # captured environment with a binding to themselves.
            assert isinstance(value.env, dict)
            value.env[exp.name.name] = value
        return EnvObject({**env, exp.name.name: value})
    if isinstance(exp, Where):
        res_env = eval_exp(env, exp.binding)
        assert isinstance(res_env, EnvObject)
        new_env = {**env, **res_env.env}
        return eval_exp(new_env, exp.body)
    if isinstance(exp, Assert):
        cond = eval_exp(env, exp.cond)
        if cond != Bool(True):
            raise AssertionError(f"condition {exp.cond} failed")
        return eval_exp(env, exp.value)
    if isinstance(exp, Function):
        if not isinstance(exp.arg, Var):
            raise RuntimeError(f"expected variable in function definition {exp.arg}")
        return Closure(env, exp)
    if isinstance(exp, MatchFunction):
        return Closure(env, exp)
    if isinstance(exp, Apply):
        if isinstance(exp.func, Var) and exp.func.name == "$$quote":
            return exp.arg
        callee = eval_exp(env, exp.func)
        if isinstance(callee, NativeFunction):
            arg = eval_exp(env, exp.arg)
            return callee.func(arg)
        if not isinstance(callee, Closure):
            raise TypeError(f"attempted to apply a non-closure of type {type(callee).__name__}")
        arg = eval_exp(env, exp.arg)
        if isinstance(callee.func, Function):
            assert isinstance(callee.func.arg, Var)
            new_env = {**callee.env, callee.func.arg.name: arg}
            return eval_exp(new_env, callee.func.body)
        elif isinstance(callee.func, MatchFunction):
            arg = eval_exp(env, exp.arg)
            for case in callee.func.cases:
                m = match(arg, case.pattern)
                if m is None:
                    continue
                return eval_exp({**callee.env, **m}, case.body)
            raise MatchError("no matching cases")
        else:
            raise TypeError(f"attempted to apply a non-function of type {type(callee.func).__name__}")
    if isinstance(exp, Access):
        obj = eval_exp(env, exp.obj)
        if isinstance(obj, Record):
            if not isinstance(exp.at, Var):
                raise TypeError(f"cannot access record field using {type(exp.at).__name__}, expected a field name")
            if exp.at.name not in obj.data:
                raise NameError(f"no assignment to {exp.at.name} found in record")
            return obj.data[exp.at.name]
        elif isinstance(obj, List):
            access_at = eval_exp(env, exp.at)
            if not isinstance(access_at, Int):
                raise TypeError(f"cannot index into list using type {type(access_at).__name__}, expected integer")
            if access_at.value < 0 or access_at.value >= len(obj.items):
                raise ValueError(f"index {access_at.value} out of bounds for list")
            return obj.items[access_at.value]
        raise TypeError(f"attempted to access from type {type(obj).__name__}")
    if isinstance(exp, Compose):
        clo_inner = eval_exp(env, exp.inner)
        clo_outer = eval_exp(env, exp.outer)
        return Closure({}, Function(Var("x"), Apply(clo_outer, Apply(clo_inner, Var("x")))))
    raise NotImplementedError(f"eval_exp not implemented for {exp}")


def bencode(obj: object) -> bytes:
    if isinstance(obj, int):
        return b"i" + str(int(obj)).encode("ascii") + b"e"
    if isinstance(obj, bytes):
        return str(len(obj)).encode("ascii") + b":" + obj
    if isinstance(obj, list):
        return b"l" + b"".join(bencode(x) for x in obj) + b"e"
    if isinstance(obj, dict):
        sorted_items = sorted(obj.items(), key=lambda x: x[0])
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in sorted_items) + b"e"
    raise NotImplementedError(f"bencode not implemented for {type(obj)}")


class Bdecoder:
    def __init__(self, msg: str) -> None:
        self.msg: str = msg
        self.idx: int = 0

    def peek(self) -> str:
        return self.msg[self.idx]

    def read(self) -> str:
        c = self.peek()
        self.idx += 1
        return c

    def decode_int(self) -> int:
        buf = ""
        while (c := self.read()) != "e":
            buf += c
        return int(buf)

    def decode_list(self) -> typing.List[Any]:
        result = []
        while (c := self.peek()) != "e":
            result.append(self.decode())
        assert self.read() == "e"
        return result

    def decode_dict(self) -> typing.Dict[Any, Any]:
        result: Dict[Any, Any] = {}
        while (c := self.peek()) != "e":
            key = self.decode()
            value = self.decode()
            result[key] = value
        assert self.read() == "e"
        return result

    def decode_str(self, start: str) -> str:
        len_buf = start
        while (c := self.peek()) != ":":
            assert c.isdigit()
            len_buf += c
            self.read()
        assert self.read() == ":"
        buf = ""
        for _ in range(int(len_buf)):
            buf += self.read()
        return buf

    def decode(self) -> object:
        ty = self.read()
        if ty == "i":
            return self.decode_int()
        if ty == "l":
            return self.decode_list()
        if ty == "d":
            return self.decode_dict()
        if ty.isdigit():
            return self.decode_str(ty)
        raise NotImplementedError(ty)


def bdecode(msg: str) -> object:
    return Bdecoder(msg).decode()


def serialize(obj: Object) -> bytes:
    return bencode(obj.serialize())


class ScrapMonad:
    def __init__(self, env: Env):
        assert isinstance(env, dict)  # for .copy()
        self.env: Env = env.copy()

    def bind(self, exp: Object) -> "ScrapMonad":
        env = self.env
        result = eval_exp(env, exp)
        if isinstance(result, EnvObject):
            return ScrapMonad({**env, **result.env})
        return ScrapMonad({**env, "_": result})


class ScrapReplServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        logger.debug("GET %s", self.path)
        parsed_path = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        logging.debug("PATH %s", parsed_path)
        logging.debug("QUERY %s", query)
        if parsed_path.path == "/repl":
            return self.do_repl()
        if parsed_path.path == "/eval":
            return self.do_eval(query)
        return self.do_404()

    def do_repl(self) -> None:
        html = rb"""
<html>
<head>
</head>
<body>
Output:
<div>
<pre id="output">
>>> </pre>
</div>
<div>
Input: <input id="input" onkeyup=""/>
</div>
<script type="module">
"use strict";

async function sendRequest(exp) {
    const response = await fetch("/eval?" + new URLSearchParams({exp}));
    return response.json();
}

const input = document.getElementById("input");
const output = document.getElementById("output");
input.addEventListener("keyup", async ({key}) => {
    if (key == "Enter") {
        const response = await sendRequest(input.value);
        const {env, result} = response;
        output.innerHTML += input.value + "\n";
        output.innerHTML += result + "\n>>> ";
        input.value = "";
    }
});
</script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html)
        return

    def do_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"try hitting /eval?exp=EXP&env=ENV")
        return

    def do_eval(self, query: Dict[str, Any]) -> None:
        exp = query.get("exp")
        if exp is None:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Need expression to evaluate")
            return
        if len(exp) != 1:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Need exactly one expression to evaluate")
            return
        exp = exp[0]
        tokens = tokenize(exp)
        ast = parse(tokens)
        env = query.get("env")
        if env is None:
            env = STDLIB
        else:
            if len(env) != 1:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"Need exactly one env")
                return
            env = env[0]
            # TODO(max): Deserialize env
        logging.debug("ast is %s and env is %s", ast, env)
        monad = ScrapMonad(env)
        try:
            result = monad.bind(ast)
        except Exception as e:
            serialized = serialize_env(env)
            encoded = bencode(serialized)
            response = {"env": encoded.decode("utf-8"), "result": str(e)}
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            serialized = serialize_env(result.env)
            encoded = bencode(serialized)
            expr_result = result.env.get("_")
            response = {"env": encoded.decode("utf-8"), "result": repr(expr_result) if expr_result is not None else ""}
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return


class TokenizerTests(unittest.TestCase):
    def test_tokenize_digit(self) -> None:
        self.assertEqual(tokenize("1"), [NumLit(1)])

    def test_tokenize_multiple_digits(self) -> None:
        self.assertEqual(tokenize("123"), [NumLit(123)])

    def test_tokenize_negative_int(self) -> None:
        self.assertEqual(tokenize("-123"), [Operator("-"), NumLit(123)])

    def test_tokenize_binop(self) -> None:
        self.assertEqual(tokenize("1 + 2"), [NumLit(1), Operator("+"), NumLit(2)])

    def test_tokenize_binop_no_spaces(self) -> None:
        self.assertEqual(tokenize("1+2"), [NumLit(1), Operator("+"), NumLit(2)])

    def test_tokenize_two_oper_chars_returns_two_ops(self) -> None:
        self.assertEqual(tokenize(",:"), [Operator(","), Operator(":")])

    def test_tokenize_binary_sub_no_spaces(self) -> None:
        self.assertEqual(tokenize("1-2"), [NumLit(1), Operator("-"), NumLit(2)])

    def test_tokenize_binop_var(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                self.assertEqual(tokenize(f"a {op} b"), [Name("a"), Operator(op), Name("b")])
                self.assertEqual(tokenize(f"a{op}b"), [Name("a"), Operator(op), Name("b")])

    def test_tokenize_var(self) -> None:
        self.assertEqual(tokenize("abc"), [Name("abc")])

    @unittest.skip("TODO: make this fail to tokenize")
    def test_tokenize_var_with_quote(self) -> None:
        self.assertEqual(tokenize("sha1'abc"), [Name("sha1'abc")])

    def test_tokenize_dollar_sha1_var(self) -> None:
        self.assertEqual(tokenize("$sha1'foo"), [Name("$sha1'foo")])

    def test_tokenize_dollar_dollar_var(self) -> None:
        self.assertEqual(tokenize("$$bills"), [Name("$$bills")])

    def test_tokenize_dot_dot_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token '..'")):
            tokenize("..")

    def test_tokenize_spread(self) -> None:
        self.assertEqual(tokenize("..."), [Operator("...")])

    def test_ignore_whitespace(self) -> None:
        self.assertEqual(tokenize("1\n+\t2"), [NumLit(1), Operator("+"), NumLit(2)])

    def test_ignore_line_comment(self) -> None:
        self.assertEqual(tokenize("-- 1\n2"), [NumLit(2)])

    def test_tokenize_string(self) -> None:
        self.assertEqual(tokenize('"hello"'), [StringLit("hello")])

    def test_tokenize_string_with_spaces(self) -> None:
        self.assertEqual(tokenize('"hello world"'), [StringLit("hello world")])

    def test_tokenize_string_missing_end_quote_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(UnexpectedEOFError, "while reading string"):
            tokenize('"hello')

    def test_tokenize_with_trailing_whitespace(self) -> None:
        self.assertEqual(tokenize("- "), [Operator("-")])
        self.assertEqual(tokenize("-- "), [])
        self.assertEqual(tokenize("+ "), [Operator("+")])
        self.assertEqual(tokenize("123 "), [NumLit(123)])
        self.assertEqual(tokenize("abc "), [Name("abc")])
        self.assertEqual(tokenize("[ "), [LeftBracket()])
        self.assertEqual(tokenize("] "), [RightBracket()])

    def test_tokenize_empty_list(self) -> None:
        self.assertEqual(tokenize("[ ]"), [LeftBracket(), RightBracket()])

    def test_tokenize_empty_list_with_spaces(self) -> None:
        self.assertEqual(tokenize("[ ]"), [LeftBracket(), RightBracket()])

    def test_tokenize_list_with_items(self) -> None:
        self.assertEqual(tokenize("[ 1 , 2 ]"), [LeftBracket(), NumLit(1), Operator(","), NumLit(2), RightBracket()])

    def test_tokenize_list_with_no_spaces(self) -> None:
        self.assertEqual(tokenize("[1,2]"), [LeftBracket(), NumLit(1), Operator(","), NumLit(2), RightBracket()])

    def test_tokenize_function(self) -> None:
        self.assertEqual(
            tokenize("a -> b -> a + b"),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_function_with_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("a->b->a+b"),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_where(self) -> None:
        self.assertEqual(tokenize("a . b"), [Name("a"), Operator("."), Name("b")])

    def test_tokenize_assert(self) -> None:
        self.assertEqual(tokenize("a ? b"), [Name("a"), Operator("?"), Name("b")])

    def test_tokenize_hastype(self) -> None:
        self.assertEqual(tokenize("a : b"), [Name("a"), Operator(":"), Name("b")])

    def test_tokenize_minus_returns_minus(self) -> None:
        self.assertEqual(tokenize("-"), [Operator("-")])

    def test_tokenize_tilde_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~")

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~=")

    def test_tokenize_tilde_tilde_returns_empty_bytes(self) -> None:
        self.assertEqual(tokenize("~~"), [BytesLit("", 64)])

    def test_tokenize_bytes_returns_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~QUJD"), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base85(self) -> None:
        self.assertEqual(tokenize("~~85'K|(_"), [BytesLit("K|(_", 85)])

    def test_tokenize_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~64'QUJD"), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base32(self) -> None:
        self.assertEqual(tokenize("~~32'IFBEG==="), [BytesLit("IFBEG===", 32)])

    def test_tokenize_bytes_base16(self) -> None:
        self.assertEqual(tokenize("~~16'414243"), [BytesLit("414243", 16)])

    def test_tokenize_hole(self) -> None:
        self.assertEqual(tokenize("()"), [LeftParen(), RightParen()])

    def test_tokenize_hole_with_spaces(self) -> None:
        self.assertEqual(tokenize("( )"), [LeftParen(), RightParen()])

    def test_tokenize_parenthetical_expression(self) -> None:
        self.assertEqual(tokenize("(1+2)"), [LeftParen(), NumLit(1), Operator("+"), NumLit(2), RightParen()])

    def test_tokenize_pipe(self) -> None:
        self.assertEqual(
            tokenize("1 |> f . f = a -> a + 1"),
            [
                NumLit(1),
                Operator("|>"),
                Name("f"),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                NumLit(1),
            ],
        )

    def test_tokenize_reverse_pipe(self) -> None:
        self.assertEqual(
            tokenize("f <| 1 . f = a -> a + 1"),
            [
                Name("f"),
                Operator("<|"),
                NumLit(1),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                NumLit(1),
            ],
        )

    def test_tokenize_record_no_fields(self) -> None:
        self.assertEqual(
            tokenize("{ }"),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_no_fields_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("{}"),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_one_field(self) -> None:
        self.assertEqual(
            tokenize("{ a = 4 }"),
            [LeftBrace(), Name("a"), Operator("="), NumLit(4), RightBrace()],
        )

    def test_tokenize_record_multiple_fields(self) -> None:
        self.assertEqual(
            tokenize('{ a = 4, b = "z" }'),
            [
                LeftBrace(),
                Name("a"),
                Operator("="),
                NumLit(4),
                Operator(","),
                Name("b"),
                Operator("="),
                StringLit("z"),
                RightBrace(),
            ],
        )

    def test_tokenize_record_access(self) -> None:
        self.assertEqual(
            tokenize("r@a"),
            [Name("r"), Operator("@"), Name("a")],
        )

    def test_tokenize_right_eval(self) -> None:
        self.assertEqual(tokenize("a!b"), [Name("a"), Operator("!"), Name("b")])

    def test_tokenize_match(self) -> None:
        self.assertEqual(
            tokenize("g = | 1 -> 2 | 2 -> 3"),
            [
                Name("g"),
                Operator("="),
                Operator("|"),
                NumLit(1),
                Operator("->"),
                NumLit(2),
                Operator("|"),
                NumLit(2),
                Operator("->"),
                NumLit(3),
            ],
        )

    def test_tokenize_compose(self) -> None:
        self.assertEqual(
            tokenize("f >> g"),
            [Name("f"), Operator(">>"), Name("g")],
        )

    def test_tokenize_compose_reverse(self) -> None:
        self.assertEqual(
            tokenize("f << g"),
            [Name("f"), Operator("<<"), Name("g")],
        )

    def test_first_lineno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.lineno, 1)

    def test_first_colno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.colno, 1)

    def test_first_line_is_empty(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.line, "")

    def test_read_char_increments_colno(self) -> None:
        l = Lexer("abc")
        l.read_char()
        self.assertEqual(l.colno, 2)
        self.assertEqual(l.lineno, 1)

    def test_read_newline_increments_lineno(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        l.read_char()
        self.assertEqual(l.lineno, 2)
        self.assertEqual(l.colno, 1)

    def test_read_char_appends_to_line(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        self.assertEqual(l.line, "ab")
        l.read_char()
        self.assertEqual(l.line, "")

    def read_one_sets_lineno(self) -> None:
        l = Lexer("a b \n c d")
        a = l.read_one()
        b = l.read_one()
        c = l.read_one()
        d = l.read_one()
        self.assertEqual(a.lineno, 1)
        self.assertEqual(b.lineno, 1)
        self.assertEqual(c.lineno, 2)
        self.assertEqual(d.lineno, 2)


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedEOFError) as ctx:
            parse([])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_digit_returns_int(self) -> None:
        self.assertEqual(parse([NumLit(1)]), Int(1))

    def test_parse_digits_returns_int(self) -> None:
        self.assertEqual(parse([NumLit(123)]), Int(123))

    def test_parse_negative_int_returns_binary_sub_int(self) -> None:
        self.assertEqual(parse([Operator("-"), NumLit(123)]), Binop(BinopKind.SUB, Int(0), Int(123)))

    def test_parse_negative_var_returns_binary_sub_int(self) -> None:
        self.assertEqual(parse([Operator("-"), Name("x")]), Binop(BinopKind.SUB, Int(0), Var("x")))

    def test_parse_negative_int_binds_tighter_than_plus(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("+"), Name("r")]),
            Binop(BinopKind.ADD, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_mul(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("*"), Name("r")]),
            Binop(BinopKind.MUL, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_index(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Operator("@"), Name("r")]),
            Access(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_apply(self) -> None:
        self.assertEqual(
            parse([Operator("-"), Name("l"), Name("r")]),
            Apply(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("abc_123")]), Var("abc_123"))

    def test_parse_sha_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("$sha1'abc")]), Var("$sha1'abc"))

    def test_parse_sha_var_without_quote_returns_var(self) -> None:
        self.assertEqual(parse([Name("$sha1abc")]), Var("$sha1abc"))

    def test_parse_dollar_returns_var(self) -> None:
        self.assertEqual(parse([Name("$")]), Var("$"))

    def test_parse_dollar_dollar_returns_var(self) -> None:
        self.assertEqual(parse([Name("$$")]), Var("$$"))

    @unittest.skip("TODO: make this fail to parse")
    def test_parse_sha_var_without_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse([Name("sha1'abc")])

    def test_parse_dollar_dollar_var_returns_var(self) -> None:
        self.assertEqual(parse([Name("$$bills")]), Var("$$bills"))

    def test_parse_bytes_returns_bytes(self) -> None:
        self.assertEqual(parse([BytesLit("QUJD", 64)]), Bytes(b"ABC"))

    def test_parse_binary_add_returns_binop(self) -> None:
        self.assertEqual(parse([NumLit(1), Operator("+"), NumLit(2)]), Binop(BinopKind.ADD, Int(1), Int(2)))

    def test_parse_binary_sub_returns_binop(self) -> None:
        self.assertEqual(parse([NumLit(1), Operator("-"), NumLit(2)]), Binop(BinopKind.SUB, Int(1), Int(2)))

    def test_parse_binary_add_right_returns_binop(self) -> None:
        self.assertEqual(
            parse([NumLit(1), Operator("+"), NumLit(2), Operator("+"), NumLit(3)]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_right(self) -> None:
        self.assertEqual(
            parse([NumLit(1), Operator("+"), NumLit(2), Operator("*"), NumLit(3)]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_left(self) -> None:
        self.assertEqual(
            parse([NumLit(1), Operator("*"), NumLit(2), Operator("+"), NumLit(3)]),
            Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3)),
        )

    def test_exp_binds_tighter_than_mul_right(self) -> None:
        self.assertEqual(
            parse([NumLit(5), Operator("*"), NumLit(2), Operator("^"), NumLit(3)]),
            Binop(BinopKind.MUL, Int(5), Binop(BinopKind.EXP, Int(2), Int(3))),
        )

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("+<"), Name("ls"), Operator("@"), NumLit(0)]),
            Binop(BinopKind.LIST_APPEND, Var("a"), Access(Var("ls"), Int(0))),
        )

    def test_parse_binary_str_concat_returns_binop(self) -> None:
        self.assertEqual(
            parse([StringLit("abc"), Operator("++"), StringLit("def")]),
            Binop(BinopKind.STRING_CONCAT, String("abc"), String("def")),
        )

    def test_parse_binary_list_cons_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator(">+"), Name("b")]),
            Binop(BinopKind.LIST_CONS, Var("a"), Var("b")),
        )

    def test_parse_binary_list_append_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("+<"), Name("b")]),
            Binop(BinopKind.LIST_APPEND, Var("a"), Var("b")),
        )

    def test_parse_binary_op_returns_binop(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                kind = BinopKind.from_str(op)
                self.assertEqual(parse([Name("a"), Operator(op), Name("b")]), Binop(kind, Var("a"), Var("b")))

    def test_parse_empty_list(self) -> None:
        self.assertEqual(
            parse([LeftBracket(), RightBracket()]),
            List([]),
        )

    def test_parse_list_of_ints_returns_list(self) -> None:
        self.assertEqual(
            parse([LeftBracket(), NumLit(1), Operator(","), NumLit(2), RightBracket()]),
            List([Int(1), Int(2)]),
        )

    def test_parse_assign(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("="), NumLit(1)]),
            Assign(Var("a"), Int(1)),
        )

    def test_parse_function_one_arg_returns_function(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("->"), Name("a"), Operator("+"), NumLit(1)]),
            Function(Var("a"), Binop(BinopKind.ADD, Var("a"), Int(1))),
        )

    def test_parse_function_two_args_returns_functions(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")]),
            Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))),
        )

    def test_parse_assign_function(self) -> None:
        self.assertEqual(
            parse([Name("id"), Operator("="), Name("x"), Operator("->"), Name("x")]),
            Assign(Var("id"), Function(Var("x"), Var("x"))),
        )

    def test_parse_function_application_one_arg(self) -> None:
        self.assertEqual(parse([Name("f"), Name("a")]), Apply(Var("f"), Var("a")))

    def test_parse_function_application_two_args(self) -> None:
        self.assertEqual(parse([Name("f"), Name("a"), Name("b")]), Apply(Apply(Var("f"), Var("a")), Var("b")))

    def test_parse_where(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("."), Name("b")]), Where(Var("a"), Var("b")))

    def test_parse_nested_where(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("."), Name("b"), Operator("."), Name("c")]),
            Where(Where(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_assert(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("?"), Name("b")]), Assert(Var("a"), Var("b")))

    def test_parse_nested_assert(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("?"), Name("b"), Operator("?"), Name("c")]),
            Assert(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_mixed_assert_where(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("?"), Name("b"), Operator("."), Name("c")]),
            Where(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_hastype(self) -> None:
        self.assertEqual(parse([Name("a"), Operator(":"), Name("b")]), Binop(BinopKind.HASTYPE, Var("a"), Var("b")))

    def test_parse_hole(self) -> None:
        self.assertEqual(parse([LeftParen(), RightParen()]), Hole())

    def test_parse_parenthesized_expression(self) -> None:
        self.assertEqual(
            parse([LeftParen(), NumLit(1), Operator("+"), NumLit(2), RightParen()]),
            Binop(BinopKind.ADD, Int(1), Int(2)),
        )

    def test_parse_parenthesized_add_mul(self) -> None:
        self.assertEqual(
            parse([LeftParen(), NumLit(1), Operator("+"), NumLit(2), RightParen(), Operator("*"), NumLit(3)]),
            Binop(BinopKind.MUL, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)),
        )

    def test_parse_pipe(self) -> None:
        self.assertEqual(
            parse([NumLit(1), Operator("|>"), Name("f")]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_pipe(self) -> None:
        self.assertEqual(
            parse([NumLit(1), Operator("|>"), Name("f"), Operator("|>"), Name("g")]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_reverse_pipe(self) -> None:
        self.assertEqual(
            parse([Name("f"), Operator("<|"), NumLit(1)]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_reverse_pipe(self) -> None:
        self.assertEqual(
            parse([Name("g"), Operator("<|"), Name("f"), Operator("<|"), NumLit(1)]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_empty_record(self) -> None:
        self.assertEqual(parse([LeftBrace(), RightBrace()]), Record({}))

    def test_parse_record_single_field(self) -> None:
        self.assertEqual(parse([LeftBrace(), Name("a"), Operator("="), NumLit(4), RightBrace()]), Record({"a": Int(4)}))

    def test_parse_record_with_expression(self) -> None:
        self.assertEqual(
            parse([LeftBrace(), Name("a"), Operator("="), NumLit(1), Operator("+"), NumLit(2), RightBrace()]),
            Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))}),
        )

    def test_parse_record_multiple_fields(self) -> None:
        self.assertEqual(
            parse(
                [
                    LeftBrace(),
                    Name("a"),
                    Operator("="),
                    NumLit(4),
                    Operator(","),
                    Name("b"),
                    Operator("="),
                    StringLit("z"),
                    RightBrace(),
                ]
            ),
            Record({"a": Int(4), "b": String("z")}),
        )

    def test_non_variable_in_assignment_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([NumLit(3), Operator("="), NumLit(4)])
        self.assertEqual(ctx.exception.args[0], "expected variable in assignment Int(value=3)")

    def test_non_assign_in_record_constructor_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([LeftBrace(), NumLit(1), Operator(","), NumLit(2), RightBrace()])
        self.assertEqual(ctx.exception.args[0], "failed to parse variable assignment in record constructor")

    def test_parse_right_eval_returns_binop(self) -> None:
        self.assertEqual(parse([Name("a"), Operator("!"), Name("b")]), Binop(BinopKind.RIGHT_EVAL, Var("a"), Var("b")))

    def test_parse_right_eval_with_defs_returns_binop(self) -> None:
        self.assertEqual(
            parse([Name("a"), Operator("!"), Name("b"), Operator("."), Name("c")]),
            Binop(BinopKind.RIGHT_EVAL, Var("a"), Where(Var("b"), Var("c"))),
        )

    def test_parse_match_no_cases_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([Operator("|")])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_match_one_case(self) -> None:
        self.assertEqual(
            parse([Operator("|"), NumLit(1), Operator("->"), NumLit(2)]),
            MatchFunction([MatchCase(Int(1), Int(2))]),
        )

    def test_parse_match_two_cases(self) -> None:
        self.assertEqual(
            parse(
                [
                    Operator("|"),
                    NumLit(1),
                    Operator("->"),
                    NumLit(2),
                    Operator("|"),
                    NumLit(2),
                    Operator("->"),
                    NumLit(3),
                ]
            ),
            MatchFunction(
                [
                    MatchCase(Int(1), Int(2)),
                    MatchCase(Int(2), Int(3)),
                ]
            ),
        )

    def test_parse_compose(self) -> None:
        self.assertEqual(parse([Name("f"), Operator(">>"), Name("g")]), Compose(Var("f"), Var("g")))

    def test_parse_compose_reverse(self) -> None:
        self.assertEqual(parse([Name("f"), Operator("<<"), Name("g")]), Compose(Var("g"), Var("f")))

    def test_parse_double_compose(self) -> None:
        self.assertEqual(
            parse([Name("f"), Operator("<<"), Name("g"), Operator("<<"), Name("h")]),
            Compose(Compose(Var("h"), Var("g")), Var("f")),
        )

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(
            parse([Name("true"), Operator("||"), Name("true"), Operator("&&"), Name("false")]),
            Binop(BinopKind.BOOL_OR, Var("true"), Binop(BinopKind.BOOL_AND, Var("true"), Var("false"))),
        )


class MatchTests(unittest.TestCase):
    def test_match_with_equal_ints_returns_empty_dict(self) -> None:
        self.assertEqual(match(Int(1), pattern=Int(1)), {})

    def test_match_with_inequal_ints_returns_none(self) -> None:
        self.assertEqual(match(Int(2), pattern=Int(1)), None)

    def test_match_int_with_non_int_returns_none(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Int(1)), None)

    def test_match_with_equal_strings_returns_empty_dict(self) -> None:
        self.assertEqual(match(String("a"), pattern=String("a")), {})

    def test_match_with_inequal_strings_returns_none(self) -> None:
        self.assertEqual(match(String("b"), pattern=String("a")), None)

    def test_match_string_with_non_string_returns_none(self) -> None:
        self.assertEqual(match(Int(1), pattern=String("abc")), None)

    def test_match_var_returns_dict_with_var_name(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Var("a")), {"a": String("abc")})

    def test_match_record_with_non_record_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            None,
        )

    def test_match_record_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y"), "z": Var("z")}),
            ),
            None,
        )

    def test_match_record_with_fewer_fields_in_pattern_none(self) -> None:
        self.assertIs(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x")}),
            ),
            None,
        )

    def test_match_record_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_record_with_matching_const_returns_dict_with_other_keys(self) -> None:
        # TODO(max): Should this be the case? I feel like we should return all
        # the keys.
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(1), "y": Var("y")}),
            ),
            {"y": Int(2)},
        )

    def test_match_record_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(3), "y": Var("y")}),
            ),
            None,
        )

    def test_match_list_with_non_list_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=List([Var("x"), Var("y")]),
            ),
            None,
        )

    def test_match_list_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y"), Var("z")]),
            ),
            None,
        )

    def test_match_list_with_fewer_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x")]),
            ),
            None,
        )

    def test_match_list_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y")]),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_list_with_matching_const_returns_dict_with_other_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(1), Var("y")]),
            ),
            {"y": Int(2)},
        )

    def test_match_list_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(3), Var("y")]),
            ),
            None,
        )

    def test_parse_right_pipe(self) -> None:
        text = "3 + 4 |> $$quote"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_left_pipe(self) -> None:
        text = "$$quote <| 3 + 4"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_match_with_left_apply(self) -> None:
        text = """| a -> b <| c
                  | d -> e"""
        tokens = tokenize(text)
        self.assertEqual(
            tokens,
            [
                Operator("|"),
                Name("a"),
                Operator("->"),
                Name("b"),
                Operator("<|"),
                Name("c"),
                Operator("|"),
                Name("d"),
                Operator("->"),
                Name("e"),
            ],
        )
        ast = parse(tokens)
        self.assertEqual(
            ast, MatchFunction([MatchCase(Var("a"), Apply(Var("b"), Var("c"))), MatchCase(Var("d"), Var("e"))])
        )

    def test_parse_match_with_right_apply(self) -> None:
        text = """
| 1 -> 19
| a -> a |> (x -> x + 1)
"""
        tokens = tokenize(text)
        ast = parse(tokens)
        self.assertEqual(
            ast,
            MatchFunction(
                [
                    MatchCase(Int(1), Int(19)),
                    MatchCase(
                        Var("a"),
                        Apply(
                            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))),
                            Var("a"),
                        ),
                    ),
                ]
            ),
        )


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        self.assertEqual(eval_exp({}, exp), String("xyz"))

    def test_eval_bytes_returns_bytes(self) -> None:
        exp = Bytes(b"xyz")
        self.assertEqual(eval_exp({}, exp), Bytes(b"xyz"))

    def test_eval_true_returns_true(self) -> None:
        self.assertEqual(eval_exp({}, Bool(True)), Bool(True))

    def test_eval_false_returns_false(self) -> None:
        self.assertEqual(eval_exp({}, Bool(False)), Bool(False))

    def test_eval_with_non_existent_var_raises_name_error(self) -> None:
        exp = Var("no")
        with self.assertRaises(NameError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "name 'no' is not defined")

    def test_eval_with_bound_var_returns_value(self) -> None:
        exp = Var("yes")
        env = {"yes": Int(123)}
        self.assertEqual(eval_exp(env, exp), Int(123))

    def test_eval_with_binop_add_returns_sum(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_with_nested_binop(self) -> None:
        exp = Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_add_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), String("hello"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected Int, got String")

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(-1))

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(0))

    def test_eval_with_binop_exp(self) -> None:
        exp = Binop(BinopKind.EXP, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(8))

    def test_eval_with_binop_mod(self) -> None:
        exp = Binop(BinopKind.MOD, Int(10), Int(4))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_with_binop_equal_with_equal_returns_true(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), Bool(True))

    def test_eval_with_binop_equal_with_inequal_returns_false(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Bool(False))

    def test_eval_with_binop_not_equal_with_equal_returns_false(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), Bool(False))

    def test_eval_with_binop_not_equal_with_inequal_returns_true(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Bool(True))

    def test_eval_with_binop_concat_with_strings_returns_string(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String("hello"), String(" world"))
        self.assertEqual(eval_exp({}, exp), String("hello world"))

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, Int(123), String(" world"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String(" world"), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_cons_with_int_list_returns_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, Int(1), List([Int(2), Int(3)]))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_binop_cons_with_list_list_returns_nested_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), List([]))
        self.assertEqual(eval_exp({}, exp), List([List([])]))

    def test_eval_with_binop_cons_with_list_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected List, got Int")

    def test_eval_with_list_append(self) -> None:
        exp = Binop(BinopKind.LIST_APPEND, List([Int(1), Int(2)]), Int(3))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        self.assertEqual(eval_exp({}, exp), List([Int(3), Int(7)]))

    def test_eval_with_function_returns_function(self) -> None:
        exp = Function(Var("x"), Var("x"))
        self.assertEqual(eval_exp({}, exp), Closure({}, Function(Var("x"), Var("x"))))

    def test_eval_assign_returns_env_object(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        result = eval_exp(env, exp)
        self.assertEqual(result, EnvObject({"a": Int(1)}))

    def test_eval_assign_function_returns_closure_with_function_in_env(self) -> None:
        exp = Assign(Var("a"), Function(Var("x"), Var("x")))
        result = eval_exp({}, exp)
        assert isinstance(result, EnvObject)
        closure = result.env["a"]
        self.assertIsInstance(closure, Closure)
        self.assertEqual(closure, Closure({"a": closure}, Function(Var("x"), Var("x"))))

    def test_eval_assign_does_not_modify_env(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        eval_exp(env, exp)
        self.assertEqual(env, {})

    def test_eval_where_evaluates_in_order(self) -> None:
        exp = Where(Binop(BinopKind.ADD, Var("a"), Int(2)), Assign(Var("a"), Int(1)))
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_nested_where(self) -> None:
        exp = Where(
            Where(
                Binop(BinopKind.ADD, Var("a"), Var("b")),
                Assign(Var("a"), Int(1)),
            ),
            Assign(Var("b"), Int(2)),
        )
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_assert_with_truthy_cond_returns_value(self) -> None:
        exp = Assert(Int(123), Bool(True))
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        exp = Assert(Int(123), Bool(False))
        with self.assertRaisesRegex(AssertionError, re.escape("condition Bool(value=False) failed")):
            eval_exp({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), Bool(True)), Bool(True))
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_hole(self) -> None:
        exp = Hole()
        self.assertEqual(eval_exp({}, exp), Hole())

    def test_eval_function_application_one_arg(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_function_application_two_args(self) -> None:
        exp = Apply(
            Apply(Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))), Int(3)),
            Int(2),
        )
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_function_returns_closure_with_captured_env(self) -> None:
        exp = Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        res = eval_exp({"y": Int(5)}, exp)
        self.assertIsInstance(res, Closure)
        assert isinstance(res, Closure)  # for mypy
        self.assertEqual(res.env, {"y": Int(5)})

    def test_eval_function_capture_env(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y"))), Int(2))
        self.assertEqual(eval_exp({"y": Int(5)}, exp), Int(7))

    def test_eval_non_function_raises_type_error(self) -> None:
        exp = Apply(Int(3), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to apply a non-closure of type Int")):
            eval_exp({}, exp)

    def test_eval_access_from_invalid_object_raises_type_error(self) -> None:
        exp = Access(Int(4), String("x"))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to access from type Int")):
            eval_exp({}, exp)

    def test_eval_record_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Int(0))
        with self.assertRaisesRegex(
            TypeError, re.escape("cannot access record field using Int, expected a field name")
        ):
            eval_exp({}, exp)

    def test_eval_record_access_with_unknown_accessor_raises_name_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("b"))
        with self.assertRaisesRegex(NameError, re.escape("no assignment to b found in record")):
            eval_exp({}, exp)

    def test_eval_record_access(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(eval_exp({}, exp), Int(4))

    def test_eval_list_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(List([Int(4)]), String("hello"))
        with self.assertRaisesRegex(TypeError, re.escape("cannot index into list using type String, expected integer")):
            eval_exp({}, exp)

    def test_eval_list_access_with_out_of_bounds_accessor_raises_value_error(self) -> None:
        exp = Access(List([Int(1), Int(2), Int(3)]), Int(4))
        with self.assertRaisesRegex(ValueError, re.escape("index 4 out of bounds for list")):
            eval_exp({}, exp)

    def test_eval_list_access(self) -> None:
        exp = Access(List([String("a"), String("b"), String("c")]), Int(2))
        self.assertEqual(eval_exp({}, exp), String("c"))

    def test_right_eval_evaluates_right_hand_side(self) -> None:
        exp = Binop(BinopKind.RIGHT_EVAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_no_cases_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([]), Int(1))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_int_with_equal_int_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(1))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_int_with_inequal_int_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(3))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_string_with_equal_string_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("a"))
        self.assertEqual(eval_exp({}, exp), String("b"))

    def test_match_string_with_inequal_string_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("c"))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_falls_through_to_next(self) -> None:
        exp = Apply(
            MatchFunction([MatchCase(pattern=Int(3), body=Int(4)), MatchCase(pattern=Int(1), body=Int(2))]), Int(1)
        )
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_compose(self) -> None:
        exp = Compose(
            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
            Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
        )
        env = {"a": Int(1)}
        expected = Closure(
            {},
            Function(
                Var("x"),
                Apply(
                    Closure(env, Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2)))),
                    Apply(Closure(env, Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3)))), Var("x")),
                ),
            ),
        )
        self.assertEqual(eval_exp(env, exp), expected)

    def test_eval_compose_apply(self) -> None:
        exp = Apply(
            Compose(
                Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
                Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
            ),
            Int(4),
        )
        self.assertEqual(
            eval_exp({}, exp),
            Int(14),
        )

    def test_eval_native_function_returns_function(self) -> None:
        exp = NativeFunction("times2", lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertIs(eval_exp({}, exp), exp)

    def test_eval_apply_native_function_calls_function(self) -> None:
        exp = Apply(NativeFunction("times2", lambda x: Int(x.value * 2)), Int(3))  # type: ignore [attr-defined]
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_apply_quote_returns_ast(self) -> None:
        ast = Binop(BinopKind.ADD, Int(1), Int(2))
        exp = Apply(Var("$$quote"), ast)
        self.assertIs(eval_exp({}, exp), ast)

    def test_eval_apply_closure_with_match_function_has_access_to_closure_vars(self) -> None:
        ast = Apply(Closure({"x": Int(1)}, MatchFunction([MatchCase(Var("y"), Var("x"))])), Int(2))
        self.assertEqual(eval_exp({}, ast), Int(1))

    def test_eval_less_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Bool(True))

    def test_eval_less_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS, Bool(True), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int, got Bool")):
            eval_exp({}, ast)

    def test_eval_less_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Bool(True))

    def test_eval_less_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, Bool(True), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int, got Bool")):
            eval_exp({}, ast)

    def test_eval_greater_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Bool(False))

    def test_eval_greater_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER, Bool(True), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int, got Bool")):
            eval_exp({}, ast)

    def test_eval_greater_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), Bool(False))

    def test_eval_greater_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, Bool(True), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int, got Bool")):
            eval_exp({}, ast)

    def test_boolean_and_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_AND, Bool(True), Var("a"))
        self.assertEqual(eval_exp({"a": Bool(False)}, ast), Bool(False))

        ast = Binop(BinopKind.BOOL_AND, Var("a"), Bool(False))
        self.assertEqual(eval_exp({"a": Bool(True)}, ast), Bool(False))

    def test_boolean_or_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_OR, Bool(False), Var("a"))
        self.assertEqual(eval_exp({"a": Bool(True)}, ast), Bool(True))

        ast = Binop(BinopKind.BOOL_OR, Var("a"), Bool(True))
        self.assertEqual(eval_exp({"a": Bool(False)}, ast), Bool(True))

    def test_boolean_and_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction(raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_AND, Bool(False), apply)
        self.assertEqual(eval_exp({"error": error}, ast), Bool(False))

    def test_boolean_or_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction(raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_OR, Bool(True), apply)
        self.assertEqual(eval_exp({"error": error}, ast), Bool(True))

    def test_boolean_and_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_AND, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected Bool, got Int")):
            eval_exp({}, exp)

    def test_boolean_or_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_OR, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected Bool, got Int")):
            eval_exp({}, exp)


class EndToEndTests(unittest.TestCase):
    def _run(self, text: str, env: Optional[Env] = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        return eval_exp(env or {}, ast)

    def test_int_returns_int(self) -> None:
        self.assertEqual(self._run("1"), Int(1))

    def test_bytes_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~QUJD"), Bytes(b"ABC"))

    def test_bytes_base85_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~85'K|(_"), Bytes(b"ABC"))

    def test_bytes_base64_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~64'QUJD"), Bytes(b"ABC"))

    def test_bytes_base32_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~32'IFBEG==="), Bytes(b"ABC"))

    def test_bytes_base16_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~16'414243"), Bytes(b"ABC"))

    def test_int_add_returns_int(self) -> None:
        self.assertEqual(self._run("1 + 2"), Int(3))

    def test_string_concat_returns_string(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), String("abcdef"))

    def test_list_cons_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ [2,3]"), List([Int(1), Int(2), Int(3)]))

    def test_list_cons_nested_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ 2 >+ [3,4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_list_append_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3"), List([Int(1), Int(2), Int(3)]))

    def test_list_append_nested_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3 +< 4"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_empty_list(self) -> None:
        self.assertEqual(self._run("[ ]"), List([]))

    def test_empty_list_with_no_spaces(self) -> None:
        self.assertEqual(self._run("[]"), List([]))

    def test_list_of_ints(self) -> None:
        self.assertEqual(self._run("[ 1 , 2 ]"), List([Int(1), Int(2)]))

    def test_list_of_exprs(self) -> None:
        self.assertEqual(
            self._run("[ 1 + 2 , 3 + 4 ]"),
            List([Int(3), Int(7)]),
        )

    def test_where(self) -> None:
        self.assertEqual(self._run("a + 2 . a = 1"), Int(3))

    def test_nested_where(self) -> None:
        self.assertEqual(self._run("a + b . a = 1 . b = 2"), Int(3))

    def test_assert_with_truthy_cond_returns_value(self) -> None:
        self.assertEqual(self._run("a + 1 ? a == 1 . a = 1"), Int(2))

    def test_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        with self.assertRaisesRegex(AssertionError, "condition Binop"):
            self._run("a + 1 ? a == 2 . a = 1")

    def test_nested_assert(self) -> None:
        self.assertEqual(self._run("a + b ? a == 1 ? b == 2 . a = 1 . b = 2"), Int(3))

    def test_hole(self) -> None:
        self.assertEqual(self._run("()"), Hole())

    def test_bindings_behave_like_letstar(self) -> None:
        with self.assertRaises(NameError) as ctx:
            self._run("b . a = 1 . b = a")
        self.assertEqual(ctx.exception.args[0], "name 'a' is not defined")

    def test_function_application_two_args(self) -> None:
        self.assertEqual(self._run("(a -> b -> a + b) 3 2"), Int(5))

    def test_function_create_list_correct_order(self) -> None:
        self.assertEqual(self._run("(a -> b -> [a, b]) 3 2"), List([Int(3), Int(2)]))

    def test_create_record(self) -> None:
        self.assertEqual(self._run("{a = 1 + 3}"), Record({"a": Int(4)}))

    def test_access_record(self) -> None:
        self.assertEqual(self._run('rec@b . rec = { a = 1, b = "x" }'), String("x"))

    def test_access_list(self) -> None:
        self.assertEqual(self._run("xs@1 . xs = [1, 2, 3]"), Int(2))

    def test_access_list_var(self) -> None:
        self.assertEqual(self._run("xs@y . y = 2 . xs = [1, 2, 3]"), Int(3))

    def test_access_list_expr(self) -> None:
        self.assertEqual(self._run("xs@(1+1) . xs = [1, 2, 3]"), Int(3))

    def test_functions_eval_arguments(self) -> None:
        self.assertEqual(self._run("(x -> x) c . c = 1"), Int(1))

    def test_non_var_function_arg_raises_parse_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._run("1 -> a")
        self.assertEqual(ctx.exception.args[0], "expected variable in function definition Int(value=1)")

    def test_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (b -> b * 2)) 3"), Int(8))

    def test_compose_does_not_expose_internal_x(self) -> None:
        with self.assertRaisesRegex(NameError, "name 'x' is not defined"):
            self._run("f 3 . f = ((y -> x) >> (z -> x))")

    def test_double_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (x -> x) >> (b -> b * 2)) 3"), Int(8))

    def test_reverse_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) << (b -> b * 2)) 3"), Int(7))

    def test_simple_int_match(self) -> None:
        self.assertEqual(
            self._run(
                """
                inc 2
                . inc =
                  | 1 -> 2
                  | 2 -> 3
                  """
            ),
            Int(3),
        )

    def test_match_var_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                """
            ),
            Int(3),
        )

    def test_match_var_binds_first_arm(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                  | y -> y * 2
                """
            ),
            Int(3),
        )

    def test_match_function_can_close_over_variables(self) -> None:
        self.assertEqual(
            self._run(
                """
        f 1 2
        . f = a ->
          | b -> a + b
        """
            ),
            Int(3),
        )

    def test_match_record_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { x = 3 }
                . get_x =
                  | { x = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_record_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 3, y = 4 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            ),
            Int(12),
        )

    def test_match_record_with_extra_fields_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult rec
                . rec = { x = 3 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            )

    def test_match_record_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 4, y = 5 }
                . mult =
                  | { x = 3, y = y } -> 1
                  | { x = 4, y = y } -> 2
                """
            ),
            Int(2),
        )

    def test_match_record_with_non_record_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | { x = x } -> x
                """
            )

    def test_match_record_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { a = 3, b = 3 }
                . get_x =
                  | { a = x, b = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_list_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 4, 5]
                . mult =
                  | [1, x, 3, y, 5] -> x * y
                """
            ),
            Int(8),
        )

    def test_match_list_incorrect_length_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult xs
                . xs = [1, 2, 3]
                . mult =
                  | [1, 2] -> 1
                  | [1, 2, 3, 4] -> 1
                  | [1, 3] -> 1
                """
            )

    def test_match_list_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                middle xs
                . xs = [4, 5, 6]
                . middle =
                  | [1, x, 3] -> x
                  | [4, x, 6] -> x
                  | [7, x, 9] -> x
                """
            ),
            Int(5),
        )

    def test_match_list_with_non_list_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | [2, x] -> x
                """
            )

    def test_match_list_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 2, 1]
                . mult =
                  | [1, x, 3, x, 1] -> x
                """
            ),
            Int(2),
        )

    def test_pipe(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2)"), Int(3))

    def test_pipe_nested(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2) |> (b -> b * 2)"), Int(6))

    def test_reverse_pipe(self) -> None:
        self.assertEqual(self._run("(a -> a + 2) <| 1"), Int(3))

    def test_reverse_pipe_nested(self) -> None:
        self.assertEqual(self._run("(b -> b * 2) <| (a -> a + 2) <| 1"), Int(6))

    def test_function_can_reference_itself(self) -> None:
        result = self._run(
            """
    f 1
    . f = n -> f
    """
        )
        self.assertEqual(result, Closure({"f": result}, Function(Var("n"), Var("f"))))

    def test_function_can_call_itself(self) -> None:
        with self.assertRaises(RecursionError):
            self._run(
                """
        f 1
        . f = n -> f n
        """
            )

    def test_match_function_can_call_itself(self) -> None:
        self.assertEqual(
            self._run(
                """
        fac 5
        . fac =
          | 0 -> 1
          | 1 -> 1
          | n -> n * fac (n - 1)
        """
            ),
            Int(120),
        )

    def test_stdlib_add(self) -> None:
        self.assertEqual(self._run("$$add 3 4", STDLIB), Int(7))

    def test_stdlib_quote(self) -> None:
        self.assertEqual(self._run("$$quote (3 + 4)"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_pipe(self) -> None:
        self.assertEqual(self._run("3 + 4 |> $$quote"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_reverse_pipe(self) -> None:
        self.assertEqual(self._run("$$quote <| 3 + 4"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_serialize(self) -> None:
        self.assertEqual(self._run("$$serialize 3", STDLIB), Bytes(value=b"d4:type3:Int5:valuei3ee"))

    def test_stdlib_serialize_expr(self) -> None:
        self.assertEqual(
            self._run("(1+2) |> $$quote |> $$serialize", STDLIB),
            Bytes(value=b"d4:leftd4:type3:Int5:valuei1ee2:op3:ADD5:rightd4:type3:Int5:valuei2ee4:type5:Binope"),
        )

    def test_stdlib_listlength_empty_list_returns_zero(self) -> None:
        self.assertEqual(self._run("$$listlength []", STDLIB), Int(0))

    def test_stdlib_listlength_returns_length(self) -> None:
        self.assertEqual(self._run("$$listlength [1,2,3]", STDLIB), Int(3))

    def test_stdlib_listlength_with_non_list_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._run("$$listlength 1", STDLIB)
        self.assertEqual(ctx.exception.args[0], "listlength expected List, but got Int")

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(self._run("[1, 2, 3] +< xs@0 . xs = [4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_exponentiation(self) -> None:
        self.assertEqual(self._run("6 ^ 2"), Int(36))

    def test_modulus(self) -> None:
        self.assertEqual(self._run("11 % 3"), Int(2))

    def test_exp_binds_tighter_than_mul(self) -> None:
        self.assertEqual(self._run("5 * 2 ^ 3"), Int(40))

    def test_stdlib_true_returns_true(self) -> None:
        # TODO: replace true/false stdlib def with alternate
        self.assertEqual(self._run("true", STDLIB), Bool(True))

    def test_stdlib_false_returns_false(self) -> None:
        # TODO: replace true/false stdlib def with alternate
        self.assertEqual(self._run("false", STDLIB), Bool(False))

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(self._run("true || true && false", STDLIB), Bool(True))

    def test_compare_binds_tighter_than_boolean_and(self) -> None:
        self.assertEqual(self._run("1 < 2 && 2 < 1"), Bool(False))


class BencodeTests(unittest.TestCase):
    def test_bencode_int(self) -> None:
        self.assertEqual(bencode(123), b"i123e")

    def test_bencode_bool(self) -> None:
        # TODO(max): Should we discriminate between bool and int?
        self.assertEqual(bencode(True), b"i1e")

    def test_bencode_negative_int(self) -> None:
        self.assertEqual(bencode(-123), b"i-123e")

    def test_bencode_bytes(self) -> None:
        self.assertEqual(bencode(b"abc"), b"3:abc")

    def test_bencode_empty_list(self) -> None:
        self.assertEqual(bencode([]), b"le")

    def test_bencode_list_of_ints(self) -> None:
        self.assertEqual(bencode([1, 2, 3]), b"li1ei2ei3ee")

    def test_bencode_list_of_lists(self) -> None:
        self.assertEqual(bencode([[1, 2], [3, 4]]), b"lli1ei2eeli3ei4eee")

    def test_bencode_dict_sorts_keys(self) -> None:
        d = {}
        d[b"b"] = 1
        d[b"a"] = 2
        # It's sorted by insertion order (guaranteed Python 3.6+)
        self.assertEqual([*d], [b"b", b"a"])
        # It's sorted lexicographically
        self.assertEqual(bencode(d), b"d1:ai2e1:bi1ee")


class BdecodeTests(unittest.TestCase):
    def test_bdecode_int(self) -> None:
        self.assertEqual(bdecode("i123e"), 123)

    def test_bdecode_bool(self) -> None:
        # TODO(max): Should we discriminate between bool and int?
        self.assertEqual(bdecode("i1e"), 1)

    def test_bdecode_negative_int(self) -> None:
        self.assertEqual(bdecode("i-123e"), -123)

    def test_bdecode_bytes(self) -> None:
        self.assertEqual(bdecode("3:abc"), "abc")

    def test_bdecode_empty_list(self) -> None:
        self.assertEqual(bdecode("le"), [])

    def test_bdecode_list_of_ints(self) -> None:
        self.assertEqual(bdecode("li1ei2ei3ee"), [1, 2, 3])

    def test_bdecode_list_of_lists(self) -> None:
        self.assertEqual(bdecode("lli1ei2eeli3ei4eee"), [[1, 2], [3, 4]])

    def test_bdecode_dict_sorts_keys(self) -> None:
        self.assertEqual(bdecode("d1:ai2e1:bi1ee"), {"b": 1, "a": 2})


class ObjectSerializeTests(unittest.TestCase):
    def test_serialize_int(self) -> None:
        obj = Int(123)
        self.assertEqual(obj.serialize(), {b"type": b"Int", b"value": 123})

    def test_serialize_negative_int(self) -> None:
        obj = Int(-123)
        self.assertEqual(obj.serialize(), {b"type": b"Int", b"value": -123})

    def test_serialize_str(self) -> None:
        obj = String("abc")
        self.assertEqual(obj.serialize(), {b"type": b"String", b"value": b"abc"})

    def test_serialize_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(obj.serialize(), {b"type": b"Bytes", b"value": b"abc"})

    def test_serialize_var(self) -> None:
        obj = Var("abc")
        self.assertEqual(obj.serialize(), {b"type": b"Var", b"name": b"abc"})

    def test_serialize_bool(self) -> None:
        obj = Bool(True)
        self.assertEqual(obj.serialize(), {b"type": b"Bool", b"value": True})

    def test_serialize_binary_add(self) -> None:
        obj = Binop(BinopKind.ADD, Int(123), Int(456))
        self.assertEqual(
            obj.serialize(),
            {
                b"left": {b"type": b"Int", b"value": 123},
                b"op": b"ADD",
                b"right": {b"type": b"Int", b"value": 456},
                b"type": b"Binop",
            },
        )

    def test_serialize_list(self) -> None:
        obj = List([Int(1), Int(2)])
        self.assertEqual(
            obj.serialize(),
            {b"type": b"List", b"items": [{b"type": b"Int", b"value": 1}, {b"type": b"Int", b"value": 2}]},
        )

    def test_serialize_assign(self) -> None:
        obj = Assign(Var("x"), Int(2))
        self.assertEqual(
            obj.serialize(),
            {b"type": b"Assign", b"name": {b"name": b"x", b"type": b"Var"}, b"value": {b"type": b"Int", b"value": 2}},
        )

    def test_serialize_record(self) -> None:
        obj = Record({"x": Int(1)})
        self.assertEqual(obj.serialize(), {b"data": {b"x": {b"type": b"Int", b"value": 1}}, b"type": b"Record"})


class ObjectDeserializeTests(unittest.TestCase):
    def test_deserialize_int(self) -> None:
        msg = {"type": "Int", "value": 123}
        self.assertEqual(Object.deserialize(msg), Int(123))

    def test_deserialize_negative_int(self) -> None:
        msg = {"type": "Int", "value": -123}
        self.assertEqual(Object.deserialize(msg), Int(-123))

    def test_deserialize_str(self) -> None:
        msg = {"type": "String", "value": "abc"}
        self.assertEqual(Object.deserialize(msg), String("abc"))

    def test_deserialize_bytes(self) -> None:
        msg = {"type": "Bytes", "value": b"abc"}
        self.assertEqual(Object.deserialize(msg), Bytes(b"abc"))

    def test_deserialize_var(self) -> None:
        msg = {"type": "Var", "name": "abc"}
        self.assertEqual(Object.deserialize(msg), Var("abc"))

    def test_deserialize_bool(self) -> None:
        msg = {"type": "Bool", "value": True}
        self.assertEqual(Object.deserialize(msg), Bool(True))

    def test_deserialize_binary_add(self) -> None:
        msg = {
            "left": {"type": "Int", "value": 123},
            "op": "ADD",
            "right": {"type": "Int", "value": 456},
            "type": "Binop",
        }
        obj = Binop(BinopKind.ADD, Int(123), Int(456))
        self.assertEqual(Object.deserialize(msg), obj)


class SerializeTests(unittest.TestCase):
    def test_serialize_int(self) -> None:
        obj = Int(3)
        self.assertEqual(serialize(obj), b"d4:type3:Int5:valuei3ee")

    def test_serialize_str(self) -> None:
        obj = String("abc")
        self.assertEqual(serialize(obj), b"d4:type6:String5:value3:abce")

    def test_serialize_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(serialize(obj), b"d4:type5:Bytes5:value3:abce")

    def test_serialize_var(self) -> None:
        obj = Var("abc")
        self.assertEqual(serialize(obj), b"d4:name3:abc4:type3:Vare")

    def test_serialize_bool(self) -> None:
        obj = Bool(True)
        self.assertEqual(serialize(obj), b"d4:type4:Bool5:valuei1ee")

    def test_serialize_function(self) -> None:
        obj = Function(Var("x"), Binop(BinopKind.ADD, Int(1), Var("x")))
        self.assertEqual(
            serialize(obj),
            b"d3:argd4:name1:x4:type3:Vare4:bodyd4:leftd4:type3:Int5:valuei1ee2:op3:ADD5:rightd4:name1:x4:type3:Vare4:type5:Binope4:type8:Functione",
        )


class ScrapMonadTests(unittest.TestCase):
    def test_create_copies_env(self) -> None:
        env = {"a": Int(123)}
        result = ScrapMonad(env)
        self.assertEqual(result.env, env)
        self.assertIsNot(result.env, env)

    def test_bind_returns_new_monad(self) -> None:
        env = {"a": Int(123)}
        orig = ScrapMonad(env)
        result = orig.bind(Assign(Var("b"), Int(456)))
        self.assertEqual(orig.env, {"a": Int(123)})
        self.assertEqual(result.env, {"a": Int(123), "b": Int(456)})


def eval_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    program = args.program_file.read()
    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval_exp({}, ast)
    print(result)


def apply_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    tokens = tokenize(args.program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval_exp({}, ast)
    print(result)


def fetch(url: Object) -> Object:
    if not isinstance(url, String):
        raise TypeError(f"fetch expected String, but got {type(url).__name__}")
    with urllib.request.urlopen(url.value) as f:
        return String(f.read().decode("utf-8"))


def make_object(pyobj: object) -> Object:
    assert not isinstance(pyobj, Object)
    if isinstance(pyobj, bool):
        return Bool(pyobj)
    if isinstance(pyobj, int):
        return Int(pyobj)
    if isinstance(pyobj, str):
        return String(pyobj)
    if isinstance(pyobj, list):
        return List([make_object(o) for o in pyobj])
    if isinstance(pyobj, dict):
        # Assumed to only be called with JSON, so string keys.
        return Record({key: make_object(value) for key, value in pyobj.items()})
    raise NotImplementedError(type(pyobj))


def jsondecode(obj: Object) -> Object:
    if not isinstance(obj, String):
        raise TypeError(f"jsondecode expected String, but got {type(obj).__name__}")
    data = json.loads(obj.value)
    return make_object(data)


def listlength(obj: Object) -> Object:
    # TODO(max): Implement in scrapscript once list pattern matching is
    # implemented.
    if not isinstance(obj, List):
        raise TypeError(f"listlength expected List, but got {type(obj).__name__}")
    return Int(len(obj.items))


STDLIB = {
    "$$add": Closure({}, Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))),
    "$$fetch": NativeFunction("$$fetch", fetch),
    "$$jsondecode": NativeFunction("$$jsondecode", jsondecode),
    "$$serialize": NativeFunction("$$serialize", lambda obj: Bytes(serialize(obj))),
    "$$listlength": NativeFunction("$$listlength", listlength),
    "true": Bool(True),
    "false": Bool(False),
}


class Completer:
    def __init__(self, env: Env) -> None:
        self.env: Env = env
        self.matches: typing.List[str] = []

    def complete(self, text: str, state: int) -> Optional[str]:
        assert "@" not in text, "TODO: handle attr/index access"
        if state == 0:
            options = sorted(self.env.keys())
            if not text:
                self.matches = options[:]
            else:
                self.matches = [key for key in options if key.startswith(text)]
        try:
            return self.matches[state]
        except IndexError:
            return None


REPL_HISTFILE = os.path.expanduser(".scrap-history")


class ScrapRepl(code.InteractiveConsole):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.env: Env = STDLIB.copy()

    def enable_readline(self) -> None:
        assert readline, "Can't enable readline without readline module"
        if os.path.exists(REPL_HISTFILE):
            readline.read_history_file(REPL_HISTFILE)
        # what determines the end of a word; need to set so $ can be part of a
        # variable name
        readline.set_completer_delims(" \t\n;")
        # TODO(max): Add completion per scope, not just for global environment.
        readline.set_completer(Completer(self.env).complete)
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("tab: menu-complete")

    def finish_readline(self) -> None:
        assert readline, "Can't finish readline without readline module"
        histfile_size = 1000
        readline.set_history_length(histfile_size)
        readline.write_history_file(REPL_HISTFILE)

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        try:
            tokens = tokenize(source)
            logger.debug("Tokens: %s", tokens)
            ast = parse(tokens)
            logger.debug("AST: %s", ast)
            result = eval_exp(self.env, ast)
            assert isinstance(self.env, dict)  # for .update()/__setitem__
            if isinstance(result, EnvObject):
                self.env.update(result.env)
            else:
                self.env["_"] = result
            print(result)
        except UnexpectedEOFError:
            # Need to read more text
            return True
        except ParseError as e:
            print(f"Parse error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        return False


def repl_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    repl = ScrapRepl()
    if readline:
        repl.enable_readline()
    repl.interact(banner="")
    if readline:
        repl.finish_readline()


def test_command(args: argparse.Namespace) -> None:
    if args.debug:
        # pylint: disable=protected-access
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    # Pass on the rest of the positionals (for filtering tests and so on)
    unittest.main(argv=[__file__, *args.unittest_args])


def serve_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    with socketserver.TCPServer(("", args.port), ScrapReplServer) as httpd:
        print("serving at port", args.port)
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scrapscript")
    subparsers = parser.add_subparsers(dest="command")

    repl = subparsers.add_parser("repl")
    repl.set_defaults(func=repl_command)
    repl.add_argument("--debug", action="store_true")

    test = subparsers.add_parser("test")
    test.set_defaults(func=test_command)
    test.add_argument("unittest_args", nargs="*")
    test.add_argument("--debug", action="store_true")

    eval_ = subparsers.add_parser("eval")
    eval_.set_defaults(func=eval_command)
    eval_.add_argument("program_file", type=argparse.FileType("r"))
    eval_.add_argument("--debug", action="store_true")

    apply = subparsers.add_parser("apply")
    apply.set_defaults(func=apply_command)
    apply.add_argument("program")
    apply.add_argument("--debug", action="store_true")

    serve = subparsers.add_parser("serve")
    serve.set_defaults(func=serve_command)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if not args.command:
        args.debug = False
        repl_command(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
