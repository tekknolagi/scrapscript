#!/usr/bin/env python3.10
import argparse
import base64
import code
import enum
import json
import logging
import os
import re
import sys
import unittest
import urllib.request
from dataclasses import dataclass
from enum import auto
from types import ModuleType
from typing import Any, Callable, Mapping, Optional

readline: Optional[ModuleType]
try:
    import readline
except ImportError:
    readline = None


logger = logging.getLogger(__name__)


def is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in ("$", "'", "_")


class Lexer:
    # TODO(chris): Add position information to tokens, enable showing position of error in program
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
            raise UnexpectedEOFError("while reading token")
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
            if self.has_input() and self.peek_char() == "-":
                self.read_comment()
                return self.read_one()
            if self.has_input() and self.peek_char().isdigit():
                return self.read_number(c)
            return self.read_op(c)
        if c == "~":
            if self.has_input() and self.peek_char() == "~":
                self.read_char()
                return self.read_bytes()
            raise ParseError(f"unexpected token {c!r}")
        if c.isdigit():
            return self.read_number(c)
        if c in "()[]{}":
            return c
        if c in OPER_CHARS:
            return self.read_op(c)
        if is_identifier_char(c):
            return self.read_var(c)
        raise ParseError(f"unexpected token {c!r}")

    def read_string(self) -> str:
        buf = ""
        while self.has_input():
            if (c := self.read_char()) == '"':
                break
            buf += c
        else:
            raise UnexpectedEOFError("while reading string")
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
        while self.has_input() and is_identifier_char(c := self.peek_char()):
            self.read_char()
            buf += c
        return buf

    def read_bytes(self) -> str:
        buf = ""
        while self.has_input():
            if (c := self.read_char()).isspace():
                break
            buf += c
        if not (len(buf) >= 2 and buf[:2].isnumeric()):
            buf = "64'" + buf
        return "~~" + buf


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
    ">>": lp(14),
    "<<": lp(14),
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
    ">+": lp(10),
    "+<": rp(10),
    "->": lp(5),
    "|": rp(4.5),
    ":": lp(4.5),
    "|>": rp(4.11),
    "<|": lp(4.11),
    "=": rp(4),
    "@": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    "]": xp(1),
}


OPER_CHARS = set("".join(PS.keys()))
assert " " not in OPER_CHARS


class ParseError(Exception):
    pass


class UnexpectedEOFError(ParseError):
    pass


def parse_assign(tokens: list[str], p: float = 0) -> "Assign":
    assign = parse(tokens, p)
    if not isinstance(assign, Assign):
        raise ParseError("failed to parse variable assignment in record constructor")
    return assign


def parse(tokens: list[str], p: float = 0) -> "Object":
    if not tokens:
        raise UnexpectedEOFError("unexpected end of input")
    token = tokens.pop(0)
    l: Object
    sha_prefix = "$sha1'"
    dollar_dollar_prefix = "$$"
    tilde_tilde_prefix = "~~"
    # TODO(max): Tag tokens out of the lexer so we don't have to re-interpret
    # them here.
    if token.isnumeric() or (token[0] == "-" and token[1:].isnumeric()):
        l = Int(int(token))
    elif token.isidentifier():
        # TODO: Handle kebab case vars
        l = Var(token)
    elif token.startswith(sha_prefix) and token[len(sha_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith(dollar_dollar_prefix) and token[len(dollar_dollar_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith(tilde_tilde_prefix):
        assert len(token) >= len("~~XX'")
        assert "'" in token, "expected base in bytes"
        base, without_base = token[len(tilde_tilde_prefix) :].split("'")
        assert base.isnumeric(), f"unexpected base {base!r} in {token!r}"
        if base == "85":
            l = Bytes(base64.b85decode(without_base))
        elif base == "64":
            l = Bytes(base64.b64decode(without_base))
        elif base == "32":
            l = Bytes(base64.b32decode(without_base))
        elif base == "16":
            l = Bytes(base64.b16decode(without_base))
        else:
            raise ParseError(f"unexpected base {base!r} in {token!r}")
    elif token.startswith('"') and token.endswith('"'):
        l = String(token[1:-1])
    elif token == "|":
        expr = parse(tokens, 5)  # TODO: make this work for larger arities
        if not isinstance(expr, Function):
            raise ParseError(f"expected function in match expression {expr!r}")
        cases = [MatchCase(expr.arg, expr.body)]
        while tokens and tokens[0] == "|":
            tokens.pop(0)
            expr = parse(tokens, 5)  # TODO: make this work for larger arities
            if not isinstance(expr, Function):
                raise ParseError(f"expected function in match expression {expr!r}")
            cases.append(MatchCase(expr.arg, expr.body))
        l = MatchFunction(cases)
    elif token == "(":
        if tokens[0] == ")":
            l = Hole()
        else:
            l = parse(tokens, 0)
        tokens.pop(0)
    elif token == "[":
        l = List([])
        token = tokens[0]
        if token == "]":
            tokens.pop(0)
        else:
            l.items.append(parse(tokens, 2))
            while tokens.pop(0) != "]":
                l.items.append(parse(tokens, 2))
    elif token == "{":
        l = Record({})
        token = tokens[0]
        if token == "}":
            tokens.pop(0)
        else:
            assign = parse_assign(tokens, 2)
            l.data[assign.name.name] = assign.value
            while tokens.pop(0) != "}":
                # TODO: Implement .. and ... operators
                assign = parse_assign(tokens, 2)
                l.data[assign.name.name] = assign.value
    else:
        raise ParseError(f"unexpected token {token!r}")

    while True:
        if not tokens:
            break
        op = tokens[0]
        if op in ")]}":
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
            if not isinstance(l, Var):
                raise ParseError(f"expected variable in assignment {l!r}")
            l = Assign(l, parse(tokens, pr))
        elif op == "->":
            l = Function(l, parse(tokens, pr))
        elif op == "":
            l = Apply(l, parse(tokens, pr))
        elif op == "|>":
            l = Apply(parse(tokens, pr), l)
        elif op == "<|":
            l = Apply(l, parse(tokens, pr))
        elif op == ">>":
            l = Compose(l, parse(tokens, pr))
        elif op == "<<":
            l = Compose(parse(tokens, pr), l)
        elif op == ".":
            l = Where(l, parse(tokens, pr))
        elif op == "?":
            l = Assert(l, parse(tokens, pr))
        elif op == "@":
            # TODO: revisit whether to use @ or . for field access
            l = Access(l, parse(tokens, pr))
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
class Bytes(Object):
    value: bytes


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bool(Object):
    value: bool


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Hole(Object):
    pass


Env = Mapping[str, Object]


class BinopKind(enum.Enum):
    ADD = auto()
    STRING_CONCAT = auto()
    LIST_CONS = auto()
    LIST_APPEND = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    HASTYPE = auto()
    PIPE = auto()
    REVERSE_PIPE = auto()
    RIGHT_EVAL = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "++": cls.STRING_CONCAT,
            ">+": cls.LIST_CONS,
            "+<": cls.LIST_APPEND,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
            "==": cls.EQUAL,
            "/=": cls.NOT_EQUAL,
            "<": cls.LESS,
            ">": cls.GREATER,
            "<=": cls.LESS_EQUAL,
            ">=": cls.GREATER_EQUAL,
            ":": cls.HASTYPE,
            "|>": cls.PIPE,
            "<|": cls.REVERSE_PIPE,
            "!": cls.RIGHT_EVAL,
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


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchCase(Object):
    pattern: Object
    body: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchFunction(Object):
    cases: list[MatchCase]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunction(Object):
    func: Callable[[Object], Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Closure(Object):
    env: Env
    func: Function | MatchFunction


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Record(Object):
    data: dict[str, Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Access(Object):
    obj: Object
    at: Object


def unpack_int(obj: Object) -> int:
    if not isinstance(obj, Int):
        raise TypeError(f"expected Int, got {type(obj).__name__}")
    return obj.value


def eval_int(env: Env, exp: Object) -> int:
    result = eval(env, exp)
    return unpack_int(result)


def eval_str(env: Env, exp: Object) -> str:
    result = eval(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


def eval_list(env: Env, exp: Object) -> list[Object]:
    result = eval(env, exp)
    if not isinstance(result, List):
        raise TypeError(f"expected List, got {type(result).__name__}")
    return result.items


BINOP_HANDLERS: dict[BinopKind, Callable[[Env, Object, Object], Object]] = {
    BinopKind.ADD: lambda env, x, y: Int(eval_int(env, x) + eval_int(env, y)),
    BinopKind.STRING_CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    # We have type: ignore because we haven't (re)defined eval yet.
    BinopKind.LIST_CONS: lambda env, x, y: List([eval(env, x)] + eval_list(env, y)),  # type: ignore [arg-type]
    BinopKind.LIST_APPEND: lambda env, x, y: List(eval_list(env, x) + [eval(env, y)]),  # type: ignore [arg-type]
    BinopKind.SUB: lambda env, x, y: Int(eval_int(env, x) - eval_int(env, y)),
    BinopKind.MUL: lambda env, x, y: Int(eval_int(env, x) * eval_int(env, y)),
    BinopKind.DIV: lambda env, x, y: Int(eval_int(env, x) // eval_int(env, y)),
    # We have type: ignore because we haven't (re)defined eval yet.
    BinopKind.EQUAL: lambda env, x, y: Bool(eval(env, x) == eval(env, y)),  # type: ignore [arg-type]
    # We have type: ignore because we haven't (re)defined eval yet.
    BinopKind.RIGHT_EVAL: lambda env, x, y: eval(env, y),  # type: ignore [arg-type]
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
    raise NotImplementedError("TODO: match")


# pylint: disable=redefined-builtin
def eval(env: Env, exp: Object) -> Object:
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
        return List([eval(env, item) for item in exp.items])
    if isinstance(exp, Record):
        return Record({k: eval(env, exp.data[k]) for k in exp.data})
    if isinstance(exp, Assign):
        # TODO(max): Rework this. There's something about matching that we need
        # to figure out and implement.
        assert isinstance(exp.name, Var)
        value = eval(env, exp.value)
        return EnvObject({**env, exp.name.name: value})
    if isinstance(exp, Where):
        res_env = eval(env, exp.binding)
        assert isinstance(res_env, EnvObject)
        new_env = {**env, **res_env.env}
        return eval(new_env, exp.body)
    if isinstance(exp, Assert):
        cond = eval(env, exp.cond)
        if cond != Bool(True):
            raise AssertionError(f"condition {exp.cond} failed")
        return eval(env, exp.value)
    if isinstance(exp, Function):
        if not isinstance(exp.arg, Var):
            raise RuntimeError(f"expected variable in function definition {exp.arg}")
        return Closure(env, exp)
    if isinstance(exp, MatchFunction):
        return Closure(env, exp)
    if isinstance(exp, Apply):
        if isinstance(exp.func, Var) and exp.func.name == "$$quote":
            return exp.arg
        callee = eval(env, exp.func)
        if isinstance(callee, NativeFunction):
            arg = eval(env, exp.arg)
            return callee.func(arg)
        if not isinstance(callee, Closure):
            raise TypeError(f"attempted to apply a non-closure of type {type(callee).__name__}")
        arg = eval(env, exp.arg)
        if isinstance(callee.func, Function):
            assert isinstance(callee.func.arg, Var)
            new_env = {**callee.env, callee.func.arg.name: arg}
            return eval(new_env, callee.func.body)
        elif isinstance(callee.func, MatchFunction):
            # TODO(max): Implement MatchClosure
            arg = eval(env, exp.arg)
            for case in callee.func.cases:
                m = match(arg, case.pattern)
                if m is None:
                    continue
                return eval({**env, **m}, case.body)
            raise MatchError("no matching cases")
        else:
            raise TypeError(f"attempted to apply a non-function of type {type(callee.func).__name__}")
    if isinstance(exp, Access):
        obj = eval(env, exp.obj)
        if isinstance(obj, Record):
            if not isinstance(exp.at, Var):
                raise TypeError(f"cannot access record field using {type(exp.at).__name__}, expected a field name")
            if exp.at.name not in obj.data:
                raise NameError(f"no assignment to {exp.at.name} found in record")
            return obj.data[exp.at.name]
        elif isinstance(obj, List):
            access_at = eval(env, exp.at)
            if not isinstance(access_at, Int):
                raise TypeError(f"cannot index into list using type {type(access_at).__name__}, expected integer")
            if access_at.value < 0 or access_at.value >= len(obj.items):
                raise ValueError(f"index {access_at.value} out of bounds for list")
            return obj.items[access_at.value]
        raise TypeError(f"attempted to access from type {type(obj).__name__}")
    if isinstance(exp, Compose):
        clo_inner = eval(env, exp.inner)
        clo_outer = eval(env, exp.outer)
        return Closure({}, Function(Var("x"), Apply(clo_outer, Apply(clo_inner, Var("x")))))
    raise NotImplementedError(f"eval not implemented for {exp}")


class TokenizerTests(unittest.TestCase):
    def test_tokenize_digit(self) -> None:
        self.assertEqual(tokenize("1"), ["1"])

    def test_tokenize_multiple_digits(self) -> None:
        self.assertEqual(tokenize("123"), ["123"])

    def test_tokenize_negative_int(self) -> None:
        self.assertEqual(tokenize("-123"), ["-123"])

    def test_tokenize_binop(self) -> None:
        self.assertEqual(tokenize("1 + 2"), ["1", "+", "2"])

    def test_tokenize_binop_no_spaces(self) -> None:
        self.assertEqual(tokenize("1+2"), ["1", "+", "2"])

    @unittest.skip("TODO(max): Move negative integers into the parser")
    def test_tokenize_binary_sub_no_spaces(self) -> None:
        self.assertEqual(tokenize("1-2"), ["1", "-", "2"])

    def test_tokenize_binop_var(self) -> None:
        ops = ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                self.assertEqual(tokenize(f"a {op} b"), ["a", op, "b"])
                self.assertEqual(tokenize(f"a{op}b"), ["a", op, "b"])

    def test_tokenize_var(self) -> None:
        self.assertEqual(tokenize("abc"), ["abc"])

    def test_tokenize_dollar_sha1_var(self) -> None:
        self.assertEqual(tokenize("$sha1'foo"), ["$sha1'foo"])

    def test_tokenize_dollar_dollar_var(self) -> None:
        self.assertEqual(tokenize("$$bills"), ["$$bills"])

    def test_ignore_whitespace(self) -> None:
        self.assertEqual(tokenize("1\n+\t2"), ["1", "+", "2"])

    def test_ignore_line_comment(self) -> None:
        self.assertEqual(tokenize("-- 1\n2"), ["2"])

    def test_tokenize_string(self) -> None:
        self.assertEqual(tokenize('"hello"'), ['"hello"'])

    def test_tokenize_string_with_spaces(self) -> None:
        self.assertEqual(tokenize('"hello world"'), ['"hello world"'])

    def test_tokenize_string_missing_end_quote_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(UnexpectedEOFError, "while reading string"):
            tokenize('"hello')

    def test_tokenize_with_trailing_whitespace(self) -> None:
        self.assertEqual(tokenize("- "), ["-"])
        self.assertEqual(tokenize("-- "), [])
        self.assertEqual(tokenize("+ "), ["+"])
        self.assertEqual(tokenize("123 "), ["123"])
        self.assertEqual(tokenize("abc "), ["abc"])
        self.assertEqual(tokenize("[ "), ["["])
        self.assertEqual(tokenize("] "), ["]"])

    def test_tokenize_empty_list(self) -> None:
        self.assertEqual(tokenize("[ ]"), ["[", "]"])

    def test_tokenize_empty_list_with_no_spaces(self) -> None:
        self.assertEqual(tokenize("[ ]"), ["[", "]"])

    def test_tokenize_list_with_items(self) -> None:
        self.assertEqual(tokenize("[ 1 , 2 ]"), ["[", "1", ",", "2", "]"])

    def test_tokenize_list_with_no_spaces(self) -> None:
        self.assertEqual(tokenize("[1,2]"), ["[", "1", ",", "2", "]"])

    def test_tokenize_function(self) -> None:
        self.assertEqual(tokenize("a -> b -> a + b"), ["a", "->", "b", "->", "a", "+", "b"])

    def test_tokenize_function_with_no_spaces(self) -> None:
        self.assertEqual(tokenize("a->b->a+b"), ["a", "->", "b", "->", "a", "+", "b"])

    def test_tokenize_where(self) -> None:
        self.assertEqual(tokenize("a . b"), ["a", ".", "b"])

    def test_tokenize_assert(self) -> None:
        self.assertEqual(tokenize("a ? b"), ["a", "?", "b"])

    def test_tokenize_hastype(self) -> None:
        self.assertEqual(tokenize("a : b"), ["a", ":", "b"])

    def test_tokenize_minus_returns_minus(self) -> None:
        self.assertEqual(tokenize("-"), ["-"])

    def test_tokenize_tilde_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~")

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~=")

    def test_tokenize_tilde_tilde_equals_returns_empty_bytes(self) -> None:
        self.assertEqual(tokenize("~~"), ["~~64'"])

    def test_tokenize_bytes_returns_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~QUJD"), ["~~64'QUJD"])

    def test_tokenize_bytes_base85(self) -> None:
        self.assertEqual(tokenize("~~85'K|(_"), ["~~85'K|(_"])

    def test_tokenize_bytes_base64(self) -> None:
        self.assertEqual(tokenize("~~64'QUJD"), ["~~64'QUJD"])

    def test_tokenize_bytes_base32(self) -> None:
        self.assertEqual(tokenize("~~32'IFBEG==="), ["~~32'IFBEG==="])

    def test_tokenize_bytes_base16(self) -> None:
        self.assertEqual(tokenize("~~16'414243"), ["~~16'414243"])

    def test_tokenize_hole(self) -> None:
        self.assertEqual(tokenize("()"), ["(", ")"])

    def test_tokenize_hole_with_spaces(self) -> None:
        self.assertEqual(tokenize("( )"), ["(", ")"])

    def test_tokenize_parenthetical_expression(self) -> None:
        self.assertEqual(tokenize("(1+2)"), ["(", "1", "+", "2", ")"])

    def test_tokenize_pipe(self) -> None:
        self.assertEqual(
            tokenize("1 |> f . f = a -> a + 1"),
            ["1", "|>", "f", ".", "f", "=", "a", "->", "a", "+", "1"],
        )

    def test_tokenize_reverse_pipe(self) -> None:
        self.assertEqual(
            tokenize("f <| 1 . f = a -> a + 1"),
            ["f", "<|", "1", ".", "f", "=", "a", "->", "a", "+", "1"],
        )

    def test_tokenize_record_no_fields(self) -> None:
        self.assertEqual(
            tokenize("{ }"),
            ["{", "}"],
        )

    def test_tokenize_record_no_fields_no_spaces(self) -> None:
        self.assertEqual(
            tokenize("{}"),
            ["{", "}"],
        )

    def test_tokenize_record_one_field(self) -> None:
        self.assertEqual(
            tokenize("{ a = 4 }"),
            ["{", "a", "=", "4", "}"],
        )

    def test_tokenize_record_multiple_fields(self) -> None:
        self.assertEqual(
            tokenize('{ a = 4, b = "z" }'),
            ["{", "a", "=", "4", ",", "b", "=", '"z"', "}"],
        )

    def test_tokenize_record_access(self) -> None:
        self.assertEqual(
            tokenize("r@a"),
            ["r", "@", "a"],
        )

    def test_tokenize_right_eval(self) -> None:
        self.assertEqual(tokenize("a!b"), ["a", "!", "b"])

    def test_tokenize_match(self) -> None:
        self.assertEqual(
            tokenize("g = | 1 -> 2 | 2 -> 3"),
            ["g", "=", "|", "1", "->", "2", "|", "2", "->", "3"],
        )

    def test_tokenize_compose(self) -> None:
        self.assertEqual(
            tokenize("f >> g"),
            ["f", ">>", "g"],
        )

    def test_tokenize_compose_reverse(self) -> None:
        self.assertEqual(
            tokenize("f << g"),
            ["f", "<<", "g"],
        )


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedEOFError) as ctx:
            parse([])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_digit_returns_int(self) -> None:
        self.assertEqual(parse(["1"]), Int(1))

    def test_parse_digits_returns_int(self) -> None:
        self.assertEqual(parse(["123"]), Int(123))

    def test_parse_negative_int_returns_int(self) -> None:
        self.assertEqual(parse(["-123"]), Int(-123))

    def test_parse_var_returns_var(self) -> None:
        self.assertEqual(parse(["abc_123"]), Var("abc_123"))

    def test_parse_sha_var_returns_var(self) -> None:
        self.assertEqual(parse(["$sha1'abc"]), Var("$sha1'abc"))

    def test_parse_sha_var_without_quote_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse(["$sha1abc"])

    def test_parse_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse(["$"])

    def test_parse_dollar_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse(["$$"])

    def test_parse_sha_var_without_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse(["sha1'abc"])

    def test_parse_dollar_dollar_var_returns_var(self) -> None:
        self.assertEqual(parse(["$$bills"]), Var("$$bills"))

    def test_parse_bytes_returns_bytes(self) -> None:
        self.assertEqual(parse(["~~64'QUJD"]), Bytes(b"ABC"))

    def test_parse_binary_add_returns_binop(self) -> None:
        self.assertEqual(parse(["1", "+", "2"]), Binop(BinopKind.ADD, Int(1), Int(2)))

    def test_parse_binary_add_right_returns_binop(self) -> None:
        self.assertEqual(
            parse(["1", "+", "2", "+", "3"]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_right(self) -> None:
        self.assertEqual(
            parse(["1", "+", "2", "*", "3"]),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_left(self) -> None:
        self.assertEqual(
            parse(["1", "*", "2", "+", "3"]),
            Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3)),
        )

    def test_parse_binary_str_concat_returns_binop(self) -> None:
        self.assertEqual(
            parse(['"abc"', "++", '"def"']),
            Binop(BinopKind.STRING_CONCAT, String("abc"), String("def")),
        )

    def test_parse_binary_list_cons_returns_binop(self) -> None:
        self.assertEqual(
            parse(["a", ">+", "b"]),
            Binop(BinopKind.LIST_CONS, Var("a"), Var("b")),
        )

    def test_parse_binary_list_append_returns_binop(self) -> None:
        self.assertEqual(
            parse(["a", "+<", "b"]),
            Binop(BinopKind.LIST_APPEND, Var("a"), Var("b")),
        )

    def test_parse_binary_op_returns_binop(self) -> None:
        ops = ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                kind = BinopKind.from_str(op)
                self.assertEqual(parse(["a", op, "b"]), Binop(kind, Var("a"), Var("b")))

    def test_parse_empty_list(self) -> None:
        self.assertEqual(
            parse(["[", "]"]),
            List([]),
        )

    def test_parse_list_of_ints_returns_list(self) -> None:
        self.assertEqual(
            parse(["[", "1", ",", "2", "]"]),
            List([Int(1), Int(2)]),
        )

    def test_parse_assign(self) -> None:
        self.assertEqual(
            parse(["a", "=", "1"]),
            Assign(Var("a"), Int(1)),
        )

    def test_parse_function_one_arg_returns_function(self) -> None:
        self.assertEqual(
            parse(["a", "->", "a", "+", "1"]),
            Function(Var("a"), Binop(BinopKind.ADD, Var("a"), Int(1))),
        )

    def test_parse_function_two_args_returns_functions(self) -> None:
        self.assertEqual(
            parse(["a", "->", "b", "->", "a", "+", "b"]),
            Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))),
        )

    def test_parse_assign_function(self) -> None:
        self.assertEqual(
            parse(["id", "=", "x", "->", "x"]),
            Assign(Var("id"), Function(Var("x"), Var("x"))),
        )

    def test_parse_function_application_one_arg(self) -> None:
        self.assertEqual(parse(["f", "a"]), Apply(Var("f"), Var("a")))

    def test_parse_function_application_two_args(self) -> None:
        self.assertEqual(parse(["f", "a", "b"]), Apply(Apply(Var("f"), Var("a")), Var("b")))

    def test_parse_where(self) -> None:
        self.assertEqual(parse(["a", ".", "b"]), Where(Var("a"), Var("b")))

    def test_parse_nested_where(self) -> None:
        self.assertEqual(parse(["a", ".", "b", ".", "c"]), Where(Where(Var("a"), Var("b")), Var("c")))

    def test_parse_assert(self) -> None:
        self.assertEqual(parse(["a", "?", "b"]), Assert(Var("a"), Var("b")))

    def test_parse_nested_assert(self) -> None:
        self.assertEqual(parse(["a", "?", "b", "?", "c"]), Assert(Assert(Var("a"), Var("b")), Var("c")))

    def test_parse_mixed_assert_where(self) -> None:
        self.assertEqual(parse(["a", "?", "b", ".", "c"]), Where(Assert(Var("a"), Var("b")), Var("c")))

    def test_parse_hastype(self) -> None:
        self.assertEqual(parse(["a", ":", "b"]), Binop(BinopKind.HASTYPE, Var("a"), Var("b")))

    def test_parse_hole(self) -> None:
        self.assertEqual(parse(["(", ")"]), Hole())

    def test_parse_parenthesized_expression(self) -> None:
        self.assertEqual(parse(["(", "1", "+", "2", ")"]), Binop(BinopKind.ADD, Int(1), Int(2)))

    def test_parse_parenthesized_add_mul(self) -> None:
        self.assertEqual(
            parse(["(", "1", "+", "2", ")", "*", "3"]),
            Binop(BinopKind.MUL, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)),
        )

    def test_parse_pipe(self) -> None:
        self.assertEqual(
            parse(["1", "|>", "f"]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_pipe(self) -> None:
        self.assertEqual(
            parse(["1", "|>", "f", "|>", "g"]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_reverse_pipe(self) -> None:
        self.assertEqual(
            parse(["f", "<|", "1"]),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_reverse_pipe(self) -> None:
        self.assertEqual(
            parse(["g", "<|", "f", "<|", "1"]),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_empty_record(self) -> None:
        self.assertEqual(parse(["{", "}"]), Record({}))

    def test_parse_record_single_field(self) -> None:
        self.assertEqual(parse(["{", "a", "=", "4", "}"]), Record({"a": Int(4)}))

    def test_parse_record_with_expression(self) -> None:
        self.assertEqual(
            parse(["{", "a", "=", "1", "+", "2", "}"]),
            Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))}),
        )

    def test_parse_record_multiple_fields(self) -> None:
        self.assertEqual(
            parse(["{", "a", "=", "4", ",", "b", "=", '"z"', "}"]), Record({"a": Int(4), "b": String("z")})
        )

    def test_non_variable_in_assignment_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(["3", "=", "4"])
        self.assertEqual(ctx.exception.args[0], "expected variable in assignment Int(value=3)")

    def test_non_assign_in_record_constructor_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(["{", "1", ",", "2", "}"])
        self.assertEqual(ctx.exception.args[0], "failed to parse variable assignment in record constructor")

    def test_parse_right_eval_returns_binop(self) -> None:
        self.assertEqual(parse(["a", "!", "b"]), Binop(BinopKind.RIGHT_EVAL, Var("a"), Var("b")))

    def test_parse_right_eval_with_defs_returns_binop(self) -> None:
        self.assertEqual(
            parse(["a", "!", "b", ".", "c"]),
            Binop(BinopKind.RIGHT_EVAL, Var("a"), Where(Var("b"), Var("c"))),
        )

    def test_parse_match_no_cases_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(["|"])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_match_one_case(self) -> None:
        self.assertEqual(
            parse(["|", "1", "->", "2"]),
            MatchFunction([MatchCase(Int(1), Int(2))]),
        )

    def test_parse_match_two_cases(self) -> None:
        self.assertEqual(
            parse(["|", "1", "->", "2", "|", "2", "->", "3"]),
            MatchFunction(
                [
                    MatchCase(Int(1), Int(2)),
                    MatchCase(Int(2), Int(3)),
                ]
            ),
        )

    def test_parse_compose(self) -> None:
        self.assertEqual(parse(["f", ">>", "g"]), Compose(Var("f"), Var("g")))

    def test_parse_compose_reverse(self) -> None:
        self.assertEqual(parse(["f", "<<", "g"]), Compose(Var("g"), Var("f")))

    def test_parse_double_compose(self) -> None:
        self.assertEqual(
            parse(["f", "<<", "g", "<<", "h"]),
            Compose(Compose(Var("h"), Var("g")), Var("f")),
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

    def test_match_record_with_fewer_fields_in_pattern_returns_intersection(self) -> None:
        # TODO(max): Should this be the case? I feel like we should not match
        # without explicitly using spread.
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x")}),
            ),
            {"x": Int(1)},
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

    def test_match_record_with_non_matching_const_returns_empty_dict(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(3), "y": Var("y")}),
            ),
            None,
        )


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval({}, exp), Int(5))

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        self.assertEqual(eval({}, exp), String("xyz"))

    def test_eval_bytes_returns_bytes(self) -> None:
        exp = Bytes(b"xyz")
        self.assertEqual(eval({}, exp), Bytes(b"xyz"))

    def test_eval_true_returns_true(self) -> None:
        self.assertEqual(eval({}, Bool(True)), Bool(True))

    def test_eval_false_returns_false(self) -> None:
        self.assertEqual(eval({}, Bool(False)), Bool(False))

    def test_eval_with_non_existent_var_raises_name_error(self) -> None:
        exp = Var("no")
        with self.assertRaises(NameError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "name 'no' is not defined")

    def test_eval_with_bound_var_returns_value(self) -> None:
        exp = Var("yes")
        env = {"yes": Int(123)}
        self.assertEqual(eval(env, exp), Int(123))

    def test_eval_with_binop_add_returns_sum(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(eval({}, exp), Int(3))

    def test_eval_with_nested_binop(self) -> None:
        exp = Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        self.assertEqual(eval({}, exp), Int(6))

    def test_eval_with_binop_add_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), String("hello"))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected Int, got String")

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        self.assertEqual(eval({}, exp), Int(-1))

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        self.assertEqual(eval({}, exp), Int(6))

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(2), Int(3))
        self.assertEqual(eval({}, exp), Int(0))

    def test_eval_with_binop_equal_with_equal_returns_true(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(1))
        self.assertEqual(eval({}, exp), Bool(True))

    def test_eval_with_binop_equal_with_inequal_returns_false(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(2))
        self.assertEqual(eval({}, exp), Bool(False))

    def test_eval_with_binop_concat_with_strings_returns_string(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String("hello"), String(" world"))
        self.assertEqual(eval({}, exp), String("hello world"))

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, Int(123), String(" world"))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String(" world"), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_cons_with_int_list_returns_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, Int(1), List([Int(2), Int(3)]))
        self.assertEqual(eval({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_binop_cons_with_list_list_returns_nested_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), List([]))
        self.assertEqual(eval({}, exp), List([List([])]))

    def test_eval_with_binop_cons_with_list_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected List, got Int")

    def test_eval_with_list_append(self) -> None:
        exp = Binop(BinopKind.LIST_APPEND, List([Int(1), Int(2)]), Int(3))
        self.assertEqual(eval({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        self.assertEqual(eval({}, exp), List([Int(3), Int(7)]))

    def test_eval_with_function_returns_function(self) -> None:
        exp = Function(Var("x"), Var("x"))
        self.assertEqual(eval({}, exp), Closure({}, Function(Var("x"), Var("x"))))

    def test_eval_assign_returns_env_object(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        result = eval(env, exp)
        self.assertEqual(result, EnvObject({"a": Int(1)}))

    def test_eval_assign_does_not_modify_env(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        eval(env, exp)
        self.assertEqual(env, {})

    def test_eval_where_evaluates_in_order(self) -> None:
        exp = Where(Binop(BinopKind.ADD, Var("a"), Int(2)), Assign(Var("a"), Int(1)))
        env: Env = {}
        self.assertEqual(eval(env, exp), Int(3))
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
        self.assertEqual(eval(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_assert_with_truthy_cond_returns_value(self) -> None:
        exp = Assert(Int(123), Bool(True))
        self.assertEqual(eval({}, exp), Int(123))

    def test_eval_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        exp = Assert(Int(123), Bool(False))
        with self.assertRaisesRegex(AssertionError, re.escape("condition Bool(value=False) failed")):
            eval({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), Bool(True)), Bool(True))
        self.assertEqual(eval({}, exp), Int(123))

    def test_eval_hole(self) -> None:
        exp = Hole()
        self.assertEqual(eval({}, exp), Hole())

    def test_eval_function_application_one_arg(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))), Int(2))
        self.assertEqual(eval({}, exp), Int(3))

    def test_eval_function_application_two_args(self) -> None:
        exp = Apply(
            Apply(Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))), Int(3)),
            Int(2),
        )
        self.assertEqual(eval({}, exp), Int(5))

    def test_eval_function_returns_closure_with_captured_env(self) -> None:
        exp = Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        res = eval({"y": Int(5)}, exp)
        self.assertIsInstance(res, Closure)
        assert isinstance(res, Closure)  # for mypy
        self.assertEqual(res.env, {"y": Int(5)})

    def test_eval_function_capture_env(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y"))), Int(2))
        self.assertEqual(eval({"y": Int(5)}, exp), Int(7))

    def test_eval_non_function_raises_type_error(self) -> None:
        exp = Apply(Int(3), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to apply a non-closure of type Int")):
            eval({}, exp)

    def test_eval_access_from_invalid_object_raises_type_error(self) -> None:
        exp = Access(Int(4), String("x"))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to access from type Int")):
            eval({}, exp)

    def test_eval_record_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Int(0))
        with self.assertRaisesRegex(
            TypeError, re.escape("cannot access record field using Int, expected a field name")
        ):
            eval({}, exp)

    def test_eval_record_access_with_unknown_accessor_raises_name_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("b"))
        with self.assertRaisesRegex(NameError, re.escape("no assignment to b found in record")):
            eval({}, exp)

    def test_eval_record_access(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(eval({}, exp), Int(4))

    def test_eval_list_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(List([Int(4)]), String("hello"))
        with self.assertRaisesRegex(TypeError, re.escape("cannot index into list using type String, expected integer")):
            eval({}, exp)

    def test_eval_list_access_with_out_of_bounds_accessor_raises_value_error(self) -> None:
        exp = Access(List([Int(1), Int(2), Int(3)]), Int(4))
        with self.assertRaisesRegex(ValueError, re.escape("index 4 out of bounds for list")):
            eval({}, exp)

    def test_eval_list_access(self) -> None:
        exp = Access(List([String("a"), String("b"), String("c")]), Int(2))
        self.assertEqual(eval({}, exp), String("c"))

    def test_right_eval_evaluates_right_hand_side(self) -> None:
        exp = Binop(BinopKind.RIGHT_EVAL, Int(1), Int(2))
        self.assertEqual(eval({}, exp), Int(2))

    def test_match_no_cases_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([]), Int(1))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval({}, exp)

    def test_match_int_with_equal_int_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(1))
        self.assertEqual(eval({}, exp), Int(2))

    def test_match_int_with_inequal_int_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(3))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval({}, exp)

    def test_match_string_with_equal_string_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("a"))
        self.assertEqual(eval({}, exp), String("b"))

    def test_match_string_with_inequal_string_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("c"))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval({}, exp)

    def test_match_falls_through_to_next(self) -> None:
        exp = Apply(
            MatchFunction([MatchCase(pattern=Int(3), body=Int(4)), MatchCase(pattern=Int(1), body=Int(2))]), Int(1)
        )
        self.assertEqual(eval({}, exp), Int(2))

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
        self.assertEqual(eval(env, exp), expected)

    def test_eval_compose_apply(self) -> None:
        exp = Apply(
            Compose(
                Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
                Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))),
            ),
            Int(4),
        )
        self.assertEqual(
            eval({}, exp),
            Int(14),
        )

    def test_eval_native_function_returns_function(self) -> None:
        exp = NativeFunction(lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertIs(eval({}, exp), exp)

    def test_eval_apply_native_function_calls_function(self) -> None:
        exp = Apply(NativeFunction(lambda x: Int(x.value * 2)), Int(3))  # type: ignore [attr-defined]
        self.assertEqual(eval({}, exp), Int(6))

    def test_eval_apply_quote_returns_ast(self) -> None:
        ast = Binop(BinopKind.ADD, Int(1), Int(2))
        exp = Apply(Var("$$quote"), ast)
        self.assertIs(eval({}, exp), ast)


class EndToEndTests(unittest.TestCase):
    def _run(self, text: str, env: Optional[Env] = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        return eval(env or {}, ast)

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

    def test_pipe(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2)"), Int(3))

    def test_pipe_nested(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2) |> (b -> b * 2)"), Int(6))

    def test_reverse_pipe(self) -> None:
        self.assertEqual(self._run("(a -> a + 2) <| 1"), Int(3))

    def test_reverse_pipe_nested(self) -> None:
        self.assertEqual(self._run("(b -> b * 2) <| (a -> a + 2) <| 1"), Int(6))

    def test_recursive_function(self) -> None:
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


def eval_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    program = args.program_file.read()
    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval({}, ast)
    print(result)


def apply_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    tokens = tokenize(args.program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval({}, ast)
    print(result)


def fetch(url: Object) -> Object:
    if not isinstance(url, String):
        raise TypeError(f"fetch Expected String, but got {type(url).__name__}")
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
        raise TypeError(f"jsondecode Expected String, but got {type(obj).__name__}")
    data = json.loads(obj.value)
    return make_object(data)


STDLIB = {
    "$$add": NativeFunction(lambda x: NativeFunction(lambda y: Int(unpack_int(x) + unpack_int(y)))),
    "$$fetch": NativeFunction(fetch),
    "$$jsondecode": NativeFunction(jsondecode),
}


class ScrapRepl(code.InteractiveConsole):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.env: Env = STDLIB

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        try:
            tokens = tokenize(source)
            logger.debug("Tokens: %s", tokens)
            ast = parse(tokens)
            logger.debug("AST: %s", ast)
            result = eval(self.env, ast)
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

    histfile = os.path.expanduser(".scrap-history")
    histfile_size = 1000
    if readline and os.path.exists(histfile):
        readline.read_history_file(histfile)
    repl = ScrapRepl()
    repl.interact(banner="")
    if readline:
        readline.set_history_length(histfile_size)
        readline.write_history_file(histfile)


def test_command(args: argparse.Namespace) -> None:
    if args.debug:
        # pylint: disable=protected-access
        __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main(argv=[__file__])


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    repl = subparsers.add_parser("repl")
    repl.set_defaults(func=repl_command)
    repl.add_argument("--debug", action="store_true")

    test = subparsers.add_parser("test")
    test.set_defaults(func=test_command)
    test.add_argument("--debug", action="store_true")

    eval_ = subparsers.add_parser("eval")
    eval_.set_defaults(func=eval_command)
    eval_.add_argument("program_file", type=argparse.FileType("r"))
    eval_.add_argument("--debug", action="store_true")

    apply = subparsers.add_parser("apply")
    apply.set_defaults(func=apply_command)
    apply.add_argument("program")
    apply.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
