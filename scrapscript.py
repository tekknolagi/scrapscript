#!/usr/bin/env python3.10
import base64
import enum
import logging
import sys
import unittest
from dataclasses import dataclass
from enum import auto
from typing import Callable, Mapping, Optional

import click
from click import File

logger = logging.getLogger(__name__)


FULL_TEST_OUTPUT = False


if FULL_TEST_OUTPUT:
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999


def is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in ("$", "'", "_")


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
        if c == "(":
            if self.has_input() and self.peek_char() == ")":
                self.read_char()
                return "()"
        if c.isdigit():
            return self.read_number(c)
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
        while self.has_input() and is_identifier_char(c := self.peek_char()):
            self.read_char()
            buf += c
        return buf

    def read_bytes(self) -> str:
        buf = "~~"
        while self.has_input():
            if (c := self.read_char()) == "=":
                break
            buf += c
        else:
            raise ParseError("unexpected EOF while reading bytes")
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
    "|": rp(4.5),
    ":": lp(4.5),
    "|>": rp(4.11),
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
    tilde_tilde_prefix = "~~"
    # TODO(max): Tag tokens out of the lexer so we don't have to re-interpret
    # them here.
    if token.isnumeric() or (token[0] == "-" and token[1:].isnumeric()):
        l = Int(int(token))
    elif token.isidentifier():
        l = Var(token)
    elif token.startswith(sha_prefix) and token[len(sha_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith(dollar_dollar_prefix) and token[len(dollar_dollar_prefix) :].isidentifier():
        l = Var(token)
    elif token.startswith(tilde_tilde_prefix):
        l = Bytes(base64.b64decode(token[len(tilde_tilde_prefix) :]))
    elif token.startswith('"') and token.endswith('"'):
        l = String(token[1:-1])
    elif token == "()":
        l = Hole()
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
        raise ParseError(f"unexpected token {token!r}")
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
        elif op == "|>":
            l = Apply(parse(tokens, pr), l)
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
    HASTYPE = auto()
    PIPE = auto()

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
            ":": cls.HASTYPE,
            "|>": cls.PIPE,
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
    if isinstance(exp, (Int, Bool, String, Bytes, Function, Hole)):
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

    def test_tokenize_binop_var(self) -> None:
        ops = ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++"]
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
        with self.assertRaisesRegex(ParseError, "unexpected EOF while reading string"):
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

    def test_tokenize_tilde_tilde_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected EOF while reading bytes"):
            tokenize("~~")

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            tokenize("~=")

    def test_tokenize_tilde_tilde_equals_returns_empty_bytes(self) -> None:
        self.assertEqual(tokenize("~~="), ["~~"])

    def test_tokenize_bytes_returns_bytes(self) -> None:
        self.assertEqual(tokenize("~~QUJD="), ["~~QUJD"])

    def test_tokenize_hole(self) -> None:
        self.assertEqual(tokenize("()"), ["()"])

    def test_tokenize_pipe(self) -> None:
        self.assertEqual(
            tokenize("1 |> f . f = a -> a + 1"),
            ["1", "|>", "f", ".", "f", "=", "a", "->", "a", "+", "1"],
        )


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
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
        self.assertEqual(parse(["~~QUJD"]), Bytes(b"ABC"))

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

    def test_parse_binary_concat_returns_binop(self) -> None:
        self.assertEqual(
            parse(['"abc"', "++", '"def"']),
            Binop(BinopKind.CONCAT, String("abc"), String("def")),
        )

    def test_parse_binary_op_returns_binop(self) -> None:
        ops = ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++"]
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
        self.assertEqual(parse(["()"]), Hole())

    def test_parse_pipe(self) -> None:
        self.assertEqual(
            parse(["x", "|>", "f"]),
            Apply(Var("f"), Var("x")),
        )

    def test_parse_pipe_nested(self) -> None:
        self.assertEqual(
            parse(["x", "|>", "f", "|>", "g"]),
            Apply(Var("g"), Apply(Var("f"), Var("x"))),
        )

    def test_parse_pipe_nested_mixed_with_normal_application(self) -> None:
        self.assertEqual(
            parse(["f", "x", "|>", "g", "y", "|>", "h", "z"]),
            Apply(
                Apply(Var("h"), Var("z")),
                Apply(
                    Apply(Var("g"), Var("y")),
                    Apply(Var("f"), Var("x")),
                ),
            ),
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
        exp = Binop(BinopKind.CONCAT, String("hello"), String(" world"))
        self.assertEqual(eval({}, exp), String("hello world"))

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.CONCAT, Int(123), String(" world"))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.CONCAT, String(" world"), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

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
        self.assertEqual(eval({}, exp), exp)

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
        with self.assertRaisesRegex(AssertionError, r"condition Bool\(value=False\) failed"):
            eval({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), Bool(True)), Bool(True))
        self.assertEqual(eval({}, exp), Int(123))

    def test_eval_hole(self) -> None:
        exp = Hole()
        self.assertEqual(eval({}, exp), Hole())


class EndToEndTests(unittest.TestCase):
    def _run(self, text: str, env: Optional[Env] = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        return eval(env or {}, ast)

    def test_int_returns_int(self) -> None:
        self.assertEqual(self._run("1"), Int(1))

    def test_bytes_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~QUJD="), Bytes(b"ABC"))

    def test_int_add_returns_int(self) -> None:
        self.assertEqual(self._run("1 + 2"), Int(3))

    def test_string_concat_returns_string(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), String("abcdef"))

    def test_empty_list(self) -> None:
        self.assertEqual(self._run("[ ]"), List([]))

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


@click.group()
def main() -> None:
    """Main CLI entrypoint."""


@main.command(name="eval")
@click.argument("program-file", type=File(), default=sys.stdin)
@click.option("--debug", is_flag=True)
def eval_command(program_file: File, debug: bool) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    program = program_file.read()  # type: ignore [attr-defined]
    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval({}, ast)
    print(result)


@main.command(name="apply")
@click.argument("program", type=str, required=True)
@click.option("--debug", is_flag=True)
def apply_command(program: str, debug: bool) -> None:
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval({}, ast)
    print(result)


@main.command(name="test")
def eval_test_command() -> None:
    unittest.main(argv=[__file__])


if __name__ == "__main__":
    main()
