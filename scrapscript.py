import click
import enum
import re
import sys
import unittest
from dataclasses import dataclass
from enum import auto
from typing import Mapping


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


def eval_int(env: dict[str, Object], exp: Object) -> int:
    result = eval(env, exp)
    if not isinstance(result, Int):
        raise TypeError(f"expected Int, got {type(result).__name__}")
    return result.value


def eval_str(env: dict[str, Object], exp: Object) -> str:
    result = eval(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


BINOP_HANDLERS = {
    BinopKind.ADD: lambda env, x, y: Int(eval_int(env, x) + eval_int(env, y)),
    BinopKind.CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    BinopKind.SUB: lambda env, x, y: Int(eval_int(env, x) - eval_int(env, y)),
    BinopKind.MUL: lambda env, x, y: Int(eval_int(env, x) * eval_int(env, y)),
    BinopKind.DIV: lambda env, x, y: Int(eval_int(env, x) // eval_int(env, y)),
}


def eval(env: Mapping[str, Object], exp: Object) -> Object:
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

    def test_ignore_whitespace(self) -> None:
        self.assertEqual(tokenize("1\n+\t2"), ["1", "+", "2"])

    def test_ignore_line_comment(self) -> None:
        self.assertEqual(tokenize("-- 1\n2"), ["2"])

    def test_tokenize_string(self) -> None:
        self.assertEqual(tokenize('"hello"'), ['"hello"'])

    @unittest.skip("TODO(max): Support spaces in strings")
    def test_tokenize_string_with_spaces(self) -> None:
        self.assertEqual(tokenize('"hello world"'), ['"hello world"'])

    def test_tokenize_empty_list(self) -> None:
        self.assertEqual(tokenize("[ ] "), ["[", "]"])

    def test_tokenize_list_with_items(self) -> None:
        self.assertEqual(tokenize("[ 1 , 2 ] "), ["[", "1", ",", "2", "]"])


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse([])
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_digit_returns_int(self) -> None:
        self.assertEqual(parse(["1"]), Int(1))

    def test_parse_digits_returns_int(self) -> None:
        self.assertEqual(parse(["123"]), Int(123))

    @unittest.skip("TODO(max): negatives")
    def test_parse_negative_int_returns_int(self) -> None:
        self.assertEqual(parse(["-123"]), Int(123))

    def test_parse_var_returns_var(self) -> None:
        self.assertEqual(parse(["abc_123"]), Var("abc_123"))

    def test_parse_sha_var_returns_var(self) -> None:
        self.assertEqual(parse(["$sha1'abc"]), Var("$sha1'abc"))

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


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval({}, exp), Int(5))

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        self.assertEqual(eval({}, exp), String("xyz"))

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


class EndToEndTests(unittest.TestCase):
    def _run(self, text: str, env: dict[str, Object] = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        return eval(env or {}, ast)

    def test_int_returns_int(self) -> None:
        self.assertEqual(self._run("1"), Int(1))

    def test_int_add_returns_int(self) -> None:
        self.assertEqual(self._run("1 + 2"), Int(3))

    def test_string_concat_returns_string(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), String("abcdef"))


@click.group()
def main() -> None:
    """Main CLI entrypoint."""


@main.command(name="eval")
@click.argument("program-file", type=click.File(), default=sys.stdin)
def eval_command(program_file: click.File) -> None:
    program = program_file.read()  # type: ignore [attr-defined]
    tokens = tokenize(program)
    ast = parse(tokens)
    print(ast)


@main.command(name="apply")
@click.argument("program", type=str, required=True)
def eval_apply_command(program: str) -> None:
    tokens = tokenize(program)
    ast = parse(tokens)
    print(ast)


@main.command(name="test")
def eval_test_command() -> None:
    unittest.main(argv=[__file__])


if __name__ == "__main__":
    main()
