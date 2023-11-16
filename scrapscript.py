import enum
import unittest
import re
from dataclasses import dataclass
from enum import auto
from typing import Mapping


def tokenize(x):
    # TODO: Make this a proper tokenizer that handles strings with blankspace.
    stripped = re.sub(r" *--[^\n]*", "", x).strip()
    return re.split(r"[\s\n]+", stripped)


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Object:
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Int(Object):
    value: int


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class String(Object):
    value: str


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Var(Object):
    name: str


class BinopKind(enum.Enum):
    ADD = auto()
    CONCAT = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Binop(Object):
    op: BinopKind
    left: Object
    right: Object


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

    def test_tokenize_string(self):
        self.assertEqual(tokenize('"hello"'), ['"hello"'])


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


if __name__ == "__main__":
    unittest.main()
