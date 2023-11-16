import enum
import unittest
from dataclasses import dataclass
from enum import auto


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Object:
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Int(Object):
    value: int


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Var(Object):
    name: str


class BinopKind(enum.Enum):
    ADD = auto()
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
        raise TypeError(f"expected Int, got {result}")
    return result.value


BINOP_HANDLERS = {
    BinopKind.ADD: lambda env, x, y: Int(eval_int(env, x) + eval_int(env, y)),
    BinopKind.SUB: lambda env, x, y: Int(eval_int(env, x) - eval_int(env, y)),
    BinopKind.MUL: lambda env, x, y: Int(eval_int(env, x) * eval_int(env, y)),
    BinopKind.DIV: lambda env, x, y: Int(eval_int(env, x) // eval_int(env, y)),
}


def eval(env: dict[str, Object], exp: Object) -> Object:
    if isinstance(exp, Int):
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


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval({}, exp), Int(5))

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

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        self.assertEqual(eval({}, exp), Int(-1))

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        self.assertEqual(eval({}, exp), Int(6))

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(2), Int(3))
        self.assertEqual(eval({}, exp), Int(0))


if __name__ == "__main__":
    unittest.main()
