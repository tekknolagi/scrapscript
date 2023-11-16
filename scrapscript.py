import unittest
from dataclasses import dataclass


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Object:
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Int(Object):
    value: int


@dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)
class Var(Object):
    name: str


def eval(env: dict[str, Object], exp: Object) -> Object:
    if isinstance(exp, Int):
        return exp
    if isinstance(exp, Var):
        value = env.get(exp.name)
        if value is None:
            raise NameError(f"name '{exp.name}' is not defined")
        return value
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


if __name__ == "__main__":
    unittest.main()
