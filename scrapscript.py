import unittest
from dataclasses import dataclass


obj = dataclass(eq=True, frozen=True, unsafe_hash=True, repr=False)


@obj
class Object:
    pass


@obj
class Int(Object):
    value: int


def eval(env: dict[str, Object], exp: Object) -> Object:
    if isinstance(exp, Int):
        return exp
    raise NotImplementedError(f"eval not implemented for {exp}")


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval({}, exp), Int(5))


if __name__ == "__main__":
    unittest.main()
