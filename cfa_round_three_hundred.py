from __future__ import annotations
import dataclasses
from scrapscript import (
    Access,
    Apply,
    Assign,
    Binop,
    BinopKind,
    Function,
    Hole,
    Int,
    List,
    MatchFunction,
    Object,
    Record,
    Spread,
    String,
    Var,
    Variant,
    Where,
)
import unittest
from typing import Mapping

next_label = 0

def new_label():
    global next_label
    result = next_label
    next_label += 1
    return result

def label_of(expr):
    if not hasattr(expr, "label"):
        object.__setattr__(expr, "label", new_label())
    return expr.label

@dataclasses.dataclass
class AbstractValue:
    def meet(self, other: AbstractValue) -> AbstractValue:
        raise NotImplementedError

    @staticmethod
    def bottom() -> AbstractValue:
        raise NotImplementedError

AnyFunction = Function|MatchFunction

@dataclasses.dataclass
class FunctionSet(AbstractValue):
    functions: frozenset[AnyFunction] = dataclasses.field(default_factory=frozenset)

    def meet(self, other: FunctionSet) -> FunctionSet:
        return FunctionSet(self.functions | other.functions)

    @staticmethod
    def bottom() -> FunctionSet:
        return FunctionSet()

AbstractEnv = Mapping[str, int]

# (lambda (x[0]) x[1])[2]

@dataclasses.dataclass
class Cache:
    cache: Mapping[int, AbstractValue] = dataclasses.field(default_factory=dict)

    def set(self, key: int, value: AbstractValue) -> AbstractValue:
        if key not in self.cache:
            self.cache[key] = value
        else:
            self.cache[key] = self.cache[key].meet(value)
        return self.cache[key]

    def at(self, key: int) -> AbstractValue:
        return self.cache[key]

def eval_exp(env: AbstractEnv, cache: Cache, exp: Object) -> AbstractValue:
    if isinstance(exp, (Int, String)):
        return cache.set(label_of(exp), FunctionSet.bottom())
    if isinstance(exp, Var):
        return cache.set(label_of(exp), cache.at(env[exp.name]))
    if isinstance(exp, Function):
        cache.set(label_of(exp.arg), FunctionSet.bottom())
        eval_exp({**env, exp.arg.name: label_of(exp.arg)}, cache, exp.body)
        return cache.set(label_of(exp), FunctionSet({exp}))
    if isinstance(exp, Apply):
        fun = eval_exp(env, cache, exp.func)
        arg = eval_exp(env, cache, exp.arg)
        for f in fun.functions:  # TODO(max): Figure out how to make this generic over AbstractValue
            cache.set(label_of(f.arg), arg)
            f_result = eval_exp({**env, f.arg.name: label_of(f.arg)}, cache, f.body)
            cache.set(label_of(exp), f_result)
        return cache.at(label_of(exp))
    raise NotImplementedError


class CFATests(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(
            eval_exp({}, Cache(), Int(1)),
            FunctionSet.bottom(),
        )

    def test_function_flows_to_itself(self) -> None:
        f = Function(Var("x"), Int(1))
        cache = Cache()
        self.assertEqual(
            eval_exp({}, cache, f),
            FunctionSet({f}),
        )
        self.assertEqual(cache.at(label_of(f.body)), FunctionSet.bottom())

    def test_function_arg_flows_to_var(self) -> None:
        f = Function(Var("arg"), Var("arg"))
        g = Function(Var("x"), Var("x"))
        cache = Cache({label_of(f.arg): FunctionSet({g})})
        eval_exp({}, cache, f)
        self.assertEqual(cache.at(label_of(f.body)), FunctionSet({g}))

    def test_apply_id_to_itself(self) -> None:
        f = Function(Var("arg"), Var("arg"))
        exp = Apply(f, f)
        cache = Cache()
        eval_exp({}, cache, exp)
        self.assertEqual(cache.at(label_of(exp)), FunctionSet({f}))

    def test_apply_id_chained(self) -> None:
        f = Function(Var("arg"), Var("arg"))
        exp = Apply(Apply(f, f), f)
        cache = Cache()
        eval_exp({}, cache, exp)
        self.assertEqual(cache.at(label_of(exp)), FunctionSet({f}))

    def test_apply_id_crossed(self) -> None:
        f = Function(Var("x"), Var("x"))
        g = Function(Var("y"), Var("y"))
        exp = Apply(Apply(f, g), f)
        cache = Cache()
        eval_exp({}, cache, exp)
        self.assertEqual(cache.at(label_of(f.arg)), FunctionSet({g}))
        self.assertEqual(cache.at(label_of(g.arg)), FunctionSet({f}))

    def test_apply_function_meet(self) -> None:
        f = Function(Var("x"), Var("x"))
        g = Function(Var("y"), Var("y"))
        exp = Apply(Apply(Apply(f, g), Apply(f, g)), f)
        cache = Cache()
        eval_exp({}, cache, exp)
        self.assertEqual(cache.at(label_of(f.arg)), FunctionSet({g}))
        self.assertEqual(cache.at(label_of(g.arg)), FunctionSet({f, g}))


if __name__ == "__main__":
    unittest.main()
