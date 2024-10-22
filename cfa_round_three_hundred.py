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
    parse,
    tokenize,
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
    changed: bool = False

    def set(self, key: int, value: AbstractValue) -> AbstractValue:
        if key not in self.cache:
            self.changed = True
            self.cache[key] = value
        else:
            old = self.cache[key]
            self.cache[key] = self.cache[key].meet(value)
            if old != self.cache[key]:
                self.changed = True
        return self.cache[key]

    def at(self, key: int) -> AbstractValue:
        if key not in self.cache:
            self.cache[key] = FunctionSet.bottom()
        return self.cache[key]

# def eval_exp(env: AbstractEnv, cache: Cache, exp: Object) -> AbstractValue:
#     if isinstance(exp, (Int, String)):
#         return cache.set(label_of(exp), FunctionSet.bottom())
#     if isinstance(exp, Var):
#         return cache.set(label_of(exp), cache.at(env[exp.name]))
#     if isinstance(exp, Function):
#         cache.set(label_of(exp.arg), FunctionSet.bottom())
#         eval_exp({**env, exp.arg.name: label_of(exp.arg)}, cache, exp.body)
#         return cache.set(label_of(exp), FunctionSet({exp}))
#     if isinstance(exp, Apply):
#         cache.set(label_of(exp), FunctionSet.bottom())
#         fun = eval_exp(env, cache, exp.func)
#         arg = eval_exp(env, cache, exp.arg)
#         for f in fun.functions:  # TODO(max): Figure out how to make this generic over AbstractValue
#             cache.set(label_of(f.arg), arg)
#             f_result = eval_exp({**env, f.arg.name: label_of(f.arg)}, cache, f.body)
#             cache.set(label_of(exp), f_result)
#         return cache.at(label_of(exp))
#     raise NotImplementedError

# flow is [to, from]
def emit_flow(env: AbstractEnv, flow: set[tuple[int, int]], exp: Object):
    if isinstance(exp, (Int, String)):
        return
    print("label", label_of(exp), exp)
    if isinstance(exp, Var):
        flow.add(("var", label_of(exp), env[exp.name]))
        return
    if isinstance(exp, Function):
        print("label", label_of(exp.arg), exp.arg)
        emit_flow({**env, exp.arg.name: label_of(exp.arg)}, flow, exp.body)
        flow.add(("singleton", label_of(exp), exp))
        return
    if isinstance(exp, Apply):
        # f x
        # f == exp.func
        # x == exp.arg
        # f = y -> y
        emit_flow(env, flow, exp.func)
        emit_flow(env, flow, exp.arg)
        # TODO(max): Make constraints lambdas that get real values or something
        flow.add(("rhs_flows_to_lhs_arg", label_of(exp.func), label_of(exp.arg)))
        flow.add(("rhs_body_flows_to_lhs", label_of(exp), label_of(exp.func)))
        return


        # for f in fun.functions:  # TODO(max): Figure out how to make this generic over AbstractValue
        #     cache.set(label_of(f.arg), arg)
        #     f_result = eval_exp({**env, f.arg.name: label_of(f.arg)}, cache, f.body)
        #     cache.set(label_of(exp), f_result)
        # return cache.at(label_of(exp))
    raise NotImplementedError


def eval_exp(env: AbstractEnv, cache: Cache, exp: Object) -> AbstractValue:
    flow = set()
    emit_flow(env, flow, exp)
    for eqn in flow:
        print(eqn)
    cache.changed = True
    while cache.changed:
        cache.changed = False
        for (eq, to_label, from_label) in flow:
            print("eqn", (eq, to_label, from_label))
            if eq == "var":
                from_value = cache.at(from_label)
                cache.set(to_label, from_value)
            elif eq == "singleton":
                from_value = from_label  # lies
                cache.set(to_label, FunctionSet({from_value}))
            elif eq == "rhs_body_flows_to_lhs":
                from_value = cache.at(from_label)
                for func in from_value.functions:
                    cache.set(to_label, cache.at(label_of(func.body)))
            else:
                assert eq == "rhs_flows_to_lhs_arg"
                from_value = cache.at(from_label)
                to_value = cache.at(to_label)
                for func in to_value.functions:
                    if label_of(func.arg) == 0:
                        print("arg", label_of(func.arg), "<-", from_value)
                    cache.set(label_of(func.arg), from_value)
    return cache.at(label_of(exp))


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
        x = Var("x")
        print("x is labeled", label_of(x))
        f = Function(x, Var("x"))
        print("f is labeled", label_of(f))
        g = Function(Var("y"), Var("y"))
        print("g is labeled", label_of(g))
        # (((f g) (f g)) f)
        # ((g g) f)
        # (g f)
        exp = Apply(Apply(Apply(f, g), Apply(f, g)), f)
        cache = Cache()
        eval_exp({}, cache, exp)
        self.assertEqual(cache.at(label_of(f.arg)), FunctionSet({g}))
        self.assertEqual(cache.at(label_of(g.arg)), FunctionSet({f, g}))

    def test_apply_fixpoint(self) -> None:
        exp = parse(tokenize("(x -> x x) (y -> y y)"))
        # exp = parse(tokenize("f f . f = x -> x x"))
        cache = Cache()
        eval_exp({}, cache, exp)


if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
