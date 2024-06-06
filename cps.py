import dataclasses
import itertools
import unittest
from collections import Counter
from scrapscript import (
    parse,
    tokenize,
    Assign,
    Int,
    Var as ScrapVar,
    Object,
    Binop,
    BinopKind,
    Where,
    Apply,
    Function,
    List,
    Variant,
)


@dataclasses.dataclass
class CPSExpr:
    pass


@dataclasses.dataclass
class Atom(CPSExpr):
    value: object

    def __repr__(self) -> str:
        return repr(self.value)


@dataclasses.dataclass
class Var(CPSExpr):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclasses.dataclass
class Prim(CPSExpr):
    op: str
    args: list[CPSExpr]

    def __repr__(self) -> str:
        return f"(${self.op} {' '.join(map(repr, self.args))})"


fun_counter = itertools.count()


@dataclasses.dataclass
class Fun(CPSExpr):
    args: list[Var]
    body: CPSExpr
    annotations: dict[str, object] = dataclasses.field(default_factory=dict)
    id: int = dataclasses.field(default_factory=lambda: next(fun_counter), compare=False)

    def name(self) -> str:
        return f"fun{self.id}"

    def freevars(self) -> set[str]:
        result = self.annotations["freevars"]
        assert isinstance(result, set)
        return result

    def kind(self) -> str:
        result = self.annotations["kind"]
        assert isinstance(result, str)
        return result

    def __repr__(self) -> str:
        args = " ".join(map(repr, self.args))
        return f"(fun ({args}) {self.body!r})"


@dataclasses.dataclass
class App(CPSExpr):
    fun: CPSExpr
    args: list[CPSExpr]

    def __repr__(self) -> str:
        return f"({self.fun!r} {' '.join(map(repr, self.args))})"


cps_counter = itertools.count()


def gensym() -> str:
    return f"v{next(cps_counter)}"


def cont(arg: Var, body: CPSExpr) -> CPSExpr:
    return Fun([arg], body)


def cps(exp: Object, k: CPSExpr) -> CPSExpr:
    if isinstance(exp, Int):
        return App(k, [Atom(exp.value)])
    if isinstance(exp, ScrapVar):
        return App(k, [Var(exp.name)])
    if isinstance(exp, Binop):
        left = Var(gensym())
        right = Var(gensym())
        return cps(exp.left, cont(left, cps(exp.right, cont(right, Prim(BinopKind.to_str(exp.op), [left, right, k])))))
    if isinstance(exp, Where):
        assert isinstance(exp.binding, Assign)
        assert isinstance(exp.binding.name, ScrapVar)
        name = exp.binding.name.name
        value = exp.binding.value
        body = exp.body
        return cps(value, cont(Var(name), cps(body, k)))
    if isinstance(exp, Apply):
        fun = Var(gensym())
        arg = Var(gensym())
        return cps(exp.func, cont(fun, cps(exp.arg, cont(arg, App(fun, [arg, k])))))
    if isinstance(exp, Function):
        assert isinstance(exp.arg, ScrapVar)
        arg = Var(exp.arg.name)
        subk = Var(gensym())
        return App(k, [Fun([arg, subk], cps(exp.body, subk))])
    if isinstance(exp, List) and not exp.items:
        return App(k, [Atom([])])
    if isinstance(exp, List):
        items = exp.items
        head = Var(gensym())
        tail = Var(gensym())
        return cps(items[0], cont(head, cps(List(items[1:]), cont(tail, Prim("cons", [head, tail, k])))))
    if isinstance(exp, Variant):
        tag_value = Var(gensym())
        return cps(exp.value, cont(tag_value, Prim("tag", [Atom(exp.tag), tag_value, k])))
    raise NotImplementedError(f"cps: {exp}")


class CPSTests(unittest.TestCase):
    def setUp(self) -> None:
        global cps_counter
        cps_counter = itertools.count()

    def test_atom(self) -> None:
        self.assertEqual(cps(Int(42), Var("k")), App(Var("k"), [Atom(42)]))

    def test_var(self) -> None:
        self.assertEqual(cps(ScrapVar("x"), Var("k")), App(Var("k"), [Var("x")]))

    def test_binop(self) -> None:
        self.assertEqual(
            cps(parse(tokenize("1 + 2")), Var("k")),
            # ((fun (v0) ((fun (v1) (+ v0 v1 k)) 2)) 1)
            App(
                Fun(
                    [Var("v0")],
                    App(
                        Fun(
                            [Var("v1")],
                            Prim("+", [Var("v0"), Var("v1"), Var("k")]),
                        ),
                        [Atom(2)],
                    ),
                ),
                [Atom(1)],
            ),
        )

    def test_where(self) -> None:
        exp = parse(tokenize("a + b . a = 1 . b = 2"))
        self.assertEqual(
            cps(exp, Var("k")),
            # ((fun (b) ((fun (a) ((fun (v0) ((fun (v1) (+ v0 v1 k)) b)) a)) 1)) 2)
            App(
                Fun(
                    [Var("b")],
                    App(
                        Fun(
                            [Var("a")],
                            App(
                                Fun(
                                    [Var("v0")],
                                    App(
                                        Fun(
                                            [Var("v1")],
                                            Prim("+", [Var("v0"), Var("v1"), Var("k")]),
                                        ),
                                        [Var("b")],
                                    ),
                                ),
                                [Var("a")],
                            ),
                        ),
                        [Atom(1)],
                    ),
                ),
                [Atom(2)],
            ),
        )

    def test_empty_list(self) -> None:
        self.assertEqual(cps(List([]), Var("k")), App(Var("k"), [Atom([])]))

    def test_variant(self) -> None:
        self.assertEqual(
            cps(parse(tokenize("# a_tag 123")), Var("k")),
            # ((fun (v0) ($tag 'a_tag' v0 k)) 123)
            App(Fun([Var("v0")], Prim("tag", [Atom("a_tag"), Var("v0"), Var("k")])), [Atom(123)]),
        )


def arg_name(arg: CPSExpr) -> str:
    assert isinstance(arg, Var)
    return arg.name


def alphatise_(exp: CPSExpr, env: dict[str, str]) -> CPSExpr:
    if isinstance(exp, Atom):
        return exp
    if isinstance(exp, Var):
        return Var(env.get(exp.name, exp.name))
    if isinstance(exp, Prim):
        return Prim(exp.op, [alphatise_(arg, env) for arg in exp.args])
    if isinstance(exp, Fun):
        new_env = {arg_name(arg): gensym() for arg in exp.args}
        new_body = alphatise_(exp.body, {**env, **new_env})
        return Fun([Var(new_env[arg_name(arg)]) for arg in exp.args], new_body)
    if isinstance(exp, App):
        return App(alphatise_(exp.fun, env), [alphatise_(arg, env) for arg in exp.args])
    raise NotImplementedError(f"alphatise: {exp}")


def alphatise(exp: CPSExpr) -> CPSExpr:
    return alphatise_(exp, {})


class AlphatiseTests(unittest.TestCase):
    def setUp(self) -> None:
        global cps_counter
        cps_counter = itertools.count()

    def test_atom(self) -> None:
        self.assertEqual(alphatise(Atom(42)), Atom(42))

    def test_var(self) -> None:
        self.assertEqual(alphatise(Var("x")), Var("x"))

    def test_prim(self) -> None:
        exp = Prim("+", [Var("x"), Var("y"), Var("z")])
        self.assertEqual(
            alphatise_(exp, {"x": "v0", "y": "v1"}),
            Prim("+", [Var("v0"), Var("v1"), Var("z")]),
        )

    def test_fun(self) -> None:
        exp = Fun([Var("x"), Var("y")], Prim("+", [Var("x"), Var("y"), Var("z")]))
        self.assertEqual(
            alphatise(exp),
            Fun(
                [Var("v0"), Var("v1")],
                Prim("+", [Var("v0"), Var("v1"), Var("z")]),
            ),
        )

    def test_app(self) -> None:
        exp = App(Var("f"), [Var("x"), Var("y")])
        self.assertEqual(alphatise_(exp, {"x": "v0", "y": "v1"}), App(Var("f"), [Var("v0"), Var("v1")]))


def subst(exp: CPSExpr, env: dict[str, CPSExpr]) -> CPSExpr:
    if isinstance(exp, Atom):
        return exp
    if isinstance(exp, Var):
        return env.get(exp.name, exp)
    if isinstance(exp, Prim):
        return Prim(exp.op, [subst(arg, env) for arg in exp.args])
    if isinstance(exp, Fun):
        new_env = {arg_name(arg): Var(gensym()) for arg in exp.args}
        new_body = subst(exp.body, {**env, **new_env})
        return Fun([Var(new_env[arg_name(arg)].name) for arg in exp.args], new_body)
    if isinstance(exp, App):
        return App(subst(exp.fun, env), [subst(arg, env) for arg in exp.args])
    raise NotImplementedError(f"subst: {exp}")


class SubstTests(unittest.TestCase):
    def setUp(self) -> None:
        global cps_counter
        cps_counter = itertools.count()

    def test_atom(self) -> None:
        self.assertEqual(subst(Atom(42), {}), Atom(42))

    def test_var(self) -> None:
        self.assertEqual(subst(Var("x"), {}), Var("x"))
        self.assertEqual(subst(Var("x"), {"x": Atom(42)}), Atom(42))

    def test_prim(self) -> None:
        exp = Prim("+", [Var("x"), Var("y"), Var("z")])
        self.assertEqual(
            subst(exp, {"x": Atom(1), "z": Atom(3)}),
            Prim("+", [Atom(1), Var("y"), Atom(3)]),
        )

    def test_fun(self) -> None:
        exp = Fun([Var("x"), Var("y")], Prim("+", [Var("x"), Var("y"), Var("z")]))
        self.assertEqual(
            subst(exp, {"z": Atom(3)}),
            Fun(
                [Var("v0"), Var("v1")],
                Prim("+", [Var("v0"), Var("v1"), Atom(3)]),
            ),
        )

    def test_app(self) -> None:
        exp = App(Var("f"), [Var("x"), Var("y")])
        self.assertEqual(subst(exp, {"x": Atom(1), "y": Atom(2)}), App(Var("f"), [Atom(1), Atom(2)]))


def is_simple(exp: CPSExpr) -> bool:
    return isinstance(exp, (Atom, Var, Fun)) or (isinstance(exp, Prim) and exp.op in {"clo", "tag"})


def census(exp: CPSExpr) -> Counter[str]:
    if isinstance(exp, Atom):
        return Counter()
    if isinstance(exp, Var):
        return Counter({exp.name: 1})
    if isinstance(exp, Prim):
        return sum((census(arg) for arg in exp.args), Counter())
    if isinstance(exp, Fun):
        return census(exp.body)
    if isinstance(exp, App):
        return sum((census(arg) for arg in exp.args), census(exp.fun))
    raise NotImplementedError(f"census: {exp}")


class CensusTests(unittest.TestCase):
    def test_atom(self) -> None:
        self.assertEqual(census(Atom(42)), {})

    def test_var(self) -> None:
        self.assertEqual(census(Var("x")), {"x": 1})

    def test_prim(self) -> None:
        exp = Prim("+", [Var("x"), Var("y"), Var("x")])
        self.assertEqual(census(exp), {"x": 2, "y": 1})

    def test_fun(self) -> None:
        exp = Fun([Var("x"), Var("y")], Prim("+", [Var("x"), Var("y"), Var("x")]))
        self.assertEqual(census(exp), {"x": 2, "y": 1})

    def test_app(self) -> None:
        exp = App(Var("f"), [Var("x"), Var("y")])
        self.assertEqual(census(exp), {"f": 1, "x": 1, "y": 1})


def opt(exp: CPSExpr) -> CPSExpr:
    if isinstance(exp, Atom):
        return exp
    if isinstance(exp, Var):
        return exp
    if isinstance(exp, Prim):
        args = [opt(arg) for arg in exp.args[:-1]]
        cont = exp.args[-1]
        if exp.op == "cons":
            assert len(args) == 2
            if all(isinstance(arg, Atom) for arg in args):
                return App(cont, [Atom(args)])
        if exp.op == "+":
            consts = [arg for arg in args if isinstance(arg, Atom)]
            vars = [arg for arg in args if not isinstance(arg, Atom)]
            if consts:
                # TODO(max): Only sum ints
                consts = [Atom(sum(c.value for c in consts))]  # type: ignore
                args = consts + vars
        if len(args) == 1:
            return App(cont, args)
        return Prim(exp.op, args + [cont])
    if isinstance(exp, App) and isinstance(exp.fun, Fun):
        fun = opt(exp.fun)
        assert isinstance(fun, Fun)
        formals = exp.fun.args
        actuals = [opt(arg) for arg in exp.args]
        if len(formals) != len(actuals):
            return App(fun, actuals)
        if all(is_simple(arg) for arg in actuals):
            new_env = {arg_name(formal): actual for formal, actual in zip(formals, actuals)}
            return subst(fun.body, new_env)
        return App(fun, actuals)
    if isinstance(exp, App):
        fun = opt(exp.fun)
        args = [opt(arg) for arg in exp.args]
        return App(fun, args)
    if isinstance(exp, Fun):
        body = opt(exp.body)
        return Fun(exp.args, body)
    return exp


def spin_opt(exp: CPSExpr) -> CPSExpr:
    while True:
        new_exp = opt(exp)
        if new_exp == exp:
            return exp
        exp = new_exp


class OptTests(unittest.TestCase):
    def setUp(self) -> None:
        global cps_counter
        cps_counter = itertools.count()

    def test_prim(self) -> None:
        exp = Prim("+", [Atom(1), Atom(2), Atom(3), Var("k")])
        self.assertEqual(opt(exp), App(Var("k"), [Atom(6)]))

    def test_prim_var(self) -> None:
        exp = Prim("+", [Atom(1), Var("x"), Atom(3), Var("k")])
        self.assertEqual(opt(exp), Prim("+", [Atom(4), Var("x"), Var("k")]))

    def test_subst(self) -> None:
        exp = App(Fun([Var("x")], Prim("+", [Atom(1), Var("x"), Atom(2), Var("k")])), [Atom(3)])
        self.assertEqual(spin_opt(exp), App(Var("k"), [Atom(6)]))

    def test_add(self) -> None:
        exp = parse(tokenize("1 + 2 + c"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            Prim("+", [Atom(2), Var("c"), Fun([Var("v6")], Prim("+", [Atom(1), Var("v6"), Var("k")]))]),
        )

    def test_simple_fun(self) -> None:
        exp = cps(parse(tokenize("_ -> 1")), Var("k"))
        self.assertEqual(
            spin_opt(exp),
            # (k (fun (_ v0) (v0 1)))
            App(
                Var("k"),
                [
                    Fun(
                        [Var("_"), Var("v0")],
                        App(Var("v0"), [Atom(1)]),
                    )
                ],
            ),
        )

    def test_fun(self) -> None:
        exp = cps(parse(tokenize("_ -> 1 + 2 + 3")), Var("k"))
        self.assertEqual(
            spin_opt(exp),
            # (k (fun (_ v0) (v0 6)))
            App(
                Var("k"),
                [
                    Fun(
                        [Var("_"), Var("v0")],
                        App(Var("v0"), [Atom(6)]),
                    )
                ],
            ),
        )

    def test_add_function(self) -> None:
        exp = parse(tokenize("x -> y -> x + y"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            # (k (fun (x v0) (v0 (fun (y v1) ($+ x y v1)))))
            App(
                Var("k"),
                [
                    Fun(
                        [Var("x"), Var("v0")],
                        App(
                            Var("v0"),
                            [
                                Fun(
                                    [Var("y"), Var("v1")],
                                    Prim("+", [Var("x"), Var("y"), Var("v1")]),
                                )
                            ],
                        ),
                    )
                ],
            ),
        )

    def test_fold_add_function_curried(self) -> None:
        exp = parse(tokenize("(x -> y -> x + y) 3"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            # (k (fun (v6 v7) ($+ 3 v6 v7)))
            App(
                Var("k"),
                [
                    Fun(
                        [Var("v6"), Var("v7")],
                        Prim("+", [Atom(3), Var("v6"), Var("v7")]),
                    )
                ],
            ),
        )

    def test_fold_add_function_var(self) -> None:
        exp = parse(tokenize("(x -> y -> x + y) a b"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            # ($+ a b k)
            Prim("+", [Var("a"), Var("b"), Var("k")]),
        )

    def test_fold_add_function_int(self) -> None:
        exp = parse(tokenize("add a b . add = x -> y -> x + y . a = 3 . b = 4"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            App(Var("k"), [Atom(7)]),
        )

    def test_make_empty_list(self) -> None:
        exp = parse(tokenize("[]"))
        self.assertEqual(spin_opt(cps(exp, Var("k"))), App(Var("k"), [Atom([])]))

    def test_make_const_list(self) -> None:
        exp = parse(tokenize("[1+2, 2+3, 3+4]"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            App(Var("k"), [Atom([Atom(3), Atom([Atom(5), Atom([Atom(7), Atom([])])])])]),
        )

    def test_make_list(self) -> None:
        exp = parse(tokenize("[1+2, x, 3+4]"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            # ($cons x [7, []] (fun (v46) ($cons 3 v46 k)))
            Prim(
                "cons",
                [
                    Var("x"),
                    Atom([Atom(7), Atom([])]),
                    Fun([Var("v46")], Prim("cons", [Atom(3), Var("v46"), Var("k")])),
                ],
            ),
        )

    def test_variant(self) -> None:
        exp = parse(tokenize("# a_tag 123"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            # ($tag 'a_tag' 123 k)
            Prim("tag", [Atom("a_tag"), Atom(123), Var("k")]),
        )


def free_in(exp: CPSExpr) -> set[str]:
    match exp:
        case Atom(_):
            return set()
        case Var(name):
            return {name}
        case Prim(_, args):
            return {name for arg in args for name in free_in(arg)}
        case Fun(args, body):
            return free_in(body) - {arg_name(arg) for arg in args}
        case App(fun, args):
            return free_in(fun) | {name for arg in args for name in free_in(arg)}
    raise NotImplementedError(f"free_in: {exp}")


def annotate_free_in(exp: CPSExpr) -> None:
    match exp:
        case Atom(_):
            return
        case Var(_):
            return
        case Prim(_, args):
            for arg in args:
                annotate_free_in(arg)
        case Fun(args, body):
            freevars = free_in(exp)
            exp.annotations["freevars"] = freevars
            for arg in args:
                annotate_free_in(arg)
            annotate_free_in(body)
        case App(fun, args):
            for arg in args:
                annotate_free_in(arg)
            annotate_free_in(fun)


class FreeInTests(unittest.TestCase):
    def test_atom(self) -> None:
        self.assertEqual(free_in(Atom(42)), set())

    def test_var(self) -> None:
        self.assertEqual(free_in(Var("x")), {"x"})

    def test_prim(self) -> None:
        exp = Prim("+", [Var("x"), Var("y"), Var("z")])
        self.assertEqual(free_in(exp), {"x", "y", "z"})

    def test_fun(self) -> None:
        exp = Fun([Var("x"), Var("y")], Prim("+", [Var("x"), Var("y"), Var("z")]))
        self.assertEqual(free_in(exp), {"z"})

    def test_fun_annotate(self) -> None:
        exp = Fun([Var("x"), Var("y")], Prim("+", [Var("x"), Var("y"), Var("z")]))
        annotate_free_in(exp)
        self.assertEqual(exp.freevars(), {"z"})

    def test_app(self) -> None:
        exp = App(Var("f"), [Var("x"), Var("y")])
        self.assertEqual(free_in(exp), {"f", "x", "y"})


def classify_lambdas(exp: CPSExpr) -> None:
    match exp:
        case Atom(_):
            return
        case Var(_):
            return
        case App(Fun(_, body) as lam, args):
            lam.annotations["kind"] = "open"
            classify_lambdas(body)
            for arg in args:
                classify_lambdas(arg)
        case Prim(_, [*args, Fun(_, _) as lam]):
            lam.annotations["kind"] = "open"
            for arg in args:
                classify_lambdas(arg)
        case App(f, args):
            classify_lambdas(f)
            for arg in args:
                classify_lambdas(arg)
        case Fun(_, body) as lam:
            lam.annotations["kind"] = "closed"
            classify_lambdas(body)
        case Prim(_, args):
            for arg in args:
                classify_lambdas(arg)
        case _:
            raise NotImplementedError(f"classify_lambdas: {exp}")


class ClassificationTests(unittest.TestCase):
    def test_open(self) -> None:
        lam = Fun([Var("x")], Var("x"))
        exp = App(lam, [Atom(42)])
        classify_lambdas(exp)
        self.assertEqual(lam.kind(), "open")

    def test_open_prim(self) -> None:
        lam = Fun([Var("x")], Var("x"))
        exp = Prim("+", [Var("x"), Var("y"), lam])
        classify_lambdas(exp)
        self.assertEqual(lam.kind(), "open")

    def test_closed_arg(self) -> None:
        lam = Fun([Var("x")], Var("x"))
        exp = App(Var("f"), [lam])
        classify_lambdas(exp)
        self.assertEqual(lam.kind(), "closed")

    def test_closed(self) -> None:
        exp = Fun([Var("x")], Var("x"))
        classify_lambdas(exp)
        self.assertEqual(exp.kind(), "closed")


#
#
# def make_closures_explicit(exp: CPSExpr, replacements: dict[str, CPSExpr]) -> CPSExpr:
#     def rec(exp: CPSExpr) -> CPSExpr:
#         return make_closures_explicit(exp, replacements)
#
#     match exp:
#         case Atom(_):
#             return exp
#         case Var(name):
#             if name in replacements:
#                 return replacements[name]
#             return exp
#         case Prim(op, args):
#             return Prim(op, [rec(arg) for arg in args])
#         case Fun(args, body):
#             freevars = sorted(free_in(exp))
#             this = Var("this")
#             new_replacements = {fv: Prim("clo", [this, Atom(idx)]) for idx, fv in enumerate(freevars)}
#             body = make_closures_explicit(body, {**replacements, **new_replacements})
#             return Fun([this] + args, body)
#         case App(fun, args):
#             return App(rec(fun), [rec(arg) for arg in args])
#     raise NotImplementedError(f"make_closures_explicit: {exp}")
#
#
# class ClosureTests(unittest.TestCase):
#     def test_no_freevars(self) -> None:
#         exp = Fun([Var("x")], Var("x"))
#         # (fun (this x) x)
#         self.assertEqual(make_closures_explicit(exp, {}), Fun([Var("this"), Var("x")], Var("x")))
#
#     def test_freevars(self) -> None:
#         exp = Fun([Var("k")], Prim("+", [Var("x"), Var("y"), Var("k")]))
#         # (fun (this k) ($+ ($clo this 0) ($clo this 1) k))
#         self.assertEqual(
#             make_closures_explicit(exp, {}),
#             Fun(
#                 [Var("this"), Var("k")],
#                 Prim(
#                     "+",
#                     [
#                         Prim("clo", [Var("this"), Atom(0)]),
#                         Prim("clo", [Var("this"), Atom(1)]),
#                         Var("k"),
#                     ],
#                 ),
#             ),
#         )
#
#     def test_app_fun(self) -> None:
#         exp = App(Fun([Var("x")], Var("x")), [Atom(42)])
#         # ((fun (this x) x) 42)
#         self.assertEqual(
#             make_closures_explicit(exp, {}),
#             App(Fun([Var("this"), Var("x")], Var("x")), [Atom(42)]),
#         )
#
#     def test_app(self) -> None:
#         exp = App(Var("f"), [Atom(42)])
#         # (f 42)
#         self.assertEqual(make_closures_explicit(exp, {}), App(Var("f"), [Atom(42)]))
#
#     def test_add_function(self) -> None:
#         exp = cps(parse(tokenize("x -> y -> x + y")), Var("k"))
#         exp = spin_opt(exp)
#         # (k (fun (this x v2)
#         #      (v2 (fun (this y v3)
#         #            ($+ ($clo this 0) y v3)))))
#         self.assertEqual(
#             make_closures_explicit(exp, {}),
#             App(
#                 Var("k"),
#                 [
#                     Fun(
#                         [Var("this"), Var("x"), Var("v2")],
#                         App(
#                             Var("v2"),
#                             [
#                                 Fun(
#                                     [Var("this"), Var("y"), Var("v3")],
#                                     Prim("+", [Prim("clo", [Var("this"), Atom(0)]), Var("y"), Var("v3")]),
#                                 )
#                             ],
#                         ),
#                     )
#                 ],
#             ),
#         )
#
#
# class C:
#     def __init__(self) -> None:
#         self.funs: list[str] = []
#
#     def G(self, exp: CPSExpr) -> str:
#         match exp:
#             case Atom(int(value)):
#                 return str(value)
#             case Var(name):
#                 return name
#             case App(k, [Fun(_, _)]):
#                 assert isinstance(k, Var)
#                 assert isinstance(exp.args[0], Fun)
#                 fun, name = self.G_proc(exp.args[0])
#                 self.funs.append(fun)
#                 return f"return mkclosure({name});"
#             case App(k, [E]):
#                 assert is_simple(E)
#                 return f"return {E};"
#             case App(E, [*args, k]):
#                 assert isinstance(E, Var)
#                 assert all(is_simple(arg) for arg in args)
#                 return self.G_cont(f"{E.name}({', '.join(str(arg) for arg in args)})", k)
#             case Prim("+", [x, y, k]):
#                 assert is_simple(x)
#                 assert is_simple(y)
#                 return self.G_cont(f"{x} + {y}", k)
#             # TODO(max): j case
#             # TODO(max): Split cont and fun or annotate
#             case Prim("if", [cond, tk, fk]):
#                 return f"if ({cond}) {{ {self.G(tk)} }} else {{ {self.G(fk)} }}"
#             case _:
#                 raise NotImplementedError(f"G: {exp}")
#
#     def G_cont(self, val: str, exp: CPSExpr) -> str:
#         match exp:
#             case Fun([res], M1):
#                 return f"{res} <- {val}; {self.G(M1)}"
#             case Var(_):
#                 return f"return {val};"
#             case _:
#                 raise NotImplementedError(f"G_cont: {exp}")
#
#     def G_proc(self, exp: Fun) -> tuple[str, str]:
#         match exp:
#             case Fun([*args, _], M1):
#                 return f"proc fun{exp.id}({', '.join(arg.name for arg in args)}) {{ {self.G(M1)} " + "}", f"fun{exp.id}"
#             case _:
#                 raise NotImplementedError(f"G_proc: {exp}")
#
#     def code(self) -> str:
#         return "\n\n".join(self.funs)
#
#
# class GTests(unittest.TestCase):
#     def setUp(self) -> None:
#         global cps_counter
#         cps_counter = itertools.count()
#
#         global fun_counter
#         fun_counter = itertools.count()
#
#     def test_app_cont(self) -> None:
#         # (E ... (fun (x) M1))
#         exp = App(Var("f"), [Atom(1), Fun([Var("x")], App(Var("k"), [Var("x")]))])
#         self.assertEqual(C().G(exp), "x <- f(1); return x;")
#
#     def test_tailcall(self) -> None:
#         # (E ... k)
#         exp = App(Var("f"), [Atom(1), Var("k")])
#         self.assertEqual(C().G(exp), "return f(1);")
#
#     def test_return(self) -> None:
#         # (k E)
#         exp = App(Var("k"), [Atom(1)])
#         self.assertEqual(C().G(exp), "return 1;")
#
#     def test_if(self) -> None:
#         # ($if cond t f)
#         exp = Prim(
#             "if",
#             [
#                 Atom(1),
#                 App(Var("k"), [Atom(2)]),
#                 App(Var("k"), [Atom(3)]),
#             ],
#         )
#         self.assertEqual(C().G(exp), "if (1) { return 2; } else { return 3; }")
#
#     def test_add_cont(self) -> None:
#         # ($+ x y (fun (res) M1))
#         exp = Prim("+", [Atom(1), Atom(2), Fun([Var("res")], App(Var("k"), [Var("res")]))])
#         self.assertEqual(C().G(exp), "res <- 1 + 2; return res;")
#
#     def test_add_cont_var(self) -> None:
#         # ($+ x y k)
#         exp = Prim("+", [Atom(1), Atom(2), Var("k")])
#         self.assertEqual(C().G(exp), "return 1 + 2;")
#
#     def test_proc(self) -> None:
#         exp = App(Var("k"), [Fun([Var("x"), Var("j")], Prim("+", [Var("x"), Atom(1), Var("j")]))])
#         c = C()
#         code = c.G(exp)
#         self.assertEqual(c.code(), "proc fun0(x) { return x + 1; }")
#         self.assertEqual(code, "return mkclosure(fun0);")


if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
