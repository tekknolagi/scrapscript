import dataclasses
import itertools
import unittest
from scrapscript import parse, tokenize, Assign, Int, Var as ScrapVar, Object, Binop, BinopKind, Where, Apply, Function


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
        return f"({self.op} {' '.join(map(repr, self.args))})"


@dataclasses.dataclass
class Fun(CPSExpr):
    args: list[CPSExpr]
    body: CPSExpr

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
    return isinstance(exp, (Atom, Var, Fun))


def opt(exp: CPSExpr) -> CPSExpr:
    if isinstance(exp, Atom):
        return exp
    if isinstance(exp, Var):
        return exp
    if isinstance(exp, Prim):
        args = [opt(arg) for arg in exp.args]
        consts = [arg for arg in args if isinstance(arg, Atom)]
        vars = [arg for arg in args if not isinstance(arg, Atom)]
        if exp.op == "+":
            consts = [Atom(sum(c.value for c in consts))]  # type: ignore
            args = consts + vars
        if len(args) == 2:
            # Last argument is a cont
            return App(args[1], [args[0]])
        return Prim(exp.op, args)
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
        exp = parse(tokenize("add a b . add = x -> y -> x + y . a = 3 . b = 4"))
        self.assertEqual(
            spin_opt(cps(exp, Var("k"))),
            App(Var("k"), [Atom(7)]),
        )


if __name__ == "__main__":
    unittest.main()
