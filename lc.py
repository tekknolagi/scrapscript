from __future__ import annotations
import dataclasses
import unittest
import typing
from scrapscript import (
    Object,
    Int,
    Var,
    Function,
    Apply,
    Where,
    Assign,
)


@dataclasses.dataclass
class Ty:
    pass


@dataclasses.dataclass
class MonoType(Ty):
    pass


@dataclasses.dataclass
class TyVar(MonoType):
    name: str

    def __repr__(self) -> str:
        return f"'{self.name}"


@dataclasses.dataclass
class TyCon(MonoType):
    name: str
    args: list[MonoType]

    def __repr__(self) -> str:
        if not self.args:
            return self.name
        return f"({self.name.join(map(repr, self.args))})"


@dataclasses.dataclass
class Forall(Ty):
    tyvars: list[TyVar]
    ty: MonoType

    def __repr__(self) -> str:
        return f"(forall {', '.join(map(repr, self.tyvars))}. {self.ty})"


UnitType = TyCon("()", [])
IntType = TyCon("int", [])
BoolType = TyCon("bool", [])
IdFunc = Forall([TyVar("a")], TyCon("->", [TyVar("a"), TyVar("a")]))
NotFunc = TyCon("->", [BoolType, BoolType])


class ReprTest(unittest.TestCase):
    def test_tyvar(self) -> None:
        self.assertEqual(repr(TyVar("a")), "'a")

    def test_tycon(self) -> None:
        self.assertEqual(repr(TyCon("int", [])), "int")

    def test_tycon_args(self) -> None:
        self.assertEqual(repr(TyCon("->", [IntType, IntType])), "(int->int)")

    def test_forall(self) -> None:
        self.assertEqual(repr(Forall([TyVar("a"), TyVar("b")], TyVar("a"))), "(forall 'a, 'b. 'a)")


def func_type(*args: MonoType) -> TyCon:
    return TyCon("->", list(args))


def tuple_type(*args: MonoType) -> TyCon:
    return TyCon("*", list(args))


def ftv_ty(ty: Ty) -> set[str]:
    if isinstance(ty, TyVar):
        return {ty.name}
    if isinstance(ty, TyCon):
        return set().union(*map(ftv_ty, ty.args))
    if isinstance(ty, Forall):
        return ftv_ty(ty.ty) - set(tyvar.name for tyvar in ty.tyvars)
    raise TypeError(f"Unknown type: {ty}")


Context = typing.Mapping[str, Forall]


def ftv_ctx(ctx: Context) -> set[str]:
    return set().union(*(ftv_ty(scheme) for scheme in ctx.values()))


class FtvTest(unittest.TestCase):
    def test_tyvar(self) -> None:
        self.assertEqual(ftv_ty(TyVar("a")), {"a"})

    def test_tycon(self) -> None:
        self.assertEqual(ftv_ty(TyCon("int", [])), set())

    def test_tycon_args(self) -> None:
        self.assertEqual(ftv_ty(TyCon("->", [TyVar("a"), TyVar("b")])), {"a", "b"})

    def test_forall(self) -> None:
        self.assertEqual(ftv_ty(Forall([TyVar("a"), TyVar("b")], TyVar("a"))), set())
        self.assertEqual(ftv_ty(Forall([TyVar("a"), TyVar("b")], TyVar("c"))), {"c"})

    def test_context(self) -> None:
        self.assertEqual(ftv_ctx({"id": IdFunc}), set())
        self.assertEqual(ftv_ctx({"f": Forall([TyVar("a")], TyVar("b"))}), {"b"})


Subst = typing.Mapping[str, MonoType]


def apply_ty(ty: MonoType, subst: Subst) -> MonoType:
    if isinstance(ty, TyVar):
        return subst.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [apply_ty(arg, subst) for arg in ty.args])
    raise TypeError(f"Unknown type: {ty}")


def apply_scheme(ty: Forall, subst: Subst) -> Forall:
    ty_args = {arg.name for arg in ty.tyvars}
    new_subst = {name: ty for name, ty in subst.items() if name not in ty_args}
    return Forall(ty.tyvars, apply_ty(ty.ty, new_subst))


class ApplyTest(unittest.TestCase):
    def test_tyvar(self) -> None:
        self.assertEqual(apply_ty(TyVar("a"), {"a": TyVar("b")}), TyVar("b"))
        self.assertEqual(apply_ty(TyVar("a"), {}), TyVar("a"))

    def test_tycon(self) -> None:
        self.assertEqual(apply_ty(TyCon("int", []), {}), TyCon("int", []))

    def test_tycon_args(self) -> None:
        self.assertEqual(
            apply_ty(TyCon("->", [TyVar("a"), TyVar("b")]), {"a": TyVar("c")}),
            TyCon("->", [TyVar("c"), TyVar("b")]),
        )

    def test_forall(self) -> None:
        ty = Forall([TyVar("a")], func_type(TyVar("a"), TyVar("b")))
        self.assertEqual(apply_scheme(ty, {"a": TyVar("c")}), ty)
        self.assertEqual(apply_scheme(ty, {"b": TyVar("c")}), Forall([TyVar("a")], func_type(TyVar("a"), TyVar("c"))))


def compose(s1: Subst, s2: Subst) -> Subst:
    result = {tyvar: apply_ty(ty, s1) for tyvar, ty in s2.items()}
    result.update(s1)
    return result


class ComposeTest(unittest.TestCase):
    def test_s1_applied_inside_s2(self) -> None:
        s1 = {"a": TyVar("b")}
        s2 = {"c": TyVar("a")}
        self.assertEqual(compose(s1, s2), {"a": TyVar("b"), "c": TyVar("b")})

    def test_left_takes_precedence(self) -> None:
        s1 = {"a": TyVar("b")}
        s2 = {"a": TyVar("c")}
        self.assertEqual(compose(s1, s2), {"a": TyVar("b")})

    def test_union(self) -> None:
        s1 = {"a": TyVar("b")}
        s2 = {"c": TyVar("d")}
        self.assertEqual(compose(s1, s2), {"a": TyVar("b"), "c": TyVar("d")})


fresh_var_counter = 0


def fresh_tyvar(prefix: str = "t") -> TyVar:
    global fresh_var_counter
    result = f"{prefix}{fresh_var_counter}"
    fresh_var_counter += 1
    return TyVar(result)


class FreshTests(unittest.TestCase):
    def setUp(self) -> None:
        global fresh_var_counter
        fresh_var_counter = 0

    def test_fresh(self) -> None:
        self.assertEqual(fresh_tyvar(), TyVar("t0"))
        self.assertEqual(fresh_tyvar("x"), TyVar("x1"))


def bind_var(ty: MonoType, name: str) -> Subst:
    if isinstance(ty, TyVar) and ty.name == name:
        return {}
    if name in ftv_ty(ty):
        raise TypeError(f"Occurs check failed for {name} in {ty}")
    return {name: ty}


class BindTests(unittest.TestCase):
    def test_tyvar_with_matching_name_returns_empty_subst(self) -> None:
        self.assertEqual(bind_var(TyVar("a"), "a"), {})

    def test_tyvar_with_non_matching_name_returns_singleton(self) -> None:
        self.assertEqual(bind_var(TyVar("a"), "b"), {"b": TyVar("a")})

    def test_tycon_returns_singleton(self) -> None:
        self.assertEqual(
            bind_var(func_type(TyVar("a"), TyVar("b")), "c"),
            {"c": func_type(TyVar("a"), TyVar("b"))},
        )

    def test_name_in_freevars_raises_type_error(self) -> None:
        ty = func_type(TyVar("a"), TyVar("b"))
        with self.assertRaisesRegex(TypeError, "Occurs check"):
            bind_var(ty, "a")


def generalize(ty: MonoType, ctx: Context) -> Forall:
    tyvars = ftv_ty(ty) - ftv_ctx(ctx)
    return Forall([TyVar(name) for name in sorted(tyvars)], ty)


class GeneralizeTests(FreshTests):
    def test_tyvar(self) -> None:
        self.assertEqual(generalize(TyVar("a"), {}), Forall([TyVar("a")], TyVar("a")))

    def test_tyvar_bound_in_ctx(self) -> None:
        self.assertEqual(
            generalize(TyVar("a"), {"f": Forall([TyVar("a")], TyVar("a"))}),
            Forall([TyVar("a")], TyVar("a")),
        )

    def test_tyvar_free_in_ctx(self) -> None:
        self.assertEqual(
            generalize(TyVar("a"), {"f": Forall([TyVar("b")], TyVar("a"))}),
            Forall([], TyVar("a")),
        )


def instantiate(scheme: Forall) -> MonoType:
    fresh = {tyvar.name: fresh_tyvar() for tyvar in scheme.tyvars}
    return apply_ty(scheme.ty, fresh)


class InstantiateTests(FreshTests):
    def test_freshen_type_variables(self) -> None:
        scheme = Forall([TyVar("a")], func_type(TyVar("a"), TyVar("b")))
        self.assertEqual(instantiate(scheme), func_type(TyVar("t0"), TyVar("b")))


def unify_fail(ty1: MonoType, ty2: MonoType) -> None:
    raise TypeError(f"Unification failed for {ty1} and {ty2}")


def unify(ty1: MonoType, ty2: MonoType) -> Subst:
    if isinstance(ty1, TyVar):
        return bind_var(ty2, ty1.name)
    if isinstance(ty2, TyVar):  # Mirror
        return unify(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name:
            unify_fail(ty1, ty2)
        if len(ty1.args) != len(ty2.args):
            unify_fail(ty1, ty2)
        result: Subst = {}
        for l, r in zip(ty1.args, ty2.args):
            result = compose(
                unify(apply_ty(l, result), apply_ty(r, result)),
                result,
            )
        return result
    raise TypeError(f"ICE: Unexpected type {type(ty1)}")


class UnifyTests(FreshTests):
    def test_tyvar_tyvar(self) -> None:
        self.assertEqual(unify(TyVar("a"), TyVar("b")), {"a": TyVar("b")})
        self.assertEqual(unify(TyVar("b"), TyVar("a")), {"b": TyVar("a")})

    def test_tyvar_tycon(self) -> None:
        self.assertEqual(unify(TyVar("a"), IntType), {"a": IntType})
        self.assertEqual(unify(IntType, TyVar("a")), {"a": IntType})

    def test_tycon_tycon_name_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify(IntType, BoolType)

    def test_tycon_tycon_arity_mismatch(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [])
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify(l, r)

    def test_tycon_tycon_unifies_arg(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [TyVar("b")])
        self.assertEqual(unify(l, r), {"a": TyVar("b")})

    def test_tycon_tycon_unifies_args(self) -> None:
        l = func_type(TyVar("a"), TyVar("b"))
        r = func_type(TyVar("c"), TyVar("d"))
        self.assertEqual(unify(l, r), {"a": TyVar("c"), "b": TyVar("d")})


def infer_w(expr: Object, ctx: Context) -> tuple[Subst, MonoType]:
    if isinstance(expr, Var):
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise TypeError(f"Unbound variable {expr.name}")
        return {}, instantiate(scheme)
    if isinstance(expr, Function):
        arg_tyvar = fresh_tyvar("a")
        assert isinstance(expr.arg, Var)
        body_ctx = {**ctx, expr.arg.name: Forall([], arg_tyvar)}
        body_subst, body_ty = infer_w(expr.body, body_ctx)
        return body_subst, func_type(apply_ty(arg_tyvar, body_subst), body_ty)
    raise TypeError(f"Unexpected type {type(expr)}")


class InferTests(FreshTests):
    def test_unbound_var(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unbound variable"):
            infer_w(Var("a"), {})

    def test_var_instantiates_scheme(self) -> None:
        subst, ty = infer_w(Var("a"), {"a": Forall([TyVar("b")], TyVar("b"))})
        self.assertEqual(subst, {})
        self.assertEqual(ty, TyVar("t0"))

    def test_function_returns_arg(self) -> None:
        subst, ty = infer_w(Function(Var("x"), Var("x")), {})
        self.assertEqual(subst, {})
        self.assertEqual(ty, func_type(TyVar("a0"), TyVar("a0")))

    def test_nested_function_outer(self) -> None:
        subst, ty = infer_w(Function(Var("x"), Function(Var("y"), Var("x"))), {})
        self.assertEqual(subst, {})
        self.assertEqual(ty, func_type(TyVar("a0"), func_type(TyVar("a1"), TyVar("a0"))))

    def test_nested_function_inner(self) -> None:
        subst, ty = infer_w(Function(Var("x"), Function(Var("y"), Var("y"))), {})
        self.assertEqual(subst, {})
        self.assertEqual(ty, func_type(TyVar("a0"), func_type(TyVar("a1"), TyVar("a1"))))


if __name__ == "__main__":
    unittest.main()
