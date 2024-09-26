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
    Binop,
    BinopKind,
)


@dataclasses.dataclass
class Ty:
    forwarded: Ty | None = dataclasses.field(init=False, default=None)

    def find(self) -> Ty:
        result: Ty = self
        while isinstance(result, TyVar):
            it = result.forwarded
            if it is None:
                return result
            result = it
        return result

    def _set_forwarded(self, other: Ty) -> None:
        assert type(self) == type(other)
        assert self == other, f"{self} != {other}"


@dataclasses.dataclass
class MonoType(Ty):
    pass


@dataclasses.dataclass
class TyVar(MonoType):
    name: str

    def __repr__(self) -> str:
        return f"'{self.name}"

    def make_equal_to(self, other: Ty) -> None:
        self.find()._set_forwarded(other)

    def _set_forwarded(self, other: Ty) -> None:
        self.forwarded = other


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
    assert len(args) >= 2
    if len(args) == 2:
        return TyCon("->", list(args))
    return TyCon("->", [args[0], func_type(*args[1:])])


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


def apply_ctx(ctx: Context, subst: Subst) -> Context:
    return {name: apply_scheme(scheme, subst) for name, scheme in ctx.items()}


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

    def assertTyEqual(self, l: Ty, r: Ty) -> bool:
        l = l.find()
        r = r.find()
        if isinstance(l, TyVar) and isinstance(r, TyVar):
            if l != r:
                self.fail(f"Type mismatch: {l} != {r}")
            return True
        if isinstance(l, TyCon) and isinstance(r, TyCon):
            if l.name != r.name:
                self.fail(f"Type mismatch: {l} != {r}")
            if len(l.args) != len(r.args):
                self.fail(f"Type mismatch: {l} != {r}")
            for l_arg, r_arg in zip(l.args, r.args):
                self.assertTyEqual(l_arg, r_arg)
            return True
        if isinstance(l, Forall) or isinstance(r, Forall):
            raise NotImplementedError
        self.fail(f"Type mismatch: {l} != {r}")


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
    # TODO(max): Freshen?
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


def unify_w(ty1: MonoType, ty2: MonoType) -> Subst:
    if isinstance(ty1, TyVar):
        return bind_var(ty2, ty1.name)
    if isinstance(ty2, TyVar):  # Mirror
        return unify_w(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name:
            unify_fail(ty1, ty2)
        if len(ty1.args) != len(ty2.args):
            unify_fail(ty1, ty2)
        result: Subst = {}
        for l, r in zip(ty1.args, ty2.args):
            result = compose(
                unify_w(apply_ty(l, result), apply_ty(r, result)),
                result,
            )
        return result
    raise TypeError(f"ICE: Unexpected type {type(ty1)}")


class UnifyWTests(FreshTests):
    def test_tyvar_tyvar(self) -> None:
        self.assertEqual(unify_w(TyVar("a"), TyVar("b")), {"a": TyVar("b")})
        self.assertEqual(unify_w(TyVar("b"), TyVar("a")), {"b": TyVar("a")})

    def test_tyvar_tycon(self) -> None:
        self.assertEqual(unify_w(TyVar("a"), IntType), {"a": IntType})
        self.assertEqual(unify_w(IntType, TyVar("a")), {"a": IntType})

    def test_tycon_tycon_name_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify_w(IntType, BoolType)

    def test_tycon_tycon_arity_mismatch(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [])
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify_w(l, r)

    def test_tycon_tycon_unifies_arg(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [TyVar("b")])
        self.assertEqual(unify_w(l, r), {"a": TyVar("b")})

    def test_tycon_tycon_unifies_args(self) -> None:
        l = func_type(TyVar("a"), TyVar("b"))
        r = func_type(TyVar("c"), TyVar("d"))
        self.assertEqual(unify_w(l, r), {"a": TyVar("c"), "b": TyVar("d")})


def infer_w(expr: Object, ctx: Context) -> tuple[Subst, MonoType]:
    if isinstance(expr, Var):
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise TypeError(f"Unbound variable {expr.name}")
        return {}, instantiate(scheme)
    if isinstance(expr, Int):
        return {}, IntType
    if isinstance(expr, Function):
        arg_tyvar = fresh_tyvar("a")
        assert isinstance(expr.arg, Var)
        body_ctx = {**ctx, expr.arg.name: Forall([], arg_tyvar)}
        body_subst, body_ty = infer_w(expr.body, body_ctx)
        return body_subst, func_type(apply_ty(arg_tyvar, body_subst), body_ty)
    if isinstance(expr, Apply):
        s1, ty = infer_w(expr.func, ctx)
        s2, p = infer_w(expr.arg, apply_ctx(ctx, s1))
        r = fresh_tyvar("a")
        s3 = unify_w(apply_ty(ty, s2), TyCon("->", [p, r]))
        return compose(compose(s3, s2), s1), apply_ty(r, s3)
    if isinstance(expr, Binop):
        left, right = expr.left, expr.right
        op = Var(BinopKind.to_str(expr.op))
        return infer_w(Apply(Apply(op, left), right), ctx)
    raise TypeError(f"Unexpected type {type(expr)}")


class InferWTests(FreshTests):
    def test_unbound_var(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unbound variable"):
            infer_w(Var("a"), {})

    def test_var_instantiates_scheme(self) -> None:
        subst, _ = infer_w(Var("a"), {"a": Forall([TyVar("b")], TyVar("b"))})
        self.assertEqual(subst, {})

    def test_int(self) -> None:
        subst, _ = infer_w(Int(123), {})
        self.assertEqual(subst, {})

    def test_function_returns_arg(self) -> None:
        subst, _ = infer_w(Function(Var("x"), Var("x")), {})
        self.assertEqual(subst, {})

    def test_nested_function_outer(self) -> None:
        subst, _ = infer_w(Function(Var("x"), Function(Var("y"), Var("x"))), {})
        self.assertEqual(subst, {})

    def test_nested_function_inner(self) -> None:
        subst, _ = infer_w(Function(Var("x"), Function(Var("y"), Var("y"))), {})
        self.assertEqual(subst, {})

    def test_apply_id_int(self) -> None:
        func = Function(Var("x"), Var("x"))
        arg = Int(123)
        subst, _ = infer_w(Apply(func, arg), {})
        self.assertEqual(subst, {"a0": IntType, "a1": IntType})

    def test_apply_two_arg_returns_function(self) -> None:
        func = Function(Var("x"), Function(Var("y"), Var("x")))
        arg = Int(123)
        subst, _ = infer_w(Apply(func, arg), {})
        self.assertEqual(subst, {"a0": IntType, "a2": func_type(TyVar("a1"), IntType)})

    def test_binop_add_constrains_int(self) -> None:
        expr = Binop(BinopKind.ADD, Var("x"), Var("y"))
        subst, _ = infer_w(
            expr,
            {
                "x": Forall([], TyVar("a")),
                "y": Forall([], TyVar("b")),
                "+": Forall([], func_type(IntType, IntType, IntType)),
            },
        )
        self.assertEqual(
            subst,
            {"a": IntType, "a0": func_type(IntType, IntType), "b": IntType, "a1": IntType},
        )

    def test_binop_add_function_constrains_int(self) -> None:
        expr = Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))
        subst, _ = infer_w(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertEqual(
            subst,
            {"a0": IntType, "a2": func_type(IntType, IntType), "a1": IntType, "a3": IntType},
        )


def unify_j(ty1: Ty, ty2: Ty) -> None:
    ty1 = ty1.find()
    ty2 = ty2.find()
    if isinstance(ty1, TyVar):
        ty1.make_equal_to(ty2)
        return
    if isinstance(ty2, TyVar):  # Mirror
        return unify_j(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name:
            unify_fail(ty1, ty2)
            return
        if len(ty1.args) != len(ty2.args):
            unify_fail(ty1, ty2)
            return
        for l, r in zip(ty1.args, ty2.args):
            unify_j(l, r)
        return
    raise TypeError(f"ICE: Unexpected type {type(ty1)}")


class UnifyJTests(FreshTests):
    def test_tyvar_tyvar(self) -> None:
        a = TyVar("a")
        b = TyVar("b")
        unify_j(a, b)
        self.assertIs(a.find(), b.find())

    def test_tyvar_tycon(self) -> None:
        a = TyVar("a")
        unify_j(a, IntType)
        self.assertIs(a.find(), IntType)
        b = TyVar("b")
        unify_j(b, IntType)
        self.assertIs(b.find(), IntType)

    def test_tycon_tycon_name_mismatch(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify_j(IntType, BoolType)

    def test_tycon_tycon_arity_mismatch(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [])
        with self.assertRaisesRegex(TypeError, "Unification failed"):
            unify_j(l, r)

    def test_tycon_tycon_unifies_arg(self) -> None:
        a = TyVar("a")
        b = TyVar("b")
        l = TyCon("x", [a])
        r = TyCon("x", [b])
        unify_j(l, r)
        self.assertIs(a.find(), b.find())

    def test_tycon_tycon_unifies_args(self) -> None:
        a, b, c, d = map(TyVar, "abcd")
        l = func_type(a, b)
        r = func_type(c, d)
        unify_j(l, r)
        self.assertIs(a.find(), c.find())
        self.assertIs(b.find(), d.find())
        self.assertIsNot(a.find(), b.find())


def infer_j(expr: Object, ctx: Context) -> TyVar:
    result = fresh_tyvar()
    if isinstance(expr, Var):
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise TypeError(f"Unbound variable {expr.name}")
        unify_j(result, instantiate(scheme))
        return result
    if isinstance(expr, Int):
        unify_j(result, IntType)
        return result
    if isinstance(expr, Function):
        arg_tyvar = fresh_tyvar("a")
        assert isinstance(expr.arg, Var)
        body_ctx = {**ctx, expr.arg.name: Forall([], arg_tyvar)}
        body_ty = infer_j(expr.body, body_ctx)
        unify_j(result, func_type(arg_tyvar, body_ty))
        return result
    if isinstance(expr, Apply):
        func_ty = infer_j(expr.func, ctx)
        arg_ty = infer_j(expr.arg, ctx)
        unify_j(func_ty, TyCon("->", [arg_ty, result]))
        return result
    if isinstance(expr, Binop):
        left, right = expr.left, expr.right
        op = Var(BinopKind.to_str(expr.op))
        return infer_j(Apply(Apply(op, left), right), ctx)
    raise TypeError(f"Unexpected type {type(expr)}")


class BaseTestCases:
    class InferTestsBase(FreshTests):
        def infer(self, expr: Object, ctx: Context) -> Ty:
            raise NotImplementedError

        def test_unbound_var(self) -> None:
            with self.assertRaisesRegex(TypeError, "Unbound variable"):
                self.infer(Var("a"), {})

        def test_var_instantiates_scheme(self) -> None:
            ty = self.infer(Var("a"), {"a": Forall([TyVar("b")], TyVar("b"))})
            self.assertEqual(ty, TyVar("t0"))

        def test_int(self) -> None:
            ty = self.infer(Int(123), {})
            self.assertEqual(ty, IntType)

        def test_function_returns_arg(self) -> None:
            ty = self.infer(Function(Var("x"), Var("x")), {})
            self.assertEqual(ty, func_type(TyVar("a0"), TyVar("a0")))

        def test_nested_function_outer(self) -> None:
            ty = self.infer(Function(Var("x"), Function(Var("y"), Var("x"))), {})
            self.assertEqual(ty, func_type(TyVar("a0"), TyVar("a1"), TyVar("a0")))

        def test_nested_function_inner(self) -> None:
            ty = self.infer(Function(Var("x"), Function(Var("y"), Var("y"))), {})
            self.assertEqual(ty, func_type(TyVar("a0"), TyVar("a1"), TyVar("a1")))

        def test_apply_id_int(self) -> None:
            func = Function(Var("x"), Var("x"))
            arg = Int(123)
            ty = self.infer(Apply(func, arg), {})
            self.assertEqual(ty, IntType)

        def test_apply_two_arg_returns_function(self) -> None:
            func = Function(Var("x"), Function(Var("y"), Var("x")))
            arg = Int(123)
            ty = self.infer(Apply(func, arg), {})
            self.assertEqual(ty, func_type(TyVar("a1"), IntType))

        def test_binop_add_constrains_int(self) -> None:
            expr = Binop(BinopKind.ADD, Var("x"), Var("y"))
            ty = self.infer(
                expr,
                {
                    "x": Forall([], TyVar("a")),
                    "y": Forall([], TyVar("b")),
                    "+": Forall([], func_type(IntType, IntType, IntType)),
                },
            )
            self.assertEqual(ty, IntType)

        def test_binop_add_function_constrains_int(self) -> None:
            expr = Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))
            ty = self.infer(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
            self.assertEqual(ty, func_type(IntType, IntType, IntType))


class InferWSideBySideTests(BaseTestCases.InferTestsBase):
    def infer(self, expr: Object, ctx: Context) -> Ty:
        _, ty = infer_w(expr, ctx)
        return ty


class InferJSideBySideTests(FreshTests):
    def test_unbound_var(self) -> None:
        with self.assertRaisesRegex(TypeError, "Unbound variable"):
            infer_j(Var("a"), {})

    def test_var_instantiates_scheme(self) -> None:
        ty = infer_j(Var("a"), {"a": Forall([TyVar("b")], TyVar("b"))}).find()
        self.assertEqual(ty, TyVar("t1"))

    def test_int(self) -> None:
        ty = infer_j(Int(123), {}).find()
        self.assertEqual(ty, IntType)

    def test_function_returns_arg(self) -> None:
        ty = infer_j(Function(Var("x"), Var("x")), {}).find()
        self.assertTyEqual(ty, func_type(TyVar("a1"), TyVar("a1")))

    def test_nested_function_outer(self) -> None:
        ty = infer_j(Function(Var("x"), Function(Var("y"), Var("x"))), {}).find()
        self.assertTyEqual(ty, func_type(TyVar("a1"), TyVar("a3"), TyVar("a1")))

    def test_nested_function_inner(self) -> None:
        ty = infer_j(Function(Var("x"), Function(Var("y"), Var("y"))), {}).find()
        self.assertTyEqual(ty, func_type(TyVar("a1"), TyVar("a3"), TyVar("a3")))

    def test_apply_id_int(self) -> None:
        func = Function(Var("x"), Var("x"))
        arg = Int(123)
        ty = infer_j(Apply(func, arg), {}).find()
        self.assertIs(ty, IntType)

    def test_apply_two_arg_returns_function(self) -> None:
        func = Function(Var("x"), Function(Var("y"), Var("x")))
        arg = Int(123)
        ty = infer_j(Apply(func, arg), {}).find()
        self.assertTyEqual(ty, func_type(TyVar("a4"), IntType))

    def test_binop_add_constrains_int(self) -> None:
        expr = Binop(BinopKind.ADD, Var("x"), Var("y"))
        ty = infer_j(
            expr,
            {
                "x": Forall([], TyVar("a")),
                "y": Forall([], TyVar("b")),
                "+": Forall([], func_type(IntType, IntType, IntType)),
            },
        )
        self.assertEqual(ty.find(), IntType)

    def test_binop_add_function_constrains_int(self) -> None:
        expr = Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))
        ty = infer_j(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(IntType, IntType, IntType))


if __name__ == "__main__":
    unittest.main()
