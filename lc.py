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
    args: list[Ty]

    def __repr__(self) -> str:
        if not self.args:
            return self.name
        return f"({self.name.join(map(repr, self.args))})"


@dataclasses.dataclass
class Forall(Ty):
    tyvars: list[TyVar]
    ty: Ty

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


Subst = typing.Mapping[str, Ty]


def apply_ty(ty: Ty, subst: Subst) -> Ty:
    if isinstance(ty, TyVar):
        return subst.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [apply_ty(arg, subst) for arg in ty.args])
    if isinstance(ty, Forall):
        ty_args = {arg.name for arg in ty.tyvars}
        new_subst = {name: ty for name, ty in subst.items() if name not in ty_args}
        return Forall(ty.tyvars, apply_ty(ty.ty, new_subst))
    raise TypeError(f"Unknown type: {ty}")


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
        self.assertEqual(apply_ty(ty, {"a": TyVar("c")}), ty)
        self.assertEqual(apply_ty(ty, {"b": TyVar("c")}), Forall([TyVar("a")], func_type(TyVar("a"), TyVar("c"))))


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


if __name__ == "__main__":
    unittest.main()
