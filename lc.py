from __future__ import annotations
import dataclasses
import unittest
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


if __name__ == "__main__":
    unittest.main()
