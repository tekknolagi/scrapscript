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

    def __repr__(self):
        return f"'{self.name}"


@dataclasses.dataclass
class TyCon(MonoType):
    name: str
    args: list[MonoType]

    def __repr__(self):
        if not self.args:
            return self.name
        return f"({self.name.join(map(repr, self.args))})"


@dataclasses.dataclass
class Forall(Ty):
    tyvars: list[TyVar]
    ty: Ty

    def __repr__(self):
        return f"(forall {', '.join(map(repr, self.tyvars))}. {self.ty})"


UnitType = TyCon("()", [])
IntType = TyCon("int", [])
BoolType = TyCon("bool", [])
IdFunc = Forall([TyVar("a")], TyCon("->", [TyVar("a"), TyVar("a")]))
NotFunc = TyCon("->", [BoolType, BoolType])


class ReprTest(unittest.TestCase):
    def test_tyvar(self):
        self.assertEqual(repr(TyVar("a")), "'a")

    def test_tycon(self):
        self.assertEqual(repr(TyCon("int", [])), "int")

    def test_tycon_args(self):
        self.assertEqual(repr(TyCon("->", [IntType, IntType])), "(int->int)")

    def test_forall(self):
        self.assertEqual(repr(Forall([TyVar("a"), TyVar("b")], TyVar("a"))), "(forall 'a, 'b. 'a)")


def func_type(*args):
    return TyCon("->", list(args))


def tuple_type(*args):
    return TyCon("*", list(args))


if __name__ == "__main__":
    unittest.main()
