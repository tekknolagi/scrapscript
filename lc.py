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
    tyvar: TyVar
    ty: Ty


IntType = TyCon("int", [])
BoolType = TyCon("bool", [])
IdFunc = Forall(TyVar("a"), TyCon("->", [TyVar("a"), TyVar("a")]))
NotFunc = TyCon("->", [BoolType, BoolType])
