import dataclasses
import unittest
from scrapscript import (
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

@dataclasses.dataclass
class TyCon(MonoType):
    name: str
    args: list[MonoType]

def ty_func(tys: list[MonoType]) -> TyCon:
    return TyCon('->', tys)

def ty_list(ty: MonoType) -> TyCon:
    return TyCon('List', [ty])

@dataclasses.dataclass
class Forall(Ty):
    tyvar: TyVar
    ty: Ty

TyEnv = dict[str, Ty]

@dataclasses.dataclass
class Substitution:
    raw: TyEnv

def apply(sub: Substitution, ty: Ty) -> Ty:
    if isinstance(ty, TyVar):
        return sub.raw.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [apply(sub, arg) for arg in ty.args])
    if isinstance(ty, Forall):
        return Forall(ty.tyvar, apply(sub, ty.ty))
    raise NotImplementedError

def apply_tyenv(sub: Substitution, ty: TyEnv) -> TyEnv:
    return {k: apply(sub, v) for k, v in ty.items()}

current_tyvar = 0

def fresh_tyvar() -> TyVar:
    global current_tyvar
    current_tyvar += 1
    return TyVar(f't{current_tyvar}')

def instantiate_sub(ty: Forall, sub: dict[str, MonoType]) -> MonoType:
    if isinstance(ty, TyVar):
        return sub.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [instantiate_sub(arg, sub) for arg in ty.args])
    if isinstance(ty, Forall):
        return instantiate_sub(ty.ty, {**sub, ty.tyvar.name: fresh_tyvar()})
    raise NotImplementedError

def instantiate(ty: Forall) -> MonoType:
    return instantiate_sub(ty, {})

def free_vars(ty: Ty) -> set[str]:
    if isinstance(ty, TyVar):
        return {ty.name}
    if isinstance(ty, TyCon):
        return set().union(*[free_vars(arg) for arg in ty.args])
    if isinstance(ty, Forall):
        return free_vars(ty.ty) - {ty.tyvar.name}
    raise ValueError(f"unexpected type {ty}")

def generalize(ty: Ty) -> Forall:
    fv = reversed(sorted(free_vars(ty)))
    result = ty
    for var in fv:
        result = Forall(TyVar(var), result)
    return result

class ApplyTest(unittest.TestCase):
    def test_apply_var(self):
        sub = Substitution({'a': TyVar('b')})
        ty = TyVar('a')
        self.assertEqual(apply(sub, ty), TyVar('b'))

    def test_apply_tycon(self):
        sub = Substitution({'a': TyVar('b')})
        ty = TyCon('List', [TyVar('a')])
        self.assertEqual(apply(sub, ty), TyCon('List', [TyVar('b')]))

    def test_apply_forall(self):
        sub = Substitution({'a': TyVar('b')})
        ty = Forall(TyVar('a'), TyVar('a'))
        self.assertEqual(apply(sub, ty), Forall(TyVar('a'), TyVar('b')))

    def test_apply_tyenv(self):
        sub = Substitution({'a': TyVar('b')})
        tyenv = {'x': TyVar('a')}
        self.assertEqual(apply_tyenv(sub, tyenv), {'x': TyVar('b')})

class InstantiateTest(unittest.TestCase):
    def setUp(self):
        global current_tyvar
        current_tyvar = 0

    def test_instantiate_forall_tyvar(self):
        ty = Forall(TyVar('a'), TyVar('a'))
        self.assertEqual(instantiate(ty), TyVar('t1'))

    def test_instantiate_forall_tycon(self):
        ty = Forall(TyVar('a'), TyCon('List', [TyVar('a')]))
        self.assertEqual(instantiate(ty), TyCon('List', [TyVar('t1')]))

    def test_instantiate_forall_forall(self):
        ty = Forall(TyVar('a'), Forall(TyVar('b'), ty_func([TyVar('a'), TyVar('b')])))
        self.assertEqual(instantiate(ty), ty_func([TyVar('t1'), TyVar('t2')]))

class FreeVarsTest(unittest.TestCase):
    def test_free_vars_tyvar(self):
        ty = TyVar('a')
        self.assertEqual(free_vars(ty), {'a'})

    def test_free_vars_tycon(self):
        ty = TyCon('List', [TyVar('a')])
        self.assertEqual(free_vars(ty), {'a'})

    def test_free_vars_forall(self):
        ty = Forall(TyVar('a'), ty_func([TyVar('a'), TyVar('b')]))
        self.assertEqual(free_vars(ty), {'b'})

class GeneralizeTest(unittest.TestCase):
    def test_generalize_tyvar(self):
        ty = TyVar('a')
        self.assertEqual(generalize(ty), Forall(TyVar('a'), TyVar('a')))

    def test_generalize_tycon(self):
        ty = TyCon('->', [TyVar('b'), TyVar('a')])
        self.assertEqual(generalize(ty), Forall(TyVar('a'), Forall(TyVar('b'), ty)))

    def test_generalize_forall(self):
        ty = Forall(TyVar('a'), ty_func([TyVar('a'), TyVar('b')]))
        self.assertEqual(generalize(ty), Forall(TyVar('b'), ty))

if __name__ == '__main__':
    unittest.main()
