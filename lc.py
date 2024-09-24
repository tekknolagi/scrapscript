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

IntType = TyCon("int", [])
BoolType = TyCon("bool", [])

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

    def __call__(self, other):
        if isinstance(other, Ty):
            return apply(self, other)
        if isinstance(other, Substitution):
            return Substitution(apply_tyenv(self, other.raw))
        assert isinstance(other, dict), f"it's a {type(other)}"
        return apply_tyenv(self, other)

    def union(self, other: Substitution) -> Substitution:
        return Substitution({**self.raw, **other.raw})

def apply(sub: Substitution, ty: Ty) -> Ty:
    if isinstance(ty, TyVar):
        return sub.raw.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [apply(sub, arg) for arg in ty.args])
    if isinstance(ty, Forall):
        q = ty.tyvar.name
        if q in sub.raw or any(q in free_vars(t) for t in sub.raw.values()):
            # Rename
            q_prime = fresh_tyvar()
            renamed_ty = apply(Substitution({q: q_prime}), ty.ty)
            return Forall(q_prime, apply(sub, renamed_ty))
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

def contains(ty: MonoType, var: TyVar) -> bool:
    if isinstance(ty, TyVar):
        return ty.name == var.name
    if isinstance(ty, TyCon):
        return any(contains(arg, var) for arg in ty.args)
    if isinstance(ty, Forall):
        return var.name in free_vars(ty)
    raise NotImplementedError

def unify(ty1: MonoType, ty2: MonoType) -> Substitution:
    if isinstance(ty1, TyVar) and isinstance(ty2, TyVar) and ty1.name == ty2.name:
        return Substitution({})
    if isinstance(ty1, TyVar):
        if contains(ty2, ty1):
            raise ValueError(f"recursive type {ty1} ; {ty2}")
        return Substitution({ty1.name: ty2})
    if isinstance(ty2, TyVar):
        return unify(ty2, ty1)  # flip it and reverse it
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name:
            raise TypeError(f"TyCon mismatch: {ty1.name} != {ty2.name}")
        if len(ty1.args) != len(ty2.args):
            raise TypeError(f"TyCon arity mismatch: {len(ty1.args)} != {len(ty2.args)}")
        s = Substitution({})
        for l, r in zip(ty1.args, ty2.args):
            s1 = unify(l, r)
            s2 = unify(apply(s1, l), apply(s1, r))
            # s = s1.union(s2).union(s)
            # s = unify(s(l), s(r))(s)
            # TODO(max): Iteratively apply(s) to l and r and unify(l, r)???
            # unify(apply(Substitution(s), l),
            #           apply(Substitution(s), r))
            # a = apply_tyenv(u, s)
            # s.update(a)
        return s
    raise NotImplementedError

def algorithm_w(env: TyEnv, expr: Object) -> tuple[Substitution, MonoType]:
    if isinstance(expr, Int):
        return Substitution({}), IntType
    if isinstance(expr, Var):
        ty = env.get(expr.name)
        if ty is None:
            raise TypeError(f"unbound local {expr.name}")
        return Substitution({}), instantiate(ty)
    if isinstance(expr, Function):
        beta = fresh_tyvar()
        assert isinstance(expr.arg, Var)
        s1, t1 = algorithm_w({**env, expr.arg.name: beta}, expr.body)
        return s1, apply(s1, ty_func([beta, t1]))
    if isinstance(expr, Apply):
        s1, t1 = algorithm_w(env, expr.func)
        s2, t2 = algorithm_w(apply_tyenv(s1, env), expr.arg)
        beta = fresh_tyvar()
        s3 = unify(apply(s2, t1), ty_func([t2, beta]))
        return Substitution(apply_tyenv(s3, apply_tyenv(s2, s1.raw))), apply(s3, beta)
    raise NotImplementedError

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
        self.assertEqual(apply(sub, ty), Forall(TyVar('t1'), TyVar('t1')))

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

class ContainsTest(unittest.TestCase):
    def test_contains_tyvar(self):
        self.assertTrue(contains(TyVar("a"), TyVar("a")))
        self.assertFalse(contains(TyVar("a"), TyVar("b")))

    def test_contains_tycon(self):
        self.assertTrue(contains(ty_func([TyVar("a"), TyVar("b")]), TyVar("a")))
        self.assertTrue(contains(ty_func([TyVar("a"), TyVar("b")]), TyVar("b")))
        self.assertFalse(contains(ty_func([TyVar("a"), TyVar("b")]), TyVar("c")))

    def test_contains_forall(self):
        self.assertFalse(contains(Forall(TyVar("a"), TyVar("a")), TyVar("a")))
        self.assertFalse(contains(Forall(TyVar("a"), TyVar("a")), TyVar("b")))
        self.assertTrue(contains(Forall(TyVar("a"), TyVar("b")), TyVar("b")))

class UnifyTest(unittest.TestCase):
    def test_unify_same_tyvar(self):
        self.assertEqual(unify(TyVar('a'), TyVar('a')), Substitution({}))

    def test_unify_recursive(self):
        with self.assertRaisesRegex(ValueError, "recursive type"):
            unify(TyVar('a'), TyCon('List', [TyVar('a')]))

        with self.assertRaisesRegex(ValueError, "recursive type"):
            unify(TyCon('List', [TyVar('a')]), TyVar('a'))

    def test_unify_different_tyvar(self):
        self.assertEqual(unify(TyVar('a'), TyVar('b')), Substitution({'a': TyVar('b')}))
        self.assertEqual(unify(TyVar('b'), TyVar('a')), Substitution({'b': TyVar('a')}))

    def test_unify_mismatched_tycon_name(self):
        l = TyCon("foo", [TyVar("a"), TyVar("b")])
        r = TyCon("bar", [TyVar("c"), TyVar("d")])
        with self.assertRaisesRegex(TypeError, "TyCon mismatch"):
            unify(l, r)

    def test_unify_mismatched_tycon_args(self):
        l = TyCon("foo", [TyVar("a"), TyVar("b")])
        r = TyCon("foo", [TyVar("c")])
        with self.assertRaisesRegex(TypeError, "TyCon arity mismatch"):
            unify(l, r)

    def test_unify_matching_tycon_of_tyvar(self):
        l = ty_func([TyVar("a"), TyVar("b")])
        r = ty_func([TyVar("c"), TyVar("d")])
        self.assertEqual(
            unify(l, r),
            Substitution({"a": TyVar("c"), "b": TyVar("d")}),
        )

    def test_unify_matching_tycon(self):
        func = ty_func([TyVar("a"), TyVar("b")])
        l = ty_func([func, TyVar("c")])
        r = ty_func([TyVar("d"), TyVar("e")])
        self.assertEqual(
            unify(l, r),
            Substitution({"d": func, "c": TyVar("e")}),
        )
        self.assertEqual(
            unify(r, l),
            Substitution({"d": func, "e": TyVar("c")}),
        )

class AlgorithmWTest(unittest.TestCase):
    def setUp(self):
        global current_tyvar
        current_tyvar = 0

    def test_int(self):
        subst, ty = algorithm_w({}, Int(123))
        self.assertEqual(subst, Substitution({}))
        self.assertEqual(ty, IntType)

    def test_unbound_var_raises(self):
        with self.assertRaisesRegex(TypeError, "unbound local"):
            algorithm_w({}, Var("x"))

    def test_var_returns_empty_subst(self):
        subst, _ = algorithm_w({"x": IntType}, Var("x"))
        self.assertEqual(subst, Substitution({}))

    def test_var_returns_associated_type(self):
        _, ty = algorithm_w({"x": IntType}, Var("x"))
        self.assertEqual(ty, IntType)

    def test_var_returns_associated_instantiated_type(self):
        _, ty = algorithm_w({"x": Forall(TyVar("a"), TyVar("a"))}, Var("x"))
        self.assertEqual(ty, TyVar("t1"))

    def test_function_returning_int(self):
        subst, ty = algorithm_w({}, Function(Var("x"), Int(123)))
        self.assertEqual(subst, Substitution({}))
        self.assertEqual(ty, ty_func([TyVar("t1"), IntType]))

    def test_function_returning_arg(self):
        subst, ty = algorithm_w({}, Function(Var("x"), Var("x")))
        self.assertEqual(subst, Substitution({}))
        self.assertEqual(ty, ty_func([TyVar("t1"), TyVar("t1")]))

    def test_apply(self):
        env = {"not": ty_func([BoolType, BoolType]),
               "even": ty_func([IntType, BoolType]),
               }
        subst, ty = algorithm_w(env,
                                Function(Var("x"), Apply(Var("not"),
                                                         Var("x"))))
        self.assertEqual(subst, Substitution({}))
        self.assertEqual(ty, ty_func([BoolType, BoolType]))

if __name__ == '__main__':
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
