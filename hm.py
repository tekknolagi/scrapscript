from __future__ import annotations
import dataclasses
import unittest


@dataclasses.dataclass
class Ty:
    forwarded: typing.Optional[Ty] = dataclasses.field(default=None, init=False)

    def find(self) -> Ty:
        result = self
        while isinstance(result, Ty):
            next = result.forwarded
            if next is None:
                return result
            result = next
        return result

    def _set_forwarded(self, other: Ty) -> None:
        self.forwarded = other

    def make_equal_to(self, other: Ty) -> None:
        self.find()._set_forwarded(other)


@dataclasses.dataclass
class TyVar(Ty):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, TyVar) and self.name == other.name

    def __repr__(self) -> str:
        return f"'{self.name}"


@dataclasses.dataclass
class TyCon(Ty):
    params: list[Ty]
    name: str

    def __repr__(self) -> str:
        if self.params:
            return f"{' '.join(map(lambda x: str(x.find()), self.params))} {self.name}"
        return self.name


@dataclasses.dataclass
class TyFun(Ty):
    arg: Ty
    ret: Ty

    def __repr__(self) -> str:
        return f"{self.arg.find()} -> {self.ret.find()}"


# @dataclasses.dataclass
# class TyApp(Ty):
#     args: list[Ty]
#     func: Ty
#
#     def __repr__(self) -> str:
#         return f"{' '.join(map(str, self.args))} {self.func}"


@dataclasses.dataclass
class Forall(Ty):
    bound: list[TyVar]
    ty: Ty

    def __repr__(self) -> str:
        return f"forall {', '.join(map(lambda x: str(x.find()), self.bound))}. {self.ty.find()}"


var_counter = iter(range(1000))


def fresh_var():
    return TyVar(f"t{next(var_counter)}")


def instantiate(scheme: Forall) -> Ty:
    bound = {var: fresh_var() for var in scheme.bound}
    return substitute(bound, scheme.ty)


def free_vars(ty: Ty) -> set[TyVar]:
    if isinstance(ty, TyVar):
        return {ty}
    if isinstance(ty, TyCon):
        return set().union(*[free_vars(param) for param in ty.params])
    if isinstance(ty, TyFun):
        return free_vars(ty.arg).union(free_vars(ty.ret))
    raise ValueError(f"unexpected type {ty}")


def generalize(ty: Ty) -> Forall:
    return Forall(sorted(free_vars(ty), key=lambda tv: tv.name), ty)


def substitute(subs: dict[TyVar, Ty], ty: Ty) -> Ty:
    if isinstance(ty, TyVar):
        return subs.get(ty, ty)
    if isinstance(ty, TyCon):
        return TyCon([substitute(subs, param) for param in ty.params], ty.name)
    if isinstance(ty, TyFun):
        return TyFun(substitute(subs, ty.arg), substitute(subs, ty.ret))
    raise ValueError(f"unexpected type {ty}")


class BasicTests(unittest.TestCase):
    def setUp(self):
        global var_counter
        var_counter = iter(range(1000))

    def test_tyvar(self):
        self.assertEqual(TyVar("a"), TyVar("a"))
        self.assertNotEqual(TyVar("a"), TyVar("b"))
        self.assertEqual(repr(TyVar("a")), "'a")

    def test_tycon(self):
        self.assertEqual(TyCon([], "int"), TyCon([], "int"))
        self.assertNotEqual(TyCon([], "int"), TyCon([], "float"))
        self.assertEqual(repr(TyCon([], "int")), "int")
        self.assertEqual(repr(TyCon([TyVar("a"), TyVar("b")], "list")), "'a 'b list")

    def test_tyfun(self):
        self.assertEqual(TyFun(TyVar("a"), TyVar("b")), TyFun(TyVar("a"), TyVar("b")))
        self.assertNotEqual(TyFun(TyVar("a"), TyVar("b")), TyFun(TyVar("a"), TyVar("c")))
        self.assertEqual(repr(TyFun(TyVar("a"), TyVar("b"))), "'a -> 'b")

    def test_forall(self):
        self.assertEqual(Forall([TyVar("a")], TyVar("a")), Forall([TyVar("a")], TyVar("a")))
        self.assertNotEqual(Forall([TyVar("a")], TyVar("a")), Forall([TyVar("b")], TyVar("a")))
        self.assertEqual(repr(Forall([TyVar("a"), TyVar("b")], TyVar("a"))), "forall 'a, 'b. 'a")

    def test_substitute_missing_tyvar(self):
        subs = {}
        self.assertEqual(substitute(subs, TyVar("a")), TyVar("a"))

    def test_substitute_tyvar(self):
        subs = {TyVar("a"): TyVar("b")}
        self.assertEqual(substitute(subs, TyVar("a")), TyVar("b"))

    def test_substitute_tycon(self):
        subs = {TyVar("a"): TyVar("b")}
        self.assertEqual(substitute(subs, TyCon([TyVar("a")], "list")), TyCon([TyVar("b")], "list"))

    def test_substitute_tyfun(self):
        subs = {TyVar("a"): TyVar("b")}
        self.assertEqual(substitute(subs, TyFun(TyVar("a"), TyVar("a"))), TyFun(TyVar("b"), TyVar("b")))

    def test_instantiate_tyvar(self):
        scheme = Forall([TyVar("a")], TyVar("a"))
        self.assertEqual(instantiate(scheme), TyVar("t0"))

    def test_instantiate_tyfun(self):
        scheme = Forall([TyVar("a"), TyVar("b")], TyFun(TyVar("a"), TyVar("b")))
        self.assertEqual(instantiate(scheme), TyFun(TyVar("t0"), TyVar("t1")))

    def test_free_vars_tyvar(self):
        self.assertEqual(free_vars(TyVar("a")), {TyVar("a")})

    def test_free_vars_tycon(self):
        self.assertEqual(free_vars(TyCon([TyVar("a"), TyVar("b")], "list")), {TyVar("a"), TyVar("b")})

    def test_free_vars_tyfun(self):
        self.assertEqual(free_vars(TyFun(TyVar("a"), TyVar("b"))), {TyVar("a"), TyVar("b")})

    def test_generalize_tyvar(self):
        self.assertEqual(
            generalize(TyVar("a")),
            Forall([TyVar("a")], TyVar("a")),
        )

    def test_generalize_tyfun(self):
        self.assertEqual(
            generalize(TyFun(TyVar("a"), TyVar("b"))),
            Forall([TyVar("a"), TyVar("b")], TyFun(TyVar("a"), TyVar("b"))),
        )

    def test_generalize_tycon(self):
        self.assertEqual(
            generalize(TyCon([TyVar("b"), TyVar("a")], "list")),
            Forall([TyVar("a"), TyVar("b")], TyCon([TyVar("b"), TyVar("a")], "list")),
        )


from scrapscript import (
    Object,
    Int,
    Float,
    String,
    Bytes,
    Var,
    Hole,
    Spread,
    Binop,
    BinopKind,
    List,
    Assign,
    Function,
    Apply,
    Where,
    Assert,
    EnvObject,
    MatchCase,
    MatchFunction,
    Closure,
    Record,
    Access,
    Variant,
    parse,
    tokenize,
)


IntType = TyCon([], "int")
StringType = TyCon([], "string")
FloatType = TyCon([], "float")


@dataclasses.dataclass
class Typer:
    # Map object ids to types.
    env: dict[int, Ty] = dataclasses.field(default_factory=dict)

    def annotation(self, exp: Object) -> Ty:
        result = self.env.get(exp.id)
        if result is None:
            result = fresh_var()
            self.env[exp.id] = result
        return result

    def constrain(self, varenv: dict[str, Ty], exp: Object):
        ann = self.annotation(exp)
        if isinstance(exp, Int):
            return self.unify(ann, IntType)
        if isinstance(exp, Float):
            return self.unify(ann, FloatType)
        if isinstance(exp, String):
            return self.unify(ann, StringType)
        if isinstance(exp, List):
            item_type = fresh_var()
            if not exp.items:
                return self.unify(ann, Forall([item_type], TyCon([item_type], "list")))
            for item in exp.items:
                self.unify(item_type, self.constrain(varenv, item))
            self.unify(ann, TyCon([item_type], "list"))
            return ann
        if isinstance(exp, Binop):
            left = self.constrain(varenv, exp.left)
            right = self.constrain(varenv, exp.right)
            if exp.op == BinopKind.ADD:
                self.unify(left, IntType)
                self.unify(right, IntType)
                return self.unify(ann, IntType)
            if exp.op == BinopKind.STRING_CONCAT:
                self.unify(left, StringType)
                self.unify(right, StringType)
                return self.unify(ann, StringType)
            if exp.op == BinopKind.LIST_APPEND:
                self.unify(left, TyCon([right], "list"))
                self.unify(ann, left)
                return ann
            if exp.op == BinopKind.LIST_CONS:
                ty = TyCon([left], "list")
                self.unify(right, ty)
                self.unify(ann, ty)
                return ann
            raise ValueError(f"unexpected binop {exp.op}")
        if isinstance(exp, Function):
            arg = fresh_var()
            body = self.constrain({**varenv, exp.arg.name: arg}, exp.body)
            return self.unify(ann, TyFun(arg, body))
        if isinstance(exp, Var):
            if var_ty := varenv.get(exp.name):
                return self.unify(ann, var_ty)
            raise NameError(f"unbound variable {exp.name}")
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            name = exp.binding.name
            name_ty = self.annotation(name)
            value = exp.binding.value
            value_ty = self.constrain(varenv, value)
            if isinstance(value, Function):
                value_ty = generalize(value_ty)
            body = exp.body
            self.unify(name_ty, value_ty)
            body_ty = self.constrain({**varenv, name.name: name_ty}, body)
            return self.unify(ann, body_ty)
        if isinstance(exp, Apply):
            func_ty = self.constrain(varenv, exp.func)
            arg_ty = self.constrain(varenv, exp.arg)
            self.unify(func_ty, TyFun(arg_ty, ann))
            return ann
        raise ValueError(f"unexpected expression {type(exp)} {exp}")

    def unify(self, left: Ty, right: Ty) -> Ty:
        left = left.find()
        right = right.find()
        if left == right:
            return left
        if isinstance(left, TyVar):
            left.make_equal_to(right)
            return right
        if isinstance(right, TyVar):
            right.make_equal_to(left)
            return left
        if isinstance(left, Forall):
            return self.unify(instantiate(left), right)
        if isinstance(right, Forall):
            return self.unify(left, instantiate(right))
        if isinstance(left, TyFun) and isinstance(right, TyFun):
            left.make_equal_to(right)
            self.unify(left.arg, right.arg)
            self.unify(left.ret, right.ret)
            return left
        if isinstance(left, TyCon) and isinstance(right, TyCon):
            if len(left.params) != len(right.params):
                raise TypeError(f"cannot unify {left} and {right}")
            if left.name != right.name:
                raise TypeError(f"cannot unify {left} and {right}")
            left.make_equal_to(right)
            for l, r in zip(left.params, right.params):
                self.unify(l, r)
            return left
        raise TypeError(f"cannot unify {left} and {right}")


class TyperTests(unittest.TestCase):
    def setUp(self):
        global var_counter
        var_counter = iter(range(1000))

    def test_unify_int_str_raises_type_error(self):
        typer = Typer()
        with self.assertRaisesRegex(TypeError, "cannot unify int and string"):
            typer.unify(IntType, StringType)

    def test_annotation(self):
        self.assertEqual(Typer().annotation(Int(42)), TyVar("t0"))

    def test_constrain_int(self):
        typer = Typer()
        exp = Int(42)
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[exp.id].find(), IntType)

    def test_constrain_string(self):
        typer = Typer()
        exp = String("hello")
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[exp.id].find(), StringType)

    def test_constrain_add(self):
        typer = Typer()
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[exp.left.id].find(), IntType)
        self.assertEqual(typer.env[exp.right.id].find(), IntType)
        self.assertEqual(typer.env[exp.id].find(), IntType)

    def test_constrain_function(self):
        typer = Typer()
        exp = Function(Var("x"), Var("x"))
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[exp.id].find(), TyFun(TyVar("t1"), TyVar("t1")))

    def test_constrain_var_does_not_instantiate_forall(self):
        typer = Typer()
        exp = Var("x")
        exp_ty = Forall([TyVar("a")], TyVar("a"))
        result = typer.constrain({"x": exp_ty}, exp)
        self.assertEqual(result, exp_ty)

    def test_constrain_var_looks_in_varenv(self):
        typer = Typer()
        exp = Var("x")
        result = typer.constrain({"x": IntType}, exp)
        self.assertEqual(result, IntType)

        with self.assertRaisesRegex(NameError, "unbound variable y"):
            typer.constrain({}, Var("y"))

    def test_constrain_where_binds_name(self):
        typer = Typer()
        exp = Where(Var("x"), Assign(Var("x"), Int(42)))
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[exp.id].find(), IntType)

    def test_constrain_where_function_generalizes(self):
        typer = Typer()
        exp = Where(Var("f"), Assign(Var("f"), Function(Var("x"), Var("x"))))
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(
            typer.env[exp.id].find(),
            Forall([TyVar("t3")], TyFun(TyVar("t3"), TyVar("t3"))),
        )

    def test_constrain_apply_function(self):
        typer = Typer()
        exp = Apply(Var("f"), Int(42))
        f_ty = TyFun(IntType, TyFun(StringType, IntType))
        result = typer.constrain({"f": f_ty}, exp)
        self.assertEqual(typer.env[exp.id].find(), TyFun(StringType, IntType))
        self.assertEqual(result.find(), TyFun(StringType, IntType))

    def test_constrain_forall_function(self):
        typer = Typer()
        x = Var("x")
        apply = Apply(Var("f"), Int(42))
        # Applying f to two different types, using Where for sequence
        exp = Where(
            apply,
            Assign(x, Apply(Var("f"), String("hello"))),
        )
        exp_ty = Forall([TyVar("a")], TyFun(TyVar("a"), TyVar("a")))
        result = typer.constrain({"f": exp_ty}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(typer.env[apply.id].find(), IntType)
        self.assertEqual(typer.env[x.id].find(), StringType)

    def test_constrain_forall_function_application(self):
        typer = Typer()
        exp = parse(tokenize("f 1 . f = x -> x"))
        result = typer.constrain({}, exp)
        self.assertEqual(typer.env[exp.id].find(), IntType)

    def test_constrain_empty_list(self):
        typer = Typer()
        exp = List([])
        result = typer.constrain({}, exp)
        self.assertIn(exp.id, typer.env)
        self.assertEqual(
            typer.env[exp.id].find(),
            Forall([TyVar("t1")], TyCon([TyVar("t1")], "list")),
        )

    def test_constrain_list_of_ints(self):
        pass

    def test_constrain_polymorphic_empty_list(self):
        typer = Typer()
        exp = parse(tokenize("""
        l1
        . l1 = "hello" >+ empty
        . l0 = 1 >+ empty
        . empty = []"""))
        result = typer.constrain({}, exp)
        self.assertEqual(typer.env[exp.id].find(), TyCon([StringType], "list"))

        typer = Typer()
        exp = parse(tokenize("""
        l0
        . l1 = "hello" >+ empty
        . l0 = 1 >+ empty
        . empty = []"""))
        result = typer.constrain({}, exp)
        self.assertEqual(typer.env[exp.id].find(), TyCon([IntType], "list"))


if __name__ == "__main__":
    unittest.main()
