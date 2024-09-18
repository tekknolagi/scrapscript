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

    def __repr__(self) -> str:
        return f"'{self.name}"


@dataclasses.dataclass
class TyCon(Ty):
    _params: list[Ty]
    name: str

    def __repr__(self) -> str:
        if self.params:
            return f"{' '.join(map(lambda x: str(x), self.params))} {self.name}"
        return self.name

    @property
    def params(self) -> list[Ty]:
        return [param.find() for param in self._params]

    def __eq__(self, other) -> bool:
        if not isinstance(other, TyCon):
            return NotImplemented
        return self.name == other.name and self.params == other.params


@dataclasses.dataclass
class TyRecord(Ty):
    _data: dict[str, Ty]

    def __repr__(self) -> str:
        return str(self.data)

    @property
    def data(self) -> dict[str, Ty]:
        return {name: ty.find() for name, ty in self._data.items()}

    def __eq__(self, other) -> bool:
        if not isinstance(other, TyRecord):
            return NotImplemented
        return self.data == other.data


@dataclasses.dataclass
class TyUnion(Ty):
    _types: list[Ty]

    def __repr__(self) -> str:
        return f"{' | '.join(map(str, self.types))}"

    @property
    def types(self) -> list[Ty]:
        return [ty.find() for ty in self._types]

    def __eq__(self, other) -> bool:
        if not isinstance(other, TyUnion):
            return NotImplemented
        return self.types == other.types


@dataclasses.dataclass
class TyFun(Ty):
    _arg: Ty
    _ret: Ty

    def __repr__(self) -> str:
        return f"{self.arg.find()} -> {self.ret.find()}"

    @property
    def arg(self) -> Ty:
        return self._arg.find()

    @property
    def ret(self) -> Ty:
        return self._ret.find()

    def __eq__(self, other) -> bool:
        if not isinstance(other, TyFun):
            return NotImplemented
        return self.arg == other.arg and self.ret == other.ret


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
                item_type = self.unify(item_type, self.constrain(varenv, item))
            self.unify(ann, TyCon([item_type], "list"))
            return ann
        if isinstance(exp, Record):
            data_ty = {name: self.constrain(varenv, value) for name, value in exp.data.items()}
            return self.unify(ann, TyRecord(data_ty))
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
            new_env = {**varenv, name.name: name_ty}
            value_ty = self.constrain(new_env, value)
            if isinstance(value, Function):
                value_ty = generalize(value_ty)
            body = exp.body
            self.unify(name_ty, value_ty)
            body_ty = self.constrain(new_env, body)
            return self.unify(ann, body_ty)
        if isinstance(exp, Apply):
            func_ty = self.constrain(varenv, exp.func)
            arg_ty = self.constrain(varenv, exp.arg)
            self.unify(func_ty, TyFun(arg_ty, ann))
            return ann
        if isinstance(exp, MatchFunction):
            assert exp.cases, "empty match function"
            case_tys = [self.constrain_case(varenv, case) for case in exp.cases]
            arg = self.union([case_ty.arg for case_ty in case_tys])
            ret = self.union([case_ty.ret for case_ty in case_tys])
            return self.unify(ann, TyFun(arg, ret))
        raise ValueError(f"unexpected expression {type(exp)} {exp}")

    def union(self, tys: list[Ty]) -> Ty:
        flat = []
        for ty in tys:
            if isinstance(ty, TyUnion):
                flat.extend(ty.types)
            else:
                flat.append(ty)
        unique = []
        for ty in flat:
            if ty not in unique:
                unique.append(ty)
        if len(unique) == 1:
            return unique[0]
        return TyUnion(unique)

    def constrain_case(self, varenv: dict[str, Ty], case: MatchCase) -> Ty:
        pattern_ty, pattern_env = self.constrain_pattern(varenv, case.pattern)
        body_ty = self.constrain({**varenv, **pattern_env}, case.body)
        return TyFun(pattern_ty, body_ty)

    def constrain_pattern(self, varenv: dict[str, Ty], pattern: Object) -> tuple[ty, dict[str, Ty]]:
        if isinstance(pattern, Int):
            return IntType, {}
        if isinstance(pattern, String):
            return StringType, {}
        if isinstance(pattern, Var):
            var_ty = fresh_var()
            return var_ty, {pattern.name: var_ty}
        if isinstance(pattern, List):
            item_ty = fresh_var()
            if not pattern.items:
                return Forall([item_ty], TyCon([item_ty], "list")), {}
            list_item_ty = fresh_var()
            result_env = {}
            for item in pattern.items:
                item_ty, item_env = self.constrain_pattern(varenv, item)
                list_item_ty = self.unify(list_item_ty, item_ty)
                result_env.update(item_env)
            return TyCon([list_item_ty], "list"), result_env
        raise ValueError(f"unexpected pattern {type(pattern)} {pattern}")

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
                self.error(left, right)
            if left.name != right.name:
                self.error(left, right)
            left.make_equal_to(right)
            for l, r in zip(left.params, right.params):
                self.unify(l, r)
            return left
        self.error(left, right)

    def error(self, left: Ty, right: Ty) -> None:
        raise TypeError(f"cannot unify {left} and {right}")


class TyperTests(unittest.TestCase):
    def setUp(self):
        global var_counter
        var_counter = iter(range(1000))

    def test_unify_int_str_raises_type_error(self):
        typer = Typer()
        with self.assertRaisesRegex(TypeError, "cannot unify int and string"):
            typer.unify(IntType, StringType)

    def test_unify_tycon_different_params(self):
        typer = Typer()
        with self.assertRaisesRegex(TypeError, "cannot unify foo and int string bar"):
            typer.unify(TyCon([], "foo"), TyCon([IntType, StringType], "bar"))

    def test_union_one_type_returns_type(self):
        typer = Typer()
        self.assertEqual(typer.union([IntType]), IntType)

    def test_union_two_same_types_returns_type(self):
        typer = Typer()
        self.assertEqual(typer.union([IntType, IntType]), IntType)

    def test_union_two_different_types_returns_union(self):
        typer = Typer()
        self.assertEqual(typer.union([IntType, StringType]), TyUnion([IntType, StringType]))

    def test_union_union_types_returns_union(self):
        typer = Typer()
        result = typer.union(
            [
                TyUnion([IntType, StringType]),
                StringType,
                TyUnion([IntType, FloatType]),
            ]
        )
        self.assertEqual(result, TyUnion([IntType, StringType, FloatType]))

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

    def test_constrain_where_recursive_function_returns_a_to_b(self):
        typer = Typer()
        exp = parse(tokenize("f . f = x -> f x"))
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyFun(TyVar("t7"), TyVar("t8")))

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
        typer = Typer()
        exp = List([Int(1), Int(2)])
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyCon([IntType], "list"))

    def test_constrain_mixed_list(self):
        typer = Typer()
        exp = List([Int(1), String("hi")])
        with self.assertRaisesRegex(TypeError, "cannot unify int and string"):
            typer.constrain({}, exp)

    def test_constrain_polymorphic_empty_list(self):
        typer = Typer()
        exp = parse(
            tokenize("""
        l1
        . l1 = "hello" >+ empty
        . l0 = 1 >+ empty
        . empty = []""")
        )
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyCon([StringType], "list"))

        typer = Typer()
        exp = parse(
            tokenize("""
        l0
        . l1 = "hello" >+ empty
        . l0 = 1 >+ empty
        . empty = []""")
        )
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyCon([IntType], "list"))

        typer = Typer()
        exp = parse(
            tokenize("""
        empty
        . l1 = "hello" >+ empty
        . l0 = 1 >+ empty
        . empty = []""")
        )
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), Forall([TyVar("t37")], TyCon([TyVar("t37")], "list")))

    def test_constrain_empty_rec(self):
        typer = Typer()
        exp = Record({})
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyRecord({}))

    def test_constrain_one_field_rec(self):
        typer = Typer()
        exp = Record({"a": Int(123)})
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyRecord({"a": IntType}))

    def test_constrain_two_field_rec(self):
        typer = Typer()
        exp = Record({"a": Int(123), "b": String("hello")})
        result = typer.constrain({}, exp)
        self.assertEqual(
            result.find(),
            TyRecord({"a": IntType, "b": StringType}),
        )

    def test_constrain_match_case_int_int(self):
        typer = Typer()
        exp = MatchCase(Int(1), Int(2))
        result = typer.constrain_case({}, exp)
        self.assertEqual(result.find(), TyFun(IntType, IntType))

    def test_constrain_match_case_string_int(self):
        typer = Typer()
        exp = MatchCase(String("x"), Int(2))
        result = typer.constrain_case({}, exp)
        self.assertEqual(result.find(), TyFun(StringType, IntType))

    def test_constrain_match_case_var_int(self):
        typer = Typer()
        exp = MatchCase(Var("_"), Int(2))
        result = typer.constrain_case({}, exp)
        self.assertEqual(result.find(), TyFun(TyVar("t0"), IntType))

    def test_constrain_match_case_var_var(self):
        typer = Typer()
        exp = MatchCase(Var("_"), Var("_"))
        result = typer.constrain_case({}, exp)
        self.assertEqual(result.find(), TyFun(TyVar("t0"), TyVar("t0")))

    def test_constrain_match_case_var_plus_one(self):
        typer = Typer()
        exp = MatchCase(Var("_"), Binop(BinopKind.ADD, Var("_"), Int(1)))
        result = typer.constrain_case({}, exp)
        self.assertEqual(result.find(), TyFun(IntType, IntType))

    def test_constrain_match_case_empty_list(self):
        typer = Typer()
        exp = MatchCase(List([]), Int(1))
        result = typer.constrain_case({}, exp)
        self.assertEqual(
            result.find(),
            TyFun(Forall([TyVar("t0")], TyCon([TyVar("t0")], "list")), IntType),
        )

    def test_constrain_match_case_var_list(self):
        typer = Typer()
        exp = MatchCase(List([Var("_")]), Int(1))
        result = typer.constrain_case({}, exp)
        self.assertEqual(
            result.find(),
            TyFun(TyCon([TyVar("t2")], "list"), IntType),
        )

    def test_constrain_match_case_var_list_plus_one(self):
        typer = Typer()
        exp = MatchCase(List([Var("_")]), Binop(BinopKind.ADD, Var("_"), Int(2)))
        result = typer.constrain_case({}, exp)
        self.assertEqual(
            result.find(),
            TyFun(TyCon([IntType], "list"), IntType),
        )

    def test_constrain_match_case_var_int_list(self):
        typer = Typer()
        exp = MatchCase(List([Int(1), Var("x")]), Var("x"))
        result = typer.constrain_case({}, exp)
        self.assertEqual(
            result.find(),
            TyFun(TyCon([IntType], "list"), IntType),
        )

    def test_constrain_match_function_one_case(self):
        typer = Typer()
        exp = MatchFunction([MatchCase(Int(1), Int(2))])
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyFun(IntType, IntType))

    def test_constrain_match_function_mismatched_cases(self):
        typer = Typer()
        exp = MatchFunction(
            [
                MatchCase(Int(1), Int(2)),
                MatchCase(Int(2), Int(2)),
            ]
        )
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyFun(IntType, IntType))

    def test_constrain_match_function_mismatched_cases(self):
        typer = Typer()
        exp = MatchFunction(
            [
                MatchCase(Int(1), Int(2)),
                MatchCase(String("x"), Int(2)),
                MatchCase(Int(3), Float(2.0)),
            ]
        )
        result = typer.constrain({}, exp)
        self.assertEqual(result.find(), TyFun(TyUnion([IntType, StringType]), TyUnion([IntType, FloatType])))


if __name__ == "__main__":
    unittest.main()
