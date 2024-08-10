import dataclasses
import unittest
from scrapscript import (
    Access,
    Apply,
    Assign,
    Binop,
    BinopKind,
    Function,
    Hole,
    Int,
    List,
    MatchFunction,
    Object,
    Record,
    Spread,
    String,
    Var,
    Variant,
    Where,
    free_in,
    parse,
    tokenize,
)

Env = dict[str, str]

@dataclasses.dataclass
class Instr:
    opcode: str
    operands: tuple[str, ...]

    def __repr__(self) -> str:
        if not self.operands:
            return f"({self.opcode})"
        return f"({self.opcode} {' '.join(self.operands)})"

@dataclasses.dataclass
class Const(Instr):
    value: Object

    def __init__(self, value: Object) -> None:
        super().__init__("const", ())
        self.value = value

    def __repr__(self) -> str:
        return f"({self.opcode} {self.value})"

@dataclasses.dataclass
class ReadLocal(Instr):
    index: int

    def __init__(self, index: int) -> None:
        super().__init__("read-local", ())
        self.index = index

    def __repr__(self) -> str:
        return f"({self.opcode} {self.index})"

@dataclasses.dataclass
class WriteLocal(Instr):
    index: int

    def __init__(self, index: int) -> None:
        super().__init__("write-local", ())
        self.index = index

    def __repr__(self) -> str:
        return f"({self.opcode} {self.index})"

@dataclasses.dataclass
class CompiledFunction:
    name: str
    code: list[Instr]
    locals: list[str] = dataclasses.field(default_factory=list)

function_counter = iter(range(1000))

@dataclasses.dataclass
class Compiler:
    functions: list[CompiledFunction]
    function: CompiledFunction
    scope: list[int] = dataclasses.field(default_factory=lambda: [0])

    def __post_init__(self) -> None:
        self.functions.append(self.function)

    def _emit(self, opcode: str, *operands: str) -> None:
        self._emit_instr(Instr(opcode, operands))

    def _emit_instr(self, instr: Instr) -> None:
        self.function.code.append(instr)

    def _new_local(self, name: str) -> int:
        assert name not in self.function.locals
        index = len(self.function.locals)
        self.function.locals.append(name)
        self.scope[-1] += 1
        return index

    def enter_scope(self) -> None:
        self.scope.append(0)

    def exit_scope(self) -> None:
        num_locals = self.scope[-1]
        del self.function.locals[-num_locals:]
        self.scope.pop()

    def _local(self, name: str) -> int:
        return self.function.locals.index(name)

    def compile_assign(self, env: Env, exp: Assign) -> Env:
        assert isinstance(exp.name, Var)
        name = exp.name.name
        value = self.compile(env, exp.value)
        return {**env, name: value}

    def try_match(self, env: Env, pattern: Object, fail_label: str) -> None:
        if isinstance(pattern, Int):
            self._emit_instr(Const(pattern))
            self._emit("binop", "==")
            self._emit("branch-if-nonzero", fail_label)
            return
        if isinstance(pattern, Var):
            index = self._new_local(pattern.name)
            self._emit_instr(WriteLocal(index))
            return
        if isinstance(pattern, List):
            self._emit("dup")
            self._emit("is-list")
            self._emit("branch-if-nonzero", fail_label)
            use_spread = False
            for i, pattern_item in enumerate(pattern.items):
                if isinstance(pattern_item, Spread):
                    use_spread = True
                    break
                self._emit("dup")
                self._emit("is-empty-list")
                self._emit("branch-if-nonzero", fail_label)
                self._emit("split-list")  # car on top, cdr under
                self.try_match(env, pattern_item, fail_label)
            if not use_spread:
                self._emit("is-empty-list")
                self._emit("branch-if-nonzero", fail_label)
            return
        raise NotImplementedError(type(pattern))

    def compile(self, env: Env, exp: Object) -> None:
        if isinstance(exp, Int):
            self._emit_instr(Const(exp))
            return
        if isinstance(exp, String):
            self._emit_instr(Const(exp))
            return
        if isinstance(exp, Variant):
            self.compile(env, exp.value)
            self._emit("make-variant", exp.tag)
            return
        if isinstance(exp, Binop):
            self.compile(env, exp.left)
            self.compile(env, exp.right)
            self._emit("binop", exp.op.name)
            return
        if isinstance(exp, Function):
            prev = self.function
            self.function = new_fn = CompiledFunction(f"fn_{next(function_counter)}", [], [])
            self._new_local(exp.arg.name)
            self.compile({}, exp.body)
            self.functions.append(new_fn)
            self.function = prev
            self._emit("fun-ref", new_fn.name)
            return
        if isinstance(exp, MatchFunction):
            prev = self.function
            self.function = new_fn = CompiledFunction(f"fn_{next(function_counter)}", [], [])
            self._new_local("__match_arg__")

            for i, case in enumerate(exp.cases):
                self._emit("label", f"case_{i}")
                self.enter_scope()
                self._emit_instr(ReadLocal(0))
                fallthrough = f"case_{i+1}" if i < len(exp.cases) - 1 else "no_match"
                self.try_match(env, case.pattern, fallthrough)
                self._emit("label", f"body_{i}")
                self.compile({}, case.body)
                self._emit("return")
                self.exit_scope()
            self._emit("label", "no_match")
            self._emit("abort", "no matching cases")

            self.functions.append(new_fn)
            self.function = prev
            self._emit("fun-ref", new_fn.name)
            return
        if isinstance(exp, Apply):
            self.compile(env, exp.func)
            self.compile(env, exp.arg)
            self._emit("call-function")
            return
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            binding = exp.binding
            assert isinstance(binding.name, Var)
            name = binding.name.name
            index = self._new_local(name)
            self.compile(env, binding.value)
            self._emit_instr(WriteLocal(index))
            self.compile(env, exp.body)
            return
        if isinstance(exp, Var):
            self._emit_instr(ReadLocal(self._local(exp.name)))
            return
        raise NotImplementedError(type(exp))


class CompilerTests(unittest.TestCase):
    def test_scope(self) -> None:
        c = Compiler([], CompiledFunction("main", []))
        c._new_local("a")
        c.enter_scope()
        self.assertEqual(c._new_local("b"), 1)
        self.assertEqual(c._local("a"), 0)
        self.assertEqual(c._local("b"), 1)
        c.exit_scope()
        self.assertRaises(ValueError, c._local, "b")


unittest.main(exit=False)


code = """
f 3
-- . a = 1
-- . b = "2"
. f =
-- | 0 -> 1
-- | 1 -> 2
| [x] -> 4
| _ -> 3
-- . f = x -> x * 2
"""
c = Compiler([], CompiledFunction("main", []))
c.compile({}, parse(tokenize(code)))
for f in c.functions:
    print(f"{f.name}:")
    for op in f.code:
        print(op)
    print()
