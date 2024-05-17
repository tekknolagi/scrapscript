import dataclasses
import itertools
from scrapscript import (
    Apply,
    Assign,
    Binop,
    BinopKind,
    Function,
    Int,
    List,
    MatchCase,
    MatchFunction,
    Object,
    String,
    Var,
    Where,
    free_in,
    parse,
    tokenize,
)

Env = dict[str, str]


fn_counter = itertools.count()


@dataclasses.dataclass
class CompiledFunction:
    id: int = dataclasses.field(default=0, init=False, compare=False, hash=False)
    params: list[str]
    fields: list[str] = dataclasses.field(default_factory=list)
    code: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = next(fn_counter)
        self.code.append("HANDLES();")

    def name(self) -> str:
        return f"fn_{self.id}"

    def decl(self) -> str:
        args = ", ".join(f"struct gc_obj* {arg}" for arg in self.params)
        return f"struct gc_obj* {self.name()}({args})"


# TODO(max): Only pass around handles, not raw pointers; arguments might be
# moved at any time by collection


class Compiler:
    def __init__(self, main: CompiledFunction) -> None:
        self.gensym_counter: int = 0
        self.functions: list[CompiledFunction] = [main]
        self.function: CompiledFunction = main

    def gensym(self) -> str:
        self.gensym_counter += 1
        return f"tmp_{self.gensym_counter-1}"

    def _emit(self, line: str) -> None:
        self.function.code.append(line)

    def _debug(self, line: str) -> None:
        self._emit("#ifndef NDEBUG")
        self._emit(line)
        self._emit("#endif")

    def _mktemp(self, exp: str) -> str:
        temp = self.gensym()
        self._emit(f"GC_HANDLE(struct gc_obj*, {temp}, {exp});")
        return temp

    def compile_assign(self, env: Env, exp: Assign) -> Env:
        assert isinstance(exp.name, Var)
        value = self.compile(env, exp.value)
        return {**env, exp.name.name: value}

    def make_closure(self, env: Env, fn: CompiledFunction) -> str:
        name = self._mktemp(f"mkclosure(heap, {fn.name()}, {len(fn.fields)})")
        for i, field in enumerate(fn.fields):
            self._emit(f"closure_set({name}, {i}, {env[field]});")
        self._debug("collect(heap);")
        return name

    def compile_case(self, env: Env, exp: MatchCase, arg: str) -> None:
        # Only called from inside MatchFunction; has explicit early return
        if not isinstance(exp.pattern, Int):
            raise NotImplementedError("pattern {type(exp.pattern)}")
        self._emit(f"if (is_num({arg}) && ((struct num*){arg})->value == {exp.pattern.value}) {{")
        result = self.compile(env, exp.body)
        self._emit(f"return {result};")
        self._emit("}")

    def compile(self, env: Env, exp: Object) -> str:
        if isinstance(exp, Int):
            # TODO(max): Bignum
            self._debug("collect(heap);")
            return self._mktemp(f"mknum(heap, {exp.value})")
        if isinstance(exp, Binop):
            left = self.compile(env, exp.left)
            right = self.compile(env, exp.right)
            if exp.op == BinopKind.ADD:
                self._debug("collect(heap);")
                return self._mktemp(f"num_add({left}, {right})")
            if exp.op == BinopKind.MUL:
                self._debug("collect(heap);")
                return self._mktemp(f"num_mul({left}, {right})")
            raise NotImplementedError(f"binop {exp.op}")
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            res_env = self.compile_assign(env, exp.binding)
            new_env = {**env, **res_env}
            return self.compile(new_env, exp.body)
        if isinstance(exp, Var):
            value = env.get(exp.name)
            if value is None:
                raise NameError(f"name '{exp.name}' is not defined")
            return value
        if isinstance(exp, Apply):
            if isinstance(exp.func, Var):
                if exp.func.name == "runtime":
                    assert isinstance(exp.arg, String)
                    return exp.arg.value
            callee = self.compile(env, exp.func)
            arg = self.compile(env, exp.arg)
            return self._mktemp(f"((struct closure*){callee})->fn((struct gc_obj*){callee}, {arg})")
            raise NotImplementedError(f"apply {type(callee)} {callee}")
        if isinstance(exp, List):
            num_items = len(exp.items)
            items = [self.compile(env, item) for item in exp.items]
            result = self._mktemp(f"mklist(heap, {num_items})")
            for i, item in enumerate(items):
                self._emit(f"list_set({result}, {i}, {item});")
            self._debug("collect(heap);")
            return result
        if isinstance(exp, Function):
            fields = sorted(free_in(exp))
            assert isinstance(exp.arg, Var)
            fn = CompiledFunction(params=["this", exp.arg.name], fields=fields)
            self.functions.append(fn)
            cur = self.function
            self.function = fn
            funcenv = {exp.arg.name: exp.arg.name}
            for i, field in enumerate(fields):
                funcenv[field] = self._mktemp(f"closure_get(this, {i})")
            val = self.compile(funcenv, exp.body)
            fn.code.append(f"return {val};")
            self.function = cur
            return self.make_closure(env, fn)
        if isinstance(exp, MatchFunction):
            fields = sorted(free_in(exp))
            arg = self.gensym()
            fn = CompiledFunction(params=["this", arg], fields=fields)
            self.functions.append(fn)
            cur = self.function
            self.function = fn
            funcenv = {arg: arg}
            for i, field in enumerate(fields):
                funcenv[field] = self._mktemp(f"closure_get(this, {i})")
            for case in exp.cases:
                self.compile_case(funcenv, case, arg)
            # TODO(max): (non-fatal?) exceptions
            self._emit(r'fprintf(stderr, "no matching cases\n");')
            self._emit("abort();")
            self.function = cur
            return self.make_closure(env, fn)
        raise NotImplementedError(f"exp {type(exp)} {exp}")


program = parse(
    tokenize(
        """
println (mklist 3 4)
. mklist = x -> y -> [is_even x, is_even y]
. println = runtime "builtin_println_wrapper"
. is_even = | 0 -> 1
            | 1 -> 0
            | 2 -> 1
            | 3 -> 0
            | 4 -> 1
"""
    )
)

BUILTINS = [
    "print_wrapper",
    "println_wrapper",
]


def main() -> None:
    main = CompiledFunction(params=[])
    compiler = Compiler(main)
    result = compiler.compile({}, program)
    main.code.append(f"return {result};")
    with open("output.c", "w") as f:
        print('#include "runtime.c"\n', file=f)
        # Declare all functions
        for function in compiler.functions:
            print(function.decl() + ";", file=f)
        for builtin in BUILTINS:
            print(f"struct closure* builtin_{builtin} = NULL;", file=f)
        for function in compiler.functions:
            print(f"{function.decl()} {{", file=f)
            for line in function.code:
                print(line, file=f)
            print("}", file=f)
        print("int main() {", file=f)
        print("heap = make_heap(1024);", file=f)
        print("HANDLES();", file=f)
        for builtin in BUILTINS:
            print(f"builtin_{builtin} = (struct closure*)mkclosure(heap, {builtin}, 0);", file=f)
            print(f"GC_PROTECT(builtin_{builtin});", file=f)
        print(f"{main.name()}();", file=f)
        print("destroy_heap(heap);", file=f)
        print("}", file=f)


if __name__ == "__main__":
    main()
