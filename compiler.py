import argparse
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
from typing import Optional

Env = dict[str, str]


fn_counter = itertools.count()


@dataclasses.dataclass
class CompiledFunction:
    id: int = dataclasses.field(default=0, init=False, compare=False, hash=False)
    params: list[str]
    fields: list[str] = dataclasses.field(default_factory=list)
    code: list[str] = dataclasses.field(default_factory=list)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = next(fn_counter)
        self.code.append("HANDLES();")
        for param in self.params:
            # The parameters are raw pointers and must be updated on GC
            self.code.append(f"GC_PROTECT({param});")
        if self.name is None:
            self.name = f"fn_{self.id}"

    def decl(self) -> str:
        args = ", ".join(f"struct gc_obj* {arg}" for arg in self.params)
        return f"struct gc_obj* {self.name}({args})"


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

    def _handle(self, name: str, exp: str) -> str:
        self._emit(f"GC_HANDLE(struct gc_obj*, {name}, {exp});")
        return name

    def _mktemp(self, exp: str) -> str:
        temp = self.gensym()
        return self._handle(temp, exp)

    def compile_assign(self, env: Env, exp: Assign) -> Env:
        assert isinstance(exp.name, Var)
        name = exp.name.name
        if isinstance(exp.value, Function):
            # Named function
            value = self.compile_function(env, exp.value, name)
            return {**env, name: value}
        if isinstance(exp.value, MatchFunction):
            # Named match function
            value = self.compile_match_function(env, exp.value, name)
            return {**env, name: value}
        value = self.compile(env, exp.value)
        return {**env, name: value}

    def make_compiled_function(self, env: Env, arg: str, exp: Object, name: Optional[str]) -> CompiledFunction:
        assert isinstance(exp, (Function, MatchFunction))
        free = free_in(exp)
        if name is not None and name in free:
            free.remove(name)
        fields = sorted(free)
        return CompiledFunction(params=["this", arg], fields=fields)

    def compile_function_env(self, fn: CompiledFunction, name: Optional[str]) -> Env:
        result = {param: param for param in fn.params}
        if name is not None:
            result[name] = "this"
        for i, field in enumerate(fn.fields):
            result[field] = self._mktemp(f"closure_get(this, {i})")
        return result

    def compile_function(self, env: Env, exp: Function, name: Optional[str]) -> str:
        assert isinstance(exp.arg, Var)
        fn = self.make_compiled_function(env, exp.arg.name, exp, name)
        self.functions.append(fn)
        cur = self.function
        self.function = fn
        funcenv = self.compile_function_env(fn, name)
        val = self.compile(funcenv, exp.body)
        fn.code.append(f"return {val};")
        self.function = cur
        return self.make_closure(env, fn)

    def compile_match_function(self, env: Env, exp: MatchFunction, name: Optional[str]) -> str:
        arg = self.gensym()
        fn = self.make_compiled_function(env, arg, exp, name)
        self.functions.append(fn)
        cur = self.function
        self.function = fn
        funcenv = self.compile_function_env(fn, name)
        for case in exp.cases:
            self.compile_case(funcenv, case, arg)
        # TODO(max): (non-fatal?) exceptions
        self._emit(r'fprintf(stderr, "no matching cases\n");')
        self._emit("abort();")
        self.function = cur
        return self.make_closure(env, fn)

    def make_closure(self, env: Env, fn: CompiledFunction) -> str:
        name = self._mktemp(f"mkclosure(heap, {fn.name}, {len(fn.fields)})")
        for i, field in enumerate(fn.fields):
            self._emit(f"closure_set({name}, {i}, {env[field]});")
        self._debug("collect(heap);")
        return name

    def compile_case(self, env: Env, exp: MatchCase, arg: str) -> None:
        # Only called from inside MatchFunction; has explicit early return
        if isinstance(exp.pattern, Int):
            self._emit(f"if (is_num({arg}) && num_value({arg}) == {exp.pattern.value}) {{")
            result = self.compile(env, exp.body)
            self._emit(f"return {result};")
            self._emit("}")
            return
        if isinstance(exp.pattern, Var):
            new_env = {**env, exp.pattern.name: arg}
            result = self.compile(new_env, exp.body)
            self._emit(f"return {result};")
            return
        raise NotImplementedError(f"pattern {type(exp.pattern)}")

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
            if exp.op == BinopKind.SUB:
                self._debug("collect(heap);")
                return self._mktemp(f"num_sub({left}, {right})")
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
                    return f"builtin_{exp.arg.value}"
            callee = self.compile(env, exp.func)
            arg = self.compile(env, exp.arg)
            fn = self.gensym()
            self._emit(f"ClosureFn {fn} = closure_fn({callee});")
            return self._mktemp(f"(*{fn})((struct gc_obj*){callee}, {arg})")
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
            # Anonymous function
            return self.compile_function(env, exp, name=None)
        if isinstance(exp, MatchFunction):
            # Anonymous match function
            return self.compile_match_function(env, exp, name=None)
        raise NotImplementedError(f"exp {type(exp)} {exp}")


# TODO(max): Emit constants into the const heap
# The const heap must only point within the const heap
# The const heap will never be scanned
# The const heap can be serialized to disk and mmap'd

BUILTINS = [
    "print",
    "println",
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="scrapscript")
    parser.add_argument("file")
    parser.add_argument("-o", "--output", default="output.c")
    parser.add_argument("--format", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        source = f.read()
    program = parse(tokenize(source))

    main = CompiledFunction(params=[], name="scrap_main")
    compiler = Compiler(main)
    result = compiler.compile({}, program)
    main.code.append(f"return {result};")

    for builtin in BUILTINS:
        fn = CompiledFunction(params=["this", "arg"], name=f"builtin_{builtin}_wrapper")
        fn.code.append(f"return {builtin}(arg);")
        compiler.functions.append(fn)

    with open(args.output, "w") as f:
        print('#include "runtime.c"\n', file=f)
        # Declare all functions
        for function in compiler.functions:
            print(function.decl() + ";", file=f)
        for builtin in BUILTINS:
            print(f"struct gc_obj* builtin_{builtin} = NULL;", file=f)
        for function in compiler.functions:
            print(f"{function.decl()} {{", file=f)
            for line in function.code:
                print(line, file=f)
            print("}", file=f)
        print("int main() {", file=f)
        print("heap = make_heap(1024);", file=f)
        print("HANDLES();", file=f)
        for builtin in BUILTINS:
            print(f"builtin_{builtin} = mkclosure(heap, builtin_{builtin}_wrapper, 0);", file=f)
            print(f"GC_PROTECT(builtin_{builtin});", file=f)
        print(f"{main.name}();", file=f)
        print("destroy_heap(heap);", file=f)
        print("}", file=f)

    if args.format:
        import subprocess

        subprocess.run(["clang-format-15", "-i", args.output], check=True)

    if args.compile:
        import os
        import shlex
        import subprocess

        cc = os.environ.get("CC", "clang")
        cflags = os.environ.get("CFLAGS", "-O0 -ggdb -DNDEBUG")
        subprocess.run([cc, "-o", "a.out", *shlex.split(cflags), args.output], check=True)

    if args.run:
        import subprocess

        subprocess.run(["./a.out"], check=True)


if __name__ == "__main__":
    main()
