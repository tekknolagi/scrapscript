#!/usr/bin/env python3
import argparse
import dataclasses
import itertools
from scrapscript import (
    Access,
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
    Record,
    Spread,
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


class Compiler:
    def __init__(self, main: CompiledFunction) -> None:
        self.gensym_counter: int = 0
        self.functions: list[CompiledFunction] = [main]
        self.function: CompiledFunction = main
        self.record_keys: dict[str, int] = {}

    def record_key(self, key: str) -> int:
        result = self.record_keys.get(key)
        if result is not None:
            return result
        result = self.record_keys[key] = len(self.record_keys)
        return result

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
        # TODO(max): Liveness analysis to avoid unnecessary handles
        self._emit(f"OBJECT_HANDLE({name}, {exp});")
        return name

    def _guard(self, cond: str, msg: Optional[str] = None) -> None:
        if msg is None:
            msg = f"assertion {cond!s} failed"
        self._emit(f"if (!({cond})) {{")
        self._emit(f'fprintf(stderr, "{msg}\\n");')
        self._emit("abort();")
        self._emit("}")

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

    def try_match(self, env: Env, arg: str, pattern: Object, fallthrough: str) -> Env:
        if isinstance(pattern, Int):
            self._emit(f"if (!is_num({arg})) {{ goto {fallthrough}; }}")
            self._emit(f"if (num_value({arg}) != {pattern.value}) {{ goto {fallthrough}; }}")
            return {}
        if isinstance(pattern, Var):
            return {pattern.name: arg}
        if isinstance(pattern, List):
            self._emit(f"if (!is_list({arg})) {{ goto {fallthrough}; }}")
            use_spread = sum(isinstance(item, Spread) for item in pattern.items)
            if use_spread:
                assert use_spread == 1
                # check min # of list elements
                num_real_patterns = len(pattern.items) - 1
                self._emit(f"if (list_size({arg}) < {num_real_patterns}) {{ goto {fallthrough}; }}")
            else:
                # check exact # of list elements
                self._emit(f"if (list_size({arg}) != {len(pattern.items)}) {{ goto {fallthrough}; }}")
            updates = {}
            for i, pattern_item in enumerate(pattern.items):
                if isinstance(pattern_item, Spread):
                    if pattern_item.name:
                        # TODO(max): Use cons cells for list or find a way to
                        # make stack-allocated lightweight views
                        list_rest = self._mktemp(f"list_rest({arg})")
                        updates[pattern_item.name] = list_rest
                    break
                list_item = self._mktemp(f"list_get({arg}, {i})")
                updates.update(self.try_match(env, list_item, pattern_item, fallthrough))
            return updates
        raise NotImplementedError("try_match", pattern)

    def compile_match_function(self, env: Env, exp: MatchFunction, name: Optional[str]) -> str:
        arg = self.gensym()
        fn = self.make_compiled_function(env, arg, exp, name)
        self.functions.append(fn)
        cur = self.function
        self.function = fn
        funcenv = self.compile_function_env(fn, name)
        for i, case in enumerate(exp.cases):
            self._emit(f"// case {i}")
            fallthrough = self.gensym()
            env_updates = self.try_match(funcenv, arg, case.pattern, fallthrough)
            case_result = self.compile({**funcenv, **env_updates}, case.body)
            self._emit(f"return {case_result};")
            self._emit(f"{fallthrough}:;")
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
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_add({left}, {right})")
            if exp.op == BinopKind.MUL:
                self._debug("collect(heap);")
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_mul({left}, {right})")
            if exp.op == BinopKind.SUB:
                self._debug("collect(heap);")
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_sub({left}, {right})")
            if exp.op == BinopKind.LIST_CONS:
                self._debug("collect(heap);")
                return self._mktemp(f"list_cons({left}, {right})")
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
            self._guard(f"is_closure({callee})", "attempting to call a non-closure")
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
        if isinstance(exp, Record):
            values: dict[str, str] = {}
            for key, value_exp in exp.data.items():
                values[key] = self.compile(env, value_exp)
            result = self._mktemp(f"mkrecord(heap, {len(values)})")
            for i, (key, value) in enumerate(values.items()):
                key_idx = self.record_key(key)
                self._emit(f"record_set({result}, /*index=*/{i}, /*key=*/{key_idx}, /*value=*/{value});")
            self._debug("collect(heap);")
            return result
        if isinstance(exp, Access):
            assert isinstance(exp.at, Var), f"only Var access is supported, got {type(exp.at)}"
            record = self.compile(env, exp.obj)
            key_idx = self.record_key(exp.at.name)
            # Check if the record is a record
            self._guard(f"is_record({record})", "not a record")
            value = self._mktemp(f"record_get({record}, {key_idx})")
            self._guard(f"{value} != NULL", f"missing key {exp.at.name!s}")
            return value
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
        print(f"#define OBJECT_HANDLE(name, exp) GC_HANDLE(struct gc_obj*, name, exp)", file=f)
        # Declare all functions
        print("const char* record_keys[] = {", file=f)
        for key in compiler.record_keys:
            print(f'"{key}",', file=f)
        print("};", file=f)
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
        print(f"struct gc_obj* result = {main.name}();", file=f)
        print("println(result);", file=f)
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
        cflags = os.environ.get("CFLAGS", "-O0 -ggdb")
        subprocess.run([cc, "-o", "a.out", *shlex.split(cflags), args.output], check=True)

    if args.run:
        import subprocess

        subprocess.run(["./a.out"], check=True)


if __name__ == "__main__":
    main()
