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

    def _mktemp(self, exp: str) -> str:
        temp = self.gensym()
        self._emit(f"GC_HANDLE(struct gc_obj*, {temp}, {exp});")
        return temp

    def compile_assign(self, env: Env, exp: Assign) -> Env:
        assert isinstance(exp.name, Var)
        value = self.compile(env, exp.value)
        return {**env, exp.name.name: value}

    def compile(self, env: Env, exp: Object) -> str:
        if isinstance(exp, Int):
            # TODO(max): Bignum
            return self._mktemp(f"mknum(heap, {exp.value})")
        if isinstance(exp, Binop):
            left = self.compile(env, exp.left)
            right = self.compile(env, exp.right)
            if exp.op == BinopKind.ADD:
                return self._mktemp(f"num_add({left}, {right})")
            if exp.op == BinopKind.MUL:
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
            return self._mktemp(f"((struct closure*){callee})->fn({arg})")
            raise NotImplementedError(f"apply {type(callee)} {callee}")
        if isinstance(exp, List):
            num_items = len(exp.items)
            items = [self.compile(env, item) for item in exp.items]
            result = self._mktemp(f"mklist(heap, {num_items})")
            for i, item in enumerate(items):
                self._emit(f"list_set({result}, {i}, {item});")
            return result
        if isinstance(exp, Function):
            fields = sorted(free_in(exp))
            assert isinstance(exp.arg, Var)
            fn = CompiledFunction(params=[exp.arg.name], fields=fields)
            self.functions.append(fn)
            cur = self.function
            self.function = fn
            val = self.compile({exp.arg.name: exp.arg.name}, exp.body)
            fn.code.append(f"return {val};")
            self.function = cur

            name = self._mktemp(f"mkclosure(heap, {fn.name()}, {len(fields)})")
            for i, field in enumerate(fields):
                self._emit(f"closure_set({name}, {i}, {env[field]});")
            return name
        raise NotImplementedError(f"exp {type(exp)} {exp}")


program = parse(
    tokenize(
        """
println l
. l = [inc a, b, c, a + b + c]
. inc = x -> x + 1
. a = 1
. b = 2
. c = 3
. print = runtime "print"
. println = runtime "builtin_println"
"""
    )
)

BUILTINS = [
    "print",
    "println",
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
