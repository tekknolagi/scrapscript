from dataclasses import dataclass
from scrapscript import Object, Binop, Int, String, BinopKind, parse, tokenize, Where, Assign, Var, Apply, List

Env = dict[str, str]


class Compiler:
    def __init__(self) -> None:
        self.code: list[str] = []
        self.gensym_counter: int = 0

    def gensym(self) -> str:
        self.gensym_counter += 1
        return f"tmp_{self.gensym_counter-1}"

    def _mktemp(self, exp: str) -> str:
        temp = self.gensym()
        # self.code.append(f"struct gc_obj* {temp} = {exp};")
        self.code.append(f"GC_HANDLE(struct gc_obj*, {temp}, {exp});")
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
            return self._mktemp(f"{callee}({arg})")
            raise NotImplementedError(f"apply {type(callee)} {callee}")
        if isinstance(exp, List):
            num_items = len(exp.items)
            items = [self.compile(env, item) for item in exp.items]
            result = self._mktemp(f"mklist(heap, {num_items})")
            for i, item in enumerate(items):
                self.code.append(f"list_set({result}, {i}, {item});")
            return result
        raise NotImplementedError(f"exp {type(exp)} {exp}")


program = parse(
    tokenize(
        """
println l
. d = a + b + c
. a = 1
. b = 2
. c = 3
. l = [1, 2, 3]
. print = runtime "print"
. println = runtime "println"
"""
    )
)


def main() -> None:
    compiler = Compiler()
    compiler.compile({}, program)
    with open("output.c", "w") as f:
        print('#include "runtime.c"\n', file=f)
        print("int main() {", file=f)
        print("HANDLES();", file=f)
        print("heap = make_heap(1024);", file=f)
        for line in compiler.code:
            print(line, file=f)
        print("}", file=f)


if __name__ == "__main__":
    main()
