from scrapscript import Object, Binop, Int, BinopKind, parse, tokenize, Where, Assign, Var

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
        raise NotImplementedError(f"exp {type(exp)} {exp}")


def main() -> None:
    exp = parse(tokenize("d*2 . d = a + b + c . a = 1 . b = 2 . c = 3"))
    compiler = Compiler()
    compiler.compile({}, exp)
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
