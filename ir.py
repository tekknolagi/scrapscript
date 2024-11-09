import typing
import dataclasses

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
    type_of,
    IntType,
    StringType,
    parse,  # needed for /compilerepl
    tokenize,  # needed for /compilerepl
)


class Value:
    pass


class Constant(Value):
    def __init__(self, value: typing.Any):
        self.value = value

    def __repr__(self):
        return f"Constant({self.value})"


class Operation(Value):
    def __init__(self, name: str, args: list[Value]):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Operation({self.name}, {self.args})"

    def arg(self, index: int):
        return self.args[index]


Env = typing.Dict[str, Operation]


class Block(list):
    def opbuilder(opname):
        def wraparg(arg):
            if not isinstance(arg, Value):
                arg = Constant(arg)
            return arg

        def build(self, *args):
            op = Operation(opname, [wraparg(arg) for arg in args])
            self.append(op)
            return op

        return build

    box_small_int = opbuilder("box_small_int")
    int_add = opbuilder("int_add")
    return_ = opbuilder("return")
    load_arg = opbuilder("load_arg")
    apply = opbuilder("apply")


@dataclasses.dataclass
class CompiledFunction:
    id: int = dataclasses.field(default=0, init=False, compare=False, hash=False)
    name: str
    params: typing.List[str]
    fields: typing.List[str] = dataclasses.field(default_factory=list)
    blocks: typing.List[Block] = dataclasses.field(default_factory=list)

    def new_block(self) -> Block:
        block = Block()
        self.blocks.append(block)
        return block


class Compiler:
    def __init__(self, main_fn: CompiledFunction) -> None:
        self.gensym_counter: int = 0
        self.record_keys: typing.Dict[str, int] = {}
        self.record_builders: typing.Dict[Tuple[str, ...], CompiledFunction] = {}
        self.variant_tags: typing.Dict[str, int] = {}
        self.debug: bool = False
        self.functions: typing.List[CompiledFunction] = [main_fn]

    def gensym(self, stem: str = "tmp") -> str:
        self.gensym_counter += 1
        return f"{stem}_{self.gensym_counter-1}"

    def compile(self, block: Block, env: Env, exp: Object) -> str:
        if isinstance(exp, Int):
            return block.box_small_int(exp.value)
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            name, value, body = exp.binding.name.name, exp.binding.value, exp.body
            value = self.compile(block, env, value)
            return self.compile(block, {**env, name: value}, body)
        if isinstance(exp, Var):
            return env[exp.name]
        if isinstance(exp, Binop):
            left = self.compile(block, env, exp.left)
            right = self.compile(block, env, exp.right)
            if exp.op == BinopKind.ADD:
                return block.int_add(left, right)
            if exp.op == BinopKind.MUL:
                return block.mul(left, right)
            raise NotImplementedError(f"compile: {exp.op}")
        if isinstance(exp, Function):
            fn = CompiledFunction(self.gensym(), [exp.arg])
            new_block = fn.new_block()
            arg = new_block.load_arg(0)
            self.compile(new_block, {exp.arg.name: arg}, exp.body)
            new_block.return_(new_block[-1])
            self.functions.append(fn)
            return Constant(fn)
        if isinstance(exp, Apply):
            func = self.compile(block, env, exp.func)
            arg = self.compile(block, env, exp.arg)
            return block.apply(func, arg)
        raise NotImplementedError(f"compile: {type(exp)}")


def bb_to_str(bb: Block, varprefix: str = "v", indent=""):
    def arg_to_str(arg: Value):
        if isinstance(arg, Constant):
            if isinstance(arg.value, CompiledFunction):
                return f"fn {arg.value.name}"
            return str(arg.value)
        return varnames[arg]

    varnames = {}
    res = []
    for index, op in enumerate(bb):
        var = f"{varprefix}{index}"
        varnames[op] = var
        arguments = ", ".join(arg_to_str(op.arg(i)) for i in range(len(op.args)))
        strop = f"{indent}{var} = {op.name}({arguments})"
        res.append(strop)
    return "\n".join(res)


expr = parse(tokenize("inc (a + b) . a = 1 . b = 2 . inc = x -> x + 1"))
fn = CompiledFunction("main", [])
block = fn.new_block()
compiler = Compiler(fn)
compiler.compile(block, {}, expr)
block.return_(block[-1])

for fn in compiler.functions:
    print(f"fn {fn.name} {fn.params}")
    for index, block in enumerate(fn.blocks):
        print(f"  bb{index}:")
        print(bb_to_str(block, "v", indent=" " * 4))
    print()
