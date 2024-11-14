import dataclasses
import sys
import typing

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


opcounter = 0


class Operation(Value):
    def __init__(self, name: str, args: list[Value], comment=None):
        self.name = name
        self.args = args
        self.comment = comment
        global opcounter
        self.id = opcounter
        opcounter += 1

    def v(self):
        return f"v{self.id}"

    def __repr__(self):
        return f"Operation({self.name}, {self.args})"

    def arg(self, index: int):
        return self.args[index]


Env = typing.Dict[str, Operation]


blockcounter = 0


class Block(list):
    def __init__(self):
        super().__init__()
        self.function = None
        global blockcounter
        self.id = blockcounter
        blockcounter += 1

    def set_function(self, function):
        self.function = function

    def opbuilder(opname):
        def wraparg(arg):
            if not isinstance(arg, Value):
                arg = Constant(arg)
            return arg

        def build(self, *args, comment=None):
            op = Operation(opname, [wraparg(arg) for arg in args], comment=comment)
            self.append(op)
            return op

        return build

    def __str__(self):
        return f"bb{self.id}"

    is_smallint = opbuilder("is_smallint")
    box_smallint = opbuilder("box_smallint")
    unbox_smallint = opbuilder("unbox_smallint")
    int_add = opbuilder("int_add")
    int_equal = opbuilder("int_equal")
    return_ = opbuilder("return")
    load_arg = opbuilder("load_arg")
    apply = opbuilder("apply")
    alloc_closure = opbuilder("alloc_closure")
    write_field = opbuilder("write_field")
    cond_branch = opbuilder("cond_branch")
    branch = opbuilder("branch")
    abort = opbuilder("abort")


@dataclasses.dataclass
class CompiledFunction:
    id: int = dataclasses.field(default=0, init=False, compare=False, hash=False)
    name: str
    params: typing.List[str]
    freevars: typing.Dict[str, Value] = dataclasses.field(default_factory=dict)
    blocks: typing.List[Block] = dataclasses.field(default_factory=list)

    def new_block(self) -> Block:
        block = Block()
        block.set_function(self)
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

    def lookup_name(self, block: Block, env: Env, name: str) -> Value:
        value = env.get(name)
        if value is not None:
            return value
        value = block.function.freevars.get(name)
        if value is not None:
            return value
        raise NameError(f"lookup_name: {name}")

    def try_match(self, fn: CompiledFunction, pattern_entry: Block, body_entry: Block, env: Env, arg: Value, pattern: Object, fallthrough: Block) -> Env:
        if isinstance(pattern, Int):
            assert -2**63 < pattern.value < 2**63, "not a smallint"
            is_smallint = pattern_entry.is_smallint(arg)
            block = fn.new_block()
            pattern_entry.cond_branch(is_smallint, block, fallthrough)
            unboxed = block.unbox_smallint(arg)
            eq = block.int_equal(unboxed, pattern.value)
            # TODO(max): Not body_entry but instead whatever the next block is
            # to match in the pattern; might be recursive
            block.cond_branch(eq, body_entry, fallthrough)
            return {}
        raise NotImplementedError(f"try_match: {type(pattern)}")

    def compile(self, block: Block, env: Env, exp: Object) -> Value:
        if isinstance(exp, Int):
            return block.box_smallint(exp.value)
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            name, value, body = exp.binding.name.name, exp.binding.value, exp.body
            value = self.compile(block, env, value)
            return self.compile(block, {**env, name: value}, body)
        if isinstance(exp, Var):
            return self.lookup_name(block, env, exp.name)
        if isinstance(exp, Binop):
            left = self.compile(block, env, exp.left)
            right = self.compile(block, env, exp.right)
            if exp.op == BinopKind.ADD:
                return block.int_add(left, right)
            if exp.op == BinopKind.MUL:
                return block.mul(left, right)
            raise NotImplementedError(f"compile: {exp.op}")
        if isinstance(exp, Function):
            argname = exp.arg.name
            freevars = free_in(exp)
            freevar_values = {var: self.lookup_name(block, env, var) for var in freevars}
            fn = CompiledFunction(self.gensym(), [argname], freevars=freevar_values)
            new_block = fn.new_block()
            arg = new_block.load_arg(0, comment=argname)
            self.compile(new_block, {argname: arg}, exp.body)
            new_block.return_(new_block[-1])
            self.functions.append(fn)
            result = block.alloc_closure(fn, len(freevars))
            for idx, var in enumerate(freevars):
                block.write_field(result, idx, freevar_values[var])
            return result
        if isinstance(exp, Apply):
            func = self.compile(block, env, exp.func)
            arg = self.compile(block, env, exp.arg)
            return block.apply(func, arg)
        if isinstance(exp, MatchFunction):
            argname = self.gensym()
            fn = CompiledFunction(self.gensym(), [argname])
            entry = fn.new_block()
            arg = entry.load_arg(0)
            case_blocks = [fn.new_block() for _ in exp.cases]
            case_blocks.append(fn.new_block())  # no match
            entry.branch(case_blocks[0])
            funcenv = {argname: arg}
            for i, case in enumerate(exp.cases):
                pattern_entry = case_blocks[i]
                fallthrough = case_blocks[i + 1]
                body_entry = fn.new_block()
                env_updates = self.try_match(fn, pattern_entry, body_entry,
                                             funcenv, arg, case.pattern,
                                             fallthrough)
                case_result = self.compile(body_entry, {**funcenv, **env_updates},
                                           case.body)
                body_entry.return_(case_result)
            case_blocks[-1].abort("No match")
            self.functions.append(fn)
            return Constant(fn)
        raise NotImplementedError(f"compile: {type(exp)}")


def bb_to_str(bb: Block, varprefix: str = "v", indent=""):
    def arg_to_str(arg: Value):
        if isinstance(arg, Constant):
            if isinstance(arg.value, CompiledFunction):
                return f"fn {arg.value.name}"
            return str(arg.value)
        return arg.v()

    def is_terminator(op):
        return op.name in ("return", "cond_branch", "branch", "abort")

    varnames = {}
    res = []
    for index, op in enumerate(bb):
        var = f"{varprefix}{index}"
        # varnames[op] = var
        arguments = ", ".join(arg_to_str(op.arg(i)) for i in range(len(op.args)))
        comment = f"  # {op.comment}" if op.comment else ""
        if is_terminator(op):
            strop = f"{indent}{op.name}({arguments}){comment}"
        else:
            strop = f"{indent}{op.v()} = {op.name}({arguments}){comment}"
        res.append(strop)
    return "\n".join(res)


# expr = parse(tokenize("inc (a + b) . a = 1 . b = 2 . inc = x -> x + 1 --| 0 -> 1 | 3 -> 4"))
with open(sys.argv[1]) as f:
    source = f.read()
expr = parse(tokenize(source))
fn = CompiledFunction("main", [])
block = fn.new_block()
compiler = Compiler(fn)
result = compiler.compile(block, {}, expr)
block.return_(result)

for fn in compiler.functions:
    if fn.freevars:
        fv_string = " "+", ".join([f"{v.v()}={k}" for k, v in fn.freevars.items()])
    else:
        fv_string = ""
    print(f"fn {fn.name}({', '.join(fn.params)}){fv_string}:")
    for block in fn.blocks:
        print(f"  {block}:")
        print(bb_to_str(block, "v", indent=" " * 4))
    print()
