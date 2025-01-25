#!/usr/bin/env python3
from __future__ import annotations
import dataclasses
import io
import itertools
import json
import os
import typing
import unittest

from typing import Dict, Optional, Tuple

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
    parse,
    tokenize,
)


@dataclasses.dataclass
class InstrId:
    data: dict[Instr, int] = dataclasses.field(default_factory=dict)

    def __getitem__(self, instr: Instr) -> int:
        id = self.data.get(instr)
        if id is not None:
            return id
        id = len(self.data)
        self.data[instr] = id
        return id


@dataclasses.dataclass(eq=False)
class Instr:
    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def to_string(self, gvn: InstrId) -> str:
        return type(self).__name__


@dataclasses.dataclass(eq=False)
class Const(Instr):
    value: Object

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.value}>"


@dataclasses.dataclass(eq=False)
class Param(Instr):
    idx: int
    name: str

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.idx}; {self.name}>"


@dataclasses.dataclass(eq=False)
class MatchFail(Instr):
    pass


@dataclasses.dataclass(eq=False)
class HasOperands(Instr):
    operands: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def __init__(self, *operands: Instr) -> None:
        self.operands = list(operands)

    def to_string(self, gvn: InstrId) -> str:
        stem = f"{type(self).__name__}"
        if not self.operands:
            return stem
        return stem + " " + ", ".join(f"v{gvn[op]}" for op in self.operands)


@dataclasses.dataclass(init=False, eq=False)
class IntAdd(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class IntSub(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class IntMul(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class IntLess(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class IsNumEqualWord(HasOperands):
    expected: int

    def __init__(self, value: Instr, expected: int) -> None:
        self.operands = [value]
        self.expected = expected

    def to_string(self, gvn: InstrId) -> str:
        return super().to_string(gvn) + f", {self.expected}"


@dataclasses.dataclass(eq=False)
class ClosureRef(HasOperands):
    idx: int
    name: str

    def __init__(self, closure: Instr, idx: int, name: str) -> None:
        self.operands = [closure]
        self.idx = idx
        self.name = name

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.idx}; {self.name}> v{gvn[self.operands[0]]}"


@dataclasses.dataclass(init=False, eq=False)
class IsList(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class IsEmptyList(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class ListCons(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class ListFirst(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class ListRest(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class Call(HasOperands):
    pass


@dataclasses.dataclass(eq=False)
class Control(Instr):
    def succs(self) -> tuple[Block, ...]:
        raise NotImplementedError("succs")


@dataclasses.dataclass(eq=False)
class NewClosure(HasOperands):
    fn: IRFunction

    def __init__(self, fn: IRFunction, bound: list[Instr]) -> None:
        self.fn = fn
        self.operands = bound.copy()

    def to_string(self, gvn: InstrId) -> str:
        stem = f"{type(self).__name__}<{self.fn.name()}>"
        if not self.operands:
            return stem
        return f"{stem} " + ", ".join(f"v{gvn[op]}" for op in self.operands)


Env = Dict[str, Instr]


@dataclasses.dataclass(eq=False)
class Block:
    id: int
    instrs: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def append(self, instr: Instr) -> Instr:
        self.instrs.append(instr)
        return instr

    def name(self) -> str:
        return f"bb{self.id}"

    def terminator(self) -> Control:
        result = self.instrs[-1]
        assert isinstance(result, Control)
        return result


@dataclasses.dataclass(eq=False)
class Jump(Control):
    target: Block

    def to_string(self, gvn: InstrId) -> str:
        return super().to_string(gvn) + f" {self.target.name()}"

    def succs(self) -> tuple[Block, ...]:
        return (self.target,)


@dataclasses.dataclass(init=False, eq=False)
class Return(HasOperands, Control):
    pass

    def succs(self) -> tuple[Block, ...]:
        return ()


@dataclasses.dataclass(init=False, eq=False)
class CondBranch(Control, HasOperands):
    conseq: Block
    alt: Block

    def __init__(self, cond: Instr, conseq: Block, alt: Block) -> None:
        self.conseq = conseq
        self.alt = alt
        self.operands = [cond]

    def to_string(self, gvn: InstrId) -> str:
        return super().to_string(gvn) + f", {self.conseq.name()}, {self.alt.name()}"

    def succs(self) -> tuple[Block, ...]:
        return (self.conseq, self.alt)


@dataclasses.dataclass
class CFG:
    blocks: list[Block] = dataclasses.field(init=False, default_factory=list)
    entry: Block = dataclasses.field(init=False)

    def __init__(self) -> None:
        self.blocks = []
        self.entry = self.new_block()

    def new_block(self) -> Block:
        result = Block(len(self.blocks))
        self.blocks.append(result)
        return result

    def to_string(self, fn: IRFunction, gvn: InstrId) -> str:
        result = ""
        for block in self.blocks:
            result += f"  {block.name()} {{\n"
            for instr in block.instrs:
                if isinstance(instr, Control):
                    result += f"    {instr.to_string(gvn)}\n"
                else:
                    result += f"    v{gvn[instr]} = {instr.to_string(gvn)}\n"
            result += "  }\n"
        return result

    def rpo(self) -> list[Block]:
        result: list[Block] = []
        self.po_from(self.entry, result, set())
        result.reverse()
        return result

    def po_from(self, block: Block, result: list[Block], visited: set[Block]) -> None:
        visited.add(block)
        terminator = block.terminator()
        for succ in terminator.succs():
            if succ not in visited:
                self.po_from(succ, result, visited)
        result.append(block)

    def preds(self) -> dict[Block, set[Block]]:
        rpo = self.rpo()
        result: dict[Block, set[Block]] = {block: set() for block in rpo}
        for block in rpo:
            for succ in block.terminator().succs():
                result[succ].add(block)
        return result

    def doms(self) -> dict[Block, set[Block]]:
        preds = self.preds()
        entry = [block for block, block_preds in preds.items() if not block_preds][0]
        other_blocks = set(preds.keys()) - {entry}
        result = {entry: {entry}}
        for block in other_blocks:
            result[block] = set(preds.keys())
        change = True
        while change:
            change = False
            for block in other_blocks:
                tmp = {block} | set.intersection(*(result[pred] for pred in preds[block]))
                if tmp != result[block]:
                    result[block] = tmp
                    change = True
        return result


@dataclasses.dataclass(eq=False)
class IRFunction:
    id: int
    params: list[str]
    cfg: CFG = dataclasses.field(init=False, default_factory=CFG)

    def name(self) -> str:
        return f"fn{self.id}"

    def to_string(self, gvn: InstrId) -> str:
        result = f"{self.name()} {{\n"
        result += self.cfg.to_string(self, gvn)
        return result + "}"


class Compiler:
    def __init__(self) -> None:
        self.fns: list[IRFunction] = []
        entry = self.new_function([])
        self.gensym_counter: int = 0
        self.fn: IRFunction = entry
        self.block: Block = entry.cfg.entry

    def new_function(self, params: list[str]) -> IRFunction:
        result = IRFunction(len(self.fns), params)
        self.fns.append(result)
        return result

    def gensym(self, stem: str = "tmp") -> str:
        self.gensym_counter += 1
        return f"{stem}_{self.gensym_counter-1}"

    def push_fn(self, fn: IRFunction) -> tuple[IRFunction, Block]:
        prev_fn = self.fn
        prev_block = self.block
        self.restore_fn(fn, fn.cfg.entry)
        return prev_fn, prev_block

    def restore_fn(self, fn: IRFunction, block: Block) -> None:
        self.fn = fn
        self.block = block

    def emit(self, instr: Instr) -> Instr:
        self.block.append(instr)
        return instr

    def compile_match_pattern(self, env: Env, param: Instr, pattern: Object, success: Block, fallthrough: Block) -> Env:
        if isinstance(pattern, Int):
            cond = self.emit(IsNumEqualWord(param, pattern.value))
            self.emit(CondBranch(cond, success, fallthrough))
            return {}
        if isinstance(pattern, Var):
            self.emit(Jump(success))
            return {pattern.name: param}
        if isinstance(pattern, List):
            is_list = self.emit(IsList(param))
            is_list_block = self.fn.cfg.new_block()
            self.emit(CondBranch(is_list, is_list_block, fallthrough))
            self.block = is_list_block
            updates = {}
            the_list = param
            for i, pattern_item in enumerate(pattern.items):
                assert not isinstance(pattern_item, Spread)
                # Not enough elements
                is_empty = self.emit(IsEmptyList(the_list))
                is_nonempty_block = self.fn.cfg.new_block()
                self.emit(CondBranch(is_empty, fallthrough, is_nonempty_block))
                self.block = is_nonempty_block
                list_item = self.emit(ListFirst(the_list))
                pattern_success = self.fn.cfg.new_block()
                # Recursive pattern match
                updates.update(self.compile_match_pattern(env, list_item, pattern_item, pattern_success, fallthrough))
                self.block = pattern_success
                the_list = self.emit(ListRest(the_list))
            # Too many elements
            is_empty = self.emit(IsEmptyList(the_list))
            self.emit(CondBranch(is_empty, success, fallthrough))
            return updates
        raise NotImplementedError(f"pattern {type(pattern)} {pattern}")

    def compile_body(self, env: Env, exp: Object) -> None:
        self.emit(Return(self.compile(env, exp)))

    def compile_function(self, env: Env, exp: Function | MatchFunction, func_name: Optional[str]) -> Instr:
        if isinstance(exp, Function):
            assert isinstance(exp.arg, Var)
            param = exp.arg.name
        else:
            param = self.gensym("arg")
        clo = "$clo"
        fn = self.new_function([clo, param])
        freevars = free_in(exp)
        if func_name is not None and func_name in freevars:
            # Functions can refer to themselves; we close the loop below in the
            # funcenv
            freevars.remove(func_name)
        ordered_freevars = sorted(freevars)
        prev_fn, prev_block = self.push_fn(fn)
        #
        funcenv = {}
        for idx, name in enumerate(fn.params):
            funcenv[name] = self.emit(Param(idx, name))
        closure = funcenv[clo]
        if func_name is not None:
            funcenv[func_name] = closure
        for idx, name in enumerate(ordered_freevars):
            funcenv[name] = self.emit(ClosureRef(closure, idx, name))
        #
        if isinstance(exp, Function):
            self.compile_body(funcenv, exp.body)
        else:
            no_match = self.fn.cfg.new_block()
            no_match.append(MatchFail())
            case_blocks = [self.fn.cfg.new_block() for case in exp.cases]
            case_blocks.append(no_match)
            self.emit(Jump(case_blocks[0]))
            for i, case in enumerate(exp.cases):
                self.block = case_blocks[i]
                fallthrough = case_blocks[i + 1]
                body_block = self.fn.cfg.new_block()
                env_updates = self.compile_match_pattern(funcenv, funcenv[param], case.pattern, body_block, fallthrough)
                self.block = body_block
                self.compile_body({**funcenv, **env_updates}, case.body)
        self.restore_fn(prev_fn, prev_block)
        bound = [env[name] for name in ordered_freevars]
        result = self.emit(NewClosure(fn, bound))
        return result

    def compile(self, env: Env, exp: Object) -> Instr:
        if isinstance(exp, (Int, String)):
            return self.emit(Const(exp))
        if isinstance(exp, Var):
            return env[exp.name]
        if isinstance(exp, Binop):
            left = self.compile(env, exp.left)
            right = self.compile(env, exp.right)
            if exp.op == BinopKind.ADD:
                return self.emit(IntAdd(left, right))
            if exp.op == BinopKind.SUB:
                return self.emit(IntSub(left, right))
            if exp.op == BinopKind.MUL:
                return self.emit(IntMul(left, right))
            if exp.op == BinopKind.LESS:
                return self.emit(IntLess(left, right))
        if isinstance(exp, List):
            result = self.emit(Const(List([])))
            if not exp.items:
                return result
            for elt_exp in reversed(exp.items):
                elt = self.compile(env, elt_exp)
                result = self.emit(ListCons(elt, result))
            return result
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            name, value_exp, body_exp = exp.binding.name.name, exp.binding.value, exp.body
            if isinstance(value_exp, (Function, MatchFunction)):
                value = self.compile_function(env, value_exp, func_name=name)
            else:
                value = self.compile(env, value_exp)
            return self.compile({**env, name: value}, body_exp)
        if isinstance(exp, Apply):
            fn = self.compile(env, exp.func)
            arg = self.compile(env, exp.arg)
            return self.emit(Call(fn, arg))
        if isinstance(exp, (Function, MatchFunction)):
            # Anonymous function
            return self.compile_function(env, exp, func_name=None)
        raise NotImplementedError(f"exp {type(exp)} {exp}")


class IRTests(unittest.TestCase):
    def _parse(self, source: str) -> Object:
        return parse(tokenize(source))

    def test_int(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, Int(1))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    Return v0
  }
}""",
        )

    def test_str(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, String("hello"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<"hello">
    Return v0
  }
}""",
        )

    def test_add_int(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("1 + 2"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    v1 = Const<2>
    v2 = IntAdd v0, v1
    Return v2
  }
}""",
        )

    def test_sub_int(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("1 - 2"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    v1 = Const<2>
    v2 = IntSub v0, v1
    Return v2
  }
}""",
        )

    def test_less_int(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("1 < 2"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    v1 = Const<2>
    v2 = IntLess v0, v1
    Return v2
  }
}""",
        )

    def test_empty_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("[]"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<[]>
    Return v0
  }
}""",
        )

    def test_const_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("[1, 2]"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<[]>
    v1 = Const<2>
    v2 = ListCons v1, v0
    v3 = Const<1>
    v4 = ListCons v3, v2
    Return v4
  }
}""",
        )

    def test_non_const_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("a -> [a]"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; a>
    v2 = Const<[]>
    v3 = ListCons v1, v2
    Return v3
  }
}""",
        )

    def test_let(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("a . a = 1"))
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    Return v0
  }
}""",
        )

    def test_fun_id(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("a -> a"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    Return v0
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; a>
    Return v1
  }
}""",
        )

    def test_fun_closure(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("a -> b -> a + b"))
        self.assertEqual(len(compiler.fns), 3)
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    Return v0
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; a>
    v2 = NewClosure<fn2> v1
    Return v2
  }
}""",
        )
        self.assertEqual(
            compiler.fns[2].to_string(InstrId()),
            """\
fn2 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; b>
    v2 = ClosureRef<0; a> v0
    v3 = IntAdd v2, v1
    Return v3
  }
}""",
        )

    def test_fun_const_closure(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("(a -> a + b) . b = 1"))
        self.assertEqual(len(compiler.fns), 2)
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    v1 = NewClosure<fn1> v0
    Return v1
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; a>
    v2 = ClosureRef<0; b> v0
    v3 = IntAdd v1, v2
    Return v3
  }
}""",
        )

    def test_match_no_cases(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, MatchFunction([]))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    Return v0
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb1
  }
  bb1 {
    v2 = MatchFail
  }
}""",
        )

    def test_match_one_case(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| 1 -> 2 + 3"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsNumEqualWord v1, 1
    CondBranch v3, bb3, bb1
  }
  bb3 {
    v4 = Const<2>
    v5 = Const<3>
    v6 = IntAdd v4, v5
    Return v6
  }
}""",
        )

    def test_match_two_cases(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| 1 -> 2 | 3 -> 4"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsNumEqualWord v1, 1
    CondBranch v3, bb4, bb3
  }
  bb3 {
    v4 = IsNumEqualWord v1, 3
    CondBranch v4, bb5, bb1
  }
  bb4 {
    v5 = Const<2>
    Return v5
  }
  bb5 {
    v6 = Const<4>
    Return v6
  }
}""",
        )

    def test_match_var(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| a -> a + 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    Jump bb3
  }
  bb3 {
    v3 = Const<1>
    v4 = IntAdd v1, v3
    Return v4
  }
}""",
        )

    def test_match_empty_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| [] -> 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsList v1
    CondBranch v3, bb4, bb1
  }
  bb3 {
    v4 = Const<1>
    Return v4
  }
  bb4 {
    v5 = IsEmptyList v1
    CondBranch v5, bb3, bb1
  }
}""",
        )

    def test_match_one_item_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| [a] -> a + 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsList v1
    CondBranch v3, bb4, bb1
  }
  bb3 {
    v4 = Const<1>
    v5 = IntAdd v6, v4
    Return v5
  }
  bb4 {
    v7 = IsEmptyList v1
    CondBranch v7, bb1, bb5
  }
  bb5 {
    v6 = ListFirst v1
    Jump bb6
  }
  bb6 {
    v8 = ListRest v1
    v9 = IsEmptyList v8
    CondBranch v9, bb3, bb1
  }
}""",
        )

    def test_match_two_item_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("| [a, b] -> a + b"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsList v1
    CondBranch v3, bb4, bb1
  }
  bb3 {
    v4 = IntAdd v5, v6
    Return v4
  }
  bb4 {
    v7 = IsEmptyList v1
    CondBranch v7, bb1, bb5
  }
  bb5 {
    v5 = ListFirst v1
    Jump bb6
  }
  bb6 {
    v8 = ListRest v1
    v9 = IsEmptyList v8
    CondBranch v9, bb1, bb7
  }
  bb7 {
    v6 = ListFirst v8
    Jump bb8
  }
  bb8 {
    v10 = ListRest v8
    v11 = IsEmptyList v10
    CondBranch v11, bb3, bb1
  }
}""",
        )

    def test_apply_fn(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("f 1 . f  = x -> x + 1"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<1>
    v2 = Call v0, v1
    Return v2
  }
}""",
        )

    def test_recursive_call(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("fact 5 . fact = | 0 -> 1 | n -> n * fact (n - 1)"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<5>
    v2 = Call v0, v1
    Return v2
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb1 {
    v2 = MatchFail
  }
  bb2 {
    v3 = IsNumEqualWord v1, 0
    CondBranch v3, bb4, bb3
  }
  bb3 {
    Jump bb5
  }
  bb4 {
    v4 = Const<1>
    Return v4
  }
  bb5 {
    v5 = Const<1>
    v6 = IntSub v1, v5
    v7 = Call v0, v6
    v8 = IntMul v1, v7
    Return v8
  }
}""",
        )

    def test_apply_anonymous_function(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, self._parse("((x -> x + 1) 1)"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<1>
    v2 = Call v0, v1
    Return v2
  }
}""",
        )


class RPOTests(unittest.TestCase):
    def test_one_block(self) -> None:
        fn = IRFunction(0, [])
        entry = fn.cfg.entry
        one = entry.append(Const(Int(1)))
        entry.append(Return(one))
        self.assertEqual(fn.cfg.rpo(), [entry])

    def test_jump(self) -> None:
        fn = IRFunction(0, [])
        entry = fn.cfg.entry
        one = entry.append(Const(Int(1)))
        exit = fn.cfg.new_block()
        entry.append(Jump(exit))
        exit.append(Return(one))
        self.assertEqual(fn.cfg.rpo(), [entry, exit])

    def test_cond_branch(self) -> None:
        fn = IRFunction(0, [])
        entry = fn.cfg.entry
        one = entry.append(Const(Int(1)))
        left = fn.cfg.new_block()
        right = fn.cfg.new_block()
        entry.append(CondBranch(one, left, right))
        left.append(Return(one))
        right.append(Return(one))
        self.assertEqual(fn.cfg.rpo(), [entry, right, left])


class PredTests(unittest.TestCase):
    def test_preds(self) -> None:
        fn = IRFunction(0, [])
        entry = fn.cfg.entry
        one = entry.append(Const(Int(1)))
        bb1 = fn.cfg.new_block()
        entry.append(Jump(bb1))
        two = bb1.append(Const(Int(2)))
        bb2 = fn.cfg.new_block()
        bb3 = fn.cfg.new_block()
        bb1.append(CondBranch(two, bb2, bb3))
        bb4 = fn.cfg.new_block()
        bb2.append(Jump(bb4))
        bb3.append(Jump(bb4))
        three = bb4.append(Const(Int(3)))
        bb5 = fn.cfg.new_block()
        bb6 = fn.cfg.new_block()
        bb4.append(CondBranch(three, bb5, bb6))
        bb7 = fn.cfg.new_block()
        bb5.append(Jump(bb7))
        bb6.append(Jump(bb7))
        four = bb7.append(Const(Int(4)))
        exit = fn.cfg.new_block()
        bb7.append(CondBranch(four, exit, bb4))
        five = exit.append(Const(Int(5)))
        exit.append(Return(five))
        preds = fn.cfg.preds()
        self.assertEqual(
            preds,
            {
                entry: set(),
                bb1: {entry},
                bb2: {bb1},
                bb3: {bb1},
                bb4: {bb2, bb3, bb7},
                bb5: {bb4},
                bb6: {bb4},
                bb7: {bb5, bb6},
                exit: {bb7},
            },
        )


class DominatorTests(unittest.TestCase):
    def test_dom(self) -> None:
        fn = IRFunction(0, [])
        entry = fn.cfg.entry
        one = entry.append(Const(Int(1)))
        bb1 = fn.cfg.new_block()
        entry.append(Jump(bb1))
        two = bb1.append(Const(Int(2)))
        bb2 = fn.cfg.new_block()
        bb3 = fn.cfg.new_block()
        bb1.append(CondBranch(two, bb2, bb3))
        bb4 = fn.cfg.new_block()
        bb2.append(Jump(bb4))
        bb3.append(Jump(bb4))
        three = bb4.append(Const(Int(3)))
        bb5 = fn.cfg.new_block()
        bb6 = fn.cfg.new_block()
        bb4.append(CondBranch(three, bb5, bb6))
        bb7 = fn.cfg.new_block()
        bb5.append(Jump(bb7))
        bb6.append(Jump(bb7))
        four = bb7.append(Const(Int(4)))
        exit = fn.cfg.new_block()
        bb7.append(CondBranch(four, exit, bb4))
        five = exit.append(Const(Int(5)))
        exit.append(Return(five))
        doms = fn.cfg.doms()
        self.assertEqual(
            doms,
            {
                entry: {entry},
                bb1: {bb1, entry},
                bb2: {bb1, entry, bb2},
                bb3: {bb3, bb1, entry},
                bb4: {bb4, bb1, entry},
                bb5: {bb4, bb1, bb5, entry},
                bb6: {bb4, bb1, bb6, entry},
                bb7: {bb4, bb1, entry, bb7},
                exit: {bb4, bb1, entry, exit, bb7},
            },
        )


if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
