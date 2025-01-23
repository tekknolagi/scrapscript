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
class Control(Instr):
    pass


@dataclasses.dataclass(eq=False)
class NewClosure(Instr):
    fn: IRFunction

    def to_string(self, gvn: InstrId) -> str:
        return super().to_string(gvn) + f" {self.fn.name()}"


Env = Dict[str, Instr]


@dataclasses.dataclass(eq=False)
class Block:
    id: int
    instrs: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def append(self, instr: Instr) -> None:
        self.instrs.append(instr)

    def name(self) -> str:
        return f"bb{self.id}"


@dataclasses.dataclass(eq=False)
class Jump(Control):
    target: Block

    def to_string(self, gvn: InstrId) -> str:
        return super().to_string(gvn) + f" {self.target.name()}"


@dataclasses.dataclass(init=False, eq=False)
class Return(HasOperands, Control):
    pass


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

    def push_fn(self, fn: IRFunction) -> IRFunction:
        self.fns.append(fn)
        prev_fn = self.fn
        self.restore_fn(fn)
        return prev_fn

    def restore_fn(self, fn: IRFunction) -> None:
        self.fn = fn
        self.block = fn.cfg.entry

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
        raise NotImplementedError(f"pattern {type(pattern)} {pattern}")

    def compile_body(self, env: Env, exp: Object) -> None:
        self.emit(Return(self.compile(env, exp)))

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
            if exp.op == BinopKind.LESS:
                return self.emit(IntLess(left, right))
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            name, value_exp, body_exp = exp.binding.name.name, exp.binding.value, exp.body
            value = self.compile(env, value_exp)
            return self.compile({**env, name: value}, body_exp)
        if isinstance(exp, MatchFunction):
            param = self.gensym("arg")
            fn = self.new_function([param])
            prev_fn = self.push_fn(fn)
            self.block = fn.cfg.entry
            #
            funcenv = {}
            for idx, name in enumerate(fn.params):
                funcenv[name] = self.emit(Param(idx, name))
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
            #
            self.restore_fn(prev_fn)
            return self.emit(NewClosure(fn))
        if isinstance(exp, Function):
            assert isinstance(exp.arg, Var)
            param = exp.arg.name
            fn = self.new_function([param])
            prev_fn = self.push_fn(fn)
            self.block = fn.cfg.entry
            #
            funcenv = {}
            for idx, name in enumerate(fn.params):
                funcenv[name] = self.emit(Param(idx, name))
            self.compile_body(funcenv, exp.body)
            #
            self.restore_fn(prev_fn)
            return self.emit(NewClosure(fn))
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
    v0 = NewClosure fn1
    Return v0
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; a>
    Return v0
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
    v0 = NewClosure fn1
    Return v0
  }
}""",
        )
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; arg_0>
    Jump bb1
  }
  bb1 {
    v1 = MatchFail
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
    v0 = Param<0; arg_0>
    Jump bb2
  }
  bb1 {
    v1 = MatchFail
  }
  bb2 {
    v2 = IsNumEqualWord v0, 1
    CondBranch v2, bb3, bb1
  }
  bb3 {
    v3 = Const<2>
    v4 = Const<3>
    v5 = IntAdd v3, v4
    Return v5
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
    v0 = Param<0; arg_0>
    Jump bb2
  }
  bb1 {
    v1 = MatchFail
  }
  bb2 {
    v2 = IsNumEqualWord v0, 1
    CondBranch v2, bb4, bb3
  }
  bb3 {
    v3 = IsNumEqualWord v0, 3
    CondBranch v3, bb5, bb1
  }
  bb4 {
    v4 = Const<2>
    Return v4
  }
  bb5 {
    v5 = Const<4>
    Return v5
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
    v0 = Param<0; arg_0>
    Jump bb2
  }
  bb1 {
    v1 = MatchFail
  }
  bb2 {
    Jump bb3
  }
  bb3 {
    v2 = Const<1>
    v3 = IntAdd v0, v2
    Return v3
  }
}""",
        )


if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
