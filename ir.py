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
class Instr:
    pass

@dataclasses.dataclass
class Const(Instr):
    value: Object

@dataclasses.dataclass
class Param(Instr):
    idx: int
    name: str

@dataclasses.dataclass
class MatchFail(Instr):
    pass

@dataclasses.dataclass
class HasOperands(Instr):
    operands: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def __init__(self, *operands: Instr) -> None:
        self.operands = list(operands)

@dataclasses.dataclass(init=False)
class IntAdd(HasOperands):
    pass

@dataclasses.dataclass(init=False)
class IntLess(HasOperands):
    pass

@dataclasses.dataclass(init=False)
class IsNumEqualWord(HasOperands):
    expected: int

    def __init__(self, value: Instr, expected: int) -> None:
        self.operands = [value]
        self.expected = expected

@dataclasses.dataclass
class Control(Instr):
    pass

Env = Dict[str, Instr]

@dataclasses.dataclass
class Block:
    id: int
    instrs: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def append(self, instr: Instr) -> None:
        self.instrs.append(instr)

@dataclasses.dataclass
class Jump(Control):
    target: Block

@dataclasses.dataclass(init=False)
class Return(HasOperands, Control):
    pass

@dataclasses.dataclass(init=False)
class CondBranch(Control, HasOperands):
    conseq: Block
    alt: Block

    def __init__(self, cond: Instr, conseq: Block, alt: Block) -> None:
        self.conseq = conseq
        self.alt = alt
        self.operands = [cond]

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

@dataclasses.dataclass
class Function(Instr):
    params: list[str]
    cfg: CFG = dataclasses.field(init=False, default_factory=CFG)

    # def initial_env(self) -> Env:
    #     result = {}
    #     for idx, name in enumerate(self.params):
    #         instr = Param(idx, name)
    #         result[name] = self.cfg.emit(Param(idx, name))
    #     return result

class Compiler:
    def __init__(self, entry: Function) -> None:
        self.gensym_counter: int = 0
        self.fn: Function = entry
        self.block: Block = entry.cfg.entry
        self.fns: list[Function] = [entry]

    def gensym(self, stem: str = "tmp") -> str:
        self.gensym_counter += 1
        return f"{stem}_{self.gensym_counter-1}"

    def push_fn(self, fn: Function) -> Function:
        self.fns.append(fn)
        prev_fn = self.fn
        self.restore_fn(fn)
        return prev_fn

    def restore_fn(self, fn: Function) -> None:
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
        raise NotImplementedError(f"pattern {type(pattern)} {pattern}")

    def compile(self, env: Env, exp: Object) -> Instr:
        if isinstance(exp, Int):
            return self.emit(Const(exp))
        if isinstance(exp, Binop):
            left = self.compile(env, exp.left)
            right = self.compile(env, exp.right)
            if exp.op == BinopKind.ADD:
                return IntAdd(left, right)
            if exp.op == BinopKind.LESS:
                return IntLess(left, right)
        if isinstance(exp, MatchFunction):
            param = self.gensym("arg")
            fn = Function([param])
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
                fallthrough = case_blocks[i+1]
                body_block = self.fn.cfg.new_block()
                env_updates = self.compile_match_pattern(funcenv, funcenv[param], case.pattern, body_block, fallthrough)
                self.block = body_block
                case_result = self.compile({**funcenv, **env_updates}, case.body)
                self.emit(Return(case_result))
            #
            self.restore_fn(prev_fn)
            return fn
        raise NotImplementedError(f"exp {type(exp)} {exp}")


class IRTests(unittest.TestCase):
    def _parse(self, source: str) -> Object:
        return parse(tokenize(source))

    def test_int(self) -> None:
        compiler = Compiler(Function([]))
        result = compiler.compile({}, Int(1))
        self.assertEqual(result, Const(Int(1)))

    def test_add_int(self) -> None:
        compiler = Compiler(Function([]))
        result = compiler.compile({}, self._parse("1 + 2"))
        self.assertEqual(result, IntAdd(Const(Int(1)), Const(Int(2))))

    def test_less_int(self) -> None:
        compiler = Compiler(Function([]))
        result = compiler.compile({}, self._parse("1 < 2"))
        self.assertEqual(result, IntLess(Const(Int(1)), Const(Int(2))))

    # def test_match_no_cases(self) -> None:
    #     compiler = Compiler()
    #     result = compiler.compile({}, MatchFunction([]))
    #     self.assertEqual(result, IntLess(Const(Int(1)), Const(Int(2))))

    def test_match_one_case(self) -> None:
        compiler = Compiler(Function([]))
        result = compiler.compile({}, self._parse("| 1 -> 2"))
        self.assertIsInstance(result, Function)
        self.assertEqual(result.cfg.entry.instrs, [
            Param(0, "arg_0")
        ])

if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
