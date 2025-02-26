#!/usr/bin/env python3
from __future__ import annotations
import dataclasses
import io
import itertools
import json
import os
import typing
import unittest

from typing import Dict, Optional

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
        instr = instr.find()
        id = self.data.get(instr)
        if id is not None:
            return id
        id = len(self.data)
        self.data[instr] = id
        return id

    def name(self, instr: Instr) -> str:
        return f"v{self[instr]}"


@dataclasses.dataclass(eq=False)
class Instr:
    forwarded: Optional[Instr] = dataclasses.field(init=False, default=None)

    def find(self) -> Instr:
        result = self
        while True:
            it = result.forwarded
            if it is None:
                return result
            result = it

    def make_equal_to(self, other: Instr) -> None:
        self.find().forwarded = other

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def to_string(self, gvn: InstrId) -> str:
        return type(self).__name__


@dataclasses.dataclass(eq=False)
class Nop(Instr):
    pass


@dataclasses.dataclass(eq=False)
class Const(Instr):
    value: Object

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.value}>"


@dataclasses.dataclass(eq=False)
class CConst(Instr):
    type: str
    value: str

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.type}; {self.value}>"


@dataclasses.dataclass(eq=False)
class Param(Instr):
    idx: int
    name: str

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.idx}; {self.name}>"


@dataclasses.dataclass(eq=False)
class HasOperands(Instr):
    operands: list[Instr] = dataclasses.field(init=False, default_factory=list)

    def __init__(self, *operands: Instr) -> None:
        self.operands = list(operands)

    def to_string(self, gvn: InstrId) -> str:
        stem = f"{type(self).__name__}"
        if not self.operands:
            return stem
        return stem + " " + ", ".join(f"{gvn.name(op)}" for op in self.operands)


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


# TODO(max): Maybe start work on boxing/unboxing in the IR.
@dataclasses.dataclass(init=False, eq=False)
class CEqual(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class RefineType(HasOperands):
    def __init__(self, value: Instr, ty: ConstantLattice) -> None:
        self.operands = [value]
        self.ty = ty

    def to_string(self, gvn: InstrId) -> str:
        return f"{type(self).__name__}<{self.ty.__class__.__name__}> " + ", ".join(
            f"{gvn.name(op)}" for op in self.operands
        )


@dataclasses.dataclass(init=False, eq=False)
class IsIntEqualWord(HasOperands):
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
        return f"{type(self).__name__}<{self.idx}; {self.name}> {gvn.name(self.operands[0])}"


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
class ClosureCall(HasOperands):
    pass


@dataclasses.dataclass(eq=False)
class Control(Instr):
    def succs(self) -> tuple[Block, ...]:
        raise NotImplementedError("succs")


@dataclasses.dataclass(eq=False)
class MatchFail(Control):
    def succs(self) -> tuple[Block, ...]:
        return ()


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
        return f"{stem} " + ", ".join(f"{gvn.name(op)}" for op in self.operands)


@dataclasses.dataclass(eq=False)
class NewRecord(Instr):
    num_fields: int


@dataclasses.dataclass(init=False, eq=False)
class IsRecord(HasOperands):
    pass


@dataclasses.dataclass(eq=False)
class RecordSet(HasOperands):
    idx: int
    name: str

    def __init__(self, rec: Instr, idx: int, name: str, value: Instr) -> None:
        self.operands = [rec, value]
        self.idx = idx
        self.name = name

    def to_string(self, gvn: InstrId) -> str:
        stem = f"{type(self).__name__}<{self.idx}; {self.name}> "
        return stem + ", ".join(f"{gvn.name(op)}" for op in self.operands)


@dataclasses.dataclass(eq=False)
class RecordGet(HasOperands):
    name: str

    def __init__(self, rec: Instr, name: str) -> None:
        self.operands = [rec]
        self.name = name

    def to_string(self, gvn: InstrId) -> str:
        stem = f"{type(self).__name__}<{self.name}> "
        return stem + ", ".join(f"{gvn.name(op)}" for op in self.operands)


@dataclasses.dataclass(init=False, eq=False)
class RecordNumFields(HasOperands):
    pass


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
        assert isinstance(result, Control), f"Expected Control but found {result}"
        return result

    def succs(self) -> tuple[Block, ...]:
        if not self.instrs:
            return ()
        return self.terminator().succs()


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


@dataclasses.dataclass(init=False, eq=False)
class Guard(HasOperands):
    pass


@dataclasses.dataclass(init=False, eq=False)
class GuardNonNull(Guard):
    pass


@dataclasses.dataclass
class CFG:
    blocks: list[Block] = dataclasses.field(init=False, default_factory=list)
    entry: Block = dataclasses.field(init=False)
    next_block_id: int = 0

    def __init__(self) -> None:
        self.blocks = []
        self.next_block_id = 0
        self.entry = self.new_block()

    def new_block(self) -> Block:
        result = Block(self.next_block_id)
        self.next_block_id += 1
        self.blocks.append(result)
        return result

    def to_string(self, fn: IRFunction, gvn: InstrId) -> str:
        result = ""
        for block in self.rpo():
            result += f"  {block.name()} {{\n"
            for instr in block.instrs:
                instr = instr.find()
                if isinstance(instr, Nop):
                    continue
                if isinstance(instr, Control):
                    result += f"    {instr.to_string(gvn)}\n"
                else:
                    result += f"    {gvn.name(instr)} = {instr.to_string(gvn)}\n"
            result += "  }\n"
        return result

    def rpo(self) -> list[Block]:
        result: list[Block] = []
        self.po_from(self.entry, result, set())
        result.reverse()
        return result

    def po_from(self, block: Block, result: list[Block], visited: set[Block]) -> None:
        visited.add(block)
        for succ in block.succs():
            if succ not in visited:
                self.po_from(succ, result, visited)
        result.append(block)

    def preds(self) -> dict[Block, set[Block]]:
        rpo = self.rpo()
        result: dict[Block, set[Block]] = {block: set() for block in rpo}
        for block in rpo:
            for succ in block.succs():
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

    def to_c(self) -> str:
        with io.StringIO() as f:
            f.write(f"{self.c_decl()} {{\n")
            f.write("HANDLES();\n")
            for param in self.params:
                f.write(f"GC_PROTECT({param});\n")
            gvn = InstrId()
            for block in self.cfg.rpo():
                self._to_c(f, block, gvn)
            f.write("}")
            return f.getvalue()
        return

    def c_decl(self) -> str:
        params = ", ".join(f"struct object *{param}" for param in self.params)
        return f"struct object *fn{self.id}({params})\n"

    def _instr_to_c(self, instr: Instr, gvn: InstrId) -> str:
        def _handle(rhs: str) -> str:
            return f"OBJECT_HANDLE({gvn.name(instr)}, {rhs});\n"

        def _decl(ty: str, rhs: str) -> str:
            return f"{ty} {gvn.name(instr)} = {rhs};\n"

        def op(idx: int) -> str:
            assert isinstance(instr, HasOperands)
            return gvn.name(instr.operands[idx])

        if isinstance(instr, Const):
            value = instr.value
            if isinstance(value, Int):
                return _handle(f"mksmallint({value.value})")
            if isinstance(value, Hole):
                return _decl("struct object*", "hole()")
            if isinstance(value, String):
                string_repr = json.dumps(value.value)
                return _handle(f"mkstring(heap, {string_repr}, {len(value.value)})")
            if isinstance(value, List):
                if not value.items:
                    return _decl("struct object*", "empty_list()")
            raise NotImplementedError("const", type(value))
        if isinstance(instr, IntAdd):
            operands = ", ".join(gvn.name(op) for op in instr.operands)
            return _handle(f"num_add({operands})")
        if isinstance(instr, Param):
            return _handle(self.params[instr.idx])
        if isinstance(instr, NewClosure):
            result = _handle(f"mkclosure(heap, {instr.fn.name()}, {len(instr.operands)})")
            for idx, opnd in enumerate(instr.operands):
                result += f"closure_set({gvn.name(instr)}, {idx}, {gvn.name(opnd)});\n"
            return result
        if isinstance(instr, ClosureRef):
            return _handle(f"closure_get({op(0)}, {instr.idx})")
        if isinstance(instr, ClosureCall):
            return _handle(f"closure_call({op(0)}, {op(1)})")
        if isinstance(instr, IsIntEqualWord):
            return _decl("bool", f"{op(0)} == mksmallint({instr.expected})")
        if isinstance(instr, NewRecord):
            return _handle(f"mkrecord(heap, {instr.num_fields})")
        if isinstance(instr, RecordSet):
            return f"record_set({op(0)}, {instr.idx}, (struct record_field){{.key={instr.name}, .value={op(1)}}});\n"
        if isinstance(instr, RecordGet):
            return _handle(f"record_get({op(0)}, {instr.name})")
        if isinstance(instr, GuardNonNull):
            return f"if ({op(0)} == NULL) {{ abort(); }}\n" + _handle(op(0))
        if isinstance(instr, ListCons):
            return _handle(f"list_cons({op(0)}, {op(1)})")
        if isinstance(instr, ListFirst):
            return _handle(f"list_first({op(0)})")
        if isinstance(instr, ListRest):
            return _handle(f"list_rest({op(0)})")
        if isinstance(instr, IsList):
            return _decl("bool", f"is_list({op(0)})")
        if isinstance(instr, IsEmptyList):
            return _decl("bool", f"{op(0)} == empty_list()")
        if isinstance(instr, IsRecord):
            return _decl("bool", f"is_record({op(0)})")
        if isinstance(instr, RecordNumFields):
            return _decl("uword", f"record_num_fields({op(0)})")
        if isinstance(instr, CConst):
            return _decl(instr.type, instr.value)
        if isinstance(instr, CEqual):
            return _decl("bool", f"{op(0)} == {op(1)}")
        if isinstance(instr, Return):
            return f"return {op(0)};\n"
        if isinstance(instr, Jump):
            return f"goto {instr.target.name()};\n"
        if isinstance(instr, CondBranch):
            return f"if ({op(0)}) {{ goto {instr.conseq.name()}; }} else {{ goto {instr.alt.name()}; }}\n"
        if isinstance(instr, MatchFail):
            return "\n".join(
                [
                    """fprintf(stderr, "no matching cases\\n");""",
                    "abort();",
                    "return NULL;\n",  # Pacify the C compiler
                ]
            )
        raise NotImplementedError(type(instr))

    def _to_c(self, f: io.StringIO, block: Block, gvn: InstrId) -> None:
        f.write(f"{block.name()}:;\n")
        for instr in block.instrs:
            instr = instr.find()
            if isinstance(instr, Nop):
                continue
            f.write(self._instr_to_c(instr, gvn))


class Compiler:
    def __init__(self) -> None:
        self.fns: list[IRFunction] = []
        self.entry = entry = self.new_function([])
        self.gensym_counter: int = 0
        self.fn: IRFunction = entry
        self.block: Block = entry.cfg.entry
        self.record_keys: Dict[str, int] = {}

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
            cond = self.emit(IsIntEqualWord(param, pattern.value))
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
            # the_list = self.emit(RefineType(param, CList()))
            the_list = param
            for i, pattern_item in enumerate(pattern.items):
                if isinstance(pattern_item, Spread):
                    if pattern_item.name:
                        updates[pattern_item.name] = the_list
                    self.emit(Jump(success))
                    return updates
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
        if isinstance(pattern, Record):
            is_record = self.emit(IsRecord(param))
            updates = {}
            is_record_block = self.fn.cfg.new_block()
            self.emit(CondBranch(is_record, is_record_block, fallthrough))
            self.block = is_record_block
            for key, pattern_value in pattern.data.items():
                if isinstance(pattern_value, Spread):
                    if pattern_value.name:
                        raise NotImplementedError("named record spread not yet supported")
                    self.emit(Jump(success))
                    return updates
                key_idx = self.record_key(key)
                record_value = self.emit(RecordGet(param, key_idx))
                is_null = self.emit(CEqual(record_value, self.emit(CConst("struct object*", "NULL"))))
                recursive_block = self.fn.cfg.new_block()
                self.emit(CondBranch(is_null, fallthrough, recursive_block))
                self.block = recursive_block
                pattern_success = self.fn.cfg.new_block()
                # Recursive pattern match
                updates.update(
                    self.compile_match_pattern(env, record_value, pattern_value, pattern_success, fallthrough)
                )
                self.block = pattern_success
            # Too many fields
            num_fields = self.emit(RecordNumFields(param))
            cmp = self.emit(CEqual(num_fields, self.emit(CConst("uword", str(len(pattern.data))))))
            self.emit(CondBranch(cmp, success, fallthrough))
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
        if isinstance(exp, (Int, String, Hole)):
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
            # TODO(max): Separate out into ClosureFn and DirectCall and then we
            # can later replace the ClosureFn with known C function pointer in
            # an optimization pass
            return self.emit(ClosureCall(fn, arg))
        if isinstance(exp, (Function, MatchFunction)):
            # Anonymous function
            return self.compile_function(env, exp, func_name=None)
        if isinstance(exp, Record):
            num_fields = len(exp.data)
            result = self.emit(NewRecord(num_fields))
            for idx, (key, value_exp) in enumerate(exp.data.items()):
                value = self.compile(env, value_exp)
                self.emit(RecordSet(result, idx, self.record_key(key), value))
            return result
        if isinstance(exp, Access):
            assert isinstance(exp.at, Var), f"List access not supported"
            record = self.compile(env, exp.obj)
            key_idx = self.record_key(exp.at.name)
            # TODO(max): Guard that it's a Record
            value = self.emit(RecordGet(record, key_idx))
            return self.emit(GuardNonNull(value))
        raise NotImplementedError(f"exp {type(exp)} {exp}")

    def record_key(self, key: str) -> str:
        if key not in self.record_keys:
            self.record_keys[key] = len(self.record_keys)
        return f"Record_{key}"

    def to_c(self) -> str:
        with io.StringIO() as f:
            if self.record_keys:
                print("const char* record_keys[] = {", file=f)
                for key in self.record_keys:
                    print(f'"{key}",', file=f)
                print("};", file=f)
                print("enum {", file=f)
                for key, idx in self.record_keys.items():
                    print(f"Record_{key} = {idx},", file=f)
                print("};", file=f)
            else:
                # Pacify the C compiler
                print("const char* record_keys[] = { NULL };", file=f)
            for fn in self.fns:
                print(fn.to_c(), file=f)
            return f.getvalue()


@dataclasses.dataclass
class ConstantLattice:
    pass


@dataclasses.dataclass
class CBottom(ConstantLattice):
    pass


@dataclasses.dataclass
class CTop(ConstantLattice):
    pass


@dataclasses.dataclass
class CList(ConstantLattice):
    pass


@dataclasses.dataclass
class CInt(ConstantLattice):
    value: Optional[int] = None

    def has_value(self) -> bool:
        return self.value is not None


@dataclasses.dataclass
class CCInt(ConstantLattice):
    value: Optional[int] = None

    def has_value(self) -> bool:
        return self.value is not None


@dataclasses.dataclass
class CCBool(ConstantLattice):
    value: Optional[bool] = None


@dataclasses.dataclass
class CClo(ConstantLattice):
    value: Optional[IRFunction] = None


def union(self: ConstantLattice, other: ConstantLattice) -> ConstantLattice:
    if isinstance(self, CBottom):
        return other
    if isinstance(other, CBottom):
        return self
    if isinstance(self, CTop):
        return self
    if isinstance(self, CInt) and isinstance(other, CInt):
        return self if self.value == other.value else CInt()
    if isinstance(self, CCBool) and isinstance(other, CCBool):
        return self if self.value == other.value else CCBool()
    return CBottom()


@dataclasses.dataclass
class SCCP:
    fn: IRFunction
    instr_type: dict[Instr, ConstantLattice] = dataclasses.field(init=False, default_factory=dict)
    block_executable: set[Block] = dataclasses.field(init=False, default_factory=set)
    instr_uses: dict[Instr, set[Instr]] = dataclasses.field(init=False, default_factory=dict)

    def type_of(self, instr: Instr) -> ConstantLattice:
        result = self.instr_type.get(instr)
        if result is not None:
            return result
        result = self.instr_type[instr] = CBottom()
        return result

    def run(self) -> dict[Instr, ConstantLattice]:
        block_worklist: list[Block] = [self.fn.cfg.entry]
        instr_worklist: list[Instr] = []

        while block_worklist or instr_worklist:
            if instr_worklist and (instr := instr_worklist.pop(0)):
                instr = instr.find()
                if isinstance(instr, HasOperands):
                    for operand in instr.operands:
                        if operand not in self.instr_uses:
                            self.instr_uses[operand] = set()
                        self.instr_uses[operand].add(instr)
                new_type: ConstantLattice = CBottom()
                if isinstance(instr, Const):
                    value = instr.value
                    if isinstance(value, Int):
                        new_type = CInt(value.value)
                    if isinstance(value, List):
                        new_type = CList()
                elif isinstance(instr, Return):
                    pass
                elif isinstance(instr, MatchFail):
                    pass
                elif isinstance(instr, CondBranch):
                    match self.type_of(instr.operands[0]):
                        case CCBool(True):
                            instr.make_equal_to(Jump(instr.conseq))
                            block_worklist.append(instr.conseq)
                        case CCBool(False):
                            instr.make_equal_to(Jump(instr.alt))
                            block_worklist.append(instr.alt)
                        case CBottom():
                            pass
                        case _:
                            block_worklist.append(instr.conseq)
                            block_worklist.append(instr.alt)
                elif isinstance(instr, IntAdd):
                    match (self.type_of(instr.operands[0]), self.type_of(instr.operands[1])):
                        case (CInt(int(l)), CInt(int(r))):
                            new_type = CInt(l + r)
                            instr.make_equal_to(Const(Int(l+r)))
                        case (CInt(_), CInt(_)):
                            new_type = CInt()
                elif isinstance(instr, IntSub):
                    match (self.type_of(instr.operands[0]), self.type_of(instr.operands[1])):
                        case (CInt(int(l)), CInt(int(r))):
                            new_type = CInt(l - r)
                        case (CInt(_), CInt(_)):
                            new_type = CInt()
                elif isinstance(instr, ListCons):
                    if isinstance(self.type_of(instr.operands[1]), CList):
                        new_type = CList()
                elif isinstance(instr, NewClosure):
                    new_type = CClo(instr.fn)
                elif isinstance(instr, ClosureCall):
                    new_type = CTop()
                elif isinstance(instr, Param):
                    new_type = CTop()
                elif isinstance(instr, ClosureRef):
                    new_type = CTop()
                elif isinstance(instr, NewRecord):
                    new_type = CTop()
                elif isinstance(instr, RecordSet):
                    new_type = CTop()
                elif isinstance(instr, RecordGet):
                    new_type = CTop()
                elif isinstance(instr, GuardNonNull):
                    new_type = CTop()
                elif isinstance(instr, IsIntEqualWord):
                    match self.type_of(instr.operands[0]):
                        case CInt(int(i)) if i == instr.expected:
                            new_type = CCBool(True)
                        case _:
                            new_type = CCBool()
                elif isinstance(instr, IsList):
                    match self.type_of(instr.operands[0]):
                        case CList():
                            new_type = CCBool(True)
                        case _:
                            new_type = CCBool()
                elif isinstance(instr, IsEmptyList):
                    new_type = CCBool()
                elif isinstance(instr, ListFirst):
                    new_type = CTop()
                elif isinstance(instr, ListRest):
                    new_type = CTop()
                elif isinstance(instr, IsRecord):
                    new_type = CCBool()
                elif isinstance(instr, CConst):
                    new_type = CTop()
                elif isinstance(instr, CEqual):
                    new_type = CCBool()
                elif isinstance(instr, RecordNumFields):
                    new_type = CCInt()
                else:
                    raise NotImplementedError(f"SCCP {instr}")
                old_type = self.type_of(instr)
                if union(old_type, new_type) != old_type:
                    self.instr_type[instr] = new_type
                    for use in self.instr_uses.get(instr, set()):
                        instr_worklist.append(use)
            if block_worklist and (block := block_worklist.pop(0)):
                if block not in self.block_executable:
                    self.block_executable.add(block)
                    instr_worklist.extend(block.instrs)

        return self.instr_type


@dataclasses.dataclass
class CleanCFG:
    fn: IRFunction

    def run(self) -> None:
        changed = True
        while changed:
            changed = False
            for block in self.fn.cfg.rpo():
                if not block.instrs:
                    # Ignore transient empty blocks.
                    continue
                # Keep working on the current block until no further changes are made.
                while self.absorb_dst_block(block):
                    pass
            changed = self.remove_unreachable_blocks()

    def absorb_dst_block(self, block: Block) -> bool:
        terminator = block.terminator()
        if not isinstance(terminator, Jump):
            return False
        target = terminator.target
        if target == block:
            return False
        preds = self.fn.cfg.preds()
        if len(preds[target]) > 1:
            return False
        block.instrs.pop(-1)
        block.instrs.extend(target.instrs)
        target.instrs.clear()
        # No Phi to fix up
        return True

    def remove_unreachable_blocks(self) -> bool:
        num_blocks = len(self.fn.cfg.blocks)
        self.fn.cfg.blocks = self.fn.cfg.rpo()
        return len(self.fn.cfg.blocks) != num_blocks


@dataclasses.dataclass
class DeadCodeElimination:
    fn: IRFunction

    def is_critical(self, instr: Instr) -> bool:
        if isinstance(instr, Const):
            return False
        if isinstance(instr, IntAdd):
            return False
        # TODO(max): Add more. Track heap effects?
        return True

    def run(self) -> None:
        worklist: list[Instr] = []
        marked: set[Instr] = set()
        blocks = self.fn.cfg.rpo()
        # Mark
        for block in blocks:
            for instr in block.instrs:
                instr = instr.find()
                if self.is_critical(instr):
                    marked.add(instr)
                    worklist.append(instr)
        while worklist:
            instr = worklist.pop(0).find()
            if isinstance(instr, HasOperands):
                for op in instr.operands:
                    op = op.find()
                    if op not in marked:
                        marked.add(op)
                        worklist.append(op)
        # Sweep
        for block in blocks:
            for instr in block.instrs:
                instr = instr.find()
                if instr not in marked:
                    instr.make_equal_to(Nop())


def _parse(source: str) -> Object:
    return parse(tokenize(source))


class IRTests(unittest.TestCase):
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
        compiler.compile_body({}, _parse("1 + 2"))
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
        compiler.compile_body({}, _parse("1 - 2"))
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
        compiler.compile_body({}, _parse("1 < 2"))
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
        compiler.compile_body({}, _parse("[]"))
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
        compiler.compile_body({}, _parse("[1, 2]"))
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
        compiler.compile_body({}, _parse("a -> [a]"))
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
        compiler.compile_body({}, _parse("a . a = 1"))
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
        compiler.compile_body({}, _parse("a -> a"))
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
        compiler.compile_body({}, _parse("a -> b -> a + b"))
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
        compiler.compile_body({}, _parse("(a -> a + b) . b = 1"))
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
    MatchFail
  }
}""",
        )
        CleanCFG(compiler.fns[1]).run()
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    MatchFail
  }
}""",
        )

    def test_match_one_case(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| 1 -> 2 + 3"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsIntEqualWord v1, 1
    CondBranch v2, bb3, bb1
  }
  bb1 {
    MatchFail
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
        compiler.compile_body({}, _parse("| 1 -> 2 | 3 -> 4"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsIntEqualWord v1, 1
    CondBranch v2, bb4, bb3
  }
  bb3 {
    v3 = IsIntEqualWord v1, 3
    CondBranch v3, bb5, bb1
  }
  bb1 {
    MatchFail
  }
  bb5 {
    v4 = Const<4>
    Return v4
  }
  bb4 {
    v5 = Const<2>
    Return v5
  }
}""",
        )

    def test_match_var(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| a -> a + 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    Jump bb3
  }
  bb3 {
    v2 = Const<1>
    v3 = IntAdd v1, v2
    Return v3
  }
}""",
        )

    def test_match_empty_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| [] -> 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb3, bb1
  }
  bb1 {
    MatchFail
  }
  bb3 {
    v4 = Const<1>
    Return v4
  }
}""",
        )

    def test_match_one_item_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| [a] -> a + 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb1, bb5
  }
  bb5 {
    v4 = ListFirst v1
    Jump bb6
  }
  bb6 {
    v5 = ListRest v1
    v6 = IsEmptyList v5
    CondBranch v6, bb3, bb1
  }
  bb3 {
    v7 = Const<1>
    v8 = IntAdd v4, v7
    Return v8
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_match_two_item_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| [a, b] -> a + b"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb1, bb5
  }
  bb5 {
    v4 = ListFirst v1
    Jump bb6
  }
  bb6 {
    v5 = ListRest v1
    v6 = IsEmptyList v5
    CondBranch v6, bb1, bb7
  }
  bb7 {
    v7 = ListFirst v5
    Jump bb8
  }
  bb8 {
    v8 = ListRest v5
    v9 = IsEmptyList v8
    CondBranch v9, bb3, bb1
  }
  bb3 {
    v10 = IntAdd v4, v7
    Return v10
  }
  bb1 {
    MatchFail
  }
}""",
        )
        CleanCFG(compiler.fns[1]).run()
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb1, bb5
  }
  bb5 {
    v4 = ListFirst v1
    v5 = ListRest v1
    v6 = IsEmptyList v5
    CondBranch v6, bb1, bb7
  }
  bb7 {
    v7 = ListFirst v5
    v8 = ListRest v5
    v9 = IsEmptyList v8
    CondBranch v9, bb3, bb1
  }
  bb3 {
    v10 = IntAdd v4, v7
    Return v10
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_match_list_spread(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| [_, ...xs] -> xs"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb1, bb5
  }
  bb5 {
    v4 = ListFirst v1
    Jump bb6
  }
  bb6 {
    v5 = ListRest v1
    Jump bb3
  }
  bb3 {
    Return v5
  }
  bb1 {
    MatchFail
  }
}""",
        )
        CleanCFG(compiler.fns[1]).run()
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    v2 = IsList v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = IsEmptyList v1
    CondBranch v3, bb1, bb5
  }
  bb5 {
    v4 = ListFirst v1
    v5 = ListRest v1
    Return v5
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_match_empty_record(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| {} -> 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsRecord v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = RecordNumFields v1
    v4 = CConst<uword; 0>
    v5 = CEqual v3, v4
    CondBranch v5, bb3, bb1
  }
  bb1 {
    MatchFail
  }
  bb3 {
    v6 = Const<1>
    Return v6
  }
}""",
        )

    def test_match_one_item_record(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| {a=1} -> 1"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsRecord v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = RecordGet<Record_a> v1
    v4 = CConst<struct object*; NULL>
    v5 = CEqual v3, v4
    CondBranch v5, bb1, bb5
  }
  bb5 {
    v6 = IsIntEqualWord v3, 1
    CondBranch v6, bb6, bb1
  }
  bb6 {
    v7 = RecordNumFields v1
    v8 = CConst<uword; 1>
    v9 = CEqual v7, v8
    CondBranch v9, bb3, bb1
  }
  bb3 {
    v10 = Const<1>
    Return v10
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_match_two_item_record(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| {a=1, b=2} -> 3"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsRecord v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = RecordGet<Record_a> v1
    v4 = CConst<struct object*; NULL>
    v5 = CEqual v3, v4
    CondBranch v5, bb1, bb5
  }
  bb5 {
    v6 = IsIntEqualWord v3, 1
    CondBranch v6, bb6, bb1
  }
  bb6 {
    v7 = RecordGet<Record_b> v1
    v8 = CConst<struct object*; NULL>
    v9 = CEqual v7, v8
    CondBranch v9, bb1, bb7
  }
  bb7 {
    v10 = IsIntEqualWord v7, 2
    CondBranch v10, bb8, bb1
  }
  bb8 {
    v11 = RecordNumFields v1
    v12 = CConst<uword; 2>
    v13 = CEqual v11, v12
    CondBranch v13, bb3, bb1
  }
  bb3 {
    v14 = Const<3>
    Return v14
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_match_record_spread(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("| {a=a, ...} -> a"))
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    Jump bb2
  }
  bb2 {
    v2 = IsRecord v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = RecordGet<Record_a> v1
    v4 = CConst<struct object*; NULL>
    v5 = CEqual v3, v4
    CondBranch v5, bb1, bb5
  }
  bb5 {
    Jump bb6
  }
  bb6 {
    Jump bb3
  }
  bb3 {
    Return v3
  }
  bb1 {
    MatchFail
  }
}""",
        )
        CleanCFG(compiler.fns[1]).run()
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    v2 = IsRecord v1
    CondBranch v2, bb4, bb1
  }
  bb4 {
    v3 = RecordGet<Record_a> v1
    v4 = CConst<struct object*; NULL>
    v5 = CEqual v3, v4
    CondBranch v5, bb1, bb5
  }
  bb5 {
    Return v3
  }
  bb1 {
    MatchFail
  }
}""",
        )

    def test_apply_fn(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("f 1 . f  = x -> x + 1"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<1>
    v2 = ClosureCall v0, v1
    Return v2
  }
}""",
        )

    def test_recursive_call(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("fact 5 . fact = | 0 -> 1 | n -> n * fact (n - 1)"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<5>
    v2 = ClosureCall v0, v1
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
  bb2 {
    v2 = IsIntEqualWord v1, 0
    CondBranch v2, bb4, bb3
  }
  bb3 {
    Jump bb5
  }
  bb5 {
    v3 = Const<1>
    v4 = IntSub v1, v3
    v5 = ClosureCall v0, v4
    v6 = IntMul v1, v5
    Return v6
  }
  bb4 {
    v7 = Const<1>
    Return v7
  }
}""",
        )
        CleanCFG(compiler.fns[1]).run()
        self.assertEqual(
            compiler.fns[1].to_string(InstrId()),
            """\
fn1 {
  bb0 {
    v0 = Param<0; $clo>
    v1 = Param<1; arg_0>
    v2 = IsIntEqualWord v1, 0
    CondBranch v2, bb4, bb3
  }
  bb3 {
    v3 = Const<1>
    v4 = IntSub v1, v3
    v5 = ClosureCall v0, v4
    v6 = IntMul v1, v5
    Return v6
  }
  bb4 {
    v7 = Const<1>
    Return v7
  }
}""",
        )

    def test_apply_anonymous_function(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("((x -> x + 1) 1)"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewClosure<fn1>
    v1 = Const<1>
    v2 = ClosureCall v0, v1
    Return v2
  }
}""",
        )
        analysis = SCCP(compiler.fns[0])
        analysis.run()
        entry = compiler.fns[0].cfg.entry
        self.assertEqual(analysis.instr_type[entry.instrs[0]], CClo(compiler.fns[1]))

    def test_empty_record(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("{}"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewRecord
    Return v0
  }
}""",
        )

    def test_record_with_one_field(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("{a=1}"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewRecord
    v1 = Const<1>
    v2 = RecordSet<0; Record_a> v0, v1
    Return v0
  }
}""",
        )

    def test_record_with_two_fields(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("{a=1, b=2}"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewRecord
    v1 = Const<1>
    v2 = RecordSet<0; Record_a> v0, v1
    v3 = Const<2>
    v4 = RecordSet<1; Record_b> v0, v3
    Return v0
  }
}""",
        )

    def test_record_access(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("{a=1, b=2}@a"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = NewRecord
    v1 = Const<1>
    v2 = RecordSet<0; Record_a> v0, v1
    v3 = Const<2>
    v4 = RecordSet<1; Record_b> v0, v3
    v5 = RecordGet<Record_a> v0
    v6 = GuardNonNull v5
    Return v6
  }
}""",
        )

    def test_hole(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("()"))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<()>
    Return v0
  }
}""",
        )

    def test_string(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse('"hello"'))
        self.assertEqual(
            compiler.fns[0].to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<"hello">
    Return v0
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


class SCCPTests(unittest.TestCase):
    def test_int(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, Int(1))
        analysis = SCCP(compiler.fn)
        result = analysis.run()
        entry = compiler.fn.cfg.entry
        self.assertEqual(result, {entry.instrs[0]: CInt(1), entry.instrs[1]: CBottom()})

    def test_int_add(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("1 + 2 + 3"))
        analysis = SCCP(compiler.fn)
        result = analysis.run()
        entry = compiler.fn.cfg.entry
        self.assertEqual(
            result,
            {
                entry.instrs[0]: CInt(1),
                entry.instrs[1]: CInt(2),
                entry.instrs[2]: CInt(3),
                entry.instrs[3]: CInt(5),
                entry.instrs[4]: CInt(6),
                entry.instrs[5]: CBottom(),
            },
        )

        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<1>
    v1 = Const<2>
    v2 = Const<3>
    v3 = Const<5>
    v4 = Const<6>
    Return v4
  }
}""",
        )

    def test_empty_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("[]"))
        analysis = SCCP(compiler.fn)
        analysis.run()
        return_instr = compiler.fn.cfg.entry.instrs[-1]
        self.assertIsInstance(return_instr, Return)
        assert isinstance(return_instr, Return)
        returned = return_instr.operands[0]
        self.assertEqual(analysis.instr_type[returned], CList())

    def test_const_list(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("[1, 2]"))
        analysis = SCCP(compiler.fn)
        analysis.run()
        return_instr = compiler.fn.cfg.entry.instrs[-1]
        self.assertIsInstance(return_instr, Return)
        assert isinstance(return_instr, Return)
        returned = return_instr.operands[0]
        self.assertEqual(analysis.instr_type[returned], CList())


class DeadCodeEliminationTests(unittest.TestCase):
    def test_remove_const(self) -> None:
        compiler = Compiler()
        compiler.emit(Const(1))
        compiler.emit(Const(2))
        compiler.emit(Const(3))
        four = compiler.emit(Const(4))
        compiler.emit(Return(four))
        DeadCodeElimination(compiler.fn).run()
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<4>
    Return v0
  }
}""",
        )

    def test_remove_int_add(self) -> None:
        compiler = Compiler()
        one = compiler.emit(Const(1))
        two = compiler.emit(Const(2))
        compiler.emit(IntAdd(one, two))
        four = compiler.emit(Const(4))
        compiler.emit(Return(four))
        DeadCodeElimination(compiler.fn).run()
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<4>
    Return v0
  }
}""",
        )


def opt(fn: IRFunction) -> None:
    CleanCFG(fn).run()
    instr_type = SCCP(fn).run()
    for block in fn.cfg.rpo():
        for instr in block.instrs:
            match instr_type[instr]:
                case CInt(int(i)):
                    instr.make_equal_to(Const(Int(i)))
    DeadCodeElimination(fn).run()


class OptTests(unittest.TestCase):
    def test_int_add(self) -> None:
        compiler = Compiler()
        compiler.compile_body({}, _parse("1 + 2 + 3"))
        opt(compiler.fn)
        self.assertEqual(
            compiler.fn.to_string(InstrId()),
            """\
fn0 {
  bb0 {
    v0 = Const<6>
    Return v0
  }
}""",
        )


def compile_to_c(source: str) -> str:
    import subprocess
    import tempfile

    program = parse(tokenize(source))
    compiler = Compiler()
    compiler.compile_body({}, program)
    for fn in compiler.fns:
        opt(fn)
    c_code = compiler.to_c()
    dirname = os.path.dirname(__file__)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as c_file:
        constants = [
            ("uword", "kKiB", 1024),
            ("uword", "kMiB", "kKiB * kKiB"),
            ("uword", "kGiB", "kKiB * kKiB * kKiB"),
            ("uword", "kPageSize", "4 * kKiB"),
            ("uword", "kSmallIntTagBits", 1),
            ("uword", "kPrimaryTagBits", 3),
            ("uword", "kObjectAlignmentLog2", 3),  # bits
            ("uword", "kObjectAlignment", "1ULL << kObjectAlignmentLog2"),
            ("uword", "kImmediateTagBits", 5),
            ("uword", "kSmallIntTagMask", "(1ULL << kSmallIntTagBits) - 1"),
            ("uword", "kPrimaryTagMask", "(1ULL << kPrimaryTagBits) - 1"),
            ("uword", "kImmediateTagMask", "(1ULL << kImmediateTagBits) - 1"),
            ("uword", "kWordSize", "sizeof(word)"),
            ("uword", "kMaxSmallStringLength", "kWordSize - 1"),
            ("uword", "kBitsPerByte", 8),
            # Up to the five least significant bits are used to tag the object's layout.
            # The three low bits make up a primary tag, used to differentiate gc_obj
            # from immediate objects. All even tags map to SmallInt, which is
            # optimized by checking only the lowest bit for parity.
            ("uword", "kSmallIntTag", 0),  # 0b****0
            ("uword", "kHeapObjectTag", 1),  # 0b**001
            ("uword", "kEmptyListTag", 5),  # 0b00101
            ("uword", "kHoleTag", 7),  # 0b00111
            ("uword", "kSmallStringTag", 13),  # 0b01101
            ("uword", "kVariantTag", 15),  # 0b01111
            # TODO(max): Fill in 21
            # TODO(max): Fill in 23
            # TODO(max): Fill in 29
            # TODO(max): Fill in 31
            ("uword", "kBitsPerPointer", "kBitsPerByte * kWordSize"),
            ("word", "kSmallIntBits", "kBitsPerPointer - kSmallIntTagBits"),
            ("word", "kSmallIntMinValue", "-(((word)1) << (kSmallIntBits - 1))"),
            ("word", "kSmallIntMaxValue", "(((word)1) << (kSmallIntBits - 1)) - 1"),
        ]
        for type_, name, value in constants:
            print(f"#define {name} ({type_})({value})", file=c_file)
        # The runtime is in the same directory as this file
        with open(os.path.join(dirname, "runtime.c"), "r") as runtime:
            c_file.write(runtime.read())
        c_file.write("\n")
        for fn in compiler.fns:
            c_file.write(fn.c_decl() + ";\n")
        c_file.write(c_code)
        c_file.write("\n")
        # The platform is in the same directory as this file
        print(
            f"""

const char* variant_names[] = {{
  "UNDEF",
}};
int main() {{
  struct space space = make_space(MEMORY_SIZE);
  init_heap(heap, space);
  HANDLES();
  GC_HANDLE(struct object*, result, {compiler.entry.name()}());
  println(result);
  destroy_space(space);
  return 0;
}}
""",
            file=c_file,
        )
    return c_file.name


def compile_to_binary(c_name: str) -> str:
    import subprocess
    import tempfile

    cc = os.environ.get("CC", "tcc")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as out_file:
        subprocess.run([cc, "-o", out_file.name, c_name], check=True)
    return out_file.name


def _run(code: str) -> str:
    import subprocess
    import tempfile

    c_name = compile_to_c(code)
    binary_name = compile_to_binary(c_name)
    result = subprocess.run([binary_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return result.stdout


class CompilerEndToEndTests(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(_run("1"), "1\n")

    def test_int_add(self) -> None:
        self.assertEqual(_run("1 + 2"), "3\n")

    def test_int_sub(self) -> None:
        self.assertEqual(_run("1 - 2"), "-1\n")

    def test_fun_id(self) -> None:
        self.assertEqual(_run("a -> a"), "<closure>\n")

    def test_closed_vars(self) -> None:
        self.assertEqual(_run("((a -> a + b) 3) . b = 4"), "7\n")

    def test_call_fun_id(self) -> None:
        self.assertEqual(_run("(a -> a) 3"), "3\n")

    def test_match_int(self) -> None:
        self.assertEqual(_run("| 1 -> 2"), "<closure>\n")

    def test_call_match_int(self) -> None:
        self.assertEqual(_run("(| 1 -> 2) 1"), "2\n")
        self.assertEqual(_run("(| 1 -> 2 | 3 -> 4) 3"), "4\n")

    def test_match_list(self) -> None:
        self.assertEqual(_run("f [1, 2] . f = | [1, 2] -> 3 | [4, 5] -> 6"), "3\n")
        self.assertEqual(_run("f [4, 5] . f = | [1, 2] -> 3 | [4, 5] -> 6"), "6\n")

    def test_match_list_spread(self) -> None:
        self.assertEqual(_run("f [4, 5] . f = | [_, ...xs] -> xs"), "[5]\n")

    def test_var(self) -> None:
        self.assertEqual(_run("a . a = 1"), "1\n")

    def test_function(self) -> None:
        self.assertEqual(_run("f 1 . f = x -> x + 1"), "2\n")

    def test_match_int_fallthrough(self) -> None:
        self.assertEqual(_run("f 3 . f = | 1 -> 2 | 3 -> 4"), "4\n")

    def test_empty_record(self) -> None:
        self.assertEqual(_run("{}"), "{}\n")

    def test_record_with_one_field(self) -> None:
        self.assertEqual(_run("{a=1}"), "{a = 1}\n")

    def test_record_with_two_fields(self) -> None:
        self.assertEqual(_run("{a=1, b=2}"), "{a = 1, b = 2}\n")

    def test_record_builder(self) -> None:
        self.assertEqual(_run("f 1 2 . f = x -> y -> {a = x, b = y}"), "{a = 1, b = 2}\n")

    def test_record_access(self) -> None:
        self.assertEqual(_run("rec@a . rec = {a = 1, b = 2}"), "1\n")

    def test_record_builder_access(self) -> None:
        self.assertEqual(_run("(f 1 2)@a . f = x -> y -> {a = x, b = y}"), "1\n")

    def test_match_record(self) -> None:
        self.assertEqual(_run("f {a = 4, b = 5} . f = | {a = 1, b = 2} -> 3 | {a = 4, b = 5} -> 6"), "6\n")

    def test_match_record_too_few_keys(self) -> None:
        self.assertEqual(_run("f {a = 4, b = 5} . f = | {a = _} -> 3 | {a = _, b = _} -> 6"), "6\n")

    def test_match_record_spread(self) -> None:
        self.assertEqual(_run("f {a=1, b=2, c=3} . f = | {a=a, ...} -> a"), "1\n")

    def test_hole(self) -> None:
        self.assertEqual(_run("()"), "()\n")

    def test_string(self) -> None:
        self.assertEqual(_run('"hello"'), '"hello"\n')


if __name__ == "__main__":
    __import__("sys").modules["unittest.util"]._MAX_LENGTH = 999999999
    unittest.main()
