#!/usr/bin/env python3.10
from __future__ import annotations
import argparse
import base64
import code
import copy
import dataclasses
import enum
import functools
import json
import logging
import os
import struct
import sys
import typing
import urllib.request
from dataclasses import dataclass
from enum import auto
from types import ModuleType
from typing import Any, Callable, Dict, Generator, Iterator, Mapping, Optional, Set, Tuple, Union

readline: Optional[ModuleType]
try:
    import readline
except ImportError:
    readline = None


logger = logging.getLogger(__name__)


def is_identifier_char(c: str) -> bool:
    return c.isalnum() or c in ("$", "'", "_")


@dataclass(eq=True, order=True, unsafe_hash=True)
class SourceLocation:
    lineno: int = dataclasses.field(default=-1)
    colno: int = dataclasses.field(default=-1)
    byteno: int = dataclasses.field(default=-1)


@dataclass(eq=True, unsafe_hash=True)
class SourceExtent:
    start: SourceLocation = dataclasses.field(default_factory=SourceLocation)
    end: SourceLocation = dataclasses.field(default_factory=SourceLocation)

    def coalesce(self, other: Optional[SourceExtent]) -> Optional[SourceExtent]:
        return SourceExtent(min(self.start, other.start), max(self.end, other.end)) if other else None


def join_source_extents(
    source_extent_one: Optional[SourceExtent], source_extent_two: Optional[SourceExtent]
) -> Optional[SourceExtent]:
    return source_extent_one.coalesce(source_extent_two) if source_extent_one else None


@dataclass(eq=True)
class Token:
    source_extent: SourceExtent = dataclasses.field(default_factory=SourceExtent, init=False, compare=False)

    def with_source(self, source_extent: SourceExtent) -> Token:
        self.source_extent = source_extent
        return self


@dataclass(eq=True)
class IntLit(Token):
    value: int


@dataclass(eq=True)
class FloatLit(Token):
    value: float


@dataclass(eq=True)
class StringLit(Token):
    value: str


@dataclass(eq=True)
class BytesLit(Token):
    value: str
    base: int


@dataclass(eq=True)
class Operator(Token):
    value: str


@dataclass(eq=True)
class Name(Token):
    value: str


@dataclass(eq=True)
class LeftParen(Token):
    # (
    pass


@dataclass(eq=True)
class RightParen(Token):
    # )
    pass


@dataclass(eq=True)
class LeftBrace(Token):
    # {
    pass


@dataclass(eq=True)
class RightBrace(Token):
    # }
    pass


@dataclass(eq=True)
class LeftBracket(Token):
    # [
    pass


@dataclass(eq=True)
class RightBracket(Token):
    # ]
    pass


@dataclass(eq=True)
class Hash(Token):
    # #
    pass


@dataclass(eq=True)
class EOF(Token):
    pass


def num_bytes_as_utf8(s: str) -> int:
    return len(s.encode(encoding="UTF-8"))


class Lexer:
    def __init__(self, text: str):
        self.text: str = text
        self.idx: int = 0
        self._lineno: int = 1
        self._colno: int = 1
        self.line: str = ""
        self._byteno: int = 0
        self.current_token_source_extent: SourceExtent = SourceExtent(
            start=SourceLocation(
                lineno=self._lineno,
                colno=self._colno,
                byteno=self._byteno,
            ),
            end=SourceLocation(
                lineno=self._lineno,
                colno=self._colno,
                byteno=self._byteno,
            ),
        )
        self.token_start_idx: int = self.idx
        self.token_end_idx: int = self.token_start_idx

    @property
    def lineno(self) -> int:
        return self._lineno

    @property
    def colno(self) -> int:
        return self._colno

    @property
    def byteno(self) -> int:
        return self._byteno

    def mark_token_start(self) -> None:
        self.current_token_source_extent.start.lineno = self._lineno
        self.current_token_source_extent.start.colno = self._colno
        self.current_token_source_extent.start.byteno = self._byteno
        self.token_start_idx = self.idx

    def mark_token_end(self) -> None:
        self.current_token_source_extent.end.lineno = self._lineno
        self.current_token_source_extent.end.colno = self._colno
        self.current_token_source_extent.end.byteno = self._byteno
        self.token_end_idx = self.idx

    def has_input(self) -> bool:
        return self.idx < len(self.text)

    def read_char(self) -> str:
        self.mark_token_end()
        c = self.peek_char()
        if c == "\n":
            self._lineno += 1
            self._colno = 1
            self.line = ""
        else:
            self.line += c
            self._colno += 1
        self.idx += 1
        self._byteno += num_bytes_as_utf8(c)
        return c

    def peek_char(self) -> str:
        if not self.has_input():
            raise UnexpectedEOFError("while reading token")
        return self.text[self.idx]

    def make_token(self, cls: type, *args: Any) -> Token:
        result: Token = cls(*args)
        return result.with_source(copy.deepcopy(self.current_token_source_extent))

    def read_tokens(self) -> Generator[Token, None, None]:
        while (token := self.read_token()) and not isinstance(token, EOF):
            yield token

    def read_token(self) -> Token:
        # Consume all whitespace
        while self.has_input():
            # Keep updating the token start location until we exhaust all whitespace
            self.mark_token_start()
            c = self.read_char()
            if not c.isspace():
                break
        else:
            return self.make_token(EOF)
        if c == '"':
            return self.read_string()
        if c == "-":
            if self.has_input() and self.peek_char() == "-":
                self.read_comment()
                # Need to start reading a new token
                return self.read_token()
            return self.read_op(c)
        if c == "#":
            return self.make_token(Hash)
        if c == "~":
            if self.has_input() and self.peek_char() == "~":
                self.read_char()
                return self.read_bytes()
            raise ParseError(f"unexpected token {c!r}")
        if c.isdigit():
            return self.read_number(c)
        if c in "()[]{}":
            custom = {
                "(": LeftParen,
                ")": RightParen,
                "{": LeftBrace,
                "}": RightBrace,
                "[": LeftBracket,
                "]": RightBracket,
            }
            return self.make_token(custom[c])
        if c in OPER_CHARS:
            return self.read_op(c)
        if is_identifier_char(c):
            return self.read_var(c)
        raise InvalidTokenError(
            SourceExtent(
                start=SourceLocation(
                    lineno=self.current_token_source_extent.start.lineno,
                    colno=self.current_token_source_extent.start.colno,
                    byteno=self.current_token_source_extent.start.byteno,
                ),
                end=SourceLocation(
                    lineno=self.current_token_source_extent.end.lineno,
                    colno=self.current_token_source_extent.end.colno,
                    byteno=self.current_token_source_extent.end.byteno,
                ),
            )
        )

    def read_string(self) -> Token:
        buf = ""
        while self.has_input():
            if (c := self.read_char()) == '"':
                break
            buf += c
        else:
            raise UnexpectedEOFError("while reading string")
        return self.make_token(StringLit, buf)

    def read_comment(self) -> None:
        while self.has_input() and self.read_char() != "\n":
            pass

    def read_number(self, first_digit: str) -> Token:
        # TODO: Support floating point numbers with no integer part
        buf = first_digit
        has_decimal = False
        while self.has_input():
            c = self.peek_char()
            if c == ".":
                if has_decimal:
                    raise ParseError(f"unexpected token {c!r}")
                has_decimal = True
            elif not c.isdigit():
                break
            self.read_char()
            buf += c

        if has_decimal:
            return self.make_token(FloatLit, float(buf))
        return self.make_token(IntLit, int(buf))

    def _starts_operator(self, buf: str) -> bool:
        # TODO(max): Rewrite using trie
        return any(op.startswith(buf) for op in PS.keys())

    def read_op(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input():
            c = self.peek_char()
            if not self._starts_operator(buf + c):
                break
            self.read_char()
            buf += c
        if buf in PS.keys():
            return self.make_token(Operator, buf)
        raise ParseError(f"unexpected token {buf!r}")

    def read_var(self, first_char: str) -> Token:
        buf = first_char
        while self.has_input() and is_identifier_char(c := self.peek_char()):
            self.read_char()
            buf += c
        return self.make_token(Name, buf)

    def read_bytes(self) -> Token:
        buf = ""
        while self.has_input():
            if self.peek_char().isspace():
                break
            buf += self.read_char()
        base, _, value = buf.rpartition("'")
        return self.make_token(BytesLit, value, int(base) if base else 64)


PEEK_EMPTY = object()


class Peekable:
    def __init__(self, iterator: Iterator[Any]) -> None:
        self.iterator = iterator
        self.cache = PEEK_EMPTY

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self.cache is not PEEK_EMPTY:
            result = self.cache
            self.cache = PEEK_EMPTY
            return result
        return next(self.iterator)

    def peek(self) -> Any:
        result = self.cache = next(self)
        return result


def tokenize(x: str) -> Peekable:
    lexer = Lexer(x)
    return Peekable(lexer.read_tokens())


@dataclass(frozen=True)
class Prec:
    pl: float
    pr: float


def lp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n - 0.1)


def rp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n + 0.1)


def np(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, n)


def xp(n: float) -> Prec:
    # TODO(max): Rewrite
    return Prec(n, 0)


PS = {
    "::": lp(2000),
    "@": rp(1001),
    "": rp(1000),
    ">>": lp(14),
    "<<": lp(14),
    "^": rp(13),
    "*": rp(12),
    "/": rp(12),
    "//": lp(12),
    "%": lp(12),
    "+": lp(11),
    "-": lp(11),
    ">*": rp(10),
    "++": rp(10),
    ">+": lp(10),
    "+<": rp(10),
    "==": np(9),
    "/=": np(9),
    "<": np(9),
    ">": np(9),
    "<=": np(9),
    ">=": np(9),
    "&&": rp(8),
    "||": rp(7),
    "|>": rp(6),
    "<|": lp(6),
    "#": lp(5.5),
    "->": lp(5),
    "|": rp(4.5),
    ":": lp(4.5),
    "=": rp(4),
    "!": lp(3),
    ".": rp(3),
    "?": rp(3),
    ",": xp(1),
    # TODO: Fix precedence for spread
    "...": xp(0),
}


HIGHEST_PREC: float = max(max(p.pl, p.pr) for p in PS.values())


OPER_CHARS = set("".join(PS.keys()))
assert " " not in OPER_CHARS


class SyntacticError(Exception):
    pass


class ParseError(SyntacticError):
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class UnexpectedTokenError(ParseError):
    unexpected_token: Token


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class InvalidTokenError(ParseError):
    unexpected_token: SourceExtent = dataclasses.field(default_factory=SourceExtent, compare=False)


# TODO(max): Replace with EOFError?
class UnexpectedEOFError(ParseError):
    pass


def parse_assign(tokens: Peekable, p: float = 0) -> "Assign":
    assign = parse_binary(tokens, p)
    if isinstance(assign, Spread):
        return Assign(RECORD_SPREAD_KEY_PLACEHOLDER, assign)
    if not isinstance(assign, Assign):
        raise ParseError("failed to parse variable assignment in record constructor")
    return assign


def gensym() -> str:
    gensym.counter += 1  # type: ignore
    return f"$v{gensym.counter}"  # type: ignore


def gensym_reset() -> None:
    gensym.counter = -1  # type: ignore


gensym_reset()


def make_source_annotated_object(cls: type, source_extent: Optional[SourceExtent], *args: Any) -> Object:
    result: Object = cls(*args)
    object.__setattr__(result, "source_extent", source_extent)
    return result


def parse_unary(tokens: Peekable, p: float) -> "Object":
    token = next(tokens)
    l: Object
    if isinstance(token, IntLit):
        return make_source_annotated_object(Int, token.source_extent, token.value)
    elif isinstance(token, FloatLit):
        return make_source_annotated_object(Float, token.source_extent, token.value)
    elif isinstance(token, Name):
        # TODO: Handle kebab case vars
        return make_source_annotated_object(Var, token.source_extent, token.value)
    elif isinstance(token, Hash):
        hash_source_extent = token.source_extent
        if isinstance(variant_tag := next(tokens), Name):
            # It needs to be higher than the precedence of the -> operator so that
            # we can match variants in MatchFunction
            # It needs to be higher than the precedence of the && operator so that
            # we can use #true() and #false() in boolean expressions
            # It needs to be higher than the precedence of juxtaposition so that
            # f #true() #false() is parsed as f(TRUE)(FALSE)
            variant_payload = parse_binary(tokens, PS[""].pr + 1)
            return make_source_annotated_object(
                Variant,
                hash_source_extent.coalesce(variant_payload.source_extent),
                variant_tag.value,
                variant_payload,
            )
        else:
            raise UnexpectedTokenError(variant_tag)
    elif isinstance(token, BytesLit):
        base = token.base
        if base == 85:
            l = Bytes(base64.b85decode(token.value))
        elif base == 64:
            l = Bytes(base64.b64decode(token.value))
        elif base == 32:
            l = Bytes(base64.b32decode(token.value))
        elif base == 16:
            l = Bytes(base64.b16decode(token.value))
        else:
            raise ParseError(f"unexpected base {base!r} in {token!r}")
        object.__setattr__(l, "source_extent", token.source_extent)
        return l
    elif isinstance(token, StringLit):
        return make_source_annotated_object(String, token.source_extent, token.value)
    elif token == Operator("..."):
        try:
            if isinstance(tokens.peek(), Name):
                spread_variable = next(tokens)
                return make_source_annotated_object(
                    Spread, token.source_extent.coalesce(spread_variable.source_extent), spread_variable.value
                )
            else:
                return make_source_annotated_object(Spread, token.source_extent)
        except StopIteration:
            return Spread()
    elif token == Operator("|"):
        pipe_source_extent = token.source_extent
        expr = parse_binary(tokens, PS["|"].pr)  # TODO: make this work for larger arities
        if not isinstance(expr, Function):
            raise ParseError(f"expected function in match expression {expr!r}")
        match_case = make_source_annotated_object(
            MatchCase, pipe_source_extent.coalesce(expr.source_extent), expr.arg, expr.body
        )
        cases = [match_case]
        match_function_source_extent = match_case.source_extent
        while True:
            try:
                if tokens.peek() != Operator("|"):
                    break
            except StopIteration:
                break
            pipe_source_extent = next(tokens).source_extent
            expr = parse_binary(tokens, PS["|"].pr)  # TODO: make this work for larger arities
            if not isinstance(expr, Function):
                raise ParseError(f"expected function in match expression {expr!r}")
            match_case = make_source_annotated_object(
                MatchCase, pipe_source_extent.coalesce(expr.source_extent), expr.arg, expr.body
            )
            cases.append(match_case)
            match_function_source_extent = join_source_extents(match_function_source_extent, match_case.source_extent)
        return make_source_annotated_object(
            MatchFunction,
            match_function_source_extent,
            cases,
        )
    elif isinstance(token, LeftParen):
        left_paren_source_extent = token.source_extent
        if isinstance(tokens.peek(), RightParen):
            l = make_source_annotated_object(Hole, left_paren_source_extent.coalesce(next(tokens).source_extent))
        else:
            l = parse(tokens)
            object.__setattr__(l, "source_extent", left_paren_source_extent.coalesce(next(tokens).source_extent))
        return l
    elif isinstance(token, LeftBracket):
        list_start_source_extent = token.source_extent
        l = List([])
        token = tokens.peek()
        if isinstance(token, RightBracket):
            list_end_source_extent = next(tokens).source_extent
        else:
            l.items.append(parse_binary(tokens, 2))
            while not isinstance(token := next(tokens), RightBracket):
                if isinstance(l.items[-1], Spread):
                    raise ParseError("spread must come at end of list match")
                # TODO: Implement .. operator
                l.items.append(parse_binary(tokens, 2))
            list_end_source_extent = token.source_extent
        object.__setattr__(l, "source_extent", list_start_source_extent.coalesce(list_end_source_extent))
        return l
    elif isinstance(token, LeftBrace):
        record_start_source_extent = token.source_extent
        l = Record({})
        token = tokens.peek()
        if isinstance(token, RightBrace):
            record_end_source_extent = next(tokens).source_extent
        else:
            assign = parse_assign(tokens, 2)
            l.data[assign.name.name] = assign.value
            while not isinstance(token := next(tokens), RightBrace):
                if isinstance(assign.value, Spread):
                    raise ParseError("spread must come at end of record match")
                # TODO: Implement .. operator
                assign = parse_assign(tokens, 2)
                l.data[assign.name.name] = assign.value
            record_end_source_extent = token.source_extent
        object.__setattr__(l, "source_extent", record_start_source_extent.coalesce(record_end_source_extent))
        return l
    elif token == Operator("-"):
        # Unary minus
        # Precedence was chosen to be higher than binary ops so that -a op
        # b is (-a) op b and not -(a op b).
        # Precedence was chosen to be higher than function application so that
        # -a b is (-a) b and not -(a b).
        r = parse_binary(tokens, HIGHEST_PREC + 1)
        source_extent = token.source_extent.coalesce(r.source_extent)
        if isinstance(r, Int):
            assert r.value >= 0, "Tokens should never have negative values"
            return make_source_annotated_object(Int, source_extent, -r.value)
        if isinstance(r, Float):
            assert r.value >= 0, "Tokens should never have negative values"
            return make_source_annotated_object(Float, source_extent, -r.value)
        return make_source_annotated_object(Binop, source_extent, BinopKind.SUB, Int(0), r)
    else:
        raise UnexpectedTokenError(token)


def parse_binary(tokens: Peekable, p: float) -> "Object":
    l: Object = parse_unary(tokens, p)
    while True:
        op: Token
        try:
            op = tokens.peek()
        except StopIteration:
            break
        if isinstance(op, (RightParen, RightBracket, RightBrace)):
            break
        if not isinstance(op, Operator):
            prec = PS[""]
            pl, pr = prec.pl, prec.pr
            if pl < p:
                break
            arg = parse_binary(tokens, pr)
            l = make_source_annotated_object(Apply, join_source_extents(l.source_extent, arg.source_extent), l, arg)
            continue
        prec = PS[op.value]
        pl, pr = prec.pl, prec.pr
        if pl < p:
            break
        next(tokens)
        if op == Operator("="):
            if not isinstance(l, Var):
                raise ParseError(f"expected variable in assignment {l!r}")
            value = parse_binary(tokens, pr)
            l = make_source_annotated_object(
                Assign, join_source_extents(l.source_extent, value.source_extent), l, value
            )
        elif op == Operator("->"):
            body = parse_binary(tokens, pr)
            l = make_source_annotated_object(
                Function, join_source_extents(l.source_extent, body.source_extent), l, body
            )
        elif op == Operator("|>"):
            func = parse_binary(tokens, pr)
            l = make_source_annotated_object(Apply, join_source_extents(func.source_extent, l.source_extent), func, l)
        elif op == Operator("<|"):
            arg = parse_binary(tokens, pr)
            l = make_source_annotated_object(Apply, join_source_extents(l.source_extent, arg.source_extent), l, arg)
        elif op == Operator(">>"):
            r = parse_binary(tokens, pr)
            varname = gensym()
            l = make_source_annotated_object(
                Function,
                join_source_extents(l.source_extent, r.source_extent),
                Var(varname),
                Apply(r, Apply(l, Var(varname))),
            )
        elif op == Operator("<<"):
            r = parse_binary(tokens, pr)
            varname = gensym()
            l = make_source_annotated_object(
                Function,
                join_source_extents(l.source_extent, r.source_extent),
                Var(varname),
                Apply(l, Apply(r, Var(varname))),
            )
        elif op == Operator("."):
            binding = parse_binary(tokens, pr)
            l = make_source_annotated_object(
                Where, join_source_extents(l.source_extent, binding.source_extent), l, binding
            )
        elif op == Operator("?"):
            cond = parse_binary(tokens, pr)
            l = make_source_annotated_object(Assert, join_source_extents(l.source_extent, cond.source_extent), l, cond)
        elif op == Operator("@"):
            # TODO: revisit whether to use @ or . for field access
            at = parse_binary(tokens, pr)
            l = make_source_annotated_object(Access, join_source_extents(l.source_extent, at.source_extent), l, at)
        else:
            assert isinstance(op, Operator)
            right = parse_binary(tokens, pr)
            l = make_source_annotated_object(
                Binop,
                join_source_extents(l.source_extent, right.source_extent),
                BinopKind.from_str(op.value),
                l,
                right,
            )
    return l


def parse(tokens: Peekable) -> "Object":
    try:
        return parse_binary(tokens, 0)
    except StopIteration:
        raise UnexpectedEOFError("unexpected end of input")


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Object:
    source_extent: Optional[SourceExtent] = dataclasses.field(default=None, compare=False, init=False, repr=False)

    def __str__(self) -> str:
        return pretty(self)


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Int(Object):
    value: int


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Float(Object):
    value: float


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class String(Object):
    value: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Bytes(Object):
    value: bytes


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Var(Object):
    name: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Hole(Object):
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Spread(Object):
    name: Optional[str] = None


RECORD_SPREAD_KEY_PLACEHOLDER = Var("...")


Env = Mapping[str, Object]


# TODO(max): Add source extents for BinopKind?
class BinopKind(enum.Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    FLOOR_DIV = auto()
    EXP = auto()
    MOD = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    BOOL_AND = auto()
    BOOL_OR = auto()
    STRING_CONCAT = auto()
    LIST_CONS = auto()
    LIST_APPEND = auto()
    RIGHT_EVAL = auto()
    HASTYPE = auto()
    PIPE = auto()
    REVERSE_PIPE = auto()

    @classmethod
    def from_str(cls, x: str) -> "BinopKind":
        return {
            "+": cls.ADD,
            "-": cls.SUB,
            "*": cls.MUL,
            "/": cls.DIV,
            "//": cls.FLOOR_DIV,
            "^": cls.EXP,
            "%": cls.MOD,
            "==": cls.EQUAL,
            "/=": cls.NOT_EQUAL,
            "<": cls.LESS,
            ">": cls.GREATER,
            "<=": cls.LESS_EQUAL,
            ">=": cls.GREATER_EQUAL,
            "&&": cls.BOOL_AND,
            "||": cls.BOOL_OR,
            "++": cls.STRING_CONCAT,
            ">+": cls.LIST_CONS,
            "+<": cls.LIST_APPEND,
            "!": cls.RIGHT_EVAL,
            ":": cls.HASTYPE,
            "|>": cls.PIPE,
            "<|": cls.REVERSE_PIPE,
        }[x]

    @classmethod
    def to_str(cls, binop_kind: "BinopKind") -> str:
        return {
            cls.ADD: "+",
            cls.SUB: "-",
            cls.MUL: "*",
            cls.DIV: "/",
            cls.EXP: "^",
            cls.MOD: "%",
            cls.EQUAL: "==",
            cls.NOT_EQUAL: "/=",
            cls.LESS: "<",
            cls.GREATER: ">",
            cls.LESS_EQUAL: "<=",
            cls.GREATER_EQUAL: ">=",
            cls.BOOL_AND: "&&",
            cls.BOOL_OR: "||",
            cls.STRING_CONCAT: "++",
            cls.LIST_CONS: ">+",
            cls.LIST_APPEND: "+<",
            cls.RIGHT_EVAL: "!",
            cls.HASTYPE: ":",
            cls.PIPE: "|>",
            cls.REVERSE_PIPE: "<|",
        }[binop_kind]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Binop(Object):
    op: BinopKind
    left: Object
    right: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class List(Object):
    items: typing.List[Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assign(Object):
    name: Var
    value: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Function(Object):
    arg: Object
    body: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Apply(Object):
    func: Object
    arg: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Where(Object):
    body: Object
    binding: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Assert(Object):
    value: Object
    cond: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class EnvObject(Object):
    env: Env

    def __str__(self) -> str:
        return f"EnvObject(keys={self.env.keys()})"


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchCase(Object):
    pattern: Object
    body: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class MatchFunction(Object):
    cases: typing.List[MatchCase]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Relocation(Object):
    name: str


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunctionRelocation(Relocation):
    pass


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class NativeFunction(Object):
    name: str
    func: Callable[[Object], Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Closure(Object):
    env: Env
    func: Union[Function, MatchFunction]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Record(Object):
    data: Dict[str, Object]


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Access(Object):
    obj: Object
    at: Object


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Variant(Object):
    tag: str
    value: Object


tags = [
    TYPE_SHORT := b"i",  # fits in 64 bits
    TYPE_LONG := b"l",  # bignum
    TYPE_FLOAT := b"d",
    TYPE_STRING := b"s",
    TYPE_REF := b"r",
    TYPE_LIST := b"[",
    TYPE_RECORD := b"{",
    TYPE_VARIANT := b"#",
    TYPE_VAR := b"v",
    TYPE_FUNCTION := b"f",
    TYPE_MATCH_FUNCTION := b"m",
    TYPE_CLOSURE := b"c",
    TYPE_BYTES := b"b",
    TYPE_HOLE := b"(",
    TYPE_ASSIGN := b"=",
    TYPE_BINOP := b"+",
    TYPE_APPLY := b" ",
    TYPE_WHERE := b".",
    TYPE_ACCESS := b"@",
    TYPE_SPREAD := b"S",
    TYPE_NAMED_SPREAD := b"R",
    TYPE_TRUE := b"T",
    TYPE_FALSE := b"F",
]
FLAG_REF = 0x80


BITS_PER_BYTE = 8
BYTES_PER_DIGIT = 8
BITS_PER_DIGIT = BYTES_PER_DIGIT * BITS_PER_BYTE
DIGIT_MASK = (1 << BITS_PER_DIGIT) - 1


def ref(tag: bytes) -> bytes:
    return (tag[0] | FLAG_REF).to_bytes(1, "little")


tags = tags + [ref(v) for v in tags]
assert len(tags) == len(set(tags)), "Duplicate tags"
assert all(len(v) == 1 for v in tags), "Tags must be 1 byte"
assert all(isinstance(v, bytes) for v in tags)


def zigzag_encode(val: int) -> int:
    if val < 0:
        return -2 * val - 1
    return 2 * val


def zigzag_decode(val: int) -> int:
    if val & 1 == 1:
        return -val // 2
    return val // 2


@dataclass
class Serializer:
    refs: typing.List[Object] = dataclasses.field(default_factory=list)
    output: bytearray = dataclasses.field(default_factory=bytearray)

    def ref(self, obj: Object) -> Optional[int]:
        for idx, ref in enumerate(self.refs):
            if ref is obj:
                return idx
        return None

    def add_ref(self, ty: bytes, obj: Object) -> int:
        assert len(ty) == 1
        assert self.ref(obj) is None
        self.emit(ref(ty))
        result = len(self.refs)
        self.refs.append(obj)
        return result

    def emit(self, obj: bytes) -> None:
        self.output.extend(obj)

    def _fits_in_nbits(self, obj: int, nbits: int) -> bool:
        return -(1 << (nbits - 1)) <= obj < (1 << (nbits - 1))

    def _short(self, number: int) -> bytes:
        # From Peter Ruibal, https://github.com/fmoo/python-varint
        number = zigzag_encode(number)
        buf = bytearray()
        while True:
            towrite = number & 0x7F
            number >>= 7
            if number:
                buf.append(towrite | 0x80)
            else:
                buf.append(towrite)
                break
        return bytes(buf)

    def _long(self, number: int) -> bytes:
        digits = []
        number = zigzag_encode(number)
        while number:
            digits.append(number & DIGIT_MASK)
            number >>= BITS_PER_DIGIT
        buf = bytearray(self._short(len(digits)))
        for digit in digits:
            buf.extend(digit.to_bytes(BYTES_PER_DIGIT, "little"))
        return bytes(buf)

    def _string(self, obj: str) -> bytes:
        encoded = obj.encode("utf-8")
        return self._short(len(encoded)) + encoded

    def serialize(self, obj: Object) -> None:
        assert isinstance(obj, Object), type(obj)
        if (ref := self.ref(obj)) is not None:
            return self.emit(TYPE_REF + self._short(ref))
        if isinstance(obj, Int):
            if self._fits_in_nbits(obj.value, 64):
                self.emit(TYPE_SHORT)
                self.emit(self._short(obj.value))
                return
            self.emit(TYPE_LONG)
            self.emit(self._long(obj.value))
            return
        if isinstance(obj, String):
            return self.emit(TYPE_STRING + self._string(obj.value))
        if isinstance(obj, List):
            self.add_ref(TYPE_LIST, obj)
            self.emit(self._short(len(obj.items)))
            for item in obj.items:
                self.serialize(item)
            return
        if isinstance(obj, Variant):
            if obj.tag == "true" and isinstance(obj.value, Hole):
                return self.emit(TYPE_TRUE)
            if obj.tag == "false" and isinstance(obj.value, Hole):
                return self.emit(TYPE_FALSE)
            # TODO(max): Determine if this should be a ref
            self.emit(TYPE_VARIANT)
            # TODO(max): String pool (via refs) for strings longer than some length?
            self.emit(self._string(obj.tag))
            return self.serialize(obj.value)
        if isinstance(obj, Record):
            # TODO(max): Determine if this should be a ref
            self.emit(TYPE_RECORD)
            self.emit(self._short(len(obj.data)))
            for key, value in obj.data.items():
                self.emit(self._string(key))
                self.serialize(value)
            return
        if isinstance(obj, Var):
            return self.emit(TYPE_VAR + self._string(obj.name))
        if isinstance(obj, Function):
            self.emit(TYPE_FUNCTION)
            self.serialize(obj.arg)
            return self.serialize(obj.body)
        if isinstance(obj, MatchFunction):
            self.emit(TYPE_MATCH_FUNCTION)
            self.emit(self._short(len(obj.cases)))
            for case in obj.cases:
                self.serialize(case.pattern)
                self.serialize(case.body)
            return
        if isinstance(obj, Closure):
            self.add_ref(TYPE_CLOSURE, obj)
            self.serialize(obj.func)
            self.emit(self._short(len(obj.env)))
            for key, value in obj.env.items():
                self.emit(self._string(key))
                self.serialize(value)
            return
        if isinstance(obj, Bytes):
            self.emit(TYPE_BYTES)
            self.emit(self._short(len(obj.value)))
            self.emit(obj.value)
            return
        if isinstance(obj, Float):
            self.emit(TYPE_FLOAT)
            self.emit(struct.pack("<d", obj.value))
            return
        if isinstance(obj, Hole):
            self.emit(TYPE_HOLE)
            return
        if isinstance(obj, Assign):
            self.emit(TYPE_ASSIGN)
            self.serialize(obj.name)
            self.serialize(obj.value)
            return
        if isinstance(obj, Binop):
            self.emit(TYPE_BINOP)
            self.emit(self._string(BinopKind.to_str(obj.op)))
            self.serialize(obj.left)
            self.serialize(obj.right)
            return
        if isinstance(obj, Apply):
            self.emit(TYPE_APPLY)
            self.serialize(obj.func)
            self.serialize(obj.arg)
            return
        if isinstance(obj, Where):
            self.emit(TYPE_WHERE)
            self.serialize(obj.body)
            self.serialize(obj.binding)
            return
        if isinstance(obj, Access):
            self.emit(TYPE_ACCESS)
            self.serialize(obj.obj)
            self.serialize(obj.at)
            return
        if isinstance(obj, Spread):
            if obj.name is not None:
                self.emit(TYPE_NAMED_SPREAD)
                self.emit(self._string(obj.name))
                return
            self.emit(TYPE_SPREAD)
            return
        raise NotImplementedError(type(obj))


@dataclass
class Deserializer:
    flat: Union[bytes, memoryview]
    idx: int = 0
    refs: typing.List[Object] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.flat, bytes):
            self.flat = memoryview(self.flat)

    def read(self, size: int) -> memoryview:
        result = memoryview(self.flat[self.idx : self.idx + size])
        self.idx += size
        return result

    def read_tag(self) -> Tuple[bytes, bool]:
        tag = self.read(1)[0]
        is_ref = bool(tag & FLAG_REF)
        return (tag & ~FLAG_REF).to_bytes(1, "little"), is_ref

    def _string(self) -> str:
        length = self._short()
        encoded = self.read(length)
        return str(encoded, "utf-8")

    def _short(self) -> int:
        # From Peter Ruibal, https://github.com/fmoo/python-varint
        shift = 0
        result = 0
        while True:
            i = self.read(1)[0]
            result |= (i & 0x7F) << shift
            shift += 7
            if not (i & 0x80):
                break
        return zigzag_decode(result)

    def _long(self) -> int:
        num_digits = self._short()
        digits = []
        for _ in range(num_digits):
            digit = int.from_bytes(self.read(BYTES_PER_DIGIT), "little")
            digits.append(digit)
        result = 0
        for digit in reversed(digits):
            result <<= BITS_PER_DIGIT
            result |= digit
        return zigzag_decode(result)

    def parse(self) -> Object:
        ty, is_ref = self.read_tag()
        if ty == TYPE_REF:
            idx = self._short()
            return self.refs[idx]
        if ty == TYPE_SHORT:
            assert not is_ref
            return Int(self._short())
        if ty == TYPE_LONG:
            assert not is_ref
            return Int(self._long())
        if ty == TYPE_STRING:
            assert not is_ref
            return String(self._string())
        if ty == TYPE_LIST:
            length = self._short()
            result_list = List([])
            assert is_ref
            self.refs.append(result_list)
            for i in range(length):
                result_list.items.append(self.parse())
            return result_list
        if ty == TYPE_RECORD:
            assert not is_ref
            length = self._short()
            result_rec = Record({})
            for i in range(length):
                key = self._string()
                value = self.parse()
                result_rec.data[key] = value
            return result_rec
        if ty == TYPE_VARIANT:
            assert not is_ref
            tag = self._string()
            value = self.parse()
            return Variant(tag, value)
        if ty == TYPE_VAR:
            assert not is_ref
            return Var(self._string())
        if ty == TYPE_FUNCTION:
            assert not is_ref
            arg = self.parse()
            body = self.parse()
            return Function(arg, body)
        if ty == TYPE_MATCH_FUNCTION:
            assert not is_ref
            length = self._short()
            result_matchfun = MatchFunction([])
            for i in range(length):
                pattern = self.parse()
                body = self.parse()
                result_matchfun.cases.append(MatchCase(pattern, body))
            return result_matchfun
        if ty == TYPE_CLOSURE:
            func = self.parse()
            length = self._short()
            assert isinstance(func, (Function, MatchFunction))
            result_closure = Closure({}, func)
            assert is_ref
            self.refs.append(result_closure)
            for i in range(length):
                key = self._string()
                value = self.parse()
                assert isinstance(result_closure.env, dict)  # For mypy
                result_closure.env[key] = value
            return result_closure
        if ty == TYPE_BYTES:
            assert not is_ref
            length = self._short()
            return Bytes(self.read(length))
        if ty == TYPE_FLOAT:
            assert not is_ref
            return Float(struct.unpack("<d", self.read(8))[0])
        if ty == TYPE_HOLE:
            assert not is_ref
            return Hole()
        if ty == TYPE_ASSIGN:
            assert not is_ref
            name = self.parse()
            value = self.parse()
            assert isinstance(name, Var)
            return Assign(name, value)
        if ty == TYPE_BINOP:
            assert not is_ref
            op = BinopKind.from_str(self._string())
            left = self.parse()
            right = self.parse()
            return Binop(op, left, right)
        if ty == TYPE_APPLY:
            assert not is_ref
            func = self.parse()
            arg = self.parse()
            return Apply(func, arg)
        if ty == TYPE_WHERE:
            assert not is_ref
            body = self.parse()
            binding = self.parse()
            return Where(body, binding)
        if ty == TYPE_ACCESS:
            assert not is_ref
            obj = self.parse()
            at = self.parse()
            return Access(obj, at)
        if ty == TYPE_SPREAD:
            return Spread()
        if ty == TYPE_NAMED_SPREAD:
            return Spread(self._string())
        if ty == TYPE_TRUE:
            assert not is_ref
            return Variant("true", Hole())
        if ty == TYPE_FALSE:
            assert not is_ref
            return Variant("false", Hole())
        raise NotImplementedError(bytes(ty))


TRUE = Variant("true", Hole())


FALSE = Variant("false", Hole())


def unpack_number(obj: Object) -> Union[int, float]:
    if not isinstance(obj, (Int, Float)):
        raise TypeError(f"expected Int or Float, got {type(obj).__name__}")
    return obj.value


def eval_number(env: Env, exp: Object) -> Union[int, float]:
    result = eval_exp(env, exp)
    return unpack_number(result)


def eval_str(env: Env, exp: Object) -> str:
    result = eval_exp(env, exp)
    if not isinstance(result, String):
        raise TypeError(f"expected String, got {type(result).__name__}")
    return result.value


def eval_bool(env: Env, exp: Object) -> bool:
    result = eval_exp(env, exp)
    if not isinstance(result, Variant):
        raise TypeError(f"expected #true or #false, got {type(result).__name__}")
    if result.tag not in ("true", "false"):
        raise TypeError(f"expected #true or #false, got {type(result).__name__}")
    return result.tag == "true"


def eval_list(env: Env, exp: Object) -> typing.List[Object]:
    result = eval_exp(env, exp)
    if not isinstance(result, List):
        raise TypeError(f"expected List, got {type(result).__name__}")
    return result.items


def make_bool(x: bool) -> Object:
    return TRUE if x else FALSE


def wrap_inferred_number_type(x: Union[int, float]) -> Object:
    # TODO: Since this is intended to be a reference implementation
    # we should avoid relying heavily on Python's implementation of
    # arithmetic operations, type inference, and multiple dispatch.
    # Update this to make the interpreter more language agnostic.
    if isinstance(x, int):
        return Int(x)
    return Float(x)


BINOP_HANDLERS: Dict[BinopKind, Callable[[Env, Object, Object], Object]] = {
    BinopKind.ADD: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) + eval_number(env, y)),
    BinopKind.SUB: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) - eval_number(env, y)),
    BinopKind.MUL: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) * eval_number(env, y)),
    BinopKind.DIV: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) / eval_number(env, y)),
    BinopKind.FLOOR_DIV: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) // eval_number(env, y)),
    BinopKind.EXP: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) ** eval_number(env, y)),
    BinopKind.MOD: lambda env, x, y: wrap_inferred_number_type(eval_number(env, x) % eval_number(env, y)),
    BinopKind.EQUAL: lambda env, x, y: make_bool(eval_exp(env, x) == eval_exp(env, y)),
    BinopKind.NOT_EQUAL: lambda env, x, y: make_bool(eval_exp(env, x) != eval_exp(env, y)),
    BinopKind.LESS: lambda env, x, y: make_bool(eval_number(env, x) < eval_number(env, y)),
    BinopKind.GREATER: lambda env, x, y: make_bool(eval_number(env, x) > eval_number(env, y)),
    BinopKind.LESS_EQUAL: lambda env, x, y: make_bool(eval_number(env, x) <= eval_number(env, y)),
    BinopKind.GREATER_EQUAL: lambda env, x, y: make_bool(eval_number(env, x) >= eval_number(env, y)),
    BinopKind.BOOL_AND: lambda env, x, y: make_bool(eval_bool(env, x) and eval_bool(env, y)),
    BinopKind.BOOL_OR: lambda env, x, y: make_bool(eval_bool(env, x) or eval_bool(env, y)),
    BinopKind.STRING_CONCAT: lambda env, x, y: String(eval_str(env, x) + eval_str(env, y)),
    BinopKind.LIST_CONS: lambda env, x, y: List([eval_exp(env, x)] + eval_list(env, y)),
    BinopKind.LIST_APPEND: lambda env, x, y: List(eval_list(env, x) + [eval_exp(env, y)]),
    BinopKind.RIGHT_EVAL: lambda env, x, y: eval_exp(env, y),
}


class MatchError(Exception):
    pass


def match(obj: Object, pattern: Object) -> Optional[Env]:
    if isinstance(pattern, Hole):
        return {} if isinstance(obj, Hole) else None
    if isinstance(pattern, Int):
        return {} if isinstance(obj, Int) and obj.value == pattern.value else None
    if isinstance(pattern, Float):
        raise MatchError("pattern matching is not supported for Floats")
    if isinstance(pattern, String):
        return {} if isinstance(obj, String) and obj.value == pattern.value else None
    if isinstance(pattern, Var):
        return {pattern.name: obj}
    if isinstance(pattern, Variant):
        if not isinstance(obj, Variant):
            return None
        if obj.tag != pattern.tag:
            return None
        return match(obj.value, pattern.value)
    if isinstance(pattern, Record):
        if not isinstance(obj, Record):
            return None
        result: Env = {}
        seen_keys: set[str] = set()
        for key, pattern_item in pattern.data.items():
            if isinstance(pattern_item, Spread):
                if pattern_item.name is not None:
                    assert isinstance(result, dict)  # for .update()
                    rest_keys = set(obj.data.keys()) - seen_keys
                    result.update({pattern_item.name: Record({key: obj.data[key] for key in rest_keys})})
                return result
            seen_keys.add(key)
            obj_item = obj.data.get(key)
            if obj_item is None:
                return None
            part = match(obj_item, pattern_item)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        if len(pattern.data) != len(obj.data):
            return None
        return result
    if isinstance(pattern, List):
        if not isinstance(obj, List):
            return None
        result: Env = {}  # type: ignore
        for i, pattern_item in enumerate(pattern.items):
            if isinstance(pattern_item, Spread):
                if pattern_item.name is not None:
                    assert isinstance(result, dict)  # for .update()
                    result.update({pattern_item.name: List(obj.items[i:])})
                return result
            if i >= len(obj.items):
                return None
            obj_item = obj.items[i]
            part = match(obj_item, pattern_item)
            if part is None:
                return None
            assert isinstance(result, dict)  # for .update()
            result.update(part)
        if len(pattern.items) != len(obj.items):
            return None
        return result
    raise NotImplementedError(f"match not implemented for {type(pattern).__name__}")


def free_in(exp: Object) -> Set[str]:
    if isinstance(exp, (Int, Float, String, Bytes, Hole, NativeFunction)):
        return set()
    if isinstance(exp, Variant):
        return free_in(exp.value)
    if isinstance(exp, Var):
        return {exp.name}
    if isinstance(exp, Spread):
        if exp.name is not None:
            return {exp.name}
        return set()
    if isinstance(exp, Binop):
        return free_in(exp.left) | free_in(exp.right)
    if isinstance(exp, List):
        if not exp.items:
            return set()
        return set.union(*(free_in(item) for item in exp.items))
    if isinstance(exp, Record):
        if not exp.data:
            return set()
        return set.union(*(free_in(value) for key, value in exp.data.items()))
    if isinstance(exp, Function):
        assert isinstance(exp.arg, Var)
        return free_in(exp.body) - {exp.arg.name}
    if isinstance(exp, MatchFunction):
        if not exp.cases:
            return set()
        return set.union(*(free_in(case) for case in exp.cases))
    if isinstance(exp, MatchCase):
        return free_in(exp.body) - free_in(exp.pattern)
    if isinstance(exp, Apply):
        return free_in(exp.func) | free_in(exp.arg)
    if isinstance(exp, Access):
        # For records, y is not free in x@y; it is a field name.
        # For lists, y *is* free in x@y; it is an index expression (could be a
        # var).
        # For now, we'll assume it might be an expression and mark it as a
        # (possibly extra) freevar.
        return free_in(exp.obj) | free_in(exp.at)
    if isinstance(exp, Where):
        assert isinstance(exp.binding, Assign)
        return (free_in(exp.body) - {exp.binding.name.name}) | free_in(exp.binding)
    if isinstance(exp, Assign):
        return free_in(exp.value)
    if isinstance(exp, Closure):
        # TODO(max): Should this remove the set of keys in the closure env?
        return free_in(exp.func)
    raise NotImplementedError(("free_in", type(exp)))


def improve_closure(closure: Closure) -> Closure:
    freevars = free_in(closure.func)
    env = {boundvar: value for boundvar, value in closure.env.items() if boundvar in freevars}
    return Closure(env, closure.func)


def eval_exp(env: Env, exp: Object) -> Object:
    logger.debug(exp)
    if isinstance(exp, (Int, Float, String, Bytes, Hole, Closure, NativeFunction)):
        return exp
    if isinstance(exp, Variant):
        return Variant(exp.tag, eval_exp(env, exp.value))
    if isinstance(exp, Var):
        value = env.get(exp.name)
        if value is None:
            raise NameError(f"name '{exp.name}' is not defined")
        return value
    if isinstance(exp, Binop):
        handler = BINOP_HANDLERS.get(exp.op)
        if handler is None:
            raise NotImplementedError(f"no handler for {exp.op}")
        return handler(env, exp.left, exp.right)
    if isinstance(exp, List):
        return List([eval_exp(env, item) for item in exp.items])
    if isinstance(exp, Record):
        return Record({k: eval_exp(env, exp.data[k]) for k in exp.data})
    if isinstance(exp, Assign):
        # TODO(max): Rework this. There's something about matching that we need
        # to figure out and implement.
        assert isinstance(exp.name, Var)
        value = eval_exp(env, exp.value)
        if isinstance(value, Closure):
            # We want functions to be able to call themselves without using the
            # Y combinator or similar, so we bind functions (and only
            # functions) using a letrec-like strategy. We augment their
            # captured environment with a binding to themselves.
            assert isinstance(value.env, dict)
            value.env[exp.name.name] = value
            # We still improve_closure here even though we also did it on
            # Closure creation because the Closure might not need a binding for
            # itself (it might not be recursive).
            value = improve_closure(value)
        return EnvObject({**env, exp.name.name: value})
    if isinstance(exp, Where):
        assert isinstance(exp.binding, Assign)
        res_env = eval_exp(env, exp.binding)
        assert isinstance(res_env, EnvObject)
        new_env = {**env, **res_env.env}
        return eval_exp(new_env, exp.body)
    if isinstance(exp, Assert):
        cond = eval_exp(env, exp.cond)
        if cond != TRUE:
            raise AssertionError(f"condition {exp.cond} failed")
        return eval_exp(env, exp.value)
    if isinstance(exp, Function):
        if not isinstance(exp.arg, Var):
            raise RuntimeError(f"expected variable in function definition {exp.arg}")
        value = Closure(env, exp)
        value = improve_closure(value)
        return value
    if isinstance(exp, MatchFunction):
        value = Closure(env, exp)
        value = improve_closure(value)
        return value
    if isinstance(exp, Apply):
        if isinstance(exp.func, Var) and exp.func.name == "$$quote":
            return exp.arg
        callee = eval_exp(env, exp.func)
        arg = eval_exp(env, exp.arg)
        if isinstance(callee, NativeFunction):
            return callee.func(arg)
        if not isinstance(callee, Closure):
            raise TypeError(f"attempted to apply a non-closure of type {type(callee).__name__}")
        if isinstance(callee.func, Function):
            assert isinstance(callee.func.arg, Var)
            new_env = {**callee.env, callee.func.arg.name: arg}
            return eval_exp(new_env, callee.func.body)
        elif isinstance(callee.func, MatchFunction):
            for case in callee.func.cases:
                m = match(arg, case.pattern)
                if m is None:
                    continue
                return eval_exp({**callee.env, **m}, case.body)
            raise MatchError("no matching cases")
        else:
            raise TypeError(f"attempted to apply a non-function of type {type(callee.func).__name__}")
    if isinstance(exp, Access):
        obj = eval_exp(env, exp.obj)
        if isinstance(obj, Record):
            if not isinstance(exp.at, Var):
                raise TypeError(f"cannot access record field using {type(exp.at).__name__}, expected a field name")
            if exp.at.name not in obj.data:
                raise NameError(f"no assignment to {exp.at.name} found in record")
            return obj.data[exp.at.name]
        elif isinstance(obj, List):
            access_at = eval_exp(env, exp.at)
            if not isinstance(access_at, Int):
                raise TypeError(f"cannot index into list using type {type(access_at).__name__}, expected integer")
            if access_at.value < 0 or access_at.value >= len(obj.items):
                raise ValueError(f"index {access_at.value} out of bounds for list")
            return obj.items[access_at.value]
        raise TypeError(f"attempted to access from type {type(obj).__name__}")
    elif isinstance(exp, Spread):
        raise RuntimeError("cannot evaluate a spread")
    raise NotImplementedError(f"eval_exp not implemented for {exp}")


class ScrapMonad:
    def __init__(self, env: Env) -> None:
        assert isinstance(env, dict)  # for .copy()
        self.env: Env = env.copy()

    def bind(self, exp: Object) -> Tuple[Object, "ScrapMonad"]:
        env = self.env
        result = eval_exp(env, exp)
        if isinstance(result, EnvObject):
            return result, ScrapMonad({**env, **result.env})
        return result, ScrapMonad({**env, "_": result})


class InferenceError(Exception):
    pass


@dataclasses.dataclass
class MonoType:
    def find(self) -> MonoType:
        return self


@dataclasses.dataclass
class TyVar(MonoType):
    forwarded: MonoType | None = dataclasses.field(init=False, default=None)
    name: str

    def find(self) -> MonoType:
        result: MonoType = self
        while isinstance(result, TyVar):
            it = result.forwarded
            if it is None:
                return result
            result = it
        return result

    def __str__(self) -> str:
        return f"'{self.name}"

    def make_equal_to(self, other: MonoType) -> None:
        chain_end = self.find()
        if not isinstance(chain_end, TyVar):
            raise InferenceError(f"{self} is already resolved to {chain_end}")
        chain_end.forwarded = other

    def is_unbound(self) -> bool:
        return self.forwarded is None


@dataclasses.dataclass
class TyCon(MonoType):
    name: str
    args: list[MonoType]

    def __str__(self) -> str:
        # TODO(max): Precedence pretty-print type constructors
        if not self.args:
            return self.name
        if len(self.args) == 1:
            return f"({self.args[0]} {self.name})"
        return f"({self.name.join(map(str, self.args))})"


@dataclasses.dataclass
class TyEmptyRow(MonoType):
    def __str__(self) -> str:
        return "{}"


@dataclasses.dataclass
class TyRow(MonoType):
    fields: dict[str, MonoType]
    rest: TyVar | TyEmptyRow = dataclasses.field(default_factory=TyEmptyRow)

    def __post_init__(self) -> None:
        if not self.fields and isinstance(self.rest, TyEmptyRow):
            raise InferenceError("Empty row must have a rest type")

    def __str__(self) -> str:
        flat, rest = row_flatten(self)
        # sort to make tests deterministic
        result = [f"{key}={val}" for key, val in sorted(flat.items())]
        if isinstance(rest, TyVar):
            result.append(f"...{rest}")
        else:
            assert isinstance(rest, TyEmptyRow)
        return "{" + ", ".join(result) + "}"


def row_flatten(rec: MonoType) -> tuple[dict[str, MonoType], TyVar | TyEmptyRow]:
    if isinstance(rec, TyVar):
        rec = rec.find()
        if isinstance(rec, TyVar):
            return {}, rec
    if isinstance(rec, TyRow):
        flat, rest = row_flatten(rec.rest)
        flat.update(rec.fields)
        return flat, rest
    if isinstance(rec, TyEmptyRow):
        return {}, rec
    raise InferenceError(f"Expected record type, got {type(rec)}")


@dataclasses.dataclass
class Forall:
    tyvars: list[TyVar]
    ty: MonoType

    def __str__(self) -> str:
        return f"(forall {', '.join(map(str, self.tyvars))}. {self.ty})"


def func_type(*args: MonoType) -> TyCon:
    assert len(args) >= 2
    if len(args) == 2:
        return TyCon("->", list(args))
    return TyCon("->", [args[0], func_type(*args[1:])])


def list_type(arg: MonoType) -> TyCon:
    return TyCon("list", [arg])


def unify_fail(ty1: MonoType, ty2: MonoType) -> None:
    raise InferenceError(f"Unification failed for {ty1} and {ty2}")


def occurs_in(tyvar: TyVar, ty: MonoType) -> bool:
    if isinstance(ty, TyVar):
        return tyvar == ty
    if isinstance(ty, TyCon):
        return any(occurs_in(tyvar, arg) for arg in ty.args)
    if isinstance(ty, TyEmptyRow):
        return False
    if isinstance(ty, TyRow):
        return any(occurs_in(tyvar, val) for val in ty.fields.values()) or occurs_in(tyvar, ty.rest)
    raise InferenceError(f"Unknown type: {ty}")


def unify_type(ty1: MonoType, ty2: MonoType) -> None:
    ty1 = ty1.find()
    ty2 = ty2.find()
    if isinstance(ty1, TyVar):
        if occurs_in(ty1, ty2):
            raise InferenceError(f"Occurs check failed for {ty1} and {ty2}")
        ty1.make_equal_to(ty2)
        return
    if isinstance(ty2, TyVar):  # Mirror
        return unify_type(ty2, ty1)
    if isinstance(ty1, TyCon) and isinstance(ty2, TyCon):
        if ty1.name != ty2.name:
            unify_fail(ty1, ty2)
            return
        if len(ty1.args) != len(ty2.args):
            unify_fail(ty1, ty2)
            return
        for l, r in zip(ty1.args, ty2.args):
            unify_type(l, r)
        return
    if isinstance(ty1, TyEmptyRow) and isinstance(ty2, TyEmptyRow):
        return
    if isinstance(ty1, TyRow) and isinstance(ty2, TyRow):
        ty1_fields, ty1_rest = row_flatten(ty1)
        ty2_fields, ty2_rest = row_flatten(ty2)
        ty1_missing = {}
        ty2_missing = {}
        all_field_names = set(ty1_fields.keys()) | set(ty2_fields.keys())
        for key in sorted(all_field_names):  # Sort for deterministic error messages
            ty1_val = ty1_fields.get(key)
            ty2_val = ty2_fields.get(key)
            if ty1_val is not None and ty2_val is not None:
                unify_type(ty1_val, ty2_val)
            elif ty1_val is None:
                assert ty2_val is not None
                ty1_missing[key] = ty2_val
            elif ty2_val is None:
                assert ty1_val is not None
                ty2_missing[key] = ty1_val
        # In general, we want to:
        # 1) Add missing fields from one row to the other row
        # 2) "Keep the rows unified" by linking each row's rest to the other
        #    row's rest
        if not ty1_missing and not ty2_missing:
            # The rests are either both empty (rows were closed) or both
            # unbound type variables (rows were open); unify the rest variables
            unify_type(ty1_rest, ty2_rest)
            return
        if not ty1_missing:
            # The first row has fields that the second row doesn't have; add
            # them to the second row
            unify_type(ty2_rest, TyRow(ty2_missing, ty1_rest))
            return
        if not ty2_missing:
            # The second row has fields that the first row doesn't have; add
            # them to the first row
            unify_type(ty1_rest, TyRow(ty1_missing, ty2_rest))
            return
        # They each have fields the other lacks; create new rows sharing a rest
        # and add the missing fields to each row
        rest = fresh_tyvar()
        unify_type(ty1_rest, TyRow(ty1_missing, rest))
        unify_type(ty2_rest, TyRow(ty2_missing, rest))
        return
    if isinstance(ty1, TyRow) and isinstance(ty2, TyEmptyRow):
        raise InferenceError(f"Unifying row {ty1} with empty row")
    if isinstance(ty1, TyEmptyRow) and isinstance(ty2, TyRow):
        raise InferenceError(f"Unifying empty row with row {ty2}")
    raise InferenceError(f"Cannot unify {ty1} and {ty2}")


Context = typing.Mapping[str, Forall]


fresh_var_counter = 0


def fresh_tyvar(prefix: str = "t") -> TyVar:
    global fresh_var_counter
    result = f"{prefix}{fresh_var_counter}"
    fresh_var_counter += 1
    return TyVar(result)


def reset_tyvar_counter() -> None:
    global fresh_var_counter
    fresh_var_counter = 0


IntType = TyCon("int", [])
StringType = TyCon("string", [])
FloatType = TyCon("float", [])
BytesType = TyCon("bytes", [])
HoleType = TyCon("hole", [])


Subst = typing.Mapping[str, MonoType]


def apply_ty(ty: MonoType, subst: Subst) -> MonoType:
    ty = ty.find()
    if isinstance(ty, TyVar):
        return subst.get(ty.name, ty)
    if isinstance(ty, TyCon):
        return TyCon(ty.name, [apply_ty(arg, subst) for arg in ty.args])
    if isinstance(ty, TyEmptyRow):
        return ty
    if isinstance(ty, TyRow):
        rest = apply_ty(ty.rest, subst)
        assert isinstance(rest, (TyVar, TyEmptyRow))
        return TyRow({key: apply_ty(val, subst) for key, val in ty.fields.items()}, rest)
    raise InferenceError(f"Unknown type: {ty}")


def instantiate(scheme: Forall) -> MonoType:
    fresh = {tyvar.name: fresh_tyvar() for tyvar in scheme.tyvars}
    return apply_ty(scheme.ty, fresh)


def ftv_ty(ty: MonoType) -> set[str]:
    ty = ty.find()
    if isinstance(ty, TyVar):
        return {ty.name}
    if isinstance(ty, TyCon):
        return set().union(*map(ftv_ty, ty.args))
    if isinstance(ty, TyEmptyRow):
        return set()
    if isinstance(ty, TyRow):
        return set().union(*map(ftv_ty, ty.fields.values()), ftv_ty(ty.rest))
    raise InferenceError(f"Unknown type: {ty}")


def generalize(ty: MonoType, ctx: Context) -> Forall:
    def ftv_scheme(ty: Forall) -> set[str]:
        return ftv_ty(ty.ty) - set(tyvar.name for tyvar in ty.tyvars)

    def ftv_ctx(ctx: Context) -> set[str]:
        return set().union(*(ftv_scheme(scheme) for scheme in ctx.values()))

    # TODO(max): Freshen?
    tyvars = ftv_ty(ty) - ftv_ctx(ctx)
    return Forall([TyVar(name) for name in sorted(tyvars)], ty)


def type_of(expr: Object) -> MonoType:
    ty = getattr(expr, "inferred_type", None)
    if ty is not None:
        assert isinstance(ty, MonoType)
        return ty.find()
    return set_type(expr, fresh_tyvar())


def set_type(expr: Object, ty: MonoType) -> MonoType:
    object.__setattr__(expr, "inferred_type", ty)
    return ty


def infer_common(expr: Object) -> MonoType:
    if isinstance(expr, Int):
        return set_type(expr, IntType)
    if isinstance(expr, Float):
        return set_type(expr, FloatType)
    if isinstance(expr, Bytes):
        return set_type(expr, BytesType)
    if isinstance(expr, Hole):
        return set_type(expr, HoleType)
    if isinstance(expr, String):
        return set_type(expr, StringType)
    raise InferenceError(f"{type(expr)} can't be simply inferred")


def infer_pattern_type(pattern: Object, ctx: Context) -> MonoType:
    assert isinstance(ctx, dict)
    if isinstance(pattern, (Int, Float, Bytes, Hole, String)):
        return infer_common(pattern)
    if isinstance(pattern, Var):
        result = fresh_tyvar()
        ctx[pattern.name] = Forall([], result)
        return set_type(pattern, result)
    if isinstance(pattern, List):
        list_item_ty = fresh_tyvar()
        result_ty = list_type(list_item_ty)
        for item in pattern.items:
            if isinstance(item, Spread):
                if item.name is not None:
                    ctx[item.name] = Forall([], result_ty)
                break
            item_ty = infer_pattern_type(item, ctx)
            unify_type(list_item_ty, item_ty)
        return set_type(pattern, result_ty)
    if isinstance(pattern, Record):
        fields = {}
        rest: TyVar | TyEmptyRow = TyEmptyRow()  # Default closed row
        for key, value in pattern.data.items():
            if isinstance(value, Spread):
                # Open row
                rest = fresh_tyvar()
                if value.name is not None:
                    ctx[value.name] = Forall([], rest)
                break
            fields[key] = infer_pattern_type(value, ctx)
        return set_type(pattern, TyRow(fields, rest))
    raise InferenceError(f"{type(pattern)} isn't allowed in a pattern")


def infer_type(expr: Object, ctx: Context) -> MonoType:
    if isinstance(expr, (Int, Float, Bytes, Hole, String)):
        return infer_common(expr)
    if isinstance(expr, Var):
        scheme = ctx.get(expr.name)
        if scheme is None:
            raise InferenceError(f"Unbound variable {expr.name}")
        return set_type(expr, instantiate(scheme))
    if isinstance(expr, Function):
        arg_tyvar = fresh_tyvar()
        assert isinstance(expr.arg, Var)
        body_ctx = {**ctx, expr.arg.name: Forall([], arg_tyvar)}
        body_ty = infer_type(expr.body, body_ctx)
        return set_type(expr, func_type(arg_tyvar, body_ty))
    if isinstance(expr, Binop):
        left, right = expr.left, expr.right
        op = Var(BinopKind.to_str(expr.op))
        return set_type(expr, infer_type(Apply(Apply(op, left), right), ctx))
    if isinstance(expr, Where):
        assert isinstance(expr.binding, Assign)
        name, value, body = expr.binding.name.name, expr.binding.value, expr.body
        if isinstance(value, (Function, MatchFunction)):
            # Letrec
            func_ty: MonoType = fresh_tyvar()
            value_ty = infer_type(value, {**ctx, name: Forall([], func_ty)})
        else:
            # Let
            value_ty = infer_type(value, ctx)
        value_scheme = generalize(value_ty, ctx)
        body_ty = infer_type(body, {**ctx, name: value_scheme})
        return set_type(expr, body_ty)
    if isinstance(expr, List):
        list_item_ty = fresh_tyvar()
        for item in expr.items:
            assert not isinstance(item, Spread), "Spread can only occur in list match (for now)"
            item_ty = infer_type(item, ctx)
            unify_type(list_item_ty, item_ty)
        return set_type(expr, list_type(list_item_ty))
    if isinstance(expr, MatchCase):
        pattern_ctx: Context = {}
        pattern_ty = infer_pattern_type(expr.pattern, pattern_ctx)
        body_ty = infer_type(expr.body, {**ctx, **pattern_ctx})
        return set_type(expr, func_type(pattern_ty, body_ty))
    if isinstance(expr, Apply):
        func_ty = infer_type(expr.func, ctx)
        arg_ty = infer_type(expr.arg, ctx)
        result = fresh_tyvar()
        unify_type(func_ty, func_type(arg_ty, result))
        return set_type(expr, result)
    if isinstance(expr, MatchFunction):
        result = fresh_tyvar()
        for case in expr.cases:
            case_ty = infer_type(case, ctx)
            unify_type(result, case_ty)
        return set_type(expr, result)
    if isinstance(expr, Record):
        fields = {}
        rest: TyVar | TyEmptyRow = TyEmptyRow()
        for key, value in expr.data.items():
            assert not isinstance(value, Spread), "Spread can only occur in record match (for now)"
            fields[key] = infer_type(value, ctx)
        return set_type(expr, TyRow(fields, rest))
    if isinstance(expr, Access):
        obj_ty = infer_type(expr.obj, ctx)
        value_ty = fresh_tyvar()
        assert isinstance(expr.at, Var)
        # "has field" constraint in the form of an open row
        unify_type(obj_ty, TyRow({expr.at.name: value_ty}, fresh_tyvar()))
        return value_ty
    raise InferenceError(f"Unexpected type {type(expr)}")


def minimize(ty: MonoType) -> MonoType:
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    free = ftv_ty(ty)
    subst = {ftv: TyVar(next(letters)) for ftv in sorted(free)}
    return apply_ty(ty, subst)


Number = typing.Union[int, float]


class Repr(typing.Protocol):
    def __call__(self, obj: Object, prec: Number = 0) -> str: ...


# Can't use reprlib.recursive_repr because it doesn't work if the print
# function has more than one argument (for example, prec)
def handle_recursion(func: Repr) -> Repr:
    cache: typing.List[Object] = []

    @functools.wraps(func)
    def wrapper(obj: Object, prec: Number = 0) -> str:
        for cached in cache:
            if obj is cached:
                return "..."
        cache.append(obj)
        result = func(obj, prec)
        cache.remove(obj)
        return result

    return wrapper


@handle_recursion
def pretty(obj: Object, prec: Number = 0) -> str:
    if isinstance(obj, Int):
        return str(obj.value)
    if isinstance(obj, Float):
        return str(obj.value)
    if isinstance(obj, String):
        return json.dumps(obj.value)
    if isinstance(obj, Bytes):
        return f"~~{base64.b64encode(obj.value).decode()}"
    if isinstance(obj, Var):
        return obj.name
    if isinstance(obj, Hole):
        return "()"
    if isinstance(obj, Spread):
        return f"...{obj.name}" if obj.name else "..."
    if isinstance(obj, List):
        return f"[{', '.join(pretty(item) for item in obj.items)}]"
    if isinstance(obj, Record):
        return f"{{{', '.join(f'{key} = {pretty(value)}' for key, value in obj.data.items())}}}"
    if isinstance(obj, Closure):
        keys = list(obj.env.keys())
        return f"Closure({keys}, {pretty(obj.func)})"
    if isinstance(obj, EnvObject):
        return f"EnvObject({repr(obj.env)})"
    if isinstance(obj, NativeFunction):
        return f"NativeFunction(name={obj.name})"
    if isinstance(obj, Relocation):
        return f"Relocation(name={repr(obj.name)})"
    if isinstance(obj, Variant):
        op_prec = PS["#"]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"#{obj.tag} {pretty(obj.value, right_prec)}"
    if isinstance(obj, Assign):
        op_prec = PS["="]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.name, left_prec)} = {pretty(obj.value, right_prec)}"
    if isinstance(obj, Binop):
        op_prec = PS[BinopKind.to_str(obj.op)]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.left, left_prec)} {BinopKind.to_str(obj.op)} {pretty(obj.right, right_prec)}"
    if isinstance(obj, Function):
        op_prec = PS["->"]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        assert isinstance(obj.arg, Var)
        result = f"{obj.arg.name} -> {pretty(obj.body, right_prec)}"
    if isinstance(obj, MatchFunction):
        op_prec = PS["|"]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = "\n".join(
            f"| {pretty(case.pattern, left_prec)} -> {pretty(case.body, right_prec)}" for case in obj.cases
        )
    if isinstance(obj, Where):
        op_prec = PS["."]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.body, left_prec)} . {pretty(obj.binding, right_prec)}"
    if isinstance(obj, Assert):
        op_prec = PS["!"]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.value, left_prec)} ! {pretty(obj.cond, right_prec)}"
    if isinstance(obj, Apply):
        op_prec = PS[""]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.func, left_prec)} {pretty(obj.arg, right_prec)}"
    if isinstance(obj, Access):
        op_prec = PS["@"]
        left_prec, right_prec = op_prec.pl, op_prec.pr
        result = f"{pretty(obj.obj, left_prec)} @ {pretty(obj.at, right_prec)}"
    if prec >= op_prec.pl:
        return f"({result})"
    return result


def fetch(url: Object) -> Object:
    if not isinstance(url, String):
        raise TypeError(f"fetch expected String, but got {type(url).__name__}")
    with urllib.request.urlopen(url.value) as f:
        return String(f.read().decode("utf-8"))


def make_object(pyobj: object) -> Object:
    assert not isinstance(pyobj, Object)
    if isinstance(pyobj, int):
        return Int(pyobj)
    if isinstance(pyobj, str):
        return String(pyobj)
    if isinstance(pyobj, list):
        return List([make_object(o) for o in pyobj])
    if isinstance(pyobj, dict):
        # Assumed to only be called with JSON, so string keys.
        return Record({key: make_object(value) for key, value in pyobj.items()})
    raise NotImplementedError(type(pyobj))


def jsondecode(obj: Object) -> Object:
    if not isinstance(obj, String):
        raise TypeError(f"jsondecode expected String, but got {type(obj).__name__}")
    data = json.loads(obj.value)
    return make_object(data)


def listlength(obj: Object) -> Object:
    # TODO(max): Implement in scrapscript once list pattern matching is
    # implemented.
    if not isinstance(obj, List):
        raise TypeError(f"listlength expected List, but got {type(obj).__name__}")
    return Int(len(obj.items))


def serialize(obj: Object) -> bytes:
    serializer = Serializer()
    serializer.serialize(obj)
    return bytes(serializer.output)


def deserialize(data: bytes) -> Object:
    deserializer = Deserializer(data)
    return deserializer.parse()


def deserialize_object(obj: Object) -> Object:
    assert isinstance(obj, Bytes)
    return deserialize(obj.value)


STDLIB = {
    "$$add": Closure({}, Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))),
    "$$fetch": NativeFunction("$$fetch", fetch),
    "$$jsondecode": NativeFunction("$$jsondecode", jsondecode),
    "$$serialize": NativeFunction("$$serialize", lambda obj: Bytes(serialize(obj))),
    "$$deserialize": NativeFunction("$$deserialize", deserialize_object),
    "$$listlength": NativeFunction("$$listlength", listlength),
}


PRELUDE = """
id = x -> x

. quicksort =
  | [] -> []
  | [p, ...xs] -> (concat ((quicksort (ltp xs p)) +< p) (quicksort (gtp xs p))
    . gtp = xs -> p -> filter (x -> x >= p) xs
    . ltp = xs -> p -> filter (x -> x < p) xs)

. filter = f ->
  | [] -> []
  | [x, ...xs] -> f x |> | #true () -> x >+ filter f xs
                         | #false () -> filter f xs

. concat = xs ->
  | [] -> xs
  | [y, ...ys] -> concat (xs +< y) ys

. map = f ->
  | [] -> []
  | [x, ...xs] -> f x >+ map f xs

. range =
  | 0 -> []
  | i -> range (i - 1) +< (i - 1)

. foldr = f -> a ->
  | [] -> a
  | [x, ...xs] -> f x (foldr f a xs)

. take =
  | 0 -> xs -> []
  | n ->
    | [] -> []
    | [x, ...xs] -> x >+ take (n - 1) xs

. all = f ->
  | [] -> #true ()
  | [x, ...xs] -> f x && all f xs

. any = f ->
  | [] -> #false ()
  | [x, ...xs] -> f x || any f xs
"""


def boot_env() -> Env:
    env_object = eval_exp(STDLIB, parse(tokenize(PRELUDE)))
    assert isinstance(env_object, EnvObject)
    return env_object.env


class Completer:
    def __init__(self, env: Env) -> None:
        self.env: Env = env
        self.matches: typing.List[str] = []

    def complete(self, text: str, state: int) -> Optional[str]:
        assert "@" not in text, "TODO: handle attr/index access"
        if state == 0:
            options = sorted(self.env.keys())
            if not text:
                self.matches = options[:]
            else:
                self.matches = [key for key in options if key.startswith(text)]
        try:
            return self.matches[state]
        except IndexError:
            return None


REPL_HISTFILE = os.path.expanduser(".scrap-history")


class ScrapRepl(code.InteractiveConsole):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.env: Env = boot_env()

    def enable_readline(self) -> None:
        assert readline, "Can't enable readline without readline module"
        if os.path.exists(REPL_HISTFILE):
            readline.read_history_file(REPL_HISTFILE)
        # what determines the end of a word; need to set so $ can be part of a
        # variable name
        readline.set_completer_delims(" \t\n;")
        # TODO(max): Add completion per scope, not just for global environment.
        readline.set_completer(Completer(self.env).complete)
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("tab: menu-complete")

    def finish_readline(self) -> None:
        assert readline, "Can't finish readline without readline module"
        histfile_size = 1000
        readline.set_history_length(histfile_size)
        readline.write_history_file(REPL_HISTFILE)

    def runsource(self, source: str, filename: str = "<input>", symbol: str = "single") -> bool:
        try:
            tokens = tokenize(source)
            logger.debug("Tokens: %s", tokens)
            ast = parse(tokens)
            if isinstance(ast, MatchFunction) and not source.endswith("\n"):
                # User might be in the middle of typing a multi-line match...
                # wait for them to hit Enter once after the last case
                return True
            logger.debug("AST: %s", ast)
            result = eval_exp(self.env, ast)
            assert isinstance(self.env, dict)  # for .update()/__setitem__
            if isinstance(result, EnvObject):
                self.env.update(result.env)
            else:
                self.env["_"] = result
                print(pretty(result))
        except UnexpectedEOFError:
            # Need to read more text
            return True
        except ParseError as e:
            print(f"Parse error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
        return False


def eval_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    program = args.program_file.read()
    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval_exp(boot_env(), ast)
    print(pretty(result))


def check_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    program = args.program_file.read()
    tokens = tokenize(program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = infer_type(ast, OP_ENV)
    result = minimize(result)
    print(result)


def apply_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    tokens = tokenize(args.program)
    logger.debug("Tokens: %s", tokens)
    ast = parse(tokens)
    logger.debug("AST: %s", ast)
    result = eval_exp(boot_env(), ast)
    print(pretty(result))


def repl_command(args: argparse.Namespace) -> None:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    repl = ScrapRepl()
    if readline:
        repl.enable_readline()
    repl.interact(banner="")
    if readline:
        repl.finish_readline()


def env_get_split(key: str, default: Optional[typing.List[str]] = None) -> typing.List[str]:
    import shlex

    cflags = os.environ.get(key)
    if cflags:
        return shlex.split(cflags)
    if default:
        return default
    return []


def discover_cflags(cc: typing.List[str], debug: bool = True) -> typing.List[str]:
    default_cflags = ["-Wall", "-Wextra", "-fno-strict-aliasing", "-Wno-unused-function"]
    # -fno-strict-aliasing is needed because we do pointer casting a bunch
    # -Wno-unused-function is needed because we have a bunch of unused
    # functions depending on what code is compiled
    if debug:
        default_cflags += ["-O0", "-ggdb"]
    else:
        default_cflags += ["-O2", "-DNDEBUG"]
        if "cosmo" not in cc[0]:
            # cosmocc does not support LTO
            default_cflags.append("-flto")
    if "mingw" in cc[0]:
        # Windows does not support mmap
        default_cflags.append("-DSTATIC_HEAP")
    return env_get_split("CFLAGS", default_cflags)


OP_ENV = {
    "+": Forall([], func_type(IntType, IntType, IntType)),
    "-": Forall([], func_type(IntType, IntType, IntType)),
    "*": Forall([], func_type(IntType, IntType, IntType)),
    "/": Forall([], func_type(IntType, IntType, FloatType)),
    "++": Forall([], func_type(StringType, StringType, StringType)),
    ">+": Forall([TyVar("a")], func_type(TyVar("a"), list_type(TyVar("a")), list_type(TyVar("a")))),
    "+<": Forall([TyVar("a")], func_type(list_type(TyVar("a")), TyVar("a"), list_type(TyVar("a")))),
}


def compile_command(args: argparse.Namespace) -> None:
    if args.run:
        args.compile = True
    from compiler import compile_to_string

    with open(args.file, "r") as f:
        source = f.read()

    program = parse(tokenize(source))
    if args.check:
        infer_type(program, OP_ENV)
    c_program = compile_to_string(program, args.debug)

    with open(args.platform, "r") as f:
        platform = f.read()

    with open(args.output, "w") as f:
        f.write(c_program)
        f.write(platform)

    if args.format:
        import subprocess

        subprocess.run(["clang-format-15", "-i", args.output], check=True)

    if args.compile:
        import subprocess

        cc = env_get_split("CC", ["clang"])
        cflags = discover_cflags(cc, args.debug)
        if args.memory:
            cflags += [f"-DMEMORY_SIZE={args.memory}"]
        ldflags = env_get_split("LDFLAGS")
        subprocess.run([*cc, "-o", "a.out", *cflags, args.output, *ldflags], check=True)

    if args.run:
        import subprocess

        subprocess.run(["sh", "-c", "./a.out"], check=True)


def flat_command(args: argparse.Namespace) -> None:
    prog = parse(tokenize(sys.stdin.read()))
    serializer = Serializer()
    serializer.serialize(prog)
    sys.stdout.buffer.write(serializer.output)


def server_command(args: argparse.Namespace) -> None:
    import http.server
    import socketserver
    import hashlib

    dir = os.path.abspath(args.directory)
    if not os.path.isdir(dir):
        print(f"Error: {dir} is not a valid directory")
        sys.exit(1)

    scraps = {}
    for root, _, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir)
            if file.startswith("$"):
                logger.debug(f"Skipping {rel_path}")
                continue
            rel_path_without_ext = os.path.splitext(rel_path)[0]
            with open(file_path, "r") as f:
                try:
                    program = parse(tokenize(f.read()))
                    serializer = Serializer()
                    serializer.serialize(program)
                    serialized = bytes(serializer.output)
                    scraps[rel_path_without_ext] = serialized
                    logger.debug(f"Loaded {rel_path_without_ext}")
                    file_hash = hashlib.sha256(serialized).hexdigest()
                    scraps[f"${file_hash}"] = serialized
                    logger.debug(f"Loaded {rel_path_without_ext} as ${file_hash}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

    keep_serving = True

    class ScrapHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_QUIT(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Quitting")
            nonlocal keep_serving
            keep_serving = False

        def do_GET(self) -> None:
            path = self.path.lstrip("/")
            scrap = scraps.get(path)
            if scrap is not None:
                self.send_response(200)
                self.send_header("Content-Type", "application/scrap; charset=binary")
                self.send_header("Content-Disposition", f"attachment; filename={json.dumps(f'{path}.scrap')}")
                self.send_header("Content-Length", str(len(scrap)))
                self.end_headers()
                self.wfile.write(scrap)
            else:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"File not found")

    handler = ScrapHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        logger.info(f"Serving {dir} at http://{args.host}:{args.port}")
        while keep_serving:
            httpd.handle_request()


def main() -> None:
    parser = argparse.ArgumentParser(prog="scrapscript")
    subparsers = parser.add_subparsers(dest="command")

    repl = subparsers.add_parser("repl")
    repl.set_defaults(func=repl_command)
    repl.add_argument("--debug", action="store_true")

    eval_ = subparsers.add_parser("eval")
    eval_.set_defaults(func=eval_command)
    eval_.add_argument("program_file", type=argparse.FileType("r"))
    eval_.add_argument("--debug", action="store_true")

    check = subparsers.add_parser("check")
    check.set_defaults(func=check_command)
    check.add_argument("program_file", type=argparse.FileType("r"))
    check.add_argument("--debug", action="store_true")

    apply = subparsers.add_parser("apply")
    apply.set_defaults(func=apply_command)
    apply.add_argument("program")
    apply.add_argument("--debug", action="store_true")

    comp = subparsers.add_parser("compile")
    comp.set_defaults(func=compile_command)
    comp.add_argument("file")
    comp.add_argument("-o", "--output", default="output.c")
    comp.add_argument("--format", action="store_true")
    comp.add_argument("--compile", action="store_true")
    comp.add_argument("--memory", type=int)
    comp.add_argument("--run", action="store_true")
    comp.add_argument("--debug", action="store_true", default=False)
    comp.add_argument("--check", action="store_true", default=False)
    # The platform is in the same directory as this file
    comp.add_argument("--platform", default=os.path.join(os.path.dirname(__file__), "cli.c"))

    flat = subparsers.add_parser("flat")
    flat.set_defaults(func=flat_command)

    yard = subparsers.add_parser("yard")
    yard.set_defaults(func=lambda _: yard.print_help())
    yard_subparsers = yard.add_subparsers(dest="yard_command")

    yard_server = yard_subparsers.add_parser("server")
    yard_server.set_defaults(func=server_command)
    yard_server.add_argument("directory", type=str, nargs="?", default=".", help="Directory to serve")
    yard_server.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    yard_server.add_argument("--port", type=int, default=8080, help="Port to listen on")

    args = parser.parse_args()
    if not args.command:
        args.debug = False
        repl_command(args)
    else:
        args.func(args)


if __name__ == "__main__":
    # This is so that we can use scrapscript.py as a main but also import
    # things from `scrapscript` and not have that be a separate module.
    sys.modules["scrapscript"] = sys.modules[__name__]
    main()
