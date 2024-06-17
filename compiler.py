#!/usr/bin/env python3
import dataclasses
import io
import itertools
import json
import os
import typing

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
    parse,
    tokenize,
)

Env = Dict[str, str]


fn_counter = itertools.count()


@dataclasses.dataclass
class CompiledFunction:
    id: int = dataclasses.field(default=0, init=False, compare=False, hash=False)
    name: str
    params: typing.List[str]
    fields: typing.List[str] = dataclasses.field(default_factory=list)
    code: typing.List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = next(fn_counter)
        self.code.append("HANDLES();")
        for param in self.params:
            # The parameters are raw pointers and must be updated on GC
            self.code.append(f"GC_PROTECT({param});")

    def decl(self) -> str:
        args = ", ".join(f"struct object* {arg}" for arg in self.params)
        return f"struct object* {self.name}({args})"


class Compiler:
    def __init__(self, main_fn: CompiledFunction) -> None:
        self.gensym_counter: int = 0
        self.functions: typing.List[CompiledFunction] = [main_fn]
        self.function: CompiledFunction = main_fn
        self.record_keys: Dict[str, int] = {}
        self.record_builders: Dict[Tuple[str, ...], CompiledFunction] = {}
        self.variant_tags: Dict[str, int] = {}
        self.debug: bool = False

    def record_key(self, key: str) -> str:
        if key not in self.record_keys:
            self.record_keys[key] = len(self.record_keys)
        return f"Record_{key}"

    def record_builder(self, keys: Tuple[str, ...]) -> CompiledFunction:
        builder = self.record_builders.get(keys)
        if builder is not None:
            return builder

        builder = CompiledFunction(f"Record_builder_{'_'.join(keys)}", list(keys))
        self.functions.append(builder)
        cur = self.function
        self.function = builder

        result = self._mktemp(f"mkrecord(heap, {len(keys)})")
        for i, key in enumerate(keys):
            key_idx = self.record_key(key)
            self._emit(f"record_set({result}, /*index=*/{i}, (struct record_field){{.key={key_idx}, .value={key}}});")
        self._debug("collect(heap);")
        self._emit(f"return {result};")

        self.function = cur
        self.record_builders[keys] = builder
        return builder

    def variant_tag(self, key: str) -> int:
        result = self.variant_tags.get(key)
        if result is not None:
            return result
        result = self.variant_tags[key] = len(self.variant_tags)
        return result

    def gensym(self, stem: str = "tmp") -> str:
        self.gensym_counter += 1
        return f"{stem}_{self.gensym_counter-1}"

    def _emit(self, line: str) -> None:
        self.function.code.append(line)

    def _debug(self, line: str) -> None:
        if not self.debug:
            return
        self._emit("#ifndef NDEBUG")
        self._emit(line)
        self._emit("#endif")

    def _handle(self, name: str, exp: str) -> str:
        # TODO(max): Liveness analysis to avoid unnecessary handles
        self._emit(f"OBJECT_HANDLE({name}, {exp});")
        return name

    def _guard(self, cond: str, msg: Optional[str] = None) -> None:
        if msg is None:
            msg = f"assertion {cond!s} failed"
        self._emit(f"if (!({cond})) {{")
        self._emit(f'fprintf(stderr, "{msg}\\n");')
        self._emit("abort();")
        self._emit("}")

    def _mktemp(self, exp: str) -> str:
        temp = self.gensym()
        return self._handle(temp, exp)

    def compile_assign(self, env: Env, exp: Assign) -> Env:
        assert isinstance(exp.name, Var)
        name = exp.name.name
        if isinstance(exp.value, Function):
            # Named function
            value = self.compile_function(env, exp.value, name)
            return {**env, name: value}
        if isinstance(exp.value, MatchFunction):
            # Named match function
            value = self.compile_match_function(env, exp.value, name)
            return {**env, name: value}
        value = self.compile(env, exp.value)
        return {**env, name: value}

    def make_compiled_function(self, arg: str, exp: Object, name: Optional[str]) -> CompiledFunction:
        assert isinstance(exp, (Function, MatchFunction))
        free = free_in(exp)
        if name is not None and name in free:
            free.remove(name)
        fields = sorted(free)
        fn_name = self.gensym(name if name else "fn")  # must be globally unique
        return CompiledFunction(fn_name, params=["this", arg], fields=fields)

    def compile_function_env(self, fn: CompiledFunction, name: Optional[str]) -> Env:
        result = {param: param for param in fn.params}
        if name is not None:
            result[name] = "this"
        for i, field in enumerate(fn.fields):
            result[field] = self._mktemp(f"closure_get(this, /*{field}=*/{i})")
        return result

    def compile_function(self, env: Env, exp: Function, name: Optional[str]) -> str:
        assert isinstance(exp.arg, Var)
        fn = self.make_compiled_function(exp.arg.name, exp, name)
        self.functions.append(fn)
        cur = self.function
        self.function = fn
        funcenv = self.compile_function_env(fn, name)
        val = self.compile(funcenv, exp.body)
        fn.code.append(f"return {val};")
        self.function = cur
        return self.make_closure(env, fn)

    def try_match(self, env: Env, arg: str, pattern: Object, fallthrough: str) -> Env:
        if isinstance(pattern, Int):
            self._emit(f"if (!is_num_equal_word({arg}, {pattern.value})) {{ goto {fallthrough}; }}")
            return {}
        if isinstance(pattern, Hole):
            self._emit(f"if (!is_hole({arg})) {{ goto {fallthrough}; }}")
            return {}
        if isinstance(pattern, Variant):
            self.variant_tag(pattern.tag)  # register it for the big enum
            self._emit(f"if (!is_variant({arg})) {{ goto {fallthrough}; }}")
            self._emit(f"if (variant_tag({arg}) != Tag_{pattern.tag}) {{ goto {fallthrough}; }}")
            return self.try_match(env, self._mktemp(f"variant_value({arg})"), pattern.value, fallthrough)

        if isinstance(pattern, String):
            value = pattern.value
            if len(value) < 8:
                self._emit(f"if ({arg} != mksmallstring({json.dumps(value)}, {len(value)})) {{ goto {fallthrough}; }}")
                return {}
            self._emit(f"if (!is_string({arg})) {{ goto {fallthrough}; }}")
            self._emit(
                f"if (!string_equal_cstr_len({arg}, {json.dumps(value)}, {len(value)})) {{ goto {fallthrough}; }}"
            )
            return {}
        if isinstance(pattern, Var):
            return {pattern.name: arg}
        if isinstance(pattern, List):
            self._emit(f"if (!is_list({arg})) {{ goto {fallthrough}; }}")
            updates = {}
            the_list = arg
            use_spread = False
            for i, pattern_item in enumerate(pattern.items):
                if isinstance(pattern_item, Spread):
                    use_spread = True
                    if pattern_item.name:
                        updates[pattern_item.name] = the_list
                    break
                # Not enough elements
                self._emit(f"if (is_empty_list({the_list})) {{ goto {fallthrough}; }}")
                list_item = self._mktemp(f"list_first({the_list})")
                updates.update(self.try_match(env, list_item, pattern_item, fallthrough))
                the_list = self._mktemp(f"list_rest({the_list})")
            if not use_spread:
                # Too many elements
                self._emit(f"if (!is_empty_list({the_list})) {{ goto {fallthrough}; }}")
            return updates
        if isinstance(pattern, Record):
            self._emit(f"if (!is_record({arg})) {{ goto {fallthrough}; }}")
            updates = {}
            for key, pattern_value in pattern.data.items():
                assert not isinstance(pattern_value, Spread), "record spread not yet supported"
                key_idx = self.record_key(key)
                record_value = self._mktemp(f"record_get({arg}, {key_idx})")
                self._emit(f"if ({record_value} == NULL) {{ goto {fallthrough}; }}")
                updates.update(self.try_match(env, record_value, pattern_value, fallthrough))
            # TODO(max): Check that there are no other fields in the record,
            # perhaps by length check
            return updates
        raise NotImplementedError("try_match", pattern)

    def compile_match_function(self, env: Env, exp: MatchFunction, name: Optional[str]) -> str:
        arg = self.gensym()
        fn = self.make_compiled_function(arg, exp, name)
        self.functions.append(fn)
        cur = self.function
        self.function = fn
        funcenv = self.compile_function_env(fn, name)
        for i, case in enumerate(exp.cases):
            fallthrough = f"case_{i+1}" if i < len(exp.cases) - 1 else "no_match"
            env_updates = self.try_match(funcenv, arg, case.pattern, fallthrough)
            case_result = self.compile({**funcenv, **env_updates}, case.body)
            self._emit(f"return {case_result};")
            self._emit(f"{fallthrough}:;")
        self._emit(r'fprintf(stderr, "no matching cases\n");')
        self._emit("abort();")
        # Pacify the C compiler
        self._emit("return NULL;")
        self.function = cur
        return self.make_closure(env, fn)

    def make_closure(self, env: Env, fn: CompiledFunction) -> str:
        name = self._mktemp(f"mkclosure(heap, {fn.name}, {len(fn.fields)})")
        for i, field in enumerate(fn.fields):
            self._emit(f"closure_set({name}, /*{field}=*/{i}, {env[field]});")
        self._debug("collect(heap);")
        return name

    def compile(self, env: Env, exp: Object) -> str:
        if isinstance(exp, Int):
            # TODO(max): Bignum
            self._debug("collect(heap);")
            return self._mktemp(f"mknum(heap, {exp.value})")
        if isinstance(exp, Hole):
            return self._mktemp("hole()")
        if isinstance(exp, Variant):
            self._debug("collect(heap);")
            self.variant_tag(exp.tag)
            value = self.compile(env, exp.value)
            result = self._mktemp(f"mkvariant(heap, Tag_{exp.tag})")
            self._emit(f"variant_set({result}, {value});")
            return result
        if isinstance(exp, String):
            self._debug("collect(heap);")
            string_repr = json.dumps(exp.value)
            return self._mktemp(f"mkstring(heap, {string_repr}, {len(exp.value)});")
        if isinstance(exp, Binop):
            left = self.compile(env, exp.left)
            right = self.compile(env, exp.right)
            if exp.op == BinopKind.ADD:
                self._debug("collect(heap);")
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_add({left}, {right})")
            if exp.op == BinopKind.MUL:
                self._debug("collect(heap);")
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_mul({left}, {right})")
            if exp.op == BinopKind.SUB:
                self._debug("collect(heap);")
                self._guard(f"is_num({left})")
                self._guard(f"is_num({right})")
                return self._mktemp(f"num_sub({left}, {right})")
            if exp.op == BinopKind.LIST_CONS:
                self._debug("collect(heap);")
                return self._mktemp(f"list_cons({left}, {right})")
            if exp.op == BinopKind.STRING_CONCAT:
                self._debug("collect(heap);")
                self._guard(f"is_string({left})")
                self._guard(f"is_string({right})")
                return self._mktemp(f"string_concat({left}, {right})")
            raise NotImplementedError(f"binop {exp.op}")
        if isinstance(exp, Where):
            assert isinstance(exp.binding, Assign)
            res_env = self.compile_assign(env, exp.binding)
            new_env = {**env, **res_env}
            return self.compile(new_env, exp.body)
        if isinstance(exp, Var):
            var_value = env.get(exp.name)
            if var_value is None:
                raise NameError(f"name '{exp.name}' is not defined")
            return var_value
        if isinstance(exp, Apply):
            callee = self.compile(env, exp.func)
            arg = self.compile(env, exp.arg)
            return self._mktemp(f"closure_call({callee}, {arg})")
        if isinstance(exp, List):
            items = [self.compile(env, item) for item in exp.items]
            result = self._mktemp("empty_list()")
            for item in reversed(items):
                result = self._mktemp(f"list_cons({item}, {result})")
            self._debug("collect(heap);")
            return result
        if isinstance(exp, Record):
            values: Dict[str, str] = {}
            for key, value_exp in exp.data.items():
                values[key] = self.compile(env, value_exp)
            keys = tuple(sorted(exp.data.keys()))
            builder = self.record_builder(keys)
            return self._mktemp(f"{builder.name}({', '.join(values[key] for key in keys)})")
        if isinstance(exp, Access):
            assert isinstance(exp.at, Var), f"only Var access is supported, got {type(exp.at)}"
            record = self.compile(env, exp.obj)
            key_idx = self.record_key(exp.at.name)
            # Check if the record is a record
            self._guard(f"is_record({record})", "not a record")
            value = self._mktemp(f"record_get({record}, {key_idx})")
            self._guard(f"{value} != NULL", f"missing key {exp.at.name!s}")
            return value
        if isinstance(exp, Function):
            # Anonymous function
            return self.compile_function(env, exp, name=None)
        if isinstance(exp, MatchFunction):
            # Anonymous match function
            return self.compile_match_function(env, exp, name=None)
        raise NotImplementedError(f"exp {type(exp)} {exp}")


# TODO(max): Emit constants into the const heap
# The const heap must only point within the const heap
# The const heap will never be scanned
# The const heap can be serialized to disk and mmap'd


def compile_to_string(source: str, debug: bool) -> str:
    program = parse(tokenize(source))

    main_fn = CompiledFunction("scrap_main", params=[])
    compiler = Compiler(main_fn)
    compiler.debug = debug
    result = compiler.compile({}, program)
    main_fn.code.append(f"return {result};")

    f = io.StringIO()
    constants = [
        ("uword", "kKiB", 1024),
        ("uword", "kMiB", "kKiB * kKiB"),
        ("uword", "kGiB", "kKiB * kKiB * kKiB"),
        ("uword", "kPageSize", "4 * kKiB"),
        ("uword", "kSmallIntTagBits", 1),
        ("uword", "kPrimaryTagBits", 3),
        ("uword", "kImmediateTagBits", 5),
        ("uword", "kSmallIntTagMask", "(1 << kSmallIntTagBits) - 1"),
        ("uword", "kPrimaryTagMask", "(1 << kPrimaryTagBits) - 1"),
        ("uword", "kImmediateTagMask", "(1 << kImmediateTagBits) - 1"),
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
        ("uword", "kBitsPerPointer", "kBitsPerByte * kWordSize"),
        ("word", "kSmallIntBits", "kBitsPerPointer - kSmallIntTagBits"),
        ("word", "kSmallIntMinValue", "-(((word)1) << (kSmallIntBits - 1))"),
        ("word", "kSmallIntMaxValue", "(((word)1) << (kSmallIntBits - 1)) - 1"),
    ]
    for type_, name, value in constants:
        print(f"#define {name} ({type_})({value})", file=f)
    # The runtime is in the same directory as this file
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "runtime.c"), "r") as runtime:
        print(runtime.read(), file=f)
    print("#define OBJECT_HANDLE(name, exp) GC_HANDLE(struct object*, name, exp)", file=f)
    if compiler.record_keys:
        print("const char* record_keys[] = {", file=f)
        for key in compiler.record_keys:
            print(f'"{key}",', file=f)
        print("};", file=f)
        print("enum {", file=f)
        for key, idx in compiler.record_keys.items():
            print(f"Record_{key} = {idx},", file=f)
        print("};", file=f)
    else:
        # Pacify the C compiler
        print("const char* record_keys[] = { NULL };", file=f)
    if compiler.variant_tags:
        print("const char* variant_names[] = {", file=f)
        for key in compiler.variant_tags:
            print(f'"{key}",', file=f)
        print("};", file=f)
        print("enum {", file=f)
        for key, idx in compiler.variant_tags.items():
            print(f"Tag_{key} = {idx},", file=f)
        print("};", file=f)
    else:
        # Pacify the C compiler
        print("const char* variant_names[] = { NULL };", file=f)
    # Declare all functions
    for function in compiler.functions:
        print(function.decl() + ";", file=f)
    for function in compiler.functions:
        print(f"{function.decl()} {{", file=f)
        for line in function.code:
            print(line, file=f)
        print("}", file=f)
    return f.getvalue()
