#!/usr/bin/env python3.10
import logging
import re
from typing import Optional

import pytest

# pylint: disable=redefined-builtin
from scrapscript.lib.scrapscript import (
    Apply,
    Assert,
    Assign,
    Binop,
    BinopKind,
    Bool,
    Bytes,
    Env,
    EnvObject,
    Function,
    Hole,
    Int,
    List,
    Object,
    ParseError,
    String,
    Var,
    Where,
    eval,
    parse,
    tokenize,
)

logger = logging.getLogger(__name__)


class TestTokenizer:
    def test_tokenize_digit(self) -> None:
        assert tokenize("1") == ["1"]

    def test_tokenize_multiple_digits(self) -> None:
        assert tokenize("123") == ["123"]

    def test_tokenize_negative_int(self) -> None:
        assert tokenize("-123") == ["-123"]

    def test_tokenize_binop(self) -> None:
        assert tokenize("1 + 2") == ["1", "+", "2"]

    def test_tokenize_binop_no_spaces(self) -> None:
        assert tokenize("1+2") == ["1", "+", "2"]

    @pytest.mark.parametrize("op", ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++"])
    def test_tokenize_binop_var(self, op: str) -> None:
        assert tokenize(f"a {op} b") == ["a", op, "b"]
        assert tokenize(f"a{op}b") == ["a", op, "b"]

    def test_tokenize_var(self) -> None:
        assert tokenize("abc") == ["abc"]

    def test_tokenize_dollar_sha1_var(self) -> None:
        assert tokenize("$sha1'foo") == ["$sha1'foo"]

    def test_tokenize_dollar_dollar_var(self) -> None:
        assert tokenize("$$bills") == ["$$bills"]

    def test_ignore_whitespace(self) -> None:
        assert tokenize("1\n+\t2") == ["1", "+", "2"]

    def test_ignore_line_comment(self) -> None:
        assert tokenize("-- 1\n2") == ["2"]

    def test_tokenize_string(self) -> None:
        assert tokenize('"hello"') == ['"hello"']

    def test_tokenize_string_with_spaces(self) -> None:
        assert tokenize('"hello world"') == ['"hello world"']

    def test_tokenize_string_missing_end_quote_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected EOF while reading string")):
            tokenize('"hello')

    def test_tokenize_with_trailing_whitespace(self) -> None:
        assert tokenize("- ") == ["-"]
        assert tokenize("-- ") == []
        assert tokenize("+ ") == ["+"]
        assert tokenize("123 ") == ["123"]
        assert tokenize("abc ") == ["abc"]
        assert tokenize("[ ") == ["["]
        assert tokenize("] ") == ["]"]

    def test_tokenize_empty_list(self) -> None:
        assert tokenize("[ ]") == ["[", "]"]

    def test_tokenize_list_with_items(self) -> None:
        assert tokenize("[ 1 , 2 ]") == ["[", "1", ",", "2", "]"]

    def test_tokenize_list_with_no_spaces(self) -> None:
        assert tokenize("[1,2]") == ["[", "1", ",", "2", "]"]

    def test_tokenize_function(self) -> None:
        assert tokenize("a -> b -> a + b") == ["a", "->", "b", "->", "a", "+", "b"]

    def test_tokenize_function_with_no_spaces(self) -> None:
        assert tokenize("a->b->a+b") == ["a", "->", "b", "->", "a", "+", "b"]

    def test_tokenize_where(self) -> None:
        assert tokenize("a . b") == ["a", ".", "b"]

    def test_tokenize_assert(self) -> None:
        assert tokenize("a ? b") == ["a", "?", "b"]

    def test_tokenize_hastype(self) -> None:
        assert tokenize("a : b") == ["a", ":", "b"]

    def test_tokenize_minus_returns_minus(self) -> None:
        assert tokenize("-") == ["-"]

    def test_tokenize_tilde_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token '~'")):
            tokenize("~")

    def test_tokenize_tilde_tilde_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected EOF while reading bytes")):
            tokenize("~~")

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token '~'")):
            tokenize("~=")

    def test_tokenize_tilde_tilde_equals_returns_empty_bytes(self) -> None:
        assert tokenize("~~=") == ["~~"]

    def test_tokenize_bytes_returns_bytes(self) -> None:
        assert tokenize("~~QUJD=") == ["~~QUJD"]

    def test_tokenize_hole(self) -> None:
        assert tokenize("()") == ["()"]

    def test_tokenize_pipe(self) -> None:
        assert tokenize("1 |> f . f = a -> a + 1") == ["1", "|>", "f", ".", "f", "=", "a", "->", "a", "+", "1"]


class TestParser:
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected end of input")):
            parse([])

    def test_parse_digit_returns_int(self) -> None:
        assert parse(["1"]) == Int(1)

    def test_parse_digits_returns_int(self) -> None:
        assert parse(["123"]) == Int(123)

    def test_parse_negative_int_returns_int(self) -> None:
        assert parse(["-123"]) == Int(-123)

    def test_parse_var_returns_var(self) -> None:
        assert parse(["abc_123"]) == Var("abc_123")

    def test_parse_sha_var_returns_var(self) -> None:
        assert parse(["$sha1'abc"]) == Var("$sha1'abc")

    def test_parse_sha_var_without_quote_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token")):
            parse(["$sha1abc"])

    def test_parse_dollar_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token")):
            parse(["$"])

    def test_parse_dollar_dollar_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token")):
            parse(["$$"])

    def test_parse_sha_var_without_dollar_raises_parse_error(self) -> None:
        with pytest.raises(ParseError, match=re.escape("unexpected token")):
            parse(["sha1'abc"])

    def test_parse_dollar_dollar_var_returns_var(self) -> None:
        assert parse(["$$bills"]) == Var("$$bills")

    def test_parse_bytes_returns_bytes(self) -> None:
        assert parse(["~~QUJD"]) == Bytes(b"ABC")

    def test_parse_binary_add_returns_binop(self) -> None:
        assert parse(["1", "+", "2"]) == Binop(BinopKind.ADD, Int(1), Int(2))

    def test_parse_binary_add_right_returns_binop(self) -> None:
        assert parse(["1", "+", "2", "+", "3"]) == Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3)))

    def test_mul_binds_tighter_than_add_right(self) -> None:
        assert parse(["1", "+", "2", "*", "3"]) == Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3)))

    def test_mul_binds_tighter_than_add_left(self) -> None:
        assert parse(["1", "*", "2", "+", "3"]) == Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3))

    def test_parse_binary_concat_returns_binop(self) -> None:
        assert parse(['"abc"', "++", '"def"']) == Binop(BinopKind.CONCAT, String("abc"), String("def"))

    @pytest.mark.parametrize("op", ["+", "-", "*", "/", "==", "/=", "<", ">", "<=", ">=", "++"])
    def test_parse_binary_op_returns_binop(self, op: str) -> None:
        kind = BinopKind.from_str(op)
        assert parse(["a", op, "b"]) == Binop(kind, Var("a"), Var("b"))

    def test_parse_empty_list(self) -> None:
        assert parse(["[", "]"]) == List([])

    def test_parse_list_of_ints_returns_list(self) -> None:
        assert parse(["[", "1", ",", "2", "]"]) == List([Int(1), Int(2)])

    def test_parse_assign(self) -> None:
        assert parse(["a", "=", "1"]) == Assign(Var("a"), Int(1))

    def test_parse_function_one_arg_returns_function(self) -> None:
        assert parse(["a", "->", "a", "+", "1"]) == Function(Var("a"), Binop(BinopKind.ADD, Var("a"), Int(1)))

    def test_parse_function_two_args_returns_functions(self) -> None:
        assert parse(["a", "->", "b", "->", "a", "+", "b"]) == Function(
            Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))
        )

    def test_parse_assign_function(self) -> None:
        assert parse(["id", "=", "x", "->", "x"]) == Assign(Var("id"), Function(Var("x"), Var("x")))

    def test_parse_function_application_one_arg(self) -> None:
        assert parse(["f", "a"]) == Apply(Var("f"), Var("a"))

    def test_parse_function_application_two_args(self) -> None:
        assert parse(["f", "a", "b"]) == Apply(Apply(Var("f"), Var("a")), Var("b"))

    def test_parse_where(self) -> None:
        assert parse(["a", ".", "b"]) == Where(Var("a"), Var("b"))

    def test_parse_nested_where(self) -> None:
        assert parse(["a", ".", "b", ".", "c"]) == Where(Where(Var("a"), Var("b")), Var("c"))

    def test_parse_assert(self) -> None:
        assert parse(["a", "?", "b"]) == Assert(Var("a"), Var("b"))

    def test_parse_nested_assert(self) -> None:
        assert parse(["a", "?", "b", "?", "c"]) == Assert(Assert(Var("a"), Var("b")), Var("c"))

    def test_parse_mixed_assert_where(self) -> None:
        assert parse(["a", "?", "b", ".", "c"]) == Where(Assert(Var("a"), Var("b")), Var("c"))

    def test_parse_hastype(self) -> None:
        assert parse(["a", ":", "b"]) == Binop(BinopKind.HASTYPE, Var("a"), Var("b"))

    def test_parse_hole(self) -> None:
        assert parse(["()"]) == Hole()

    def test_parse_pipe(self) -> None:
        assert parse(["1", "|>", "f", ".", "f", "=", "a", "->", "a", "+", "1"]) == Where(
            first=Apply(func=Var(name="f"), arg=Int(value=1)),
            second=Assign(
                name=Var(name="f"),
                value=Function(arg=Var(name="a"), body=Binop(op=BinopKind.ADD, left=Var(name="a"), right=Int(value=1))),
            ),
        )


class TestEval:
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        assert eval({}, exp) == Int(5)

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        assert eval({}, exp) == String("xyz")

    def test_eval_bytes_returns_bytes(self) -> None:
        exp = Bytes(b"xyz")
        assert eval({}, exp) == Bytes(b"xyz")

    def test_eval_true_returns_true(self) -> None:
        assert eval({}, Bool(True)) == Bool(True)

    def test_eval_false_returns_false(self) -> None:
        assert eval({}, Bool(False)) == Bool(False)

    def test_eval_with_non_existent_var_raises_name_error(self) -> None:
        exp = Var("no")
        with pytest.raises(NameError, match=re.escape("name 'no' is not defined")):
            eval({}, exp)

    def test_eval_with_bound_var_returns_value(self) -> None:
        exp = Var("yes")
        env = {"yes": Int(123)}
        assert eval(env, exp) == Int(123)

    def test_eval_with_binop_add_returns_sum(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        assert eval({}, exp) == Int(3)

    def test_eval_with_nested_binop(self) -> None:
        exp = Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        assert eval({}, exp) == Int(6)

    def test_eval_with_binop_add_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), String("hello"))
        with pytest.raises(TypeError, match=re.escape("expected Int, got String")):
            eval({}, exp)

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        assert eval({}, exp) == Int(-1)

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        assert eval({}, exp) == Int(6)

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(2), Int(3))
        assert eval({}, exp) == Int(0)

    def test_eval_with_binop_equal_with_equal_returns_true(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(1))
        assert eval({}, exp) == Bool(True)

    def test_eval_with_binop_equal_with_inequal_returns_false(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(2))
        assert eval({}, exp) == Bool(False)

    def test_eval_with_binop_concat_with_strings_returns_string(self) -> None:
        exp = Binop(BinopKind.CONCAT, String("hello"), String(" world"))
        assert eval({}, exp) == String("hello world")

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.CONCAT, Int(123), String(" world"))
        with pytest.raises(TypeError, match=re.escape("expected String, got Int")):
            eval({}, exp)

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.CONCAT, String(" world"), Int(123))
        with pytest.raises(TypeError, match=re.escape("expected String, got Int")):
            eval({}, exp)

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        assert eval({}, exp) == List([Int(3), Int(7)])

    def test_eval_with_function_returns_function(self) -> None:
        exp = Function(Var("x"), Var("x"))
        assert eval({}, exp) == exp

    def test_eval_assign_returns_env_object(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        result = eval(env, exp)
        assert result == EnvObject({"a": Int(1)})

    def test_eval_assign_does_not_modify_env(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        eval(env, exp)
        assert env == {}

    def test_eval_where_evaluates_in_order(self) -> None:
        exp = Where(Binop(BinopKind.ADD, Var("a"), Int(2)), Assign(Var("a"), Int(1)))
        env: Env = {}
        assert eval(env, exp) == Int(3)
        assert env == {}

    def test_eval_nested_where(self) -> None:
        exp = Where(
            Where(
                Binop(BinopKind.ADD, Var("a"), Var("b")),
                Assign(Var("a"), Int(1)),
            ),
            Assign(Var("b"), Int(2)),
        )
        env: Env = {}
        assert eval(env, exp) == Int(3)
        assert env == {}

    def test_eval_assert_with_truthy_cond_returns_value(self) -> None:
        exp = Assert(Int(123), Bool(True))
        assert eval({}, exp) == Int(123)

    def test_eval_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        exp = Assert(Int(123), Bool(False))
        with pytest.raises(AssertionError, match=re.escape("condition Bool(value=False) failed")):
            eval({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), Bool(True)), Bool(True))
        assert eval({}, exp) == Int(123)

    def test_eval_hole(self) -> None:
        exp = Hole()
        assert eval({}, exp) == Hole()


class TestEndToEnd:
    def _run(self, text: str, env: Optional[Env] = None) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        return eval(env or {}, ast)

    def test_int_returns_int(self) -> None:
        assert self._run("1") == Int(1)

    def test_bytes_returns_bytes(self) -> None:
        assert self._run("~~QUJD=") == Bytes(b"ABC")

    def test_int_add_returns_int(self) -> None:
        assert self._run("1 + 2") == Int(3)

    def test_string_concat_returns_string(self) -> None:
        assert self._run('"abc" ++ "def"') == String("abcdef")

    def test_empty_list(self) -> None:
        assert self._run("[ ]") == List([])

    def test_list_of_ints(self) -> None:
        assert self._run("[ 1 , 2 ]") == List([Int(1), Int(2)])

    def test_list_of_exprs(self) -> None:
        assert self._run("[ 1 + 2 , 3 + 4 ]") == List([Int(3), Int(7)])

    def test_where(self) -> None:
        assert self._run("a + 2 . a = 1") == Int(3)

    def test_nested_where(self) -> None:
        assert self._run("a + b . a = 1 . b = 2") == Int(3)

    def test_assert_with_truthy_cond_returns_value(self) -> None:
        assert self._run("a + 1 ? a == 1 . a = 1") == Int(2)

    def test_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        with pytest.raises(AssertionError, match=re.escape("condition Binop")):
            self._run("a + 1 ? a == 2 . a = 1")

    def test_nested_assert(self) -> None:
        assert self._run("a + b ? a == 1 ? b == 2 . a = 1 . b = 2") == Int(3)

    def test_hole(self) -> None:
        assert self._run("()") == Hole()
