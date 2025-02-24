import base64
import unittest
import re
from typing import Optional

# ruff: noqa: F405
# ruff: noqa: F403
from scrapscript import *


class PeekableTests(unittest.TestCase):
    def test_can_create_peekable(self) -> None:
        Peekable(iter([1, 2, 3]))

    def test_can_iterate_over_peekable(self) -> None:
        sequence = [1, 2, 3]
        for idx, e in enumerate(Peekable(iter(sequence))):
            self.assertEqual(sequence[idx], e)

    def test_peek_next(self) -> None:
        iterator = Peekable(iter([1, 2, 3]))
        self.assertEqual(iterator.peek(), 1)
        self.assertEqual(next(iterator), 1)
        self.assertEqual(iterator.peek(), 2)
        self.assertEqual(next(iterator), 2)
        self.assertEqual(iterator.peek(), 3)
        self.assertEqual(next(iterator), 3)
        with self.assertRaises(StopIteration):
            iterator.peek()
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_can_peek_peekable(self) -> None:
        sequence = [1, 2, 3]
        p = Peekable(iter(sequence))
        self.assertEqual(p.peek(), 1)
        # Ensure we can peek repeatedly
        self.assertEqual(p.peek(), 1)
        for idx, e in enumerate(p):
            self.assertEqual(sequence[idx], e)

    def test_peek_on_empty_peekable_raises_stop_iteration(self) -> None:
        empty = Peekable(iter([]))
        with self.assertRaises(StopIteration):
            empty.peek()

    def test_next_on_empty_peekable_raises_stop_iteration(self) -> None:
        empty = Peekable(iter([]))
        with self.assertRaises(StopIteration):
            next(empty)


class TokenizerTests(unittest.TestCase):
    def test_tokenize_digit(self) -> None:
        self.assertEqual(list(tokenize("1")), [IntLit(1)])

    def test_tokenize_multiple_digits(self) -> None:
        self.assertEqual(list(tokenize("123")), [IntLit(123)])

    def test_tokenize_negative_int(self) -> None:
        self.assertEqual(list(tokenize("-123")), [Operator("-"), IntLit(123)])

    def test_tokenize_float(self) -> None:
        self.assertEqual(list(tokenize("3.14")), [FloatLit(3.14)])

    def test_tokenize_negative_float(self) -> None:
        self.assertEqual(list(tokenize("-3.14")), [Operator("-"), FloatLit(3.14)])

    @unittest.skip("TODO: support floats with no integer part")
    def test_tokenize_float_with_no_integer_part(self) -> None:
        self.assertEqual(list(tokenize(".14")), [FloatLit(0.14)])

    def test_tokenize_float_with_no_decimal_part(self) -> None:
        self.assertEqual(list(tokenize("10.")), [FloatLit(10.0)])

    def test_tokenize_float_with_multiple_decimal_points_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token '.'")):
            list(tokenize("1.0.1"))

    def test_tokenize_binop(self) -> None:
        self.assertEqual(list(tokenize("1 + 2")), [IntLit(1), Operator("+"), IntLit(2)])

    def test_tokenize_binop_no_spaces(self) -> None:
        self.assertEqual(list(tokenize("1+2")), [IntLit(1), Operator("+"), IntLit(2)])

    def test_tokenize_two_oper_chars_returns_two_ops(self) -> None:
        self.assertEqual(list(tokenize(",:")), [Operator(","), Operator(":")])

    def test_tokenize_binary_sub_no_spaces(self) -> None:
        self.assertEqual(list(tokenize("1-2")), [IntLit(1), Operator("-"), IntLit(2)])

    def test_tokenize_binop_var(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                self.assertEqual(list(tokenize(f"a {op} b")), [Name("a"), Operator(op), Name("b")])
                self.assertEqual(list(tokenize(f"a{op}b")), [Name("a"), Operator(op), Name("b")])

    def test_tokenize_var(self) -> None:
        self.assertEqual(list(tokenize("abc")), [Name("abc")])

    @unittest.skip("TODO: make this fail to tokenize")
    def test_tokenize_var_with_quote(self) -> None:
        self.assertEqual(list(tokenize("sha1'abc")), [Name("sha1'abc")])

    def test_tokenize_dollar_sha1_var(self) -> None:
        self.assertEqual(list(tokenize("$sha1'foo")), [Name("$sha1'foo")])

    def test_tokenize_dollar_dollar_var(self) -> None:
        self.assertEqual(list(tokenize("$$bills")), [Name("$$bills")])

    def test_tokenize_dot_dot_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token '..'")):
            list(tokenize(".."))

    def test_tokenize_spread(self) -> None:
        self.assertEqual(list(tokenize("...")), [Operator("...")])

    def test_ignore_whitespace(self) -> None:
        self.assertEqual(list(tokenize("1\n+\t2")), [IntLit(1), Operator("+"), IntLit(2)])

    def test_ignore_line_comment(self) -> None:
        self.assertEqual(list(tokenize("-- 1\n2")), [IntLit(2)])

    def test_tokenize_string(self) -> None:
        self.assertEqual(list(tokenize('"hello"')), [StringLit("hello")])

    def test_tokenize_string_with_spaces(self) -> None:
        self.assertEqual(list(tokenize('"hello world"')), [StringLit("hello world")])

    def test_tokenize_string_missing_end_quote_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(UnexpectedEOFError, "while reading string"):
            list(tokenize('"hello'))

    def test_tokenize_with_trailing_whitespace(self) -> None:
        self.assertEqual(list(tokenize("- ")), [Operator("-")])
        self.assertEqual(list(tokenize("-- ")), [])
        self.assertEqual(list(tokenize("+ ")), [Operator("+")])
        self.assertEqual(list(tokenize("123 ")), [IntLit(123)])
        self.assertEqual(list(tokenize("abc ")), [Name("abc")])
        self.assertEqual(list(tokenize("[ ")), [LeftBracket()])
        self.assertEqual(list(tokenize("] ")), [RightBracket()])

    def test_tokenize_empty_list(self) -> None:
        self.assertEqual(list(tokenize("[ ]")), [LeftBracket(), RightBracket()])

    def test_tokenize_empty_list_with_spaces(self) -> None:
        self.assertEqual(list(tokenize("[ ]")), [LeftBracket(), RightBracket()])

    def test_tokenize_list_with_items(self) -> None:
        self.assertEqual(
            list(tokenize("[ 1 , 2 ]")), [LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()]
        )

    def test_tokenize_list_with_no_spaces(self) -> None:
        self.assertEqual(list(tokenize("[1,2]")), [LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()])

    def test_tokenize_function(self) -> None:
        self.assertEqual(
            list(tokenize("a -> b -> a + b")),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_function_with_no_spaces(self) -> None:
        self.assertEqual(
            list(tokenize("a->b->a+b")),
            [Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")],
        )

    def test_tokenize_where(self) -> None:
        self.assertEqual(list(tokenize("a . b")), [Name("a"), Operator("."), Name("b")])

    def test_tokenize_assert(self) -> None:
        self.assertEqual(list(tokenize("a ? b")), [Name("a"), Operator("?"), Name("b")])

    def test_tokenize_hastype(self) -> None:
        self.assertEqual(list(tokenize("a : b")), [Name("a"), Operator(":"), Name("b")])

    def test_tokenize_minus_returns_minus(self) -> None:
        self.assertEqual(list(tokenize("-")), [Operator("-")])

    def test_tokenize_tilde_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            list(tokenize("~"))

    def test_tokenize_tilde_equals_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token '~'"):
            list(tokenize("~="))

    def test_tokenize_tilde_tilde_returns_empty_bytes(self) -> None:
        self.assertEqual(list(tokenize("~~")), [BytesLit("", 64)])

    def test_tokenize_bytes_returns_bytes_base64(self) -> None:
        self.assertEqual(list(tokenize("~~QUJD")), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base85(self) -> None:
        self.assertEqual(list(tokenize("~~85'K|(_")), [BytesLit("K|(_", 85)])

    def test_tokenize_bytes_base64(self) -> None:
        self.assertEqual(list(tokenize("~~64'QUJD")), [BytesLit("QUJD", 64)])

    def test_tokenize_bytes_base32(self) -> None:
        self.assertEqual(list(tokenize("~~32'IFBEG===")), [BytesLit("IFBEG===", 32)])

    def test_tokenize_bytes_base16(self) -> None:
        self.assertEqual(list(tokenize("~~16'414243")), [BytesLit("414243", 16)])

    def test_tokenize_hole(self) -> None:
        self.assertEqual(list(tokenize("()")), [LeftParen(), RightParen()])

    def test_tokenize_hole_with_spaces(self) -> None:
        self.assertEqual(list(tokenize("( )")), [LeftParen(), RightParen()])

    def test_tokenize_parenthetical_expression(self) -> None:
        self.assertEqual(list(tokenize("(1+2)")), [LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen()])

    def test_tokenize_pipe(self) -> None:
        self.assertEqual(
            list(tokenize("1 |> f . f = a -> a + 1")),
            [
                IntLit(1),
                Operator("|>"),
                Name("f"),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                IntLit(1),
            ],
        )

    def test_tokenize_reverse_pipe(self) -> None:
        self.assertEqual(
            list(tokenize("f <| 1 . f = a -> a + 1")),
            [
                Name("f"),
                Operator("<|"),
                IntLit(1),
                Operator("."),
                Name("f"),
                Operator("="),
                Name("a"),
                Operator("->"),
                Name("a"),
                Operator("+"),
                IntLit(1),
            ],
        )

    def test_tokenize_record_no_fields(self) -> None:
        self.assertEqual(
            list(tokenize("{ }")),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_no_fields_no_spaces(self) -> None:
        self.assertEqual(
            list(tokenize("{}")),
            [LeftBrace(), RightBrace()],
        )

    def test_tokenize_record_one_field(self) -> None:
        self.assertEqual(
            list(tokenize("{ a = 4 }")),
            [LeftBrace(), Name("a"), Operator("="), IntLit(4), RightBrace()],
        )

    def test_tokenize_record_multiple_fields(self) -> None:
        self.assertEqual(
            list(tokenize('{ a = 4, b = "z" }')),
            [
                LeftBrace(),
                Name("a"),
                Operator("="),
                IntLit(4),
                Operator(","),
                Name("b"),
                Operator("="),
                StringLit("z"),
                RightBrace(),
            ],
        )

    def test_tokenize_record_access(self) -> None:
        self.assertEqual(
            list(tokenize("r@a")),
            [Name("r"), Operator("@"), Name("a")],
        )

    def test_tokenize_right_eval(self) -> None:
        self.assertEqual(list(tokenize("a!b")), [Name("a"), Operator("!"), Name("b")])

    def test_tokenize_match(self) -> None:
        self.assertEqual(
            list(tokenize("g = | 1 -> 2 | 2 -> 3")),
            [
                Name("g"),
                Operator("="),
                Operator("|"),
                IntLit(1),
                Operator("->"),
                IntLit(2),
                Operator("|"),
                IntLit(2),
                Operator("->"),
                IntLit(3),
            ],
        )

    def test_tokenize_compose(self) -> None:
        self.assertEqual(
            list(tokenize("f >> g")),
            [Name("f"), Operator(">>"), Name("g")],
        )

    def test_tokenize_compose_reverse(self) -> None:
        self.assertEqual(
            list(tokenize("f << g")),
            [Name("f"), Operator("<<"), Name("g")],
        )

    def test_first_lineno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.lineno, 1)

    def test_first_colno_is_one(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.colno, 1)

    def test_first_line_is_empty(self) -> None:
        l = Lexer("abc")
        self.assertEqual(l.line, "")

    def test_read_char_increments_colno(self) -> None:
        l = Lexer("abc")
        l.read_char()
        self.assertEqual(l.colno, 2)
        self.assertEqual(l.lineno, 1)

    def test_read_newline_increments_lineno(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        l.read_char()
        self.assertEqual(l.lineno, 2)
        self.assertEqual(l.colno, 1)

    def test_read_char_increments_byteno(self) -> None:
        l = Lexer("abc")
        l.read_char()
        self.assertEqual(l.byteno, 1)
        l.read_char()
        self.assertEqual(l.byteno, 2)
        l.read_char()
        self.assertEqual(l.byteno, 3)

    def test_read_char_appends_to_line(self) -> None:
        l = Lexer("ab\nc")
        l.read_char()
        l.read_char()
        self.assertEqual(l.line, "ab")
        l.read_char()
        self.assertEqual(l.line, "")

    def test_read_token_sets_start_and_end_linenos(self) -> None:
        l = Lexer("a b \n c d")
        a = l.read_token()
        b = l.read_token()
        c = l.read_token()
        d = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)

        self.assertEqual(b.source_extent.start.lineno, 1)
        self.assertEqual(b.source_extent.end.lineno, 1)

        self.assertEqual(c.source_extent.start.lineno, 2)
        self.assertEqual(c.source_extent.end.lineno, 2)

        self.assertEqual(d.source_extent.start.lineno, 2)
        self.assertEqual(d.source_extent.end.lineno, 2)

    def test_read_token_sets_source_extents_for_variables(self) -> None:
        l = Lexer("aa bbbb \n ccccc ddddddd")

        a = l.read_token()
        b = l.read_token()
        c = l.read_token()
        d = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)
        self.assertEqual(a.source_extent.start.colno, 1)
        self.assertEqual(a.source_extent.end.colno, 2)
        self.assertEqual(a.source_extent.start.byteno, 0)
        self.assertEqual(a.source_extent.end.byteno, 1)

        self.assertEqual(b.source_extent.start.lineno, 1)
        self.assertEqual(b.source_extent.end.lineno, 1)
        self.assertEqual(b.source_extent.start.colno, 4)
        self.assertEqual(b.source_extent.end.colno, 7)
        self.assertEqual(b.source_extent.start.byteno, 3)
        self.assertEqual(b.source_extent.end.byteno, 6)

        self.assertEqual(c.source_extent.start.lineno, 2)
        self.assertEqual(c.source_extent.end.lineno, 2)
        self.assertEqual(c.source_extent.start.colno, 2)
        self.assertEqual(c.source_extent.end.colno, 6)
        self.assertEqual(c.source_extent.start.byteno, 10)
        self.assertEqual(c.source_extent.end.byteno, 14)

        self.assertEqual(d.source_extent.start.lineno, 2)
        self.assertEqual(d.source_extent.end.lineno, 2)
        self.assertEqual(d.source_extent.start.colno, 8)
        self.assertEqual(d.source_extent.end.colno, 14)
        self.assertEqual(d.source_extent.start.byteno, 16)
        self.assertEqual(d.source_extent.end.byteno, 22)

    def test_read_token_correctly_sets_source_extents_for_variants(self) -> None:
        l = Lexer("# \n\r\n\t abc")

        a = l.read_token()
        b = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)
        self.assertEqual(a.source_extent.start.colno, 1)
        # TODO(max): Should tabs count as one column?
        self.assertEqual(a.source_extent.end.colno, 1)

        self.assertEqual(b.source_extent.start.lineno, 3)
        self.assertEqual(b.source_extent.end.lineno, 3)
        self.assertEqual(b.source_extent.start.colno, 3)
        self.assertEqual(b.source_extent.end.colno, 5)

    def test_read_token_correctly_sets_source_extents_for_strings(self) -> None:
        l = Lexer('"今日は、Maxさん。"')
        a = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)

        self.assertEqual(a.source_extent.start.colno, 1)
        self.assertEqual(a.source_extent.end.colno, 12)

        self.assertEqual(a.source_extent.start.byteno, 0)
        self.assertEqual(a.source_extent.end.byteno, 25)

    def test_read_token_correctly_sets_source_extents_for_byte_literals(self) -> None:
        l = Lexer("~~QUJD ~~85'K|(_ ~~64'QUJD\n ~~32'IFBEG=== ~~16'414243")
        a = l.read_token()
        b = l.read_token()
        c = l.read_token()
        d = l.read_token()
        e = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)
        self.assertEqual(a.source_extent.start.colno, 1)
        self.assertEqual(a.source_extent.end.colno, 6)
        self.assertEqual(a.source_extent.start.byteno, 0)
        self.assertEqual(a.source_extent.end.byteno, 5)

        self.assertEqual(b.source_extent.start.lineno, 1)
        self.assertEqual(b.source_extent.end.lineno, 1)
        self.assertEqual(b.source_extent.start.colno, 8)
        self.assertEqual(b.source_extent.end.colno, 16)
        self.assertEqual(b.source_extent.start.byteno, 7)
        self.assertEqual(b.source_extent.end.byteno, 15)

        self.assertEqual(c.source_extent.start.lineno, 1)
        self.assertEqual(c.source_extent.end.lineno, 1)
        self.assertEqual(c.source_extent.start.colno, 18)
        self.assertEqual(c.source_extent.end.colno, 26)
        self.assertEqual(c.source_extent.start.byteno, 17)
        self.assertEqual(c.source_extent.end.byteno, 25)

        self.assertEqual(d.source_extent.start.lineno, 2)
        self.assertEqual(d.source_extent.end.lineno, 2)
        self.assertEqual(d.source_extent.start.colno, 2)
        self.assertEqual(d.source_extent.end.colno, 14)
        self.assertEqual(d.source_extent.start.byteno, 28)
        self.assertEqual(d.source_extent.end.byteno, 40)

        self.assertEqual(e.source_extent.start.lineno, 2)
        self.assertEqual(e.source_extent.end.lineno, 2)
        self.assertEqual(e.source_extent.start.colno, 16)
        self.assertEqual(e.source_extent.end.colno, 26)
        self.assertEqual(e.source_extent.start.byteno, 42)
        self.assertEqual(e.source_extent.end.byteno, 52)

    def test_read_token_correctly_sets_source_extents_for_numbers(self) -> None:
        l = Lexer("123 123.456")
        a = l.read_token()
        b = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)
        self.assertEqual(a.source_extent.start.colno, 1)
        self.assertEqual(a.source_extent.end.colno, 3)
        self.assertEqual(a.source_extent.start.byteno, 0)
        self.assertEqual(a.source_extent.end.byteno, 2)

        self.assertEqual(b.source_extent.start.lineno, 1)
        self.assertEqual(b.source_extent.end.lineno, 1)
        self.assertEqual(b.source_extent.start.colno, 5)
        self.assertEqual(b.source_extent.end.colno, 11)
        self.assertEqual(b.source_extent.start.byteno, 4)
        self.assertEqual(b.source_extent.end.byteno, 10)

    def test_read_token_correctly_sets_source_extents_for_operators(self) -> None:
        l = Lexer("> >>")
        a = l.read_token()
        b = l.read_token()

        self.assertEqual(a.source_extent.start.lineno, 1)
        self.assertEqual(a.source_extent.end.lineno, 1)
        self.assertEqual(a.source_extent.start.colno, 1)
        self.assertEqual(a.source_extent.end.colno, 1)
        self.assertEqual(a.source_extent.start.byteno, 0)
        self.assertEqual(a.source_extent.end.byteno, 0)

        self.assertEqual(b.source_extent.start.lineno, 1)
        self.assertEqual(b.source_extent.end.lineno, 1)
        self.assertEqual(b.source_extent.start.colno, 3)
        self.assertEqual(b.source_extent.end.colno, 4)
        self.assertEqual(b.source_extent.start.byteno, 2)
        self.assertEqual(b.source_extent.end.byteno, 3)

    def test_tokenize_list_with_only_spread(self) -> None:
        self.assertEqual(list(tokenize("[ ... ]")), [LeftBracket(), Operator("..."), RightBracket()])

    def test_tokenize_list_with_spread(self) -> None:
        self.assertEqual(
            list(tokenize("[ 1 , ... ]")),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBracket(),
            ],
        )

    def test_tokenize_list_with_spread_no_spaces(self) -> None:
        self.assertEqual(
            list(tokenize("[ 1,... ]")),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBracket(),
            ],
        )

    def test_tokenize_list_with_named_spread(self) -> None:
        self.assertEqual(
            list(tokenize("[1,...rest]")),
            [
                LeftBracket(),
                IntLit(1),
                Operator(","),
                Operator("..."),
                Name("rest"),
                RightBracket(),
            ],
        )

    def test_tokenize_record_with_only_spread(self) -> None:
        self.assertEqual(
            list(tokenize("{ ... }")),
            [
                LeftBrace(),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_record_with_spread(self) -> None:
        self.assertEqual(
            list(tokenize("{ x = 1, ...}")),
            [
                LeftBrace(),
                Name("x"),
                Operator("="),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_record_with_spread_no_spaces(self) -> None:
        self.assertEqual(
            list(tokenize("{x=1,...}")),
            [
                LeftBrace(),
                Name("x"),
                Operator("="),
                IntLit(1),
                Operator(","),
                Operator("..."),
                RightBrace(),
            ],
        )

    def test_tokenize_variant_with_whitespace(self) -> None:
        self.assertEqual(list(tokenize("# \n\r\n\t abc")), [Hash(), Name("abc")])

    def test_tokenize_variant_with_no_space(self) -> None:
        self.assertEqual(list(tokenize("#abc")), [Hash(), Name("abc")])


class ParserTests(unittest.TestCase):
    def test_parse_with_empty_tokens_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedEOFError) as ctx:
            parse(Peekable(iter([])))
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_digit_returns_int(self) -> None:
        self.assertEqual(parse(Peekable(iter([IntLit(1)]))), Int(1))

    def test_parse_digits_returns_int(self) -> None:
        self.assertEqual(parse(Peekable(iter([IntLit(123)]))), Int(123))

    def test_parse_negative_int_returns_negative_int(self) -> None:
        self.assertEqual(parse(Peekable(iter([Operator("-"), IntLit(123)]))), Int(-123))

    def test_parse_negative_var_returns_binary_sub_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Operator("-"), Name("x")]))), Binop(BinopKind.SUB, Int(0), Var("x")))

    def test_parse_negative_int_binds_tighter_than_plus(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Operator("-"), Name("l"), Operator("+"), Name("r")]))),
            Binop(BinopKind.ADD, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_mul(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Operator("-"), Name("l"), Operator("*"), Name("r")]))),
            Binop(BinopKind.MUL, Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_index(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Operator("-"), Name("l"), Operator("@"), Name("r")]))),
            Access(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_negative_int_binds_tighter_than_apply(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Operator("-"), Name("l"), Name("r")]))),
            Apply(Binop(BinopKind.SUB, Int(0), Var("l")), Var("r")),
        )

    def test_parse_decimal_returns_float(self) -> None:
        self.assertEqual(parse(Peekable(iter([FloatLit(3.14)]))), Float(3.14))

    def test_parse_negative_float_returns_binary_sub_float(self) -> None:
        self.assertEqual(parse(Peekable(iter([Operator("-"), FloatLit(3.14)]))), Float(-3.14))

    def test_parse_var_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("abc_123")]))), Var("abc_123"))

    def test_parse_sha_var_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("$sha1'abc")]))), Var("$sha1'abc"))

    def test_parse_sha_var_without_quote_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("$sha1abc")]))), Var("$sha1abc"))

    def test_parse_dollar_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("$")]))), Var("$"))

    def test_parse_dollar_dollar_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("$$")]))), Var("$$"))

    @unittest.skip("TODO: make this fail to parse")
    def test_parse_sha_var_without_dollar_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, "unexpected token"):
            parse(Peekable(iter([Name("sha1'abc")])))

    def test_parse_dollar_dollar_var_returns_var(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("$$bills")]))), Var("$$bills"))

    def test_parse_bytes_returns_bytes(self) -> None:
        self.assertEqual(parse(Peekable(iter([BytesLit("QUJD", 64)]))), Bytes(b"ABC"))

    def test_parse_binary_add_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("+"), IntLit(2)]))), Binop(BinopKind.ADD, Int(1), Int(2))
        )

    def test_parse_binary_sub_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("-"), IntLit(2)]))), Binop(BinopKind.SUB, Int(1), Int(2))
        )

    def test_parse_binary_add_right_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("+"), IntLit(2), Operator("+"), IntLit(3)]))),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.ADD, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_right(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("+"), IntLit(2), Operator("*"), IntLit(3)]))),
            Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3))),
        )

    def test_mul_binds_tighter_than_add_left(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("*"), IntLit(2), Operator("+"), IntLit(3)]))),
            Binop(BinopKind.ADD, Binop(BinopKind.MUL, Int(1), Int(2)), Int(3)),
        )

    def test_mul_and_div_bind_left_to_right(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("/"), IntLit(3), Operator("*"), IntLit(3)]))),
            Binop(BinopKind.MUL, Binop(BinopKind.DIV, Int(1), Int(3)), Int(3)),
        )

    def test_exp_binds_tighter_than_mul_right(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(5), Operator("*"), IntLit(2), Operator("^"), IntLit(3)]))),
            Binop(BinopKind.MUL, Int(5), Binop(BinopKind.EXP, Int(2), Int(3))),
        )

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("+<"), Name("ls"), Operator("@"), IntLit(0)]))),
            Binop(BinopKind.LIST_APPEND, Var("a"), Access(Var("ls"), Int(0))),
        )

    def test_parse_binary_str_concat_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([StringLit("abc"), Operator("++"), StringLit("def")]))),
            Binop(BinopKind.STRING_CONCAT, String("abc"), String("def")),
        )

    def test_parse_binary_list_cons_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator(">+"), Name("b")]))),
            Binop(BinopKind.LIST_CONS, Var("a"), Var("b")),
        )

    def test_parse_binary_list_append_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("+<"), Name("b")]))),
            Binop(BinopKind.LIST_APPEND, Var("a"), Var("b")),
        )

    def test_parse_binary_op_returns_binop(self) -> None:
        ops = ["+", "-", "*", "/", "^", "%", "==", "/=", "<", ">", "<=", ">=", "&&", "||", "++", ">+", "+<"]
        for op in ops:
            with self.subTest(op=op):
                kind = BinopKind.from_str(op)
                self.assertEqual(
                    parse(Peekable(iter([Name("a"), Operator(op), Name("b")]))), Binop(kind, Var("a"), Var("b"))
                )

    def test_parse_empty_list(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([LeftBracket(), RightBracket()]))),
            List([]),
        )

    def test_parse_list_of_ints_returns_list(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([LeftBracket(), IntLit(1), Operator(","), IntLit(2), RightBracket()]))),
            List([Int(1), Int(2)]),
        )

    def test_parse_list_with_only_comma_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBracket(), Operator(","), RightBracket()])))

        self.assertEqual(parse_error.exception.unexpected_token, Operator(","))

    def test_parse_list_with_two_commas_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBracket(), Operator(","), Operator(","), RightBracket()])))

        self.assertEqual(parse_error.exception.unexpected_token, Operator(","))

    def test_parse_list_with_trailing_comma_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBracket(), IntLit(1), Operator(","), RightBracket()])))

        self.assertEqual(parse_error.exception.unexpected_token, RightBracket())

    def test_parse_assign(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("="), IntLit(1)]))),
            Assign(Var("a"), Int(1)),
        )

    def test_parse_function_one_arg_returns_function(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("->"), Name("a"), Operator("+"), IntLit(1)]))),
            Function(Var("a"), Binop(BinopKind.ADD, Var("a"), Int(1))),
        )

    def test_parse_function_two_args_returns_functions(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter([Name("a"), Operator("->"), Name("b"), Operator("->"), Name("a"), Operator("+"), Name("b")])
                )
            ),
            Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))),
        )

    def test_parse_assign_function(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("id"), Operator("="), Name("x"), Operator("->"), Name("x")]))),
            Assign(Var("id"), Function(Var("x"), Var("x"))),
        )

    def test_parse_function_application_one_arg(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("f"), Name("a")]))), Apply(Var("f"), Var("a")))

    def test_parse_function_application_two_args(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("f"), Name("a"), Name("b")]))), Apply(Apply(Var("f"), Var("a")), Var("b"))
        )

    def test_parse_where(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("a"), Operator("."), Name("b")]))), Where(Var("a"), Var("b")))

    def test_parse_nested_where(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("."), Name("b"), Operator("."), Name("c")]))),
            Where(Where(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_assert(self) -> None:
        self.assertEqual(parse(Peekable(iter([Name("a"), Operator("?"), Name("b")]))), Assert(Var("a"), Var("b")))

    def test_parse_nested_assert(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("?"), Name("b"), Operator("?"), Name("c")]))),
            Assert(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_mixed_assert_where(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("?"), Name("b"), Operator("."), Name("c")]))),
            Where(Assert(Var("a"), Var("b")), Var("c")),
        )

    def test_parse_hastype(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator(":"), Name("b")]))), Binop(BinopKind.HASTYPE, Var("a"), Var("b"))
        )

    def test_parse_hole(self) -> None:
        self.assertEqual(parse(Peekable(iter([LeftParen(), RightParen()]))), Hole())

    def test_parse_parenthesized_expression(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen()]))),
            Binop(BinopKind.ADD, Int(1), Int(2)),
        )

    def test_parse_parenthesized_add_mul(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter([LeftParen(), IntLit(1), Operator("+"), IntLit(2), RightParen(), Operator("*"), IntLit(3)])
                )
            ),
            Binop(BinopKind.MUL, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3)),
        )

    def test_parse_pipe(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("|>"), Name("f")]))),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_pipe(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([IntLit(1), Operator("|>"), Name("f"), Operator("|>"), Name("g")]))),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_reverse_pipe(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("f"), Operator("<|"), IntLit(1)]))),
            Apply(Var("f"), Int(1)),
        )

    def test_parse_nested_reverse_pipe(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("g"), Operator("<|"), Name("f"), Operator("<|"), IntLit(1)]))),
            Apply(Var("g"), Apply(Var("f"), Int(1))),
        )

    def test_parse_empty_record(self) -> None:
        self.assertEqual(parse(Peekable(iter([LeftBrace(), RightBrace()]))), Record({}))

    def test_parse_record_single_field(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([LeftBrace(), Name("a"), Operator("="), IntLit(4), RightBrace()]))),
            Record({"a": Int(4)}),
        )

    def test_parse_record_with_expression(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter([LeftBrace(), Name("a"), Operator("="), IntLit(1), Operator("+"), IntLit(2), RightBrace()])
                )
            ),
            Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))}),
        )

    def test_parse_record_multiple_fields(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter(
                        [
                            LeftBrace(),
                            Name("a"),
                            Operator("="),
                            IntLit(4),
                            Operator(","),
                            Name("b"),
                            Operator("="),
                            StringLit("z"),
                            RightBrace(),
                        ]
                    )
                )
            ),
            Record({"a": Int(4), "b": String("z")}),
        )

    def test_non_variable_in_assignment_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(Peekable(iter([IntLit(3), Operator("="), IntLit(4)])))
        self.assertEqual(ctx.exception.args[0], "expected variable in assignment Int(value=3)")

    def test_non_assign_in_record_constructor_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(Peekable(iter([LeftBrace(), IntLit(1), Operator(","), IntLit(2), RightBrace()])))
        self.assertEqual(ctx.exception.args[0], "failed to parse variable assignment in record constructor")

    def test_parse_right_eval_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("!"), Name("b")]))),
            Binop(BinopKind.RIGHT_EVAL, Var("a"), Var("b")),
        )

    def test_parse_right_eval_with_defs_returns_binop(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("a"), Operator("!"), Name("b"), Operator("."), Name("c")]))),
            Binop(BinopKind.RIGHT_EVAL, Var("a"), Where(Var("b"), Var("c"))),
        )

    def test_parse_match_no_cases_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse(Peekable(iter([Operator("|")])))
        self.assertEqual(ctx.exception.args[0], "unexpected end of input")

    def test_parse_match_one_case(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Operator("|"), IntLit(1), Operator("->"), IntLit(2)]))),
            MatchFunction([MatchCase(Int(1), Int(2))]),
        )

    def test_parse_match_two_cases(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter(
                        [
                            Operator("|"),
                            IntLit(1),
                            Operator("->"),
                            IntLit(2),
                            Operator("|"),
                            IntLit(2),
                            Operator("->"),
                            IntLit(3),
                        ]
                    )
                )
            ),
            MatchFunction(
                [
                    MatchCase(Int(1), Int(2)),
                    MatchCase(Int(2), Int(3)),
                ]
            ),
        )

    def test_parse_compose(self) -> None:
        gensym_reset()
        self.assertEqual(
            parse(Peekable(iter([Name("f"), Operator(">>"), Name("g")]))),
            Function(Var("$v0"), Apply(Var("g"), Apply(Var("f"), Var("$v0")))),
        )

    def test_parse_compose_reverse(self) -> None:
        gensym_reset()
        self.assertEqual(
            parse(Peekable(iter([Name("f"), Operator("<<"), Name("g")]))),
            Function(Var("$v0"), Apply(Var("f"), Apply(Var("g"), Var("$v0")))),
        )

    def test_parse_double_compose(self) -> None:
        gensym_reset()
        self.assertEqual(
            parse(Peekable(iter([Name("f"), Operator("<<"), Name("g"), Operator("<<"), Name("h")]))),
            Function(
                Var("$v1"),
                Apply(Var("f"), Apply(Function(Var("$v0"), Apply(Var("g"), Apply(Var("h"), Var("$v0")))), Var("$v1"))),
            ),
        )

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([Name("x"), Operator("||"), Name("y"), Operator("&&"), Name("z")]))),
            Binop(BinopKind.BOOL_OR, Var("x"), Binop(BinopKind.BOOL_AND, Var("y"), Var("z"))),
        )

    def test_parse_list_spread(self) -> None:
        self.assertEqual(
            parse(Peekable(iter([LeftBracket(), IntLit(1), Operator(","), Operator("..."), RightBracket()]))),
            List([Int(1), Spread()]),
        )

    @unittest.skip("TODO(max): Raise if ...x is used with non-name")
    def test_parse_list_with_non_name_expr_after_spread_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("unexpected token IntLit(lineno=-1, value=1)")):
            parse(Peekable(iter([LeftBracket(), IntLit(1), Operator(","), Operator("..."), IntLit(2), RightBracket()])))

    def test_parse_list_with_named_spread(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter(
                        [
                            LeftBracket(),
                            IntLit(1),
                            Operator(","),
                            Operator("..."),
                            Name("rest"),
                            RightBracket(),
                        ]
                    )
                )
            ),
            List([Int(1), Spread("rest")]),
        )

    def test_parse_list_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse(Peekable(iter([LeftBracket(), Operator("..."), Operator(","), IntLit(1), RightBracket()])))

    def test_parse_list_named_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse(
                Peekable(iter([LeftBracket(), Operator("..."), Name("rest"), Operator(","), IntLit(1), RightBracket()]))
            )

    def test_parse_list_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse(
                Peekable(
                    iter(
                        [
                            LeftBracket(),
                            IntLit(1),
                            Operator(","),
                            Operator("..."),
                            Operator(","),
                            IntLit(1),
                            RightBracket(),
                        ]
                    )
                )
            )

    def test_parse_list_named_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of list match")):
            parse(
                Peekable(
                    iter(
                        [
                            LeftBracket(),
                            IntLit(1),
                            Operator(","),
                            Operator("..."),
                            Name("rest"),
                            Operator(","),
                            IntLit(1),
                            RightBracket(),
                        ]
                    )
                )
            )

    def test_parse_record_spread(self) -> None:
        self.assertEqual(
            parse(
                Peekable(
                    iter(
                        [LeftBrace(), Name("x"), Operator("="), IntLit(1), Operator(","), Operator("..."), RightBrace()]
                    )
                )
            ),
            Record({"x": Int(1), "...": Spread()}),
        )

    def test_parse_record_spread_beginning_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of record match")):
            parse(
                Peekable(
                    iter(
                        [LeftBrace(), Operator("..."), Operator(","), Name("x"), Operator("="), IntLit(1), RightBrace()]
                    )
                )
            )

    def test_parse_record_spread_middle_raises_parse_error(self) -> None:
        with self.assertRaisesRegex(ParseError, re.escape("spread must come at end of record match")):
            parse(
                Peekable(
                    iter(
                        [
                            LeftBrace(),
                            Name("x"),
                            Operator("="),
                            IntLit(1),
                            Operator(","),
                            Operator("..."),
                            Operator(","),
                            Name("y"),
                            Operator("="),
                            IntLit(2),
                            RightBrace(),
                        ]
                    )
                )
            )

    def test_parse_record_with_only_comma_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBrace(), Operator(","), RightBrace()])))

        self.assertEqual(parse_error.exception.unexpected_token, Operator(","))

    def test_parse_record_with_two_commas_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBrace(), Operator(","), Operator(","), RightBrace()])))

        self.assertEqual(parse_error.exception.unexpected_token, Operator(","))

    def test_parse_record_with_trailing_comma_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([LeftBrace(), Name("x"), Operator("="), IntLit(1), Operator(","), RightBrace()])))

        self.assertEqual(parse_error.exception.unexpected_token, RightBrace())

    def test_parse_variant_returns_variant(self) -> None:
        self.assertEqual(parse(Peekable(iter([Hash(), Name("abc"), IntLit(1)]))), Variant("abc", Int(1)))

    def test_parse_hash_raises_unexpected_eof_error(self) -> None:
        tokens = Peekable(iter([Hash()]))
        with self.assertRaises(UnexpectedEOFError):
            parse(tokens)

    def test_parse_variant_non_name_raises_parse_error(self) -> None:
        with self.assertRaises(UnexpectedTokenError) as parse_error:
            parse(Peekable(iter([Hash(), IntLit(1)])))

        self.assertEqual(parse_error.exception.unexpected_token, IntLit(1))

    def test_parse_variant_eof_raises_unexpected_eof_error(self) -> None:
        with self.assertRaises(UnexpectedEOFError):
            parse(Peekable(iter([Hash()])))

    def test_match_with_variant(self) -> None:
        ast = parse(tokenize("| #true () -> 123"))
        self.assertEqual(ast, MatchFunction([MatchCase(TRUE, Int(123))]))

    def test_binary_and_with_variant_args(self) -> None:
        ast = parse(tokenize("#true() && #false()"))
        self.assertEqual(ast, Binop(BinopKind.BOOL_AND, TRUE, FALSE))

    def test_apply_with_variant_args(self) -> None:
        ast = parse(tokenize("f #true() #false()"))
        self.assertEqual(ast, Apply(Apply(Var("f"), TRUE), FALSE))

    def test_parse_int_preserves_source_extent(self) -> None:
        int_lit = IntLit(1)
        source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        int_lit.source_extent = source_extent
        int_ast = make_source_annotated_object(Int, source_extent, 1)
        self.assertTrue(parse(Peekable(iter([int_lit]))).source_extent == int_ast.source_extent)

    def test_parse_float_preserves_source_extent(self) -> None:
        float_lit = FloatLit(3.2)
        source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=3, byteno=2)
        )
        float_lit.source_extent = source_extent
        float_ast = make_source_annotated_object(Float, source_extent, 3.2)
        self.assertTrue(parse(Peekable(iter([float_lit]))).source_extent == float_ast.source_extent)

    def test_parse_string_preserves_source_extent(self) -> None:
        string_lit = StringLit("Hello")
        source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=7, byteno=6)
        )
        string_lit.source_extent = source_extent
        string_ast = make_source_annotated_object(String, source_extent, "Hello")
        self.assertTrue(parse(Peekable(iter([string_lit]))).source_extent == string_ast.source_extent)

    def test_parse_bytes_preserves_source_extent(self) -> None:
        bytes_lit = BytesLit("QUJD", 64)
        source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=9, byteno=8)
        )
        bytes_lit.source_extent = source_extent
        bytes_ast = make_source_annotated_object(Bytes, source_extent, base64.b64decode("QUJD"))
        self.assertTrue(parse(Peekable(iter([bytes_lit]))).source_extent == bytes_ast.source_extent)

    def test_parse_var_preserves_source_extent(self) -> None:
        var = Name("x")
        source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        var.source_extent = source_extent
        var_ast = make_source_annotated_object(Var, source_extent, "x")
        self.assertTrue(parse(Peekable(iter([var]))).source_extent == var_ast.source_extent)

    def test_parse_hole_preserves_source_extent(self) -> None:
        left_paren = LeftParen()
        left_paren.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        right_paren = RightParen()
        right_paren.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=2, byteno=1), end=SourceLocation(lineno=1, colno=2, byteno=1)
        )
        hole = make_source_annotated_object(
            Hole,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=2, byteno=1)
            ),
        )
        self.assertTrue(parse(Peekable(iter([left_paren, right_paren]))).source_extent == hole.source_extent)

    def test_parenthesized_expression_preserves_source_extent(self) -> None:
        left_paren = LeftParen()
        left_paren.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        num = IntLit(1)
        num_ast = make_source_annotated_object(
            Int,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=2, byteno=1), end=SourceLocation(lineno=1, colno=2, byteno=1)
            ),
            1,
        )
        right_paren = RightParen()
        right_paren.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=3, byteno=2), end=SourceLocation(lineno=1, colno=3, byteno=2)
        )
        parenthesized_num = make_source_annotated_object(
            Int,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=3, byteno=2)
            ),
            1,
        )
        self.assertTrue(
            parse(Peekable(iter([left_paren, num, right_paren]))).source_extent == parenthesized_num.source_extent
        )

    def test_parse_spread_preserves_source_extent(self) -> None:
        ellipsis = Operator("...")
        ellipsis.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=3, byteno=2)
        )
        name = Name("x")
        name.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=4, byteno=3), end=SourceLocation(lineno=1, colno=4, byteno=3)
        )
        spread = make_source_annotated_object(
            Spread,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=4, byteno=3)
            ),
            "x",
        )
        self.assertTrue(parse(Peekable(iter([ellipsis, name]))).source_extent == spread.source_extent)

    def test_parse_binop_preserves_source_extent(self) -> None:
        first_addend = IntLit(1)
        first_addend.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        operator = Operator("+")
        operator.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=3, byteno=2), end=SourceLocation(lineno=1, colno=3, byteno=2)
        )
        second_addend = IntLit(2)
        second_addend.source_extent = SourceExtent(
            start=SourceLocation(lineno=2, colno=5, byteno=4), end=SourceLocation(lineno=2, colno=5, byteno=4)
        )
        first_addend_ast = make_source_annotated_object(Int, first_addend.source_extent, 1)
        operator_ast = BinopKind.ADD
        second_addend_ast = make_source_annotated_object(Int, second_addend.source_extent, 2)
        binop = make_source_annotated_object(
            Binop,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=2, colno=5, byteno=4)
            ),
            operator_ast,
            first_addend_ast,
            second_addend_ast,
        )
        self.assertTrue(
            parse(Peekable(iter([first_addend, operator, second_addend]))).source_extent == binop.source_extent
        )

    def test_parse_list_preserves_source_extent(self) -> None:
        left_bracket = LeftBracket()
        left_bracket.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=1, byteno=0)
        )
        one = IntLit(1)
        one.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=2, byteno=1), end=SourceLocation(lineno=1, colno=2, byteno=1)
        )
        comma = Operator(",")
        comma.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=3, byteno=2), end=SourceLocation(lineno=1, colno=3, byteno=2)
        )
        two = IntLit(2)
        two.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=5, byteno=4), end=SourceLocation(lineno=1, colno=5, byteno=4)
        )
        right_bracket = RightBracket()
        right_bracket.source_extent = SourceExtent(
            start=SourceLocation(lineno=1, colno=6, byteno=5), end=SourceLocation(lineno=1, colno=6, byteno=5)
        )
        one_ast = make_source_annotated_object(Int, one.source_extent, 1)
        two_ast = make_source_annotated_object(Int, two.source_extent, 2)
        list_ast = make_source_annotated_object(
            List,
            SourceExtent(
                start=SourceLocation(lineno=1, colno=1, byteno=0), end=SourceLocation(lineno=1, colno=6, byteno=5)
            ),
            [one_ast, two_ast],
        )
        self.assertTrue(
            parse(Peekable(iter([left_bracket, one, comma, two, right_bracket]))).source_extent
            == list_ast.source_extent
        )


class MatchTests(unittest.TestCase):
    def test_match_hole_with_non_hole_returns_none(self) -> None:
        self.assertEqual(match(Int(1), pattern=Hole()), None)

    def test_match_hole_with_hole_returns_empty_dict(self) -> None:
        self.assertEqual(match(Hole(), pattern=Hole()), {})

    def test_match_with_equal_ints_returns_empty_dict(self) -> None:
        self.assertEqual(match(Int(1), pattern=Int(1)), {})

    def test_match_with_inequal_ints_returns_none(self) -> None:
        self.assertEqual(match(Int(2), pattern=Int(1)), None)

    def test_match_int_with_non_int_returns_none(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Int(1)), None)

    def test_match_with_equal_floats_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(Float(1), pattern=Float(1))

    def test_match_with_inequal_floats_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(Float(2), pattern=Float(1))

    def test_match_float_with_non_float_raises_match_error(self) -> None:
        with self.assertRaisesRegex(MatchError, re.escape("pattern matching is not supported for Floats")):
            match(String("abc"), pattern=Float(1))

    def test_match_with_equal_strings_returns_empty_dict(self) -> None:
        self.assertEqual(match(String("a"), pattern=String("a")), {})

    def test_match_with_inequal_strings_returns_none(self) -> None:
        self.assertEqual(match(String("b"), pattern=String("a")), None)

    def test_match_string_with_non_string_returns_none(self) -> None:
        self.assertEqual(match(Int(1), pattern=String("abc")), None)

    def test_match_var_returns_dict_with_var_name(self) -> None:
        self.assertEqual(match(String("abc"), pattern=Var("a")), {"a": String("abc")})

    def test_match_record_with_non_record_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            None,
        )

    def test_match_record_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y"), "z": Var("z")}),
            ),
            None,
        )

    def test_match_record_with_fewer_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x")}),
            ),
            None,
        )

    def test_match_record_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Var("x"), "y": Var("y")}),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_record_with_matching_const_returns_dict_with_other_keys(self) -> None:
        # TODO(max): Should this be the case? I feel like we should return all
        # the keys.
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(1), "y": Var("y")}),
            ),
            {"y": Int(2)},
        )

    def test_match_record_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"x": Int(1), "y": Int(2)}),
                pattern=Record({"x": Int(3), "y": Var("y")}),
            ),
            None,
        )

    def test_match_list_with_non_list_returns_none(self) -> None:
        self.assertEqual(
            match(
                Int(2),
                pattern=List([Var("x"), Var("y")]),
            ),
            None,
        )

    def test_match_list_with_more_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y"), Var("z")]),
            ),
            None,
        )

    def test_match_list_with_fewer_fields_in_pattern_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x")]),
            ),
            None,
        )

    def test_match_list_with_vars_returns_dict_with_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("x"), Var("y")]),
            ),
            {"x": Int(1), "y": Int(2)},
        )

    def test_match_list_with_matching_const_returns_dict_with_other_keys(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(1), Var("y")]),
            ),
            {"y": Int(2)},
        )

    def test_match_list_with_non_matching_const_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Int(3), Var("y")]),
            ),
            None,
        )

    def test_parse_right_pipe(self) -> None:
        text = "3 + 4 |> $$quote"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_left_pipe(self) -> None:
        text = "$$quote <| 3 + 4"
        ast = parse(tokenize(text))
        self.assertEqual(ast, Apply(Var("$$quote"), Binop(BinopKind.ADD, Int(3), Int(4))))

    def test_parse_match_with_left_apply(self) -> None:
        text = """| a -> b <| c
                  | d -> e"""
        tokens = tokenize(text)
        self.assertEqual(
            list(tokens),
            [
                Operator("|"),
                Name("a"),
                Operator("->"),
                Name("b"),
                Operator("<|"),
                Name("c"),
                Operator("|"),
                Name("d"),
                Operator("->"),
                Name("e"),
            ],
        )
        tokens = tokenize(text)
        ast = parse(tokens)
        self.assertEqual(
            ast, MatchFunction([MatchCase(Var("a"), Apply(Var("b"), Var("c"))), MatchCase(Var("d"), Var("e"))])
        )

    def test_parse_match_with_right_apply(self) -> None:
        text = """
| 1 -> 19
| a -> a |> (x -> x + 1)
"""
        tokens = tokenize(text)
        ast = parse(tokens)
        self.assertEqual(
            ast,
            MatchFunction(
                [
                    MatchCase(Int(1), Int(19)),
                    MatchCase(
                        Var("a"),
                        Apply(
                            Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))),
                            Var("a"),
                        ),
                    ),
                ]
            ),
        )

    def test_match_list_with_spread_returns_empty_dict(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4), Int(5)]),
                pattern=List([Int(1), Spread()]),
            ),
            {},
        )

    def test_match_list_with_named_spread_returns_name_bound_to_rest(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4)]),
                pattern=List([Var("a"), Int(2), Spread("rest")]),
            ),
            {"a": Int(1), "rest": List([Int(3), Int(4)])},
        )

    def test_match_list_with_named_spread_returns_name_bound_to_empty_rest(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2)]),
                pattern=List([Var("a"), Int(2), Spread("rest")]),
            ),
            {"a": Int(1), "rest": List([])},
        )

    def test_match_list_with_mismatched_spread_returns_none(self) -> None:
        self.assertEqual(
            match(
                List([Int(1), Int(2), Int(3), Int(4), Int(5)]),
                pattern=List([Int(1), Int(6), Spread()]),
            ),
            None,
        )

    def test_match_record_with_constant_and_spread_returns_empty_dict(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"a": Int(1), "...": Spread()}),
            ),
            {},
        )

    def test_match_record_with_var_and_spread_returns_match(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"a": Var("x"), "...": Spread()}),
            ),
            {"x": Int(1)},
        )

    def test_match_record_with_mismatched_spread_returns_none(self) -> None:
        self.assertEqual(
            match(
                Record({"a": Int(1), "b": Int(2), "c": Int(3)}),
                pattern=Record({"d": Var("x"), "...": Spread()}),
            ),
            None,
        )

    def test_match_variant_with_equal_tag_returns_empty_dict(self) -> None:
        self.assertEqual(match(Variant("abc", Hole()), pattern=Variant("abc", Hole())), {})

    def test_match_variant_with_inequal_tag_returns_none(self) -> None:
        self.assertEqual(match(Variant("def", Hole()), pattern=Variant("abc", Hole())), None)

    def test_match_variant_matches_value(self) -> None:
        self.assertEqual(match(Variant("abc", Int(123)), pattern=Variant("abc", Hole())), None)
        self.assertEqual(match(Variant("abc", Int(123)), pattern=Variant("abc", Int(123))), {})

    def test_match_variant_with_different_type_returns_none(self) -> None:
        self.assertEqual(match(Int(123), pattern=Variant("abc", Hole())), None)


class EvalTests(unittest.TestCase):
    def test_eval_int_returns_int(self) -> None:
        exp = Int(5)
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_float_returns_float(self) -> None:
        exp = Float(3.14)
        self.assertEqual(eval_exp({}, exp), Float(3.14))

    def test_eval_str_returns_str(self) -> None:
        exp = String("xyz")
        self.assertEqual(eval_exp({}, exp), String("xyz"))

    def test_eval_bytes_returns_bytes(self) -> None:
        exp = Bytes(b"xyz")
        self.assertEqual(eval_exp({}, exp), Bytes(b"xyz"))

    def test_eval_with_non_existent_var_raises_name_error(self) -> None:
        exp = Var("no")
        with self.assertRaises(NameError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "name 'no' is not defined")

    def test_eval_with_bound_var_returns_value(self) -> None:
        exp = Var("yes")
        env = {"yes": Int(123)}
        self.assertEqual(eval_exp(env, exp), Int(123))

    def test_eval_with_binop_add_returns_sum(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_with_nested_binop(self) -> None:
        exp = Binop(BinopKind.ADD, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_add_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.ADD, Int(1), String("hello"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected Int or Float, got String")

    def test_eval_with_binop_sub(self) -> None:
        exp = Binop(BinopKind.SUB, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(-1))

    def test_eval_with_binop_mul(self) -> None:
        exp = Binop(BinopKind.MUL, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_with_binop_div(self) -> None:
        exp = Binop(BinopKind.DIV, Int(3), Int(10))
        self.assertEqual(eval_exp({}, exp), Float(0.3))

    def test_eval_with_binop_floor_div(self) -> None:
        exp = Binop(BinopKind.FLOOR_DIV, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(0))

    def test_eval_with_binop_exp(self) -> None:
        exp = Binop(BinopKind.EXP, Int(2), Int(3))
        self.assertEqual(eval_exp({}, exp), Int(8))

    def test_eval_with_binop_mod(self) -> None:
        exp = Binop(BinopKind.MOD, Int(10), Int(4))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_with_binop_equal_with_equal_returns_true(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), TRUE)

    def test_eval_with_binop_equal_with_inequal_returns_false(self) -> None:
        exp = Binop(BinopKind.EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), FALSE)

    def test_eval_with_binop_not_equal_with_equal_returns_false(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(1))
        self.assertEqual(eval_exp({}, exp), FALSE)

    def test_eval_with_binop_not_equal_with_inequal_returns_true(self) -> None:
        exp = Binop(BinopKind.NOT_EQUAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), TRUE)

    def test_eval_with_binop_concat_with_strings_returns_string(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String("hello"), String(" world"))
        self.assertEqual(eval_exp({}, exp), String("hello world"))

    def test_eval_with_binop_concat_with_int_string_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, Int(123), String(" world"))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_concat_with_string_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.STRING_CONCAT, String(" world"), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected String, got Int")

    def test_eval_with_binop_cons_with_int_list_returns_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, Int(1), List([Int(2), Int(3)]))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_binop_cons_with_list_list_returns_nested_list(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), List([]))
        self.assertEqual(eval_exp({}, exp), List([List([])]))

    def test_eval_with_binop_cons_with_list_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.LIST_CONS, List([]), Int(123))
        with self.assertRaises(TypeError) as ctx:
            eval_exp({}, exp)
        self.assertEqual(ctx.exception.args[0], "expected List, got Int")

    def test_eval_with_list_append(self) -> None:
        exp = Binop(BinopKind.LIST_APPEND, List([Int(1), Int(2)]), Int(3))
        self.assertEqual(eval_exp({}, exp), List([Int(1), Int(2), Int(3)]))

    def test_eval_with_list_evaluates_elements(self) -> None:
        exp = List(
            [
                Binop(BinopKind.ADD, Int(1), Int(2)),
                Binop(BinopKind.ADD, Int(3), Int(4)),
            ]
        )
        self.assertEqual(eval_exp({}, exp), List([Int(3), Int(7)]))

    def test_eval_with_function_returns_closure_with_improved_env(self) -> None:
        exp = Function(Var("x"), Var("x"))
        self.assertEqual(eval_exp({"a": Int(1), "b": Int(2)}, exp), Closure({}, exp))

    def test_eval_with_match_function_returns_closure_with_improved_env(self) -> None:
        exp = MatchFunction([])
        self.assertEqual(eval_exp({"a": Int(1), "b": Int(2)}, exp), Closure({}, exp))

    def test_eval_assign_returns_env_object(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        result = eval_exp(env, exp)
        self.assertEqual(result, EnvObject({"a": Int(1)}))

    def test_eval_assign_function_returns_closure_without_function_in_env(self) -> None:
        exp = Assign(Var("a"), Function(Var("x"), Var("x")))
        result = eval_exp({}, exp)
        assert isinstance(result, EnvObject)
        closure = result.env["a"]
        self.assertIsInstance(closure, Closure)
        self.assertEqual(closure, Closure({}, Function(Var("x"), Var("x"))))

    def test_eval_assign_function_returns_closure_with_function_in_env(self) -> None:
        exp = Assign(Var("a"), Function(Var("x"), Var("a")))
        result = eval_exp({}, exp)
        assert isinstance(result, EnvObject)
        closure = result.env["a"]
        self.assertIsInstance(closure, Closure)
        self.assertEqual(closure, Closure({"a": closure}, Function(Var("x"), Var("a"))))

    def test_eval_assign_does_not_modify_env(self) -> None:
        exp = Assign(Var("a"), Int(1))
        env: Env = {}
        eval_exp(env, exp)
        self.assertEqual(env, {})

    def test_eval_where_evaluates_in_order(self) -> None:
        exp = Where(Binop(BinopKind.ADD, Var("a"), Int(2)), Assign(Var("a"), Int(1)))
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_nested_where(self) -> None:
        exp = Where(
            Where(
                Binop(BinopKind.ADD, Var("a"), Var("b")),
                Assign(Var("a"), Int(1)),
            ),
            Assign(Var("b"), Int(2)),
        )
        env: Env = {}
        self.assertEqual(eval_exp(env, exp), Int(3))
        self.assertEqual(env, {})

    def test_eval_assert_with_truthy_cond_returns_value(self) -> None:
        exp = Assert(Int(123), TRUE)
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        exp = Assert(Int(123), FALSE)
        with self.assertRaisesRegex(AssertionError, re.escape("condition #false () failed")):
            eval_exp({}, exp)

    def test_eval_nested_assert(self) -> None:
        exp = Assert(Assert(Int(123), TRUE), TRUE)
        self.assertEqual(eval_exp({}, exp), Int(123))

    def test_eval_hole(self) -> None:
        exp = Hole()
        self.assertEqual(eval_exp({}, exp), Hole())

    def test_eval_function_application_one_arg(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(1))), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(3))

    def test_eval_function_application_two_args(self) -> None:
        exp = Apply(
            Apply(Function(Var("a"), Function(Var("b"), Binop(BinopKind.ADD, Var("a"), Var("b")))), Int(3)),
            Int(2),
        )
        self.assertEqual(eval_exp({}, exp), Int(5))

    def test_eval_function_returns_closure_with_captured_env(self) -> None:
        exp = Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        res = eval_exp({"y": Int(5)}, exp)
        self.assertIsInstance(res, Closure)
        assert isinstance(res, Closure)  # for mypy
        self.assertEqual(res.env, {"y": Int(5)})

    def test_eval_function_capture_env(self) -> None:
        exp = Apply(Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y"))), Int(2))
        self.assertEqual(eval_exp({"y": Int(5)}, exp), Int(7))

    def test_eval_non_function_raises_type_error(self) -> None:
        exp = Apply(Int(3), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to apply a non-closure of type Int")):
            eval_exp({}, exp)

    def test_eval_access_from_invalid_object_raises_type_error(self) -> None:
        exp = Access(Int(4), String("x"))
        with self.assertRaisesRegex(TypeError, re.escape("attempted to access from type Int")):
            eval_exp({}, exp)

    def test_eval_record_evaluates_value_expressions(self) -> None:
        exp = Record({"a": Binop(BinopKind.ADD, Int(1), Int(2))})
        self.assertEqual(eval_exp({}, exp), Record({"a": Int(3)}))

    def test_eval_record_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Int(0))
        with self.assertRaisesRegex(
            TypeError, re.escape("cannot access record field using Int, expected a field name")
        ):
            eval_exp({}, exp)

    def test_eval_record_access_with_unknown_accessor_raises_name_error(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("b"))
        with self.assertRaisesRegex(NameError, re.escape("no assignment to b found in record")):
            eval_exp({}, exp)

    def test_eval_record_access(self) -> None:
        exp = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(eval_exp({}, exp), Int(4))

    def test_eval_list_access_with_invalid_accessor_raises_type_error(self) -> None:
        exp = Access(List([Int(4)]), String("hello"))
        with self.assertRaisesRegex(TypeError, re.escape("cannot index into list using type String, expected integer")):
            eval_exp({}, exp)

    def test_eval_list_access_with_out_of_bounds_accessor_raises_value_error(self) -> None:
        exp = Access(List([Int(1), Int(2), Int(3)]), Int(4))
        with self.assertRaisesRegex(ValueError, re.escape("index 4 out of bounds for list")):
            eval_exp({}, exp)

    def test_eval_list_access(self) -> None:
        exp = Access(List([String("a"), String("b"), String("c")]), Int(2))
        self.assertEqual(eval_exp({}, exp), String("c"))

    def test_right_eval_evaluates_right_hand_side(self) -> None:
        exp = Binop(BinopKind.RIGHT_EVAL, Int(1), Int(2))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_no_cases_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([]), Int(1))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_int_with_equal_int_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(1))
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_match_int_with_inequal_int_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=Int(1), body=Int(2))]), Int(3))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_string_with_equal_string_matches(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("a"))
        self.assertEqual(eval_exp({}, exp), String("b"))

    def test_match_string_with_inequal_string_raises_match_error(self) -> None:
        exp = Apply(MatchFunction([MatchCase(pattern=String("a"), body=String("b"))]), String("c"))
        with self.assertRaisesRegex(MatchError, "no matching cases"):
            eval_exp({}, exp)

    def test_match_falls_through_to_next(self) -> None:
        exp = Apply(
            MatchFunction([MatchCase(pattern=Int(3), body=Int(4)), MatchCase(pattern=Int(1), body=Int(2))]), Int(1)
        )
        self.assertEqual(eval_exp({}, exp), Int(2))

    def test_eval_compose(self) -> None:
        gensym_reset()
        exp = parse(tokenize("(x -> x + 3) << (x -> x * 2)"))
        env = {"a": Int(1)}
        expected = Closure(
            {},
            Function(
                Var("$v0"),
                Apply(
                    Function(Var("x"), Binop(BinopKind.ADD, Var("x"), Int(3))),
                    Apply(Function(Var("x"), Binop(BinopKind.MUL, Var("x"), Int(2))), Var("$v0")),
                ),
            ),
        )
        self.assertEqual(eval_exp(env, exp), expected)

    def test_eval_native_function_returns_function(self) -> None:
        exp = NativeFunction("times2", lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertIs(eval_exp({}, exp), exp)

    def test_eval_apply_native_function_calls_function(self) -> None:
        exp = Apply(NativeFunction("times2", lambda x: Int(x.value * 2)), Int(3))  # type: ignore [attr-defined]
        self.assertEqual(eval_exp({}, exp), Int(6))

    def test_eval_apply_quote_returns_ast(self) -> None:
        ast = Binop(BinopKind.ADD, Int(1), Int(2))
        exp = Apply(Var("$$quote"), ast)
        self.assertIs(eval_exp({}, exp), ast)

    def test_eval_apply_closure_with_match_function_has_access_to_closure_vars(self) -> None:
        ast = Apply(Closure({"x": Int(1)}, MatchFunction([MatchCase(Var("y"), Var("x"))])), Int(2))
        self.assertEqual(eval_exp({}, ast), Int(1))

    def test_eval_less_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), TRUE)

    def test_eval_less_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_less_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), TRUE)

    def test_eval_less_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.LESS_EQUAL, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_greater_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), FALSE)

    def test_eval_greater_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_eval_greater_equal_returns_bool(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, Int(3), Int(4))
        self.assertEqual(eval_exp({}, ast), FALSE)

    def test_eval_greater_equal_on_non_bool_raises_type_error(self) -> None:
        ast = Binop(BinopKind.GREATER_EQUAL, String("xyz"), Int(4))
        with self.assertRaisesRegex(TypeError, re.escape("expected Int or Float, got String")):
            eval_exp({}, ast)

    def test_boolean_and_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_AND, TRUE, Var("a"))
        self.assertEqual(eval_exp({"a": FALSE}, ast), FALSE)

        ast = Binop(BinopKind.BOOL_AND, Var("a"), FALSE)
        self.assertEqual(eval_exp({"a": TRUE}, ast), FALSE)

    def test_boolean_or_evaluates_args(self) -> None:
        ast = Binop(BinopKind.BOOL_OR, FALSE, Var("a"))
        self.assertEqual(eval_exp({"a": TRUE}, ast), TRUE)

        ast = Binop(BinopKind.BOOL_OR, Var("a"), TRUE)
        self.assertEqual(eval_exp({"a": FALSE}, ast), TRUE)

    def test_boolean_and_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction("error", raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_AND, FALSE, apply)
        self.assertEqual(eval_exp({"error": error}, ast), FALSE)

    def test_boolean_or_short_circuit(self) -> None:
        def raise_func(message: Object) -> Object:
            if not isinstance(message, String):
                raise TypeError(f"raise_func expected String, but got {type(message).__name__}")
            raise RuntimeError(message)

        error = NativeFunction("error", raise_func)
        apply = Apply(Var("error"), String("expected failure"))
        ast = Binop(BinopKind.BOOL_OR, TRUE, apply)
        self.assertEqual(eval_exp({"error": error}, ast), TRUE)

    def test_boolean_and_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_AND, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected #true or #false, got Int")):
            eval_exp({}, exp)

    def test_boolean_or_on_int_raises_type_error(self) -> None:
        exp = Binop(BinopKind.BOOL_OR, Int(1), Int(2))
        with self.assertRaisesRegex(TypeError, re.escape("expected #true or #false, got Int")):
            eval_exp({}, exp)

    def test_eval_record_with_spread_fails(self) -> None:
        exp = Record({"x": Spread()})
        with self.assertRaisesRegex(RuntimeError, "cannot evaluate a spread"):
            eval_exp({}, exp)

    def test_eval_variant_returns_variant(self) -> None:
        self.assertEqual(
            eval_exp(
                {},
                Variant("abc", Binop(BinopKind.ADD, Int(1), Int(2))),
            ),
            Variant("abc", Int(3)),
        )

    def test_eval_float_and_float_addition_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.ADD, Float(1.0), Float(2.0))), Float(3.0))

    def test_eval_int_and_float_addition_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.ADD, Int(1), Float(2.0))), Float(3.0))

    def test_eval_int_and_float_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Int(1), Float(2.0))), Float(0.5))

    def test_eval_float_and_int_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Float(1.0), Int(2))), Float(0.5))

    def test_eval_int_and_int_division_returns_float(self) -> None:
        self.assertEqual(eval_exp({}, Binop(BinopKind.DIV, Int(1), Int(2))), Float(0.5))


class EndToEndTestsBase(unittest.TestCase):
    def _run(self, text: str, env: Optional[Env] = None, check: bool = False) -> Object:
        tokens = tokenize(text)
        ast = parse(tokens)
        if check:
            infer_type(ast, OP_ENV)
        if env is None:
            env = boot_env()
        return eval_exp(env, ast)


class EndToEndTests(EndToEndTestsBase):
    def test_int_returns_int(self) -> None:
        self.assertEqual(self._run("1"), Int(1))

    def test_float_returns_float(self) -> None:
        self.assertEqual(self._run("3.14"), Float(3.14))

    def test_bytes_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~QUJD"), Bytes(b"ABC"))

    def test_bytes_base85_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~85'K|(_"), Bytes(b"ABC"))

    def test_bytes_base64_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~64'QUJD"), Bytes(b"ABC"))

    def test_bytes_base32_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~32'IFBEG==="), Bytes(b"ABC"))

    def test_bytes_base16_returns_bytes(self) -> None:
        self.assertEqual(self._run("~~16'414243"), Bytes(b"ABC"))

    def test_int_add_returns_int(self) -> None:
        self.assertEqual(self._run("1 + 2"), Int(3))

    def test_int_sub_returns_int(self) -> None:
        self.assertEqual(self._run("1 - 2"), Int(-1))

    def test_string_concat_returns_string(self) -> None:
        self.assertEqual(self._run('"abc" ++ "def"'), String("abcdef"))

    def test_list_cons_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ [2,3]"), List([Int(1), Int(2), Int(3)]))

    def test_list_cons_nested_returns_list(self) -> None:
        self.assertEqual(self._run("1 >+ 2 >+ [3,4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_list_append_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3"), List([Int(1), Int(2), Int(3)]))

    def test_list_append_nested_returns_list(self) -> None:
        self.assertEqual(self._run("[1,2] +< 3 +< 4"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_empty_list(self) -> None:
        self.assertEqual(self._run("[ ]"), List([]))

    def test_empty_list_with_no_spaces(self) -> None:
        self.assertEqual(self._run("[]"), List([]))

    def test_list_of_ints(self) -> None:
        self.assertEqual(self._run("[ 1 , 2 ]"), List([Int(1), Int(2)]))

    def test_list_of_exprs(self) -> None:
        self.assertEqual(
            self._run("[ 1 + 2 , 3 + 4 ]"),
            List([Int(3), Int(7)]),
        )

    def test_where(self) -> None:
        self.assertEqual(self._run("a + 2 . a = 1"), Int(3))

    def test_nested_where(self) -> None:
        self.assertEqual(self._run("a + b . a = 1 . b = 2"), Int(3))

    def test_assert_with_truthy_cond_returns_value(self) -> None:
        self.assertEqual(self._run("a + 1 ? a == 1 . a = 1"), Int(2))

    def test_assert_with_falsey_cond_raises_assertion_error(self) -> None:
        with self.assertRaisesRegex(AssertionError, "condition a == 2 failed"):
            self._run("a + 1 ? a == 2 . a = 1")

    def test_nested_assert(self) -> None:
        self.assertEqual(self._run("a + b ? a == 1 ? b == 2 . a = 1 . b = 2"), Int(3))

    def test_hole(self) -> None:
        self.assertEqual(self._run("()"), Hole())

    def test_bindings_behave_like_letstar(self) -> None:
        with self.assertRaises(NameError) as ctx:
            self._run("b . a = 1 . b = a")
        self.assertEqual(ctx.exception.args[0], "name 'a' is not defined")

    def test_function_application_two_args(self) -> None:
        self.assertEqual(self._run("(a -> b -> a + b) 3 2"), Int(5))

    def test_function_create_list_correct_order(self) -> None:
        self.assertEqual(self._run("(a -> b -> [a, b]) 3 2"), List([Int(3), Int(2)]))

    def test_create_record(self) -> None:
        self.assertEqual(self._run("{a = 1 + 3}"), Record({"a": Int(4)}))

    def test_access_record(self) -> None:
        self.assertEqual(self._run('rec@b . rec = { a = 1, b = "x" }'), String("x"))

    def test_access_list(self) -> None:
        self.assertEqual(self._run("xs@1 . xs = [1, 2, 3]"), Int(2))

    def test_access_list_var(self) -> None:
        self.assertEqual(self._run("xs@y . y = 2 . xs = [1, 2, 3]"), Int(3))

    def test_access_list_expr(self) -> None:
        self.assertEqual(self._run("xs@(1+1) . xs = [1, 2, 3]"), Int(3))

    def test_access_list_closure_var(self) -> None:
        self.assertEqual(
            self._run("list_at 1 [1,2,3] . list_at = idx -> ls -> ls@idx"),
            Int(2),
        )

    def test_functions_eval_arguments(self) -> None:
        self.assertEqual(self._run("(x -> x) c . c = 1"), Int(1))

    def test_non_var_function_arg_raises_parse_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._run("1 -> a")
        self.assertEqual(ctx.exception.args[0], "expected variable in function definition 1")

    def test_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (b -> b * 2)) 3"), Int(8))

    def test_compose_does_not_expose_internal_x(self) -> None:
        with self.assertRaisesRegex(NameError, "name 'x' is not defined"):
            self._run("f 3 . f = ((y -> x) >> (z -> x))")

    def test_double_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) >> (x -> x) >> (b -> b * 2)) 3"), Int(8))

    def test_reverse_compose(self) -> None:
        self.assertEqual(self._run("((a -> a + 1) << (b -> b * 2)) 3"), Int(7))

    def test_simple_int_match(self) -> None:
        self.assertEqual(
            self._run(
                """
                inc 2
                . inc =
                  | 1 -> 2
                  | 2 -> 3
                  """
            ),
            Int(3),
        )

    def test_match_var_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                """
            ),
            Int(3),
        )

    def test_match_var_binds_first_arm(self) -> None:
        self.assertEqual(
            self._run(
                """
                id 3
                . id =
                  | x -> x
                  | y -> y * 2
                """
            ),
            Int(3),
        )

    def test_match_function_can_close_over_variables(self) -> None:
        self.assertEqual(
            self._run(
                """
        f 1 2
        . f = a ->
          | b -> a + b
        """
            ),
            Int(3),
        )

    def test_match_record_binds_var(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { x = 3 }
                . get_x =
                  | { x = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_record_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 3, y = 4 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            ),
            Int(12),
        )

    def test_match_record_with_extra_fields_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult rec
                . rec = { x = 3 }
                . mult =
                  | { x = x, y = y } -> x * y
                """
            )

    def test_match_record_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult rec
                . rec = { x = 4, y = 5 }
                . mult =
                  | { x = 3, y = y } -> 1
                  | { x = 4, y = y } -> 2
                """
            ),
            Int(2),
        )

    def test_match_record_with_non_record_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | { x = x } -> x
                """
            )

    def test_match_record_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                get_x rec
                . rec = { a = 3, b = 3 }
                . get_x =
                  | { a = x, b = x } -> x
                """
            ),
            Int(3),
        )

    def test_match_record_spread_binds_spread(self) -> None:
        self.assertEqual(self._run("(| { a=1, ...rest } -> rest) {a=1, b=2, c=3}"), Record({"b": Int(2), "c": Int(3)}))

    def test_match_list_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 4, 5]
                . mult =
                  | [1, x, 3, y, 5] -> x * y
                """
            ),
            Int(8),
        )

    def test_match_list_incorrect_length_does_not_match(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                mult xs
                . xs = [1, 2, 3]
                . mult =
                  | [1, 2] -> 1
                  | [1, 2, 3, 4] -> 1
                  | [1, 3] -> 1
                """
            )

    def test_match_list_with_constant(self) -> None:
        self.assertEqual(
            self._run(
                """
                middle xs
                . xs = [4, 5, 6]
                . middle =
                  | [1, x, 3] -> x
                  | [4, x, 6] -> x
                  | [7, x, 9] -> x
                """
            ),
            Int(5),
        )

    def test_match_list_with_non_list_fails(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
                get_x 3
                . get_x =
                  | [2, x] -> x
                """
            )

    def test_match_list_doubly_binds_vars(self) -> None:
        self.assertEqual(
            self._run(
                """
                mult xs
                . xs = [1, 2, 3, 2, 1]
                . mult =
                  | [1, x, 3, x, 1] -> x
                """
            ),
            Int(2),
        )

    def test_match_list_spread_binds_spread(self) -> None:
        self.assertEqual(self._run("(| [x, ...xs] -> xs) [1, 2]"), List([Int(2)]))

    def test_pipe(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2)"), Int(3))

    def test_pipe_nested(self) -> None:
        self.assertEqual(self._run("1 |> (a -> a + 2) |> (b -> b * 2)"), Int(6))

    def test_reverse_pipe(self) -> None:
        self.assertEqual(self._run("(a -> a + 2) <| 1"), Int(3))

    def test_reverse_pipe_nested(self) -> None:
        self.assertEqual(self._run("(b -> b * 2) <| (a -> a + 2) <| 1"), Int(6))

    def test_function_can_reference_itself(self) -> None:
        result = self._run(
            """
    f 1
    . f = n -> f
    """,
            {},
        )
        self.assertEqual(result, Closure({"f": result}, Function(Var("n"), Var("f"))))

    def test_function_can_call_itself(self) -> None:
        with self.assertRaises(RecursionError):
            self._run(
                """
        f 1
        . f = n -> f n
        """
            )

    def test_match_function_can_call_itself(self) -> None:
        self.assertEqual(
            self._run(
                """
        fac 5
        . fac =
          | 0 -> 1
          | 1 -> 1
          | n -> n * fac (n - 1)
        """
            ),
            Int(120),
        )

    def test_list_access_binds_tighter_than_append(self) -> None:
        self.assertEqual(self._run("[1, 2, 3] +< xs@0 . xs = [4]"), List([Int(1), Int(2), Int(3), Int(4)]))

    def test_exponentiation(self) -> None:
        self.assertEqual(self._run("6 ^ 2"), Int(36))

    def test_modulus(self) -> None:
        self.assertEqual(self._run("11 % 3"), Int(2))

    def test_exp_binds_tighter_than_mul(self) -> None:
        self.assertEqual(self._run("5 * 2 ^ 3"), Int(40))

    def test_variant_true_returns_true(self) -> None:
        self.assertEqual(self._run("# true ()", {}), TRUE)

    def test_variant_false_returns_false(self) -> None:
        self.assertEqual(self._run("#false ()", {}), FALSE)

    def test_boolean_and_binds_tighter_than_or(self) -> None:
        self.assertEqual(self._run("#true () || #true () && boom", {}), TRUE)

    def test_compare_binds_tighter_than_boolean_and(self) -> None:
        self.assertEqual(self._run("1 < 2 && 2 < 1"), FALSE)

    def test_match_list_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        f [2, 4, 6]
        . f =
          | [] -> 0
          | [x, ...] -> x
          | c -> 1
        """
            ),
            Int(2),
        )

    def test_match_list_named_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        tail [1,2,3]
        . tail =
          | [first, ...rest] -> rest
        """
            ),
            List([Int(2), Int(3)]),
        )

    def test_match_record_spread(self) -> None:
        self.assertEqual(
            self._run(
                """
        f {x = 4, y = 5}
        . f =
          | {} -> 0
          | {x = a, ...} -> a
          | c -> 1
        """
            ),
            Int(4),
        )

    def test_match_expr_as_boolean_variants(self) -> None:
        self.assertEqual(
            self._run(
                """
        say (1 < 2)
        . say =
          | #false () -> "oh no"
          | #true () -> "omg"
        """
            ),
            String("omg"),
        )

    def test_match_variant_record(self) -> None:
        self.assertEqual(
            self._run(
                """
        f #add {x = 3, y = 4}
        . f =
          | # b () -> "foo"
          | #add {x = x, y = y} -> x + y
        """
            ),
            Int(7),
        )

    def test_int_div_returns_float(self) -> None:
        self.assertEqual(self._run("1 / 2 + 3"), Float(3.5))
        with self.assertRaisesRegex(InferenceError, "int and float"):
            self._run("1 / 2 + 3", check=True)

    def test_eval_count_bits_function_preserves_source_extents(self) -> None:
        env_object = self._run(
            """
        count_bits = counts ->
          | [1, ...bits] -> count_bits { zeros = counts@zeros, ones = 1 + counts@ones } bits
          | [0, ...bits] -> count_bits { zeros = 1 + counts@zeros, ones = counts@ones } bits
          | [] -> counts
        """
        )

        outer_function = env_object.env["count_bits"].func
        outer_function_source_extent = SourceExtent(
            start=SourceLocation(lineno=2, colno=22, byteno=22),
            end=SourceLocation(lineno=5, colno=24, byteno=241),
        )

        match_function_one = outer_function.body.cases[0]
        match_function_one_source_extent = SourceExtent(
            start=SourceLocation(lineno=3, colno=11, byteno=42), end=SourceLocation(lineno=3, colno=92, byteno=123)
        )

        match_function_two = outer_function.body.cases[1]
        match_function_two_source_extent = SourceExtent(
            start=SourceLocation(lineno=4, colno=11, byteno=135), end=SourceLocation(lineno=4, colno=92, byteno=216)
        )

        match_function_three = outer_function.body.cases[2]
        match_function_three_source_extent = SourceExtent(
            start=SourceLocation(lineno=5, colno=11, byteno=228), end=SourceLocation(lineno=5, colno=24, byteno=241)
        )

        match_functions = [match_function_one, match_function_two, match_function_three]
        match_function_source_extents = [
            match_function_one_source_extent,
            match_function_two_source_extent,
            match_function_three_source_extent,
        ]

        self.assertTrue(outer_function.source_extent == outer_function_source_extent)
        self.assertTrue(
            all(
                match_function.source_extent == source_extent
                for match_function, source_extent in zip(match_functions, match_function_source_extents)
            )
        )

    def test_eval_collatz_function_preserves_source_extents(self) -> None:
        env_object = self._run(
            """
        collatz = count ->
          | 1 -> count
          | n -> (n % 2 == 0) |> | #true () -> collatz (count + 1) (n // 2)
                                 | #false () -> collatz (count + 1) (3 * n + 1)
        """
        )

        outer_function = env_object.env["collatz"].func
        outer_function_source_extent = SourceExtent(
            start=SourceLocation(lineno=2, colno=19, byteno=19),
            end=SourceLocation(lineno=5, colno=79, byteno=205),
        )

        apply_ast = outer_function.body.cases[1].body
        arg = apply_ast.arg
        func = apply_ast.func

        arg_source_extent = SourceExtent(
            start=SourceLocation(lineno=4, colno=18, byteno=68), end=SourceLocation(lineno=4, colno=29, byteno=79)
        )

        func_source_extent = SourceExtent(
            start=SourceLocation(lineno=4, colno=34, byteno=84), end=SourceLocation(lineno=5, colno=79, byteno=205)
        )

        self.assertTrue(outer_function.source_extent == outer_function_source_extent)
        self.assertTrue(arg.source_extent == arg_source_extent)
        self.assertTrue(func.source_extent == func_source_extent)


class ClosureOptimizeTests(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(free_in(Int(1)), set())

    def test_float(self) -> None:
        self.assertEqual(free_in(Float(1.0)), set())

    def test_string(self) -> None:
        self.assertEqual(free_in(String("x")), set())

    def test_bytes(self) -> None:
        self.assertEqual(free_in(Bytes(b"x")), set())

    def test_hole(self) -> None:
        self.assertEqual(free_in(Hole()), set())

    def test_spread(self) -> None:
        self.assertEqual(free_in(Spread()), set())

    def test_spread_name(self) -> None:
        # TODO(max): Should this be assumed to always be in a place where it
        # defines a name, and therefore never have free variables?
        self.assertEqual(free_in(Spread("x")), {"x"})

    def test_nativefunction(self) -> None:
        self.assertEqual(free_in(NativeFunction("id", lambda x: x)), set())

    def test_variant(self) -> None:
        self.assertEqual(free_in(Variant("x", Var("y"))), {"y"})

    def test_var(self) -> None:
        self.assertEqual(free_in(Var("x")), {"x"})

    def test_binop(self) -> None:
        self.assertEqual(free_in(Binop(BinopKind.ADD, Var("x"), Var("y"))), {"x", "y"})

    def test_empty_list(self) -> None:
        self.assertEqual(free_in(List([])), set())

    def test_list(self) -> None:
        self.assertEqual(free_in(List([Var("x"), Var("y")])), {"x", "y"})

    def test_empty_record(self) -> None:
        self.assertEqual(free_in(Record({})), set())

    def test_record(self) -> None:
        self.assertEqual(free_in(Record({"x": Var("x"), "y": Var("y")})), {"x", "y"})

    def test_function(self) -> None:
        exp = parse(tokenize("x -> x + y"))
        self.assertEqual(free_in(exp), {"y"})

    def test_nested_function(self) -> None:
        exp = parse(tokenize("x -> y -> x + y + z"))
        self.assertEqual(free_in(exp), {"z"})

    def test_match_function(self) -> None:
        exp = parse(tokenize("| 1 -> x | 2 -> y | x -> 3 | z -> 4"))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_match_case_int(self) -> None:
        exp = MatchCase(Int(1), Var("x"))
        self.assertEqual(free_in(exp), {"x"})

    def test_match_case_var(self) -> None:
        exp = MatchCase(Var("x"), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_list(self) -> None:
        exp = MatchCase(List([Var("x")]), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_list_spread(self) -> None:
        exp = MatchCase(List([Spread()]), Binop(BinopKind.ADD, Var("xs"), Var("y")))
        self.assertEqual(free_in(exp), {"xs", "y"})

    def test_match_case_list_spread_name(self) -> None:
        exp = MatchCase(List([Spread("xs")]), Binop(BinopKind.ADD, Var("xs"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_match_case_record(self) -> None:
        exp = MatchCase(
            Record({"x": Int(1), "y": Var("y"), "a": Var("z")}),
            Binop(BinopKind.ADD, Binop(BinopKind.ADD, Var("x"), Var("y")), Var("z")),
        )
        self.assertEqual(free_in(exp), {"x"})

    def test_match_case_record_spread(self) -> None:
        exp = MatchCase(Record({"...": Spread()}), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_match_case_record_spread_name(self) -> None:
        exp = MatchCase(Record({"...": Spread("x")}), Binop(BinopKind.ADD, Var("x"), Var("y")))
        self.assertEqual(free_in(exp), {"y"})

    def test_apply(self) -> None:
        self.assertEqual(free_in(Apply(Var("x"), Var("y"))), {"x", "y"})

    def test_access(self) -> None:
        self.assertEqual(free_in(Access(Var("x"), Var("y"))), {"x", "y"})

    def test_where(self) -> None:
        exp = parse(tokenize("x . x = 1"))
        self.assertEqual(free_in(exp), set())

    def test_where_same_name(self) -> None:
        exp = parse(tokenize("x . x = x+y"))
        self.assertEqual(free_in(exp), {"x", "y"})

    def test_assign(self) -> None:
        exp = Assign(Var("x"), Int(1))
        self.assertEqual(free_in(exp), set())

    def test_assign_same_name(self) -> None:
        exp = Assign(Var("x"), Var("x"))
        self.assertEqual(free_in(exp), {"x"})

    def test_closure(self) -> None:
        # TODO(max): Should x be considered free in the closure if it's in the
        # env?
        exp = Closure({"x": Int(1)}, Function(Var("_"), List([Var("x"), Var("y")])))
        self.assertEqual(free_in(exp), {"x", "y"})


class StdLibTests(EndToEndTestsBase):
    def test_stdlib_add(self) -> None:
        self.assertEqual(self._run("$$add 3 4", STDLIB), Int(7))

    def test_stdlib_quote(self) -> None:
        self.assertEqual(self._run("$$quote (3 + 4)"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_pipe(self) -> None:
        self.assertEqual(self._run("3 + 4 |> $$quote"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_quote_reverse_pipe(self) -> None:
        self.assertEqual(self._run("$$quote <| 3 + 4"), Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_stdlib_serialize(self) -> None:
        self.assertEqual(self._run("$$serialize 3", STDLIB), Bytes(value=b"i\x06"))

    def test_stdlib_serialize_expr(self) -> None:
        self.assertEqual(
            self._run("(1+2) |> $$quote |> $$serialize", STDLIB),
            Bytes(value=b"+\x02+i\x02i\x04"),
        )

    def test_stdlib_deserialize(self) -> None:
        self.assertEqual(self._run("$$deserialize ~~aQY="), Int(3))

    def test_stdlib_deserialize_expr(self) -> None:
        self.assertEqual(self._run("$$deserialize ~~KwIraQJpBA=="), Binop(BinopKind.ADD, Int(1), Int(2)))

    def test_stdlib_listlength_empty_list_returns_zero(self) -> None:
        self.assertEqual(self._run("$$listlength []", STDLIB), Int(0))

    def test_stdlib_listlength_returns_length(self) -> None:
        self.assertEqual(self._run("$$listlength [1,2,3]", STDLIB), Int(3))

    def test_stdlib_listlength_with_non_list_raises_type_error(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self._run("$$listlength 1", STDLIB)
        self.assertEqual(ctx.exception.args[0], "listlength expected List, but got Int")


class PreludeTests(EndToEndTestsBase):
    def test_id_returns_input(self) -> None:
        self.assertEqual(self._run("id 123"), Int(123))

    def test_filter_returns_matching(self) -> None:
        self.assertEqual(
            self._run(
                """
        filter (x -> x < 4) [2, 6, 3, 7, 1, 8]
        """
            ),
            List([Int(2), Int(3), Int(1)]),
        )

    def test_filter_with_function_returning_non_bool_raises_match_error(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
        filter (x -> #no ()) [1]
        """
            )

    def test_quicksort(self) -> None:
        self.assertEqual(
            self._run(
                """
        quicksort [2, 6, 3, 7, 1, 8]
        """
            ),
            List([Int(1), Int(2), Int(3), Int(6), Int(7), Int(8)]),
        )

    def test_quicksort_with_empty_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        quicksort []
        """
            ),
            List([]),
        )

    def test_quicksort_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        quicksort ["a", "c", "b"]
        """
            )

    def test_concat(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [1, 2, 3] [4, 5, 6]
        """
            ),
            List([Int(1), Int(2), Int(3), Int(4), Int(5), Int(6)]),
        )

    def test_concat_with_first_list_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [] [4, 5, 6]
        """
            ),
            List([Int(4), Int(5), Int(6)]),
        )

    def test_concat_with_second_list_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [1, 2, 3] []
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_concat_with_both_lists_empty(self) -> None:
        self.assertEqual(
            self._run(
                """
        concat [] []
        """
            ),
            List([]),
        )

    def test_map(self) -> None:
        self.assertEqual(
            self._run(
                """
        map (x -> x * 2) [3, 1, 2]
        """
            ),
            List([Int(6), Int(2), Int(4)]),
        )

    def test_map_with_non_function_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        map 4 [3, 1, 2]
        """
            )

    def test_map_with_non_list_raises_match_error(self) -> None:
        with self.assertRaises(MatchError):
            self._run(
                """
        map (x -> x * 2) 3
        """
            )

    def test_range(self) -> None:
        self.assertEqual(
            self._run(
                """
        range 3
        """
            ),
            List([Int(0), Int(1), Int(2)]),
        )

    def test_range_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        range "a"
        """
            )

    def test_foldr(self) -> None:
        self.assertEqual(
            self._run(
                """
        foldr (x -> a -> a + x) 0 [1, 2, 3]
        """
            ),
            Int(6),
        )

    def test_foldr_on_empty_list_returns_empty_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        foldr (x -> a -> a + x) 0 []
        """
            ),
            Int(0),
        )

    def test_take(self) -> None:
        self.assertEqual(
            self._run(
                """
        take 3 [1, 2, 3, 4, 5]
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_take_n_more_than_list_length_returns_full_list(self) -> None:
        self.assertEqual(
            self._run(
                """
        take 5 [1, 2, 3]
        """
            ),
            List([Int(1), Int(2), Int(3)]),
        )

    def test_take_with_non_int_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        take "a" [1, 2, 3]
        """
            )

    def test_all_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x < 5) [1, 2, 3, 4]
        """
            ),
            TRUE,
        )

    def test_all_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x < 5) [2, 4, 6]
        """
            ),
            FALSE,
        )

    def test_all_with_empty_list_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x == 5) []
        """
            ),
            TRUE,
        )

    def test_all_with_non_bool_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        all (x -> x) [1, 2, 3]
        """
            )

    def test_all_short_circuits(self) -> None:
        self.assertEqual(
            self._run(
                """
        all (x -> x > 1) [1, "a", "b"]
        """
            ),
            FALSE,
        )

    def test_any_returns_true(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x < 4) [1, 3, 5]
        """
            ),
            TRUE,
        )

    def test_any_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x < 3) [4, 5, 6]
        """
            ),
            FALSE,
        )

    def test_any_with_empty_list_returns_false(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x == 5) []
        """
            ),
            FALSE,
        )

    def test_any_with_non_bool_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self._run(
                """
        any (x -> x) [1, 2, 3]
        """
            )

    def test_any_short_circuits(self) -> None:
        self.assertEqual(
            self._run(
                """
        any (x -> x > 1) [2, "a", "b"]
        """
            ),
            Variant("true", Hole()),
        )

    def test_mul_and_div_have_left_to_right_precedence(self) -> None:
        self.assertEqual(
            self._run(
                """
        1 / 3 * 3
        """
            ),
            Float(1.0),
        )


class TypeStrTests(unittest.TestCase):
    def test_tyvar(self) -> None:
        self.assertEqual(str(TyVar("a")), "'a")

    def test_tycon(self) -> None:
        self.assertEqual(str(TyCon("int", [])), "int")

    def test_tycon_one_arg(self) -> None:
        self.assertEqual(str(TyCon("list", [IntType])), "(int list)")

    def test_tycon_args(self) -> None:
        self.assertEqual(str(TyCon("->", [IntType, IntType])), "(int->int)")

    def test_tyrow_empty_closed(self) -> None:
        self.assertEqual(str(TyEmptyRow()), "{}")

    def test_tyrow_empty_open(self) -> None:
        self.assertEqual(str(TyRow({}, TyVar("a"))), "{...'a}")

    def test_tyrow_closed(self) -> None:
        self.assertEqual(str(TyRow({"x": IntType, "y": StringType})), "{x=int, y=string}")

    def test_tyrow_open(self) -> None:
        self.assertEqual(str(TyRow({"x": IntType, "y": StringType}, TyVar("a"))), "{x=int, y=string, ...'a}")

    def test_tyrow_chain(self) -> None:
        inner = TyRow({"x": IntType})
        inner_var = TyVar("a")
        inner_var.make_equal_to(inner)
        outer = TyRow({"y": StringType}, inner_var)
        self.assertEqual(str(outer), "{x=int, y=string}")

    def test_forall(self) -> None:
        self.assertEqual(str(Forall([TyVar("a"), TyVar("b")], TyVar("a"))), "(forall 'a, 'b. 'a)")


class InferTypeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_tyvar_counter()

    def test_unify_tyvar_tyvar(self) -> None:
        a = TyVar("a")
        b = TyVar("b")
        unify_type(a, b)
        self.assertIs(a.find(), b.find())

    def test_unify_tyvar_tycon(self) -> None:
        a = TyVar("a")
        unify_type(a, IntType)
        self.assertIs(a.find(), IntType)
        b = TyVar("b")
        unify_type(b, IntType)
        self.assertIs(b.find(), IntType)

    def test_unify_tycon_tycon_name_mismatch(self) -> None:
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            unify_type(IntType, StringType)

    def test_unify_tycon_tycon_arity_mismatch(self) -> None:
        l = TyCon("x", [TyVar("a")])
        r = TyCon("x", [])
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            unify_type(l, r)

    def test_unify_tycon_tycon_unifies_arg(self) -> None:
        a = TyVar("a")
        b = TyVar("b")
        l = TyCon("x", [a])
        r = TyCon("x", [b])
        unify_type(l, r)
        self.assertIs(a.find(), b.find())

    def test_unify_tycon_tycon_unifies_args(self) -> None:
        a, b, c, d = map(TyVar, "abcd")
        l = func_type(a, b)
        r = func_type(c, d)
        unify_type(l, r)
        self.assertIs(a.find(), c.find())
        self.assertIs(b.find(), d.find())
        self.assertIsNot(a.find(), b.find())

    def test_unify_recursive_fails(self) -> None:
        l = TyVar("a")
        r = TyCon("x", [TyVar("a")])
        with self.assertRaisesRegex(InferenceError, "Occurs check failed"):
            unify_type(l, r)

    def test_unify_empty_row(self) -> None:
        unify_type(TyEmptyRow(), TyEmptyRow())

    def test_unify_empty_row_open(self) -> None:
        l = TyRow({}, TyVar("a"))
        r = TyRow({}, TyVar("b"))
        unify_type(l, r)
        self.assertIs(l.rest.find(), r.rest.find())

    def test_unify_row_unifies_fields(self) -> None:
        a = TyVar("a")
        b = TyVar("b")
        l = TyRow({"x": a})
        r = TyRow({"x": b})
        unify_type(l, r)
        self.assertIs(a.find(), b.find())

    def test_unify_empty_right(self) -> None:
        l = TyRow({"x": IntType})
        r = TyEmptyRow()
        with self.assertRaisesRegex(InferenceError, "Unifying row {x=int} with empty row"):
            unify_type(l, r)

    def test_unify_empty_left(self) -> None:
        l = TyEmptyRow()
        r = TyRow({"x": IntType})
        with self.assertRaisesRegex(InferenceError, "Unifying empty row with row {x=int}"):
            unify_type(l, r)

    def test_unify_missing_closed(self) -> None:
        l = TyRow({"x": IntType})
        r = TyRow({"y": IntType})
        with self.assertRaisesRegex(InferenceError, "Unifying empty row with row {y=int, ...'t0}"):
            unify_type(l, r)

    def test_unify_one_open_one_closed(self) -> None:
        rest = TyVar("r1")
        l = TyRow({"x": IntType})
        r = TyRow({"x": IntType}, rest)
        unify_type(l, r)
        self.assertTyEqual(rest.find(), TyEmptyRow())

    def test_unify_one_open_more_than_one_closed(self) -> None:
        rest = TyVar("r1")
        l = TyRow({"x": IntType})
        r = TyRow({"x": IntType, "y": StringType}, rest)
        with self.assertRaisesRegex(InferenceError, "Unifying empty row with row {y=string, ...'r1}"):
            unify_type(l, r)

    def test_unify_one_open_one_closed_more(self) -> None:
        rest = TyVar("r1")
        l = TyRow({"x": IntType, "y": StringType})
        r = TyRow({"x": IntType}, rest)
        unify_type(l, r)
        self.assertTyEqual(rest.find(), TyRow({"y": StringType}))

    def test_unify_left_missing_open(self) -> None:
        l = TyRow({}, TyVar("r0"))
        r = TyRow({"y": IntType}, TyVar("r1"))
        unify_type(l, r)
        self.assertTyEqual(l.rest, TyRow({"y": IntType}, TyVar("r1")))
        assert isinstance(r.rest, TyVar)
        self.assertTrue(r.rest.is_unbound())

    def test_unify_right_missing_open(self) -> None:
        l = TyRow({"x": IntType}, TyVar("r0"))
        r = TyRow({}, TyVar("r1"))
        unify_type(l, r)
        assert isinstance(l.rest, TyVar)
        self.assertTrue(l.rest.is_unbound())
        self.assertTyEqual(r.rest, TyRow({"x": IntType}, TyVar("r0")))

    def test_unify_both_missing_open(self) -> None:
        l = TyRow({"x": IntType}, TyVar("r0"))
        r = TyRow({"y": IntType}, TyVar("r1"))
        unify_type(l, r)
        self.assertTyEqual(l.rest, TyRow({"y": IntType}, TyVar("t0")))
        self.assertTyEqual(r.rest, TyRow({"x": IntType}, TyVar("t0")))

    def test_minimize_tyvar(self) -> None:
        ty = fresh_tyvar()
        self.assertEqual(minimize(ty), TyVar("a"))

    def test_minimize_tycon(self) -> None:
        ty = func_type(TyVar("t0"), TyVar("t1"), TyVar("t0"))
        self.assertEqual(minimize(ty), func_type(TyVar("a"), TyVar("b"), TyVar("a")))

    def infer(self, expr: Object, ctx: Context) -> MonoType:
        return minimize(infer_type(expr, ctx))

    def assertTyEqual(self, l: MonoType, r: MonoType) -> bool:
        l = l.find()
        r = r.find()
        if isinstance(l, TyVar) and isinstance(r, TyVar):
            if l != r:
                self.fail(f"Type mismatch: {l} != {r}")
            return True
        if isinstance(l, TyCon) and isinstance(r, TyCon):
            if l.name != r.name:
                self.fail(f"Type mismatch: {l} != {r}")
            if len(l.args) != len(r.args):
                self.fail(f"Type mismatch: {l} != {r}")
            for l_arg, r_arg in zip(l.args, r.args):
                self.assertTyEqual(l_arg, r_arg)
            return True
        if isinstance(l, TyEmptyRow) and isinstance(r, TyEmptyRow):
            return True
        if isinstance(l, TyRow) and isinstance(r, TyRow):
            l_keys = set(l.fields.keys())
            r_keys = set(r.fields.keys())
            if l_keys != r_keys:
                self.fail(f"Type mismatch: {l} != {r}")
            for key in l_keys:
                self.assertTyEqual(l.fields[key], r.fields[key])
            self.assertTyEqual(l.rest, r.rest)
            return True
        self.fail(f"Type mismatch: {l} != {r}")

    def test_unbound_var(self) -> None:
        with self.assertRaisesRegex(InferenceError, "Unbound variable"):
            self.infer(Var("a"), {})

    def test_var_instantiates_scheme(self) -> None:
        ty = self.infer(Var("a"), {"a": Forall([TyVar("b")], TyVar("b"))})
        self.assertTyEqual(ty, TyVar("a"))

    def test_int(self) -> None:
        ty = self.infer(Int(123), {})
        self.assertTyEqual(ty, IntType)

    def test_float(self) -> None:
        ty = self.infer(Float(1.0), {})
        self.assertTyEqual(ty, FloatType)

    def test_string(self) -> None:
        ty = self.infer(String("abc"), {})
        self.assertTyEqual(ty, StringType)

    def test_function_returns_arg(self) -> None:
        ty = self.infer(Function(Var("x"), Var("x")), {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("a")))

    def test_nested_function_outer(self) -> None:
        ty = self.infer(Function(Var("x"), Function(Var("y"), Var("x"))), {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("b"), TyVar("a")))

    def test_nested_function_inner(self) -> None:
        ty = self.infer(Function(Var("x"), Function(Var("y"), Var("y"))), {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("b"), TyVar("b")))

    def test_apply_id_int(self) -> None:
        func = Function(Var("x"), Var("x"))
        arg = Int(123)
        ty = self.infer(Apply(func, arg), {})
        self.assertTyEqual(ty, IntType)

    def test_apply_two_arg_returns_function(self) -> None:
        func = Function(Var("x"), Function(Var("y"), Var("x")))
        arg = Int(123)
        ty = self.infer(Apply(func, arg), {})
        self.assertTyEqual(ty, func_type(TyVar("a"), IntType))

    def test_binop_add_constrains_int(self) -> None:
        expr = Binop(BinopKind.ADD, Var("x"), Var("y"))
        ty = self.infer(
            expr,
            {
                "x": Forall([], TyVar("a")),
                "y": Forall([], TyVar("b")),
                "+": Forall([], func_type(IntType, IntType, IntType)),
            },
        )
        self.assertTyEqual(ty, IntType)

    def test_binop_add_function_constrains_int(self) -> None:
        x = Var("x")
        y = Var("y")
        expr = Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, x, y)))
        ty = self.infer(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(IntType, IntType, IntType))
        self.assertTyEqual(type_of(x), IntType)
        self.assertTyEqual(type_of(y), IntType)

    def test_let(self) -> None:
        expr = Where(Var("f"), Assign(Var("f"), Function(Var("x"), Var("x"))))
        ty = self.infer(expr, {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("a")))

    def test_apply_monotype_to_different_types_raises(self) -> None:
        expr = Where(
            Where(Var("x"), Assign(Var("x"), Apply(Var("f"), Int(123)))),
            Assign(Var("y"), Apply(Var("f"), Float(123.0))),
        )
        ctx = {"f": Forall([], func_type(TyVar("a"), TyVar("a")))}
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            self.infer(expr, ctx)

    def test_apply_polytype_to_different_types(self) -> None:
        expr = Where(
            Where(Var("x"), Assign(Var("x"), Apply(Var("f"), Int(123)))),
            Assign(Var("y"), Apply(Var("f"), Float(123.0))),
        )
        ty = self.infer(expr, {"f": Forall([TyVar("a")], func_type(TyVar("a"), TyVar("a")))})
        self.assertTyEqual(ty, IntType)

    def test_generalization(self) -> None:
        # From https://okmij.org/ftp/ML/generalization.html
        expr = parse(tokenize("x -> (y . y = x)"))
        ty = self.infer(expr, {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("a")))

    def test_generalization2(self) -> None:
        # From https://okmij.org/ftp/ML/generalization.html
        expr = parse(tokenize("x -> (y . y = z -> x z)"))
        ty = self.infer(expr, {})
        self.assertTyEqual(ty, func_type(func_type(TyVar("a"), TyVar("b")), func_type(TyVar("a"), TyVar("b"))))

    def test_id(self) -> None:
        expr = Function(Var("x"), Var("x"))
        ty = self.infer(expr, {})
        self.assertTyEqual(ty, func_type(TyVar("a"), TyVar("a")))

    def test_empty_list(self) -> None:
        expr = List([])
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, TyCon("list", [TyVar("t0")]))

    def test_list_int(self) -> None:
        expr = List([Int(123)])
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, TyCon("list", [IntType]))

    def test_list_mismatch(self) -> None:
        expr = List([Int(123), Float(123.0)])
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            infer_type(expr, {})

    def test_recursive_fact(self) -> None:
        expr = parse(tokenize("fact . fact = | 0 -> 1 | n -> n * fact (n-1)"))
        ty = infer_type(
            expr,
            {
                "*": Forall([], func_type(IntType, IntType, IntType)),
                "-": Forall([], func_type(IntType, IntType, IntType)),
            },
        )
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_match_int_int(self) -> None:
        expr = parse(tokenize("| 0 -> 1"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_match_int_int_two_cases(self) -> None:
        expr = parse(tokenize("| 0 -> 1 | 1 -> 2"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_match_int_int_int_float(self) -> None:
        expr = parse(tokenize("| 0 -> 1 | 1 -> 2.0"))
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            infer_type(expr, {})

    def test_match_int_int_float_int(self) -> None:
        expr = parse(tokenize("| 0 -> 1 | 1.0 -> 2"))
        with self.assertRaisesRegex(InferenceError, "Unification failed"):
            infer_type(expr, {})

    def test_match_var(self) -> None:
        expr = parse(tokenize("| x -> x + 1"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_match_int_var(self) -> None:
        expr = parse(tokenize("| 0 -> 1 | x -> x"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_match_list_of_int(self) -> None:
        expr = parse(tokenize("| [x] -> x + 1"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(list_type(IntType), IntType))

    def test_match_list_of_int_to_list(self) -> None:
        expr = parse(tokenize("| [x] -> [x + 1]"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(list_type(IntType), list_type(IntType)))

    def test_match_list_of_int_to_int(self) -> None:
        expr = parse(tokenize("| [] -> 0 | [x] -> 1 | [x, y] -> x+y"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(list_type(IntType), IntType))

    def test_recursive_var_is_unbound(self) -> None:
        expr = parse(tokenize("a . a = a"))
        with self.assertRaisesRegex(InferenceError, "Unbound variable"):
            infer_type(expr, {})

    def test_recursive(self) -> None:
        expr = parse(
            tokenize(
                """
        length
        . length =
        | [] -> 0
        | [x, ...xs] -> 1 + length xs
        """
            )
        )
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(list_type(TyVar("t8")), IntType))

    def test_match_list_to_list(self) -> None:
        expr = parse(tokenize("| [] -> [] | x -> x"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(list_type(TyVar("t1")), list_type(TyVar("t1"))))

    def test_match_list_spread(self) -> None:
        expr = parse(tokenize("head . head = | [x, ...] -> x"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(list_type(TyVar("t4")), TyVar("t4")))

    def test_match_list_spread_rest(self) -> None:
        expr = parse(tokenize("tail . tail = | [x, ...xs] -> xs"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(list_type(TyVar("t4")), list_type(TyVar("t4"))))

    def test_match_list_spread_named(self) -> None:
        expr = parse(tokenize("sum . sum = | [] -> 0 | [x, ...xs] -> x + sum xs"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(list_type(IntType), IntType))

    def test_match_list_int_to_list(self) -> None:
        expr = parse(tokenize("| [] -> [3] | x -> x"))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(list_type(IntType), list_type(IntType)))

    def test_inc(self) -> None:
        expr = parse(tokenize("inc . inc = | 0 -> 1 | 1 -> 2 | a -> a + 1"))
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(IntType, IntType))

    def test_bytes(self) -> None:
        expr = Bytes(b"abc")
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, BytesType)

    def test_hole(self) -> None:
        expr = Hole()
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, HoleType)

    def test_string_concat(self) -> None:
        expr = parse(tokenize('"a" ++ "b"'))
        ty = infer_type(expr, OP_ENV)
        self.assertTyEqual(ty, StringType)

    def test_cons(self) -> None:
        expr = parse(tokenize("1 >+ [2]"))
        ty = infer_type(expr, OP_ENV)
        self.assertTyEqual(ty, list_type(IntType))

    def test_append(self) -> None:
        expr = parse(tokenize("[1] +< 2"))
        ty = infer_type(expr, OP_ENV)
        self.assertTyEqual(ty, list_type(IntType))

    def test_record(self) -> None:
        expr = Record({"a": Int(1), "b": String("hello")})
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, TyRow({"a": IntType, "b": StringType}))

    def test_match_record(self) -> None:
        expr = MatchFunction(
            [
                MatchCase(
                    Record({"x": Var("x")}),
                    Var("x"),
                )
            ]
        )
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(TyRow({"x": TyVar("t1")}), TyVar("t1")))

    def test_access_poly(self) -> None:
        expr = Function(Var("r"), Access(Var("r"), Var("x")))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, func_type(TyRow({"x": TyVar("t1")}, TyVar("t2")), TyVar("t1")))

    def test_apply_row(self) -> None:
        row0 = Record({"x": Int(1)})
        row1 = Record({"x": Int(1), "y": Int(2)})
        scheme = Forall([], func_type(TyRow({"x": IntType}, TyVar("a")), IntType))
        ty0 = infer_type(Apply(Var("f"), row0), {"f": scheme})
        self.assertTyEqual(ty0, IntType)
        with self.assertRaisesRegex(InferenceError, "Unifying empty row with row {y=int}"):
            infer_type(Apply(Var("f"), row1), {"f": scheme})

    def test_apply_row_polymorphic(self) -> None:
        row0 = Record({"x": Int(1)})
        row1 = Record({"x": Int(1), "y": Int(2)})
        row2 = Record({"x": Int(1), "y": Int(2), "z": Int(3)})
        scheme = Forall([TyVar("a")], func_type(TyRow({"x": IntType}, TyVar("a")), IntType))
        ty0 = infer_type(Apply(Var("f"), row0), {"f": scheme})
        self.assertTyEqual(ty0, IntType)
        ty1 = infer_type(Apply(Var("f"), row1), {"f": scheme})
        self.assertTyEqual(ty1, IntType)
        ty2 = infer_type(Apply(Var("f"), row2), {"f": scheme})
        self.assertTyEqual(ty2, IntType)

    def test_example_rec_access(self) -> None:
        expr = parse(tokenize('rec@a . rec = { a = 1, b = "x" }'))
        ty = infer_type(expr, {})
        self.assertTyEqual(ty, IntType)

    def test_example_rec_access_poly(self) -> None:
        expr = parse(
            tokenize(
                """
(get_x {x=1}) + (get_x {x=2,y=3})
. get_x = | { x=x, ... } -> x
"""
            )
        )
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, IntType)

    def test_example_rec_access_poly_named_bug(self) -> None:
        expr = parse(
            tokenize(
                """
(filter_x {x=1, y=2}) + 3
. filter_x = | { x=x, ...xs } -> xs
"""
            )
        )
        with self.assertRaisesRegex(InferenceError, "Cannot unify int and {y=int}"):
            infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})

    def test_example_rec_access_rest(self) -> None:
        expr = parse(
            tokenize(
                """
| { x=x, ...xs } -> xs
"""
            )
        )
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, func_type(TyRow({"x": TyVar("t1")}, TyVar("t2")), TyVar("t2")))

    def test_example_match_rec_access_rest(self) -> None:
        expr = parse(
            tokenize(
                """
filter_x {x=1, y=2}
. filter_x = | { x=x, ...xs } -> xs
"""
            )
        )
        ty = infer_type(expr, {"+": Forall([], func_type(IntType, IntType, IntType))})
        self.assertTyEqual(ty, TyRow({"y": IntType}))

    def test_example_rec_access_poly_named(self) -> None:
        expr = parse(
            tokenize(
                """
[(filter_x {x=1, y=2}), (filter_x {x=2, y=3, z=4})]
. filter_x = | { x=x, ...xs } -> xs
"""
            )
        )
        with self.assertRaisesRegex(InferenceError, "Unifying empty row with row {z=int}"):
            infer_type(expr, {})


class SerializerTests(unittest.TestCase):
    def _serialize(self, obj: Object) -> bytes:
        serializer = Serializer()
        serializer.serialize(obj)
        return bytes(serializer.output)

    def test_short(self) -> None:
        self.assertEqual(self._serialize(Int(-1)), TYPE_SHORT + b"\x01")
        self.assertEqual(self._serialize(Int(0)), TYPE_SHORT + b"\x00")
        self.assertEqual(self._serialize(Int(1)), TYPE_SHORT + b"\x02")
        self.assertEqual(self._serialize(Int(-(2**33))), TYPE_SHORT + b"\xff\xff\xff\xff?")
        self.assertEqual(self._serialize(Int(2**33)), TYPE_SHORT + b"\x80\x80\x80\x80@")
        self.assertEqual(self._serialize(Int(-(2**63))), TYPE_SHORT + b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01")
        self.assertEqual(self._serialize(Int(2**63 - 1)), TYPE_SHORT + b"\xfe\xff\xff\xff\xff\xff\xff\xff\xff\x01")

    def test_long(self) -> None:
        self.assertEqual(
            self._serialize(Int(2**100)),
            TYPE_LONG + b"\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00",
        )
        self.assertEqual(
            self._serialize(Int(-(2**100))),
            TYPE_LONG + b"\x04\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x1f\x00\x00\x00",
        )

    def test_string(self) -> None:
        self.assertEqual(self._serialize(String("hello")), TYPE_STRING + b"\nhello")

    def test_empty_list(self) -> None:
        obj = List([])
        self.assertEqual(self._serialize(obj), ref(TYPE_LIST) + b"\x00")

    def test_list(self) -> None:
        obj = List([Int(123), Int(456)])
        self.assertEqual(self._serialize(obj), ref(TYPE_LIST) + b"\x04i\xf6\x01i\x90\x07")

    def test_self_referential_list(self) -> None:
        obj = List([])
        obj.items.append(obj)
        self.assertEqual(self._serialize(obj), ref(TYPE_LIST) + b"\x02r\x00")

    def test_variant(self) -> None:
        obj = Variant("abc", Int(123))
        self.assertEqual(self._serialize(obj), TYPE_VARIANT + b"\x06abci\xf6\x01")

    def test_record(self) -> None:
        obj = Record({"x": Int(1), "y": Int(2)})
        self.assertEqual(self._serialize(obj), TYPE_RECORD + b"\x04\x02xi\x02\x02yi\x04")

    def test_var(self) -> None:
        obj = Var("x")
        self.assertEqual(self._serialize(obj), TYPE_VAR + b"\x02x")

    def test_function(self) -> None:
        obj = Function(Var("x"), Var("x"))
        self.assertEqual(self._serialize(obj), TYPE_FUNCTION + b"v\x02xv\x02x")

    def test_empty_match_function(self) -> None:
        obj = MatchFunction([])
        self.assertEqual(self._serialize(obj), TYPE_MATCH_FUNCTION + b"\x00")

    def test_match_function(self) -> None:
        obj = MatchFunction([MatchCase(Int(1), Var("x")), MatchCase(List([Int(1)]), Var("y"))])
        self.assertEqual(self._serialize(obj), TYPE_MATCH_FUNCTION + b"\x04i\x02v\x02x\xdb\x02i\x02v\x02y")

    def test_closure(self) -> None:
        obj = Closure({}, Function(Var("x"), Var("x")))
        self.assertEqual(self._serialize(obj), ref(TYPE_CLOSURE) + b"fv\x02xv\x02x\x00")

    def test_self_referential_closure(self) -> None:
        obj = Closure({}, Function(Var("x"), Var("x")))
        assert isinstance(obj.env, dict)  # For mypy
        obj.env["self"] = obj
        self.assertEqual(self._serialize(obj), ref(TYPE_CLOSURE) + b"fv\x02xv\x02x\x02\x08selfr\x00")

    def test_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(self._serialize(obj), TYPE_BYTES + b"\x06abc")

    def test_float(self) -> None:
        obj = Float(3.14)
        self.assertEqual(self._serialize(obj), TYPE_FLOAT + b"\x1f\x85\xebQ\xb8\x1e\t@")

    def test_hole(self) -> None:
        self.assertEqual(self._serialize(Hole()), TYPE_HOLE)

    def test_assign(self) -> None:
        obj = Assign(Var("x"), Int(123))
        self.assertEqual(self._serialize(obj), TYPE_ASSIGN + b"v\x02xi\xf6\x01")

    def test_binop(self) -> None:
        obj = Binop(BinopKind.ADD, Int(3), Int(4))
        self.assertEqual(self._serialize(obj), TYPE_BINOP + b"\x02+i\x06i\x08")

    def test_apply(self) -> None:
        obj = Apply(Var("f"), Var("x"))
        self.assertEqual(self._serialize(obj), TYPE_APPLY + b"v\x02fv\x02x")

    def test_where(self) -> None:
        obj = Where(Var("a"), Var("b"))
        self.assertEqual(self._serialize(obj), TYPE_WHERE + b"v\x02av\x02b")

    def test_access(self) -> None:
        obj = Access(Var("a"), Var("b"))
        self.assertEqual(self._serialize(obj), TYPE_ACCESS + b"v\x02av\x02b")

    def test_spread(self) -> None:
        self.assertEqual(self._serialize(Spread()), TYPE_SPREAD)
        self.assertEqual(self._serialize(Spread("rest")), TYPE_NAMED_SPREAD + b"\x08rest")


class RoundTripSerializationTests(unittest.TestCase):
    def _serialize(self, obj: Object) -> bytes:
        serializer = Serializer()
        serializer.serialize(obj)
        return bytes(serializer.output)

    def _deserialize(self, flat: bytes) -> Object:
        deserializer = Deserializer(flat)
        return deserializer.parse()

    def _serde(self, obj: Object) -> Object:
        flat = self._serialize(obj)
        return self._deserialize(flat)

    def _rt(self, obj: Object) -> None:
        result = self._serde(obj)
        self.assertEqual(result, obj)

    def test_short(self) -> None:
        for i in range(-(2**16), 2**16):
            self._rt(Int(i))

        self._rt(Int(-(2**63)))
        self._rt(Int(2**63 - 1))

    def test_long(self) -> None:
        self._rt(Int(2**100))
        self._rt(Int(-(2**100)))

    def test_string(self) -> None:
        self._rt(String(""))
        self._rt(String("a"))
        self._rt(String("hello"))

    def test_list(self) -> None:
        self._rt(List([]))
        self._rt(List([Int(123), Int(345)]))

    def test_self_referential_list(self) -> None:
        ls = List([])
        ls.items.append(ls)
        result = self._serde(ls)
        self.assertIsInstance(result, List)
        assert isinstance(result, List)  # For mypy
        self.assertIsInstance(result.items, list)
        self.assertEqual(len(result.items), 1)
        self.assertIs(result.items[0], result)

    def test_record(self) -> None:
        self._rt(Record({"x": Int(1), "y": Int(2)}))

    def test_variant(self) -> None:
        self._rt(Variant("abc", Int(123)))

    def test_var(self) -> None:
        self._rt(Var("x"))

    def test_function(self) -> None:
        self._rt(Function(Var("x"), Var("x")))

    def test_empty_match_function(self) -> None:
        self._rt(MatchFunction([]))

    def test_match_function(self) -> None:
        obj = MatchFunction([MatchCase(Int(1), Var("x")), MatchCase(List([Int(1)]), Var("y"))])
        self._rt(obj)

    def test_closure(self) -> None:
        self._rt(Closure({}, Function(Var("x"), Var("x"))))

    def test_self_referential_closure(self) -> None:
        obj = Closure({}, Function(Var("x"), Var("x")))
        assert isinstance(obj.env, dict)  # For mypy
        obj.env["self"] = obj
        result = self._serde(obj)
        self.assertIsInstance(result, Closure)
        assert isinstance(result, Closure)  # For mypy
        self.assertIsInstance(result.env, dict)
        self.assertEqual(len(result.env), 1)
        self.assertIs(result.env["self"], result)

    def test_bytes(self) -> None:
        self._rt(Bytes(b"abc"))

    def test_float(self) -> None:
        self._rt(Float(3.14))

    def test_hole(self) -> None:
        self._rt(Hole())

    def test_assign(self) -> None:
        self._rt(Assign(Var("x"), Int(123)))

    def test_binop(self) -> None:
        self._rt(Binop(BinopKind.ADD, Int(3), Int(4)))

    def test_apply(self) -> None:
        self._rt(Apply(Var("f"), Var("x")))

    def test_where(self) -> None:
        self._rt(Where(Var("a"), Var("b")))

    def test_access(self) -> None:
        self._rt(Access(Var("a"), Var("b")))

    def test_spread(self) -> None:
        self._rt(Spread())
        self._rt(Spread("rest"))


class ScrapMonadTests(unittest.TestCase):
    def test_create_copies_env(self) -> None:
        env = {"a": Int(123)}
        result = ScrapMonad(env)
        self.assertEqual(result.env, env)
        self.assertIsNot(result.env, env)

    def test_bind_returns_new_monad(self) -> None:
        env = {"a": Int(123)}
        orig = ScrapMonad(env)
        result, next_monad = orig.bind(Assign(Var("b"), Int(456)))
        self.assertEqual(orig.env, {"a": Int(123)})
        self.assertEqual(next_monad.env, {"a": Int(123), "b": Int(456)})


class PrettyPrintTests(unittest.TestCase):
    def test_pretty_print_int(self) -> None:
        obj = Int(1)
        self.assertEqual(pretty(obj), "1")

    def test_pretty_print_float(self) -> None:
        obj = Float(3.14)
        self.assertEqual(pretty(obj), "3.14")

    def test_pretty_print_string(self) -> None:
        obj = String("hello")
        self.assertEqual(pretty(obj), '"hello"')

    def test_pretty_print_bytes(self) -> None:
        obj = Bytes(b"abc")
        self.assertEqual(pretty(obj), "~~YWJj")

    def test_pretty_print_var(self) -> None:
        obj = Var("ref")
        self.assertEqual(pretty(obj), "ref")

    def test_pretty_print_hole(self) -> None:
        obj = Hole()
        self.assertEqual(pretty(obj), "()")

    def test_pretty_print_spread(self) -> None:
        obj = Spread()
        self.assertEqual(pretty(obj), "...")

    def test_pretty_print_named_spread(self) -> None:
        obj = Spread("rest")
        self.assertEqual(pretty(obj), "...rest")

    def test_pretty_print_binop(self) -> None:
        obj = Binop(BinopKind.ADD, Int(1), Int(2))
        self.assertEqual(pretty(obj), "1 + 2")

    def test_pretty_print_binop_precedence(self) -> None:
        obj = Binop(BinopKind.ADD, Int(1), Binop(BinopKind.MUL, Int(2), Int(3)))
        self.assertEqual(pretty(obj), "1 + 2 * 3")
        obj = Binop(BinopKind.MUL, Binop(BinopKind.ADD, Int(1), Int(2)), Int(3))
        self.assertEqual(pretty(obj), "(1 + 2) * 3")

    def test_pretty_print_int_list(self) -> None:
        obj = List([Int(1), Int(2), Int(3)])
        self.assertEqual(pretty(obj), "[1, 2, 3]")

    def test_pretty_print_str_list(self) -> None:
        obj = List([String("1"), String("2"), String("3")])
        self.assertEqual(pretty(obj), '["1", "2", "3"]')

    def test_pretty_print_recursion(self) -> None:
        obj = List([])
        obj.items.append(obj)
        self.assertEqual(pretty(obj), "[...]")

    def test_pretty_print_assign(self) -> None:
        obj = Assign(Var("x"), Int(3))
        self.assertEqual(pretty(obj), "x = 3")

    def test_pretty_print_function(self) -> None:
        obj = Function(Var("x"), Binop(BinopKind.ADD, Int(1), Var("x")))
        self.assertEqual(pretty(obj), "x -> 1 + x")

    def test_pretty_print_nested_function(self) -> None:
        obj = Function(Var("x"), Function(Var("y"), Binop(BinopKind.ADD, Var("x"), Var("y"))))
        self.assertEqual(pretty(obj), "x -> y -> x + y")

    def test_pretty_print_apply(self) -> None:
        obj = Apply(Var("x"), Var("y"))
        self.assertEqual(pretty(obj), "x y")

    def test_pretty_print_compose(self) -> None:
        gensym_reset()
        obj = parse(tokenize("(x -> x + 3) << (x -> x * 2)"))
        self.assertEqual(
            pretty(obj),
            "$v0 -> (x -> x + 3) ((x -> x * 2) $v0)",
        )
        gensym_reset()
        obj = parse(tokenize("(x -> x + 3) >> (x -> x * 2)"))
        self.assertEqual(
            pretty(obj),
            "$v0 -> (x -> x * 2) ((x -> x + 3) $v0)",
        )

    def test_pretty_print_where(self) -> None:
        obj = Where(
            Binop(BinopKind.ADD, Var("a"), Var("b")),
            Assign(Var("a"), Int(1)),
        )
        self.assertEqual(pretty(obj), "a + b . a = 1")

    def test_pretty_print_assert(self) -> None:
        obj = Assert(Int(123), Variant("true", String("foo")))
        self.assertEqual(pretty(obj), '123 ! #true "foo"')

    def test_pretty_print_envobject(self) -> None:
        obj = EnvObject({"x": Int(1)})
        self.assertEqual(pretty(obj), "EnvObject({'x': Int(value=1)})")

    def test_pretty_print_matchfunction(self) -> None:
        obj = MatchFunction([MatchCase(Var("y"), Var("x"))])
        self.assertEqual(pretty(obj), "| y -> x")

    def test_pretty_print_matchfunction_precedence(self) -> None:
        obj = MatchFunction(
            [
                MatchCase(Var("a"), MatchFunction([MatchCase(Var("b"), Var("c"))])),
                MatchCase(Var("x"), MatchFunction([MatchCase(Var("y"), Var("z"))])),
            ]
        )
        self.assertEqual(
            pretty(obj),
            """\
| a -> (| b -> c)
| x -> (| y -> z)""",
        )

    def test_pretty_print_relocation(self) -> None:
        obj = Relocation("relocate")
        self.assertEqual(pretty(obj), "Relocation(name='relocate')")

    def test_pretty_print_nativefunction(self) -> None:
        obj = NativeFunction("times2", lambda x: Int(x.value * 2))  # type: ignore [attr-defined]
        self.assertEqual(pretty(obj), "NativeFunction(name=times2)")

    def test_pretty_print_closure(self) -> None:
        obj = Closure({"a": Int(123)}, Function(Var("x"), Var("x")))
        self.assertEqual(pretty(obj), "Closure(['a'], x -> x)")

    def test_pretty_print_record(self) -> None:
        obj = Record({"a": Int(1), "b": Int(2)})
        self.assertEqual(pretty(obj), "{a = 1, b = 2}")

    def test_pretty_print_access(self) -> None:
        obj = Access(Record({"a": Int(4)}), Var("a"))
        self.assertEqual(pretty(obj), "{a = 4} @ a")

    def test_pretty_print_variant(self) -> None:
        obj = Variant("x", Int(123))
        self.assertEqual(pretty(obj), "#x 123")

        obj = Variant("x", Function(Var("a"), Var("b")))
        self.assertEqual(pretty(obj), "#x (a -> b)")


if __name__ == "__main__":
    unittest.main()
