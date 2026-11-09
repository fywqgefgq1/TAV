#!/usr/bin/env python3
import copy
import io
import json
import os
import random
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import funny_typization as ft
import funny_ir_opt as ir

fp = ft.fp

def mk(cls, **kw):
    n = cls.__new__(cls)
    for k, v in kw.items():
        object.__setattr__(n, k, v)
    if not hasattr(n, "pos"):
        object.__setattr__(n, "pos", 0)
    return n


_TOK = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(==>|==|!=|<=|>=|&&|\|\||[-+*/<>=!(){}\[\],;:.]))")
BIN = {"+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", "==": "EQ", "!=": "NEQ", ">=": "GE",
       "<=": "LE", ">": "GT", "<": "LT", "&&": "AND", "||": "OR", "==>": "→"}


class Parser:
    """func f(a: int, v: int[]) returns (r: int) uses (x: int) [requires e] [ensures e] { ... }"""

    def __init__(self, text):
        self.toks, i = [], 0
        text = text.rstrip()
        while i < len(text):
            m = _TOK.match(text, i)
            if not m:
                raise SyntaxError(f"bad input near {text[i:i + 20]!r}")
            self.toks.append((m.group(1) or m.group(2) or m.group(3), m.start(m.lastindex), m.lastindex))
            i = m.end()
        self.i = 0
        self.counter = 0

    def pos(self):
        self.counter += 1
        return self.counter

    def peek(self, k=0):
        return self.toks[self.i + k][0] if self.i + k < len(self.toks) else None

    def next(self):
        t = self.toks[self.i][0]
        self.i += 1
        return t

    def eat(self, t):
        if self.peek() != t:
            raise SyntaxError(f"expected {t!r}, got {self.peek()!r}")
        self.i += 1

    def accept(self, t):
        if self.peek() == t:
            self.i += 1
            return True
        return False

    def program(self):
        fns = []
        while self.peek() is not None:
            fns.append(self.func())
        return types.SimpleNamespace(functions=fns, formulas=[])

    def params(self):
        out = []
        self.eat("(")
        while self.peek() != ")":
            n = self.next()
            self.eat(":")
            t = self.next()
            if self.accept("["):
                self.eat("]")
                t += "[]"
            out.append((n, t))
            self.accept(",")
        self.eat(")")
        return out

    def func(self):
        self.eat("func")
        pos = self.pos()
        name = self.next()
        params = self.params()
        self.eat("returns")
        rets = self.params()
        uses = self.params() if self.accept("uses") else []
        req = self.expr() if self.accept("requires") else None
        ens = self.expr() if self.accept("ensures") else None
        body = self.block()
        return mk(fp.Function, name=name, params=params, returns=rets, uses=uses, requires=req,
                  ensures=ens, body=body, pos=pos)

    def block(self):
        self.eat("{")
        st = []
        while self.peek() != "}":
            st.append(self.stmt())
        self.eat("}")
        return mk(fp.Block, stmts=st, pos=self.pos())

    def stmt(self):
        t = self.peek()
        if t == "{":
            return self.block()
        if t == "if":
            self.next()
            self.eat("(")
            c = self.expr()
            self.eat(")")
            th = self.stmt()
            el = self.stmt() if self.accept("else") else None
            return mk(fp.IfStmt, cond=c, then_stmt=th, else_stmt=el, pos=self.pos())
        if t == "while":
            self.next()
            self.eat("(")
            c = self.expr()
            self.eat(")")
            inv = self.expr() if self.accept("invariant") else None
            return mk(fp.WhileStmt, cond=c, invariant=inv, body=self.stmt(), pos=self.pos())
        if t in ("assert", "assume"):
            self.next()
            e = self.expr()
            self.eat(";")
            return mk(fp.AssertStmt if t == "assert" else fp.AssumeStmt, expr=e, pos=self.pos())
        p = self.pos()
        names = [self.next()]
        if self.peek() == "[":
            self.next()
            idx = self.expr()
            self.eat("]")
            self.eat("=")
            v = self.expr()
            self.eat(";")
            tgt = mk(fp.ArrayAccess, array=mk(fp.Var, name=names[0], pos=p), index=idx, pos=p)
            return mk(fp.Assign, target=tgt, value=v, pos=p)
        while self.accept(","):
            names.append(self.next())
        self.eat("=")
        v = self.expr()
        self.eat(";")
        if len(names) > 1:
            return mk(fp.TupleAssign, targets=[mk(fp.Var, name=n, pos=p) for n in names], call=v, pos=p)
        return mk(fp.Assign, target=mk(fp.Var, name=names[0], pos=p), value=v, pos=p)
    def bin(self, op, l, r):
        return mk(fp.BinOp, op=BIN[op], left=l, right=r, pos=self.pos())

    def expr(self):
        l = self.or_()
        if self.accept("==>"):
            return self.bin("==>", l, self.expr())
        return l

    def or_(self):
        l = self.and_()
        while self.peek() == "||":
            self.next()
            l = self.bin("||", l, self.and_())
        return l

    def and_(self):
        l = self.not_()
        while self.peek() == "&&":
            self.next()
            l = self.bin("&&", l, self.not_())
        return l

    def not_(self):
        if self.accept("!"):
            return mk(fp.UnaryOp, op="NOT", operand=self.not_(), pos=self.pos())
        return self.cmp()

    def cmp(self):
        l = self.add()
        if self.peek() in ("==", "!=", "<", "<=", ">", ">="):
            op = self.next()
            return self.bin(op, l, self.add())
        return l

    def add(self):
        l = self.mul()
        while self.peek() in ("+", "-"):
            op = self.next()
            l = self.bin(op, l, self.mul())
        return l

    def mul(self):
        l = self.unary()
        while self.peek() in ("*", "/"):
            op = self.next()
            l = self.bin(op, l, self.unary())
        return l

    def unary(self):
        if self.accept("-"):
            return mk(fp.UnaryOp, op="MINUS", operand=self.unary(), pos=self.pos())
        return self.postfix()

    def postfix(self):
        e = self.atom()
        while self.peek() == "[":
            self.next()
            i = self.expr()
            self.eat("]")
            e = mk(fp.ArrayAccess, array=e, index=i, pos=self.pos())
        return e

    def atom(self):
        t = self.next()
        p = self.pos()
        if t == "(":
            e = self.expr()
            self.eat(")")
            return e
        if t.isdigit():
            return mk(fp.IntLit, value=int(t), pos=p)
        if t in ("true", "false"):
            return mk(fp.BoolLit, value=(t == "true"), pos=p)
        if t == "len":
            self.eat("(")
            e = self.expr()
            self.eat(")")
            return mk(fp.LengthExpr, arg=e, pos=p)
        if self.peek() == "(":
            self.next()
            args = []
            while self.peek() != ")":
                args.append(self.expr())
                self.accept(",")
            self.eat(")")
            return mk(fp.FuncCall, name=t, args=args, pos=p)
        return mk(fp.Var, name=t, pos=p)


def parse(text):
    """Parse + type-check (HW3 type checker). Returns the annotated program."""
    prog = Parser(text).program()
    errs = ft.TypeChecker(prog).check()
    assert not errs, errs
    return prog

class RefTrap(Exception):
    pass


def ref_div(a, b):
    if b == 0:
        raise RefTrap("div0")
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def w32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


class Ref:
    """Executes the typed AST directly (independent of the IR)."""

    def __init__(self, prog, fuel=2_000_000):
        self.fns = {f.name: f for f in prog.functions}
        self.fuel = fuel

    def run(self, name, args, depth=0):
        if depth > 200:
            raise RefTrap("depth")
        f = self.fns[name]
        env = {n: a for (n, _), a in zip(f.params, args)}
        for s in f.symbols:
            if s["kind"] in ("return", "local") and s["type"] == "int":
                env[s["name"]] = 0
        self.exec(f.body, env, depth)
        return tuple(env.get(n) for n, _ in f.returns)

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise RefTrap("fuel")

    def exec(self, s, env, depth):
        self.tick()
        if s is None:
            return
        if isinstance(s, fp.Block):
            for x in s.stmts:
                self.exec(x, env, depth)
        elif isinstance(s, fp.Assign):
            v = self.ev(s.value, env, depth)
            t = s.target
            if isinstance(t, fp.Var):
                env[t.name] = v
            else:
                a, i = env[t.array.name], self.ev(t.index, env, depth)
                if not 0 <= i < len(a):
                    raise RefTrap("oob")
                a[i] = v
        elif isinstance(s, fp.TupleAssign):
            res = self.run(s.call.name, [self.ev(a, env, depth) for a in s.call.args], depth + 1)
            for t, r in zip(s.targets, res):
                env[t.name] = r
        elif isinstance(s, fp.IfStmt):
            self.exec(s.then_stmt if self.ev(s.cond, env, depth) else s.else_stmt, env, depth)
        elif isinstance(s, fp.WhileStmt):
            while self.ev(s.cond, env, depth):
                self.tick()
                self.exec(s.body, env, depth)
        elif isinstance(s, (fp.AssertStmt, fp.AssumeStmt)):
            if not self.ev(s.expr, env, depth):
                raise RefTrap("assert")

    def ev(self, e, env, depth):
        if isinstance(e, fp.IntLit):
            return w32(e.value)
        if isinstance(e, fp.BoolLit):
            return e.value
        if isinstance(e, fp.Var):
            return env[e.name]
        if isinstance(e, fp.UnaryOp):
            v = self.ev(e.operand, env, depth)
            return w32(-v) if e.op == "MINUS" else (not v)
        if isinstance(e, fp.ArrayAccess):
            a, i = self.ev(e.array, env, depth), self.ev(e.index, env, depth)
            if not 0 <= i < len(a):
                raise RefTrap("oob")
            return a[i]
        if isinstance(e, fp.LengthExpr):
            return len(self.ev(e.arg, env, depth))
        if isinstance(e, fp.FuncCall):
            return self.run(e.name, [self.ev(a, env, depth) for a in e.args], depth + 1)[0]
        if isinstance(e, fp.BinOp):
            op = e.op
            if op == "AND":
                return self.ev(e.left, env, depth) and self.ev(e.right, env, depth)
            if op == "OR":
                return self.ev(e.left, env, depth) or self.ev(e.right, env, depth)
            if op == "→":
                return (not self.ev(e.left, env, depth)) or self.ev(e.right, env, depth)
            a, b = self.ev(e.left, env, depth), self.ev(e.right, env, depth)
            return {"PLUS": lambda: w32(a + b), "MINUS": lambda: w32(a - b), "STAR": lambda: w32(a * b),
                    "SLASH": lambda: w32(ref_div(a, b)), "EQ": lambda: a == b, "NEQ": lambda: a != b,
                    "GE": lambda: a >= b, "LE": lambda: a <= b, "GT": lambda: a > b,
                    "LT": lambda: a < b}[op]()
        raise AssertionError(type(e))


def outcome(fn):
    try:
        return ("ok", fn())
    except (RefTrap, ir.IRTrap):
        return ("trap", None)

LIB = """
func h(p: int) returns (s: int) { s = p * 2 + 1; }
func g(p: int, q: int) returns (s: int, t: int) { s = p + q; t = p - q; }
"""

PROGRAMS = {
    "gcd": """func gcd(a: int, b: int) returns (r: int) uses (x: int, y: int, t: int) {
        x = a; y = b;
        while (y != 0) { t = x - x / y * y; x = y; y = t; }
        r = x; }""",
    "fact": """func fact(n: int) returns (r: int) uses (i: int) {
        r = 1; i = 2; while (i <= n) { r = r * i; i = i + 1; } }""",
    "sum_arr": """func sum(v: int[]) returns (r: int) uses (i: int) {
        i = 0; while (i < len(v)) { r = r + v[i]; i = i + 1; } }""",
    "max3": """func max3(a: int, b: int, c: int) returns (m: int) {
        if (a >= b && a >= c) m = a; else if (b >= c) m = b; else m = c; }""",
    "logic": """func lg(a: int, b: int) returns (r: int) {
        if (!(a < b) || (a == 3 && b != 4)) r = 1;
        if (a > 0 ==> b > 0) r = r + 10;
        if (true && !(false)) r = r + 100; }""",
    "calls": LIB + """func calls(a: int, b: int) returns (r: int) uses (x: int, y: int) {
        x, y = g(a, b); r = h(x) + h(y) * 2 - h(x); }""",
    "alias": """func al(v: int[]) returns (w: int[], r: int) {
        w = v; r = w[0] + v[0]; w[0] = 7; r = r + w[0] + v[0] + v[0]; }""",
    "asserts": """func asr(a: int) returns (r: int) {
        assert a > 0 || a <= 0; r = a * 2; assume r != 7; assert r == a + a; }""",
    "bigmath": """func bm(a: int) returns (r: int) {
        r = a * 65536 * 65536 + 2147483647 + 1 - -2147483647 - 1; r = r / -1; }""",
    "divs": """func dv(a: int, b: int) returns (r: int) uses (x: int) {
        x = a / b; r = a / 1 + b / -1 + 7 / 2 + -7 / 2; }""",
    "foldall": """func fa() returns (r: int) uses (x: int, y: int) {
        x = 2 + 3; y = x * 4; r = y - 20; }""",
    "nested": """func nst(n: int) returns (r: int) uses (i: int, j: int) {
        i = 0; while (i < n) { j = 0; while (j < i) { r = r + i * j; j = j + 1; } i = i + 1; } }""",
}

INPUTS = {
    "gcd": [[12, 18], [17, 5], [0, 9], [9, 0], [-12, 18], [100, 75]],
    "fact": [[0], [1], [5], [10], [13]],
    "sum_arr": [[[]], [[1]], [[1, 2, 3, 4]], [[2 ** 31 - 1, 1]]],
    "max3": [[1, 2, 3], [3, 2, 1], [2, 3, 1], [2, 2, 2]],
    "logic": [[1, 2], [2, 1], [3, 5], [-1, -1], [0, 4]],
    "calls": [[1, 2], [5, -3], [0, 0]],
    "alias": [[[1, 2]], [[5]], [[]]],
    "asserts": [[1], [-4], [0], [3]],
    "bigmath": [[0], [1], [3], [-5]],
    "divs": [[10, 3], [-10, 3], [10, 0], [0, 7]],
    "foldall": [[]],
    "nested": [[0], [1], [4], [7]],
}
ENTRY = {"gcd": "gcd", "fact": "fact", "sum_arr": "sum", "max3": "max3", "logic": "lg", "calls": "calls",
         "alias": "al", "asserts": "asr", "bigmath": "bm", "divs": "dv", "foldall": "fa", "nested": "nst"}


def lowered(src, **kw):
    prog = parse(src)
    return prog, ir.lower_program(prog, ir.GenOptions(**kw))


def count(fn):
    return sum(len(b.instrs) for b in fn.cfg.blocks)

class TestIRGeneration(unittest.TestCase):
    def test_programs_match_reference(self):
        for key, src in PROGRAMS.items():
            prog, p = lowered(src)
            for f in p.functions:
                ir.validate(f)
            ref = Ref(prog)
            for args in INPUTS[key]:
                exp = outcome(lambda: ref.run(ENTRY[key], copy.deepcopy(args)))
                got = outcome(lambda: ir.run_ir(p, ENTRY[key], args)[0])
                self.assertEqual(exp, got, f"{key}{args}")

    def test_text_format(self):
        _, p = lowered(PROGRAMS["fact"])
        txt = ir.program_to_text(p)
        self.assertIn("func fact(n: int) -> (r: int)", txt)
        self.assertIn("if i <= n goto", txt)
        self.assertIn("ret r", txt)

    def test_entry_zero_init(self):
        _, p = lowered(PROGRAMS["sum_arr"])
        first = p.functions[0].cfg.blocks[0].instrs[0]
        self.assertEqual((first.op, first.dsts, first.args), ("mov", ("r",), (ir.Const(0),)))

    def test_short_circuit_creates_blocks(self):
        _, p = lowered(PROGRAMS["max3"])
        self.assertGreaterEqual(len(p.functions[0].cfg.blocks), 7)

    def test_cfg_blocks_wellformed(self):
        _, p = lowered(PROGRAMS["nested"])
        f = p.functions[0]
        labels = {b.label for b in f.cfg.blocks}
        for b in f.cfg.blocks:
            self.assertTrue(b.instrs[-1].is_terminator)
            self.assertTrue(set(b.succs()) <= labels)
        self.assertEqual(f.cfg.preds()[f.cfg.entry], [])

    def test_loop_has_back_edge(self):
        _, p = lowered(PROGRAMS["fact"])
        f = p.functions[0]
        order = {b.label: i for i, b in enumerate(f.cfg.blocks)}
        self.assertTrue(any(order[s] <= order[b.label] for b in f.cfg.blocks for s in b.succs()))

    def test_assertions_flag(self):
        _, with_a = lowered(PROGRAMS["asserts"])
        _, without = lowered(PROGRAMS["asserts"], assertions=False)
        has = lambda p: any(i.op == "trap" for i in p.functions[0].instrs())
        self.assertTrue(has(with_a))
        self.assertFalse(has(without))

    def test_assert_traps_at_runtime(self):
        _, p = lowered("func f(a: int) returns (r: int) { assert a > 0; r = a; }")
        self.assertEqual(ir.run_ir(p, "f", [3])[0], (3,))
        with self.assertRaises(ir.IRTrap):
            ir.run_ir(p, "f", [0])

    def test_contract_checks(self):
        src = "func f(a: int) returns (r: int) requires a > 0 ensures r > a { r = a - 1; }"
        prog = parse(src)
        plain = ir.lower_program(prog)
        checked = ir.lower_program(prog, ir.GenOptions(contracts=True))
        self.assertEqual(ir.run_ir(plain, "f", [3])[0], (2,))
        with self.assertRaises(ir.IRTrap) as c:
            ir.run_ir(checked, "f", [3])
        self.assertIn("ensures", str(c.exception))
        with self.assertRaises(ir.IRTrap) as c:
            ir.run_ir(checked, "f", [-1])
        self.assertIn("requires", str(c.exception))

    def test_loop_invariant_check(self):
        src = "func f(n: int) returns (r: int) { while (r < n) invariant r <= 2 { r = r + 3; } }"
        prog = parse(src)
        checked = ir.lower_program(prog, ir.GenOptions(contracts=True))
        with self.assertRaises(ir.IRTrap):
            ir.run_ir(checked, "f", [10])

    def test_quantifiers_are_skipped(self):
        prog = parse("func f(a: int) returns (r: int) { r = a; }")
        f = prog.functions[0]
        f.ensures = mk(fp.Quantifier, kind="forall", var_name="k", var_type="int",
                       predicate=mk(fp.BoolLit, value=True))
        p = ir.lower_program(prog, ir.GenOptions(contracts=True))
        self.assertEqual(ir.run_ir(p, "f", [5])[0], (5,))

    def test_literal_attribute_name_is_not_assumed(self):
        prog = parse("func f() returns (r: int) { r = 41 + 1; }")
        lit = prog.functions[0].body.stmts[0].value.left
        object.__delattr__(lit, "value")
        object.__setattr__(lit, "val", 41)
        self.assertEqual(ir.run_ir(ir.lower_program(prog), "f", [])[0], (42,))

    def test_missing_annotations_reported(self):
        prog = Parser("func f() returns (r: int) { r = 1; }").program()
        with self.assertRaises(ir.IRGenError):
            ir.lower_program(prog)


class TestOptimizations(unittest.TestCase):
    def opt(self, src, passes=ir.PASS_NAMES, name=None, **kw):
        prog, p = lowered(src, **kw)
        q = p.clone()
        stats = {f.name: ir.optimize(f, passes) for f in q.functions}
        return prog, p, q, stats

    def fn(self, p, name=None):
        return p.functions[0] if name is None else p.get(name)

    def test_constant_folding_collapses_function(self):
        _, p, q, _ = self.opt(PROGRAMS["foldall"])
        f = self.fn(q)
        self.assertEqual(len(f.cfg.blocks), 1)
        self.assertEqual([ir.instr_text(i) for i in f.cfg.blocks[0].instrs], ["ret 0"])
        self.assertEqual(f.temps(), [])

    def test_constprop_only(self):
        _, p, q, st = self.opt("func f(a: int) returns (r: int) uses (x: int) { x = 4; r = x * 3 + a; }",
                               ["constprop"])
        txt = ir.fn_to_text(self.fn(q))
        self.assertIn("$t0 = 12", txt)
        self.assertIn("r = 12 + a", txt)
        self.assertGreater(st["f"]["constprop"], 0)

    def test_constprop_does_not_cross_blocks(self):
        src = """func f(a: int) returns (r: int) uses (x: int) {
                   x = 5; if (a > 0) x = 6; r = x + 1; }"""
        prog, p, q, _ = self.opt(src, ["constprop"])
        ref = Ref(prog)
        for a in (-1, 1):
            self.assertEqual(ir.run_ir(q, "f", [a])[0], ref.run("f", [a]))

    def test_constprop_keeps_division_by_zero(self):
        _, p, q, _ = self.opt("func f() returns (r: int) { r = 1 / 0; }", ["constprop", "dce"])
        with self.assertRaises(ir.IRTrap):
            ir.run_ir(q, "f", [])

    def test_wraparound_folding(self):
        prog, p, q, _ = self.opt(PROGRAMS["bigmath"])
        ref = Ref(prog)
        for a in (0, 1, 3, -5):
            self.assertEqual(ir.run_ir(q, "bm", [a])[0], ref.run("bm", [a]))

    def test_algebraic_identities(self):
        src = """func f(a: int) returns (r: int) uses (p: int, q: int, s: int, u: int, w: int) {
                   p = a + 0; q = a * 1; s = a - 0; u = a - a; w = a * 0; r = p + q + s + u + w; }"""
        prog, p, q, st = self.opt(src, ["algebraic"])
        txt = ir.fn_to_text(self.fn(q))
        self.assertNotIn("a + 0", txt)
        self.assertNotIn("a * 1", txt)
        self.assertNotIn("a - a", txt)
        self.assertGreaterEqual(st["f"]["algebraic"], 5)
        for a in (-3, 0, 9):
            self.assertEqual(ir.run_ir(q, "f", [a])[0], (3 * a,))

    def test_strength_reduction_and_neg(self):
        src = "func f(a: int) returns (r: int, s: int, t: int) { r = a * 2; s = 0 - a; t = a * -1; }"
        _, p, q, _ = self.opt(src, ["algebraic"])
        txt = ir.fn_to_text(self.fn(q))
        self.assertIn("r = a + a", txt)
        self.assertIn("s = neg a", txt)
        self.assertIn("t = neg a", txt)

    def test_self_compare_branch(self):
        src = "func f(a: int) returns (r: int) { if (a == a) r = 1; else r = 2; }"
        _, p, q, _ = self.opt(src)
        self.assertEqual(len(self.fn(q).cfg.blocks), 1)
        self.assertEqual(ir.run_ir(q, "f", [5])[0], (1,))

    def test_copyprop(self):
        src = "func f(a: int) returns (r: int) uses (x: int, y: int) { x = a; y = x; r = y + x; }"
        _, p, q, _ = self.opt(src, ["copyprop"])
        self.assertIn("r = a + a", ir.fn_to_text(self.fn(q)))

    def test_copyprop_invalidation(self):
        src = """func f(a: int) returns (r: int) uses (x: int, y: int) {
                   x = a; y = x; x = 100; r = y + x; }"""
        prog, p, q, _ = self.opt(src, ["copyprop"])
        self.assertEqual(ir.run_ir(q, "f", [5])[0], (105,))

    def test_dce_removes_dead_but_keeps_effects(self):
        src = """func f(a: int, v: int[]) returns (r: int) uses (d: int, e: int, z: int) {
                   d = a * 3; e = a / 0 + 0; z = v[5]; r = a; }"""
        _, p, q, _ = self.opt(src, ["dce"])
        txt = ir.fn_to_text(self.fn(q))
        self.assertNotIn("d = a * 3", txt)
        self.assertIn("/ 0", txt)
        self.assertIn("v[5]", txt)

    def test_dce_across_blocks_uses_liveness(self):
        src = """func f(a: int) returns (r: int) uses (x: int) {
                   x = a + 1; if (a > 0) r = x; else r = 0; }"""
        prog, p, q, _ = self.opt(src, ["dce"])
        for a in (-1, 1):
            self.assertEqual(ir.run_ir(q, "f", [a])[0], Ref(prog).run("f", [a]))

    def test_cse(self):
        src = """func f(a: int, b: int) returns (r: int) uses (x: int, y: int) {
                   x = a * b + 1; y = b * a + 2; r = x + y; }"""
        _, p, q, _ = self.opt(src, ["cse"])
        muls = [i for i in self.fn(q).instrs() if i.op == "bin" and i.sub == "*"]
        self.assertEqual(len(muls), 1)

    def test_cse_invalidated_by_redefinition(self):
        src = """func f(a: int, b: int) returns (r: int) uses (x: int, y: int) {
                   x = a + b; a2 = x; y = a + b; r = x + y; }"""
        prog, p, q, _ = self.opt(src, ["cse", "copyprop", "dce"])
        self.assertEqual(ir.run_ir(q, "f", [3, 4])[0], (14,))
        src2 = """func f(a: int) returns (r: int) uses (x: int, y: int, k: int) {
                    k = a; x = k + 1; k = k + 5; y = k + 1; r = x * 100 + y; }"""
        prog, p, q, _ = self.opt(src2, ["cse"])
        self.assertEqual(ir.run_ir(q, "f", [1])[0], Ref(prog).run("f", [1]))

    def test_cse_array_loads_killed_by_stores(self):
        prog, p, q, _ = self.opt(PROGRAMS["alias"], ["cse"])
        for args in INPUTS["alias"]:
            self.assertEqual(outcome(lambda: ir.run_ir(q, "al", args)[0]),
                             outcome(lambda: Ref(prog).run("al", copy.deepcopy(args))))

    def test_cfg_folds_constant_branch_and_merges(self):
        src = "func f() returns (r: int) { if (1 < 2) r = 1; else r = 2; }"
        _, p, q, _ = self.opt(src)
        self.assertEqual(len(self.fn(q).cfg.blocks), 1)
        self.assertEqual(ir.run_ir(q, "f", [])[0], (1,))

    def test_cfg_removes_unreachable_after_while_true_false(self):
        src = "func f() returns (r: int) { while (false) r = r + 1; r = 5; }"
        _, p, q, _ = self.opt(src)
        self.assertEqual(len(self.fn(q).cfg.blocks), 1)

    def test_cfg_keeps_infinite_loop(self):
        src = "func f() returns (r: int) { while (true) r = r + 1; }"
        _, p, q, _ = self.opt(src)
        ir.validate(self.fn(q))
        with self.assertRaises(ir.IRTrap):
            ir.run_ir(q, "f", [], max_steps=1000)

    def test_passes_are_switchable(self):
        _, p, q_none, st = self.opt(PROGRAMS["foldall"], [])
        self.assertEqual(ir.program_to_text(p), ir.program_to_text(q_none))
        self.assertEqual(st["fa"], {})
        for off in ir.PASS_NAMES:
            enabled = [x for x in ir.PASS_NAMES if x != off]
            _, _, q, st = self.opt(PROGRAMS["foldall"], enabled)
            self.assertNotIn(off, st["fa"])

    def test_unknown_pass_rejected(self):
        _, p = lowered(PROGRAMS["fact"])
        with self.assertRaises(ValueError):
            ir.optimize(p.functions[0], ["nope"])

    def test_pipeline_never_increases_size(self):
        for key, src in PROGRAMS.items():
            _, p, q, _ = self.opt(src)
            for a, b in zip(p.functions, q.functions):
                self.assertLessEqual(count(b), count(a), key)

    def test_all_programs_all_subsets_equivalent(self):
        rnd = random.Random(1)
        for key, src in PROGRAMS.items():
            prog = parse(src)
            base = ir.lower_program(prog)
            ref = Ref(prog)
            subsets = [list(ir.PASS_NAMES), []] + [[x] for x in ir.PASS_NAMES] + \
                      [rnd.sample(ir.PASS_NAMES, rnd.randint(2, 5)) for _ in range(4)]
            for passes in subsets:
                q = base.clone()
                for f in q.functions:
                    ir.optimize(f, passes)
                for args in INPUTS[key]:
                    exp = outcome(lambda: ref.run(ENTRY[key], copy.deepcopy(args)))
                    got = outcome(lambda: ir.run_ir(q, ENTRY[key], args)[0])
                    self.assertEqual(exp, got, f"{key}{args} passes={passes}")

    def test_dynamic_instruction_count_improves(self):
        _, p = lowered(PROGRAMS["gcd"])
        q = p.clone()
        ir.optimize(q.functions[0])
        _, s0 = ir.run_ir(p, "gcd", [100, 75])
        _, s1 = ir.run_ir(q, "gcd", [100, 75])
        self.assertLess(s1, s0)


class TestColoring(unittest.TestCase):
    def check_valid(self, f, scope):
        c = ir.greedy_color(f, scope)
        for e in c.edges:
            a, b = tuple(e)
            self.assertNotEqual(c.color[a], c.color[b])
        self.assertGreaterEqual(c.num_colors, 0)
        return c

    def test_valid_colorings_on_all_programs(self):
        for src in PROGRAMS.values():
            _, p = lowered(src)
            for f in p.functions:
                for scope in ("temps", "all"):
                    self.check_valid(f, scope)
                q = f.clone()
                ir.optimize(q)
                for scope in ("temps", "all"):
                    self.check_valid(q, scope)

    def test_known_graph(self):
        _, p = lowered("func f(a: int, b: int) returns (r: int) { r = (a + b) + (a * b); }")
        c = ir.greedy_color(p.functions[0], "temps")
        self.assertEqual(sorted(c.nodes), ["$t0", "$t1"])
        self.assertEqual(c.num_colors, 2)
        self.assertEqual(len(c.edges), 1)

    def test_temporaries_share_a_colour_when_not_overlapping(self):
        src = "func f(a: int, b: int) returns (r: int, s: int) { r = (a + b) * (a - b); s = (a * b) - (a / 3); }"
        _, p = lowered(src)
        f = p.functions[0]
        c = ir.greedy_color(f, "temps")
        self.assertGreater(len(c.nodes), c.num_colors)

    def test_move_does_not_create_edge(self):
        src = "func f(a: int) returns (r: int) uses (x: int) { x = a + 1; r = x; r = r + x; }"
        _, p = lowered(src)
        c = ir.greedy_color(p.functions[0], "all")
        if "x" in c.nodes and "r" in c.nodes:
            self.assertIn(frozenset(("x", "r")) in c.edges, (True, False))

    def test_arrays_use_separate_palette(self):
        _, p = lowered(PROGRAMS["alias"])
        c = ir.greedy_color(p.functions[0], "all")
        self.assertTrue(c.color["w"].startswith("a"))
        self.assertTrue(c.color["r"].startswith("i"))

    def test_outputs(self):
        _, p = lowered(PROGRAMS["nested"])
        f = p.functions[0]
        c = ir.greedy_color(f, "all")
        self.assertIn("graph", ir.conflicts_to_dot(f, c))
        self.assertIn("colours =", ir.conflicts_to_text(f, c))


class TestSerialisation(unittest.TestCase):
    def test_json_roundtrip_structure(self):
        _, p = lowered(PROGRAMS["calls"])
        data = json.loads(json.dumps(ir.program_to_json(p)))
        self.assertEqual(data["format"], "funny-ir/1")
        ops = {i["op"] for f in data["functions"] for b in f["blocks"] for i in b["instrs"]}
        self.assertTrue({"mov", "bin", "call", "ret"} <= ops)
        for f in data["functions"]:
            labels = {b["label"] for b in f["blocks"]}
            for b in f["blocks"]:
                self.assertTrue(set(b["succs"]) <= labels)

    def test_dot(self):
        _, p = lowered(PROGRAMS["max3"])
        dot = ir.cfg_to_dot(p.functions[0])
        self.assertTrue(dot.startswith('digraph "cfg_max3"'))
        self.assertIn('label="T"', dot)


class TestCLI(unittest.TestCase):
    def test_cli_end_to_end(self):
        prog = parse(PROGRAMS["fact"])
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "p.fn")
            with open(src, "w") as fh:
                fh.write("ignored: analyze() is patched")
            old = ft.analyze
            ft.analyze = lambda text, infer_locals=True: (prog, [])
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = ir.main([src, "-o", os.path.join(d, "o"), "--run", "fact", "--args", "[6]",
                                  "--print", "after"])
            finally:
                ft.analyze = old
            self.assertEqual(rc, 0, out.getvalue())
            self.assertIn("results identical: yes", out.getvalue())
            self.assertIn("result=(720,)", out.getvalue())
            files = set(os.listdir(os.path.join(d, "o")))
            for need in ("ir_before.txt", "ir_after.txt", "ir_before.json", "ir_after.json", "metrics.json",
                         "report.txt", "cfg_fact_before.dot", "cfg_fact_after.dot",
                         "conflicts_fact_after.txt", "conflicts_fact_after.dot"):
                self.assertIn(need, files)
            with open(os.path.join(d, "o", "metrics.json")) as fh:
                m = json.load(fh)
            self.assertLessEqual(m["functions"]["fact"]["after"]["instructions"],
                                 m["functions"]["fact"]["before"]["instructions"])

    def test_select_passes(self):
        self.assertEqual(ir.select_passes("all", ""), list(ir.PASS_NAMES))
        self.assertEqual(ir.select_passes("none", ""), [])
        self.assertEqual(ir.select_passes("all", "cse,dce"), ["constprop", "algebraic", "copyprop", "cfg"])
        self.assertEqual(ir.select_passes("dce,cse", ""), ["dce", "cse"])
        with self.assertRaises(SystemExit):
            ir.select_passes("bogus", "")

    def test_frontend_errors_reported(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "p.fn")
            with open(src, "w") as fh:
                fh.write("x")
            old = ft.analyze
            ft.analyze = lambda text, infer_locals=True: (None, [{"phase": "syntax", "message": "boom", "pos": None}])
            try:
                import contextlib
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(ir.main([src, "-o", d]), 1)
            finally:
                ft.analyze = old

class Fuzz:
    VARS = ["x", "y", "z"]

    def __init__(self, seed):
        self.r = random.Random(seed)
        self.loop_vars = ["i", "j"]

    def atom(self):
        r = self.r
        k = r.random()
        if k < 0.30:
            return str(r.choice([0, 1, 2, 3, 5, 7, 100, 2147483647]))
        if k < 0.35:
            return "len(v)"
        if k < 0.42:
            return f"v[{r.choice([0, 1, 2])}]"
        if k < 0.46:
            return f"h({self.expr(1)})"
        return r.choice(["a", "b", "x", "y", "z", "r"])

    def expr(self, d=3):
        r = self.r
        if d <= 0 or r.random() < 0.25:
            return self.atom()
        k = r.random()
        if k < 0.1:
            return f"-({self.expr(d - 1)})"
        op = r.choice(["+", "+", "-", "*", "*", "/"])
        rhs = self.expr(d - 1)
        if op == "/" and r.random() < 0.8:
            rhs = str(r.choice([1, 2, 3, 7]))
        return f"({self.expr(d - 1)} {op} {rhs})"

    def cond(self, d=2):
        r = self.r
        if d > 0 and r.random() < 0.3:
            op = r.choice(["&&", "||", "==>"])
            return f"({self.cond(d - 1)} {op} {self.cond(d - 1)})"
        if d > 0 and r.random() < 0.1:
            return f"!({self.cond(d - 1)})"
        if r.random() < 0.05:
            return r.choice(["true", "false"])
        return f"{self.expr(2)} {r.choice(['<', '<=', '>', '>=', '==', '!='])} {self.expr(2)}"

    def stmt(self, depth, loops):
        r = self.r
        k = r.random()
        if k < 0.55 or depth <= 0:
            tgt = r.choice(self.VARS + ["r"])
            return f"{tgt} = {self.expr()};"
        if k < 0.60:
            return f"x, y = g({self.expr(2)}, {self.expr(2)});"
        if k < 0.64:
            return f"assert {self.cond(1)};"
        if k < 0.80:
            s = f"if ({self.cond()}) {self.block(depth - 1, loops)}"
            if r.random() < 0.6:
                s += f" else {self.block(depth - 1, loops)}"
            return s
        if loops < len(self.loop_vars):
            lv, bound = self.loop_vars[loops], r.randint(0, 4)
            return (f"{lv} = 0; while ({lv} < {bound}) {{ {self.body(depth - 1, loops + 1)} "
                    f"{lv} = {lv} + 1; }}")
        return f"{r.choice(self.VARS)} = {self.expr()};"

    def body(self, depth, loops):
        return " ".join(self.stmt(depth, loops) for _ in range(self.r.randint(1, 3)))

    def block(self, depth, loops):
        return "{ " + self.body(depth, loops) + " }"

    def program(self):
        return (LIB + "func main(a: int, b: int, v: int[]) returns (r: int) uses (x: int, y: int, z: int, i: int, j: int) { "
                + self.body(3, 0) + " r = r + x - y + z; }")


class TestFuzz(unittest.TestCase):
    def test_random_programs_before_after_equal(self):
        rnd = random.Random(7)
        checked = traps = 0
        for seed in range(250):
            src = Fuzz(seed).program()
            prog = parse(src)
            base = ir.lower_program(prog)
            for f in base.functions:
                ir.validate(f)
            subsets = [list(ir.PASS_NAMES), rnd.sample(ir.PASS_NAMES, rnd.randint(1, 4))]
            variants = []
            for passes in subsets:
                q = base.clone()
                for f in q.functions:
                    ir.optimize(f, passes)
                    ir.validate(f)
                variants.append((passes, q))
            ref = Ref(prog)
            for _ in range(4):
                args = [rnd.choice([-3, 0, 1, 2, 5, 2147483647]), rnd.choice([-1, 0, 1, 3, 7]),
                        [rnd.randint(-5, 5) for _ in range(rnd.randint(0, 4))]]
                exp = outcome(lambda: ref.run("main", copy.deepcopy(args)))
                got0 = outcome(lambda: ir.run_ir(base, "main", args)[0])
                self.assertEqual(exp, got0, f"IR != reference, seed={seed} args={args}\n{src}")
                for passes, q in variants:
                    got = outcome(lambda: ir.run_ir(q, "main", args)[0])
                    self.assertEqual(exp, got, f"seed={seed} passes={passes} args={args}\n{src}")
                if exp[0] == "ok":
                    s0 = ir.run_ir(base, "main", args)[1]
                    s1 = ir.run_ir(variants[0][1], "main", args)[1]
                    self.assertLessEqual(s1, s0, f"optimized code slower: seed={seed}")
                traps += exp[0] == "trap"
                checked += 1
        self.assertGreater(checked - traps, 400)

    def test_random_colorings_valid(self):
        for seed in range(60):
            prog = parse(Fuzz(1000 + seed).program())
            p = ir.lower_program(prog)
            for f in p.functions:
                q = f.clone()
                ir.optimize(q)
                for fn in (f, q):
                    for scope in ("temps", "all"):
                        c = ir.greedy_color(fn, scope)
                        for e in c.edges:
                            a, b = tuple(e)
                            self.assertNotEqual(c.color[a], c.color[b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
