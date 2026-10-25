from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import funny_parser as fp
from funny_parser import (
    ASTNode, Function, Formula, Block, Assign, TupleAssign, IfStmt, WhileStmt,
    AssertStmt, AssumeStmt, ErrorStmt, BinOp, UnaryOp, IntLit, BoolLit, Var,
    ArrayAccess, FuncCall, LengthExpr, Quantifier, FormulaRef,
)

@dataclass(frozen=True)
class Ty:
    name: str

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class FuncTy:
    params: tuple
    rets: tuple

    def __str__(self):
        return f"({', '.join(map(str, self.params))}) -> ({', '.join(map(str, self.rets))})"


INT, BOOL, ARR = Ty("int"), Ty("bool"), Ty("int[]")
ERR = Ty("<error>")
BASE_TYPES = {"int": INT, "int[]": ARR}


def unify(a, b):
    if a == ERR:
        return b
    if b == ERR:
        return a
    return a if a == b else None

@dataclass
class Symbol:
    name: str
    type: object
    kind: str
    pos: int
    inferred: bool = False


class Scope:
    def __init__(self, parent=None, label=""):
        self.parent = parent
        self.label = label
        self.syms = {}

    def declare(self, sym):
        self.syms[sym.name] = sym

    def lookup(self, name):
        s = self
        while s is not None:
            if name in s.syms:
                return s.syms[name]
            s = s.parent
        return None


OP_SYM = {"PLUS": "+", "MINUS": "-", "STAR": "*", "SLASH": "/", "EQ": "==", "NEQ": "!=",
          "GE": ">=", "LE": "<=", "GT": ">", "LT": "<", "AND": "and", "OR": "or",
          "NOT": "not", "→": "→"}
ARITH = {"PLUS", "MINUS", "STAR", "SLASH"}
ORDER = {"GE", "LE", "GT", "LT", "EQ", "NEQ"}
LOGIC = {"AND", "OR", "→"}


def walk(n):
    if isinstance(n, ASTNode):
        yield n
        for v in list(n.__dict__.values()):
            yield from walk(v)
    elif isinstance(n, (list, tuple)):
        for i in n:
            yield from walk(i)


def line_col(text, pos):
    if text is None or pos is None or pos < 0 or pos > len(text):
        return None
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return line, col

class TypeChecker:
    def __init__(self, program, tokens=(), text=None, infer_locals=True):
        self.program = program
        self.tokens = list(tokens)
        self.tok_index = {t[2]: i for i, t in enumerate(self.tokens)}
        self.text = text
        self.infer = infer_locals
        self.errors = []
        self.globals = Scope(None, "global")
        self.fscope = None
        self.assigned = set()
        self.ctx = None

    def err(self, pos, msg):
        self.errors.append((pos, msg))

    def where(self, pos):
        lc = line_col(self.text, pos)
        return f"line {lc[0]}, col {lc[1]}" if lc else f"position {pos}"

    def resolve_type(self, name, pos, report=True):
        t = BASE_TYPES.get(name)
        if t is None:
            if report:
                self.err(pos, f"unknown type '{name}' (supported: int, int[])")
            return ERR
        return t

    def check(self):
        self.collect_globals()
        for fn in self.program.functions:
            self.check_function(fn)
        for fm in self.program.formulas:
            self.check_formula(fm)
        self.program.globals = {n: {"kind": s.kind, "type": str(s.type)}
                                for n, s in self.globals.syms.items()}
        return self.errors

    def collect_globals(self):
        items = sorted(self.program.functions + self.program.formulas, key=lambda n: n.pos)
        for it in items:
            is_fn = isinstance(it, Function)
            params = tuple(self.resolve_type(t, it.pos, False) for _, t in it.params)
            rets = tuple(self.resolve_type(t, it.pos, False) for _, t in it.returns) if is_fn else (BOOL,)
            kind = "function" if is_fn else "formula"
            prev = self.globals.syms.get(it.name)
            if prev is not None:
                self.err(it.pos, f"redefinition of {prev.kind} '{it.name}' "
                                 f"(first defined at {self.where(prev.pos)})")
                continue
            self.globals.declare(Symbol(it.name, FuncTy(params, rets), kind, it.pos))

    def header_positions(self, item):
        res = {"params": [], "returns": [], "uses": []}
        toks = self.tokens
        i = self.tok_index.get(item.pos)
        if i is None:
            return res
        if isinstance(item, Function):
            end = self.tok_index.get(item.body.pos, len(toks))
        else:
            end = next((j for j in range(i, len(toks)) if toks[j][0] == "ARROW"), len(toks))
        section = "header"
        while i < end:
            t = toks[i][0]
            if section == "header" and t == "LPAREN":
                section = "params"
            elif section == "params" and t == "RPAREN":
                section = None
            elif section != "header" and t in ("RETURNS", "USES"):
                section = t.lower()
            elif t in ("REQUIRES", "ENSURES"):
                section = None
            elif (section in ("params", "returns", "uses") and t == "ID" and i + 2 < len(toks)
                  and toks[i + 1][0] == "COLON" and toks[i + 2][0] == "ID"):
                res[section].append(toks[i][2])
                i += 3
                continue
            i += 1
        return res

    @staticmethod
    def visible_var(scope, name):
        sym = scope.lookup(name)
        return sym if sym is not None and sym.kind not in ("function", "formula") else None

    def declare_vars(self, scope, defs, kind, positions, default_pos):
        for k, (name, tname) in enumerate(defs):
            pos = positions[k] if len(positions) == len(defs) else default_pos
            ty = self.resolve_type(tname, pos)
            prev = self.visible_var(scope, name)
            if prev is not None:
                self.err(pos, f"duplicate declaration of '{name}' "
                              f"(already declared as {prev.kind} at {self.where(prev.pos)})")
                continue
            scope.declare(Symbol(name, ty, kind, pos))

    def check_function(self, f):
        pscope = Scope(self.globals, f"params of {f.name}")
        rscope = Scope(pscope, f"returns of {f.name}")
        scope = self.fscope = Scope(rscope, f"locals of {f.name}")
        self.assigned = {n for n in self._assigned_names(f.body)}
        hp = self.header_positions(f)
        self.declare_vars(pscope, f.params, "param", hp["params"], f.pos)
        self.declare_vars(rscope, f.returns, "return", hp["returns"], f.pos)
        self.declare_vars(scope, f.uses, "local", hp["uses"], f.pos)
        if not f.returns:
            self.err(f.pos, f"function '{f.name}' must return at least one value")
        for sym in scope.syms.values():
            if sym.type not in (INT, ERR):
                self.err(sym.pos, f"local variable '{sym.name}' must be int (local arrays are not supported)")

        if f.requires is not None:
            self.ctx = "precondition (requires)"
            self.expect_bool(f.requires, pscope, self.ctx)
        if f.ensures is not None:
            self.ctx = "postcondition (ensures)"
            self.expect_bool(f.ensures, rscope, self.ctx)
        self.ctx = None
        self.stmt(f.body, scope)

        res = lambda defs: [{"name": n, "type": str(self.resolve_type(t, 0, False))} for n, t in defs]
        f.signature = {"params": res(f.params), "returns": res(f.returns)}
        f.type = str(FuncTy(tuple(self.resolve_type(t, 0, False) for _, t in f.params),
                            tuple(self.resolve_type(t, 0, False) for _, t in f.returns)))
        f.symbols = [{"name": s.name, "type": str(s.type), "kind": s.kind, "inferred": s.inferred}
                     for sc in (pscope, rscope, scope) for s in sc.syms.values()]
        self.fscope = None

    def check_formula(self, fm):
        scope = Scope(self.globals, f"formula {fm.name}")
        self.fscope, self.assigned, self.ctx = None, set(), None
        self.declare_vars(scope, fm.params, "param", self.header_positions(fm)["params"], fm.pos)
        self.expect_bool(fm.predicate, scope, f"body of formula '{fm.name}'")
        fm.signature = {"params": [{"name": n, "type": str(self.resolve_type(t, 0, False))}
                                   for n, t in fm.params], "returns": [{"name": "", "type": "bool"}]}
        fm.type = str(FuncTy(tuple(self.resolve_type(t, 0, False) for _, t in fm.params), (BOOL,)))

    @staticmethod
    def _assigned_names(body):
        for n in walk(body):
            if isinstance(n, Assign) and isinstance(n.target, Var):
                yield n.target.name
            elif isinstance(n, TupleAssign):
                for t in n.targets:
                    yield t.name

    def stmt(self, s, scope):
        if s is None or isinstance(s, ErrorStmt):
            return
        if isinstance(s, Block):
            for x in s.stmts:
                self.stmt(x, scope)
        elif isinstance(s, Assign):
            self.assign(s, scope)
        elif isinstance(s, TupleAssign):
            self.tuple_assign(s, scope)
        elif isinstance(s, IfStmt):
            self.expect_bool(s.cond, scope, "condition of 'if'")
            self.stmt(s.then_stmt, scope)
            self.stmt(s.else_stmt, scope)
        elif isinstance(s, WhileStmt):
            self.expect_bool(s.cond, scope, "condition of 'while'")
            if s.invariant is not None:
                self.expect_bool(s.invariant, scope, "loop invariant")
            self.stmt(s.body, scope)
        elif isinstance(s, AssertStmt):
            self.expect_bool(s.expr, scope, "assert condition")
        elif isinstance(s, AssumeStmt):
            self.expect_bool(s.expr, scope, "assume condition")

    def assign(self, s, scope):
        rhs = self.expr(s.value, scope)
        tgt = s.target
        if isinstance(tgt, Var):
            self.assign_var(tgt, rhs, scope, s.pos)
        else:
            self.expr(tgt, scope)
            base = tgt.array
            psym = scope.lookup(base.name) if isinstance(base, Var) else None
            if psym is not None and psym.kind == "param":
                self.err(tgt.pos, f"cannot modify an element of parameter '{base.name}' (parameters are read-only)")
            if rhs not in (INT, ERR):
                self.err(s.pos, f"cannot assign {rhs} to an element of int[] (elements are int)")

    def tuple_assign(self, s, scope):
        rets = self.call(s.call, scope, "function")
        if rets is not None and len(rets) != len(s.targets):
            self.err(s.pos, f"function '{s.call.name}' returns {len(rets)} value(s) "
                            f"but {len(s.targets)} target(s) given")
        types = []
        for k, tgt in enumerate(s.targets):
            ty = rets[k] if rets is not None and k < len(rets) else ERR
            self.assign_var(tgt, ty, scope, s.pos)
            types.append(tgt.type)
        s.types = types
        s.call.type = f"({', '.join(map(str, rets))})" if rets is not None else str(ERR)

    def assign_var(self, tgt, ty, scope, stmt_pos):
        sym = scope.lookup(tgt.name)
        if sym is None:
            if self.infer:
                sym = Symbol(tgt.name, ty, "local", tgt.pos, inferred=True)
                self.fscope.declare(sym)
                if ty not in (INT, ERR):
                    self.err(tgt.pos, f"local variable '{tgt.name}' must be int, but its first "
                                      f"assignment has type {ty}")
            else:
                self.err(tgt.pos, f"assignment to undeclared variable '{tgt.name}' "
                                  f"(declare it in 'uses'; type inference is disabled)")
                tgt.type, tgt.kind = str(ERR), None
                return
        elif sym.kind == "param":
            self.err(tgt.pos, f"cannot assign to parameter '{tgt.name}' (parameters are read-only)")
            tgt.type, tgt.kind = str(sym.type), sym.kind
            return
        elif sym.kind in ("function", "formula"):
            self.err(tgt.pos, f"cannot assign to {sym.kind} '{tgt.name}'")
            tgt.type, tgt.kind = str(ERR), None
            return
        elif sym.inferred and sym.type == ERR:
            sym.type = ty
        elif unify(sym.type, ty) is None:
            note = (f" (its type was inferred from the first assignment at {self.where(sym.pos)})"
                    if sym.inferred else "")
            self.err(stmt_pos, f"cannot assign {ty} to variable '{tgt.name}' of type {sym.type}{note}")
        tgt.type, tgt.kind = str(sym.type), sym.kind

    def expect_bool(self, e, scope, what):
        t = self.expr(e, scope)
        if t not in (BOOL, ERR):
            self.err(e.pos, f"{what} must be bool, got {t}")

    def expr(self, e, scope):
        t = self._expr(e, scope)
        e.type = str(t)
        return t

    def _expr(self, e, scope):
        if isinstance(e, IntLit):
            return INT
        if isinstance(e, BoolLit):
            return BOOL
        if isinstance(e, Var):
            return self.var(e, scope)
        if isinstance(e, UnaryOp):
            t = self.expr(e.operand, scope)
            want = INT if e.op == "MINUS" else BOOL
            if t not in (want, ERR):
                self.err(e.pos, f"operator '{OP_SYM.get(e.op, e.op)}' expects {want}, got {t}")
            return want
        if isinstance(e, BinOp):
            return self.binop(e, scope)
        if isinstance(e, ArrayAccess):
            a = self.expr(e.array, scope)
            i = self.expr(e.index, scope)
            if a not in (ARR, ERR):
                self.err(e.pos, f"cannot index a value of type {a} (only int[] can be indexed)")
            if i not in (INT, ERR):
                self.err(e.index.pos, f"array index must be int, got {i}")
            return INT
        if isinstance(e, LengthExpr):
            a = self.expr(e.arg, scope)
            if a not in (ARR, ERR):
                self.err(e.pos, f"length() expects int[], got {a}")
            return INT
        if isinstance(e, FuncCall):
            rets = self.call(e, scope, "function")
            if rets is None:
                return ERR
            if len(rets) != 1:
                what = "no value" if not rets else f"{len(rets)} values"
                self.err(e.pos, f"function '{e.name}' returns {what}; it cannot be used inside an "
                                f"expression (use a tuple assignment for several results)")
                return ERR
            return rets[0]
        if isinstance(e, FormulaRef):
            self.call(e, scope, "formula")
            return BOOL
        if isinstance(e, Quantifier):
            inner = Scope(scope, f"quantifier {e.var_name}")
            ty = self.resolve_type(e.var_type, e.pos)
            prev = self.visible_var(scope, e.var_name)
            if prev is not None:
                self.err(e.pos, f"quantifier variable '{e.var_name}' duplicates the {prev.kind} "
                                f"declared at {self.where(prev.pos)}; names must be unique in scope")
            inner.declare(Symbol(e.var_name, ty, "bound", e.pos))
            self.expect_bool(e.predicate, inner, f"body of '{e.kind}'")
            return BOOL
        raise TypeError(f"unsupported AST node {type(e).__name__}")

    def var(self, e, scope):
        sym = scope.lookup(e.name)
        if sym is None:
            full = self.fscope.lookup(e.name) if self.fscope is not None else None
            if self.ctx and full is not None and full.kind not in ("function", "formula"):
                self.err(e.pos, f"variable '{e.name}' ({full.kind}) cannot be used in the {self.ctx}")
            elif self.infer and e.name in self.assigned:
                self.err(e.pos, f"variable '{e.name}' is used before its first assignment")
            else:
                self.err(e.pos, f"undeclared variable '{e.name}'")
            e.kind = None
            return ERR
        if sym.kind in ("function", "formula"):
            self.err(e.pos, f"'{e.name}' is a {sym.kind}, not a variable")
            e.kind = None
            return ERR
        e.kind = sym.kind
        return sym.type

    def binop(self, e, scope):
        l = self.expr(e.left, scope)
        r = self.expr(e.right, scope)
        op, shown = e.op, OP_SYM.get(e.op, e.op)
        if op in ARITH or op in ORDER or op in LOGIC:
            want = BOOL if op in LOGIC else INT
            if any(t not in (want, ERR) for t in (l, r)):
                self.err(e.pos, f"operator '{shown}' expects {want} operands, got {l} and {r}")
            return INT if op in ARITH else BOOL
        raise ValueError(f"unknown binary operator {op!r}")

    def call(self, node, scope, expected_kind):
        arg_types = [self.expr(a, scope) for a in node.args]
        sym = self.globals.syms.get(node.name)
        node.ret_types, node.callee = [], None
        if sym is None:
            self.err(node.pos, f"call to undefined {expected_kind} '{node.name}'")
            return None
        if sym.kind != expected_kind:
            hint = ("formulas can only be used inside predicates" if sym.kind == "formula"
                    else "functions cannot be used as formulas")
            self.err(node.pos, f"'{node.name}' is a {sym.kind}, not a {expected_kind} ({hint})")
            return None
        sig = sym.type
        node.callee, node.ret_types = str(sig), [str(t) for t in sig.rets]
        if len(node.args) != len(sig.params):
            self.err(node.pos, f"{expected_kind} '{node.name}' expects {len(sig.params)} "
                               f"argument(s), got {len(node.args)}")
        else:
            for k, (arg, at, pt) in enumerate(zip(node.args, arg_types, sig.params), 1):
                if unify(pt, at) is None:
                    self.err(arg.pos, f"argument {k} of '{node.name}': expected {pt}, got {at}")
        return list(sig.rets)

def make_diag(phase, message, pos, text):
    d = {"phase": phase, "message": message, "pos": pos}
    lc = line_col(text, pos)
    if lc:
        d["line"], d["col"] = lc
    return d


def analyze(text, infer_locals=True):
    diags = []
    try:
        tokens = fp.tokenize(text, fp.TRANS, fp.ACCEPT, fp.START, fp.TRAP)
    except Exception as e:
        m = re.search(r"position (\d+)", str(e))
        diags.append(make_diag("lexical", str(e), int(m.group(1)) if m else None, text))
        return None, diags

    parser = fp.Parser(tokens)
    program = None
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            program = parser.parse_program()
        except fp.ParseError as e:
            parser.errors.append(str(e))
    for msg in parser.errors:
        m = re.search(r" at position (-?\d+)$", msg)
        diags.append(make_diag("syntax", msg[:m.start()] if m else msg,
                               int(m.group(1)) if m else None, text))

    if program is not None:
        checker = TypeChecker(program, tokens, text, infer_locals)
        checker.check()
        sem = sorted(checker.errors, key=lambda pm: pm[0])
        diags += [make_diag("semantic", msg, pos, text) for pos, msg in sem]
    return program, diags


def format_diag(d, text):
    where = f"line {d['line']}, col {d['col']}" if "line" in d else (
        f"position {d['pos']}" if d["pos"] is not None else "unknown position")
    out = [f"error[{d['phase']}] {where}: {d['message']}"]
    if "line" in d and text is not None:
        src = text.split("\n")[d["line"] - 1]
        gutter = f"{d['line']:>4} | "
        out.append(gutter + src)
        out.append(" " * (len(gutter) - 2) + "| " + " " * (d["col"] - 1) + "^")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HW3: semantic analysis and type checking for Funny")
    ap.add_argument("source", help="program text")
    ap.add_argument("-o", "--output", default="typed_ast.json", help="annotated AST (JSON)")
    ap.add_argument("--no-infer", action="store_true",
                    help="disable local type inference: every local must be declared in 'uses'")
    args = ap.parse_args(argv)

    with open(args.source, encoding="utf-8") as f:
        text = f.read()
    program, diags = analyze(text, infer_locals=not args.no_infer)

    result = {"ast": fp._serialize(program) if program is not None else None, "errors": diags}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for d in diags:
        print(format_diag(d, text), file=sys.stderr)
    if diags:
        if any(d["phase"] == "syntax" for d in diags):
            print("note: syntax errors present; some semantic errors may be consequences of them",
                  file=sys.stderr)
        print(f"{len(diags)} error(s)", file=sys.stderr)
        return 1
    print(f"OK: no errors; annotated AST written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())