#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import funny_typization as ft

fp = ft.fp

INT, ARRAY = "int", "int[]"

@dataclass(frozen=True)
class Const:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Reg:
    """A named IR variable: source variable (param/return/local) or temporary `$tN`."""
    name: str

    def __str__(self) -> str:
        return self.name


Operand = Union[Const, Reg]

TERMINATORS = {"jmp", "cbr", "ret", "trap"}
ARITH_OPS = ("+", "-", "*", "/")
CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")
COMMUTATIVE = {"+", "*"}


@dataclass
class Instr:
    op: str
    dsts: Tuple[str, ...] = ()
    args: Tuple[Operand, ...] = ()
    sub: str = ""
    targets: Tuple[str, ...] = ()

    def uses(self) -> List[str]:
        return [a.name for a in self.args if isinstance(a, Reg)]

    @property
    def is_terminator(self) -> bool:
        return self.op in TERMINATORS


def mov(d, s): return Instr("mov", (d,), (s,))
def binop(d, op, a, b): return Instr("bin", (d,), (a, b), op)
def neg(d, a): return Instr("neg", (d,), (a,))
def aload(d, arr, idx): return Instr("aload", (d,), (arr, idx))
def astore(arr, idx, val): return Instr("astore", (), (arr, idx, val))
def alen(d, arr): return Instr("alen", (d,), (arr,))
def call(dsts, f, args): return Instr("call", tuple(dsts), tuple(args), f)
def trap(msg): return Instr("trap", sub=msg)
def jmp(t): return Instr("jmp", targets=(t,))
def cbr(op, a, b, t, f): return Instr("cbr", (), (a, b), op, (t, f))
def ret(vals): return Instr("ret", (), tuple(vals))
def label(name): return Instr("label", sub=name)


@dataclass
class Block:
    label: str
    instrs: List[Instr]

    @property
    def term(self) -> Optional[Instr]:
        return self.instrs[-1] if self.instrs else None

    def succs(self) -> List[str]:
        t = self.term
        if t is None or t.op not in ("jmp", "cbr"):
            return []
        out: List[str] = []
        for x in t.targets:
            if x not in out:
                out.append(x)
        return out


class CFG:
    def __init__(self, blocks: List[Block]):
        self.blocks = blocks

    @property
    def entry(self) -> str:
        return self.blocks[0].label

    def by_label(self) -> Dict[str, Block]:
        return {b.label: b for b in self.blocks}

    def preds(self) -> Dict[str, List[str]]:
        p: Dict[str, List[str]] = {b.label: [] for b in self.blocks}
        for b in self.blocks:
            for s in b.succs():
                p[s].append(b.label)
        return p

    def remove_unreachable(self) -> int:
        by = self.by_label()
        seen: Set[str] = set()
        stack = [self.entry]
        while stack:
            l = stack.pop()
            if l in seen:
                continue
            seen.add(l)
            stack.extend(by[l].succs())
        before = len(self.blocks)
        self.blocks = [b for b in self.blocks if b.label in seen]
        return before - len(self.blocks)


@dataclass
class IRFunction:
    name: str
    params: List[Tuple[str, str]]
    returns: List[Tuple[str, str]]
    vars: Dict[str, str]
    cfg: CFG

    def clone(self) -> "IRFunction":
        return copy.deepcopy(self)

    def instrs(self) -> Iterable[Instr]:
        for b in self.cfg.blocks:
            yield from b.instrs

    def referenced(self) -> Set[str]:
        s: Set[str] = set()
        for i in self.instrs():
            s.update(i.dsts)
            s.update(i.uses())
        return s

    def temps(self) -> List[str]:
        return sorted((v for v in self.referenced() if is_temp(v)), key=_temp_key)

    def prune_vars(self) -> None:
        used = self.referenced()
        keep = {n for n, _ in self.params} | {n for n, _ in self.returns}
        self.vars = {n: t for n, t in self.vars.items() if n in used or n in keep or not is_temp(n)}


@dataclass
class IRProgram:
    functions: List[IRFunction]

    def get(self, name: str) -> IRFunction:
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)

    def clone(self) -> "IRProgram":
        return copy.deepcopy(self)


def is_temp(name: str) -> bool:
    return name.startswith("$")


def _temp_key(n: str):
    return (0, int(n[2:])) if n[2:].isdigit() else (1, n)

class IRError(Exception):
    pass

class IRGenError(Exception):
    pass

class IRTrap(Exception):
    pass

def instr_text(i: Instr) -> str:
    op = i.op
    if op == "mov":
        return f"{i.dsts[0]} = {i.args[0]}"
    if op == "bin":
        return f"{i.dsts[0]} = {i.args[0]} {i.sub} {i.args[1]}"
    if op == "neg":
        return f"{i.dsts[0]} = neg {i.args[0]}"
    if op == "aload":
        return f"{i.dsts[0]} = {i.args[0]}[{i.args[1]}]"
    if op == "astore":
        return f"{i.args[0]}[{i.args[1]}] = {i.args[2]}"
    if op == "alen":
        return f"{i.dsts[0]} = len {i.args[0]}"
    if op == "call":
        lhs = (", ".join(i.dsts) + " = ") if i.dsts else ""
        return f"{lhs}call {i.sub}({', '.join(map(str, i.args))})"
    if op == "trap":
        return f"trap {json.dumps(i.sub, ensure_ascii=False)}"
    if op == "jmp":
        return f"goto {i.targets[0]}"
    if op == "cbr":
        return f"if {i.args[0]} {i.sub} {i.args[1]} goto {i.targets[0]} else {i.targets[1]}"
    if op == "ret":
        return "ret" + (" " + ", ".join(map(str, i.args)) if i.args else "")
    if op == "label":
        return f"{i.sub}:"
    raise IRError(f"unknown opcode {op}")


def fn_to_text(fn: IRFunction) -> str:
    sig = ", ".join(f"{n}: {t}" for n, t in fn.params)
    rets = ", ".join(f"{n}: {t}" for n, t in fn.returns)
    lines = [f"func {fn.name}({sig}) -> ({rets})"]
    fixed = {n for n, _ in fn.params} | {n for n, _ in fn.returns}
    locs = [f"{n}: {t}" for n, t in fn.vars.items() if n not in fixed and not is_temp(n)]
    tmps = [f"{n}: {fn.vars.get(n, INT)}" for n in fn.temps()]
    if locs:
        lines.append("  locals: " + ", ".join(locs))
    if tmps:
        lines.append("  temps:  " + ", ".join(tmps))
    for b in fn.cfg.blocks:
        lines.append(f"  {b.label}:")
        lines.extend("    " + instr_text(i) for i in b.instrs)
    return "\n".join(lines)


def program_to_text(p: IRProgram) -> str:
    return "\n\n".join(fn_to_text(f) for f in p.functions) + "\n"


def opnd_json(o: Operand) -> dict:
    return {"const": o.value} if isinstance(o, Const) else {"var": o.name}


def instr_json(i: Instr) -> dict:
    a = [opnd_json(x) for x in i.args]
    op = i.op
    if op == "mov":
        return {"op": "mov", "dst": i.dsts[0], "src": a[0]}
    if op == "bin":
        return {"op": "bin", "operator": i.sub, "dst": i.dsts[0], "lhs": a[0], "rhs": a[1]}
    if op == "neg":
        return {"op": "neg", "dst": i.dsts[0], "src": a[0]}
    if op == "aload":
        return {"op": "aload", "dst": i.dsts[0], "array": a[0], "index": a[1]}
    if op == "astore":
        return {"op": "astore", "array": a[0], "index": a[1], "value": a[2]}
    if op == "alen":
        return {"op": "alen", "dst": i.dsts[0], "array": a[0]}
    if op == "call":
        return {"op": "call", "callee": i.sub, "dsts": list(i.dsts), "args": a}
    if op == "trap":
        return {"op": "trap", "message": i.sub}
    if op == "jmp":
        return {"op": "jmp", "target": i.targets[0]}
    if op == "cbr":
        return {"op": "cbr", "cmp": i.sub, "lhs": a[0], "rhs": a[1],
                "then": i.targets[0], "else": i.targets[1]}
    if op == "ret":
        return {"op": "ret", "values": a}
    raise IRError(f"cannot serialise {op}")


def fn_json(fn: IRFunction) -> dict:
    return {
        "name": fn.name,
        "params": [{"name": n, "type": t} for n, t in fn.params],
        "returns": [{"name": n, "type": t} for n, t in fn.returns],
        "vars": dict(fn.vars),
        "temps": fn.temps(),
        "entry": fn.cfg.entry,
        "blocks": [{"label": b.label, "instrs": [instr_json(i) for i in b.instrs],
                    "succs": b.succs()} for b in fn.cfg.blocks],
    }


def program_to_json(p: IRProgram) -> dict:
    return {"format": "funny-ir/1", "functions": [fn_json(f) for f in p.functions]}


def _dot_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def cfg_to_dot(fn: IRFunction) -> str:
    out = [f'digraph "cfg_{fn.name}" {{', '  node [shape=box, fontname="monospace", fontsize=10];']
    for b in fn.cfg.blocks:
        body = "\\l".join(_dot_escape(instr_text(i)) for i in b.instrs) + "\\l"
        out.append(f'  "{b.label}" [label="{_dot_escape(b.label)}:\\l{body}"];')
    for b in fn.cfg.blocks:
        t = b.term
        if t is not None and t.op == "cbr":
            out.append(f'  "{b.label}" -> "{t.targets[0]}" [label="T"];')
            out.append(f'  "{b.label}" -> "{t.targets[1]}" [label="F"];')
        else:
            for s in b.succs():
                out.append(f'  "{b.label}" -> "{s}";')
    out.append("}")
    return "\n".join(out) + "\n"

def wrap32(x: int) -> int:
    return ((x + 2 ** 31) % 2 ** 32) - 2 ** 31


def eval_bin(op: str, a: int, b: int) -> int:
    if op == "+":
        return wrap32(a + b)
    if op == "-":
        return wrap32(a - b)
    if op == "*":
        return wrap32(a * b)
    if op == "/":
        if b == 0:
            raise IRTrap("division by zero")
        q = abs(a) // abs(b)
        return wrap32(-q if (a < 0) != (b < 0) else q)
    raise IRError(f"bad arithmetic operator {op}")


_CMP = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def eval_cmp(op: str, a: int, b: int) -> bool:
    return _CMP[op](a, b)

OPMAP = {
    "PLUS": "+", "MINUS": "-", "STAR": "*", "SLASH": "/",
    "EQ": "==", "NEQ": "!=", "GE": ">=", "LE": "<=", "GT": ">", "LT": "<",
    "AND": "and", "OR": "or", "NOT": "not", "→": "imp", "IMPLIES": "imp", "ARROW": "imp",
    "+": "+", "-": "-", "*": "*", "/": "/", "==": "==", "!=": "!=", ">=": ">=", "<=": "<=",
    ">": ">", "<": "<", "&&": "and", "||": "or", "!": "not", "and": "and", "or": "or",
    "not": "not", "->": "imp", "=>": "imp",
}


def _opname(op) -> str:
    try:
        return OPMAP[op]
    except KeyError:
        raise IRGenError(f"unknown operator {op!r}")


def _lit_raw(node):
    for k in ("value", "val", "n", "num", "number", "literal", "text"):
        if hasattr(node, k):
            return getattr(node, k)
    for k, v in vars(node).items():
        if k not in ("pos", "type", "kind") and isinstance(v, (int, str, bool)):
            return v
    raise IRGenError(f"cannot read literal value of {type(node).__name__}")


def _int_value(node) -> int:
    v = _lit_raw(node)
    return int(v) if not isinstance(v, bool) else int(v)


def _bool_value(node) -> bool:
    v = _lit_raw(node)
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


@dataclass
class GenOptions:
    assertions: bool = True
    contracts: bool = False


def _supported_spec(e) -> bool:
    return not any(isinstance(n, (fp.Quantifier, fp.FormulaRef)) for n in ft.walk(e))


class _Gen:
    def __init__(self, fn, opts: GenOptions, text: Optional[str]):
        self.fn, self.opts, self.text = fn, opts, text
        self.code: List[Instr] = []
        self.nlabel = self.ntemp = 0
        syms = getattr(fn, "symbols", None)
        if syms is None:
            raise IRGenError(f"function '{fn.name}' has no type annotations (run funny_typization first)")
        self.vars: Dict[str, str] = {}
        self.kinds: Dict[str, str] = {}
        for s in syms:
            if s["type"] not in (INT, ARRAY):
                raise IRGenError(f"variable '{s['name']}' of function '{fn.name}' has type {s['type']}")
            self.vars[s["name"]] = s["type"]
            self.kinds[s["name"]] = s["kind"]

    def emit(self, i: Instr) -> None:
        self.code.append(i)

    def new_label(self, hint: str) -> str:
        self.nlabel += 1
        return f"{hint}{self.nlabel}"

    def place(self, l: str) -> None:
        self.emit(label(l))

    def new_temp(self, ty: str = INT) -> str:
        name = f"$t{self.ntemp}"
        self.ntemp += 1
        self.vars[name] = ty
        return name

    def target(self, dst: Optional[str], node) -> str:
        if dst is not None:
            return dst
        ty = getattr(node, "type", INT)
        return self.new_temp(ty if ty in (INT, ARRAY) else INT)

    def where(self, node) -> str:
        lc = ft.line_col(self.text, getattr(node, "pos", None)) if self.text else None
        return f"{lc[0]}:{lc[1]}" if lc else str(getattr(node, "pos", "?"))

    def expr(self, e, dst: Optional[str] = None) -> Operand:
        """Lower a value expression; returns an operand holding the value.
        `dst` is only a hint for the *final* instruction (saves a temporary)."""
        if isinstance(e, fp.IntLit):
            return Const(wrap32(_int_value(e)))
        if isinstance(e, fp.Var):
            if e.name not in self.vars:
                raise IRGenError(f"unknown variable '{e.name}'")
            return Reg(e.name)
        if isinstance(e, fp.UnaryOp):
            if _opname(e.op) != "-":
                raise IRGenError("boolean expression used as a value")
            if isinstance(e.operand, fp.IntLit):
                return Const(wrap32(-_int_value(e.operand)))
            a = self.expr(e.operand)
            d = self.target(dst, e)
            self.emit(neg(d, a))
            return Reg(d)
        if isinstance(e, fp.BinOp):
            op = _opname(e.op)
            if op not in ARITH_OPS:
                raise IRGenError("boolean expression used as a value")
            a = self.expr(e.left)
            b = self.expr(e.right)
            d = self.target(dst, e)
            self.emit(binop(d, op, a, b))
            return Reg(d)
        if isinstance(e, fp.ArrayAccess):
            arr = self.expr(e.array)
            idx = self.expr(e.index)
            d = self.target(dst, e)
            self.emit(aload(d, arr, idx))
            return Reg(d)
        if isinstance(e, fp.LengthExpr):
            arr = self.expr(e.arg)
            d = self.target(dst, e)
            self.emit(alen(d, arr))
            return Reg(d)
        if isinstance(e, fp.FuncCall):
            args = [self.expr(a) for a in e.args]
            d = self.target(dst, e)
            self.emit(call([d], e.name, args))
            return Reg(d)
        raise IRGenError(f"cannot lower {type(e).__name__} as a value")

    def cond(self, e, lt: str, lf: str) -> None:
        if isinstance(e, fp.BoolLit):
            self.emit(jmp(lt if _bool_value(e) else lf))
            return
        if isinstance(e, fp.UnaryOp) and _opname(e.op) == "not":
            self.cond(e.operand, lf, lt)
            return
        if isinstance(e, fp.BinOp):
            op = _opname(e.op)
            if op == "and":
                mid = self.new_label("and")
                self.cond(e.left, mid, lf)
                self.place(mid)
                self.cond(e.right, lt, lf)
                return
            if op == "or":
                mid = self.new_label("or")
                self.cond(e.left, lt, mid)
                self.place(mid)
                self.cond(e.right, lt, lf)
                return
            if op == "imp":
                mid = self.new_label("imp")
                self.cond(e.left, mid, lt)
                self.place(mid)
                self.cond(e.right, lt, lf)
                return
            if op in CMP_OPS:
                a = self.expr(e.left)
                b = self.expr(e.right)
                self.emit(cbr(op, a, b, lt, lf))
                return
        raise IRGenError(f"cannot lower {type(e).__name__} as a condition")

    def check(self, e, msg: str, node) -> None:
        if not _supported_spec(e):
            return
        ok, bad = self.new_label("ok"), self.new_label("fail")
        self.cond(e, ok, bad)
        self.place(bad)
        self.emit(trap(f"{msg} at {self.where(node)}"))
        self.place(ok)

    def stmt(self, s) -> None:
        if s is None or isinstance(s, fp.ErrorStmt):
            return
        if isinstance(s, fp.Block):
            for x in s.stmts:
                self.stmt(x)
        elif isinstance(s, fp.Assign):
            t = s.target
            if isinstance(t, fp.Var):
                v = self.expr(s.value, dst=t.name)
                if v != Reg(t.name):
                    self.emit(mov(t.name, v))
            else:
                arr = self.expr(t.array)
                idx = self.expr(t.index)
                val = self.expr(s.value)
                self.emit(astore(arr, idx, val))
        elif isinstance(s, fp.TupleAssign):
            args = [self.expr(a) for a in s.call.args]
            self.emit(call([t.name for t in s.targets], s.call.name, args))
        elif isinstance(s, fp.IfStmt):
            lthen, lend = self.new_label("then"), self.new_label("endif")
            lelse = self.new_label("else") if s.else_stmt is not None else lend
            self.cond(s.cond, lthen, lelse)
            self.place(lthen)
            self.stmt(s.then_stmt)
            if s.else_stmt is not None:
                self.emit(jmp(lend))
                self.place(lelse)
                self.stmt(s.else_stmt)
            self.place(lend)
        elif isinstance(s, fp.WhileStmt):
            lh, lb, le = self.new_label("loop"), self.new_label("body"), self.new_label("endloop")
            self.place(lh)
            inv = getattr(s, "invariant", None)
            if self.opts.contracts and inv is not None:
                self.check(inv, "loop invariant violated", s)
            self.cond(s.cond, lb, le)
            self.place(lb)
            self.stmt(s.body)
            self.emit(jmp(lh))
            self.place(le)
        elif isinstance(s, (fp.AssertStmt, fp.AssumeStmt)):
            if self.opts.assertions:
                kind = "assert" if isinstance(s, fp.AssertStmt) else "assume"
                self.check(s.expr, f"{kind} failed", s)
        else:
            raise IRGenError(f"cannot lower statement {type(s).__name__}")

    def run(self) -> IRFunction:
        f = self.fn
        self.place("entry")
        for n, _ in f.returns:
            if self.vars[n] == INT:
                self.emit(mov(n, Const(0)))
        for n, k in self.kinds.items():
            if k == "local" and self.vars[n] == INT:
                self.emit(mov(n, Const(0)))
        if self.opts.contracts and getattr(f, "requires", None) is not None:
            self.check(f.requires, "requires violated", f.requires)
        self.stmt(f.body)
        if self.opts.contracts and getattr(f, "ensures", None) is not None:
            self.check(f.ensures, "ensures violated", f.ensures)
        self.emit(ret([Reg(n) for n, _ in f.returns]))
        fn = IRFunction(
            name=f.name,
            params=[(n, self.vars[n]) for n, _ in f.params],
            returns=[(n, self.vars[n]) for n, _ in f.returns],
            vars=self.vars,
            cfg=build_cfg(self.code),
        )
        validate(fn)
        return fn


def lower_program(program, opts: Optional[GenOptions] = None, text: Optional[str] = None) -> IRProgram:
    opts = opts or GenOptions()
    return IRProgram([_Gen(f, opts, text).run() for f in program.functions])

def build_cfg(code: List[Instr]) -> CFG:
    """Split a linear instruction list (with `label` pseudo-instructions) into basic blocks.
    Leaders: labels and instructions following a terminator. Fall-through becomes `goto`."""
    blocks: List[Block] = []
    cur: Optional[Block] = None
    anon = 0
    for ins in code:
        if ins.op == "label":
            if cur is not None and not (cur.instrs and cur.instrs[-1].is_terminator):
                cur.instrs.append(jmp(ins.sub))
            cur = Block(ins.sub, [])
            blocks.append(cur)
            continue
        if cur is None or (cur.instrs and cur.instrs[-1].is_terminator):
            anon += 1
            cur = Block(f"dead{anon}", [])
            blocks.append(cur)
        cur.instrs.append(ins)
    if cur is not None and not (cur.instrs and cur.instrs[-1].is_terminator):
        cur.instrs.append(ret([]))
    cfg = CFG(blocks)
    cfg.remove_unreachable()
    return cfg


def validate(fn: IRFunction) -> None:
    """Structural well-formedness check of an IR function (raises IRError)."""
    labels = [b.label for b in fn.cfg.blocks]
    if len(set(labels)) != len(labels):
        raise IRError(f"{fn.name}: duplicate block labels")
    for b in fn.cfg.blocks:
        if not b.instrs or not b.instrs[-1].is_terminator:
            raise IRError(f"{fn.name}/{b.label}: block does not end with a terminator")
        for i in b.instrs[:-1]:
            if i.is_terminator or i.op == "label":
                raise IRError(f"{fn.name}/{b.label}: terminator in the middle of a block")
        for t in b.instrs[-1].targets:
            if t not in labels:
                raise IRError(f"{fn.name}/{b.label}: jump to unknown label {t}")
        for i in b.instrs:
            for v in list(i.dsts) + i.uses():
                if v not in fn.vars:
                    raise IRError(f"{fn.name}/{b.label}: undeclared variable {v}")

def liveness(fn: IRFunction) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    blocks = fn.cfg.blocks
    use: Dict[str, Set[str]] = {}
    defn: Dict[str, Set[str]] = {}
    for b in blocks:
        u: Set[str] = set()
        d: Set[str] = set()
        for ins in b.instrs:
            u.update(x for x in ins.uses() if x not in d)
            d.update(ins.dsts)
        use[b.label], defn[b.label] = u, d
    live_in = {b.label: set() for b in blocks}
    live_out = {b.label: set() for b in blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            out: Set[str] = set()
            for s in b.succs():
                out |= live_in[s]
            inn = use[b.label] | (out - defn[b.label])
            if out != live_out[b.label] or inn != live_in[b.label]:
                live_out[b.label], live_in[b.label] = out, inn
                changed = True
    return live_in, live_out

def _map_args(ins: Instr, f) -> Instr:
    return Instr(ins.op, ins.dsts, tuple(f(a) for a in ins.args), ins.sub, ins.targets)


def _fold(ins: Instr) -> Instr:
    a = ins.args
    if ins.op == "bin" and all(isinstance(x, Const) for x in a):
        if ins.sub == "/" and a[1].value == 0:
            return ins
        return mov(ins.dsts[0], Const(eval_bin(ins.sub, a[0].value, a[1].value)))
    if ins.op == "neg" and isinstance(a[0], Const):
        return mov(ins.dsts[0], Const(wrap32(-a[0].value)))
    if ins.op == "cbr" and all(isinstance(x, Const) for x in a):
        return jmp(ins.targets[0] if eval_cmp(ins.sub, a[0].value, a[1].value) else ins.targets[1])
    return ins


def p_constprop(fn: IRFunction) -> int:
    n = 0
    for b in fn.cfg.blocks:
        env: Dict[str, int] = {}
        out = []
        for ins in b.instrs:
            new = _map_args(ins, lambda o: Const(env[o.name]) if isinstance(o, Reg) and o.name in env else o)
            new = _fold(new)
            if new != ins:
                n += 1
            for d in new.dsts:
                env.pop(d, None)
            if new.op == "mov" and isinstance(new.args[0], Const):
                env[new.dsts[0]] = new.args[0].value
            out.append(new)
        b.instrs = out
    return n


def _algebraic(ins: Instr) -> Optional[Instr]:
    if ins.op == "bin":
        op, (a, b) = ins.sub, ins.args
        d = ins.dsts[0]
        if op in COMMUTATIVE and isinstance(a, Const) and not isinstance(b, Const):
            a, b = b, a
        bc = b.value if isinstance(b, Const) else None
        ac = a.value if isinstance(a, Const) else None
        if op == "+" and bc == 0:
            return mov(d, a)
        if op == "-":
            if bc == 0:
                return mov(d, a)
            if ac == 0 and isinstance(b, Reg):
                return neg(d, b)
            if isinstance(a, Reg) and a == b:
                return mov(d, Const(0))
        if op == "*":
            if bc == 1:
                return mov(d, a)
            if bc == 0:
                return mov(d, Const(0))
            if bc == 2 and isinstance(a, Reg):
                return binop(d, "+", a, a)
            if bc == -1 and isinstance(a, Reg):
                return neg(d, a)
        if op == "/":
            if bc == 1:
                return mov(d, a)
            if bc == -1 and isinstance(a, Reg):
                return neg(d, a)
        return Instr("bin", ins.dsts, (a, b), op)
    if ins.op == "cbr":
        a, b = ins.args
        if isinstance(a, Reg) and a == b:
            return jmp(ins.targets[0] if ins.sub in ("==", "<=", ">=") else ins.targets[1])
    if ins.op == "mov" and isinstance(ins.args[0], Reg) and ins.args[0].name == ins.dsts[0]:
        return None                                            # x = x
    return ins


def p_algebraic(fn: IRFunction) -> int:
    n = 0
    for b in fn.cfg.blocks:
        out = []
        for ins in b.instrs:
            new = _algebraic(ins)
            if new != ins:
                n += 1
            if new is not None:
                out.append(new)
        b.instrs = out
    return n


def p_copyprop(fn: IRFunction) -> int:
    n = 0
    for b in fn.cfg.blocks:
        copies: Dict[str, Reg] = {}
        out = []
        for ins in b.instrs:
            new = _map_args(ins, lambda o: copies.get(o.name, o) if isinstance(o, Reg) else o)
            if new != ins:
                n += 1
            for d in new.dsts:
                copies.pop(d, None)
                for k in [k for k, v in copies.items() if v.name == d]:
                    del copies[k]
            if new.op == "mov" and isinstance(new.args[0], Reg) and new.args[0].name != new.dsts[0]:
                copies[new.dsts[0]] = new.args[0]
            out.append(new)
        b.instrs = out
    return n


def _cse_key(ins: Instr):
    if ins.op == "bin":
        a, b = ins.args
        if ins.sub in COMMUTATIVE and repr(a) > repr(b):
            a, b = b, a
        return ("bin", ins.sub, a, b)
    if ins.op in ("neg", "alen"):
        return (ins.op, ins.args[0])
    if ins.op == "aload":
        return ("aload", ins.args[0], ins.args[1])
    return None


def p_cse(fn: IRFunction) -> int:
    n = 0
    for b in fn.cfg.blocks:
        avail: Dict[tuple, str] = {}
        out = []
        for ins in b.instrs:
            key = _cse_key(ins)
            new = ins
            if key is not None and key in avail:
                new = mov(ins.dsts[0], Reg(avail[key]))
                n += 1

            for d in new.dsts:
                for k in [k for k, h in avail.items() if h == d or Reg(d) in k]:
                    del avail[k]
            if new.op == "astore":
                for k in [k for k in avail if k[0] == "aload"]:
                    del avail[k]
            if new is ins and key is not None and ins.dsts[0] not in ins.uses():
                avail[key] = ins.dsts[0]
            out.append(new)
        b.instrs = out
    return n


def _is_pure(ins: Instr) -> bool:
    if ins.op in ("mov", "neg"):
        return True
    if ins.op == "bin":
        return ins.sub != "/" or (isinstance(ins.args[1], Const) and ins.args[1].value != 0)
    return False


def p_dce(fn: IRFunction) -> int:
    total = 0
    while True:
        _, live_out = liveness(fn)
        removed = 0
        for b in fn.cfg.blocks:
            live = set(live_out[b.label])
            keep: List[Instr] = []
            for ins in reversed(b.instrs):
                if _is_pure(ins) and ins.dsts[0] not in live:
                    removed += 1
                    continue
                live -= set(ins.dsts)
                live.update(ins.uses())
                keep.append(ins)
            keep.reverse()
            b.instrs = keep
        total += removed
        if not removed:
            return total


def p_cfg(fn: IRFunction) -> int:
    cfg = fn.cfg
    n = 0
    changed = True
    while changed:
        changed = False
        for b in cfg.blocks:
            t = b.term
            if t.op == "cbr" and t.targets[0] == t.targets[1]:
                b.instrs[-1] = jmp(t.targets[0])
                n += 1
                changed = True
        fwd = {b.label: b.instrs[0].targets[0] for b in cfg.blocks
               if len(b.instrs) == 1 and b.instrs[0].op == "jmp" and b.instrs[0].targets[0] != b.label}

        def resolve(l: str) -> str:
            seen: Set[str] = set()
            while l in fwd and l not in seen:
                seen.add(l)
                l = fwd[l]
            return l

        for b in cfg.blocks:
            t = b.term
            if t.op in ("jmp", "cbr"):
                nt = tuple(resolve(x) for x in t.targets)
                if nt != t.targets:
                    b.instrs[-1] = Instr(t.op, t.dsts, t.args, t.sub, nt)
                    n += 1
                    changed = True
        removed = cfg.remove_unreachable()
        if removed:
            n += removed
            changed = True
        preds = cfg.preds()
        by = cfg.by_label()
        for b in cfg.blocks:
            t = b.term
            if t.op == "jmp":
                s = by[t.targets[0]]
                if s is not b and s.label != cfg.entry and preds[s.label] == [b.label]:
                    b.instrs = b.instrs[:-1] + s.instrs
                    cfg.blocks.remove(s)
                    n += 1
                    changed = True
                    break
    return n


PASSES = (("constprop", p_constprop), ("algebraic", p_algebraic), ("copyprop", p_copyprop),
          ("cse", p_cse), ("dce", p_dce), ("cfg", p_cfg))
PASS_NAMES = tuple(n for n, _ in PASSES)


def optimize(fn: IRFunction, enabled: Iterable[str] = PASS_NAMES, max_rounds: int = 25) -> Dict[str, int]:
    enabled = set(enabled)
    bad = enabled - set(PASS_NAMES)
    if bad:
        raise ValueError(f"unknown optimization(s): {sorted(bad)}; available: {list(PASS_NAMES)}")
    stats = {n: 0 for n in PASS_NAMES if n in enabled}
    for _ in range(max_rounds):
        round_changes = 0
        for name, p in PASSES:
            if name in enabled:
                c = p(fn)
                stats[name] += c
                round_changes += c
                validate(fn)
        if not round_changes:
            break
    fn.prune_vars()
    return stats

@dataclass
class Coloring:
    scope: str
    nodes: List[str]
    edges: Set[frozenset]
    color: Dict[str, str]
    max_live: int

    @property
    def num_colors(self) -> int:
        return len(set(self.color.values()))


def _cls(fn: IRFunction, v: str) -> str:
    return "a" if fn.vars.get(v) == ARRAY else "i"


def build_conflicts(fn: IRFunction, scope: str = "temps") -> Tuple[List[str], Set[frozenset], int]:
    params = {n for n, _ in fn.params}
    ref = fn.referenced()
    nodes = sorted((v for v in ref if (is_temp(v) if scope == "temps" else v not in params)),
                   key=lambda v: (not is_temp(v), _temp_key(v) if is_temp(v) else (0, v)))
    in_scope = set(nodes)
    _, live_out = liveness(fn)
    edges: Set[frozenset] = set()
    max_live = 0

    def add(a: str, b: str) -> None:
        if a != b and a in in_scope and b in in_scope and _cls(fn, a) == _cls(fn, b):
            edges.add(frozenset((a, b)))

    for b in fn.cfg.blocks:
        live = set(live_out[b.label])
        max_live = max(max_live, len(live & in_scope))
        for ins in reversed(b.instrs):
            moved_from = ins.args[0].name if ins.op == "mov" and isinstance(ins.args[0], Reg) else None
            for d in ins.dsts:
                for l in live:
                    if l != moved_from:
                        add(d, l)
                for d2 in ins.dsts:
                    add(d, d2)
            live -= set(ins.dsts)
            live.update(ins.uses())
            max_live = max(max_live, len(live & in_scope))
    return nodes, edges, max_live


def greedy_color(fn: IRFunction, scope: str = "temps") -> Coloring:
    nodes, edges, max_live = build_conflicts(fn, scope)
    adj: Dict[str, Set[str]] = {v: set() for v in nodes}
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    order = sorted(nodes, key=lambda v: (-len(adj[v]), nodes.index(v)))
    idx: Dict[str, int] = {}
    for v in order:
        taken = {idx[u] for u in adj[v] if u in idx}
        c = 0
        while c in taken:
            c += 1
        idx[v] = c
    color = {v: f"{_cls(fn, v)}{idx[v]}" for v in nodes}
    for e in edges:                                           # self-check
        a, b = tuple(e)
        assert color[a] != color[b], "colouring conflict"
    return Coloring(scope, nodes, edges, color, max_live)


_PALETTE = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69",
            "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]


def conflicts_to_dot(fn: IRFunction, c: Coloring) -> str:
    names = sorted(set(c.color.values()))
    out = [f'graph "conflicts_{fn.name}" {{', '  node [style=filled, fontname="monospace"];']
    for v in c.nodes:
        fill = _PALETTE[names.index(c.color[v]) % len(_PALETTE)]
        out.append(f'  "{v}" [label="{_dot_escape(v)}\\n{c.color[v]}", fillcolor="{fill}"];')
    for e in sorted(tuple(sorted(e)) for e in c.edges):
        out.append(f'  "{e[0]}" -- "{e[1]}";')
    out.append("}")
    return "\n".join(out) + "\n"


def conflicts_to_text(fn: IRFunction, c: Coloring) -> str:
    adj = {v: sorted(u for e in c.edges if v in e for u in e if u != v) for v in c.nodes}
    lines = [f"interference graph of {fn.name} (scope={c.scope}): {len(c.nodes)} nodes, "
             f"{len(c.edges)} edges, max simultaneously live = {c.max_live}, colours = {c.num_colors}"]
    for v in c.nodes:
        lines.append(f"  {v:<8} color={c.color[v]:<4} conflicts: {', '.join(adj[v]) or '-'}")
    return "\n".join(lines) + "\n"

class Machine:
    def __init__(self, prog: IRProgram, max_steps: int = 5_000_000, max_depth: int = 300):
        self.prog, self.max_steps, self.max_depth = prog, max_steps, max_depth
        self.steps = 0
        self.index = {f.name: (f, f.cfg.by_label()) for f in prog.functions}

    def call(self, name: str, args: list, depth: int = 0):
        if depth > self.max_depth:
            raise IRTrap("call depth exceeded")
        fn, labels = self.index[name]
        if len(args) != len(fn.params):
            raise IRTrap(f"{name}: expected {len(fn.params)} arguments, got {len(args)}")
        env: Dict[str, object] = {n: a for (n, _), a in zip(fn.params, args)}

        def val(o: Operand):
            if isinstance(o, Const):
                return o.value
            if o.name not in env:
                raise IRTrap(f"use of undefined variable {o.name}")
            return env[o.name]

        def arr(o: Operand) -> list:
            a = val(o)
            if not isinstance(a, list):
                raise IRTrap("array expected")
            return a

        def index(a: list, i: int) -> int:
            if not 0 <= i < len(a):
                raise IRTrap(f"array index {i} out of bounds (length {len(a)})")
            return i

        block = fn.cfg.blocks[0]
        while True:
            nxt = None
            for ins in block.instrs:
                self.steps += 1
                if self.steps > self.max_steps:
                    raise IRTrap("step limit exceeded")
                op = ins.op
                if op == "mov":
                    env[ins.dsts[0]] = val(ins.args[0])
                elif op == "bin":
                    env[ins.dsts[0]] = eval_bin(ins.sub, val(ins.args[0]), val(ins.args[1]))
                elif op == "neg":
                    env[ins.dsts[0]] = wrap32(-val(ins.args[0]))
                elif op == "aload":
                    a = arr(ins.args[0])
                    env[ins.dsts[0]] = a[index(a, val(ins.args[1]))]
                elif op == "astore":
                    a = arr(ins.args[0])
                    a[index(a, val(ins.args[1]))] = val(ins.args[2])
                elif op == "alen":
                    env[ins.dsts[0]] = len(arr(ins.args[0]))
                elif op == "call":
                    res = self.call(ins.sub, [val(x) for x in ins.args], depth + 1)
                    for d, r in zip(ins.dsts, res):
                        env[d] = r
                elif op == "trap":
                    raise IRTrap(ins.sub)
                elif op == "jmp":
                    nxt = labels[ins.targets[0]]
                elif op == "cbr":
                    ok = eval_cmp(ins.sub, val(ins.args[0]), val(ins.args[1]))
                    nxt = labels[ins.targets[0 if ok else 1]]
                elif op == "ret":
                    return tuple(env.get(x.name) if isinstance(x, Reg) else x.value for x in ins.args)
                else:
                    raise IRError(f"cannot execute {op}")
            if nxt is None:
                raise IRError(f"{name}/{block.label}: fell off the end of a block")
            block = nxt


def run_ir(prog: IRProgram, name: str, args: list, max_steps: int = 5_000_000) -> Tuple[tuple, int]:
    m = Machine(prog, max_steps)
    res = m.call(name, copy.deepcopy(args))
    return res, m.steps

def fn_metrics(fn: IRFunction) -> dict:
    t = fn.temps()
    ct, ca = greedy_color(fn, "temps"), greedy_color(fn, "all")
    return {"instructions": sum(len(b.instrs) for b in fn.cfg.blocks),
            "blocks": len(fn.cfg.blocks), "temps": len(t),
            "temp_colors": ct.num_colors, "temp_max_live": ct.max_live,
            "var_colors": ca.num_colors, "var_max_live": ca.max_live}


def _parse_list(s: Optional[str]) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def select_passes(opt: str, disable: str) -> List[str]:
    if opt in ("all", ""):
        chosen = list(PASS_NAMES)
    elif opt == "none":
        chosen = []
    else:
        chosen = _parse_list(opt)
    chosen = [p for p in chosen if p not in _parse_list(disable)]
    bad = set(chosen) - set(PASS_NAMES)
    if bad:
        raise SystemExit(f"unknown optimization(s): {sorted(bad)}; available: {list(PASS_NAMES)}")
    return chosen


@dataclass
class Result:
    before: IRProgram
    after: IRProgram
    enabled: List[str]
    stats: Dict[str, Dict[str, int]]
    metrics: Dict[str, Dict[str, dict]]


def compile_program(program, enabled: Iterable[str] = PASS_NAMES, opts: Optional[GenOptions] = None,
                    text: Optional[str] = None) -> Result:
    enabled = list(enabled)
    before = lower_program(program, opts, text)
    after = before.clone()
    stats, metrics = {}, {}
    for fb, fa in zip(before.functions, after.functions):
        stats[fa.name] = optimize(fa, enabled)
        metrics[fa.name] = {"before": fn_metrics(fb), "after": fn_metrics(fa)}
    return Result(before, after, enabled, stats, metrics)


def report_text(r: Result) -> str:
    lines = [f"enabled optimizations: {', '.join(r.enabled) or '(none)'}", ""]
    keys = [("instructions", "instructions"), ("blocks", "basic blocks"), ("temps", "temporaries"),
            ("temp_colors", "colours (temps)"), ("temp_max_live", "max live temps"),
            ("var_colors", "colours (all vars)"), ("var_max_live", "max live vars")]
    for name, m in r.metrics.items():
        lines.append(f"function {name}")
        lines.append(f"  {'metric':<20}{'before':>8}{'after':>8}")
        for k, title in keys:
            lines.append(f"  {title:<20}{m['before'][k]:>8}{m['after'][k]:>8}")
        lines.append("  changes per pass: " + (", ".join(f"{p}={c}" for p, c in r.stats[name].items()) or "-"))
        lines.append("")
    return "\n".join(lines)


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_outputs(r: Result, outdir: str, dot: bool = True, color_scope: str = "temps") -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    files = {
        "ir_before.txt": program_to_text(r.before),
        "ir_after.txt": program_to_text(r.after),
        "ir_before.json": json.dumps(program_to_json(r.before), indent=2, ensure_ascii=False),
        "ir_after.json": json.dumps(program_to_json(r.after), indent=2, ensure_ascii=False),
        "metrics.json": json.dumps({"enabled": r.enabled, "passes": r.stats, "functions": r.metrics}, indent=2),
        "report.txt": report_text(r),
    }
    for fb, fa in zip(r.before.functions, r.after.functions):
        for tag, f in (("before", fb), ("after", fa)):
            c = greedy_color(f, color_scope)
            files[f"conflicts_{f.name}_{tag}.txt"] = conflicts_to_text(f, c)
            if dot:
                files[f"cfg_{f.name}_{tag}.dot"] = cfg_to_dot(f)
                files[f"conflicts_{f.name}_{tag}.dot"] = conflicts_to_dot(f, c)
    for fn_, content in files.items():
        _write(os.path.join(outdir, fn_), content)
    return sorted(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Funny HW4: IR generation, CFG, optimizations, colouring")
    ap.add_argument("source", help="Funny program text")
    ap.add_argument("-o", "--out-dir", default="ir_out", help="directory for results (default: ir_out)")
    ap.add_argument("--opt", default="all",
                    help=f"passes to run: all | none | comma list of {','.join(PASS_NAMES)} (default: all)")
    ap.add_argument("--disable", default="", help="comma list of passes to switch off")
    ap.add_argument("--no-assert", action="store_true", help="drop assert/assume statements instead of lowering them")
    ap.add_argument("--check-contracts", action="store_true",
                    help="also lower requires/ensures/loop invariants into runtime checks")
    ap.add_argument("--color-scope", choices=("temps", "all"), default="temps",
                    help="which variables are coloured in the written conflict graphs")
    ap.add_argument("--no-dot", action="store_true", help="do not write DOT files")
    ap.add_argument("--no-infer", action="store_true", help="forwarded to the type checker")
    ap.add_argument("--print", dest="show", choices=("before", "after", "both"), help="print IR to stdout")
    ap.add_argument("--run", metavar="FUNC", help="execute FUNC before and after optimization and compare")
    ap.add_argument("--args", default="[]", help='JSON list of arguments for --run, e.g. "[5, [1,2,3]]"')
    args = ap.parse_args(argv)

    with open(args.source, encoding="utf-8") as f:
        text = f.read()
    program, diags = ft.analyze(text, infer_locals=not args.no_infer)
    if diags:
        for d in diags:
            print(ft.format_diag(d, text), file=sys.stderr)
        print(f"{len(diags)} error(s) in the front end; IR not generated", file=sys.stderr)
        return 1

    enabled = select_passes(args.opt, args.disable)
    gen = GenOptions(assertions=not args.no_assert, contracts=args.check_contracts)
    try:
        res = compile_program(program, enabled, gen, text)
    except IRGenError as e:
        print(f"error[ir]: {e}", file=sys.stderr)
        return 1

    written = write_outputs(res, args.out_dir, dot=not args.no_dot, color_scope=args.color_scope)
    if args.show in ("before", "both"):
        print("# IR before optimization\n" + program_to_text(res.before))
    if args.show in ("after", "both"):
        print("# IR after optimization\n" + program_to_text(res.after))
    print(report_text(res))
    print(f"wrote {len(written)} files to {args.out_dir}/")

    if args.run:
        call_args = json.loads(args.args)
        outs = []
        for tag, prog in (("before", res.before), ("after", res.after)):
            try:
                outs.append((tag, run_ir(prog, args.run, call_args), None))
            except IRTrap as e:
                outs.append((tag, None, str(e)))
        for tag, ok, err in outs:
            print(f"run {args.run}{call_args} [{tag}]: " +
                  (f"result={ok[0]}, executed instructions={ok[1]}" if ok else f"trap: {err}"))
        same = (outs[0][1] and outs[1][1] and outs[0][1][0] == outs[1][1][0]) or \
               (outs[0][2] is not None and outs[1][2] is not None)
        print("results identical: " + ("yes" if same else "NO"))
        return 0 if same else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
