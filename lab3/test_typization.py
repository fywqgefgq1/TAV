import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import funny_parser as fp
from funny_typization import analyze, walk

EXPR_NODES = (fp.IntLit, fp.BoolLit, fp.Var, fp.BinOp, fp.UnaryOp, fp.ArrayAccess,
              fp.FuncCall, fp.LengthExpr, fp.Quantifier, fp.FormulaRef)


def prog(src):
    return src.strip("\n")


class Base(unittest.TestCase):
    def run_src(self, src, infer=True):
        return analyze(prog(src), infer_locals=infer)

    def assertErrors(self, src, expected, infer=True):
        """expected: list of (line, fragment); order-independent, exact count."""
        _, diags = self.run_src(src, infer)
        got = sorted((d.get("line"), d["message"]) for d in diags)
        self.assertEqual(len(got), len(expected), "\n" + "\n".join(map(str, got)))
        for (line, frag) in expected:
            self.assertTrue(any(l == line and frag in m for l, m in got),
                            f"missing error line {line}: {frag!r}; got:\n" + "\n".join(map(str, got)))

    def assertClean(self, src, infer=True):
        program, diags = self.run_src(src, infer)
        self.assertEqual(diags, [])
        for n in walk(program):
            if isinstance(n, EXPR_NODES):
                self.assertTrue(hasattr(n, "type"), f"{type(n).__name__} has no type annotation")
                self.assertNotEqual(n.type, "<error>")
        json.dumps(fp._serialize(program))
        return program


class Positive(Base):
    def test_loops_arrays_requires(self):
        p = self.assertClean("""
function sum(a: int[], n: int) returns r: int
requires n >= 0 and n <= length(a)
ensures r >= 0 or r < 0
uses i: int
{
  r = 0;
  i = 0;
  while (i < n) invariant i >= 0 {
    r = r + a[i];
    i = i + 1;
  }
}
""")
        f = p.functions[0]
        self.assertEqual(f.type, "(int[], int) -> (int)")
        kinds = {s["name"]: (s["kind"], s["type"]) for s in f.symbols}
        self.assertEqual(kinds, {"a": ("param", "int[]"), "n": ("param", "int"),
                                 "r": ("return", "int"), "i": ("local", "int")})
        acc = [n for n in walk(f.body) if isinstance(n, fp.ArrayAccess)][0]
        self.assertEqual((acc.type, acc.array.type, acc.index.type), ("int", "int[]", "int"))

    def test_calls_tuple_assign_forward_reference_recursion(self):
        p = self.assertClean("""
function inc(x: int) returns y: int { y = x + 1; }
function divmod(a: int, b: int) returns q: int, r: int {
  q = a / b;
  r = a - q * b;
}
function main(n: int) returns ok: int
{
  t = inc(n);
  u, v = divmod(t, 3);
  ok = 1;
  if (u > v and fact(3) > 0) ok = 0;
}
function fact(n: int) returns r: int {
  if (n <= 1) r = 1; else r = n * fact(n - 1);
}
""")
        main = [f for f in p.functions if f.name == "main"][0]
        loc = {s["name"]: s["type"] for s in main.symbols}
        self.assertEqual((loc["t"], loc["u"], loc["v"]), ("int", "int", "int"))
        ta = [n for n in walk(main.body) if isinstance(n, fp.TupleAssign)][0]
        self.assertEqual(ta.types, ["int", "int"])
        self.assertEqual(p.globals["divmod"]["type"], "(int, int) -> (int, int)")

    def test_formulas_and_quantifiers(self):
        p = self.assertClean("""
sorted(a: int[]) → forall(i: int i >= 0 and i + 1 < length(a) → a[i] <= a[i + 1]);
function id(a: int[]) returns r: int
requires sorted(a)
ensures exists(k: int k >= 0 → k == r)
{
  assert sorted(a);
  r = 0;
}
""")
        self.assertEqual(p.globals["sorted"], {"kind": "formula", "type": "(int[]) -> (bool)"})
        q = [n for n in walk(p) if isinstance(n, fp.Quantifier)][0]
        self.assertEqual(q.type, "bool")
        bound = [n for n in walk(q) if isinstance(n, fp.Var) and n.name == q.var_name]
        self.assertTrue(bound and all(v.kind == "bound" for v in bound))

    def test_bonus_inference_of_locals(self):
        p = self.assertClean("""
function g(a: int[]) returns r: int {
  n = length(a);
  m = n + 1;
  k, j = pair(m);
  r = k + j;
}
function pair(x: int) returns p: int, q: int { p = x; q = x; }
""")
        loc = {s["name"]: (s["type"], s["inferred"]) for s in p.functions[0].symbols}
        for v in "nmkj":
            self.assertEqual(loc[v], ("int", True))

    def test_inference_inside_branch_is_function_scoped(self):
        self.assertClean("""
function g(x: int) returns r: int {
  if (x > 0) t = 1; else t = 2;
  r = t;
}
""")

    def test_declared_local_via_uses_without_inference(self):
        self.assertClean("""
function g(x: int) returns r: int
uses t: int
{
  t = x;
  r = t;
}
""", infer=False)

    def test_array_output_parameter_is_writable(self):
        self.assertClean("""
function g(n: int) returns b: int[]
{
  b[0] = n;
}
""")

    def test_json_roundtrip_contains_types(self):
        program, diags = self.run_src("function f(x: int) returns r: int { r = x + 1; }")
        self.assertEqual(diags, [])
        d = json.loads(json.dumps(fp._serialize(program)))
        binop = d["functions"][0]["body"]["stmts"][0]["value"]
        self.assertEqual((binop["_type"], binop["type"]), ("BinOp", "int"))


class Negative(Base):
    def test_undeclared_variable(self):
        self.assertErrors("""
function f(x: int) returns r: int
{
  r = y + 1;
}
""", [(3, "undeclared variable 'y'")])

    def test_use_before_first_assignment(self):
        self.assertErrors("""
function f(x: int) returns r: int
{
  r = t + 1;
  t = 5;
}
""", [(3, "'t' is used before its first assignment")])

    def test_duplicate_declarations(self):
        self.assertErrors("""
function f(x: int, x: int[]) returns x: int
{
}
function f(y: int) returns r: int { r = y; }
""", [(1, "duplicate declaration of 'x'"), (1, "duplicate declaration of 'x'"),
      (4, "redefinition of function 'f'")])

    def test_local_duplicates_parameter(self):
        self.assertErrors("""
function f(x: int) returns r: int
uses x: int
{
  r = 1;
}
""", [(2, "duplicate declaration of 'x'")])

    def test_function_formula_name_clash(self):
        self.assertErrors("""
p(x: int) → x > 0;
function p(x: int) returns r: int { r = x; }
""", [(2, "redefinition of formula 'p'")])

    def test_incompatible_assignment(self):
        self.assertErrors("""
function f(x: int) returns r: int, b: int[]
{
  r = true;
  b = x;
}
""", [(3, "cannot assign bool to variable 'r' of type int"),
      (4, "cannot assign int to variable 'b' of type int[]")])

    def test_conditions_and_operators(self):
        self.assertErrors("""
function f(x: int, a: int[]) returns r: int
{
  if (x) r = 1;
  while (a + 1 > 0) r = 2;
  if (x == a) r = 3;
}
""", [(3, "condition of 'if' must be bool, got int"),
      (4, "operator '+' expects int operands, got int[] and int"),
      (5, "operator '==' expects int operands, got int and int[]")])

    def test_arrays_and_read_only_parameters(self):
        self.assertErrors("""
function f(x: int, a: int[]) returns r: int, b: int[]
{
  r = x[0];
  r = a[true];
  r = length(x);
  b = 5;
  b[0] = b;
  a[0] = 1;
  x = 2;
}
""", [(3, "cannot index a value of type int"),
      (4, "array index must be int, got bool"),
      (5, "length() expects int[], got int"),
      (6, "cannot assign int to variable 'b' of type int[]"),
      (7, "cannot assign int[] to an element"),
      (8, "cannot modify an element of parameter 'a'"),
      (9, "cannot assign to parameter 'x'")])

    def test_tuple_assign_to_parameter(self):
        self.assertErrors("""
function pair(a: int) returns x: int, y: int { x = a; y = a; }
function f(p: int) returns r: int
{
  p, r = pair(p);
}
""", [(4, "cannot assign to parameter 'p'")])

    def test_function_calls(self):
        self.assertErrors("""
function g(a: int, b: int[]) returns r: int { r = a; }
function pair(a: int) returns x: int, y: int { x = a; y = a; }
function one(a: int) returns r: int { r = a; }
function f(x: int) returns r: int
{
  r = g(x);
  r = g(x, x);
  r = unknown(x);
  r = pair(x);
  a, b, c = pair(x);
  a, b = one(x);
  r = f;
}
""", [(6, "function 'g' expects 2 argument(s), got 1"),
      (7, "argument 2 of 'g': expected int[], got int"),
      (8, "call to undefined function 'unknown'"),
      (9, "function 'pair' returns 2 values"),
      (10, "function 'pair' returns 2 value(s) but 3 target(s) given"),
      (11, "function 'one' returns 1 value(s) but 2 target(s) given"),
      (12, "'f' is a function, not a variable")])

    def test_formula_references(self):
        self.assertErrors("""
positive(x: int) → x > 0;
positive(y: int) → y > 1;
function f(x: int) returns r: int
requires positive(x, x) and positive(true) and nope(x)
{
  r = positive(x);
}
""", [(2, "redefinition of formula 'positive'"),
      (4, "formula 'positive' expects 1 argument(s), got 2"),
      (4, "argument 1 of 'positive': expected int, got bool"),
      (4, "call to undefined formula 'nope'"),
      (6, "'positive' is a formula, not a function")])

    def test_unknown_types_and_bad_quantifier_body(self):
        self.assertErrors("""
function f(x: float, y: bool) returns r: int
requires forall(i: int i > 0 → i)
{
  r = 1;
}
""", [(1, "unknown type 'float'"), (1, "unknown type 'bool'"),
      (2, "operator '→' expects bool operands, got bool and int")])

    def test_locals_must_be_int(self):
        self.assertErrors("""
function f(a: int[]) returns r: int
uses t: int[]
{
  b = a;
  r = 0;
}
""", [(2, "local variable 't' must be int"), (4, "local variable 'b' must be int")])

    def test_scopes_of_requires_and_ensures(self):
        self.assertErrors("""
function f(x: int) returns r: int
requires r > 0
ensures r >= x and q == 0
uses q: int
{
  r = x;
  q = 0;
}
""", [(2, "variable 'r' (return) cannot be used in the precondition"),
      (3, "variable 'q' (local) cannot be used in the postcondition")])

    def test_quantifier_variable_must_be_unique(self):
        self.assertErrors("""
function f(i: int) returns r: int
requires forall(i: int i > 0 → i > 1)
{
  r = 0;
}
""", [(2, "quantifier variable 'i' duplicates the param")])

    def test_inference_conflict(self):
        self.assertErrors("""
function f(x: int) returns r: int
{
  t = 1;
  t = true;
  r = t;
  s = t + y;
}
""", [(4, "cannot assign bool to variable 't' of type int"),
      (6, "undeclared variable 'y'")])

    def test_inference_disabled(self):
        self.assertErrors("""
function f(x: int) returns r: int
{
  t = 1;
  r = t;
}
""", [(3, "assignment to undeclared variable 't'"), (4, "undeclared variable 't'")], infer=False)

    def test_failed_inference_does_not_cascade(self):
        self.assertErrors("""
function f(x: int) returns r: int
{
  t = y;
  t = 7;
  r = t + 1;
}
""", [(3, "undeclared variable 'y'")])

    def test_syntax_and_semantic_errors_are_both_reported(self):
        _, diags = self.run_src("""
function f(x: int) returns r: int
{
  r = ;
  r = true;
}
""")
        phases = [(d["phase"], d.get("line")) for d in diags]
        self.assertIn(("syntax", 3), phases)
        self.assertIn(("semantic", 4), phases)

    def test_lexical_error_has_position(self):
        _, diags = self.run_src("function f(x: int) returns r: int { r = x # 1; }")
        self.assertEqual(diags[0]["phase"], "lexical")
        self.assertEqual((diags[0]["line"], diags[0]["col"]), (1, 43))

    def test_errors_carry_positions(self):
        _, diags = self.run_src("""
function f(x: int) returns r: int
{
  r = y;
}
""")
        d = diags[0]
        self.assertEqual((d["line"], d["col"]), (3, 7))
        self.assertIsInstance(d["pos"], int)


if __name__ == "__main__":
    unittest.main()