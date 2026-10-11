from collections import defaultdict, deque
import json

class R:
    pass

class RChar(R):
    def __init__(self, c): self.c = c

class RClass(R):
    def __init__(self, chars): self.chars = frozenset(chars)

class RAny(R):
    pass

class REps(R):
    pass

class RCat(R):
    def __init__(self, a, b): self.a, self.b = a, b

class RAlt(R):
    def __init__(self, a, b): self.a, self.b = a, b

class RStar(R):
    def __init__(self, a): self.a = a

class RegexParser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else None

    def next(self):
        c = self.peek()
        self.i += 1
        return c

    def expect(self, c):
        if self.next() != c:
            raise SyntaxError(f"expected {c!r} at {self.i-1}")

    @staticmethod
    def _unescape(c):
        return {'n': '\n', 't': '\t', 'r': '\r'}.get(c, c)

    def parse(self):
        node = self.parse_alt()
        if self.i != len(self.s):
            raise SyntaxError(f"unexpected {self.s[self.i]!r} at {self.i}")
        return node

    def parse_alt(self):
        node = self.parse_cat()
        while self.peek() == '|':
            self.next()
            rhs = self.parse_cat()
            node = RAlt(node, rhs)
        return node

    def parse_cat(self):
        node = None
        while self.peek() is not None and self.peek() not in ('|', ')'):
            rhs = self.parse_rep()
            node = rhs if node is None else RCat(node, rhs)
        return node if node is not None else REps()

    def parse_rep(self):
        node = self.parse_atom()
        while self.peek() == '*':
            op = self.next()
            node = RStar(node)
        return node

    def parse_atom(self):
        c = self.peek()
        if c == '(':
            self.next()
            node = self.parse_alt()
            self.expect(')')
            return node
        if c == '[':
            return self.parse_class()
        if c == '\\':
            self.next()
            nxt = self.next()
            return RChar(self._unescape(nxt))
        if c == '.':
            self.next()
            return RAny()
        if c in ('|', '*', '+', '?'):
            raise SyntaxError(f"unexpected {c!r} at {self.i}")
        self.next()
        return RChar(c)

    def parse_class(self):
        self.expect('[')
        negate = False
        if self.peek() == '^':
            self.next(); negate = True
        chars = set()
        while self.peek() != ']':
            if self.peek() is None:
                raise SyntaxError("unterminated [")
            c = self.next()
            if c == '\\':
                c = self._unescape(self.next())
            if self.peek() == '-' and self.i + 1 < len(self.s) and self.s[self.i+1] != ']':
                self.next()
                hi = self.next()
                if hi == '\\': hi = self._unescape(self.next())
                for code in range(ord(c), ord(hi) + 1):
                    chars.add(chr(code))
            else:
                chars.add(c)
        self.expect(']')
        if negate:
            universe = set(chr(i) for i in range(32, 127))
            chars = universe - chars
        return RClass(chars)

EPS = None

class NFA:
    def __init__(self):
        self.trans = defaultdict(list)
        self.accept = {}
        self.next_id = 0

    def new_state(self):
        s = self.next_id
        self.next_id += 1
        return s

    def add(self, src, ch, dst):
        self.trans[src].append((ch, dst))


def build_nfa(regex_ast, out: NFA):
    def rec(node):
        if isinstance(node, REps):
            s = out.new_state(); e = out.new_state()
            out.add(s, EPS, e)
            return s, e
        if isinstance(node, RChar):
            s = out.new_state(); e = out.new_state()
            out.add(s, node.c, e)
            return s, e
        if isinstance(node, RClass):
            s = out.new_state(); e = out.new_state()
            for c in node.chars:
                out.add(s, c, e)
            return s, e
        if isinstance(node, RAny):
            s = out.new_state(); e = out.new_state()
            for code in range(32, 127):
                out.add(s, chr(code), e)
            return s, e
        if isinstance(node, RCat):
            a_s, a_e = rec(node.a)
            b_s, b_e = rec(node.b)
            out.add(a_e, EPS, b_s)
            return a_s, b_e
        if isinstance(node, RAlt):
            s = out.new_state(); e = out.new_state()
            a_s, a_e = rec(node.a)
            b_s, b_e = rec(node.b)
            out.add(s, EPS, a_s); out.add(s, EPS, b_s)
            out.add(a_e, EPS, e); out.add(b_e, EPS, e)
            return s, e
        if isinstance(node, RStar):
            s = out.new_state(); e = out.new_state()
            a_s, a_e = rec(node.a)
            out.add(s, EPS, a_s); out.add(s, EPS, e)
            out.add(a_e, EPS, a_s); out.add(a_e, EPS, e)
            return s, e
        raise ValueError(f"unknown AST node {node!r}")

    return rec(regex_ast)

def eps_closure(nfa: NFA, states):
    stack = list(states)
    seen = set(states)
    while stack:
        s = stack.pop()
        for ch, t in nfa.trans.get(s, ()):
            if ch is EPS and t not in seen:
                seen.add(t); stack.append(t)
    return seen


def move(nfa: NFA, states, ch):
    out = set()
    for s in states:
        for c, t in nfa.trans.get(s, ()):
            if c is not EPS and c == ch:
                out.add(t)
    return out


def build_dfa(nfa: NFA, start_nfa_state):
    start_set = frozenset(eps_closure(nfa, {start_nfa_state}))
    state_to_idx = {start_set: 0}
    dfa_states = [start_set]
    dfa_trans = {}
    dfa_accept = {}

    def accept_of(set_of_nfa_states):
        best = None
        for s in set_of_nfa_states:
            if s in nfa.accept:
                tok, prio = nfa.accept[s]
                if best is None or prio < best[1]:
                    best = (tok, prio)
        return best[0] if best else None

    queue = deque([0])
    while queue:
        idx = queue.popleft()
        cur = dfa_states[idx]
        tok = accept_of(cur)
        if tok is not None:
            dfa_accept[idx] = tok

        chars = set()
        for s in cur:
            for c, t in nfa.trans.get(s, ()):
                if c is not EPS:
                    chars.add(c)

        for c in chars: 
            nxt = frozenset(eps_closure(nfa, move(nfa, cur, c)))
            if not nxt:
                continue
            if nxt not in state_to_idx:
                state_to_idx[nxt] = len(dfa_states)
                dfa_states.append(nxt)
                queue.append(state_to_idx[nxt])
            dfa_trans[(idx, c)] = state_to_idx[nxt]

    return dfa_states, dfa_trans, dfa_accept, 0

def minimize_dfa(dfa_states, dfa_trans, dfa_accept, start_idx):
    n = len(dfa_states)
    alphabet = sorted({c for (_, c) in dfa_trans.keys()})

    trap_old = n
    Q = set(range(n + 1))

    def delta(state, ch):
        return dfa_trans.get((state, ch), trap_old)

    groups = defaultdict(set)
    for q in Q:
        groups[dfa_accept.get(q, None)].add(q)

    P = [set(block) for block in groups.values()]
    W = deque(P)
    while W:
        A = W.popleft()

        for c in alphabet:
            X = {q for q in Q if delta(q, c) in A}
            if not X:
                continue

            new_P = []
            changed = False

            for Y in P:
                Y1 = Y & X
                Y2 = Y - X

                if not Y1 or not Y2:
                    new_P.append(Y)
                    continue

                changed = True
                new_P.append(Y1)
                new_P.append(Y2)

                replaced_in_worklist = False
                for k, B in enumerate(W):
                    if B == Y:
                        remaining = list(W)
                        remaining.pop(k)
                        W = deque(remaining)
                        W.append(Y1)
                        W.append(Y2)
                        replaced_in_worklist = True
                        break

                if not replaced_in_worklist:
                    W.append(Y1 if len(Y1) <= len(Y2) else Y2)

            if changed:
                P = new_P

    state_to_block = {
        q: bid
        for bid, block in enumerate(P)
        for q in block
    }

    new_states = list(range(len(P)))
    new_trans = {}
    new_accept = {}

    for bid, block in enumerate(P):
        rep = next(iter(block))

        tok = dfa_accept.get(rep)
        if tok is not None:
            new_accept[bid] = tok

        for c in alphabet:
            target = state_to_block[delta(rep, c)]
            new_trans[(bid, c)] = target

    new_start = state_to_block[start_idx]
    new_trap = state_to_block[trap_old]

    return new_states, new_trans, new_accept, new_start, new_trap

def emit_json(dfa_states, dfa_trans, dfa_accept, start_idx, trap, alphabet):
    out = {
        "alphabet": alphabet,
        "start": start_idx,
        "trap": trap,
        "accept": {str(k): v for k, v in dfa_accept.items()},
        "transitions": [
            {"from": s, "char": c, "to": dfa_trans[(s, c)]}
            for (s, c) in sorted(dfa_trans.keys())
        ],
    }
    return json.dumps(out, indent=2, ensure_ascii=False)

TOKENS = [
    ("COMMENT",  r"//[^\n\r]*"),
    ("WS",       r"[ \t\n\r]|[ \t\n\r][ \t\n\r]*"),
    ("IF",       r"if"),
    ("ELSE",     r"else"),
    ("WHILE",    r"while"),
    ("RETURNS",  r"returns"),
    ("REQUIRES", r"requires"),
    ("ENSURES",  r"ensures"),
    ("USES",     r"uses"),
    ("INVARIANT",r"invariant"),
    ("FORALL",   r"forall"),
    ("EXISTS",   r"exists"),
    ("TRUE",     r"true"),
    ("FALSE",    r"false"),
    ("NOT",      r"not"),
    ("AND",      r"and"),
    ("OR",       r"or"),
    ("ARROW",    r"→"),
    ("LENGTH",   r"length"),
    ("ID",       r"[a-zA-Z_][a-zA-Z0-9]*"),
    ("INT",      r"0|[1-9][0-9]*"),
    ("EQ",       r"=="),
    ("NEQ",      r"!="),
    ("GE",       r">="),
    ("LE",       r"<="),
    ("ASSIGN",   r"="),
    ("GT",       r">"),
    ("LT",       r"<"),
    ("PLUS",     r"\+"),
    ("MINUS",    r"-"),
    ("STAR",     r"\*"),
    ("SLASH",    r"/"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("LBRACK",   r"\["),
    ("RBRACK",   r"\]"),
    ("SEMI",     r";"),
    ("COMMA",    r","),
    ("COLON",    r":"),
    ("FUNCTION", r"function"),
    ("ASSERT",   r"assert"),
    ("ASSUME",   r"assume"),
]

def compile_tokens(tokens):
    nfa = NFA()
    global_start = nfa.new_state()

    for prio, (name, pattern) in enumerate(tokens):
        ast = RegexParser(pattern).parse()
        s, e = build_nfa(ast, nfa)
        nfa.add(global_start, EPS, s)
        nfa.accept[e] = (name, prio)

    print(f"[NFA] states = {nfa.next_id}, "
          f"transitions = {sum(len(v) for v in nfa.trans.values())}")

    dfa_states, dfa_trans, dfa_accept, start = build_dfa(nfa, global_start)
    print(f"[DFA] states = {len(dfa_states)}, transitions = {len(dfa_trans)}")

    min_states, min_trans, min_accept, min_start, trap = minimize_dfa(
        dfa_states, dfa_trans, dfa_accept, start
    )
    print(f"[MIN] states = {len(min_states)}, "
          f"transitions = {len(min_trans)}, trap = {trap}")

    alphabet = sorted({c for (_, c) in min_trans.keys()})
    print(f"[ALPHABET] {len(alphabet)} symbols")

    return min_states, min_trans, min_accept, min_start, trap, alphabet


def tokenize(text, trans, accept, start, trap, skip=("WS", "COMMENT")):
    out = []
    i, n = 0, len(text)
    while i < n:
        state = start
        last_ok, last_tok = -1, None
        j = i
        while j < n:
            nxt = trans.get((state, text[j]), trap)
            if nxt == trap:
                break
            state = nxt
            j += 1
            if state in accept:
                last_ok, last_tok = j, accept[state]
        if last_tok is None:
            raise ValueError(f"unexpected char {text[i]!r} at position {i}")
        if last_tok == 'INT' and last_ok < n and text[last_ok].isdigit():
            raise ValueError(f"invalid integer literal at position {i}")
        if last_tok not in skip:
            out.append((last_tok, text[i:last_ok], i))
        i = last_ok
    return out

def main():
    tokens = TOKENS
    states, trans, accept, start, trap, alphabet = compile_tokens(tokens)
    json_str = emit_json(states, trans, accept, start, trap, alphabet)
    with open("dfa.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n[OK] wrote dfa.json ({len(json_str)} bytes)")

    print("\n[TESTS]")
    for src in ["x = 1; //ERFHIU\n y = 5;", "if (x >= 1) y = x - 1;", "", "          ", "if", "\n\r\n\r\t", 
                "abs(x:int) returns r:int {\n if (x < 0) r = -x; else r = x;\n }", "0", "25", "00", "01", 
                "assert assume function x y z", "while else if invariant length", "+ - * / == != <= >= < > =", 
                "君のことµµµµµについて話す", "//µµµ君のことについて話す"]:
        try:
            print(f"  {src!r} ->", tokenize(src, trans, accept, start, trap))
        except ValueError:
            print(f"  {src!r} -> ERROR")

if __name__ == "__main__":
    main()