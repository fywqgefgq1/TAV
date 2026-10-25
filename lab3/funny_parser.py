import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funny_dfa_maker import TOKENS, compile_tokens, tokenize

_old_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    STATES, TRANS, ACCEPT, START, TRAP, ALPHABET = compile_tokens(TOKENS)
finally:
    sys.stdout = _old_stdout

class ASTNode:
    def to_dict(self):
        return {k: _serialize(v) for k, v in self.__dict__.items()}

def _serialize(obj):
    if isinstance(obj, ASTNode):
        return {"_type": obj.__class__.__name__, **obj.to_dict()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, tuple):
        return [_serialize(i) for i in obj]
    return obj

class Program(ASTNode):
    def __init__(self, functions, formulas):
        self.functions = functions
        self.formulas = formulas

class Function(ASTNode):
    def __init__(self, name, params, returns, requires, ensures, uses, body, pos):
        self.name = name
        self.params = params
        self.returns = returns
        self.requires = requires
        self.ensures = ensures
        self.uses = uses
        self.body = body
        self.pos = pos

class Formula(ASTNode):
    def __init__(self, name, params, predicate, pos):
        self.name = name
        self.params = params
        self.predicate = predicate
        self.pos = pos

class Block(ASTNode):
    def __init__(self, stmts, pos):
        self.stmts = stmts
        self.pos = pos

class Assign(ASTNode):
    def __init__(self, target, value, pos):
        self.target = target
        self.value = value
        self.pos = pos

class TupleAssign(ASTNode):
    def __init__(self, targets, call, pos):
        self.targets = targets
        self.call = call
        self.pos = pos

class IfStmt(ASTNode):
    def __init__(self, cond, then_stmt, else_stmt, pos):
        self.cond = cond
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt
        self.pos = pos

class WhileStmt(ASTNode):
    def __init__(self, cond, invariant, body, pos):
        self.cond = cond
        self.invariant = invariant
        self.body = body
        self.pos = pos

class AssertStmt(ASTNode):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class AssumeStmt(ASTNode):
    def __init__(self, expr, pos):
        self.expr = expr
        self.pos = pos

class ErrorStmt(ASTNode):
    def __init__(self, msg, pos):
        self.msg = msg
        self.pos = pos

class BinOp(ASTNode):
    def __init__(self, op, left, right, pos):
        self.op = op
        self.left = left
        self.right = right
        self.pos = pos

class UnaryOp(ASTNode):
    def __init__(self, op, operand, pos):
        self.op = op
        self.operand = operand
        self.pos = pos

class IntLit(ASTNode):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class BoolLit(ASTNode):
    def __init__(self, value, pos):
        self.value = value
        self.pos = pos

class Var(ASTNode):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

class ArrayAccess(ASTNode):
    def __init__(self, array, index, pos):
        self.array = array
        self.index = index
        self.pos = pos

class FuncCall(ASTNode):
    def __init__(self, name, args, pos):
        self.name = name
        self.args = args
        self.pos = pos

class LengthExpr(ASTNode):
    def __init__(self, arg, pos):
        self.arg = arg
        self.pos = pos

class Quantifier(ASTNode):
    def __init__(self, kind, var_name, var_type, predicate, pos):
        self.kind = kind
        self.var_name = var_name
        self.var_type = var_type
        self.predicate = predicate
        self.pos = pos

class FormulaRef(ASTNode):
    def __init__(self, name, args, pos):
        self.name = name
        self.args = args
        self.pos = pos

class ParseError(Exception):
    def __init__(self, msg, pos):
        super().__init__(f"{msg} at position {pos}")
        self.pos = pos

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errors = []

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', '', -1)

    def advance(self):
        tok = self.peek()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def expect(self, ttype):
        tok = self.peek()
        if tok[0] != ttype:
            raise ParseError(f"Expected {ttype}, got {tok[0]} ('{tok[1]}')", tok[2])
        return self.advance()

    def match(self, ttype):
        if self.peek()[0] == ttype:
            return self.advance()
        return None

    def expect_keyword(self, keyword):
        tok = self.peek()
        if tok[0] == 'ID' and tok[1] == keyword:
            return self.advance()
        if tok[0] == keyword.upper():
            return self.advance()
        raise ParseError(f"Expected '{keyword}', got {tok[0]} ('{tok[1]}')", tok[2])

    def match_keyword(self, keyword):
        tok = self.peek()
        if tok[0] == 'ID' and tok[1] == keyword:
            return self.advance()
        if tok[0] == keyword.upper():
            return self.advance()
        return None

    def error_recovery(self):
        if self.peek()[0] != 'EOF':
            self.advance()
            
        sync = {'SEMI', 'RBRACE', 'EOF', 'IF', 'WHILE', 
                'RETURNS', 'REQUIRES', 'ENSURES', 'USES', 'ID'}
        while self.peek()[0] not in sync:
            self.advance()
            
        if self.peek()[0] == 'SEMI':
            self.advance()

    def parse_program(self):
        if self.peek()[0] == 'EOF':
            raise ParseError("Empty input: expected at least one function or formula definition", 0)
            
        functions = []
        formulas = []
        while self.peek()[0] != 'EOF':
            try:
                tok = self.peek()
                if tok[0] == 'ID' and tok[1] == 'function':
                    functions.append(self.parse_function())
                elif tok[0] == 'ID':
                    self.parse_top_level_def(functions, formulas)
                else:
                    raise ParseError(f"Expected 'function' or formula definition, got {tok[0]}", tok[2])
            except ParseError as e:
                self.errors.append(str(e))
                print(f"Parse Error: {e}", file=sys.stderr)
                self.error_recovery()
        return Program(functions, formulas)

    def parse_function(self):
        pos = self.expect_keyword('function')[2]
        name = self.expect('ID')[1]
        self.expect('LPAREN')
        params = self._parse_var_defs()
        self.expect('RPAREN')

        requires = None
        if self.match('REQUIRES'):
            requires = self.parse_predicate()

        self.expect('RETURNS')
        returns = self._parse_var_defs()

        if requires is None and self.match('REQUIRES'):
            requires = self.parse_predicate()

        ensures = None
        if self.match('ENSURES'):
            ensures = self.parse_predicate()
            
        uses = []
        if self.match('USES'):
            uses = self._parse_var_defs()
            
        body = self.parse_statement()
        return Function(name, params, returns, requires, ensures, uses, body, pos)

    def parse_top_level_def(self, functions, formulas):
        pos = self.peek()[2]
        name = self.expect('ID')[1]
        self.expect('LPAREN')
        params = self._parse_var_defs()
        self.expect('RPAREN')

        if self.peek()[0] == 'ARROW':
            self.advance()
            pred = self.parse_predicate()
            self.expect('SEMI')
            formulas.append(Formula(name, params, pred, pos))
        else:
            requires = None
            if self.match('REQUIRES'):
                requires = self.parse_predicate()
            self.expect('RETURNS')
            returns = self._parse_var_defs()
            if requires is None and self.match('REQUIRES'):
                requires = self.parse_predicate()
            ensures = None
            if self.match('ENSURES'):
                ensures = self.parse_predicate()
            uses = []
            if self.match('USES'):
                uses = self._parse_var_defs()
            body = self.parse_statement()
            functions.append(Function(name, params, returns, requires, ensures, uses, body, pos))

    def _parse_var_defs(self):
        defs = []
        if self.peek()[0] in ('RPAREN', 'RETURNS', 'ENSURES', 'USES', 'ARROW'):
            return defs
        defs.append(self._parse_one_var_def())
        while self.match('COMMA'):
            defs.append(self._parse_one_var_def())
        return defs

    def _parse_one_var_def(self):
        name = self.expect('ID')[1]
        self.expect('COLON')
        ttype = self.expect('ID')[1]
        if self.peek()[0] == 'LBRACK':
            self.expect('LBRACK')
            self.expect('RBRACK')
            ttype = "int[]"
        return (name, ttype)

    def parse_statement(self):
        tok = self.peek()
        try:
            if tok[0] == 'LBRACE':
                return self.parse_block()
            if tok[0] == 'IF':
                return self.parse_if()
            if tok[0] == 'WHILE':
                return self.parse_while()
            if tok[0] == 'ID' and tok[1] == 'assert':
                return self.parse_assert()
            if tok[0] == 'ID' and tok[1] == 'assume':
                return self.parse_assume()
            if tok[0] == 'ID':
                return self.parse_assignment_or_error()
            raise ParseError(f"Unexpected token {tok[0]} ('{tok[1]}')", tok[2])
        except ParseError as e:
            self.errors.append(str(e))
            print(f"Parse Error: {e}", file=sys.stderr)
            self.error_recovery()
            return ErrorStmt(str(e), tok[2])

    def parse_block(self):
        pos = self.expect('LBRACE')[2]
        stmts = []
        while self.peek()[0] not in ('RBRACE', 'EOF'):
            stmts.append(self.parse_statement())
        self.expect('RBRACE')
        return Block(stmts, pos)

    def parse_if(self):
        pos = self.expect('IF')[2]
        self.expect('LPAREN')
        cond = self.parse_condition()
        self.expect('RPAREN')
        then_stmt = self.parse_statement()
        else_stmt = None
        if self.match('ELSE'):
            else_stmt = self.parse_statement()
        return IfStmt(cond, then_stmt, else_stmt, pos)

    def parse_while(self):
        pos = self.expect('WHILE')[2]
        self.expect('LPAREN')
        cond = self.parse_condition()
        self.expect('RPAREN')
        invariant = None
        if self.match('INVARIANT'):
            invariant = self.parse_predicate()
        body = self.parse_statement()
        return WhileStmt(cond, invariant, body, pos)

    def parse_assert(self):
        pos = self.expect_keyword('assert')[2]
        expr = self.parse_predicate()
        self.expect('SEMI')
        return AssertStmt(expr, pos)

    def parse_assume(self):
        pos = self.expect_keyword('assume')[2]
        expr = self.parse_predicate()
        self.expect('SEMI')
        return AssumeStmt(expr, pos)

    def parse_assignment_or_error(self):
        pos = self.peek()[2]
        first_name = self.expect('ID')[1]
        
        if self.peek()[0] == 'COMMA':
            targets = [Var(first_name, pos)]
            while self.match('COMMA'):
                npos = self.peek()[2]
                targets.append(Var(self.expect('ID')[1], npos))
            self.expect('ASSIGN')
            call = self._parse_func_call()
            self.expect('SEMI')
            return TupleAssign(targets, call, pos)
            
        if self.peek()[0] == 'LBRACK':
            self.advance()
            idx = self.parse_expr()
            self.expect('RBRACK')
            target = ArrayAccess(Var(first_name, pos), idx, pos)
            self.expect('ASSIGN')
            value = self.parse_expr()
            self.expect('SEMI')
            return Assign(target, value, pos)
            
        self.expect('ASSIGN')
        value = self.parse_expr()
        self.expect('SEMI')
        return Assign(Var(first_name, pos), value, pos)

    def parse_condition(self):
        left = self._parse_or()
        if self.match('ARROW'):
            right = self.parse_condition()
            return BinOp('→', left, right, left.pos)
        return left

    def _parse_or(self):
        left = self._parse_and()
        while self.peek()[0] == 'OR':
            op_tok = self.advance()
            right = self._parse_and()
            left = BinOp('OR', left, right, op_tok[2])
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self.peek()[0] == 'AND':
            op_tok = self.advance()
            right = self._parse_not()
            left = BinOp('AND', left, right, op_tok[2])
        return left

    def _parse_not(self):
        if self.peek()[0] == 'NOT':
            tok = self.advance()
            operand = self._parse_not()
            return UnaryOp('NOT', operand, tok[2])
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self.parse_expr()
        tok = self.peek()
        if tok[0] in ('EQ', 'NEQ', 'GE', 'LE', 'GT', 'LT'):
            self.advance()
            right = self.parse_expr()
            return BinOp(tok[0], left, right, tok[2])
        return left

    def parse_predicate(self):
        left = self._parse_pred_or()
        if self.match('ARROW'):
            right = self.parse_predicate()
            return BinOp('→', left, right, left.pos)
        return left

    def _parse_pred_or(self):
        left = self._parse_pred_and()
        while self.peek()[0] == 'OR':
            op_tok = self.advance()
            right = self._parse_pred_and()
            left = BinOp('OR', left, right, op_tok[2])
        return left

    def _parse_pred_and(self):
        left = self._parse_pred_not()
        while self.peek()[0] == 'AND':
            op_tok = self.advance()
            right = self._parse_pred_not()
            left = BinOp('AND', left, right, op_tok[2])
        return left

    def _parse_pred_not(self):
        if self.peek()[0] == 'NOT':
            tok = self.advance()
            operand = self._parse_pred_not()
            return UnaryOp('NOT', operand, tok[2])
        return self._parse_pred_atom()

    def _parse_pred_atom(self):
        tok = self.peek()
        if tok[0] == 'TRUE':
            self.advance()
            return BoolLit(True, tok[2])
        if tok[0] == 'FALSE':
            self.advance()
            return BoolLit(False, tok[2])
        if tok[0] == 'FORALL' or tok[0] == 'EXISTS':
            return self._parse_quantifier()
        if tok[0] == 'LPAREN':
            self.advance()
            p = self.parse_predicate()
            self.expect('RPAREN')
            return p
            
        if tok[0] == 'ID':
            saved = self.pos
            name = self.advance()[1]
            if self.peek()[0] == 'LPAREN' and name != 'length':
                self.advance()
                args = self._parse_expr_list()
                self.expect('RPAREN')
                return FormulaRef(name, args, tok[2])
            else:
                self.pos = saved
                return self._parse_comparison_in_predicate()
                
        return self._parse_comparison_in_predicate()

    def _parse_comparison_in_predicate(self):
        left = self.parse_expr()
        tok = self.peek()
        if tok[0] in ('EQ', 'NEQ', 'GE', 'LE', 'GT', 'LT'):
            self.advance()
            right = self.parse_expr()
            return BinOp(tok[0], left, right, tok[2])
        return left

    def _parse_quantifier(self):
        tok = self.advance()
        kind = tok[1]
        self.expect('LPAREN')
        var_name = self.expect('ID')[1]
        self.expect('COLON')
        var_type = self.expect('ID')[1]
        if self.peek()[0] == 'LBRACK':
            self.expect('LBRACK')
            self.expect('RBRACK')
            var_type = "int[]"
            
        pred = self.parse_predicate()
        self.expect('RPAREN')
        return Quantifier(kind, var_name, var_type, pred, tok[2])

    def parse_expr(self):
        return self._parse_additive()

    def _parse_additive(self):
        left = self._parse_multiplicative()
        while self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.advance()
            right = self._parse_multiplicative()
            left = BinOp(op[0], left, right, op[2])
        return left

    def _parse_multiplicative(self):
        left = self._parse_unary()
        while self.peek()[0] in ('STAR', 'SLASH'):
            op = self.advance()
            right = self._parse_unary()
            left = BinOp(op[0], left, right, op[2])
        return left

    def _parse_unary(self):
        if self.peek()[0] == 'MINUS':
            tok = self.advance()
            operand = self._parse_unary()
            return UnaryOp('MINUS', operand, tok[2])
        return self._parse_postfix()

    def _parse_postfix(self):
        node = self._parse_primary()
        while True:
            if self.peek()[0] == 'LBRACK':
                self.advance()
                idx = self.parse_expr()
                self.expect('RBRACK')
                node = ArrayAccess(node, idx, node.pos)
            elif self.peek()[0] == 'LPAREN' and isinstance(node, Var):
                self.advance()
                args = self._parse_expr_list()
                self.expect('RPAREN')
                node = FuncCall(node.name, args, node.pos)
            else:
                break
        return node

    def _parse_primary(self):
        tok = self.peek()
        if tok[0] == 'INT':
            self.advance()
            return IntLit(int(tok[1]), tok[2])
        if tok[0] == 'TRUE':
            self.advance()
            return BoolLit(True, tok[2])
        if tok[0] == 'FALSE':
            self.advance()
            return BoolLit(False, tok[2])
        if tok[0] == 'LENGTH':
            self.advance()
            self.expect('LPAREN')
            arg = self.parse_expr()
            self.expect('RPAREN')
            return LengthExpr(arg, tok[2])
        if tok[0] == 'ID':
            self.advance()
            return Var(tok[1], tok[2])
        if tok[0] == 'LPAREN':
            self.advance()
            e = self.parse_expr()
            self.expect('RPAREN')
            return e
        raise ParseError(f"Unexpected token in expression: {tok[0]} ('{tok[1]}')", tok[2])

    def _parse_func_call(self):
        pos = self.peek()[2]
        name = self.expect('ID')[1]
        self.expect('LPAREN')
        args = self._parse_expr_list()
        self.expect('RPAREN')
        return FuncCall(name, args, pos)

    def _parse_expr_list(self):
        exprs = []
        if self.peek()[0] == 'RPAREN':
            return exprs
        exprs.append(self.parse_expr())
        while self.match('COMMA'):
            exprs.append(self.parse_expr())
        return exprs

def main():      
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokens = tokenize(text, TRANS, ACCEPT, START, TRAP)
    parser = Parser(tokens)
    ast = parser.parse_program()
    
    result = {
        "ast": _serialize(ast),
        "errors": parser.errors
    }
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    with open("ast.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    
    if parser.errors:
        sys.exit(1)

if __name__ == "__main__":
    main()