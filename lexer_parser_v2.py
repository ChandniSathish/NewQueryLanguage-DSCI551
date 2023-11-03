import ply.lex as lex
import ply.yacc as yacc

# Define the list of token names
tokens = (
    'EXTRACT',
    'DOLLAR',
    'USING',
    'WHEN',
    'CATEGORIZE',
    'BY',
    'RANK',
    'ASC',
    'DESC',
    'IDENTIFIER',
    'STRING',
    'COMMA',
    'EQUALS',
    'LIKE',
    'BOUND',
    'NUMBER',
)

# Define regular expressions for simple tokens
t_EXTRACT = r'EXTRACT'
t_DOLLAR = r'\$'
t_USING = r'USING'
t_WHEN = r'WHEN'
t_CATEGORIZE = r'CATEGORIZE'
t_BY = r'BY'
t_RANK = r'RANK'
t_ASC = r'ASC'
t_DESC = r'DESC'
t_COMMA = r','
t_EQUALS = r'='
t_LIKE = r'LIKE'
t_BOUND = r'BOUND'
t_NUMBER = r'\d+'

t_ignore = ' \t'

# A regular expression for identifiers (table and column names)
def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

# A regular expression for string literals
def t_STRING(t):
    r"'[^']*'"
    t.value = t.value[1:-1]
    return t

# Define reserved words
reserved = {
    'EXTRACT': 'EXTRACT',
    'DOLLAR': 'DOLLAR',
    'USING': 'USING',
    'WHEN': 'WHEN',
    'CATEGORIZE': 'CATEGORIZE',
    'BY': 'BY',
    'LIKE': 'LIKE',
    'BOUND': 'BOUND',
    'RANK': 'RANK',
    'ASC': 'ASC',
    'DESC': 'DESC',
}

# Error handling for unknown characters
def t_error(t):
    print(f"Lexer Error: Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Parsing rules
def p_query(p):
    '''query : EXTRACT select_list USING table_list maybe_when maybe_categorize_by maybe_like maybe_bound maybe_rank
    '''
    p[0] = {
        'extract': p[2],
        'using': p[4],
        'when': p[5] if len(p) > 5 and p[5] else None,
        'categorize_by': p[6] if len(p) > 6 and p[6] else None,
        'like': p[7] if len(p) > 7 and p[7] else None,
        'bound': p[8] if len(p) > 8 and p[8] else None,
        'rank_by': p[9] if len(p) > 9 and p[9] else None,
    }

def p_select_list(p):
    '''select_list : DOLLAR
                   | select_list COMMA DOLLAR
                   | IDENTIFIER
                   | select_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_table_list(p):
    '''table_list : IDENTIFIER'''
    p[0] = p[1]

def p_maybe_when(p):
    '''maybe_when : WHEN condition
                  |
    '''
    p[0] = p[2] if len(p) > 1 else None

def p_maybe_categorize_by(p):
    '''maybe_categorize_by : CATEGORIZE BY column_list
                          |
    '''
    p[0] = p[3] if len(p) > 1 else None

def p_maybe_like(p):
    '''maybe_like : LIKE STRING
                  |
    '''
    if len(p) > 1:
        p[0] = {
            'operator': 'LIKE',
            'pattern': p[2],
        }
    else:
        p[0] = None

def p_maybe_bound(p):
    '''maybe_bound : BOUND NUMBER
                  |
    '''
    if len(p) > 1:
        p[0] = {
            'bound': int(p[2])
        }
    else:
        p[0] = None

def p_maybe_rank(p):
    '''maybe_rank : RANK BY IDENTIFIER order_direction
                 |
    '''
    if len(p) > 1:
        p[0] = {
            'column': p[3],
            'rank': p[4],
        }
    else:
        p[0] = None

def p_column_list(p):
    '''column_list : IDENTIFIER
                   | column_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_order_direction(p):
    '''order_direction : ASC
                       | DESC'''
    p[0] = p[1]

def p_condition(p):
    '''condition : IDENTIFIER EQUALS IDENTIFIER
                 | IDENTIFIER EQUALS STRING
                 | IDENTIFIER LIKE STRING'''
    if len(p) == 4 and p[2] == 'LIKE':
        p[0] = {
            'left': p[1],
            'operator': 'LIKE',
            'right': p[3],
        }
    else:
        p[0] = {
            'left': p[1],
            'operator': '=',
            'right': p[3],
        }

# Error handling for syntax errors
def p_error(p):
    print(f"Parser Error: Syntax error in input: {p}")

# Build the parser
parser = yacc.yacc()

# Helper function to parse a query
def parse_query(query):
    result = parser.parse(query)
    return result

# Main program
if __name__ == "__main__":
    while True:
        try:
            # Get user input
            query = input("Enter a basic SQL EXTRACT query (or 'exit' to quit): ")

            if query.lower() == 'exit':
                break

            # Parse the query
            parsed_query = parse_query(query)
            print("Parsed Query:")
            print(parsed_query)
        except Exception as e:
            print(f"An error occurred: {e}")
