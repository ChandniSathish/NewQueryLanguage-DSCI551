import ply.lex as lex
import ply.yacc as yacc
import pandas as pd
import numpy as np


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
    '''query : EXTRACT select_list USING table_list when_clause categorize_by_clause like_clause limit_clause order_clause
             | EXTRACT select_list USING table_list when_clause categorize_by_clause limit_clause order_clause
             | EXTRACT select_list USING table_list when_clause like_clause limit_clause order_clause
             | EXTRACT select_list USING table_list when_clause limit_clause order_clause
             | EXTRACT select_list USING table_list when_clause categorize_by_clause order_clause
             | EXTRACT select_list USING table_list when_clause order_clause
             | EXTRACT select_list USING table_list limit_clause order_clause
             | EXTRACT select_list USING table_list order_clause
             | EXTRACT select_list USING table_list when_clause categorize_by_clause like_clause
             | EXTRACT select_list USING table_list when_clause like_clause
             | EXTRACT select_list USING table_list when_clause categorize_by_clause
             | EXTRACT select_list USING table_list when_clause
             | EXTRACT select_list USING table_list like_clause
             | EXTRACT select_list USING table_list categorize_by_clause
             | EXTRACT select_list USING table_list limit_clause
             | EXTRACT select_list USING table_list'''
    p[0] = {
        'extract': p[2],
        'using': p[4],
        'when': p[5],
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

def p_when_clause(p):
    '''when_clause : WHEN condition'''
    p[0] = p[2]

def p_categorize_by_clause(p):
    '''categorize_by_clause : CATEGORIZE BY column_list'''
    p[0] = p[3]

def p_column_list(p):
    '''column_list : IDENTIFIER
                   | column_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_like_clause(p):
    '''like_clause : LIKE STRING'''
    p[0] = {
        'operator': 'LIKE',
        'pattern': p[2],
    }

def p_limit_clause(p):
    '''limit_clause : BOUND NUMBER'''
    p[0] = {
        'bound': int(p[2])  # Convert the number to an integer
    }

def p_order_clause(p):
    '''order_clause : RANK BY IDENTIFIER order_direction'''
    p[0] = {
        'column': p[3],
        'rank': p[4],
    }

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

def parsed_query_mapping(parsed_query):
    print("yooooooooooo")
    data = {
        'table1': {
            'column1': [10, 20, 30, 40, 60],
            'column2': [10, 20, 30, 40, 50],
        }
    }

    # Parsed query
    parsed_query = {
        'extract': ['column1', 'column2'],
        'using': 'table1',
        'when': {'left': 'column1', 'operator': '>=', 'right': 'column2'},
        'categorize_by': ['column1'],
        'like': None,
        'bound': None
    }

    # Initialize an empty result dictionary
    result = {}
    operators = {
        '=': lambda x, y: x == y,
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
    }

    # Extract the selected columns
    selected_columns = parsed_query['extract']

    # Get the table name from the query
    table_name = parsed_query['using']

    # Filter the dataset based on the 'when' condition
    left_column = parsed_query['when']['left']
    operator = parsed_query['when']['operator']
    right_column = parsed_query['when']['right']

    data_table = data[table_name]
    filtered_indices = [i for i in range(len(data_table[left_column])) if operators[operator](data_table[left_column][i], data_table[right_column][i])]
    print("filteredddd",filtered_indices)

    # Apply categorization
    categorized_by = parsed_query['categorize_by']
    if categorized_by:
        categories = set(data_table[categorized_by[0]])
        categorized_data = {}
        for category in categories:
            category_data = {}
            for col in selected_columns:
                category_data[col] = [data_table[col][i] for i in range(len(data_table[col])) if i in filtered_indices and data_table[categorized_by[0]][i] == category]
            categorized_data[category] = category_data
        result[table_name] = categorized_data
    else:
        result[table_name] = {col: [data_table[col][i] for i in filtered_indices] for col in selected_columns}

    # Print the result
    print(result)
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
            print("immaa cryy")
            result = parsed_query_mapping(parsed_query)
            print(result)

        except Exception as e:
            print(f"An error occurred: {e}")

