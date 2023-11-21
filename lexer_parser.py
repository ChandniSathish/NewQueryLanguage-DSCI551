import json
import ply.lex as lex
import ply.yacc as yacc
import sys
import pandas as pd
import numpy as np
import os
import re

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
    'GT',
    'LT',
    'GE',
    'LE',
    'PROJECT',
    'AVERAGE',
    'MIN',
    'MAX',
    'SUM',
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
t_PROJECT = r'PROJECT'
t_AVERAGE = r'AVERAGE'
t_MIN = r'MIN'
t_MAX = r'MAX'
t_SUM = r'SUM'
t_NUMBER = r'\d+(\.\d+)?'  # Allow integer or floating-point numbers
t_GT = r'>'
t_LT = r'<'
t_GE = r'>='
t_LE = r'<='

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
    'PROJECT': 'PROJECT',
    'AVERAGE': 'AVERAGE',
    'MIN': 'MIN',
    'MAX': 'MAX',
    'SUM': 'SUM',
}

# Error handling for unknown characters
def t_error(t):
    print(f"Lexer Error: Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Parsing rules
def p_query(p):
    '''query : EXTRACT select_list USING table_list maybe_when maybe_categorize_by maybe_like maybe_bound maybe_rank maybe_project maybe_average maybe_min maybe_max maybe_sum
    '''
    p[0] = {
        'extract': p[2],
        'using': p[4],
        'when': p[5] if len(p) > 5 and p[5] else None,
        'categorize_by': p[6] if len(p) > 6 and p[6] else None,
        'like': p[7] if len(p) > 7 and p[7] else None,
        'bound': p[8] if len(p) > 8 and p[8] else None,
        'rank_by': p[9] if len(p) > 9 and p[9] else None,
        'project_fields': p[10] if len(p) > 10 and p[10] else None,
        'average_field': p[11] if len(p) > 11 and p[11] else None,
        'min': p[12] if len(p) > 12 and p[12] else None,
        'max': p[13] if len(p) > 13 and p[13] else None,
        'sum_by': p[14] if len(p) > 14 and p[14] else None,
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
    p[0] = {'columns': p[3]} if len(p) > 1 else None

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

def p_maybe_average(p):
    '''maybe_average : AVERAGE BY IDENTIFIER
                   |
    '''
    if len(p) > 1:
        p[0] = {
            'field': p[3],
        }
    else:
        p[0] = None

def p_maybe_project(p):
    '''maybe_project : PROJECT project_list
                  |
    '''
    p[0] = p[2] if len(p) > 1 else None

def p_project_list(p):
    '''project_list : IDENTIFIER
                   | project_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_maybe_rank(p):
    '''maybe_rank : RANK BY IDENTIFIER maybe_order_direction
                  |
    '''
    if len(p) > 1:
        p[0] = {
            'column': p[3],
            'order_direction': p[4] if len(p) > 4 else 'ASC',  # Default to ASC
        }
    else:
        p[0] = None

def p_maybe_order_direction(p):
    '''maybe_order_direction : ASC
                             | DESC
                             |
    '''
    p[0] = p[1].upper() if len(p) > 1 else 'ASC'


def p_column_list(p):
    '''column_list : IDENTIFIER
                   | column_list COMMA IDENTIFIER'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[1].append(p[3])
        p[0] = p[1]

def p_condition(p):
    '''condition : IDENTIFIER EQUALS IDENTIFIER
                 | IDENTIFIER EQUALS STRING
                 | IDENTIFIER LIKE STRING
                 | IDENTIFIER GT NUMBER
                 | IDENTIFIER LT NUMBER
                 | IDENTIFIER GE NUMBER
                 | IDENTIFIER LE NUMBER
    '''
    if len(p) == 4 and p[2] == 'LIKE':
        p[0] = {
            'left': p[1],
            'operator': 'LIKE',
            'right': p[3],
        }
    elif len(p) == 4 and p[2] in ['GT', 'LT', 'GE', 'LE']:
        p[0] = {
            'left': p[1],
            'operator': p[2],
            'right': float(p[3]),  # Assuming the comparison involves numbers
        }
    else:
        p[0] = {
            'left': p[1],
            'operator': p[2],
            'right': p[3],
        }


def p_maybe_min(p):
    '''maybe_min : MIN BY IDENTIFIER'''
    p[0] = {
        'function': 'MIN',
        'column': p[3],
    }

def p_maybe_max(p):
    '''maybe_max : MAX BY IDENTIFIER'''
    p[0] = {
        'function': 'MAX',
        'column': p[3],
    }

def p_maybe_sum(p):
    '''maybe_sum : SUM BY IDENTIFIER
                |
    '''
    if len(p) > 1:
        p[0] = {
            'function': 'SUM',
            'column': p[3],
        }
    else:
        p[0] = None

# Error handling for syntax errors
def p_error(p):
    print(f"Parser Error: Syntax error in input: {p}")

# Build the parser
parser = yacc.yacc()

# Helper function to parse a query
def parse_query(query):
    result = parser.parse(query)
    return result

# Load data from the JSON file
def load_data_from_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def sort_data_by_field(data, field, reverse=False):
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get(field, 0), reverse=reverse)
    elif isinstance(data, dict):
        sorted_data = {}
        for category, rows in data.items():
            sorted_data[category] = sorted(rows, key=lambda x: x.get(field, 0), reverse=reverse)
        return sorted_data
    else:
        raise ValueError("Unsupported data type for sorting")

# Function to execute an "EXTRACT" query on the data
def execute_extract_query(data, conditions=None, categorize_by=None, bound=None, rank_by=None,join=None,project_fields=None,average_field=None,min_field=None,max_field=None,sum_by=None):
    result = data
    if conditions:
        conditions = conditions.replace('GT', '>').replace('LT', '<').replace('GE', '>=').replace('LE', '<=').replace('EQ', '=')
        operators = {
            '=': lambda x, y: x == y,
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
        }
        conditions = conditions.split(" ")
        print("conditions",conditions)
        left_column = conditions[0]
        operator = conditions[1]
        right_operand = conditions[2]
        # Check if right_operand is a column name or a constant
        if right_operand in result.columns:
            right_values = result[right_operand]
        else:
            try:
                # Convert to the appropriate type (int, float, etc.)
                right_value = float(right_operand) if '.' in right_operand else int(right_operand)
                right_values = [right_value] * len(result)
            except ValueError:
                raise ValueError("Right operand is neither a valid column name nor a constant.")

        # Apply the condition
        result = result[[operators[operator](left_value, right_value) for left_value, right_value in zip(result[left_column], right_values)]]

    if join:
        # join_condition = {'join_type': type,'left_column': left_column, 'right_column': right_column}
        # join_condition = join.split(" ") 
        join_type = join['join_type']
        left_column = join['left_column']
        right_column = join['right_column']

        # Sort the dataframes
        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_directory, 'iris2.csv')
        result = pd.read_csv(csv_file_path, sep=',')

        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_directory, 'iris1.csv')
        result1 = pd.read_csv(csv_file_path, sep=',')

        result_sorted = result.sort_values(by=left_column)
        other_df_sorted = result1.sort_values(by=right_column)
        
        merged_data = []
        i, j = 0, 0
        if join_type == "INNER":
            # Manual Inner Join
            while i < len(result_sorted) and j < len(other_df_sorted):
                row_left = result_sorted.iloc[i]
                row_right = other_df_sorted.iloc[j]

                if row_left[left_column] == row_right[right_column]:
                    print(row_left,row_right)
                    merged_row = pd.concat([row_left, row_right]).to_dict()
                    print("merged_row",merged_row)
                    merged_data.append(merged_row)
                    i += 1
                    j += 1
                elif row_left[left_column] < row_right[right_column]:
                    i += 1
                else:
                    j += 1

        if join_type == "LEFT":
            while i < len(result_sorted):
                row_left = result_sorted.iloc[i]
                match_found = False

                while j < len(other_df_sorted) and row_left[left_column] >= other_df_sorted.iloc[j][right_column]:
                    row_right = other_df_sorted.iloc[j]
                    if row_left[left_column] == row_right[right_column]:
                        merged_row = pd.concat([row_left, row_right]).to_dict()
                        merged_data.append(merged_row)
                        match_found = True
                        j += 1
                        sys.exit(1)  # sys.exit(1) the inner loop after finding the first match
                    j += 1

                if not match_found:
                    # For no match, include left row with NaNs for right columns
                    merged_row = pd.concat([row_left, pd.Series([pd.NA] * len(other_df_sorted.columns), index=other_df_sorted.columns)]).to_dict()
                    merged_data.append(merged_row)

                i += 1

        if join_type == "RIGHT":
            # Manual Right Outer Join
            merged_data = []
            i, j = 0, 0

            while j < len(other_df_sorted):
                row_right = other_df_sorted.iloc[j]
                match_found = False

                while i < len(result_sorted) and result_sorted.iloc[i][left_column] <= row_right[right_column]:
                    row_left = result_sorted.iloc[i]
                    if row_left[left_column] == row_right[right_column]:
                        merged_row = pd.concat([row_left, row_right]).to_dict()
                        merged_data.append(merged_row)
                        match_found = True
                        i += 1
                        sys.exit(1)  # sys.exit(1) the inner loop after finding the first match
                    i += 1

                if not match_found:
                    # For no match, include right row with NaNs for left columns
                    merged_row = pd.concat([pd.Series([pd.NA] * len(result_sorted.columns), index=result_sorted.columns), row_right]).to_dict()
                    merged_data.append(merged_row)

                j += 1

         # Convert merged data to DataFrame
        result = pd.DataFrame(merged_data)

    if categorize_by:
        grouped_data = data.groupby(categorize_by['columns'])

        # Initialize an empty dictionary for aggregation
        aggregation = {}

        grouped_data = data.groupby(categorize_by['columns'])
        # Process the groups as needed, for example, converting them to a dictionary
        categorized_data = {group: result.to_dict('records') for group, dataframe in grouped_data}
        result = categorized_data

        # if 'aggregate_function' in categorize_by and 'COUNT' in categorize_by['aggregate_function']:
        #     # Get the column name for COUNT, if specified
        #     count_column = categorize_by['aggregate_function'][1] if len(categorize_by['aggregate_function']) > 1 else None

        # if count_column:
        #     # Count non-null entries for the specified column
        #     result = grouped_data[count_column].count().reset_index(name='count')

        # Handle COUNT separately to avoid column conflict
        count_column = None
        if 'COUNT' in categorize_by['aggregate_function']:
            count_index = categorize_by['aggregate_function'].index('COUNT')
            if len(categorize_by['aggregate_function']) > count_index + 1:
                count_column = categorize_by['aggregate_function'][count_index + 1]
                if count_column not in categorize_by['columns']:
                    aggregation[count_column] = 'count'

        # Handle other aggregate functions
        for agg_func in ['MAX', 'MIN', 'AVERAGE', 'SUM']:
            if agg_func in categorize_by['aggregate_function']:
                func_index = categorize_by['aggregate_function'].index(agg_func)
                if len(categorize_by['aggregate_function']) > func_index + 1:
                    func_column = categorize_by['aggregate_function'][func_index + 1]
                    if func_column not in categorize_by['columns']:
                        if agg_func == 'AVERAGE':
                            aggregation[func_column] = 'mean'
                        else:
                            aggregation[func_column] = agg_func.lower()

        # Apply the aggregation
        if aggregation:
            result = grouped_data.agg(aggregation).reset_index()

            print(result)

        # if not isinstance(result, pd.DataFrame):
        #     result = pd.DataFrame(result)

    if bound:
        result = result[:bound['value']]
            
    if project_fields:
        # Filter the fields based on the PROJECT clause
        if isinstance(result, list):
            result = [{field: row.get(field, None) for field in project_fields} for row in result]
        elif isinstance(result, dict):
            for category, rows in result.items():
                result[category] = [{field: row.get(field, None) for field in project_fields} for row in rows]
        else:
            raise ValueError("Unsupported data type for PROJECT")

    if rank_by:
        if rank_by['order_direction'].upper() == 'ASC':
            return result.sort_values(by=rank_by['column'], ascending=True)
        else:
            return result.sort_values(by=rank_by['column'], ascending=False)

    if average_field:
        # Check if the result is a DataFrame
        if isinstance(result, pd.DataFrame):
            if average_field in result.columns:
                average_value = result[average_field].mean()
                result = pd.DataFrame({average_field: [average_value]})
            else:
                raise ValueError(f"{average_field} not found in DataFrame.")
        else:
            raise ValueError("Average calculation is only supported for pandas DataFrame.")


    if min_field:
        # Check if the result is a DataFrame
        if isinstance(result, pd.DataFrame):
            if min_field in result.columns:
                min_value = result[min_field].min()
                result = pd.DataFrame({min_field: [min_value]})
            else:
                raise ValueError(f"{min_field} not found in DataFrame.")
        else:
            raise ValueError("Minimum calculation is only supported for pandas DataFrame.")
        
    if max_field:
        # Check if the result is a DataFrame
        if isinstance(result, pd.DataFrame):
            if max_field in result.columns:
                max_value = result[max_field].max()
                result = pd.DataFrame({max_field: [max_value]})
            else:
                raise ValueError(f"{max_field} not found in DataFrame.")
        else:
            raise ValueError("Maximum calculation is only supported for pandas DataFrame.")
        
    if sum_by:
        # Check if the result is a DataFrame
        if isinstance(result, pd.DataFrame):
            if sum_by in result.columns:
                sum_value = result[sum_by].sum()
                result = pd.DataFrame({sum_by: [sum_value]})
            else:
                raise ValueError(f"{sum_by} not found in DataFrame.")
        else:
            raise ValueError("Sum calculation is only supported for pandas DataFrame.")

    return result
    # return []


# Main program
def parse_and_evaluate_query(data,query,components):
    # print("hi???")
    try:

        if len(components) < 4:
            print("Invalid query. Please provide a valid query.")
            sys.exit(1)
        # Parse the query components
        extract, dollar, using, field_name, *query_parts = components

        # Initialize variables
        conditions = None
        categorize_by = None
        bound = None
        rank_by = None
        project_fields = None
        average_field = None
        min_field = None
        max_field = None
        sum_by = None
        join_condition = None

        # Process the query parts
        while query_parts:
            keyword = query_parts.pop(0)

            if keyword == 'INNER' or keyword == "LEFT" or keyword == "RIGHT":
                type = keyword
                join_condition = {}
                if not query_parts or query_parts.pop(0) != 'JOIN':
                    print("Invalid query. 'INNER OR LEFT OR RIGHT' should be followed by 'JOIN'.")
                    sys.exit(1)

                # Assuming the JOIN condition is in the format 'columnA = columnB'
                if len(query_parts) >= 3 and query_parts[1] == '=':
                    left_column = query_parts.pop(0)  # Pops 'columnA'
                    query_parts.pop(0)  # Pops '='
                    right_column = query_parts.pop(0)  # Pops 'columnB'
                    join_condition = {'join_type': type,'left_column': left_column, 'right_column': right_column}
                else:
                    print("Invalid INNER JOIN syntax.")
                    sys.exit(1)

            if keyword == 'CATEGORIZE':
                if not query_parts or query_parts.pop(0) != 'BY':
                    print("Invalid query. 'CATEGORIZE BY' should be followed by column names.")
                    sys.exit(1)

                # Extract grouping columns
                group_columns = []
                while query_parts and query_parts[0] not in ['WHEN', 'LIKE', 'BOUND', 'RANK', 'PROJECT', 'AGGREGATE']:
                    group_columns.append(query_parts.pop(0))

                # Check if AGGREGATE is specified
                aggregate_function = []
                if query_parts and query_parts[0] == 'AGGREGATE':
                    query_parts.pop(0)  # Remove AGGREGATE keyword
                    if not query_parts or query_parts.pop(0) != 'BY':
                        print("Invalid query. 'AGGREGATE BY' should be followed by an aggregate function.")
                        sys.exit(1)
                    if query_parts:
                        while query_parts and query_parts[0] not in ['WHEN', 'LIKE', 'BOUND', 'RANK', 'PROJECT', 'AGGREGATE']:
                            aggregate_function.append(query_parts.pop(0))
                        # aggregate_function = query_parts.pop(0)

                categorize_by = {'columns': group_columns, 'aggregate_function':aggregate_function}

            if keyword == 'WHEN':
                print("hi???")
                if not query_parts:
                    print("Invalid query. Missing condition.")
                    sys.exit(1)
                conditions = ' '.join(query_parts)
                

            if keyword == 'LIKE':
                if not query_parts:
                    print("Invalid query. Missing pattern for 'LIKE'.")
                    sys.exit(1)
                if conditions:
                    conditions += f" and {categorize_by} like '{query_parts.pop(0)}'"
                else:
                    conditions = f"{categorize_by} like '{query_parts.pop(0)}'"

            if keyword == 'BOUND':
                if not query_parts:
                    print("Invalid query. Missing value for 'BOUND'.")
                    sys.exit(1)
                bound_value = query_parts.pop(0)
                try:
                    bound_value = int(bound_value)
                except ValueError:
                    print("Invalid query. 'BOUND' value must be an integer.")
                    sys.exit(1)
                bound = {'column': 'BOUND', 'value': bound_value}
        
            if keyword == 'PROJECT':
                if not query_parts:
                    print("Invalid query. Missing fields for 'PROJECT'.")
                    sys.exit(1)
                project_fields = query_parts.pop(0).split(',')

            if keyword == 'RANK':
                if not query_parts:
                    print("Invalid query. Missing column for ranking.")
                    sys.exit(1)
                rank_column = query_parts.pop(0)
                order_direction = 'ASC'
                while query_parts and query_parts[0] in ['ASC', 'DESC']:
                    if query_parts.pop(0) == 'DESC':
                        order_direction = 'DESC'
                rank_by = {'column': rank_column, 'order_direction': order_direction}

            if keyword == 'AVERAGE':
                if not query_parts:
                    print("Invalid query. Missing field for 'AVERAGE'.")
                    sys.exit(1)
                average_field = query_parts.pop(0)

            if keyword == 'MIN':
                if not query_parts:
                    print("Invalid query. Missing field for 'MIN'.")
                    sys.exit(1)
                min_field = query_parts.pop(0)

            if keyword == 'MAX':
                if not query_parts:
                    print("Invalid query. Missing field for 'MAX'.")
                    sys.exit(1)
                max_field = query_parts.pop(0)

            if keyword == 'SUM':
                if not query_parts:
                    print("Invalid query. Missing field for 'SUM'.")
                    sys.exit(1)
                sum_by = query_parts.pop(0)

        # Execute the query on the loaded data
        print("***************conditions", conditions)
        print("***************categorize_by", categorize_by)
        print("***************bound", bound)
        print("***************rank_by", rank_by)
        print("***************join", join_condition)
        print("****************min",min_field)
        result = execute_extract_query(data, conditions, categorize_by, bound, rank_by,join_condition,project_fields,average_field,min_field,max_field,sum_by)
        print("Query Result:")
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
