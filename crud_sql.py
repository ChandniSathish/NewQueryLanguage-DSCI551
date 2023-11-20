import json
import ply.lex as lex
import ply.yacc as yacc
import sys
import pandas as pd
import numpy as np
import os
import re
import ast


def evaluate_condition(row, key, condition):
    operator, value_or_column = condition
    value = row[value_or_column] if value_or_column in row else value_or_column

    if operator == ">":
        return row[key] > value
    elif operator == "<":
        return row[key] < value
    elif operator == ">=":
        return row[key] >= value
    elif operator == "<=":
        return row[key] <= value
    elif operator == "==":
        return row[key] == value
    elif operator == "!=":
        return row[key] != value
    else:
        raise ValueError(f"Unsupported operator: {operator}")

def execute_update_query(data, conditions, updates):
    for index, row in data.iterrows():
        if all(evaluate_condition(row, key, cond) for key, cond in conditions.items()):
            for update_key, update_value in updates.items():
                data.at[index, update_key] = update_value
    return data

def execute_insert_query(data, new_rows):
    print("*********************inside insert3")
    print(data.head())
    
    new_rows = ast.literal_eval(new_rows)
    print("new_rows",new_rows)

    try:
        # Attempt to create a new DataFrame from new_rows
        new_data = pd.DataFrame(new_rows)
        print("New data as DataFrame:")
        print(new_data)
    except Exception as e:
        # Print the error if DataFrame creation fails
        print("Error in creating DataFrame from new_rows:", e)
        return data 
    
    updated_data = pd.concat([data, new_data], ignore_index=True)
    print("updated_data",updated_data.tail(1))
    return updated_data

def execute_delete_query(data, conditions):
    for index, row in data.iterrows():
        if all(evaluate_condition(row, key, cond) for key, cond in conditions.items()):
            # Drop the row if all conditions are met
            data = data.drop(index)
    # Reset index after dropping rows
    return data.reset_index(drop=True)

def parse_crud_query(query):
    # Load data into dataframe
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, 'iris.csv')
    data= pd.read_csv(csv_file_path, sep=',')

    try:
        
        # Split the query into components
        components = query.split()
        print("********components",components)

        crud_funcs = ["INSERT" , "UPDATE", "DELETE"]
        if components[0] in crud_funcs:
            if components[0] == "INSERT":
                print("*********************inside insert1")
                if components[1]:
                    table_name = components[1]
                else:
                    print("Invalid query. Missing table name.")
                    
                if components[2] != 'VALUES':
                    print("Invalid query. 'VALUES' keyword missing.")
                    
                
                # Splitting the command by 'VALUES'
                parts = query.split("VALUES")

                # Extracting the part after 'VALUES'
                data_after_values = parts[1].strip() if len(parts) > 1 else ""
                # The remaining part of query_parts should be the values to insert
                if data_after_values:
                    try:
                        # print("*********************inside insert")
                        new_rows = data_after_values
                        # print("components.length",len(components))
                        data = execute_insert_query(data, new_rows)
                        
                        data.to_csv(csv_file_path, index=False)

                    except Exception as e:
                        print(f"Error parsing new rows: {e}")
                        
                else:
                    print("Invalid query. Missing values to insert.")
                    
            if components[0] == "UPDATE":
                # Handling for UPDATE operation
                if len(components) < 4 or 'VALUES' not in components or 'CONDITION' not in components:
                    print("Invalid UPDATE query. Missing 'VALUES' or 'CONDITION'.")
        

                # Extract conditions and updates
                values_part = query.split("VALUES")[1].split("CONDITION")[0].strip()
                condition_part = query.split("CONDITION")[1].strip()

                # Convert string representations to actual Python dictionaries
                updates = ast.literal_eval(values_part)
                conditions = ast.literal_eval(condition_part)

                # Execute the update query
                data = execute_update_query(data, conditions, updates)
                data.to_csv(csv_file_path, index=False)
                print(data.tail(3))
                # Example usage
                # conditions = {'variety': 'Mojito'}  # Conditions to identify rows to update
                # updates = {'variety': 'VirginMojito'}  # New values
            if components[0] == "DELETE":
                if 'CONDITION' not in components:
                    print("Invalid DELETE query. Missing 'CONDITION'.")
        
                # Extract conditions
                condition_part = query.split("CONDITION")[1].strip()
                # Convert string representation to actual Python dictionary
                conditions = ast.literal_eval(condition_part)
                # Execute the delete query
                data = execute_delete_query(data, conditions)
                data.to_csv(csv_file_path, index=False)
                print("Data after deletion:")
                print(data.tail(3))
    except Exception as e:
        print(f"An error occurred: {e}")