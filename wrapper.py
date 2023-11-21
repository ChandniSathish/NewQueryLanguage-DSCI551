import pandas as pd
import sys
import pandas as pd
import numpy as np
import os
import crud_sql
import lexer_parser
from datetime import datetime

def calculate_final_averages(final_results):
    # Calculate the average for each variety
    averages = []
    for variety, values in final_results.items():
        avg = values['sum'] / values['count']
        averages.append({'variety': variety, 'average_sepalLength': avg})
    return pd.DataFrame(averages)

def merge_aggregations(final_aggregations, chunk_aggregations):
    for key in chunk_aggregations:
        if key in final_aggregations:
            final_aggregations[key].append(chunk_aggregations[key])
        else:
            final_aggregations[key] = [chunk_aggregations[key]]

def finalize_aggregations(final_aggregations):
    final_result = {}
    for key, partials in final_aggregations.items():
        combined = pd.concat(partials, ignore_index=True)
        final_result[key] = combined.groupby(key).sum().reset_index()
    return pd.concat(final_result.values(), ignore_index=True)

def merge_dictionaries(final_dict, new_dict):
    final_dict = {}
    # Determine the max length of lists in final_dict
    max_length = max(len(lst) for lst in final_dict.values()) if final_dict else 0

    # Merge new_dict into final_dict
    for key, values in new_dict.items():
        if key in final_dict:
            # Extend the list with values
            final_dict[key].extend(values)
        else:
            # Add the new key with its values, padded to max_length
            final_dict[key] = [np.nan] * max_length + values

    # Pad all lists in final_dict to the new max length
    new_max_length = max(len(lst) for lst in final_dict.values())
    for key, values in final_dict.items():
        if len(values) < new_max_length:
            final_dict[key].extend([np.nan] * (new_max_length - len(values)))

    return final_dict

def categorize_and_aggregate(data,categorize_by):
    grouped_data = data.groupby(categorize_by['columns'])

    # Initialize an empty dictionary for aggregation
    aggregation = {}

    grouped_data = data.groupby(categorize_by['columns'])
    # Process the groups as needed, for example, converting them to a dictionary
    categorized_data = {group: data.to_dict('records') for group, dataframe in grouped_data}
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
            # print("aggregate2***********************")
            func_index = categorize_by['aggregate_function'].index(agg_func)
            if len(categorize_by['aggregate_function']) > func_index + 1:
                # print("aggregate3***********************")
                func_column = categorize_by['aggregate_function'][func_index + 1]
                print("func_column",func_column)
                if func_column not in categorize_by['columns']:
                    # print("aggregate4***********************")
                    if agg_func == 'AVERAGE':
                        aggregation[func_column] = 'mean'
                    else:
                        aggregation[func_column] = agg_func.lower()
    # print("aggregate1***********************")
    print(aggregation)
    # Apply the aggregation
    if aggregation:
        result = grouped_data.agg(aggregation).reset_index()
        # print("aggregate***********************")
        print(result)
    return result

def split_by_chunks(query,components):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, 'iris.csv')

    chunk_size = 50  # Define your chunk size
    final_result = pd.DataFrame()
    final_aggregations = pd.DataFrame()
    chunk_sum = 0
    KEYWORDS = ['WHEN', 'LIKE', 'BOUND', 'RANK', 'PROJECT', 'AGGREGATE', 'UNIQUE', 'MAX','MIN','SUM','AVERAGE','COUNT', 'INNER','LEFT','RIGHT','ASC','DESC','SLICE']
    for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
        processed_chunk = lexer_parser.parse_and_evaluate_query(chunk,query,components)
        chunk_sum += chunk_size
        

        # print("*************type",type(processed_chunk))

        if isinstance(processed_chunk, dict):
            final_result = merge_dictionaries(final_result, processed_chunk)
        else:
            final_result = pd.concat([final_result, processed_chunk])
        if  'BOUND' in components and chunk_sum > int(components[-1]):
            query_parts = query.split("BOUND")[1].strip()
            query_parts = query_parts.split()
            if 'SLICE' in components:
                bound_range = int(query_parts[-1])+int(query_parts[0])
                if bound_range <= len(final_result):
                    print("bound_range",bound_range)
                    final_result = final_result[int(query_parts[-1]):bound_range]
                    break
            else:
                bound_range = int(query_parts[0])
                final_result = final_result[:bound_range]
                break
    # final_result = finalize_aggregations(final_aggregations)
    # agg_func =  ['MAX', 'MIN', 'AVERAGE', 'SUM','COUNT']
    #EXTRACT $ USING data CATEGORIZE BY sepalLength AGGREGATE BY COUNT sepalLength

    print("******final_result1",final_result)
    if isinstance(final_result, dict):
        final_result = pd.DataFrame(final_result)

    if 'CATEGORIZE' in components:
        group_columns = []
        query_parts = query.split("BY")[1].strip()
        query_parts = query_parts.split()
        while query_parts and query_parts[0] not in KEYWORDS:
            group_columns.append(query_parts.pop(0))
            print(group_columns)
        
        # Check if AGGREGATE is specified
        aggregate_function = []
        
        if'AGGREGATE' in query_parts:
            query_parts = query.split("AGGREGATE")[1].strip()
            query_parts = query_parts.split()
            if not query_parts or query_parts.pop(0) != 'BY':
                print("Invalid query. 'AGGREGATE BY' should be followed by an aggregate function.")
                # sys.exit(1)
            if query_parts:
                while query_parts and query_parts[0] not in ['WHEN', 'LIKE', 'BOUND', 'RANK', 'PROJECT', 'AGGREGATE']:
                    aggregate_function.append(query_parts.pop(0))
                # aggregate_function = query_parts.pop(0)

        categorize_by = {'columns': group_columns, 'aggregate_function':aggregate_function}
        print("categorize_by",categorize_by)
        final_result = categorize_and_aggregate(final_result,categorize_by)

    if 'UNIQUE' in components:
        first_column_values = [row[final_result.columns[0]] for index, row in final_result.iterrows()]
        unique_values = list(set(first_column_values))

        # If you need to convert it back to a DataFrame
        final_result = pd.DataFrame(unique_values, columns=[final_result.columns[0]])
    
    if 'RANK' in components:
        rank_fields = []
        query_parts = query.split("RANK")[1].strip()
        query_parts = query_parts.split()
        while query_parts and query_parts[0] not in KEYWORDS:
            rank_fields.append(query_parts.pop(0))
        # print("query_parts",query_parts)
        # print("rank_fields",rank_fields)
        if 'ASC' in query_parts:
            # order = [True] * len(rank_fields)
            # order = [True, True]
            final_result = final_result.sort_values(by=rank_fields, ascending=True)
        elif 'DESC' in query_parts:
            final_result = final_result.sort_values(by=rank_fields, ascending=False)
        else:
            final_result = final_result.sort_values(by=rank_fields, ascending=True)
    
    if 'AGGREGATE' not in components:
        for agg_func in ['MAX', 'MIN', 'AVERAGE', 'SUM', 'COUNT']:
            if agg_func in components:
                result = final_result
            # Assuming final_result is a DataFrame and has at least one column
                first_column_values = [row[final_result.columns[0]] for index, row in final_result.iterrows()]

                if 'AVERAGE' in components:
                    final_average = sum(first_column_values) / len(first_column_values)
                    print("Final Average:", final_average)
                    final_result = final_average

                if 'MIN' in components:
                    final_min = min(first_column_values)
                    print("Final Min:", final_min)
                    final_result = final_min

                if 'SUM' in components:
                    final_sum = sum(first_column_values)
                    print("Final Sum:", final_sum)
                    final_result = final_sum

                if 'MAX' in components:
                    final_max = max(first_column_values)
                    print("Final Max:", final_max)
                    final_result = final_max
                if 'COUNT' in components:
                    # print("******************hiiiiiiii")
                    final_count = sum(first_column_values)
                    print("Final Count:", final_count)
                    final_result = final_count
                
                final_result = pd.DataFrame([final_result])

        # merge_aggregations(final_aggregations, processed_chunk)
    print("final_result************")
    print(final_result)

    # Output directory
    output_directory = os.path.join(script_directory, 'output')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.csv"
    output_filepath = os.path.join(output_directory, output_filename)

    # Save the final result to the new CSV file
    final_result.to_csv(output_filepath, index=False)

# Main program
if __name__ == "__main__":
    while True:
        # try:
        # Get user input
        query = input("Enter a basic SQL query (or 'exit' to quit): ")

        # Split the query into components
        components = query.split()
        print("********components",components)

        crud_funcs = ["INSERT" , "UPDATE", "DELETE"]
        if components[0] in crud_funcs:
            #call crud_sql.py
            print("crud func")
            crud_sql.parse_crud_query(query)
        else:
            split_by_chunks(query,components)
        if query.lower() == 'exit':
            break
        # except Exception as e:
        #     print(f"An error occurred: {e}")
       
