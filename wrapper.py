import pandas as pd
import sys
import pandas as pd
import numpy as np
import os
import crud_sql
import lexer_parser
from datetime import datetime
KEYWORDS = ['WHEN', 'LIKE', 'BOUND', 'RANK', 'PROJECT', 'AGGREGATE', 'UNIQUE', 'MAX','MIN','SUM','AVERAGE','COUNT', 'INNER','LEFT','RIGHT','ASC','DESC','SLICE','SIMILAR']

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
        index = categorize_by['aggregate_function'].index('COUNT')
        categorize_by['aggregate_function'][index] = 'SUM'

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
def join_data(result, result1, join):
    print("join type", join)
    join_type = join['join_type']
    left_column = join['left_column']
    right_column = join['right_column']

    result_sorted = result.sort_values(by=left_column)
    other_df_sorted = result1.sort_values(by=right_column)

    merged_data = []
    i, j = 0, 0

    if join_type == "INNER":
        for index_left, row_left in result_sorted.iterrows():
            for index_right, row_right in other_df_sorted.iterrows():
                if row_left[left_column] == row_right[right_column]:
                    merged_row = {**row_left.to_dict(), **row_right.to_dict()}
                    merged_data.append(merged_row)

    if join_type == "LEFT":
        for index_left, row_left in result_sorted.iterrows():
            match_found = False
            for index_right, row_right in other_df_sorted.iterrows():
                if row_left[left_column] == row_right[right_column]:
                    merged_row = {**row_left.to_dict(), **row_right.to_dict()}
                    merged_data.append(merged_row)
                    match_found = True
                    break  # Exit the inner loop after finding the first match
            if not match_found:
                merged_row = {**row_left.to_dict(), **{col: pd.NA for col in other_df_sorted.columns}}
                merged_data.append(merged_row)

    if join_type == "RIGHT":
        for index_right, row_right in other_df_sorted.iterrows():
            match_found = False
            for index_left, row_left in result_sorted.iterrows():
                if row_left[left_column] == row_right[right_column]:
                    merged_row = {**row_left.to_dict(), **row_right.to_dict()}
                    merged_data.append(merged_row)
                    match_found = True
                    break  # Exit the inner loop after finding the first match
            if not match_found:
                merged_row = {**{col: pd.NA for col in result_sorted.columns}, **row_right.to_dict()}
                merged_data.append(merged_row)

    # Convert merged data to DataFrame
    result = pd.DataFrame(merged_data)
    return result

def get_join_hash(query,components):
    rank_fields = []
    query_parts = query.split("JOIN")[1].strip()
    join_type = query.split("JOIN")[0].strip().split()[-1]
    query_parts = query_parts.split()
    if len(query_parts) >= 3 and query_parts[1] == '=':
        left_column = query_parts.pop(0)  # Pops 'columnA'
        query_parts.pop(0)  # Pops '='
        right_column = query_parts.pop(0)  # Pops 'columnB'
        join_condition = {'join_type': join_type,'left_column': left_column, 'right_column': right_column}
    else:
        print("Invalid JOIN syntax.")
        sys.exit(1)
    return join_condition


def split_by_chunks_and_join(query,components,chunk_size):
    # print("****************join chunk")
    # Get the script directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Define file paths
    csv_file_path1 = os.path.join(script_directory, 'hibiscus1.csv')
    csv_file_path2 = os.path.join(script_directory, 'hibiscus2.csv')

    # Initialize a variable to store the final result
    final_result1 = pd.DataFrame()
    final_result2 = pd.DataFrame()

    # Use pd.read_csv with chunksize for both files
    chunk_iter1 = pd.read_csv(csv_file_path1, sep=',', chunksize=chunk_size)
    chunk_iter2 = pd.read_csv(csv_file_path2, sep=',', chunksize=chunk_size)

    # Iterate over both file chunks simultaneously
    for chunk1, chunk2 in zip(chunk_iter1, chunk_iter2):
        # Process the chunks
        processed_chunk1 = lexer_parser.parse_and_evaluate_query(chunk1, query, components)
        processed_chunk2 = lexer_parser.parse_and_evaluate_query(chunk2, query, components)

        final_result1 = pd.concat([final_result1, processed_chunk1])
        final_result2 = pd.concat([final_result2, processed_chunk2])
    
    join_details = get_join_hash(query,components)
    # print("join_details",join_details)

    # Generate the iterators for both final results
    chunk_iter1 = (final_result1[i:i + chunk_size] for i in range(0, len(final_result1), chunk_size))
    chunk_iter2 = (final_result2[i:i + chunk_size] for i in range(0, len(final_result2), chunk_size))

    final_joined_result = pd.DataFrame()

    for chunk1, chunk2 in zip(chunk_iter1, chunk_iter2):
        # Process the chunks
        joined_chunk = join_data(chunk1, chunk2, join_details)
        final_joined_result = pd.concat([final_joined_result, joined_chunk])

    # print("final_joined_result")
    # print(final_joined_result)
    return final_joined_result


def split_by_chunks(query,components):
    chunk_size = 50  # Define your chunk size

    if 'JOIN' in components:
        final_result = split_by_chunks_and_join(query,components,chunk_size)
        if  'BOUND' in components:
            query_parts = query.split("BOUND")[1].strip()
            query_parts = query_parts.split()
            if 'SLICE' in components:
                bound_range = int(query_parts[-1])+int(query_parts[0])
                if bound_range <= len(final_result):
                    # print("bound_range",bound_range)
                    final_result = final_result[int(query_parts[-1]):bound_range]
            else:
                bound_range = int(query_parts[0])
                final_result = final_result[:bound_range]
    else:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_directory, 'hibiscus.csv')

        final_result = pd.DataFrame()
        final_aggregations = pd.DataFrame()
        chunk_sum = 0
        for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
            processed_chunk = lexer_parser.parse_and_evaluate_query(chunk,query,components)
            chunk_sum += chunk_size
            

            # print("*************type",type(processed_chunk))

            if isinstance(processed_chunk, dict):
                final_result = merge_dictionaries(final_result, processed_chunk)
            else:
                final_result = pd.concat([final_result, processed_chunk])
            
    # final_result = finalize_aggregations(final_aggregations)
    # agg_func =  ['MAX', 'MIN', 'AVERAGE', 'SUM','COUNT']
    #EXTRACT $ USING data CATEGORIZE BY sepalLength AGGREGATE BY COUNT sepalLength

    # print("******final_result1",final_result)
    if  'BOUND' in components:
        query_parts = query.split("BOUND")[1].strip()
        query_parts = query_parts.split()
        if 'SLICE' in components:
            bound_range = int(query_parts[-1])+int(query_parts[0])
            if bound_range <= len(final_result):
                # print("bound_range",bound_range)
                final_result = final_result[int(query_parts[-1]):bound_range]
                
        else:
            bound_range = int(query_parts[0])
            final_result = final_result[:bound_range]
            # print("bound_range",bound_range)
            # print("bound result",final_result[:bound_range])
            
    if isinstance(final_result, dict):
        final_result = pd.DataFrame(final_result)

    if 'CATEGORIZE' in components and 'AGGREGATE' in components:
        group_columns = []
        query_parts = query.split("BY")[1].strip()
        query_parts = query_parts.split()
        while query_parts and query_parts[0] not in KEYWORDS:
            group_columns.append(query_parts.pop(0))
            # print(group_columns)
        
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
        # print("categorize_by",categorize_by)
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
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(script_directory, 'output')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_{timestamp}.csv"
    output_filepath = os.path.join(output_directory, output_filename)

    # Save the final result to the new CSV file
    final_result.to_csv(output_filepath, index=False)

# def process_crud_in_chunks(query):
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     csv_file_path = os.path.join(script_directory, 'hibiscus.csv')

#     final_result = pd.DataFrame()
#     final_aggregations = pd.DataFrame()
#     chunk_sum = 0
#     for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
#         processed_chunk = lexer_parser.parse_and_evaluate_query(chunk,query,components)
#         chunk_sum += chunk_size
        

#         # print("*************type",type(processed_chunk))

#         if isinstance(processed_chunk, dict):
#             final_result = merge_dictionaries(final_result, processed_chunk)
#         else:
#             final_result = pd.concat([final_result, processed_chunk])
            

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
       
