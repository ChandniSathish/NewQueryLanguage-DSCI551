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

def split_by_chunks(query,components):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, 'iris.csv')

    chunk_size = 50  # Define your chunk size
    final_result = pd.DataFrame()
    final_aggregations = pd.DataFrame()

    for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
        print("inside loop")
        processed_chunk = lexer_parser.parse_and_evaluate_query(chunk,query,components)
        print("*************type",type(processed_chunk))
        final_result = pd.concat([final_result, processed_chunk])
    # final_result = finalize_aggregations(final_aggregations)
    # agg_func =  ['MAX', 'MIN', 'AVERAGE', 'SUM','COUNT']
    print(final_result)
    for agg_func in ['MAX', 'MIN', 'AVERAGE', 'SUM']:
        if agg_func in components:
            if 'AVERAGE' in components:
                first_column = final_result.iloc[:, 0]
                # Calculating the mean of the first column
                final_average = first_column.mean()
                print("Final Average of petalLength:", final_average)
                final_result = final_average
            if 'MIN' in components:
                

        # merge_aggregations(final_aggregations, processed_chunk)
   
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
       
