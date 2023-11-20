import pandas as pd
import sys
import pandas as pd
import numpy as np
import os
import crud_sql

# def process_chunk(chunk, conditions=None, categorize_by=None, rank_by=None, join=None):
#     # Apply various operations like conditions, categorize_by, rank_by, join, etc.
#     # Similar to what you have in execute_extract_query, but applied to the chunk
#     # Return the processed chunk
#     return processed_chunk

def split_by_chunks(query,components):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, 'iris.csv')

    chunk_size = 1000  # Define your chunk size
    final_result = pd.DataFrame()

    for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
        #call lexer_parser.py and pass query,components as parameters
        # processed_chunk = process_chunk(query,components)
        final_result = pd.concat([final_result, processed_chunk])

    # Now final_result contains the processed data from all chunks
    print(final_result)

# Main program
if __name__ == "__main__":
    while True:
        try:
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
            # else:
            #     split_by_chunks(query,components)
            # if query.lower() == 'exit':
            #     break
        except Exception as e:
            print(f"An error occurred: {e}")
       
