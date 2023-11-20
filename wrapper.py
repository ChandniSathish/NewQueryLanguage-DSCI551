import pandas as pd
import sys
import pandas as pd
import numpy as np
import os
import crud_sql
import lexer_parser
from datetime import datetime

def split_by_chunks(query,components):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_directory, 'iris.csv')

    chunk_size = 1000  # Define your chunk size
    final_result = pd.DataFrame()

    for chunk in pd.read_csv(csv_file_path, sep=',', chunksize=chunk_size):
        print("inside loop")
        processed_chunk = lexer_parser.parse_and_evaluate_query(chunk,query,components)
        final_result = pd.concat([final_result, processed_chunk])

    # Now final_result contains the processed data from all chunks
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
            else:
                split_by_chunks(query,components)
            if query.lower() == 'exit':
                break
        except Exception as e:
            print(f"An error occurred: {e}")
       
