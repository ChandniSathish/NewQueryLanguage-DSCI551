import os

# Main program
if __name__ == "__main__":
    while True:
        try:
            # Get user input
            input_line = input("Enter 'sql' for SQL or nosql for NoSQL operations (or 'exit' to quit): ")
            input_line = input_line.lower()
            if input_line == 'sql':
                # Use os.system to run the script
                    return_code = os.system(f'python wrapper_sql.py')
            else:
                # Use os.system to run the script
                    return_code = os.system(f'python wrapper_nosql.py')   

        except Exception as e:
              print(f"Error occurred: {e}")            

