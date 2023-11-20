# File: json_operations.py
import json

# Load data from the JSON file
def load_data_from_json(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []
    return data

# Function to execute an "INSERT" query on the data
def execute_insert_query(data, values, file):
    customer, product, price = map(str.strip, values.split(','))
    data.append({
        'customer': customer,
        'product': product,
        'price': int(price)
    })
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

# Function to execute a "DELETE" query on the data
def delete_entries_with_condition(file_path, condition):
    try:
        # Read the file and filter entries
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Filter entries and get the entries to delete
        entries_to_delete = [entry for entry in data if does_entry_match_condition(entry, condition)]

        if entries_to_delete:
            # Remove the entries from the list
            data = [entry for entry in data if entry not in entries_to_delete]

            # Write back the updated data
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=2)

            print(f"Entries matching the condition '{condition}' deleted successfully.")
        else:
            print(f"No entries found matching the condition '{condition}'.")

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to execute an "UPDATE" query on the data
def update_entries_with_condition(data, new_value, condition):
    updated_entries = []

    for entry in data:
        if does_entry_match_condition(entry, condition):
            updated_entry = entry.copy()
            updated_entry.update(new_value)
            updated_entries.append(updated_entry)
        else:
            updated_entries.append(entry)

    return updated_entries

# Function to write data back to file
def write_data_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def does_entry_match_condition(entry, condition):
    # Check if the entry matches the specified condition
    parts = condition.split('=')
    if len(parts) == 2:
        column = parts[0].strip()
        value = parts[1].strip()

        if column in entry and entry[column] == value:
            return True

    return False

# Main program
if __name__ == "__main__":
    # Load data from the JSON file
    json_file_path = 'sample2.json'
    data = load_data_from_json(json_file_path)

    while True:
        try:
            # Get user input
            input_line = input("Enter query (or 'exit' to quit): ")

            if input_line.lower() == 'exit':
                break

            # Parse the input line
            if input_line.startswith("INSERT VALUES"):
                values = input_line[len("INSERT VALUES"):].strip()
                execute_insert_query(data, values, json_file_path)

                # Print the updated data
                print("Updated Data:")
                print(json.dumps(data, indent=2))

            # Parse the input line
            elif input_line.startswith("DELETE WHERE"):
                condition = input_line[len("DELETE WHERE"):].strip()
                delete_entries_with_condition(json_file_path, condition)  # Correct function name

            # Parse the input line
            elif input_line.startswith("UPDATE VALUE WHERE"):
                parts = input_line[len("UPDATE VALUE WHERE"):].strip().split(" SET ")
                condition = parts[0].strip()
                new_value = json.loads(parts[1].strip())
                data = update_entries_with_condition(data, new_value, condition)
                write_data_to_file(data, json_file_path)  # Write back to file
                print(f"Entries matching the condition '{condition}' updated successfully.")

            else:
                print("Invalid query format. Please use 'INSERT VALUES', 'DELETE WHERE', or 'UPDATE VALUE WHERE'.")

        except Exception as e:
            print(f"An error occurred: {e}")



