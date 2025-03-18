import json

input_file_name = "input.json"

# Load the input data from the JSON file
with open(input_file_name, 'r') as f:
    dic = json.load(f)

# Extract 'extra_arguments' if present, otherwise use an empty dictionary
extra_arguments = dic.get('extra_arguments', {})

# Extract 'solver_params' if present, otherwise use an empty dictionary
solver_params = dic.get('solver_params', {})

# Import the run function from main.py
import main

# Execute the run function using the data contained in the "data" key of the input JSON
result = main.run(dic['data'], solver_params, extra_arguments)

print(result)