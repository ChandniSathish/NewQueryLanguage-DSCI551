import pandas as pd

# Load the genomic data from the JSON file
genomic_data = pd.read_json('sample_dataset.json')

# Select 'Gene' and 'Expression' columns
projection = genomic_data[['Gene', 'Expression']]
print("Projection:\n", projection)

# Filter genes with an expression greater than 4.0
filtered_data = genomic_data[genomic_data['Expression'] > 4.0]
print("Filtered data:\n", filtered_data)

# Group data by the 'Gene' column
grouped_data = genomic_data.groupby('Gene')

# Perform aggregation on the grouped data, such as calculating the mean expression for each gene
avg_expression_by_gene = grouped_data['Expression'].mean()
print("Mean expression for each gene:\n", avg_expression_by_gene)

# Calculate the total expression for each gene
total_expression_by_gene = grouped_data['Expression'].sum()
print("Total expression for each gene:\n", total_expression_by_gene)
