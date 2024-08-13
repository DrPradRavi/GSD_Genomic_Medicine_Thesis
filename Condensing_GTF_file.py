import pandas as pd
import re

# Read the GTF file, skipping lines until '2022-07' is found
with open('Homo_sapiens.GRCh38.108.gtf', 'r') as file:
    for line in file:
        if '2022-07' in line:
            break
    df = pd.read_csv(file, sep='\t', header=None, comment='#')

# Select and rename the required columns
df = df[[0, 2, 3, 4, 8]]
df.columns = ['CHROM', 'feature type', 'POS_start', 'POS_end', 'attribute']

# Function to extract information from the attribute column
def extract_info(attr, key):
    match = re.search(f'{key} "([^"]+)"', attr)
    return match.group(1) if match else ''

# Extract gene_id, gene_name, and gene_biotype
df['gene_id'] = df['attribute'].apply(lambda x: extract_info(x, 'gene_id'))
df['gene_name'] = df['attribute'].apply(lambda x: extract_info(x, 'gene_name'))
df['gene_biotype'] = df['attribute'].apply(lambda x: extract_info(x, 'gene_biotype'))

# Drop the attribute column
df = df.drop('attribute', axis=1)

# Add an ID column (assuming it's the index + 1)
df['ID'] = df.index + 1

# Reorder the columns
df = df[['CHROM', 'POS_start', 'POS_end', 'ID', 'gene_id', 'gene_name', 'feature type', 'gene_biotype']]

# Export to CSV
df.to_csv('GTF_file_condensed.csv', index=False)

print("File processing complete. Output saved as 'GTF_file_condensed.csv'.")