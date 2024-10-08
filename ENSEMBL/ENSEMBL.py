import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

print("Starting file import...")
# Import the required files with low_memory=False
GTF = pd.read_csv('GTF_file_condensed.csv', low_memory=False, header=0)
merged_features = pd.read_csv('ML_Merge_Output/merged_features.csv', low_memory=False, index_col=0)
print("File import completed.")

print("Starting column title extraction from merged_features...")
# Extract column titles from merged_features (excluding the last column which is 'Case1_Control0')
c_pos_genotype_values = merged_features.columns.tolist()[:-1]
print("Column title extraction completed.")

print("Creating output dataframe...")
# Create the output dataframe
output_df = pd.DataFrame({'C_POS_Genotype': c_pos_genotype_values})
output_df[['CHROM', 'POS', 'Genotype']] = output_df['C_POS_Genotype'].str.split('_', expand=True)
output_df = output_df.drop(columns=['C_POS_Genotype'])
print("Output dataframe created and processed.")

print("Processing CHROM and POS...")
# Remove rows where CHROM is not a number
output_df = output_df[output_df['CHROM'].str.isnumeric()]
# Convert CHROM and POS to numeric for comparison
output_df['CHROM'] = pd.to_numeric(output_df['CHROM'])
output_df['POS'] = pd.to_numeric(output_df['POS'])
GTF['CHROM'] = pd.to_numeric(GTF['CHROM'], errors='coerce')
print("CHROM and POS processed.")

print("Merging GTF data...")
# Merge GTF data based on the conditions
output_df = output_df.merge(
    GTF[['CHROM', 'POS_start', 'POS_end', 'gene_id', 'gene_name', 'gene_biotype']],
    how='left',
    left_on='CHROM',
    right_on='CHROM'
)

# Filter rows based on POS_start and POS_end conditions
output_df = output_df[
    (output_df['POS'] >= output_df['POS_start']) & 
    (output_df['POS'] <= output_df['POS_end'])
]

# Drop the POS_start and POS_end columns as they are no longer needed
output_df = output_df.drop(columns=['POS_start', 'POS_end'])
output_df = output_df.drop_duplicates()
print("GTF data merged and processed.")

# Ensure the output directory exists
output_dir = 'ENSEMBL'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Output directory '{output_dir}' exists or created.")

print("Saving output dataframe...")
# Save the output dataframe
output_df.to_csv(os.path.join(output_dir, 'ENSEMBL_output.csv'), index=False)
print("Output dataframe saved.")

print("Saving gene names...")
# Save the list of gene names as its own CSV file
gene_names = output_df['gene_name'].drop_duplicates().reset_index(drop=True)
gene_names.to_csv(os.path.join(output_dir, 'gene_names.csv'), index=False)
print("Gene names saved.")

print("Files imported and processing completed.")
print(f"Output dataframe shape: {output_df.shape}")
print(output_df.head())

# Print column titles
print("\nColumn titles in output_df:")
print(output_df.columns.tolist())

# Create histogram
print("Creating histogram...")

# Read the saved CSV file
output_df = pd.read_csv(os.path.join(output_dir, 'ENSEMBL_output.csv'))

# Convert CHROM to string to ensure proper sorting
output_df['CHROM'] = output_df['CHROM'].astype(str)

# Create a new column 'gene_type' based on gene_biotype
output_df['gene_type'] = np.where(output_df['gene_biotype'] == 'protein_coding', 'Intron', 'Exon')

# Group by CHROM and gene_type, and count the occurrences
grouped_data = output_df.groupby(['CHROM', 'gene_type']).size().unstack(fill_value=0)

# Create a complete index with all chromosomes from 1 to 22
all_chromosomes = [str(i) for i in range(1, 23)]
grouped_data = grouped_data.reindex(all_chromosomes, fill_value=0)

# Reorder columns to put Intron at the bottom
grouped_data = grouped_data[['Intron', 'Exon']]

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(15, 8))

# Define colors
exon_color = (33/255, 150/255, 243/255)  # Royal Blue for Exon
intron_color = (239/255, 45/255, 81/255)  # Red for Introns

# Plot the stacked bars
grouped_data.plot(kind='bar', stacked=True, ax=ax, color=[intron_color, exon_color])

plt.title('Number of Features per Chromosome')
plt.xlabel('Chromosome')
plt.ylabel('Number of Features')
plt.legend(title='Gene Type')

# Set x-axis ticks and labels
plt.xticks(range(len(all_chromosomes)), all_chromosomes, rotation=0)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'features_per_chromosome_histogram.png'))
plt.close()

print("Histogram created and saved.")