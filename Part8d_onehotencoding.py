import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import multiprocessing as mp

def one_hot_encode_column(args):
    column, column_name = args
    encoder = OneHotEncoder(sparse_output=False, categories=[[0, 1, 2]])
    encoded = encoder.fit_transform(column.values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded, index=column.index,
                              columns=[f'{column_name}_{i}' for i in range(3)])
    return encoded_df

def process_chunk(chunk):
    args = [(chunk[col], col) for col in chunk.columns]
    with mp.Pool() as pool:
        encoded_chunks = pool.map(one_hot_encode_column, args)
    return pd.concat(encoded_chunks, axis=1)

def main():
    input_file = 'Part8c_output.csv'
    output_file = 'Part8d_output.csv'

    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, index_col=0)

    # Separate the last column
    last_column = df.iloc[:, -1]
    df_to_encode = df.iloc[:, :-1]

    num_cores = max(1, mp.cpu_count() - 6)
    print(f"Using {num_cores} cores for processing")

    chunk_size = max(1, len(df_to_encode.columns) // num_cores)
    chunks = [df_to_encode.iloc[:, i:i+chunk_size] for i in range(0, df_to_encode.shape[1], chunk_size)]

    print(f"Processing {len(chunks)} chunks...")
    encoded_chunks = []
    for i, chunk in enumerate(chunks, 1):
        encoded_chunk = process_chunk(chunk)
        encoded_chunks.append(encoded_chunk)
        print(f"Processed chunk {i}/{len(chunks)}")

    print("Combining encoded chunks...")
    result_df = pd.concat(encoded_chunks, axis=1)

    # Add the last column back
    result_df = pd.concat([result_df, last_column], axis=1)

    print(f"Saving one-hot encoded data to: {output_file}")
    result_df.to_csv(output_file)

    print("Processing completed.")
    print(f"Original shape: {df.shape}")
    print(f"One-hot encoded shape: {result_df.shape}")

if __name__ == "__main__":
    main()
