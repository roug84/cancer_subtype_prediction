# process_expression_data.py
import os
import gzip
import pandas as pd
import sys
from cancer_subtype_prediction.configs import TCGA_DATA_PATH, TCGA_RESULTS_PATH


# This script processes the TCGA expression data in extreme low-memory mode
# It creates multiple smaller parquet files instead of one large file

def process_tcga_data_low_memory(
        input_gz_file,
        output_dir,
        phenotype_file,
        cancer_types,
        chunk_size=1000  # Genes per chunk
):
    """
    Process TCGA data in extreme low-memory mode by:
    1. Identifying relevant samples from phenotype data
    2. Reading expression data in small chunks
    3. Creating multiple smaller parquet files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load phenotype data to identify cancer samples
    print("Loading phenotype data...")
    phenotype_df = pd.read_table(
        phenotype_file,
        compression="gzip",
        header=0,
        sep="\t",
        encoding="ISO-8859-1",
        index_col=0,
        dtype="str"
    )

    # Print column names to help identify cancer type column
    print("Phenotype columns:", phenotype_df.columns.tolist())

    # Try to find the column with cancer type information
    cancer_col = None
    for col in phenotype_df.columns:
        if any(x in col.lower() for x in ['cancer', 'disease', 'type', 'tcga']):
            # Check if this column contains our cancer types
            sample_values = phenotype_df[col].value_counts().head(10)
            print(f"Potential cancer column: {col}")
            print(f"Sample values: {sample_values}")

            # Check for cancer type mapping
            if col == 'primary disease or tissue' and 'BRCA' in cancer_types:
                print("Found BRCA mapping in 'primary disease or tissue'")
                # Map BRCA abbreviation to full name
                cancer_mapping = {'BRCA': 'Breast Invasive Carcinoma'}
                mapped_types = [cancer_mapping.get(ct, ct) for ct in cancer_types]
                if any(cancer in phenotype_df[col].values for cancer in mapped_types):
                    cancer_col = col
                    cancer_types = mapped_types  # Use the mapped types
                    break
            elif any(cancer in phenotype_df[col].values for cancer in cancer_types):
                cancer_col = col
                break

    if not cancer_col:
        print("WARNING: Could not identify cancer type column. Using all samples.")
        relevant_samples = phenotype_df.index.tolist()
    else:
        print(f"Using column '{cancer_col}' to filter cancer samples")
        relevant_samples = phenotype_df[
            phenotype_df[cancer_col].isin(cancer_types)
        ].index.tolist()

    print(f"Found {len(relevant_samples)} samples for cancer types: {cancer_types}")

    # Step 2: Process expression data in small chunks
    # First read header to get column names
    print("Reading expression data header...")
    with gzip.open(input_gz_file, 'rt') as f:
        header = f.readline().strip().split('\t')

    # Find indices of columns we want to keep
    keep_indices = [0]  # Always keep gene ID column
    keep_columns = [header[0]]  # Gene ID column name

    for i, col in enumerate(header[1:], 1):
        if col in relevant_samples:
            keep_indices.append(i)
            keep_columns.append(col)

    print(f"Will keep {len(keep_indices)} columns out of {len(header)}")

    # Process the file
    gene_chunk = []
    chunk_num = 0
    gene_count = 0

    print("Beginning to process expression data...")
    with gzip.open(input_gz_file, 'rt') as f:
        # Skip header
        next(f)

        for line in f:
            gene_count += 1
            parts = line.strip().split('\t')

            # Extract only the columns we need
            gene_id = parts[0]
            values = [parts[i] for i in keep_indices[1:]]  # Skip gene ID in indices

            gene_chunk.append([gene_id] + values)

            # When chunk is full, process it
            if len(gene_chunk) >= chunk_size:
                process_chunk(gene_chunk, keep_columns, output_dir, chunk_num)
                gene_chunk = []  # Clear for next chunk
                chunk_num += 1
                print(f"Processed {gene_count} genes so far...")

        # Process final chunk if any
        if gene_chunk:
            process_chunk(gene_chunk, keep_columns, output_dir, chunk_num)
            chunk_num += 1

    print(f"Completed processing. Created {chunk_num} chunk files.")

    # Create an index file
    index_file = os.path.join(output_dir, "chunk_index.txt")
    with open(index_file, 'w') as f:
        f.write(f"total_chunks={chunk_num}\n")
        f.write(f"cancer_types={','.join(cancer_types)}\n")
        f.write(f"total_samples={len(keep_columns) - 1}\n")
        f.write(f"total_genes={gene_count}\n")

    print(f"Created index file: {index_file}")


def process_chunk(gene_chunk, columns, output_dir, chunk_num):
    """Process a chunk of gene expression data and save to parquet"""
    chunk_df = pd.DataFrame(gene_chunk, columns=columns)
    chunk_df.set_index(columns[0], inplace=True)

    # Convert to numeric values
    for col in chunk_df.columns:
        chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')

    # Transpose to get samples as rows (ML format)
    chunk_df = chunk_df.T

    # Save this chunk
    output_file = os.path.join(output_dir, f"expression_chunk_{chunk_num:04d}.parquet")
    chunk_df.to_parquet(output_file)

    # Clear memory
    del chunk_df


if __name__ == "__main__":
    # Get parameters from command line or use defaults
    input_file = sys.argv[1] if len(sys.argv) > 1 else "TcgaTargetGtex_rsem_gene_tpm.gz"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_chunks"
    phenotype_file = sys.argv[3] if len(sys.argv) > 3 else "TcgaTargetGTEX_phenotype.gz"
    cancer_types = sys.argv[4].split(',') if len(sys.argv) > 4 else ["BRCA"]

    data_path = TCGA_DATA_PATH

    # Build full paths
    input_path = os.path.join(data_path, input_file)
    phenotype_path = os.path.join(data_path, phenotype_file)
    output_path = os.path.join(data_path, output_dir)

    print(f"Processing {input_path} for cancer types: {cancer_types}")
    print(f"Using phenotype data: {phenotype_path}")
    print(f"Output directory: {output_path}")

    process_tcga_data_low_memory(
        input_path,
        output_path,
        phenotype_path,
        cancer_types
    )