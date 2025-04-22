# load_processed_data.py
import os
import pandas as pd
import glob


def load_processed_chunks(chunks_dir, max_chunks=None):
    """
    Load processed data chunks and combine them.

    Args:
        chunks_dir: Directory containing the chunk files
        max_chunks: Maximum number of chunks to load (for testing)

    Returns:
        Combined DataFrame with all the expression data
    """
    # Get list of all chunk files
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "expression_chunk_*.parquet")))

    if max_chunks:
        chunk_files = chunk_files[:max_chunks]

    print(f"Found {len(chunk_files)} chunk files")

    # Read and combine chunks
    dfs = []
    for file in chunk_files:
        print(f"Loading {file}...")
        df = pd.read_parquet(file)
        dfs.append(df)

    # Combine all chunks
    print("Combining chunks...")
    combined_df = pd.concat(dfs, axis=1)

    print(f"Combined data shape: {combined_df.shape}")
    return combined_df


def modify_tcga_predictor_class():
    """
    Code to modify your TCGASubtypePredictor class to use chunked data instead.
    This is just a template - you'll need to adapt it to your specific class.
    """
    return """
    def collect_data(self):
        # Modified collect_data method for TCGASubtypePredictor
        chunks_dir = os.path.join(TCGA_DATA_PATH, "processed_chunks")

        # Check if chunks have been processed
        if not os.path.exists(chunks_dir) or not os.path.exists(os.path.join(chunks_dir, "chunk_index.txt")):
            print("ERROR: You need to run process_expression_data.py first to preprocess the data")
            sys.exit(1)

        # Load phenotype data
        phenotype_gz_file_path = os.path.join(TCGA_DATA_PATH, "TcgaTargetGTEX_phenotype.gz")
        if not os.path.exists(phenotype_gz_file_path):
            download_file(
                in_url="https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGTEX_phenotype.txt.gz",
                file_path=phenotype_gz_file_path,
            )

        tcga_gtex_labels = pd.read_table(
            phenotype_gz_file_path,
            compression="gzip",
            header=0,
            sep="\t",
            encoding="ISO-8859-1",
            index_col=0,
            dtype="str",
        ).sort_index(axis="index")

        # Load processed expression data
        tcga_target_gtex_samples = load_processed_chunks(chunks_dir)

        # Rest of data loading proceeds as before
        tcga_subtypes_file_path = os.path.join(TCGA_DATA_PATH, "TCGASubtype.20170308.tsv.gz")
        survival_supplement_file_path = os.path.join(
            TCGA_DATA_PATH, "Survival_SupplementalTable_S1_20171025_xena_sp"
        )

        # Load molecular subtype data
        if not os.path.exists(tcga_subtypes_file_path):
            download_file(
                in_url="https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGASubtype.20170308.tsv.gz",
                file_path=tcga_subtypes_file_path,
            )

        molecular_subtype = pd.read_table(
            tcga_subtypes_file_path,
            compression="gzip",
            header=0,
            sep="\t",
            encoding="ISO-8859-1",
            index_col=0,
            dtype="str",
        ).sort_index(axis="index")

        # Load survival data
        if not os.path.exists(survival_supplement_file_path):
            download_file(
                in_url="https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp",
                file_path=survival_supplement_file_path,
            )

        survival_labels_tcga = pd.read_table(
            survival_supplement_file_path,
            header=0,
            sep="\t",
            encoding="ISO-8859-1",
            index_col=0,
            dtype="str",
        ).sort_index(axis="index")

        # Try to filter survival data if possible
        for col in survival_labels_tcga.columns:
            if any(cancer in col.lower() for cancer in ['cancer', 'type']):
                # Check if this column contains our cancer types
                if any(cancer in survival_labels_tcga[col].values for cancer in self.cancer_types):
                    survival_labels_tcga = survival_labels_tcga[
                        survival_labels_tcga[col].isin(self.cancer_types)
                    ]
                    break

        return (
            tcga_target_gtex_samples,
            tcga_gtex_labels,
            molecular_subtype,
            survival_labels_tcga,
        )
    """