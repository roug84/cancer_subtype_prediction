import os

import numpy as np
import pandas as pd
import wget
import requests

import ftplib

from beartype.typing import List, Tuple
import gzip

from roug_ml.utl.paths_utl import create_dir
# from roug_ml.utl.dowload_utils import download_file


def gunzip_file(in_source_path: str, in_destination_path: str) -> None:
    """
    Decompresses a gzip file from source_path and writes the decompressed data to dest_path.

    :param in_source_path: The path of the gzip file to decompress.
    :param in_destination_path: The path where the decompressed file will be saved.
    :return: None
    """
    with gzip.open(in_source_path, "rb") as src, open(in_destination_path, "wb") as dst:
        dst.writelines(src)


def download_file_from_url(url: str, destination: str) -> None:
    """
    Downloads a file from a specified URL to a in_gencode_gtf_gz_filepath.

    :param url: URL of the file to be downloaded.
    :param destination: Destination path to save the downloaded file.
    """
    wget.download(url, destination)


def extract_gene_names_from_gtf(gtf_file_path: str) -> List[str]:
    """
    Extracts protein-coding gene names from a specified GENCODE GTF file.
    Each line in the GTF file represents an annotation, with fields separated by tabs. The
    eighth field (index 7) contains a semicolon-separated list of key-value pairs providing
    detailed annotations for the feature. This function searches for the "gene_type"
    key to identify protein-coding genes, and then extracts the associated "gene_name" values.

    :param gtf_file_path: Path to the GENCODE GTF file.
    :return: List of unique protein-coding gene names.
    """
    gene_names = set()

    with open(gtf_file_path, "r") as file:
        for line in file:
            if line.startswith("#"):
                # Skip header lines that start with '#'.
                continue

            columns = line.strip().split("\t")

            # Check if this line represents a protein-coding gene.
            if 'gene_type "protein_coding"' in columns[8]:
                gene_info = columns[8]

                # Extract the gene name from the annotations.
                for info in gene_info.split(";"):
                    info = info.strip()
                    if info.startswith("gene_name"):
                        gene_name = info.split('"')[1]
                        gene_names.add(gene_name)
                        # Stop looking through the info fields once gene_name is found.
                        break

    return list(gene_names)


def download_and_extract_protein_coding_genes(
    in_path_to_save_df: str, destination: str
) -> list:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts
    protein-coding gene names.
    :param in_path_to_save_df: Path to save the dataframe containing the gene names.
    :param destination: Path to save the dataframe containing the gene names.
    :return: List of unique protein-coding gene names.
    """
    if os.path.exists(in_path_to_save_df):
        protein_coding_genes = pd.read_csv(in_path_to_save_df)
    else:
        # Unzipping the file
        os.system(f"gunzip {destination}")

        # Extract gene names
        gtf_file_path = destination.replace(".gz", "")
        genes_list = extract_gene_names_from_gtf(gtf_file_path)
        protein_coding_genes = pd.DataFrame(genes_list, columns=["gene_name"])
        protein_coding_genes.to_csv(in_path_to_save_df)

    return protein_coding_genes


def ensembl_gene_to_protein_id(ensembl_gene_id: str) -> str:
    """
    Retrieve Ensembl protein ID for a given Ensembl gene ID.

    :param ensembl_gene_id: Ensembl gene ID to look up.
    :return: Ensembl protein ID if found, otherwise None.
    """
    base_url = "https://rest.ensembl.org"
    endpoint = f"/lookup/id/{ensembl_gene_id}?expand=1"

    response = requests.get(
        base_url + endpoint, headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        data = response.json()
        if "Transcript" in data:
            transcript_info = data["Transcript"]
            for transcript in transcript_info:
                if "Translation" in transcript:
                    protein_id = transcript["Translation"]["id"]
                    return protein_id
    return None


def convert_and_save_mapping(ensembl_gene_id_list: list, in_path: str) -> pd.DataFrame:
    """
    Convert Ensembl gene IDs to protein IDs and save the mapping to a CSV file.

    :param ensembl_gene_id_list: List of Ensembl gene IDs.
    :param in_path: Path where the resulting CSV will be saved.
    :return: DataFrame containing the mapping from Ensembl gene IDs to protein IDs.
    """
    ensembl_gene_to_protein_mapping = {}

    for ensembl_gene_id in ensembl_gene_id_list:
        print(ensembl_gene_id)
        protein_id = ensembl_gene_to_protein_id(ensembl_gene_id.split(".")[0])

        if protein_id:
            print(
                f"Ensembl Gene ID: {ensembl_gene_id} -> Ensembl Protein ID: {protein_id}"
            )
            ensembl_gene_to_protein_mapping[ensembl_gene_id] = protein_id
        else:
            # print(f"No Ensembl Protein ID found for Ensembl Gene ID: {ensembl_gene_id}")
            ensembl_gene_to_protein_mapping[ensembl_gene_id] = "Not found"

    mapping_df = pd.DataFrame.from_dict(
        ensembl_gene_to_protein_mapping, orient="index", columns=["Ensembl_Protein_ID"]
    )
    mapping_df.to_csv(in_path, index=True)
    return mapping_df


# ENSEMBL to Gene names


def download_file_from_ftp(url: str, file_path: str) -> None:
    """
    Downloads a file from an FTP URL and saves it to the specified file path.

    :param url: The FTP URL from which the file will be downloaded.
    :param file_path: The path where the downloaded file should be saved.
    :return: None
    """
    url_parts = url.split("/")
    server = url_parts[2]
    file_dir = "/".join(url_parts[3:-1])
    filename = url_parts[-1]

    with ftplib.FTP(server) as ftp:
        ftp.login()
        ftp.cwd(file_dir)

        with open(file_path, "wb") as f:
            ftp.retrbinary(f"RETR {filename}", f.write)


def extract_gene_id_to_name_mapping(
    in_path_to_save_df: str, in_gencode_gtf_gz_filepath: str
) -> pd.DataFrame:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts a
    mapping from Ensembl gene IDs to gene names.

    :param in_path_to_save_df: Path to save the dataframe containing the gene ID to name mapping.
    :param in_gencode_gtf_gz_filepath: path were .gz file is stored
    :return: DataFrame mapping Ensembl gene IDs to gene names.
    """
    if os.path.exists(in_path_to_save_df):
        df = pd.read_csv(in_path_to_save_df)
    else:
        # TODO: add a check if file in_gencode_gtf_gz_filepath exists

        # Unzipping the filec
        os.system(f"gunzip {in_gencode_gtf_gz_filepath}")

        gtf_file_path = in_gencode_gtf_gz_filepath.replace(".gz", "")
        gene_id_to_name = {}
        with open(gtf_file_path, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] == "gene":
                    info = {
                        x.split()[0]: x.split()[1].replace('"', "").replace(";", "")
                        for x in fields[8].split("; ")
                    }
                    # Remove the condition to check if gene_type is protein_coding.
                    gene_id_to_name[info["gene_id"]] = info["gene_name"]

        df = pd.DataFrame(
            list(gene_id_to_name.items()), columns=["Gene ID", "Gene Name"]
        )
        df.to_csv(in_path_to_save_df, index=False)

    return df


def select_protein_coding_genes(
    results_path,
    in_tcga_target_gtex_samples: pd.DataFrame,
    gencode_release,
    in_gencode_gtf_gz_filepath,
) -> pd.DataFrame:
    """
    Filters the provided dataframe to select only protein coding genes based on a predefined
    GENCODE release.

    Notes:
    1. This method assumes that the gene IDs in the dataframe and the GENCODE mapping can be
     suffixed with version numbers using '.' (dot). The version number suffixes are stripped
     before processing.
    2. The GENCODE release version is hardcoded within the function. To use a different release,
     modify the gencode_release variable.
    3. The method will print the number of missing columns (genes that are in the GENCODE list
    but not in the input dataframe) and display the first 10 of such columns for inspection.

    Example usage:
    df = select_protein_coding_genes(sample_df)

    :param results_path: The directory path where results will be stored.
    :param in_tcga_target_gtex_samples: A dataframe with gene IDs as columns.
    :param gencode_release: The release version of GENCODE to be used.
    :param in_gencode_gtf_gz_filepath: The path to the GENCODE annotation GTF file.

    :returns: A filtered dataframe containing only protein coding genes from the input dataframe
    """
    # replace with desired release version
    base_url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{gencode_release}/gencode.v{gencode_release}.annotation.gtf.gz"

    # Specify a directory for storing the GTF files (modify as necessary)
    # Download the file
    if not os.path.exists(in_gencode_gtf_gz_filepath):
        print(f"Downloading GENCODE {gencode_release} GTF file...")
        directory_path = os.path.dirname(in_gencode_gtf_gz_filepath)
        create_dir(directory_path)
        download_file_from_ftp(base_url, file_path=in_gencode_gtf_gz_filepath)

    gunzip_file(
        in_gencode_gtf_gz_filepath, in_gencode_gtf_gz_filepath.replace(".gz", "")
    )

    # Extract the mapping of all genes (including non-protein coding)
    all_genes_mapping = extract_gene_id_to_name_mapping(
        in_path_to_save_df=os.path.join(
            results_path, "all_genes_mapping" + str(gencode_release) + ".csv"
        ),
        in_gencode_gtf_gz_filepath=in_gencode_gtf_gz_filepath,
    )

    # Remove version: for example .1 .2 etc
    all_genes_mapping["Gene ID"] = all_genes_mapping["Gene ID"].str.split(".").str[0]

    # Filter the mapping to only get protein-coding genes
    protein_coding_genes = download_and_extract_protein_coding_genes(
        in_path_to_save_df=os.path.join(
            results_path, "protein_coding_genes" + str(gencode_release) + ".csv"
        ),
        destination=in_gencode_gtf_gz_filepath,
    )

    all_genes_mapping = all_genes_mapping[
        all_genes_mapping["Gene Name"].isin(protein_coding_genes["gene_name"])
    ]

    columns_to_keep = sorted(all_genes_mapping["Gene ID"].unique())

    in_tcga_target_gtex_samples.columns = [
        col.split(".")[0] for col in in_tcga_target_gtex_samples.columns
    ]

    # Step 1: Check overlap
    missing_columns = [
        col for col in columns_to_keep if col not in in_tcga_target_gtex_samples.columns
    ]

    print(f"Number of missing columns: {len(missing_columns)}")
    print(missing_columns[:10])  # Print the first 10 missing columns for inspection

    # Step 2: Update the list by removing the missing columns
    columns_to_keep_updated = [
        col for col in columns_to_keep if col not in missing_columns
    ]

    # Step 3: Filter the dataframe
    filtered_df = in_tcga_target_gtex_samples[columns_to_keep_updated].copy()

    return filtered_df


def extract_protein_coding_genes(
    in_path_to_save_df: str, in_gencode_release: int
) -> pd.DataFrame:
    """
    Downloads the GTF file for a specified GENCODE release version and extracts a
    mapping from Ensembl gene IDs to gene names for protein-coding genes only.

    :param in_path_to_save_df: Path to save the dataframe containing the gene ID to name mapping for protein-coding genes.
    :param in_gencode_release: GENCODE release version, e.g., "38".
    :return: DataFrame mapping Ensembl gene IDs to gene names for protein-coding genes.
    """

    # If the dataframe is already saved, just read and return it
    if os.path.exists(in_path_to_save_df):
        df = pd.read_csv(in_path_to_save_df)
    else:
        str_gencode_release = str(in_gencode_release)
        base_url = f"ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{str_gencode_release}/gencode.v{str_gencode_release}.annotation.gtf.gz"

        # Specify a directory for storing the GTF files
        gtf_dir = "gtf_files"
        os.makedirs(gtf_dir, exist_ok=True)

        destination = os.path.join(
            gtf_dir, f"gencode.v{str_gencode_release}.annotation.gtf.gz"
        )

        # Download the file
        print(f"Downloading GENCODE {str_gencode_release} GTF file...")
        download_file_from_url(base_url, in_gencode_gtf_gz_filepath=destination)
        # download_file(base_url, destination)

        # Unzipping the file
        os.system(f"gunzip {destination}")

        gtf_file_path = destination.replace(".gz", "")
        gene_id_to_name = {}

        # Parse the GTF file and extract gene names for protein-coding genes only
        with open(gtf_file_path, "r") as file:
            for line in file:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] == "gene":
                    info = {
                        x.split()[0]: x.split()[1].replace('"', "").replace(";", "")
                        for x in fields[8].split("; ")
                    }
                    if (
                        "gene_type" in info and info["gene_type"] == "protein_coding"
                    ):  # Check for protein-coding genes
                        gene_id_to_name[info["gene_id"]] = info["gene_name"]

        df = pd.DataFrame(
            list(gene_id_to_name.items()), columns=["Gene ID", "Gene Name"]
        )
        df.to_csv(in_path_to_save_df, index=False)

    return df


def extract_X_y_from_dataframe(
    dataframe: pd.DataFrame, input_cols: List[str], label_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features and labels from a dataframe.
    :param dataframe: pandas DataFrame
    :param input_cols: columns that represent the input features
    :param label_col: column that represents the label
    :return: A tuple containing the array of input features and the array of labels.
    """
    # If input_cols None then the entire dataframe is X
    if input_cols is not None:
        X = dataframe[input_cols].values.astype(float)
    else:
        X = dataframe.values.astype(float)

    if label_col is not None:
        y = dataframe[label_col].values
    else:
        y = None

    return X, y
