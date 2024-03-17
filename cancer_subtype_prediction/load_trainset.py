from Bio import Entrez

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")  # Download the necessary tokenizer models


def filter_abstract(abstract, keyword):
    """
    Extracts sentences that contain the specified keyword from the abstract.

    Parameters:
        abstract (str): The text of the abstract.
        keyword (str): Keyword to search for within the abstract.

    Returns:
        str: Filtered abstract containing only relevant sentences.
    """
    sentences = sent_tokenize(abstract)
    filtered_sentences = [sentence for sentence in sentences if keyword in sentence]
    return " ".join(filtered_sentences)


def search_pubmed(query):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(pmid_list):
    # Filter out any invalid PMIDs (e.g., empty strings, None)
    valid_pmids = [pmid for pmid in pmid_list if pmid and pmid.isdigit()]

    # Check if the filtered list of PMIDs is empty
    if not valid_pmids:
        print("No valid PMIDs provided.")
        return []

    abstracts = []
    handle = Entrez.efetch(db="pubmed", id=",".join(valid_pmids), retmode="xml")
    records = Entrez.read(handle)
    for article in records.get(
        "PubmedArticle", []
    ):  # Use .get to avoid KeyError if 'PubmedArticle' is missing
        try:
            abstract_text = article["MedlineCitation"]["Article"]["Abstract"][
                "AbstractText"
            ][0]
        except KeyError:
            abstract_text = "No abstract available."
        abstracts.append(abstract_text)
    return abstracts


Entrez.email = "hector.m.romero.ugalde@gmail.com"  # Always provide your email
# # Example usage


# list_of_genes = ['KRT17', 'FGFBP1', 'KRT5', 'FABP7', 'SYT9', 'GFRA1', 'ANXA8L1', 'STAC2', 'HAS3', 'PPP1R14C', 'SOSTDC1', 'TRIM29', 'KRT16', 'PI3', 'SOX10', 'BBOX1', 'PGLYRP2', 'TP63', 'NOVA1', 'CEACAM5', 'FAT2', 'IRX1', 'DEFB132', 'NSG1', 'PROM1', 'CTB-50L17.14', 'SCUBE2', 'SFRP1', 'OSR1', 'ACE2', 'MIA', 'GABRP', 'WNT6', 'SLC7A2', 'KCNH1', 'PTX3', 'STMND1', 'TTYH1', 'ESYT3', 'TCEAL5', 'NKAIN1', 'TFF1', 'TPSG1', 'ACOX2', 'CXCL5', 'VGLL1', 'SERPINA5', 'F7', 'SLC26A3', 'GRPR']
#
# for gene in list_of_genes:
#     query = f"{gene} cancer"
#     pmids = search_pubmed(query)
#     prompt = f"Prompt: What is the biological function of gene {gene} in cancer?"
#     # Example usage
#     abstracts = fetch_abstracts(pmids)
#     # all_abstract = []
#     # for pmid, abstract in zip(pmids, abstracts):
#     #     print(f"PMID: {pmid}\nAbstract: {abstract}\n")
#     #     all_abstract.append(abstract)
#
#     with open("/Users/hector/cancer_subtype_prediction/cancer_subtype_prediction/training_file.txt", "w") as file:
#         for abstract in abstracts:
#             file.write(f"{prompt}\nResponse: {abstract}\n\n")

list_of_genes = [
    # "KRT17",
    # "FGFBP1",
    # "KRT5",
    # "FABP7",
    # "SYT9",
    # "GFRA1",
    # "ANXA8L1",
    # "STAC2",
    # "HAS3",
    # "PPP1R14C",
    # "SOSTDC1",
    # "TRIM29",
    # "KRT16",
    # "PI3",
    # "SOX10",
    # "BBOX1",
    # "PGLYRP2",
    # "TP63",
    # "NOVA1",
    # "CEACAM5",
    # "FAT2",
    # "IRX1",
    # "DEFB132",
    # "NSG1",
    # "PROM1",
    # "CTB-50L17.14",
    # "SCUBE2",
    # "SFRP1",
    # "OSR1",
    # "ACE2",
    # "MIA",
    # "GABRP",
    # "WNT6",
    # "SLC7A2",
    # "KCNH1",
    # "PTX3",
    "STMND1",
    "TTYH1",
    "ESYT3",
    "TCEAL5",
    "NKAIN1",
    "TFF1",
    "TPSG1",
    "ACOX2",
    "CXCL5",
    "VGLL1",
    "SERPINA5",
    "F7",
    "SLC26A3",
    "GRPR",
]

for gene in list_of_genes:
    query = f"{gene} cancer"
    pmids = search_pubmed(query)
    prompt = f"Prompt: What is the biological function of gene {gene} in cancer?"
    abstracts = fetch_abstracts(pmids)

    with open(
        "/Users/hector/cancer_subtype_prediction/cancer_subtype_prediction/training_file.txt",
        "a",
    ) as file:  # Open in append mode
        for abstract in abstracts:
            # Example usage
            keyword = gene
            filtered_abstract = filter_abstract(abstract, keyword)
            if len(filtered_abstract) == 0:
                continue
            file.write(f"{prompt}\nResponse: {filtered_abstract}\n\n")
