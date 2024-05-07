"""
Script used to generate training set for LLM that will explain results

"""
from Bio import Entrez
import nltk
from nltk.tokenize import sent_tokenize
from beartype.typing import List

nltk.download("punkt")  # Ensure necessary tokenizer models are downloaded


def search_pubmed(query: str) -> List[str]:
    """
    Searches PubMed for abstracts related to the given query.

    :param query: The search query to use.

    :returns a list of PubMed IDs (PMIDs) for the search results.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=1000)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def fetch_abstracts(in_pmid_list: List[str]) -> List[str]:
    """
    Fetches abstracts for the given list of PubMed IDs.

    :param in_pmid_list: A list of PubMed IDs from which to fetch abstracts.

    :returns a list of abstracts corresponding to the given PMIDs.
    """
    valid_pmids = [pmid for pmid in in_pmid_list if pmid and pmid.isdigit()]
    if not valid_pmids:
        print("No valid PMIDs provided.")
        return []

    abstracts = []
    handle = Entrez.efetch(db="pubmed", id=",".join(valid_pmids), retmode="xml")
    records = Entrez.read(handle)
    for article in records.get("PubmedArticle", []):
        try:
            abstract_text = article["MedlineCitation"]["Article"]["Abstract"][
                "AbstractText"
            ][0]
        except KeyError:
            abstract_text = "No abstract available."
        abstracts.append(abstract_text)
    return abstracts


def filter_abstract(
    abstract: str, exclude_phrases: List[str], include_patterns: List[str]
) -> str:
    """
    Filters the given abstract, returning sentences that contain the specified keyword and match
    defined patterns.

    :param abstract: The text of the abstract to filter.
    :param exclude_phrases: phrases starting with one of the elements of this list are removed.
    :param include_patterns: phrases starting with one of the elements of this list are kept.

    :returns a string containing the filtered abstract.
    """
    sentences = sent_tokenize(abstract)
    filtered_sentences = [
        sentence
        for sentence in sentences
        if not any(sentence.startswith(phrase) for phrase in exclude_phrases)
        and any(pattern in sentence for pattern in include_patterns)
    ]
    return " ".join(filtered_sentences)


class AbstractFetcher:
    """
    A class for fetching and filtering abstracts from PubMed that discuss the biological function of
    genes in cancer.
    :param in_email: The email address to use with Entrez.
    :param in_output_file_path: The path to the file where filtered abstracts will be saved.
    :param in_list_of_genes: A list of gene names for which to fetch and filter abstracts.
    """

    def __init__(
        self, in_email: str, in_output_file_path: str, in_list_of_genes: List[str]
    ):
        Entrez.email = in_email
        self.output_file_path = in_output_file_path
        self.list_of_genes = in_list_of_genes
        self.run_fetcher()

    def run_fetcher(self):
        """
        Runs the process of fetching and filtering abstracts for the list of genes.
        Saves the filtered abstracts to the output file.
        """
        #
        exclude_phrases = [
            "In this study",
            "This study",
            "We show",
            "We demonstrated",
            "Here, we",
            "Our findings",
            "Our results",
            "We found",
            "We investigate",
            "We explored",
            "We assessed",
            "It was shown",
            "This paper",
            "This article",
            "This report",
            "we",
        ]

        for gene in self.list_of_genes:
            query = f"{gene} cancer"
            pmids = search_pubmed(query)
            prompt = (
                f"Prompt: What is the biological function of gene {gene} in cancer?"
            )
            include_patterns = [
                f"{gene} is",
                f"{gene} plays",
                f"{gene} role",
                f"{gene} expression",
                f"The role of {gene}",
                f"The function of {gene}",
                f"Impact of {gene}",
                f"Significance of {gene}",
                f"{gene} in {gene} cancer",
                f"{gene} contributes",
            ]
            abstracts = fetch_abstracts(pmids)

            with open(self.output_file_path, "a") as file:  # Open in append mode
                for abstract in abstracts:
                    filtered_abstract = filter_abstract(
                        abstract,
                        exclude_phrases=exclude_phrases,
                        include_patterns=include_patterns,
                    )
                    if len(filtered_abstract) == 0:
                        continue
                    file.write(f"{prompt}\nResponse: {filtered_abstract}\n\n")


if __name__ == "__main__":
    list_of_genes = [
        "KRT17",
        "FGFBP1",
        "KRT5",
        "FABP7",
        "SYT9",
        "GFRA1",
        "ANXA8L1",
        "STAC2",
        "HAS3",
        "PPP1R14C",
        "SOSTDC1",
        "TRIM29",
        "KRT16",
        "PI3",
        "SOX10",
        "BBOX1",
        "PGLYRP2",
        "TP63",
        "NOVA1",
        "CEACAM5",
        "FAT2",
        "IRX1",
        "DEFB132",
        "NSG1",
        "PROM1",
        "CTB-50L17.14",
        "SCUBE2",
        "SFRP1",
        "OSR1",
        "ACE2",
        "MIA",
        "GABRP",
        "WNT6",
        "SLC7A2",
        "KCNH1",
        "PTX3",
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

    output_file_path = \
        "/Users/hector/cancer_subtype_prediction/cancer_subtype_prediction/training_file4.txt"
    email = "hector.m.romero.ugalde@gmail.com"
    AbstractFetcher(email, output_file_path, list_of_genes)
