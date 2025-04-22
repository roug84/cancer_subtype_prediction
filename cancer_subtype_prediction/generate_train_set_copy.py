import os
from concurrent.futures import ThreadPoolExecutor
from typing import List
from Bio import Entrez
import nltk
from nltk.tokenize import sent_tokenize
from configs import BIO_LLM_PATH
from etl import create_dir

nltk.download("punkt")  # Ensure necessary tokenizer models are downloaded


def search_pubmed(in_query: str) -> List[str]:
    """
    Searches PubMed for abstracts related to the given query.

    :param in_query: The search query to use.
    :returns a list of PubMed IDs (PMIDs) for the search results.
    """
    handle = Entrez.esearch(db="pubmed", term=in_query, retmax=1000)
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
        return []

    abstracts = []
    handle = Entrez.efetch(db="pubmed", id=",".join(valid_pmids), retmode="xml")
    records = Entrez.read(handle)
    for article in records.get("PubmedArticle", []):
        try:
            abstract_text = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
        except KeyError:
            abstract_text = "No abstract available."
        abstracts.append(abstract_text)
    handle.close()
    return abstracts


def filter_abstract(in_abstract: str, in_exclude_phrases: List[str], in_include_patterns: List[str]) -> str:
    """
    Filters the given abstract, returning sentences that contain the specified keyword and match defined patterns.

    :param in_abstract: The text of the abstract to filter.
    :param in_exclude_phrases: phrases starting with one of the elements of this list are removed.
    :param in_include_patterns: phrases starting with one of the elements of this list are kept.
    :returns a string containing the filtered abstract.
    """
    sentences = sent_tokenize(in_abstract)
    filtered_sentences = [
        sentence
        for sentence in sentences
        if not any(sentence.startswith(phrase) for phrase in in_exclude_phrases)
           and any(pattern in sentence for pattern in in_include_patterns)
    ]
    return " ".join(filtered_sentences)


class PubMedAbstractFetcher:
    """
    A class for fetching and filtering abstracts from PubMed that discuss the biological function of genes in cancer or
    details about cancer subtypes.

    :param in_email: The email address to use with Entrez.
    :param in_output_file_path: The path to the file where filtered abstracts will be saved.
    :param in_query_list: A list of query terms (genes or cancer subtypes) for which to fetch and filter abstracts.
    :param in_prompt_template: A template for the prompt that will be used for each query.
    :param in_exclude_phrases: Phrases to exclude from the filtered abstracts.
    :param in_include_patterns: Patterns to include in the filtered abstracts.
    """

    def __init__(self, in_email: str, in_output_file_path: str, in_query_list: List[str], in_prompt_template: str,
                 in_exclude_phrases: List[str], in_include_patterns: List[str]):
        Entrez.email = in_email
        self.output_file_path = in_output_file_path
        self.query_list = in_query_list
        self.prompt_template = in_prompt_template
        self.exclude_phrases = in_exclude_phrases
        self.include_patterns = in_include_patterns
        self.run_fetcher()

    def run_fetcher(self) -> None:
        """
        Runs the process of fetching and filtering abstracts for the list of queries. Saves the filtered abstracts to
         the output file.
        """
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.process_query, self.query_list)

    def process_query(self, query_item: str) -> None:
        """
        Processes a single query item by searching PubMed, fetching abstracts, and filtering the abstracts.

        :param query_item: The query term (gene or cancer subtype) to process.
        """

        query = f"{query_item}"
        pmids = search_pubmed(in_query=query)
        if not pmids:
            return

        prompt = self.prompt_template.format(query_item)
        abstracts = fetch_abstracts(pmids)
        if not abstracts:
            return

        with open(self.output_file_path, "a") as file:  # Open in append mode
            for abstract in abstracts:
                filtered_abstract = filter_abstract(
                    in_abstract=abstract,
                    in_exclude_phrases=self.exclude_phrases,
                    in_include_patterns=[pattern.format(query_item) for pattern in self.include_patterns]
                )
                if len(filtered_abstract) == 0:
                    continue
                file.write(f"{prompt}\nResponse: {filtered_abstract}\n\n")


# Example usage
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
    data_path = os.path.join(BIO_LLM_PATH, 'tcga_explain')
    create_dir(data_path)
    output_file_path = os.path.join(data_path, "training_file9.txt")
    email = "hector.m.romero.ugalde@gmail.com"

    exclude_phrases = [
        "In this study", "This study", "We show", "We demonstrated", "Here, we",
        "Our findings", "Our results", "We found", "We investigate", "We explored",
        "We assessed", "It was shown", "This paper", "This article", "This report", "we"
    ]

    include_patterns = [
        "{} is", "{} plays", "{} role", "{} expression", "The role of {}",
        "The function of {}", "Impact of {}", "Significance of {}",
        "{} in cancer", "{} contributes"
    ]

    prompt_template = "Prompt: What is the biological function of gene {} in cancer?"
    fetcher_genes = PubMedAbstractFetcher(
        in_email=email,
        in_output_file_path=output_file_path,
        in_query_list=list_of_genes,
        in_prompt_template=prompt_template,
        in_exclude_phrases=exclude_phrases,
        in_include_patterns=include_patterns
    )
