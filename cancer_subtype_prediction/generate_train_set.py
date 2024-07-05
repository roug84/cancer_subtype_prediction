import os
from generate_train_set_copy import PubMedAbstractFetcher
from configs import BIO_LLM_PATH
from etl import create_dir
# Example usage
if __name__ == "__main__":
    list_of_genes = [
        "BRCA"
    ]
    data_path = os.path.join(BIO_LLM_PATH, 'cancer_description')
    create_dir(data_path)
    output_file_path = os.path.join(data_path, "training_file12.txt")
    email = "hector.m.romero.ugalde@gmail.com"

    exclude_phrases = [
        "In this study", "This study", "We show", "We demonstrated", "Here, we",
        "Our findings", "Our results", "We found", "We investigate", "We explored",
        "We assessed", "It was shown", "This paper", "This article", "This report", "we"
    ]

    include_patterns = [
        "{}"
    ]

    prompt_template = "Prompt: {}"
    fetcher_genes = PubMedAbstractFetcher(
        in_email=email,
        in_output_file_path=output_file_path,
        in_query_list=list_of_genes,
        in_prompt_template=prompt_template,
        in_exclude_phrases=exclude_phrases,
        in_include_patterns=include_patterns
    )
