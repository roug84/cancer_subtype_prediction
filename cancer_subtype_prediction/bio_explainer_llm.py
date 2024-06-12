import os

import mlflow
import torch
from sklearn.model_selection import train_test_split
from beartype.typing import Tuple, List, Dict

from roug_ml.utl.mlflow_utils import get_or_create_experiment
from roug_ml.utl.parameter_utils import restructure_dict
from roug_ml.models.hyperoptimization import get_best_run_from_hyperoptim
from roug_ml.models.hyperoptimization import parallele_hyper_optim
from roug_ml.utl.parameter_utils import generate_param_grid_with_different_size_layers
from roug_ml.utl.mlflow_utils import get_best_run, get_top_n_runs

# data_preparation.py
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import load_dataset

from datasets import Dataset

# model_output = "Here, we show that KIT17 is expressed in a variety of cancer tissues."
# correct_gene_names = ["KRT17", "FGFBP1", "KRT5"]  # Add more correct gene names as needed

from fuzzywuzzy import process
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def evaluate_generation(reference_texts, generated_text):
    # Assume reference_texts is a list of lists of reference texts for BLEU, and a list of strings for ROUGE and BERTScore
    bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts],
                             generated_text.split())

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {key: scorer.score(ref, generated_text) for ref in reference_texts for
                    key in scorer.rouge_types}

    # BERTScore
    predictions, _, _ = bert_score([generated_text], reference_texts, lang="en",
                                   return_hash=False)
    bertscore = predictions.mean().item()

    return {"BLEU": bleu_score, "ROUGE": rouge_scores, "BERTScore": bertscore}

def correct_gene_names_based_on_similarity(text, correct_gene_names, threshold=90):
    """
    Corrects gene names in a given text based on similarity to a list of valid gene names.

    This function uses fuzzy matching to replace words in the input text with the most similar
    gene names from a provided list. It only replaces words that have a similarity score above
    the specified threshold. If no high-similarity match is found, the original word is retained.

    Parameters:
    - text (str): The input text containing gene names to be corrected.
    - correct_gene_names (List[str]): A list of valid gene names to compare against.
    - threshold (int, optional): The minimum similarity score required to replace a word. Default is 90.

    Returns:
    - corrected_text (str): The text with gene names replaced based on similarity.
    """
    words = text.split()  # Split the text into words
    corrected_words = []

    for word in words:
        # Find the most similar gene name from the correct_gene_names list
        highest_match = process.extractOne(
            word, correct_gene_names, score_cutoff=threshold
        )

        if highest_match:
            # If a high similarity match is found, replace the word with the correct gene name
            corrected_words.append(highest_match[0])
        else:
            # If no match is found, keep the original word
            corrected_words.append(word)

    # Reconstruct the text from corrected words
    corrected_text = " ".join(corrected_words)
    return corrected_text


class LLMBioExplained:
    """
    A predictor for cancer subtypes using data from The Cancer Genome Atlas (TCGA).

    This class initializes a prediction environment for cancer subtype classification
    based on genomic or proteomic data. It sets up directories for results and model artifacts,
    including SHAP values for feature importance analysis, training values, and the trained
    model itself. It leverages MLflow for experiment tracking and management.

    :param in_mlflow_experiment_name: The name of the MLflow experiment under which to log runs.

    Attributes:
    - mlflow_experiment_name (str): Stores the name of the MLflow experiment.
    - train_path (str): The path to the file storing training data values.
    - model_path (str): The path to the file storing training data values.
    - mlflow_experiment_id: The MLflow experiment ID.
    - re_optimize (bool): Flag to indicate whether to re-optimize model parameters.

    """

    def __init__(self, in_mlflow_experiment_name) -> None:
        """
        Initialize the LLMBioExplained class.

        :param in_mlflow_experiment_name: The name of the MLflow experiment for logging runs.
        """
        self.mlflow_experiment_name = in_mlflow_experiment_name
        self.train_path = "/Users/hector/cancer_subtype_prediction/cancer_subtype_prediction/training_file.txt"
        self.model_path = "/Users/hector/cancer_subtype_prediction/cancer_subtype_prediction/gpt2_finetuned2"
        self.mlflow_experiment_id = None
        self.set_mlflow_params()
        self.re_optimize = True

    def set_mlflow_params(self) -> None:
        """
        Sets the tracking URI for mlflow and initializes the mlflow experiment
        """

        mlflow.set_tracking_uri("http://localhost:8000")
        self.mlflow_experiment_id = get_or_create_experiment(
            self.mlflow_experiment_name
        )

    def run(self) -> None:
        """
        Execute the prediction pipeline.

        This function orchestrates the collection, preprocessing, splitting, optimization,
        and model loading steps, and finally generates the predicted output.
        """
        dataset = self.collect_data()
        (tokenizer, tokenized_datasets) = self.preprocess_data(
            tokenizer_name="gpt2",
            max_length=128 * 2,
            dataset=dataset,
        )

        data, data_collator = self.split_data(
            tokenizer, tokenized_datasets, split=False
        )

        if self.re_optimize:
            results = self.hyperoptimize(data, data_collator)

            best_params, best_val_accuracy, best_run_id = get_best_run_from_hyperoptim(
                results
            )

        # Retrieve the best run ID and its parameters based on validation accuracy
        best_run_id, best_params = get_best_run(
            self.mlflow_experiment_name, "val_accuracy"
        )

        # Load the pipeline from MLflow
        pipeline = mlflow.sklearn.load_model("runs:/{}/pipeline".format(best_run_id))

        model = pipeline[
            -1
        ].nn_model  # Assume the last component of the pipeline is the model

        tokenizer = pipeline[
            -1
        ].tokenizer  # Assume the tokenizer is stored in the last component

        # Ensure the tokenizer is configured to use left padding
        tokenizer.padding_side = "left"

        # Define the prompt
        prompt = "What is the biological function of gene KRT17 in Breast cancer?"

        # Tokenize the prompt with left padding
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        # Ensure that the model and inputs are on the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate the text using the model

        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=1000,  # Total length of the output text (prompt + generated)
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

        # Decode the generated text

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print(generated_text)

        reference_texts = ["This is a sample reference text.", "Another reference text."]
        results = evaluate_generation(reference_texts, generated_text)
        print(results)

    def collect_data(self) -> Dict[str, any]:
        """
        Load data from the specified training file.

        :return: A dataset object containing the collected data.
        """
        dataset = load_dataset("text", data_files={"train": self.train_path})
        return dataset

    def preprocess_data(
        self, tokenizer_name: str, max_length: int, dataset: Dict[str, any]
    ) -> Tuple[any, Dict[str, any]]:
        """
        Preprocess data for model training.

        This involves tokenizing and converting text data to input tensors for the model.

        :param tokenizer_name: Name of the tokenizer to be used.
        :param max_length: Maximum sequence length for tokenization.
        :param dataset: The dataset object containing the text data.

        :return: A tuple containing the tokenizer and the tokenized dataset.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets.set_format(
            type="torch", columns=["input_ids", "attention_mask"]
        )

        return tokenizer, tokenized_datasets

    def extract_features(self):
        """ """
        pass

    def split_data(
        self, tokenizer: any, tokenized_datasets: Dict[str, any], split: bool
    ) -> Tuple[any, any]:
        """
        Split the dataset into training and validation subsets and prepare data collators.

        :param tokenizer: The tokenizer used for preparing data.
        :param tokenized_datasets: The dataset containing tokenized data.
        :param split: Boolean flag to determine whether to split the data into training and
           validation sets.

        :return: The training dataset and a data collator for the model.
        """

        if split:
            # Assuming a simple split for demonstration, customize as needed
            train_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)[
                "train"
            ]
            return train_dataset, DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )
        else:
            return tokenized_datasets["train"], DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )

    def model_training(self):
        """
        Model Training: In this step, different machine learning algorithms are applied to the
        training data. The choice of algorithm depends on the nature of the problem (e.g.,
        classification, regression), the data, and the business context.
        """
        pass

    def evaluation(self):
        """
        Model Evaluation and Selection: After training, models are evaluated using suitable metrics
        (accuracy, precision, recall, F1 score, ROC AUC, etc., depending on the task). The best
        performing model is then selected.
        """
        pass

    def hyperoptimize(self, data: any, data_collator: any) -> List[Dict[str, any]]:
        """
        Optimize the model's hyperparameters using parallel hyperparameter tuning.

        :param data: Training dataset for hyperparameter tuning.
        :param data_collator: Data collator used for batching the data.

        :return: Results of the hyperparameter optimization process.
        """
        nn_key = ["GPt2Net"]  # 'MLP']#'CNN', 'MLP']
        tokenizer_name = ["gpt2"]
        params = {}
        batch_size = [32]
        learning_rate = [5e-5]  # 0.001,
        n_epochs = [2]  # , 10
        # Example input shape and output shape for fine-tuning GPT-2
        input_shape = [128 * 2]  # Maximum sequence length for GPT-2 fine-tuning
        output_shape = [
            128 * 2
        ]  # Output shape is typically the same as input shape for GPT-2
        dropout = 0.1  # Number of output nodes
        attention_dropout = 0.1

        list_params = generate_param_grid_with_different_size_layers(
            nn_key=nn_key,
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            cost_function=["mse"],
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            metrics=["accuracy"],
            layer_sizes=[[]],
            activations=[[]],
            cnn_filters=None,
            dropout=dropout,
            attention_dropout=attention_dropout,
            tokenizer_name=tokenizer_name,
        )

        nn_params_keys = [
            "activations",
            "in_nn",
            "input_shape",
            "output_shape",
            "dropout",
            "attention_dropout",
            "tokenizer_name",
        ]
        other_keys = [
            "batch_size",
            "cost_function",
            "learning_rate",
            "metrics",
            "n_epochs",
            "nn_key",
        ]
        list_params = [
            restructure_dict(params, nn_params_keys, other_keys, in_from_mlflow=False)
            for params in list_params
        ]
        print(list_params)

        results = parallele_hyper_optim(
            in_num_workers=1,
            x_train=None,
            y=None,
            x_val=None,
            y_val=None,
            param_grid_outer=list_params,
            in_framework="torch",
            model_save_path=self.model_path,
            in_mlflow_experiment_name=self.mlflow_experiment_name,
            in_mlflow_experiment_id=self.mlflow_experiment_id,
            use_kfold=False,
            dataset=data,
            data_collator=data_collator,
            in_scaler=None,
        )
        return results

    def validate(self):
        """ """
        pass

    def explain(self):
        """ """
        pass

    def deploy(self):
        """ """
        pass

    def maintenance(self):
        """ """
        pass


if __name__ == "__main__":
    analysis = LLMBioExplained(in_mlflow_experiment_name="llm_bio_explainer_110")
    analysis.run()
