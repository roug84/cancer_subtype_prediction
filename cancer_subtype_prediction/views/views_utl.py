"""
This script contains utl function used in views
"""

import logging

log = logging.getLogger(__name__)  # noqa: E402

logging.basicConfig(level=logging.INFO)

import os
import pickle
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from beartype.typing import Tuple, List
from mlflow import MlflowClient
from werkzeug import datastructures
import mlflow
from configs import TCGA_DATA_PATH, data_path
from etl import select_protein_coding_genes

ALLOWED_EXTENSIONS = {
    "csv",
    "parquet",
}


def load_data(file_x: datastructures.FileStorage) -> pd.DataFrame:
    """
    Load file to dataframe
    :param file_x: selected file
    :return: loaded dataframe
    """
    return pd.read_parquet(file_x)


def select_protein_coding_genes_deploy(input_df: pd.DataFrame, results_path: str) -> pd.DataFrame:
    """
    Select protein coding genes
    :param input_df: dataframe with genes
    :param results_path: path where list of protein coding genes is stored
    :return: dataframe with only protein coding genes
    """
    gencode_release = 38
    gtf_dir = os.path.join(TCGA_DATA_PATH, "gtf_files")
    destination = os.path.join(gtf_dir, f"gencode.v{gencode_release}.annotation.gtf.gz")
    log.info("select protein coding genes")
    protein_coding_genes = select_protein_coding_genes(
        results_path=results_path,
        in_tcga_target_gtex_samples=input_df,
        gencode_release=gencode_release,
        in_gencode_gtf_gz_filepath=destination,
    )
    return protein_coding_genes


def load_explainer_label_mapping_selected_feature_names_feature_names(
    res_path: str,
) -> Tuple:
    """
    Load saved shap_values, label mapping, etc
    :param res_path: path where shap_values, label_mapping, selected_feature_names and feature_names
    are stored
    :return: shap_values, label_mapping, selected_feature_names and feature_names
    """
    shap_values_path = os.path.join(res_path, "shap_values.pkl")

    explainer_path = os.path.join(res_path, "explainer.pkl")

    label_mapping_path = os.path.join(res_path, "label_mapping.pkl")

    selected_feature_names_path = os.path.join(res_path, "selected_feature_names.pkl")

    feature_names_path = os.path.join(res_path, "feature_names.pkl")

    inputs_stats_summary_path = os.path.join(res_path, 'inputs_stats_summary.csv')

    # Load shap values

    with open(shap_values_path, "rb") as f:
        shap_values = pickle.load(f)

    with open(explainer_path, 'rb') as f:
        explainer = pickle.load(f)

    # Read the label_mapping using pickle
    with open(label_mapping_path, "rb") as file:
        loaded_label_mapping = pickle.load(file)

    # Read the selected_feature_names using pickle
    with open(selected_feature_names_path, "rb") as file:
        selected_feature_names = pickle.load(file)

    # Read the feature_names using pickle
    with open(feature_names_path, "rb") as file:
        feature_names = pickle.load(file)

    inputs_stats_summary_df = pd.read_csv(inputs_stats_summary_path)

    return shap_values, loaded_label_mapping, selected_feature_names, feature_names, explainer, \
           inputs_stats_summary_df


def load_gene_id_to_name_map(results_path: str) -> dict:
    """
    Load and process gene mapping data from a CSV file.
    :param results_path: The path to the directory containing the CSV file.
    :return: A dictionary mapping Gene IDs to Gene Names.
    """
    log.info("Loading Ensembl to gene name mapping")
    gene_id_to_name_df = pd.read_csv(
        os.path.join(results_path, "all_gene_maping_Ensembl_gene_name.csv")
    )

    log.info("Removing version .")
    gene_id_to_name_df["Gene ID"] = gene_id_to_name_df["Gene ID"].str.split(".").str[0]

    log.info("Creating dict for mapping")
    gene_id_to_name_map = dict(
        zip(gene_id_to_name_df["Gene ID"], gene_id_to_name_df["Gene Name"])
    )

    return gene_id_to_name_map


def get_explanation_for_class(
    in_shap_values,
    in_selected_feature_names,
    in_class_idx: int,
    in_x_val_tensor: np.ndarray,
    in_results_path: None or str = None,
) -> None or List:
    """
    Visualize SHAP values for a specific class.
    This function plots the SHAP values for a specified class (in_class_idx).
    The SHAP values are visualized using the summary_plot function.
    :param in_shap_values: shap values
    :param in_selected_feature_names: Features selected by the first step of the predictor
    :param in_class_idx: Predicted label we want to explain
    :param in_x_val_tensor: Input tensor for which we are calculating SHAP values.
    :param in_results_path: Path where the SHAP plot should be saved. If None, no plot is saved.

    :return sorted_feature_names: list of features
    """

    # SHAP values for a class:
    shap_values_for_class = in_shap_values[in_class_idx]

    # Sum over the channel dimension to get shape
    shap_values_for_class = shap_values_for_class.sum(axis=1)

    # fig, ax = plt.subplots(figsize=(12, 8))
    #
    # # Plot
    # shap.summary_plot(shap_values_for_class, in_x_val_tensor,
    #                   feature_names=in_selected_feature_names,
    #                   show=False)
    #
    # if in_results_path is not None:
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(in_results_path, 'shap_for_id_' + str(in_class_idx) + '.png'))
    #     plt.close('all')

    # # Calculate the mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values_for_class).mean(axis=0)
    #
    # # Get the indices that would sort the mean SHAP values in descending order
    sorted_indices = np.argsort(mean_shap_values)[::-1]
    #
    # # Extract the feature names in descending order of importance
    sorted_feature_names = [in_selected_feature_names[i] for i in sorted_indices]

    return sorted_feature_names


def get_best_run(experiment_name: str, metric_key: str) -> Tuple[str, dict]:
    """
    Retrieves the best run and its parameters from a specified experiment.
    :param experiment_name: Name of the experiment
    :param metric_key: Key of the metric to use for determining the best run. The best run is
     determined by ordering the runs by this metric in descending order and picking the first one.
    :return: A tuple containing the ID of the best run and the parameters of the best run.
    :raises ValueError: If no such experiment exists.
    """
    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Search for the best run in the experiment
    runs = client.search_runs(
        [experiment.experiment_id], order_by=[f"metric.{metric_key} DESC"]
    )

    # Assuming the first run is the best one
    best_run = runs[0]
    # retrieve the best run_id
    o_best_run_id = best_run.info.run_id
    # retrieve the best parameters
    o_best_params = best_run.data.params

    return o_best_run_id, o_best_params


def load_model(
    docker: bool = True, mlflow_experiment_name: str = "TCGA_BRCA_vf_4"
) -> Pipeline:
    """
    Load ML pipeline already trained
    :param docker: True is you are running into a docker container
    :param mlflow_experiment_name: mlflow experiment name
    :return: ML Pipeline
    """
    log.info("get_best_run")

    if not docker:
        # Load the best model from mlflow
        log.info("connecting to mlflow")
        mlflow.set_tracking_uri("http://localhost:8000")

        best_run_id, best_params = get_best_run(mlflow_experiment_name, "val_accuracy")
        return mlflow.sklearn.load_model("runs:/{}/pipeline".format(best_run_id))
        # model_path = \
        #     os.path.join(
        #         data_path,
        #         "artifacts/artifacts/1/e0f3b26e080143f88a09f434b6d23641/artifacts/pipeline"
        #     )
        # if os.path.exists(model_path):
        #     log.info(f"Model path {model_path} exists.")
        #     return mlflow.sklearn.load_model(model_path)
    else:
        # mlflow.set_tracking_uri('http://host.docker.internal:8000')
        model_path = "data/artifacts/artifacts/1/e0f3b26e080143f88a09f434b6d23641/artifacts/pipeline"
        if os.path.exists(model_path):
            log.info(f"Model path {model_path} exists.")
            return mlflow.sklearn.load_model(model_path)
        else:
            log.error(f"Model path {model_path} does not exist!")
            raise FileNotFoundError(f"Model path {model_path} does not exist!")


def allowed_file(filename: str) -> bool:
    """
    Check if file is in one of the accepted formats
    :param filename: name of file to check extension
    :return: True is extension is in allowed extensions
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
