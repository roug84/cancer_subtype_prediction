import logging

log = logging.getLogger(__name__)  # noqa: E402

logging.basicConfig(level=logging.INFO)


from views.views_utl import (
    load_data,
    select_protein_coding_genes_deploy,
    load_explainer_label_mapping_selected_feature_names_feature_names,
    load_gene_id_to_name_map,
    get_explanation_for_class,
    load_model,
    allowed_file,
)

import pandas as pd
from beartype.typing import List, Tuple
import os
import shap
import pickle
import numpy as np
import torch
from werkzeug import datastructures
from sklearn.pipeline import Pipeline
from flask import Blueprint, render_template, redirect, flash
from flask import request
import re

from etl import extract_X_y_from_dataframe
from configs import TCGA_RESULTS_PATH
import gseapy as gp


unique_cancer_types = [
    "GBM",
    "OV",
    "LUAD",
    "LUSC",
    "PRAD",
    "UCEC",
    "BLCA",
    "TGCT",
    "ESCA",
    "PAAD",
    "KIRP",
    "LIHC",
    "CESC",
    "SARC",
    "BRCA",
    "THYM",
    "MESO",
    "COAD",
    "STAD",
    "SKCM",
    "CHOL",
    "KIRC",
    "THCA",
    "HNSC",
    "LAML",
    "READ",
    "LGG",
    "DLBC",
    "KICH",
    "UCS",
    "ACC",
    "PCPG",
    "UVM",
]
in_cancer_types = ["BRCA"]

UPLOAD_FOLDER = "/Users/hector/DiaHecDev/data"
mlflow_experiment_name: str = "TCGA_BRCA_vf_250"
docker = False

bp10 = Blueprint("bp10", __name__, template_folder="templates")


def predict_deployed(
    input_df: pd.DataFrame,
    pipeline: Pipeline,
    results_path: str,
    feature_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Predicts the cancer subtype based on gene expression data and returns prediction results
    along with the preprocessed data.
    This function loads the gene expression data from a file, preprocesses it by extracting
    protein-coding genes, and then uses a trained machine learning pipeline to predict the
    cancer subtypes. It also ensures that the data contains all required features for the prediction
    :param input_df: The file containing gene expression data to predict.
    :param pipeline: The trained machine learning pipeline for prediction.
    :param results_path: Path to the directory where model and preprocessing artifacts are stored.
    :param feature_names: List of feature names (genes) expected by the model.
    :return:
    - predictions: The predicted cancer subtypes.
    - df_protein_coding_genes_ordered: The preprocessed dataframe with ordered protein coding genes.
    - x_test: The feature matrix used for prediction after preprocessing.
    """
    log.info("Preparing data")

    df_protein_coding_genes = select_protein_coding_genes_deploy(
        input_df=input_df, results_path=results_path
    )

    # Reorder the columns in in_df based on feature_names
    df_protein_coding_genes_ordered = df_protein_coding_genes[feature_names]

    x_test, _ = extract_X_y_from_dataframe(df_protein_coding_genes_ordered, None, None)
    predictions = pipeline.predict(x_test)

    return predictions, df_protein_coding_genes_ordered, x_test


def explain_deployed(
    df_protein_coding_genes_ordered: pd.DataFrame,
    feature_names: List[str],
    pipeline: Pipeline,
    predictions,
    x_test: np.ndarray,
    results_path: str,
    explainer_loaded: shap.DeepExplainer,
) -> Tuple[dict, np.ndarray]:
    """
    Generates SHAP values for the deployed model's predictions and returns the data
    for visualization.

    This function uses SHAP (SHapley Additive exPlanations) to explain the output of the
    deployed model. It identifies the contribution of each feature to the prediction for a
    single sample and selects the top N features based on their impact.
    :param df_protein_coding_genes_ordered: Preprocessed dataframe with ordered protein coding genes
    :param feature_names: List of feature names (genes) expected by the model.
    :param pipeline: The trained machine learning pipeline used for prediction.
    :param predictions: The predicted cancer subtypes for the samples.
    :param x_test: The feature matrix used for prediction after preprocessing.
    :param results_path: path where results of model training are stored
    :param explainer_loaded: A previously trained and saved SHAP explainer object loaded for
    generating SHAP values.

    :return:
    - shap_data: A dictionary containing data for plotting SHAP values, formatted for compatibility
      with plotting libraries.
    - shap_values: Raw SHAP values for all features across all predictions.
    """

    number_of_samples = len(df_protein_coding_genes_ordered)

    log.info("Loading labels mapping and features names")

    selected_indices = pipeline[0].get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = [feature_names[i] for i in selected_indices]

    # Extracting the nn
    nn_model = pipeline[-1].nn_model

    x_background = pipeline[0].transform(x_test)
    # x_backgroundl_tensor = torch.tensor(x_background, dtype=torch.float32)

    # Ensure model is in evaluation mode
    nn_model.eval()
    assert len(x_test) == number_of_samples

    i = 0
    # Use the feature selector
    x_val = pipeline[0].transform(x_test[i, :].reshape(1, -1))

    # Convert data to PyTorch tensors
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)

    #
    shap.DeepExplainer(nn_model, x_val_tensor.unsqueeze(1))

    #
    shap_values = explainer_loaded.shap_values(x_val_tensor.unsqueeze(1))

    # Generate SHAP summary plot
    shap_values_for_class = shap_values[predictions[i]]

    N = 200
    mean_abs_shap_values = np.abs(shap_values_for_class).mean(axis=0).squeeze()
    top_indices = np.argsort(mean_abs_shap_values)[-N:]  # Indices of top N features
    top_features = np.array(selected_feature_names)[top_indices]
    top_shap_values = mean_abs_shap_values[top_indices]

    gene_id_to_name_map = load_gene_id_to_name_map(results_path)
    gene_names = [
        gene_id_to_name_map.get(col.split(".")[0], col) for col in top_features
    ]
    # Create a dictionary in the desired format
    shap_data = {
        "data": [{"type": "bar", "x": gene_names, "y": top_shap_values.tolist()}],
        "layout": {"title": "SHAP Bar Plot"},
    }
    return shap_data, shap_values


def enrichment_analysis_deployed(
    in_prediction: int,
    loaded_label_mapping: dict,
    shap_values: List[np.ndarray],
    selected_feature_names: List[str],
    x_test: np.ndarray,
    results_path: str,
    in_gene_sets: str = "KEGG_2019_Human",
) -> List[str]:
    """
    Conducts enrichment analysis on the genes deemed important by SHAP analysis.

    This function performs enrichment analysis using the Enrichr API to identify
    significantly enriched pathways and biological processes among the top genes
    identified by SHAP values as being important for model predictions. It generates
    HTML representations of the Enrichr results for each cancer class considered.
    :param in_prediction: prediction done by the model
    :param loaded_label_mapping: Mapping of numerical labels to their corresponding class names.
    :param shap_values: List of SHAP values for each feature across all samples.
    :param selected_feature_names: Names of the features selected by the model.
    :param x_test: The test dataset used for prediction.
    :param results_path: Path to the directory where results and intermediate files are stored.
    :param in_gene_sets: Enrichr Library name(s). or custom defined gene_sets (dict, or gmt file).
    ["KEGG_2019_Human", "Cancer_Cell_Line_Encyclopedia", "GO_Biological_Process_2018",
    "Reactome_2016"]
    :return:
        - df_htmls: A list of strings containing the HTML representation of the
      enrichment analysis results for each considered cancer class.

    """
    log.info("Ensembl to gene id mapping")
    gene_id_to_name_map = load_gene_id_to_name_map(results_path)

    df_htmls = []
    for _ in range(1):
        print("Prediction was: ", in_prediction)
        sorted_feature_names = get_explanation_for_class(
            in_shap_values=shap_values,
            in_selected_feature_names=selected_feature_names,
            in_class_idx=in_prediction,
            in_x_val_tensor=x_test,
            in_results_path=results_path,
        )

        sorted_feature_names = [
            gene_id_to_name_map.get(col.split(".")[0], col)
            for col in sorted_feature_names
        ]

        if "Numeric_type" in sorted_feature_names:
            sorted_feature_names.remove("Numeric_type")

        enr = gp.enrichr(
            gene_list=sorted_feature_names[:200],
            gene_sets=in_gene_sets,
            outdir=os.path.join(
                results_path,
                "enrichr_kegg",
                str(in_prediction) + loaded_label_mapping[in_prediction],
            ),
            cutoff=0.2,  # This is the p-value cutoff for significant pathways
        )

        # Assuming df is your DataFrame containing the enrichment analysis results
        columns_to_keep = [
            # 'Gene_set',
            "Term",
            "Overlap",
            # 'P-value',
            "Adjusted P-value",
            "Odds Ratio",
            "Combined Score",
            "Genes",
        ]
        df_filtered = enr.results[columns_to_keep]

        df_html = df_filtered.to_html(classes="dataframe")
        df_htmls.append(df_html)

    for i, html in enumerate(df_htmls):
        df_htmls[i] = html.replace("<table", '<table id="dataframe-{}"'.format(i))

    return df_htmls


@bp10.route("/upload_file", methods=["POST", "GET"])
def view():

    if request.method == "POST":
        log.info("post")
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        pipeline = load_model(
            docker=docker, mlflow_experiment_name=mlflow_experiment_name
        )
        log.info("model loaded")

        log.info("Loading labels mapping and features names")
        results_path = os.path.join(TCGA_RESULTS_PATH, mlflow_experiment_name)
        (
            shap_values_loaded,
            loaded_label_mapping,
            selected_feature_names,
            feature_names,
            explainer_loaded,
        ) = load_explainer_label_mapping_selected_feature_names_feature_names(
            res_path=results_path
        )

        log.info("Request files")
        file_x = request.files["file"]

        log.info("Loading data")
        input_df = load_data(file_x)

        log.info("Checking data")
        # Check genes are Ensembl Gene IDs
        # Asserting that all columns have values starting with "ENSG"
        cols_ensg = [col.startswith("ENSG") for col in input_df.columns]
        cols = input_df.columns

        messages = []
        if len(cols_ensg) == len(cols):
            messages.append("Not all columns start with 'ENSG'")

        # Check if all the required columns are present in the input dataframe
        cols_without_version = [col.split(".")[0] for col in cols]
        if not all(col in cols_without_version for col in selected_feature_names):
            raise ValueError(
                "Some required columns are missing from the input dataframe."
            )

        predictions, df_protein_coding_genes_ordered, x_test = predict_deployed(
            input_df=input_df,
            pipeline=pipeline,
            results_path=results_path,
            feature_names=feature_names,
        )

        shap_html, shap_values = explain_deployed(
            df_protein_coding_genes_ordered=df_protein_coding_genes_ordered,
            feature_names=feature_names,
            pipeline=pipeline,
            predictions=predictions,
            x_test=x_test,
            results_path=results_path,
            explainer_loaded=explainer_loaded,
        )

        df_htmls = enrichment_analysis_deployed(
            in_prediction=predictions[0],
            loaded_label_mapping=loaded_label_mapping,
            shap_values=shap_values,
            selected_feature_names=selected_feature_names,
            x_test=x_test,
            results_path=results_path,
            in_gene_sets=[  # "KEGG_2019_Human",
                "Cancer_Cell_Line_Encyclopedia",
                # "GO_Biological_Process_2018",
                # "Reactome_2016"
            ],
        )

        predicted_subtypes = [
            loaded_label_mapping[prediction] for prediction in predictions
        ]
        return render_template(
            "results.html",
            df_htmls=df_htmls,
            shap_plot=shap_html,
            predictions=predicted_subtypes,
        )

        # if user does not select file, browser also
        # submit an empty part without filename
        if file_x.filename == "":
            print("file empty")
            flash("No selected file")
            return redirect(request.url)

        if file_x and allowed_file(file_x.filename):
            # filename = secure_filename(file_x.filename)
            # file_x.save(os.path.join(UPLOAD_FOLDER, filename))
            # send_test_mail()
            log.info("Printing predictions")
            return render_template("temperature.html", notes=predictions)
    return render_template("upload_file.html")
