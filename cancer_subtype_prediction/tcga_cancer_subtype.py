"""
cohortTCGA TARGET GTEx
Rows (samples(patients)) x COLUMNs (identifiers(Genes)) (i.e. genomicMatrix)
60,499 identifiers X 19131 samples All IdentifiersAll Samples
"""

import os
import numpy as np
import pandas as pd
from beartype.typing import List, Tuple
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import shap
import torch
import gseapy as gp
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from etl import select_protein_coding_genes, extract_X_y_from_dataframe
from roug_ml.utl.mlflow_utils import load_top_models
from etl import download_file_from_ftp
from roug_ml.utl.dowload_utils import download_file
from roug_ml.utl.paths_utl import create_dir
from roug_ml.models.hyperoptimization import parallele_hyper_optim
from roug_ml.utl.parameter_utils import generate_param_grid_with_different_size_layers
from roug_ml.utl.parameter_utils import restructure_dict
from roug_ml.utl.mlflow_utils import get_or_create_experiment
from roug_ml.models.hyperoptimization import get_best_run_from_hyperoptim
from roug_ml.utl.mlflow_utils import get_best_run, get_top_n_runs
from roug_ml.utl.evaluation.multiclass import compute_multiclass_confusion_matrix
from roug_ml.utl.evaluation.eval_utl import calc_loss_acc_val
from roug_ml.models.feature_selection import (
    SelectKBestOneHotSelector,
)

from roug_ml.utl.data_vizualization.labels_vizualization import (
    plot_label_distribution_from_arrays,
)
from roug_ml.utl.etl.transforms_utl import one_hot_to_numeric, integer_to_onehot

from cancer_subtype_prediction.configs import TCGA_DATA_PATH, TCGA_RESULTS_PATH
from cancer_subtype_prediction.etl import extract_gene_id_to_name_mapping
from views.views_utl import get_explanation_for_class


# print(TCGA_DATA_PATH)

def feature_inspection(train_df, test_sample):
    for feature in train_df.columns:
        train_values = train_df[feature]
        test_value = test_sample[feature]
        print(f"Feature: {feature}")
        print(f"Train - Mean: {train_values.mean()}, Std: {train_values.std()}")
        print(f"Test: {test_value}")


def compute_shap_values_for_ensemble(
    models: List[Tuple], X: np.ndarray, feature_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for an ensemble of models.

    :param models: each containing a feature selector and a neural network model.
    :param X: Input data for which SHAP values need to be computed.
    :param feature_names: Names of the input features.

    :return: A tuple containing the averaged SHAP values and the names of the selected features.
    """
    all_shap_values = []
    x_val = models[0][0].transform(X)
    selected_indices = models[0][0].get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = [feature_names[i] for i in selected_indices]

    # Create a SHAP explainer for each model and get SHAP values
    for model in models:
        # Extracting the nn
        nn_model = model[-1].nn_model

        # Ensure model is in evaluation mode
        nn_model.eval()

        # Convert data to PyTorch tensors
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)

        explainer = shap.DeepExplainer(nn_model, x_val_tensor.unsqueeze(1))

        shap_values = explainer.shap_values(x_val_tensor.unsqueeze(1))

        all_shap_values.append(shap_values)

    # Combine or average the SHAP values across the ensemble
    avg_shap_values = np.mean(all_shap_values, axis=0)

    return avg_shap_values, selected_feature_names


def ensemble_predict(models: List[Pipeline], data: np.ndarray) -> np.ndarray:
    """
    Make predictions using an ensemble of models and perform majority voting.

    :param models: A list of models for making predictions.
    :param data: Input data for making predictions.

    :return: An array of final predictions after majority voting.
    """
    predictions = []
    for model in models:
        pred = model.predict(data)
        predictions.append(pred)

    # Majority voting
    stacked_predictions = np.stack(predictions, axis=0)
    final_predictions = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_predictions
    )

    return final_predictions


class TCGASubtypePredictor:
    """
    A predictor for cancer subtypes using data from The Cancer Genome Atlas (TCGA).

    This class initializes a prediction environment for cancer subtype classification
    based on genomic or proteomic data. It sets up directories for results and model artifacts,
    including SHAP values for feature importance analysis, training values, and the trained
    model itself. It leverages MLflow for experiment tracking and management.

    Parameters:
    - in_mlflow_experiment_name (str): The name of the MLflow experiment under which to log runs.
    - in_cancer_types (List[str]): A list of cancer type codes to include in the analysis.
                                   Default is ["BRCA"], which stands for Breast Invasive Carcinoma.

    Attributes:
    - mlflow_experiment_name (str): Stores the name of the MLflow experiment.
    - cancer_types (List[str]): Stores the list of cancer types considered for prediction.
    - results_path (str): The path to the directory where results and artifacts are stored.
    - shap_path (str): The path to the file storing SHAP values.
    - train_path (str): The path to the file storing training data values.
    - explainer_path (str): The path to the file storing the SHAP explainer object.
    - label_mapping_path (str): The path to the file storing label mappings.
    - selected_feature_names_path (str): The path to the file storing selected feature names.
    - feature_names_path (str): The path to the file storing all feature names.
    - loaded_models: Stores the loaded models after training.
    - x_val_tensor: Tensor containing validation data.
    - selected_feature_names: List of feature names selected for the model.
    - shap_values: SHAP values computed for model explanation.
    - ensemble_selected_feature_names: Selected feature names for an ensemble model.
    - ensemble_shap_values: SHAP values for an ensemble model.
    - pipeline: The preprocessing and modeling pipeline.
    - clf: The classifier used in the pipeline.
    - type_mapping: Mapping of cancer types to numeric labels.
    - mlflow_experiment_id: The MLflow experiment ID.
    - label_mapping: Mapping of numeric labels back to cancer types.
    - re_optimize (bool): Flag to indicate whether to re-optimize model parameters.

    """
    def __init__(self, in_mlflow_experiment_name, in_cancer_types=["BRCA"]) -> None:
        """

        :param in_mlflow_experiment_name:
        :param in_cancer_types:
        """

        self.mlflow_experiment_name = in_mlflow_experiment_name
        self.cancer_types = in_cancer_types  # example: selecting both 'BRCA' and 'LUAD'
        self.results_path = os.path.join(TCGA_RESULTS_PATH, in_mlflow_experiment_name)
        create_dir(self.results_path)
        self.shap_path = os.path.join(self.results_path, "shap_values.pkl")
        self.train_path = os.path.join(self.results_path, "train_values.pkl")
        self.explainer_path = os.path.join(self.results_path, "explainer.pkl")
        self.label_mapping_path = os.path.join(self.results_path, "label_mapping.pkl")
        self.selected_feature_names_path = os.path.join(
            self.results_path, "selected_feature_names.pkl"
        )
        self.inputs_stats_summary_path = os.path.join(self.results_path, 'inputs_stats_summary.csv')

        self.feature_names_path = os.path.join(self.results_path, "feature_names.pkl")

        self.loaded_models = None
        self.x_val_tensor = None
        self.selected_feature_names = None
        self.shap_values = None
        self.ensemble_selected_feature_names = None
        self.ensemble_shap_values = None
        self.pipeline = None
        self.clf = None
        self.type_mapping = None
        self.mlflow_experiment_id = None

        self.set_mlflow_params()
        self.label_mapping = None
        self.re_optimize = False

    def set_mlflow_params(self):
        """
        Sets the tracking URI for mlflow and initializes the mlflow experiment
        """
        mlflow.set_tracking_uri("http://localhost:8000")
        # mlflow.set_tracking_uri("http://public_ip_where_ml_flow_is_running:8000") <- public ip of VM
        # do not forget to do: export PATH=$PATH:/home/ubuntu/.local/bin in VM

        # mlflow.set_tracking_uri('http://host.docker.internal:8000')
        self.mlflow_experiment_id = get_or_create_experiment(
            self.mlflow_experiment_name
        )

    def run(self):
        (
            tcga_target_gtex_samples,
            tcga_gtex_labels,
            molecular_subtype,
            survival_labels_tcga,
        ) = self.collect_data()

        gencode_release = 38
        gtf_dir = os.path.join(TCGA_DATA_PATH, "gtf_files")
        os.makedirs(gtf_dir, exist_ok=True)

        gencode_gtf_gz_filepath = os.path.join(
            gtf_dir, f"gencode.v{gencode_release}.annotation.gtf.gz"
        )

        tcga_target_gtex_samples = select_protein_coding_genes(
            self.results_path,
            in_tcga_target_gtex_samples=tcga_target_gtex_samples,
            gencode_release=gencode_release,
            in_gencode_gtf_gz_filepath=gencode_gtf_gz_filepath,
        )
        print(" ------------- After_protein_coding ---------------")
        print(tcga_target_gtex_samples.describe())
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

        # Check that cancer types to analyse are in the list
        for cancer_type in self.cancer_types:
            assert cancer_type in unique_cancer_types

        survival_labels_tcga = survival_labels_tcga[
            survival_labels_tcga["cancer type abbreviation"].isin(self.cancer_types)
        ]

        (
            merged_df,
            genes_cols,
            molecular_subtype_cols,
            survival_labels_cols,
            tcga_gtex_labels_cols,
        ) = self.preprocess_data(
            tcga_target_gtex_samples=tcga_target_gtex_samples,
            tcga_gtex_labels=tcga_gtex_labels,
            molecular_subtype=molecular_subtype,
            survival_labels_tcga=survival_labels_tcga,
        )

        print(" ------------- merged_df ---------------")
        print(merged_df.describe())

        x_train, y_train, x_val, y_val, x_test, y_test, feature_names = self.split_data(
            merged_df=merged_df, genes_cols=genes_cols
        )

        # Fit logistic regression
        pipeline = Pipeline(
            steps=[
                ("feature_selection", SelectKBest(f_classif, k=1000)),
                (
                    "estimator",
                    LogisticRegression(max_iter=10000, class_weight="balanced"),
                ),
            ]
        )
        self.clf = pipeline.fit(x_train, one_hot_to_numeric(y_train))

        if self.re_optimize:
            results = self.hyperoptimize(
                in_merged_df=merged_df,
                X_train=x_train,
                X_val=x_val,
                y_train_labels=y_train,
                y_val_labels=y_val,
            )

            best_params, best_val_accuracy, best_run_id = get_best_run_from_hyperoptim(
                results
            )

        else:
            # 4. Get the best run from mlopt
            best_run_id, best_params = get_best_run(
                self.mlflow_experiment_name, "val_accuracy"
            )

        self.pipeline = mlflow.sklearn.load_model(
            "runs:/{}/pipeline".format(best_run_id)
        )

        # 5. Validate
        self.validate(x_test, y_test, self.label_mapping)

        self.explain(x_test, feature_names)

        df_tmp = merged_df[self.selected_feature_names].copy()

        # Compute statistics for each column
        stats_df = df_tmp.describe()

        # Additional calculations for minimum and maximum values
        min_values = df_tmp.min()
        max_values = df_tmp.max()

        # Add minimum and maximum values to the statistics DataFrame
        stats_df.loc['min'] = min_values
        stats_df.loc['max'] = max_values

        stats_df.to_csv()

        stats_df.to_csv(self.inputs_stats_summary_path )

        gene_id_to_name_df = extract_gene_id_to_name_mapping(
            in_path_to_save_df=os.path.join(
                self.results_path, "all_gene_maping_Ensembl_gene_name.csv"
            ),
            in_gencode_gtf_gz_filepath=gencode_gtf_gz_filepath,
        )  # Change the in_gencode_release if necessary

        gene_id_to_name_df["Gene ID"] = (
            gene_id_to_name_df["Gene ID"].str.split(".").str[0]
        )
        gene_id_to_name_map = dict(
            zip(gene_id_to_name_df["Gene ID"], gene_id_to_name_df["Gene Name"])
        )

        for class_id in range(len(self.label_mapping)):
            print(class_id)
            sorted_feature_names = get_explanation_for_class(
                in_shap_values=self.shap_values,
                in_selected_feature_names=self.selected_feature_names,
                in_class_idx=class_id,
                in_x_val_tensor=self.x_val_tensor,
                in_results_path=self.results_path,
            )

            print(sorted_feature_names)

            # sorted_feature_names = \
            #     get_explanation_for_class(in_shap_values=self.ensemble_shap_values,
            #                               in_selected_feature_names=self.ensemble_selected_feature_names,
            #                               in_class_idx=class_id,
            #                               in_x_val_tensor=self.x_val_tensor,
            #                               in_results_path=self.results_path)

            # mapping_df = convert_and_save_mapping(ensembl_gene_id_list=sorted_feature_names,
            #                                       in_path=os.path.join(TCGA_DATA_PATH,
            #                                                            'mapping_tcga_target_gtex_to_Ensembl_Protein_ID.csv'))

            sorted_feature_names = [
                gene_id_to_name_map.get(col.split(".")[0], col)
                for col in sorted_feature_names
            ]

            if "Numeric_type" in sorted_feature_names:
                sorted_feature_names.remove("Numeric_type")

            # important_genes` is a list of important genes from SHAP analysis
            enr = gp.enrichr(
                gene_list=sorted_feature_names[:500],
                gene_sets="KEGG_2019_Human",
                outdir=os.path.join(
                    self.results_path,
                    "enrichr_kegg",
                    str(class_id) + self.label_mapping[class_id],
                ),
                cutoff=0.5,  # This is the p-value cutoff for significant pathways
            )
            enr = gp.enrichr(
                gene_list=sorted_feature_names[:500],
                gene_sets="Reactome_2016",
                outdir=os.path.join(
                    self.results_path,
                    "enrichr_Reactome",
                    str(class_id) + self.label_mapping[class_id],
                ),
                cutoff=0.5,  # This is the p-value cutoff for significant pathways
            )

            enr = gp.enrichr(
                gene_list=sorted_feature_names[:500],
                gene_sets="GO_Biological_Process_2018",
                outdir=os.path.join(
                    self.results_path,
                    "enrichr_GO",
                    str(class_id) + self.label_mapping[class_id],
                ),
                cutoff=0.5,  # This is the p-value cutoff for significant pathways
            )

            enr = gp.enrichr(
                gene_list=sorted_feature_names[:500],
                gene_sets="Cancer_Cell_Line_Encyclopedia",
                outdir=os.path.join(
                    self.results_path,
                    "enrichr_Cancer_Cell_Line_Encyclopedia",
                    str(class_id) + self.label_mapping[class_id],
                ),
                cutoff=0.5,  # This is the p-value cutoff for significant pathways
            )

            # To visualize the results
            # barplot(enr.res2d, title='KEGG', ofname='KEGG_enrichment_results')

        print("end")

    def collect_data(self):
        """
        The first step in any machine learning pipeline is data collection. This may involve
        gathering data from various sources like databases, files, APIs, web scraping, or even
        creating synthetic data.
        """
        rsem_gene_tpm_gz_file_path = os.path.join(
            TCGA_DATA_PATH, "TcgaTargetGtex_rsem_gene_tpm.gz"
        )
        rsem_gene_tpm_parquet_file_path = os.path.join(
            TCGA_DATA_PATH, "TcgaTargetGtex_rsem_gene_tpm.parquet"
        )
        phenotype_gz_file_path = os.path.join(
            TCGA_DATA_PATH, "TcgaTargetGTEX_phenotype.gz"
        )
        tcga_subtypes_file_path = os.path.join(
            TCGA_DATA_PATH, "TCGASubtype.20170308.tsv.gz"
        )
        survival_supplement_file_path = os.path.join(
            TCGA_DATA_PATH, "Survival_SupplementalTable_S1_20171025_xena_sp"
        )

        # Convert to float32, Transpose to ML style rows = samples and hdf for significantly faster
        # reading
        if not os.path.exists(rsem_gene_tpm_parquet_file_path):
            if not os.path.exists(rsem_gene_tpm_gz_file_path):
                # Download raw files from xena
                # download_file_from_url(
                #     url="https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz",
                #     destination=rsem_gene_tpm_gz_file_path,
                # )
                download_file("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_tpm.gz",
                              file_path=rsem_gene_tpm_gz_file_path)
            data = pd.read_csv(
                rsem_gene_tpm_gz_file_path, index_col=0, compression="gzip", sep="\t"
            )
            tcga_target_gtex_samples = data.T
            # Save the dataframe to a parquet file
            tcga_target_gtex_samples.to_parquet(rsem_gene_tpm_parquet_file_path)
        else:
            tcga_target_gtex_samples = pd.read_parquet(rsem_gene_tpm_parquet_file_path)

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
        print("tcga_gtex_labels")
        [print(x) for x in tcga_gtex_labels.columns]

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
        print("molecular_subtype")
        [print(x) for x in molecular_subtype.columns]

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
        print("survival_labels_tcga")

        [print(x) for x in survival_labels_tcga.columns]

        return (
            tcga_target_gtex_samples,
            tcga_gtex_labels,
            molecular_subtype,
            survival_labels_tcga,
        )

    def preprocess_data(
        self,
        tcga_target_gtex_samples,
        tcga_gtex_labels,
        molecular_subtype,
        survival_labels_tcga,
    ):
        """
        Data Preprocessing and Cleaning: Once data has been collected, it needs to be preprocessed
        and cleaned. This can involve dealing with missing values, handling outliers, correcting
        inconsistent data formats, etc.
        """
        merged_df = tcga_target_gtex_samples.join(tcga_gtex_labels, how="inner")
        tcga_gtex_labels_cols = tcga_gtex_labels.columns.tolist()
        genes_cols = tcga_target_gtex_samples.columns.tolist()
        del tcga_target_gtex_samples
        del tcga_gtex_labels

        merged_df = merged_df.join(molecular_subtype, how="inner")
        molecular_subtype_cols = molecular_subtype.columns.tolist()
        del molecular_subtype

        merged_df = merged_df.join(survival_labels_tcga, how="inner")
        survival_labels_cols = survival_labels_tcga.columns.tolist()
        del survival_labels_tcga

        merged_df = merged_df.dropna(
            subset=["Subtype_mRNA"]
            + genes_cols
            + ["cancer type abbreviation", "primary disease or tissue"]
        )

        merged_df["type_subtype"] = (
            merged_df["Subtype_mRNA"].astype(str)
            + "_"
            + merged_df["cancer type abbreviation"].astype(str)
        )

        # Identify values in 'Numeric_Labels' that appear less than 3 times
        to_remove = (
            merged_df["type_subtype"]
            .value_counts()[merged_df["type_subtype"].value_counts() < 10]
            .index
        )

        # Filter out those values from merged_df
        merged_df = merged_df[~merged_df["type_subtype"].isin(to_remove)]

        le = LabelEncoder()

        # Fit label encoder and transform labels
        merged_df["Numeric_Labels"] = le.fit_transform(merged_df["type_subtype"])
        # merged_df['Numeric_Labels'] = le.fit_transform(merged_df['Subtype_mRNA'])

        self.label_mapping = dict(zip(le.transform(le.classes_), le.classes_))

        # Saving label_mapping using pickle
        with open(self.label_mapping_path, "wb") as file:
            pickle.dump(self.label_mapping, file)

        le_type = LabelEncoder()

        # Fit label encoder and transform labels
        merged_df["Numeric_type"] = le_type.fit_transform(
            merged_df["cancer type abbreviation"]
        )

        self.type_mapping = dict(
            zip(le_type.transform(le_type.classes_), le_type.classes_)
        )

        return (
            merged_df,
            genes_cols,
            molecular_subtype_cols,
            survival_labels_cols,
            tcga_gtex_labels_cols,
        )

    def extract_features(self):
        """
        Feature Engineering and Selection: In this step, new features are created from the existing
        data which can help improve the model's performance. Feature selection is also done in this
        stage to choose the most relevant features to train the model.
        """
        pass

    def split_data(self, merged_df: pd.DataFrame, genes_cols: List[str]) -> Tuple[np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  np.ndarray,
                                                                                  List[str]]:

        """
        Splits the dataset into training, validation, and test sets and processes the target labels
        into one-hot encoded format. It also saves the feature names and training data for later use
        The function performs the following steps:
            1. Extract features and labels from the input DataFrame based on the provided gene
                columns.
            2. Splits the data into training (60%), validation (20%), and test (20%) sets.
            3. Saves the training data to a file for later use.
            4. One-hot encodes the target labels for the training, validation, and test sets
                5. Returns the split and processed data along with the feature names used.
        :param merged_df: df containing genes cols
        :param genes_cols: Ensembl Gene IDs
        :return :
         - x_train (np.ndarray): Training feature matrix.
         - y_train (np.ndarray): One-hot encoded training labels.
         - x_val (np.ndarray): Validation feature matrix.
         - y_val (np.ndarray): One-hot encoded validation labels.
         - x_test (np.ndarray): Test feature matrix.
         - y_test (np.ndarray): One-hot encoded test labels.
         - feature_names (List[str]): List of feature names (gene columns) used in the model.

        """

        # Define your dataset
        feature_names = genes_cols  # + ['Numeric_type']

        # Assuming shap_values is your computed SHAP values
        with open(self.feature_names_path, "wb") as f:
            pickle.dump(feature_names, f)

        X, y = extract_X_y_from_dataframe(merged_df, feature_names, "Numeric_Labels")

        x_train, X_temp, y_train, y_temp = train_test_split(
            X,
            merged_df["Numeric_Labels"].values,
            test_size=0.4,  # Adjust the size for your validation set
            stratify=merged_df["Numeric_Labels"].values,
            random_state=54,
        )

        # Further split the temporary set into validation and test sets
        x_test, x_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            # Half of the temporary set will become validation, and the other half will be test
            stratify=y_temp,
            random_state=42,
        )

        # Save train data for app usage
        with open(self.train_path, "wb") as file:
            pickle.dump(x_train, file)

        # Or with just one dataset:
        fig = plot_label_distribution_from_arrays(
            label_mapping=self.label_mapping,
            Train=(x_train, y_train),
            Vaidation=(x_val, y_val),
            Test=(x_test, y_test),
        )

        fig.savefig(os.path.join(self.results_path, "distribution data"))
        plt.close("all")

        y_train = integer_to_onehot(y_train)
        y_val = integer_to_onehot(y_val)
        y_test = integer_to_onehot(y_test)

        return x_train, y_train, x_val, y_val, x_test, y_test, feature_names

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

    def hyperoptimize(self, in_merged_df, X_train, X_val, y_train_labels, y_val_labels):
        """
        Model Optimization and Hyperparameter Tuning: The selected model is further optimized to
        improve its performance. This is usually done by tuning its hyperparameters. Methods such
        as Grid Search, Random Search, or Bayesian Optimization are used for this purpose.
        """
        from sklearn.utils.class_weight import compute_class_weight

        print(y_train_labels)

        # Assuming y is a one-hot encoded numpy array
        y_labels = np.argmax(
            y_train_labels, axis=1
        )  # Convert one-hot encoded labels to class labels

        # Compute class weights
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_labels), y=y_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        # Define the loss function with class weights
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        nn_key = ["CNN"]  #'MLP']#'CNN', 'MLP']

        input_shape = [X_train.shape[1]]
        output_shape = [len([type(x) for x in in_merged_df.Numeric_Labels.unique()])]
        batch_size = [
            32,
            64,
            # 1
        ]
        # cost_function = [torch.nn.CrossEntropyLoss(weight=class_weights)]
        cost_function = [torch.nn.CrossEntropyLoss()]
        learning_rate = [0.001, 0.0001]  # 0.001,
        n_epochs = [30]  # , 10
        metrics = ["accuracy"]
        layer_sizes = [
            [200, 300],
            [20, 40],
            # [20, 40, 150, 100, 50],
            # [50, 100, 300, 100, 50],
            # [100, 200, 400, 200, 100],
            # [100, 200],
            # [400, 500],
            [300, 200, 100],
        ]
        activations = [
            ["identity", "relu"],
            # ['identity', 'tanh'],
            ["identity", "relu", "tanh"],
            # ['identity', 'relu', 'tanh', 'identity', 'identity'],
            #  ['identity', 'tanh', 'identity', 'identity', 'identity']
        ]
        cnn_filters = [1, 3]

        # n_models = [1]

        list_params = generate_param_grid_with_different_size_layers(
            nn_key,
            input_shape,
            output_shape,
            batch_size,
            cost_function,
            learning_rate,
            n_epochs,
            metrics,
            layer_sizes,
            activations,
            cnn_filters,
        )

        nn_params_keys = ["activations", "in_nn", "input_shape", "output_shape"]
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
            x_train=X_train,
            y=y_train_labels,
            x_val=X_val,
            y_val=y_val_labels,
            param_grid_outer=list_params,
            in_framework="torch",
            model_save_path=self.results_path,
            in_mlflow_experiment_id=self.mlflow_experiment_id,
            use_kfold=False,
            in_scaler=None,
            # in_selector=ModelBasedOneHotSelector(LinearSVC(penalty="l1",
            #                                             dual=False,
            #                                             random_state=42)
            #                                   )
            in_selector=SelectKBestOneHotSelector(SelectKBest(f_classif, k=1000)),
            # in_selector=AutoencoderFeatureSelector(1000, epochs=40)
        )

        return results

    def validate(self, x_test, y_test, class_labels) -> None:
        """
        Validates a model using the validation data.
        The function loads the best model obtained during the training phase, then uses it to make
        predictions on the validation data. It then calculates the validation accuracy and the
        confusion matrix.

        :param x_test: The validation feature vectors reshaped for the model's expected input.
        :param y_test: The validation labels corresponding to the validation features.
        :param class_labels: Dictionary of class labels for confusion matrix tick labels

        Prints:
        val_acc (float): The validation accuracy.

        Computes:
        Confusion matrix for the given predictions and targets using the defined activity labels.
        """

        N = 5
        top_n_runs = get_top_n_runs(self.mlflow_experiment_name, "val_accuracy", N)
        self.loaded_models = load_top_models(top_n_runs)
        final_predictions = ensemble_predict(self.loaded_models, x_test)

        predictions = self.pipeline.predict(x_test)

        predictions_clf = self.clf.predict(x_test)
        predictions_clf_one_hot = integer_to_onehot(
            data_integer=predictions_clf, n_labels=y_test.shape[1]
        )

        # label_subset = [16, 17, 18, 19]  # replace with the labels you are interested in
        label_subset = list(
            range(y_test.shape[1])
        )  # replace with the labels you are interested in
        # Create a mask for the subset of interest
        mask = np.isin(one_hot_to_numeric(y_test), label_subset)

        # Filter based on the mask
        filtered_y_val_labels = y_test[mask]
        filtered_predictions = predictions[mask]
        filtered_predictions_clf = predictions_clf_one_hot[mask]

        filtered_final_predictions = final_predictions[mask]

        test_acc_ensemble = calc_loss_acc_val(
            filtered_final_predictions, filtered_y_val_labels
        )
        print(test_acc_ensemble)

        test_acc = calc_loss_acc_val(filtered_predictions, filtered_y_val_labels)
        print(test_acc)

        test_acc_clf = calc_loss_acc_val(
            filtered_predictions_clf, filtered_y_val_labels
        )
        print("Acc Test torch: {}".format(test_acc))
        print("Acc Test rdn forest: {}".format(test_acc_clf))
        print("Acc Test ensemble: {}".format(test_acc_ensemble))

        fig = compute_multiclass_confusion_matrix(
            targets=filtered_y_val_labels,
            outputs_list=[filtered_final_predictions, filtered_predictions_clf],
            model_names=["NN", "rnd_forest"],
            class_labels=class_labels,
        )
        # Assuming you have already called the function and obtained the 'fig' object
        fig.savefig(os.path.join(self.results_path, "confusion_matrix.png"))

    def explain(self, x_test: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Compute and plot SHAP values for the given test data.

        This function uses SHAP's DeepExplainer to compute SHAP values for the provided test data.
        It then uses the summary_plot function to visualize the importance of the features.

        :param x_test: The test data to explain. It should be compatible with the pipeline's
         transform method.
        :param feature_names: Names of all the features in the original dataset.

        :returns self.shap_values: SHAP values for each class.
        """

        # Compute SHAP values for the ensemble
        self.ensemble_shap_values, self.ensemble_selected_feature_names = (
            compute_shap_values_for_ensemble(
                self.loaded_models, X=x_test, feature_names=feature_names
            )
        )
        # Use the feature selector
        x_val = self.pipeline[0].transform(x_test)
        selected_indices = self.pipeline[0].get_support(indices=True)

        # Get the names of the selected features
        self.selected_feature_names = [feature_names[i] for i in selected_indices]

        # Saving selected_feature_names using pickle
        with open(self.selected_feature_names_path, "wb") as file:
            pickle.dump(self.selected_feature_names, file)

        # Extracting the nn
        nn_model = self.pipeline[-1].nn_model

        # Ensure model is in evaluation mode
        nn_model.eval()

        # Convert data to PyTorch tensors
        self.x_val_tensor = torch.tensor(x_val, dtype=torch.float32)

        explainer = shap.DeepExplainer(nn_model, self.x_val_tensor.unsqueeze(1))

        with open(self.explainer_path, "wb") as file:
            pickle.dump(explainer, file)

        self.shap_values = explainer.shap_values(self.x_val_tensor.unsqueeze(1))

        # Assuming shap_values is your computed SHAP values
        with open(self.shap_path, "wb") as f:
            pickle.dump(self.shap_values, f)

    def deploy(self):
        """
        Model Deployment: After validation, the model is deployed in the real-world environment.
        This could be a server, a cloud-based platform, or directly embedded into an application.
        """
        pass

    def maintenance(self):
        """
        Model Monitoring and Maintenance: After deployment, the model's performance is monitored
        over time. If the model's performance degrades, it may need to be retrained or replaced.
        """
        pass


if __name__ == "__main__":
    # c_types = ['GBM', 'OV', 'LUAD', 'LUSC', 'PRAD', 'UCEC', 'BLCA', 'TGCT', 'ESCA',
    #            'PAAD', 'KIRP', 'LIHC', 'CESC', 'SARC', 'BRCA', 'THYM', 'MESO',
    #            'COAD', 'STAD', 'SKCM', 'CHOL', 'KIRC', 'THCA', 'HNSC', 'LAML',
    #            'READ', 'LGG', 'DLBC', 'KICH', 'UCS', 'ACC', 'PCPG', 'UVM']
    c_types = ["BRCA"]
    analysis = TCGASubtypePredictor(
        in_mlflow_experiment_name="TCGA_BRCA_vf_290", in_cancer_types=c_types
    )
    analysis.run()
