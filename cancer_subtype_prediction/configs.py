"""
This file contains main paths used in the project
"""

import os

root_path = os.path.abspath(os.path.join(__file__, os.pardir))

# Path to main data folder
data_path = os.path.abspath(os.path.join(root_path, "../data"))

# Path to results
RESULTS_PATH = os.path.abspath(os.path.join(root_path, "../results"))

# Path to TCGA Pan cancer
TCGA_DATA_PATH = os.path.join(data_path, "pancan-gtex-target")

# Path to save results related to this project
TCGA_RESULTS_PATH = os.path.join(RESULTS_PATH, "tcga")
