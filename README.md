# Data type agnostic imputation in heterogeneous tables

In this Thesis we explore the performance of the vanilla Transformer from 2017, a Neural Machine Translation model for data type agnostic imputation of missing values in heterogeneous tables. Therefore, we conduct experiments on several different datasets with different types of missingness. Our results show, that a transformer-based model, can impute missing values without prior knowledge of the data types and performs on par with baseline methods for missing value imputation.

# Model does not need to know your data

Our model can handle mixed data types in your table without prior specification of its types or prior categorical encoding. Just run the functions from pre-processing.py on your dataset in the form of an sklearn.bunch object and pass it to the sockeye model. Further instructions in the main.py file.

# How to run the code

1. Clone the git repository on your machine
2. Install dependencies from the requirements file
3. Run the main.py file to load all specified datasets and create synthetic datasets
4. Run sockeye_prepare_data.py to pre-process data for the Tranformer model
5. Run sockeye_train.py for training (change --use-cpu flag to --use-gpu to run on GPU)
6. Optimal model for training on GPU is in the kubernetes-experiments branch (sockeye_train.py)

# Our model's performance

Our model outperforms the baseline model (Random Forest Regressor / Classifier) in some cases, especially on large datasets. In other cases it is on par with the baseline. Ultimately, the model performs relatively poor on small datasets as the Boston house price dataset or the wine classification dataset.
