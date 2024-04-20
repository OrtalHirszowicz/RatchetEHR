# RatchetEHR: A Transformer-Based Approach for EHR Analysis

RatchetEHR is a novel transformer-based framework designed for the
predictive analysis of electronic health records (EHR) data in intensive care unit (ICU) set-
tings, with a specific focus on bloodstream infection (BSI) prediction. For the BSI task definition, we based our cohort
generation on Roimi et al.[4]

RatchetEHR is based mainly on SARD, a transformer-based model that has been presented by Kodialam et al. [1], 
and on GCT model, which is a transformer-based model that has been presented by Choi et al. [2]. The model's 
framework uses MIMICIV dataset [3] for training and testing purposes on BSI (blood stream infection) prediction task.

## Documentation
Most of the scripts and code in this repository were taken from the omop-learn package:
https://github.com/clinicalml/omop-learn. 
The repository is based on the following main parts:
* Generators - responsible for generating the data for the model
* Models - contains the EHR, RNN and LSTM models for the EHR data. 
Also include a SHAP wrapper model (used for SHAP analysis) and optimization methods for training the models.
* sql - contain SQL scripts for creating the cohort dataset.
* Tables - contain various tables for conversions required for the cohort dataset creation.
* Visualization - contains the visualization scripts for the model's performance and interpretability, also for the
visualizations presented in the article.

## Installation
First, please install the MIMICIV dataset into PostgreSQL database. The dataset can be downloaded from the following link:
https://physionet.org/content/mimiciv/2.2/
installation instructions can be found in the following link: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres

After installing the MIMICIV dataset, please install the required packages by running the following command:
pip install -r requirements.txt

## Running the code
First, please install the required packages as mentioned in the installation section.
The following scripts are used for training and testing the model:
* reconstruction_task_with_mimiciv.py - used for training and testing the RatchetEHR model on the reconstruction task.
* tl_reconstruction_task_with_mimiciv.py - used for training and testing the RatchetEHR model, LSTM or RNN on the BSI task.
* xgboost_mimiciv_raw_data.py - used for training and testing the XGBoost model on the BSI task.
* random_forest_mimiciv.py - used for training and testing the Random Forest model on the BSI task.


For the RNN, LSTM and RatchetEHR models, please use the hyper_params.py to tune the model and set the parameters.
In this file, you can choose the required model and set the hyperparameters for the model, the optimizer, the learning rate, the batch size, the number of epochs, etc.

## Reference:
[1] Kodialam, R. S., Boiarsky, R., Lim, J., Dixit, N., Sai, A., & Sontag, D. (2020). Deep Contextual Clinical Prediction with Reverse Distillation.

[2] Choi, E., Xu, Z., Li, Y., Dusenberry, M. W., Flores, G., Xue, Y., & Dai, A. M. (2020). Learning the Graphical Structure of Electronic Health Records with Graph Convolutional Transformer.

[3] Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67.
Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

[4] Roimi M, Neuberger A, Shrot A, Paul M, Geffen Y, Bar-Lavie Y. Early diagnosis of bloodstream infections in the intensive care unit using machine-learning algorithms. Intensive Care Med. 2020 Mar;46(3):454-462. doi: 10.1007/s00134-019-05876-8. Epub 2020 Jan 7. PMID: 31912208.


