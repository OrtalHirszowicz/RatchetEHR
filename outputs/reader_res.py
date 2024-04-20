import numpy as np

file = \
'outputs/tl_reconstruction_mimiciv.txt'
auc_roc = "Current ROC-AUC score on the test set:  "
auc_pr = "Current AUC PR score on the test set:  "
mimic_auc_roc = "Current ROC-AUC score on MIMIC-III: "
mimic_auc_pr =  "Current AUC PR score on MIMIC-III:  "
external_auc_roc = "Current ROC-AUC score on External: "
external_auc_pr = "Current AUC PR score on External: "
data = {
    auc_roc: [],
    auc_pr: [],
    mimic_auc_roc: [],
    mimic_auc_pr: [],
    external_auc_roc: [],
    external_auc_pr: []
}
with open(file) as f:
    for l in f:
        for line in data:
            if l[:len(line)] == line:
                data[line].append(float(l[len(line):-2]))

auc_roc_scores = np.array(data[auc_roc])
print("AUC ROC mean: ", auc_roc_scores.mean())
#print("AUC ROC median: ", np.median(auc_roc_scores))
print("AUC ROC std: ", auc_roc_scores.std())

auc_pr_scores = np.array(data[auc_pr])
print("AUC PR mean: ", auc_pr_scores.mean())
#print("AUC ROC median: ", np.median(auc_roc_scores))
print("AUC PR std: ", auc_pr_scores.std())
if len(data[external_auc_roc]) > 0:
    auc_roc_scores = np.array(data[external_auc_roc])
    print("External AUC ROC mean: ", auc_roc_scores.mean())
    #print("AUC ROC median: ", np.median(auc_roc_scores))
    print("External AUC ROC std: ", auc_roc_scores.std())

    auc_pr_scores = np.array(data[external_auc_pr])
    print("External AUC PR mean: ", auc_pr_scores.mean())
    #print("AUC ROC median: ", np.median(auc_roc_scores))
    print("External AUC PR std: ", auc_pr_scores.std())