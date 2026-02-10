
import numpy as np
from torch import nn
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix


def apply_various_thresholds(probas, gt, scores=None, threshold_range=np.arange(0.05, 1.0, 0.05), normal_class_idx=-1):
    metrics_thresholds = {}
    best_threshold = 0.0
    best_metrics = {'f1_score': -1.0}

    for proba_threshold in threshold_range:
        key = str(round(proba_threshold, 2))
        if normal_class_idx == -1:
            # process probabilites for all normal classes -> take the max class probability
            pred = [0 if max(x) >= proba_threshold else 1 for x in probas]
        elif normal_class_idx == -2:
            # the probas are just the anomaly probailities (the higher the more secure the model is that the sample is an anomaly)
            pred = [1 if x >= proba_threshold  else 0 for x in probas]
        else:
            # process probabilites for normal and anomaly class
            pred = [0 if x[normal_class_idx] < proba_threshold  else 1 for x in probas]
        assert len(pred) == len(gt), "Must be of same length"
        
        metrics = calculate_classification_metrics(y_pred=pred, y_true=gt, y_score=scores)        
        metrics_thresholds[key] = metrics

        if best_metrics['f1_score']  < metrics['f1_score']:
            best_metrics   = metrics
            best_threshold = proba_threshold


    return metrics_thresholds, best_threshold, best_metrics


def calculate_classification_metrics(y_pred, y_true, y_score=None, calc_acc_per_class=True):
    assert len(y_pred) == len(y_true), "Must be of same length"
    res_dict = {}
    # Accuracy  
    res_dict['accuracy_overalll'] = accuracy_score(y_pred=y_pred, y_true=y_true)
    # F1 Score
    res_dict['f1_score'] = f1_score(y_pred=y_pred, y_true=y_true, zero_division=0.0)
    if y_score is not None:
        # AUROC score
        res_dict['roc_auc_score'] = float(roc_auc_score(y_score=y_score, y_true=y_true))
    else:
        res_dict['roc_auc_score'] = "not available"
    # Matthews Correlation Coefficient
    res_dict['mcc'] = float(matthews_corrcoef(y_pred=y_pred, y_true=y_true))

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y_true).ravel()
    res_dict['true_positives']  = int(tp)
    res_dict['false_positives'] = int(fp)
    res_dict['false_negatives'] = int(fn)
    res_dict['true_negatives']  = int(tn)
    # False Alarm Rate
    res_dict['fpr'] = float(fp / (fp +tn))
    # True positive rate / True detection rate:   TPR = TP / (TP + FN)
    res_dict['tpr'] = float(recall_score(y_true=y_true, y_pred=y_pred, pos_label=1, zero_division=0.0))

    if calc_acc_per_class:
        # Metric Anoamly-Class 
        res_dict['anomaly_samples_precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        res_dict['anomaly_samples_recall']    = recall_score(y_true, y_pred, pos_label=1, zero_division=0.0)
        # Metric Normal-Class
        res_dict['normal_samples_precision'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0.0)
    

    res_dict['ratio'] = {'y_pred': {str(k): int(v) for k,v in   zip(np.unique(y_pred, return_counts=True)[0],  np.unique(y_pred, return_counts=True)[1])},
                         'y_true': {str(k): int(v) for k,v in   zip(np.unique(y_true, return_counts=True)[0],  np.unique(y_true, return_counts=True)[1])}}
    return res_dict


def matrix_calc_forb_norm(x, x_pred):
    # Frobenius Norm (L2 norm in the matrix space)
    # The smaller the value the more similar are x and x_pred
    assert x.shape == x_pred.shape, "Matrices must have the same shape"
    return np.linalg.norm(x - x_pred, ord='fro')
    # return np.linalg.norm(x.to('cpu').detach().numpy() - x_pred.to('cpu').detach().numpy(), ord='fro')


def matrix_calc_mse(x, x_pred):
    # Mean Sqaures Error
    assert x.shape == x_pred.shape, "Matrices must have the same shape"
    mse_loss = nn.MSELoss()
    return mse_loss(x, x_pred).item()


def matrix_calc_mae(x, x_pred):
    # Mean Absolute Error
    assert x.shape == x_pred.shape, "Matrices must have the same shape"
    mae_loss = nn.L1Loss()
    return mae_loss(x, x_pred).item()


def matrix_calc_ssim(x, x_pred):
    # Structural Similarity Index Measure (SSIM)
    # Originally used to evaluate the similarity between images, however this might be also useful to evaluate matrices
    # Returns the structural similarity index within the range [-1, 1] with 1 being the matrices (images) are the same. 
    assert x.shape == x_pred.shape, "Matrices must have the same shape"
    win_size = min(list(x.shape))
    if win_size % 2 == 0:
        # if this size is even, we decrement it by one (`win_size -= 1`), because SSIM window size should be odd.
        win_size -= 1
                
    return ssim(x.to('cpu').detach().numpy(), x_pred.to('cpu').detach().numpy(), win_size=win_size)


def association_accuracy(obj_pairs_gt, dist_mat_pred=[], obj_pairs_preds=[]):
    # Calculate the association accuracy:= TP / (total GT pairs) for the made object associations
    obj_pairs_gt = obj_pairs_gt.tolist()

    if dist_mat_pred != []:
        # Perform Linear Assignment Problem
        matched_idx_1, matched_idx_2 = linear_sum_assignment(dist_mat_pred.cpu().detach().numpy(), maximize=True)
        obj_pairs_pred = [[x1, x2] for x1, x2 in zip(matched_idx_1, matched_idx_2)]
    else:
        obj_pairs_pred = [obj_pair['idx_pair'] for obj_pair in obj_pairs_preds]


    # Check if embedding matches are in the gt obj_pairs
    true_positives = []
    for pair_gt in obj_pairs_gt:
        if pair_gt in obj_pairs_pred:
            true_positives.append(pair_gt)
            
    return len(true_positives) / len(obj_pairs_gt)


def nearest_neighbors_of_objects(x, x_pred, k_percent=0.05, k=1):
    # Check if the indices with the highest values are the same for the rows and columns of the association-(x) and embedding- (x_pred) matrix as long as the values in the association-matrix are above a certian threshold
    # Range = [0,1] and the higher the better
    threshold_min_association_score = 0.5

    # Get number of neighbors to investigate
    # k = math.ceil(min(x.shape) * k_percent)

    # Rows
    score_rows = []
    for idx_row in range(x.shape[0]):
        # Get k idx with highest values
        idx_x      = [i for i in  np.argpartition(x[idx_row,:].cpu(), -k)[-k:] if x[idx_row,i] > threshold_min_association_score]
        idx_x_pred = np.argpartition(x_pred[idx_row,:].cpu().detach().numpy(), -k)[-k:]
        # How many idx match?       
        if 0 < len(idx_x):         
            matches = sum([x in idx_x for x in idx_x_pred])
            score_rows.append(matches / k)

    # Cols
    score_cols = []
    for idx_col in range(x.shape[1]):
        # Get k idx with highest values
        idx_x      = [i for i in  np.argpartition(x[:, idx_col].cpu(), -k)[-k:] if x[i, idx_col] > threshold_min_association_score]
        idx_x_pred = np.argpartition(x_pred[:, idx_col].cpu().detach().numpy(), -k)[-k:]
        # How many idx match?
        if 0 < len(idx_x):
            matches = sum([x in idx_x for x in idx_x_pred])
            score_cols.append(matches / k)
                
    # Ensure there are no empty lists
    if score_rows == []:
        score_rows_mean = 0
    else:
        score_rows_mean = np.mean(score_rows) / 2
    if score_cols == []:
        score_cols_mean = 0
    else:
        score_cols_mean = np.mean(score_cols) / 2

    return (score_rows_mean + score_cols_mean)

  