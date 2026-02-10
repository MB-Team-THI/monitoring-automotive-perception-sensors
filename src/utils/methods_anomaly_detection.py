import numpy as np

from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from sklearn.svm import OneClassSVM

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler 

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import NearestNeighbors


from src.utils.evaluation_metrics import apply_various_thresholds
from src.utils.evaluation_metrics import calculate_classification_metrics



def calculate_lof_score(data):
    '''
    The opposite LOF of the training samples. The lower, the more abnormal. Inliers tend to have a LOF score close to 1, while outliers tend to have a larger LOF score.
    The local outlier factor (LOF) of a sample captures its supposed "degree of abnormality". 
    It is the average of the ratio of the local reachability density of a sample and those of its k-nearest neighbors.
    '''
    lof_classifier = LocalOutlierFactor()
    lof_classifier.fit(data)
    # Negative outlier factor for LOF 
    return np.mean(lof_classifier.negative_outlier_factor_)
      
    
def calculate_abod_score(data): 
    '''
     The outlier scores of the training data. The higher, the more abnormal. Outliers tend to have higher scores. This value is available once the detector is fitted.
    '''                        
    abod_classifier = ABOD()
    abod_classifier.fit(data)

    return np.mean(abod_classifier.decision_scores_)


def method_abod(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, output_dir="", anomaly_key="", params={}):    
    y_gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    y_gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)
    X_val = np.concatenate((val_data_normal, val_data_anomaly), axis=0)
    X_test = np.concatenate((test_data_normal, test_data_anomaly), axis=0)

    # Fit model on normal val-set
    model_abod = ABOD().fit(train_data_normal)

    # Define proper threshold based on val-set
    probas_val, confience_val = model_abod.predict_proba(X= X_val,
                                                         method='unify', 
                                                         return_confidence=True)
    outlier_probability_val  = probas_val[:,1] # ([proba of normal, proba of outliers])
    # metrics_val_all, best_threshold, metrics_val = apply_various_thresholds(outlier_probability_val, y_gt_val, normal_class_idx=-2, scores=outlier_probability_val)
    y_pred_val = model_abod.predict(X=X_val)
    metrics_val = calculate_classification_metrics(y_pred=y_pred_val, y_true=y_gt_val, y_score=outlier_probability_val)
    # Apply Test set and measure performance
    probas_test, confidence_test = model_abod.predict_proba(X_test,
                                                            method='unify', 
                                                            return_confidence=True)
    outlier_probability_test = probas_test[:,1]     # ([proba of normal, proba of outliers])
    y_pred_test = model_abod.predict(X=X_test)
    metrics_test = calculate_classification_metrics(y_pred=y_pred_test, y_true=y_gt_test, y_score=outlier_probability_test)

    res_dict = {'metrics_test':    metrics_test,
                'metrics_val':     metrics_val,
                # 'metrics_val_all': metrics_val_all,
                # 'threshold':       best_threshold,
                'settings':        params,
                }
    return res_dict

def method_lof(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, lof_params):    
    y_gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    y_gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)
    X_val = np.concatenate((val_data_normal, val_data_anomaly), axis=0)
    X_test = np.concatenate((test_data_normal, test_data_anomaly), axis=0)

    # Fit model on normal val-set
    model_lof = LocalOutlierFactor(n_neighbors=lof_params['n_neighbors'],
                                   novelty=lof_params['novelty'])
    model_lof.fit(train_data_normal)            
            
    # Define proper threshold based on val-set
    # .predict() -> -1 for anomalies/outliers and +1 for inliers.
    y_pred_val = model_lof.predict(X_val)
    y_pred_val = [0 if x == 1 else 1 for x in y_pred_val]
    anomaly_scores_val = -1 * model_lof.score_samples(X_val)
    metrics_val = calculate_classification_metrics(y_pred=y_pred_val, y_true=y_gt_val, y_score=anomaly_scores_val)

    # Apply Test set and measure performance
    y_pred_test = model_lof.predict(X_test)
    y_pred_test = [0 if x == 1 else 1 for x in y_pred_test]
    anomaly_scores_test = -1 * model_lof.score_samples(X_test)
    metrics_test = calculate_classification_metrics(y_pred=y_pred_test, y_true=y_gt_test, y_score=anomaly_scores_test)

    res_dict = {'metrics_test': metrics_test,
                'metrics_val':  metrics_val,
                'settings':     lof_params, }
    
    return res_dict


def method_isolation_forest(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):    
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)

    # Fit model on normal val-set
    n_estimators  = params['n_estimators']
    contamination = params['contamination']
    model_isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    model_isolation_forest.fit(train_data_normal)

    # Define proper threshold based on val-set
    pred_val = model_isolation_forest.predict(np.concatenate((val_data_normal, val_data_anomaly), axis=0))  # outlier=-1, inlier=1
    pred_val = [1 if i==-1 else 0 for i in pred_val]
    anomaly_scores_val = model_isolation_forest.decision_function(np.concatenate((val_data_normal, val_data_anomaly), axis=0))
    # The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
    anomaly_scores_val_inverse = - anomaly_scores_val

    metrics_val = calculate_classification_metrics(y_pred=pred_val, y_true=gt_val, y_score=anomaly_scores_val_inverse, calc_acc_per_class=False)


    # Apply Test set and measure performance
    pred_test = model_isolation_forest.predict(np.concatenate((test_data_normal, test_data_anomaly), axis=0)) # outlier=-1, inlier=1
    pred_test = [1 if i==-1 else 0 for i in pred_test]
    anomaly_scores_test = model_isolation_forest.decision_function(np.concatenate((test_data_normal, test_data_anomaly), axis=0))
    # The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
    anomaly_scores_test_inverse = - anomaly_scores_test
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=anomaly_scores_test_inverse, calc_acc_per_class=False)

    res_dict = {'metrics_test': metrics_test,
                'metrics_val':  metrics_val,
                'settings':     params,
                }
    
    return res_dict


def method_COPOD(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)
    # Scale data    
    scaler = StandardScaler()  
    X_scaled_train_normal = scaler.fit_transform(train_data_normal)
    X_scaled_val_normal   = scaler.transform(val_data_normal)
    X_scaled_val_anomaly  = scaler.transform(val_data_anomaly)
    X_scaled_test_normal  = scaler.transform(test_data_normal)
    X_scaled_test_anomaly = scaler.transform(test_data_anomaly)

    contamination = params['contamination']
    # Fit model on normal val-set
    model_copod = COPOD(contamination= contamination)
    model_copod.fit(X_scaled_train_normal) 
    model_copod.decision_scores_ 

    # Define proper threshold based on val-set
    probas_val = model_copod.predict_proba(np.concatenate((X_scaled_val_normal, X_scaled_val_anomaly), axis=0), method="unify")   
    probas_anomaly_val = probas_val[:,1] # ([proba of normal, proba of outliers])
    metrics_val_all, best_threshold, metrics_val = apply_various_thresholds(probas_val, gt_val, normal_class_idx=0, scores=probas_anomaly_val)
           
    # Apply Test set and measure performance
    probas_test  = model_copod.predict_proba(np.concatenate((X_scaled_test_normal, X_scaled_test_anomaly), axis=0), method="unify")
    pred_test    = [1 if x[0]<best_threshold else 0 for x in probas_test]
    probas_anomaly_test = probas_test[:,1] # ([proba of normal, proba of outliers])
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=probas_anomaly_test)
           
    res_dict = {'metrics_test':     metrics_test,
                'metrics_val':      metrics_val,
                'metrics_val_all':  metrics_val_all,
                'settings':         params,
                'threshold':        best_threshold,
                'anomaly_proba_val_normal':   probas_anomaly_val[:len(X_scaled_val_normal)].tolist(),
                'anomaly_proba_val_anomaly':  probas_anomaly_val[len(X_scaled_val_anomaly):].tolist(),
                'anomaly_proba_test_normal':  probas_anomaly_test[:len(X_scaled_test_normal)].tolist(),
                'anomaly_proba_test_anomaly': probas_anomaly_test[len(X_scaled_test_normal):].tolist(),
                }
    
    return res_dict


def method_svm(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)
    
    # Scale data    
    scaler = StandardScaler()  
    X_scaled_train_normal = scaler.fit_transform(train_data_normal)
    X_scaled_val_normal   = scaler.transform(val_data_normal)
    X_scaled_val_anomaly  = scaler.transform(val_data_anomaly)
    X_scaled_test_normal  = scaler.transform(test_data_normal)
    X_scaled_test_anomaly = scaler.transform(test_data_anomaly)

    # Model
    one_class_svm = OneClassSVM(nu = 0.01, kernel = 'rbf', gamma = 'auto')
    one_class_svm.fit(X_scaled_train_normal) 

    # Define proper threshold based on val-set
    pred_val = one_class_svm.predict(np.concatenate((X_scaled_val_normal, X_scaled_val_anomaly), axis=0))  
    #Change the anomalies' values and to make it consistent with the true values
    pred_val = [1 if i==-1 else 0 for i in pred_val]
    score_val = -1 * one_class_svm.score_samples(np.concatenate((X_scaled_val_normal, X_scaled_val_anomaly), axis=0))  
    metrics_val = calculate_classification_metrics(y_pred=pred_val, y_true=gt_val, y_score=score_val)

    # Apply Test set and measure performance
    pred_test = one_class_svm.predict(np.concatenate((X_scaled_test_normal, X_scaled_test_anomaly), axis=0))
    pred_test = [1 if i==-1 else 0 for i in pred_test]
    score_test = -1 * one_class_svm.score_samples(np.concatenate((X_scaled_test_normal, X_scaled_test_anomaly), axis=0))  
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=score_test)

    res_dict = {'metrics_test':  metrics_test,
                'metrics_val':   metrics_val,
                'settings':      params,
                }
    
    return res_dict


def method_knn(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)

    # Model    
    k = params['k_neighbors']
    metric = params['metric']
    model_knn = NearestNeighbors(n_neighbors=k,
                                 metric=metric)
    model_knn.fit(train_data_normal)  # Fit only on normal data

    # Define proper threshold based on val-set
    distances_normal, _ = model_knn.kneighbors(val_data_normal)  # Get distances to k-nearest neighbors
    anomaly_scores_normal = distances_normal[:, -1]  # Use max distance (k-th neighbor) 
    percent = 100 - params['anomaly_percentile']
    threshold = np.percentile(anomaly_scores_normal, percent)
    # Apply val set
    distances_val, _ = model_knn.kneighbors(np.concatenate((val_data_normal, val_data_anomaly), axis=0))  # Get distances to k-nearest neighbors
    anomaly_scores_val = distances_val[:, -1]
    pred_val = [1 if x > threshold else 0 for x in anomaly_scores_val]    
    metrics_val = calculate_classification_metrics(y_pred=pred_val, y_true=gt_val, y_score=anomaly_scores_val)

    # Apply Test set and measure performance
    distances_test, _ = model_knn.kneighbors(np.concatenate((test_data_normal, test_data_anomaly), axis=0))
    anomaly_scores_test = distances_test[:, -1]
    pred_test = [1 if x > threshold else 0 for x in anomaly_scores_test]    
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=anomaly_scores_test)

    res_dict = {'metrics_test':  metrics_test,
                'metrics_val':   metrics_val,
                'settings':      params,
                'threshold':     threshold,
                }
    
    return res_dict


def generate_synthetic_anomalies(n_samples, X_normal):
    min_values = X_normal.min(axis=0)  # Get min per feature
    max_values = X_normal.max(axis=0)  # Get max per feature

    # Sample uniformly within min-max range
    X_anomalies = np.random.uniform(low=min_values, high=max_values, size=(n_samples, X_normal.shape[1]))
    
    return X_anomalies


def rbf_kernels(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):
    gt_val  = [0]*len(val_data_normal)  + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal) + [1]*len(test_data_anomaly)
    X_train_anomaly_sythetic = generate_synthetic_anomalies(n_samples = len(train_data_normal), 
                                                            X_normal  = np.array(train_data_normal))
    gt_train = [0]*len(train_data_normal) + [1]*len(X_train_anomaly_sythetic)
    X_train_syntheric = np.concatenate((train_data_normal, X_train_anomaly_sythetic), axis=0)
    
    # Fit model on normal val-set 
    if params['model_type'] == "GaussianProcessClassifier":
        # O(N³)
        kernel = 1.0 * RBF(length_scale=1.0)
        model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=0, random_state=0)
        model.fit(X_train_syntheric, gt_train)

    elif params['model_type'] == "Nystroem":
        # O(NM²)
        feature_map_nystroem = Nystroem(kernel="rbf", gamma=1.0, n_components=100)
        model = make_pipeline(feature_map_nystroem, LogisticRegression())
        model.fit(X_train_syntheric, gt_train)

    # gpc.score(X_val, gt_val)
    probas_val = model.predict_proba(np.concatenate((val_data_normal, val_data_anomaly), axis=0))
    probas_anomaly_val = probas_val[:,1] # ([proba of normal, proba of outliers])
    metrics_val_all, best_threshold, metrics_val = apply_various_thresholds(probas_val, gt_val, normal_class_idx=0, scores=probas_anomaly_val)

    # Apply Test set and measure performance
    probas_test  = model.predict_proba(np.concatenate((test_data_normal, test_data_anomaly), axis=0))
    probas_anomaly_test = probas_test[:,1] # ([proba of normal, proba of outliers])
    pred_test    = [1 if x > best_threshold else 0 for x in probas_anomaly_test]
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=probas_anomaly_test)
           
    res_dict = {'metrics_test':     metrics_test,
                'metrics_val':      metrics_val,
                'metrics_val_all':  metrics_val_all,
                'threshold':        best_threshold,
                'settings':         params,
                'anomaly_proba_val_normal':   probas_anomaly_val[:len(val_data_normal)].tolist(),
                'anomaly_proba_val_anomaly':  probas_anomaly_val[len(val_data_anomaly):].tolist(),
                'anomaly_proba_test_normal':  probas_anomaly_test[:len(test_data_normal)].tolist(),
                'anomaly_proba_test_anomaly': probas_anomaly_test[len(test_data_anomaly):].tolist(),
                }

    return res_dict

