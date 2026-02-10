
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.evaluation_metrics import apply_various_thresholds
from src.utils.evaluation_metrics import calculate_classification_metrics



def method_gmm(train_data_normal, val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params, epoch=0, anomaly_key=''):
    n_comp = params['n_comp']
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)

    # Scale data    
    if params['scale_data']:
        scaler = StandardScaler()
        train_data_normal = scaler.fit_transform(train_data_normal)
        val_data_normal   = scaler.transform(val_data_normal)
        val_data_anomaly  = scaler.transform(val_data_anomaly)
        test_data_normal  = scaler.transform(test_data_normal)
        test_data_anomaly = scaler.transform(test_data_anomaly)
    
    # Fit model on normal val-set
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
    gmm.fit(train_data_normal)


    res_dict = {}
    if params['use_max_cluster_probas']:
        # Define proper threshold based on val-set - based on cluster assignment probabilities 
        probas_val = gmm.predict_proba(np.concatenate((val_data_normal, val_data_anomaly), axis=0))
        max_probas_val = np.max(probas_val, axis=1)
        anomaly_scores_val = 1 - max_probas_val  
        metrics_val_all, best_threshold, metrics_val = apply_various_thresholds(probas_val, gt_val, scores=anomaly_scores_val, threshold_range=np.arange((1.0/n_comp), 1.0, 0.05))

        # Apply Test set and measure performance
        probas_test = gmm.predict_proba(np.concatenate((test_data_normal, test_data_anomaly), axis=0))
        max_probas_test = np.max(probas_test, axis=1)
        anomaly_scores_test = 1 - max_probas_test  
        pred_test = [0 if max_proba > best_threshold else 1 for max_proba in max_probas_test]
        metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test, y_score=anomaly_scores_test)

        res_dict['class_probability'] = {'metrics_test':    metrics_test,
                                         'metrics_val':     metrics_val,
                                         'metrics_val_all': metrics_val_all,
                                         'threshold':       best_threshold,
                                         'desc':            'key = probability threshold used for anomaly detection',
                                         'settings':        params,}
    else:
        res_dict['class_probability'] = 'not executed'

    if params['use_log_likelihood']:        
        # Define proper threshold based on val-set - based on log-likelihoods
        # Calculate log-likelihoods for training inliers and outliers
        # "Log-likelihood considers the combined contribution of all clusters rather than focusing on the most probable one."
        # "It captures the overall "fit" of the data point under the model, making it more robust for anomaly detection."
        # Negative score = the higher the number the more normal / the lower the number the more anomalous
        X_train_log_likelihood_normal  = gmm.score_samples(val_data_normal)
        percentile_val = params['log_likelihood_threshold_percentil']
        threshold = np.percentile(X_train_log_likelihood_normal, percentile_val)  # Set a threshold at the 5th percentile of inliers

        # Val Set
        X_train_log_likelihood_anomaly = gmm.score_samples(val_data_anomaly)
        X_train_log_likelihood_total   = np.concatenate((X_train_log_likelihood_normal, X_train_log_likelihood_anomaly), axis=0) 
        if params['scale_log_scores']:
            scaler = MinMaxScaler()
            X_train_log_likelihood_total  = scaler.fit_transform(X_train_log_likelihood_total.reshape(-1, 1)).reshape(-1)
            X_train_log_likelihood_normal  = scaler.transform(X_train_log_likelihood_normal.reshape(-1, 1)).reshape(-1)
            X_train_log_likelihood_anomaly = scaler.transform(X_train_log_likelihood_anomaly.reshape(-1, 1)).reshape(-1)

        y_pred_train = [1 if x < threshold else 0 for x in X_train_log_likelihood_total]
        metrics_val = calculate_classification_metrics(y_pred=y_pred_train, y_true=gt_val, y_score= -X_train_log_likelihood_total)
        metrics_val['anomaly_rate_normal_train_data'] = sum([1 if x < threshold else 0 for x in X_train_log_likelihood_normal]) / len(X_train_log_likelihood_normal)

        # Test set and measure performance
        X_test_log_likelihood_normal  = gmm.score_samples(test_data_normal)
        X_test_log_likelihood_anomaly = gmm.score_samples(test_data_anomaly)    
        X_test_log_likelihood_total   = np.concatenate((X_test_log_likelihood_normal, X_test_log_likelihood_anomaly), axis=0)    
        if params['scale_log_scores']:
            X_test_log_likelihood_total  = scaler.transform(X_test_log_likelihood_total.reshape(-1, 1)).reshape(-1)            
            X_test_log_likelihood_normal  = scaler.transform(X_test_log_likelihood_normal.reshape(-1, 1)).reshape(-1)
            X_test_log_likelihood_anomaly = scaler.transform(X_test_log_likelihood_anomaly.reshape(-1, 1)).reshape(-1)

        y_pred_test = [1 if x < threshold else 0 for x in X_test_log_likelihood_total]
        metrics_test = calculate_classification_metrics(y_pred=y_pred_test, y_true=gt_test, y_score= -X_test_log_likelihood_total) 
        metrics_test['anomaly_rate_normal_test_data'] = sum([1 if x < threshold else 0 for x in X_test_log_likelihood_normal]) / len(X_test_log_likelihood_normal)


        res_dict['log_likelihood_score'] = {'metrics_test':   metrics_test,
                                            'metrics_val':    metrics_val,
                                            'desc':           'key = log likelihood score threshold used for anomaly detection',
                                            'settings':       params,
                                            'threshold':      threshold,  
                                            'log_likelihood_scores_val_normal':  X_train_log_likelihood_normal.tolist(),
                                            'log_likelihood_scores_val_aug':     X_train_log_likelihood_anomaly.tolist(),
                                            'log_likelihood_scores_test_normal': X_test_log_likelihood_normal.tolist(),
                                            'log_likelihood_scores_test_aug':    X_test_log_likelihood_anomaly.tolist(),}
        
    else:
        res_dict['log_likelihood_score'] = 'not executed'

    return {'class_probability':    res_dict['class_probability'], 
            'log_likelihood_score': res_dict['log_likelihood_score'],}


def method_dbscan(val_data_normal, val_data_anomaly, test_data_normal, test_data_anomaly, params):
    gt_val  = [0]*len(val_data_normal) + [1]*len(val_data_anomaly)
    gt_test = [0]*len(test_data_normal)  + [1]*len(test_data_anomaly)

    db_model = DBSCAN(eps           = params['db_eps'], 
                      min_samples   = params['db_min_samples'],
                      metric        = params['db_metric'],
                      metric_params = None,
                      algorithm     = 'auto',
                      leaf_size     = 30,
                      p             = None,
                      n_jobs        = None)
    
    # Train-Set
    data_val = np.concatenate((val_data_normal, val_data_anomaly), axis=0)
    db_model.fit(data_val)
    pred_val = db_model.labels_
    assert len(data_val) == len(pred_val), "They must have the same length"
    pred_val = [1 if x ==-1 else 0 for x in pred_val]
    metrics_val = calculate_classification_metrics(y_pred=pred_val, y_true=gt_val)

    # Test-Set
    data_test = np.concatenate((test_data_normal, test_data_anomaly), axis=0)
    db_model.fit(data_test)
    pred_test = db_model.labels_
    assert len(data_test) == len(pred_test), "They should have the same length"
    pred_test = [1 if x ==-1 else 0 for x in pred_test]
    metrics_test = calculate_classification_metrics(y_pred=pred_test, y_true=gt_test)

    res_dict = {'metrics_test':   metrics_test,
                'metrics_val':    metrics_val,
                'settings':       params,}    
    return res_dict
