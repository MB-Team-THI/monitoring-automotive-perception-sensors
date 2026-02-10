import os
import json
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, mean_squared_error
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from src.dataset.dataset import dataset
from src.evaluation.eval import eval

from src.utils.dimensionality_reduction import dim_red_selection

from src.utils.evaluation_metrics import matrix_calc_mse
from src.utils.evaluation_metrics import matrix_calc_mae
from src.utils.evaluation_metrics import association_accuracy
from src.utils.evaluation_metrics import matrix_calc_forb_norm
from src.utils.evaluation_metrics import nearest_neighbors_of_objects

from src.utils.visualizations import visualize_anomaly_score_distribution

from src.utils.methods_clustering import method_dbscan, method_gmm
from src.utils.methods_anomaly_detection import method_abod, method_lof, method_isolation_forest, method_COPOD, method_svm, method_knn, rbf_kernels

from src.utils.object_association.process_scene_data import create_offset_subclasses

from src.utils.sample_data import create_sample_data_augmentations, create_sample_data_subclasses, alter_object_list_eval


def save_metrics(metric_results, run_name, epoch, settings, output_dir):
    output_emb = {"name":            "Saved anomaly detection metrics",
                  "run_name":        run_name,
                  "epoch":           epoch,
                  "metrics":         metric_results,
                  "settings":        settings,}
    
    # Save as JSON
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "ad_metrics_" + str(epoch) + ".json")
    with open(filename, 'w') as f:
        json.dump(output_emb, f)



def save_clustering_embeddings(model, dataset_test_dict, run_name, epoch, eval_153, dummy_rand_vals, cluster_rep=[]):
    '''
    Two-fold evaluation:
            - Saving of the reconstructed occupancy grid
            - Saving of the embeddings for the embedding explorer
    '''
    epoch += 1
    if cluster_rep == []:
        kmeans_loss_enabled = False
    else: 
        kmeans_loss_enabled = True

    model.to(eval_153.device)
    model.eval()
    dataset_local            = [dataset(**dataset_test_dict[0])]
    res_embeddings           = []
    res_labels_gt            = []
    res_labels_cluster_loss  = []
    embedding_order          = []
    key_order           = ['dummy', 'dummy_ranges', 'gt_pred_class', 'gt_visibility', 'gt_category', 'gt_attribute', 'ego_v_mean',
                           'gt_num_lidar_pts_mean', 'gt_v_mean', 'tracking_scr_camera', 'tracking_scr_lidar', 'tracking_scr_mean',
                           'odd_day_night', 'odd_weather', 'odd_distance_ego_obj'] 
    label_dict          = {key: [] for key in key_order}

    # Iterate over the dataset
    for sample_idx, _ in enumerate(range(dataset_local[0].__len__())):       
        # The sample_idx is used for sampling an object pair from the dataset
        # Data Preprocessing -------------------------------------------------  
        input_data  = dataset_local[0].__getitem__(sample_idx)

        obj_gt      = input_data['obj_gt_org']            
        assert obj_gt != [], ("Unexpected, only for the test-set the gt_obj should not be available obj-id" + str(input_data['obj_pair_id']))

        traj_pair = {'obj_camera': input_data['obj_camera'], 
                     'obj_lidar':  input_data['obj_lidar'],
                     'obj_ego':    input_data['obj_ego'],
                     'data_order_dict':  input_data['general_info'],}           
        for key in ['obj_camera', 'obj_lidar', 'obj_ego']:
            traj_pair[key] = torch.from_numpy(traj_pair[key]).to(torch.float).to(eval_153.device)
            traj_pair[key] = torch.swapaxes(traj_pair[key], 0, 1)   
            traj_pair[key] = torch.unsqueeze(traj_pair[key], 0)     # simulate batch dimension


        dummy_label = [1]

        # Network execution -------------------------------------------------   
        [embeddings, x_pred] = model(traj_pair)
           
        # Embeddings -------------------------------------------------------            
        res_embeddings.append(embeddings.squeeze().cpu().detach().numpy().tolist())


        # Embeddings Order -------------------------------------------------
        # Save the order in which labels and embeddings are processed
        current_obj_pair = {'obj_pair_name':        input_data['obj_pair_name'],
                            'obj_pair_global_idx':  int(input_data['obj_pair_global_idx']),
                            'processing_idx':       int(sample_idx) }
        embedding_order.append(current_obj_pair)


        # Labels ----------------------------------------------------------
        # Add GT and meta-information to label dict
        res_labels_gt.append(int(dummy_label[0]))

        if kmeans_loss_enabled:
            # Find closest cluster representative for the current embedding and assign the corresponding cluster index 
            # 1. Compute distances 
            distances       = torch.cdist(embeddings.float(), cluster_rep.to(embeddings.device).float(), p=2)
            # 2. Get min distance -> idx of next cluster_rep [n_batch x 1]
            cluster_assign  = torch.argmin(distances, dim=1)
            res_labels_cluster_loss.append(int(cluster_assign.cpu().numpy()[0]))

       
    n_clusters = len(np.unique(res_labels_gt))
    kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++").fit(res_embeddings)
    res_labels_kmeans = kmeans_model.labels_.tolist()
            

    # Description GT Cluster ------------------------------------------------------
    dummy_names = ['Dummy0', 'Dummy1', 'Dummy2', 'Dummy3', 'Dummy4']
    description_gt = {'name':            'dummy_gt',
                      'ranges':          [[x, x] for x in range(len(dummy_names))],
                      'description':     dummy_names,
                      'clustercount':    []
                      }
    
    # Description Assigned Cluster ------------------------------------------------------
    if kmeans_loss_enabled:
        cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 3', 'Cluster 4']
        counts_clusters = []# count_label_dist(labels = res_labels_cluster_loss, possible_labels = cluster_names)
        description_cluster = {'name':            'cluster_assignment_ep' + str(epoch),
                            'ranges':          [[x, x] for x in range(len(cluster_names))],
                            'description':     cluster_names,
                            'clustercount':    counts_clusters
                            }
        
    cluster_names   = ['Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    counts_clusters = []  # count_label_dist(labels = res_labels_kmeans, possible_labels = cluster_names)
    description_kmeans  = {'name':            'kmeans-clustering',
                       'ranges':          [[x, x] for x in range(len(cluster_names))],
                       'description':     cluster_names,
                       'clustercount':    counts_clusters
                      }
        
    
    if kmeans_loss_enabled:
        # Compute simple metrics about clustering and track them with wandb
        ars = adjusted_rand_score(res_labels_gt, res_labels_cluster_loss)
        try:
            sc = silhouette_score(X=res_embeddings, labels=res_labels_cluster_loss)
        except:
            # For e.g., num_clusters == 1 the silhouette_score cannot be calculated
            sc = -1.0
        wandb.log({"KMeans-Loss: Adjusted Rand Score":   ars})
        wandb.log({"KMeans-Loss: Silhouette Score":      sc})

    kmeans_ars = adjusted_rand_score(res_labels_gt, res_labels_kmeans)
    try:
        kmeans_sc = silhouette_score(X=res_embeddings, labels=res_labels_kmeans)
    except:
        # For e.g., num_clusters == 1 the silhouette_score cannot be calculated
        kmeans_sc = -1.0
    wandb.log({"KMeans-Model: Adjusted Rand Score":   kmeans_ars, "Epoch": epoch})
    wandb.log({"KMeans-Model: Silhouette Score":      kmeans_sc, "Epoch": epoch})
    

    # Sanity check
    if kmeans_loss_enabled:
        assert len(res_embeddings) == len(res_labels_cluster_loss), "The length of the embeddings and labels must match - lengths are different for label " + key
    assert len(res_embeddings) == len(res_labels_kmeans)	

    dim_red_experiments =  [
            {'dim_red_method': 'pca'},
        ]

    # Save cluster representatives
    if cluster_rep != []:
        dict_cluster_rep = {'epoch':                    epoch,
                            'k cluster':                cluster_rep.shape[0],
                            'z dim':                    cluster_rep.shape[1],
                            'cluster representatives':  cluster_rep.cpu().detach().numpy().tolist()}
        output_dir = os.path.join(eval_153.output_dir, "cluster_evolution", "cluster_rep")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'cluster_rep' + str(epoch) + '.json')
        with open(filename, 'w') as f:
                json.dump(dict_cluster_rep, f)


    # Save embeddings
    for dim_red_experiment in dim_red_experiments:

        # perform dim red
        data_reduced, desc_str, dim_red_params = dim_red_selection(embeddings_original = res_embeddings,
                                                                   output_dim          = 2,
                                                                   experiment          = dim_red_experiment)
        if data_reduced == []:
            continue    

        # Create path   
        if dim_red_experiment['dim_red_method'] == 'no_dim_red':
            dim_red_string = 'original_space'    
        elif dim_red_experiment['dim_red_method'] == 'pca':
            dim_red_string = 'pca'    
        elif dim_red_experiment['dim_red_method'] == 'tsne':
            dim_red_string = 'tsne_' + str(dim_red_experiment['perplexity'])
        elif dim_red_experiment['dim_red_method'] == 'umap':
            dim_red_string = 'umap_n'+ str(dim_red_experiment['n_neighbors']) + '_' + dim_red_experiment['metric']

        output_dir = os.path.join(eval_153.output_dir, "cluster_evolution", dim_red_string)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save reduced emb
        dict_out = {'name':                  desc_str+'_ep'+str(epoch),
                    'points':                data_reduced.tolist(),
                    'dim_reduction_params':  dim_red_params}
        filename = os.path.join(output_dir, 'embeddings_' + str(epoch) + '.json')
        with open(filename, 'w') as f:
            json.dump(dict_out, f)

        if kmeans_loss_enabled:
            # Save labels Cluster loss
            filename = output_dir + "labels_" + str(epoch) + ".json"
            with open(filename, 'w') as f:
                json.dump(res_labels_cluster_loss, f)
            # Save description Cluster loss    
            filename = output_dir + "description_" + str(epoch) + ".json"
            with open(filename, 'w') as f:
                json.dump(description_cluster, f)
        else:
            # Save labels kmeans cluster
            filename = output_dir + "labels_" + str(epoch) + ".json"
            with open(filename, 'w') as f:
                json.dump(res_labels_kmeans, f)
            # Save description kmeans cluster        
            filename = output_dir + "description_" + str(epoch) + ".json"
            with open(filename, 'w') as f:
                json.dump(description_kmeans, f)

        if epoch == 1:
            # Save the GT info only once
            # Save labels GT - labels don't change random factor stays the same
            filename = output_dir + "labels_0.json"
            with open(filename, 'w') as f:
                json.dump(res_labels_gt, f)       

            # Save description GT      
            filename = output_dir + "description_0.json"
            with open(filename, 'w') as f:
                json.dump(description_gt, f)


    model.train()  
      

def save_latent_embeddings_per_scene(results_val, results_test, metric_results, run_name, epoch, settings, output_dir, res_keys_2_save, results_train=None):            
    res_keys_2_save = ['pred_single_emb', 'pred_emb_pairs']
    # Setupt Dict 
    output_emb ={"name":            "Saved embeddings from the latent space",
                 "data_structure":  "Samples X [Camera, LiDAR] X NumObjects X EmbeddingSpaceSize",
                 "run_name":        run_name,
                 "epoch":           epoch,
                 "embeddings":      {"val_set":   {k: results_val['model_preds'][k]  for k in res_keys_2_save},                                                    
                                     "test_set":  {k: results_test['model_preds'][k] for k in res_keys_2_save},
                                     "train_set": {k: results_train['model_preds'][k] for k in res_keys_2_save} },
                 "metrics":         metric_results,
                 "settings":        settings,}
    
    # Save as JSON
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "embeddings_" + str(epoch) + ".json")
    with open(filename, 'w') as f:
        json.dump(output_emb, f)
                           

def matrix_similarity():
    pass


def get_meta_info(batch_data):
    meta_info = {}
    meta_info['scene_name'] = batch_data['scene_info'][0]['scene']['name']

    data_order = batch_data['general_info'][0]['data_order_tracking_res']

    for key, data in zip(['camera', 'lidar'], [batch_data['obj_list_camera'][0], batch_data['obj_list_lidar'][0]]):
        pred_classes    = []
        tracking_scores = []
        for obj_c in data: 
            non_zero   = torch.count_nonzero(obj_c[:,data_order.index('timestamp')]).item()
            # Get predicted classes
            pred_class = int(obj_c[0,data_order.index('pred_class')].item())
            pred_classes.append(pred_class) 

            # Get tracking scores
            tracking_score_mean = sum(obj_c[0:non_zero+1,data_order.index('tracking_score')].tolist()) / non_zero
            tracking_scores.append(tracking_score_mean)

        meta_info[key] = {'pred_classes':    pred_classes,
                           'tracking_scores': tracking_scores,}

    return meta_info


def flat_data(data_nested):            
    # Input-shape:  N_scenes x N_object x sampe_size
    # Output-shape: N_objects_all x sampe_size
            
    # Convert data to python std list-type:
    if 0<len(data_nested) and isinstance(data_nested[0], np.ndarray):
        z_list = [[x.tolist() for x in sample] for sample in data_nested]
    else:
        z_list = data_nested 
    list_num_objects = [len(x) for x in data_nested]
    # Flatten data
    z_flat = [item for sublist in z_list for item in sublist]

    assert len(z_flat) == sum(list_num_objects), "Must be of same length"
    return z_flat, z_list


def perform_anomaly_detection(data_train, data_val, data_test, ad_methods_params={}, epoch=-1, partition_z_space=False, use_subclasses=False):   
    single_aug_keys = list(data_val['pred_single_embs'].keys())
    if 'z_obj_list_camera' in single_aug_keys:
        single_aug_keys.remove('z_obj_list_camera')
    if 'z_obj_list_lidar' in single_aug_keys:
        single_aug_keys.remove('z_obj_list_lidar')
    if 'z_obj_list_camera_and_lidar' in single_aug_keys:
        single_aug_keys.remove('z_obj_list_camera_and_lidar')
        
    pairs_aug_keys  = list(data_val['pred_emb_pairs'].keys())
    if 'z_object_pairs_camera_normal_lidar_normal' in pairs_aug_keys:
        pairs_aug_keys.remove('z_object_pairs_camera_normal_lidar_normal')

    if use_subclasses:
        keys_setup = {'pred_emb_pairs':         {'normal':     'z_object_pairs_sample_subclass_1',
                                                 'augmented': ['z_object_pairs_sample_subclass_2'],}, 
                     }
    else:
            
        keys_setup = {'pred_emb_pairs': {'normal':    'z_object_pairs_camera_normal_lidar_normal',
                      'augmented':      pairs_aug_keys,}, 
                    }
    res_experiment = {k: {} for k in keys_setup}

    # Perform the AD procedure for all given anomaly keys
    for z_type in keys_setup:
        if "pred_single_embs" in z_type:
            data_type_key = "pred_single_embs"
        else:
            data_type_key = z_type
        for anomaly_key in keys_setup[z_type]['augmented']:
            ###### 1. Embeddings of the val- and test-set are already given by the input data ######
            normal_key     = keys_setup[z_type]['normal']
            train_normal   = np.array(data_train[data_type_key][normal_key])
            val_normal     = np.array(data_val[data_type_key][normal_key])
            val_augmented  = np.array(data_val[data_type_key][anomaly_key])
            test_normal    = np.array(data_test[data_type_key][normal_key])
            test_augmented = np.array( data_test[data_type_key][anomaly_key])
            
                
            ###### 2. Perform Anomaly Detection ######
            res_ad = {}            
            ### Soft Classifiers
            # GMM
            if ad_methods_params["GMM"]["enabled"]:
                res_gmm =  method_gmm(train_data_normal = train_normal,
                                      val_data_normal   = val_normal, 
                                      val_data_anomaly  = val_augmented, 
                                      test_data_normal  = test_normal, 
                                      test_data_anomaly = test_augmented,
                                      params            = ad_methods_params['GMM'],
                                      anomaly_key       = anomaly_key,
                                      epoch             = epoch)
                res_ad['GMM_class_prob']           = res_gmm['class_probability']
                res_ad['GMM_log_likelihood_score'] = res_gmm['log_likelihood_score']
            # DBSCAN
            if ad_methods_params["DBSCAN"]["enabled"]:
                res_ad['DBSCAN'] = method_dbscan(val_data_normal   = val_normal, 
                                                 val_data_anomaly  = val_augmented, 
                                                 test_data_normal  = test_normal, 
                                                 test_data_anomaly = test_augmented,
                                                 params            = ad_methods_params['DBSCAN'])
            ### Anoamly Detection   

            # ABOD
            if ad_methods_params["ABOD"]["enabled"]:
                res_ad['ABOD'] = method_abod(train_data_normal = train_normal,
                                             val_data_normal   = val_normal, 
                                             val_data_anomaly  = val_augmented, 
                                             test_data_normal  = test_normal, 
                                             test_data_anomaly = test_augmented,)
            # LOF
            if ad_methods_params["LOF"]["enabled"]:
                res_ad['LOF']  =  method_lof(train_data_normal = train_normal,
                                             val_data_normal   = val_normal, 
                                             val_data_anomaly  = val_augmented, 
                                             test_data_normal  = test_normal, 
                                             test_data_anomaly = test_augmented,
                                             lof_params        = ad_methods_params['LOF'])
            # Isolation Forest
            if ad_methods_params["IsolationForest"]["enabled"]:
                res_ad['IsolationForest']  = method_isolation_forest(train_data_normal = train_normal,
                                                                     val_data_normal   = val_normal, 
                                                                     val_data_anomaly  = val_augmented, 
                                                                     test_data_normal  = test_normal, 
                                                                     test_data_anomaly = test_augmented,
                                                                     params            = ad_methods_params['IsolationForest'])
            # COPOD
            if ad_methods_params["COPOD"]["enabled"]:         
                res_ad['COPOD'] = method_COPOD(train_data_normal = train_normal,
                                               val_data_normal   = val_normal, 
                                               val_data_anomaly  = val_augmented, 
                                               test_data_normal  = test_normal, 
                                               test_data_anomaly = test_augmented,
                                               params            = ad_methods_params['COPOD'])       
            # SVM
            if ad_methods_params["SVM"]["enabled"]:
                res_ad['SVM'] = method_svm(train_data_normal = train_normal,
                                           val_data_normal   = val_normal, 
                                           val_data_anomaly  = val_augmented, 
                                           test_data_normal   = test_normal, 
                                           test_data_anomaly  = test_augmented,
                                           params             = ad_methods_params['SVM'])       
            # KNN
            if ad_methods_params["KNN"]["enabled"]:
                res_ad['KNN'] = method_knn(train_data_normal = train_normal,
                                            val_data_normal   = val_normal, 
                                            val_data_anomaly  = val_augmented, 
                                            test_data_normal   = test_normal, 
                                            test_data_anomaly  = test_augmented,
                                            params             = ad_methods_params['KNN'])   
            # RBF
            if ad_methods_params["RBF"]["enabled"]:
                res_ad['RBF'] = rbf_kernels(train_data_normal = train_normal,
                                            val_data_normal   = val_normal, 
                                            val_data_anomaly  = val_augmented, 
                                            test_data_normal   = test_normal, 
                                            test_data_anomaly  = test_augmented,
                                            params             = ad_methods_params['RBF'])   

                
            
            ###### 3. Collect results for this anomaly_key ######
            val_n, val_a, test_n, test_a = len(val_normal), len(val_augmented), len(test_normal), len(test_augmented)
            ratio_val_test = {'test #': str(test_n) + ' - '+ str(test_a),
                              'test %': str(test_n/(test_n+test_a)) + ' - '+ str(test_a/(test_n+test_a)),
                              'val #':  str(val_n) + ' - '+ str(val_a),
                              'val %':  str(val_n/(val_n+val_a)) + ' - '+ str(val_a/(val_n+val_a)),}
            
            compact_metrics = {'metrics_test': {}, 'metrics_val':  {},}
            for k in res_ad:
                # DataSplit
                for datasplit in ['metrics_test', 'metrics_val']:
                    compact_metrics[datasplit][k+'_accuracy_overalll'] = res_ad[k][datasplit]['accuracy_overalll']
                    compact_metrics[datasplit][k+'_f1-score']          = res_ad[k][datasplit]['f1_score']
                    compact_metrics[datasplit][k+'_roc_auc_score']     = res_ad[k][datasplit]['roc_auc_score']


            res_experiment[z_type][anomaly_key] =  {'anomaly_detection_results':          res_ad,
                                                    'compact_ad_metrics':                 compact_metrics,  
                                                    'data_ratio [#normal - #augmented]':  ratio_val_test,
                                                    'params ad methods':                  ad_methods_params,
                                                    'anomaly_key':                        anomaly_key,}
            
                    
            ###### 4. Plot Distribution of Anomaly Scores ######
            visualize_params = {'GMM':      True,
                                'COPOD':    True,}
            # only available for augmentation pr_s3
            visualize_anomaly_score_distribution(res_ad      = res_ad, 
                                                 output_dir  = ad_methods_params['GMM']['output_dir'], 
                                                 params      = visualize_params,
                                                 epoch       = epoch,
                                                 anomaly_key = anomaly_key,) 

    return res_experiment
        

def calculate_asso_metrics(data):
    lookupdict = {'z_object_pairs_camera_normal_lidar_normal':          'association_info_camera_lidar',
                  
                  'z_object_pairs_camera_normal_lidar_pr_s3':       'association_info_camera_lidar_pr_s3',
                  'z_object_pairs_camera_normal_lidar_pr_s2':       'association_info_camera_lidar_pr_s2',
                  'z_object_pairs_camera_normal_lidar_pr_s1':       'association_info_camera_lidar_pr_s1',
                  'z_object_pairs_camera_normal_lidar_fog_s3':      'association_info_camera_lidar_fog_s3',
                  'z_object_pairs_camera_normal_lidar_sm_s3':       'association_info_camera_lidar_sm_s3',
                  'z_object_pairs_camera_normal_lidar_tm_s3':       'association_info_camera_lidar_tm_s3',
                  'z_object_pairs_camera_normal_lidar_mb_s3':       'association_info_camera_lidar_mb_s3',
                  'z_object_pairs_camera_normal_lidar_br_s3':       'association_info_camera_lidar_br_s3',

                  'z_object_pairs_camera_br_s3_lidar_normal':       'association_info_camera_br_s3_lidar',  
                  'z_object_pairs_camera_da_s3_lidar_normal':       'association_info_camera_da_s3_lidar',     
                  'z_object_pairs_camera_fog_s3_lidar_normal':      'association_info_camera_fog_s3_lidar',     
                  'z_object_pairs_camera_mb_s3_lidar_normal':       'association_info_camera_mb_s3_lidar',     
                  'z_object_pairs_camera_mc_s3_lidar_normal':       'association_info_camera_mc_s3_lidar',     
                  'z_object_pairs_camera_snow_s3_lidar_normal':     'association_info_camera_snow_s3_lidar',     
                  'z_object_pairs_camera_tm_s3_lidar_normal':       'association_info_camera_tm_s3_lidar',        

                  'z_object_pairs_camera_fog_s3_lidar_fog_s3':      'association_info_camera_fog_s3_lidar_fog_s3',
                  }
    
    obj_pair_options = list(data['model_preds']['pred_emb_pairs'].keys())
    res_dict = {k: {} for k in obj_pair_options}

    ### Evaluation of the Object Association 
    metric_keys = ['forb_norm', 'mse', 'mae', 'knn', 'asso_acc_v1', 'asso_acc_v2', 'number_asso']
    for obj_pair_key in res_dict:
        placeholder_dict =  {k: [] for k in metric_keys}
        batch_data = data['model_input']
        num_scenes = len(batch_data)

        ## Calculate the metrics for each scene
        for idx in range(num_scenes):
            asso_key = lookupdict[obj_pair_key]
            if asso_key == "association_info_camera_lidar":
                associated_objects_idx     = batch_data[idx][asso_key][0]['associated_objects_idx']
                association_matrix         = batch_data[idx][asso_key][0]['distance_matrix']  
            else:
                associated_objects_idx     = batch_data[idx]['association_info_aug_dict'][asso_key][0]['associated_objects_idx']
                association_matrix         = batch_data[idx]['association_info_aug_dict'][asso_key][0]['distance_matrix']  

            inverse_association_matrix = association_matrix.copy()
            distance_matrix_pred       = data['model_preds']['distance_matrix'][obj_pair_key][idx]['dist_matrix']
            inverse_association_matrix = torch.from_numpy(1.0 / (1.0 + inverse_association_matrix))
            inverse_association_matrix = inverse_association_matrix.cpu().detach().float()

            # Calculate Metrics
            placeholder_dict['forb_norm'].append(matrix_calc_forb_norm(x=inverse_association_matrix, x_pred=distance_matrix_pred))
            placeholder_dict['mse']      .append(matrix_calc_mse(x=inverse_association_matrix, x_pred=distance_matrix_pred))
            placeholder_dict['mae']      .append(matrix_calc_mae(x=inverse_association_matrix, x_pred=distance_matrix_pred))
            # placeholder_dict['ssim']     .append(0) # matrix_calc_ssim(x=inverse_association_matrix, x_pred=model_output['dist_matrix_normal'[0]))
            placeholder_dict['knn']      .append(nearest_neighbors_of_objects(x=inverse_association_matrix, x_pred=distance_matrix_pred))

            # Association Accuracy
            placeholder_dict['asso_acc_v1'] .append(association_accuracy(obj_pairs_gt=associated_objects_idx, dist_mat_pred=distance_matrix_pred))
            obj_pairs = data['model_preds']['pred_emb_pairs'][obj_pair_key]['z_flat']
            placeholder_dict['asso_acc_v2'] .append(association_accuracy(obj_pairs_gt=associated_objects_idx, obj_pairs_preds=obj_pairs))            
            placeholder_dict['number_asso'] .append(len(obj_pairs))

        ## Take the mean over all scenes
        res_obj_pair = {k: float(np.mean(placeholder_dict[k])) for k in placeholder_dict}

        res_dict[obj_pair_key] = res_obj_pair

    return res_dict


def get_fixed_emb_pairs(obj_list_camera, obj_list_lidar):
    assert obj_list_camera.shape == obj_list_lidar.shape, "Must have the same Dimension"
    emb_pairs = []
    for i in range(len(obj_list_camera)):
        pair = {'obj_camera': obj_list_camera[i].tolist(),
                'obj_lidar':  obj_list_lidar[i].tolist(),
                'idx_pair':   i,}
        emb_pairs.append(pair)

    return emb_pairs


def get_emb_pairs(input_data, obj_list_camera, obj_list_lidar, params):
    # Input is scene based and will be flattet afterwards
    dist_matrix      = input_data['dist_matrix']
    emb_matrix_pairs = input_data['matrix_embedding_pairs']

    assert list(dist_matrix.shape) == [len(obj_list_camera), len(obj_list_lidar)], "the shapes must match"
    ### Get the indices of the best matches
    threshold_min_similarity = params['emb_pairs_similarity_min_threshold']
    if params['use_sim_threshold']:
        # Purely based on distance_threshold
        indices_pairs = torch.nonzero(threshold_min_similarity < dist_matrix, as_tuple=False)
    else:        
        # Apply Linear Sum Assignment to find the optimal solution
        # matched_idx_1 = camera_idx, matched_idx_2 = lidar_idx
        matched_idx_1, matched_idx_2 = linear_sum_assignment(dist_matrix, maximize=True)
        indices_pairs_raw = [[x,y] for x,y in zip(matched_idx_1, matched_idx_2)]

        # Verify that the matched indices meets the minimal similarity threshold
        indices_pairs = [indices for indices in indices_pairs_raw if threshold_min_similarity < dist_matrix[tuple(indices)]]

    ### Get the embedding of the best matches
    # Full embeddings (not partitioned)
    emb_pairs = []
    for idx_pair in indices_pairs:
        pair = {'obj_camera': obj_list_camera[idx_pair[0]].tolist(),
                'obj_lidar':  obj_list_lidar[idx_pair[1]].tolist(),
                'idx_pair':   [int(x) for x in idx_pair],}
        emb_pairs.append(pair)



    return emb_pairs


def arrange_embeddings(data_in, params):
    z_dim_ad = params['z_dim_ad']

    ### Single Embeddings
    single_embs = data_in['model_preds']['pred_single_emb']
    res_single_embs = {}
    for obj_key in single_embs: 
        if params['partition_space']:
            res_single_embs[obj_key] = np.array(single_embs[obj_key]['z_flat'])[:,z_dim_ad:]
        else:
            res_single_embs[obj_key] = np.array(single_embs[obj_key]['z_flat'])
    if 'subclass' in list(single_embs.keys())[0]:
        res_single_embs['z_obj_list_camera_and_lidar_subclass_1'] = np.concatenate((single_embs['z_obj_list_camera_subclass_1']['z_flat'], single_embs['z_obj_list_lidar_subclass_1']['z_flat']))
        res_single_embs['z_obj_list_camera_and_lidar_subclass_2'] = np.concatenate((single_embs['z_obj_list_camera_subclass_2']['z_flat'], single_embs['z_obj_list_lidar_subclass_2']['z_flat']))
    else:
        res_single_embs['z_obj_list_camera_and_lidar'] = np.concatenate((single_embs['z_obj_list_camera']['z_flat'], single_embs['z_obj_list_lidar']['z_flat']))

    ### Embedding Pairs
    obj_pairs = data_in['model_preds']['pred_emb_pairs']    
    res_obj_pairs = {}
    for obj_key in obj_pairs:
        obj_pair_list = []
        for obj in obj_pairs[obj_key]['z_flat']:
            
            # Partition latent space
            if params['partition_space']:
                obj_camera = np.array(obj['obj_camera'][z_dim_ad:])
                obj_lidar  = np.array(obj['obj_lidar'][z_dim_ad:])
            else:
                obj_camera = np.array(obj['obj_camera'])
                obj_lidar  = np.array(obj['obj_lidar'])
    
            ### Rearange the embeddings for anomaly detection
            if params['arrangement_method'] == 'diff':
                emb_pairs_arranged = obj_camera - obj_lidar #  [(x[0]-x[1]).tolist() for x in emb_pairs]
                
            elif params['arrangement_method'] == 'concat':
                emb_pairs_arranged = np.concatenate((obj_camera, obj_lidar)) # [torch.cat((x[0], x[1])).tolist() for x in emb_pairs]
                    
            elif params['arrangement_method'] == 'concat+diff':
                emb_pairs_arranged = np.concatenate((obj_camera, obj_lidar, obj_camera-obj_lidar)) # [torch.cat((x[0], x[1], x[0] - x[1] )).tolist() for x in emb_pairs]
                
            elif params['arrangement_method'] == 'camera+diff':
                emb_pairs_arranged = np.concatenate((obj_camera, obj_camera-obj_lidar))   # [torch.cat((x[0], x[0] - x[1])).tolist() for x in emb_pairs]

            else:
                assert False, "Unkown params['arrangement_method']"

            obj_pair_list.append(emb_pairs_arranged)

        res_obj_pairs[obj_key]=np.array(obj_pair_list)
    
    return {'pred_single_embs':  res_single_embs,
            'pred_emb_pairs':    res_obj_pairs}


def run_dataset_and_return_results(model, dataloader, emb_pairs_params, obj_keys_aug={}, keys_association_aug={}, train_set=False, gmm_train_set=False, framework_task="single_object", epoch=1, save_input_data=False, use_sample_data=False,  use_subclasses=False, add_noise_to_data={'enabled': False}):
    n_scenes = len(dataloader)

    
    if use_sample_data:
        obj_keys_aug = ['obj_list_lidar_____sample_aug']
        emb_pairs_params['emb_pair_keys_to_eval'] = ['z_object_pairs_camera_normal_lidar_normal', 'z_object_pairs_camera_normal_lidar_sample_augmented', 'z_object_pairs_camera_sample_aug_lidar_sample_aug']

    obj_keys_single     = ['z_obj_list_camera', 'z_obj_list_lidar'] + ['z_' + k for k in obj_keys_aug] 
    pred_single_emb     = {k: [None]*n_scenes for k in obj_keys_single}
    gmm_test_set = not gmm_train_set
    if not train_set:
        pairs_keys = emb_pairs_params['emb_pair_keys_to_eval']
    else:
        pairs_keys = ['z_object_pairs_camera_normal_lidar_normal']
    pred_emb_pairs      = {k: [None] * n_scenes for k in pairs_keys}
    preds_dist_matrix   = {k: [None] * n_scenes for k in pairs_keys}

    input_data_scenes = [None] * n_scenes

    
    res_keys = ['meta_info', 'association_info_camera_lidar', 'meta_info_normal']
    
    ### Load and iterate dataloader
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader(epoch=0)):

            ### -----------------------------------------------------------------
            ### Normal / not augmented samples -> get embeddings for normal input
            ### -----------------------------------------------------------------
            if 0 == len(input_data['association_info_camera_lidar'][0]['associated_objects_idx']):
                continue

            #--- Execute Model with normal data ---
            model_input = {'obj_list_camera':               input_data['obj_list_camera'],
                           'obj_list_lidar':                input_data['obj_list_lidar'],
                           'obj_ego':                       input_data['obj_ego'],
                           'general_info':                  input_data['general_info'],
                           'association_info_camera_lidar': input_data['association_info_camera_lidar'],}

            if use_sample_data:
                # Use sample data for training (randomly created)
                model_input = create_sample_data_augmentations(n_samples = 100, i=batch_idx, random_seq_length=False, test_set=gmm_test_set) #  5 * len(dataloader)
                model_input['general_info'] = input_data['general_info']
                model_input['obj_ego']      = input_data['obj_ego']


            model_output_normal = model(model_input, epoch)                    
            
            # Single Embeddings
            if 'z_obj_list_camera' in pred_single_emb:
                pred_single_emb['z_obj_list_camera'][batch_idx] = model_output_normal['z_objects_camera'].cpu().detach().numpy()
            if 'z_obj_list_lidar' in pred_single_emb:
                pred_single_emb['z_obj_list_lidar'][batch_idx] = model_output_normal['z_objects_lidar'].cpu().detach().numpy()
            
            if framework_task == "object_pairs":
                # Get emb pairs
                pred_emb_pairs['z_object_pairs_camera_normal_lidar_normal'][batch_idx] = get_fixed_emb_pairs(obj_list_camera=pred_single_emb['z_obj_list_camera'][batch_idx],
                                                                                                             obj_list_lidar =pred_single_emb['z_obj_list_lidar'][batch_idx])

            elif framework_task == "object_asso_object_pairs":
                # Distance Matrix
                preds_dist_matrix['z_object_pairs_camera_normal_lidar_normal'][batch_idx] = {'dist_matrix':            model_output_normal['dist_matrix'].cpu().detach(),
                                                                                             'matrix_embedding_pairs': model_output_normal['matrix_embedding_pairs'].cpu().detach(),
                                                                                             'key_camera':             'obj_list_camera',
                                                                                             'key_lidar':              'obj_list_lidar',}
                
                assert model_output_normal['matrix_embedding_pairs'].shape[:2] == model_output_normal['dist_matrix'].shape, "Must be of same dimension"
                assert pred_single_emb['z_obj_list_camera'][batch_idx].shape[0] == model_output_normal['matrix_embedding_pairs'].shape[0]
                assert pred_single_emb['z_obj_list_lidar'][batch_idx].shape[0] == model_output_normal['matrix_embedding_pairs'].shape[1]
                # Get emb pairs
                pred_emb_pairs['z_object_pairs_camera_normal_lidar_normal'][batch_idx] = get_emb_pairs(preds_dist_matrix['z_object_pairs_camera_normal_lidar_normal'][batch_idx], 
                                                                                                    obj_list_camera=pred_single_emb['z_obj_list_camera'][batch_idx],
                                                                                                    obj_list_lidar =pred_single_emb['z_obj_list_lidar'][batch_idx],
                                                                                                    params=emb_pairs_params)


            ### ------------------------------------------------------------------------------------
            ### Single augmentations (camera aug OR lidar aug) -> get embeddings for augmented input
            ### ------------------------------------------------------------------------------------
            for aug_key in obj_keys_aug:

                if use_sample_data:
                    input_data['obj_lists_aug_dict'][aug_key] = [None] * 5
                    input_data['association_info_aug_dict']['association_info_camera_lidar_' + aug_key[19:]] = [None]
                
                assert aug_key in input_data['obj_lists_aug_dict'], "The augmentation key must be in the input_data dict"
                if 0 < len(input_data['obj_lists_aug_dict'][aug_key]):
                    if 'camera' in aug_key:
                        # Camera augmentation
                        obj_list_camera = input_data['obj_lists_aug_dict'][aug_key]
                        obj_list_lidar  = input_data['obj_list_lidar']
                        key_camera = aug_key
                        key_lidar  = 'obj_list_lidar'
                        obj_pair_key = 'z_object_pairs_camera_' + aug_key[20:] + '_lidar_normal'
                        asso_key     = 'association_info_camera_' + aug_key[20:] + '_lidar'
                    elif 'lidar' in aug_key:
                        # LiDAR augmentation
                        obj_list_camera = input_data['obj_list_camera']
                        obj_list_lidar  = input_data['obj_lists_aug_dict'][aug_key]
                        key_camera = 'obj_list_camera'
                        key_lidar  = aug_key
                        obj_pair_key = 'z_object_pairs_camera_normal_lidar_'+ aug_key[19:]
                        asso_key     = 'association_info_camera_lidar_' + aug_key[19:]
                    else:
                        assert False, "unknown augmentaiton key: " + aug_key

                    assert asso_key in input_data['association_info_aug_dict'], "This key should be in the input_data['association_info_aug_dict']: " + asso_key 

                    #--- Execute Model with augmented data ---
                    model_input_aug = {'obj_list_camera':               obj_list_camera,
                                       'obj_list_lidar':                obj_list_lidar,
                                       'obj_ego':                       input_data['obj_ego'],
                                       'general_info':                  input_data['general_info'],
                                       'association_info_camera_lidar': input_data['association_info_aug_dict'][asso_key],}
                    
                    if add_noise_to_data['enabled']:
                        # add noise to alter one object list very heavily
                        obj_list_lidar_altered = alter_object_list_eval(obj_list_in = model_input_aug['obj_list_lidar'], params= add_noise_to_data)
                        model_input_aug['obj_list_lidar'] = obj_list_lidar_altered

                    if use_sample_data:
                        # Use sample data for training (randomly created)
                        model_input_aug = create_sample_data_augmentations(n_samples=100, aug_lidar=True, i=batch_idx, random_seq_length=False, test_set=gmm_test_set) #  5 * len(dataloader)
                        model_input_aug['general_info'] = input_data['general_info']
                        model_input_aug['obj_ego']      = input_data['obj_ego']
                        obj_pair_key = "z_object_pairs_camera_normal_lidar_sample_augmented"
                        input_data['association_info_aug_dict'][asso_key] = model_input_aug['association_info_camera_lidar']


                    if framework_task == "object_pairs":
                        if 0 == len(input_data['association_info_aug_dict'][asso_key][0]['associated_objects_idx']):
                            continue
                    model_output_aug = model(model_input_aug, epoch)

                    # Single Embeddings
                    key_name = 'z_' + aug_key 
                    if 'camera' in key_name:
                        pred_single_emb[key_name][batch_idx] = model_output_aug['z_objects_camera'].cpu().detach().numpy()   
                    elif 'lidar' in aug_key:        
                        pred_single_emb[key_name][batch_idx] = model_output_aug['z_objects_lidar'].cpu().detach().numpy()    
                    else:
                        assert False, "unknown augmentaiton key: " + key_name

                    
                    if framework_task == "object_pairs":
                        # Get emb pairs
                        pred_emb_pairs[obj_pair_key][batch_idx] = get_fixed_emb_pairs(obj_list_camera = model_output_aug['z_objects_camera'].cpu().detach().numpy(),
                                                                                      obj_list_lidar  = model_output_aug['z_objects_lidar'].cpu().detach().numpy())
                        
                    elif framework_task == "object_asso_object_pairs":
                        assert pred_single_emb[key_name][batch_idx].shape[0] == model_output_aug['matrix_embedding_pairs'].shape[0], "Camera: must be of same dimension" 
                        assert pred_single_emb[key_name][batch_idx].shape[0] == model_output_aug['matrix_embedding_pairs'].shape[1], "LiDAR: must be of same dimension"
                        assert model_output_aug['matrix_embedding_pairs'].shape[:2] == model_output_aug['dist_matrix'].shape, "Emb matrix: must be of same dimension"
                        


                        # Embedding Pairs (with Distance Matrix)
                        preds_dist_matrix[obj_pair_key][batch_idx] = {'dist_matrix':            model_output_aug['dist_matrix'].cpu().detach(),
                                                                      'matrix_embedding_pairs': model_output_aug['matrix_embedding_pairs'].cpu().detach(),
                                                                      'key_camera':             key_camera,
                                                                      'key_lidar':              key_lidar,}                    
                        pred_emb_pairs[obj_pair_key][batch_idx] = get_emb_pairs(preds_dist_matrix[obj_pair_key][batch_idx], 
                                                                                obj_list_camera=pred_single_emb['z_'+key_camera][batch_idx],
                                                                                obj_list_lidar =pred_single_emb['z_'+key_lidar][batch_idx],
                                                                                params=emb_pairs_params)           

            ### ------------------------------------------------------------------------------------
            ### Double augmentations (camera aug AND lidar aug)
            ### ------------------------------------------------------------------------------------
            if 'obj_list_camera_aug_fog_s3' in input_data['obj_lists_aug_dict'] and 'obj_list_lidar_aug_fog_s3' in input_data['obj_lists_aug_dict']:
                if 0 < len(input_data['obj_lists_aug_dict']['obj_list_camera_aug_fog_s3']) and 0 < len(input_data['obj_lists_aug_dict']['obj_list_lidar_aug_fog_s3']):

                    obj_pair_key = 'z_object_pairs_camera_fog_s3_lidar_fog_s3'
                    key_camera   = 'obj_list_camera_aug_fog_s3'
                    key_lidar    = 'obj_list_lidar_aug_fog_s3'                
                    obj_list_camera = input_data['obj_lists_aug_dict'][key_camera]
                    obj_list_lidar  = input_data['obj_lists_aug_dict'][key_lidar]

                    #--- Execute Model with augmented data ---
                    model_input_aug =  {'obj_list_camera':               obj_list_camera,
                                        'obj_list_lidar':                obj_list_lidar,
                                        'obj_ego':                       input_data['obj_ego'],
                                        'general_info':                  input_data['general_info'],
                                        'association_info_camera_lidar': input_data['association_info_aug_dict']['association_info_camera_fog_s3_lidar_fog_s3'],}
                    
                    if add_noise_to_data['enabled']:
                        # add noise to alter one object list very heavily
                        obj_list_camera_altered = alter_object_list_eval(obj_list_in = model_input_aug['obj_list_camera'], params= add_noise_to_data)
                        model_input_aug['obj_list_camera'] = obj_list_camera_altered
                        obj_list_lidar_altered = alter_object_list_eval(obj_list_in = model_input_aug['obj_list_lidar'], params= add_noise_to_data)
                        model_input_aug['obj_list_lidar'] = obj_list_lidar_altered
                        
                    if use_sample_data:
                        # Use sample data for training (randomly created)
                        model_input_aug = create_sample_data_augmentations(n_samples = 100, aug_camera=True, aug_lidar=True, i=batch_idx, random_seq_length=False, test_set=gmm_test_set) #  5 * len(dataloader)
                        model_input_aug['general_info'] = input_data['general_info']
                        model_input_aug['obj_ego']      = input_data['obj_ego']
                        obj_pair_key = 'z_object_pairs_camera_sample_aug_lidar_sample_aug'
                        input_data['association_info_aug_dict']['association_info_camera_fog_s3_lidar_fog_s3'] = model_input_aug['association_info_camera_lidar']
                    
                    if framework_task == "object_pairs":
                        if not use_sample_data and 0 == len(input_data['association_info_aug_dict']['association_info_camera_fog_s3_lidar_fog_s3'][0]['associated_objects_idx']):
                            continue

                    model_output_aug = model(model_input_aug, epoch)


                        
                    if framework_task == "object_pairs":
                        # Get emb pairs
                        pred_emb_pairs[obj_pair_key][batch_idx] = get_fixed_emb_pairs(obj_list_camera = model_output_aug['z_objects_camera'].cpu().detach().numpy(),
                                                                                    obj_list_lidar  = model_output_aug['z_objects_lidar'].cpu().detach().numpy())
                            
                    elif framework_task == "object_asso_object_pairs":
                        assert model_output_aug['matrix_embedding_pairs'].shape[:2] == model_output_aug['dist_matrix'].shape, "Must be of same dimension"
                        assert pred_single_emb['z_obj_list_lidar_aug_fog_s3'][batch_idx].shape[0] == model_output_aug['matrix_embedding_pairs'].shape[1]


                        # Embedding Pairs (with Distance Matrix)
                        preds_dist_matrix[obj_pair_key][batch_idx] = {'dist_matrix':            model_output_aug['dist_matrix'].cpu().detach(),
                                                                    'matrix_embedding_pairs': model_output_aug['matrix_embedding_pairs'].cpu().detach(),
                                                                    'key_camera':             key_camera,
                                                                    'key_lidar':              key_lidar,}                    
                        pred_emb_pairs[obj_pair_key][batch_idx] = get_emb_pairs(preds_dist_matrix[obj_pair_key][batch_idx], 
                                                                                obj_list_camera=pred_single_emb['z_'+key_camera][batch_idx],
                                                                                obj_list_lidar =pred_single_emb['z_'+key_lidar][batch_idx],
                                                                                params=emb_pairs_params)    


            ### Additional Information
            if save_input_data:
                input_data_scenes[batch_idx] = input_data
            else:
                # Only process and save the used features, elsewise this is expensive for the whole dataset
                input_data_scenes[batch_idx] = {'association_info_camera_lidar': input_data['association_info_camera_lidar'],
                                                'general_info':                  input_data['general_info'],
                                                'association_info_aug_dict':     input_data['association_info_aug_dict'],}
                
                for key_asso_aug in keys_association_aug:
                    if key_asso_aug in input_data['association_info_aug_dict']:
                        input_data_scenes[batch_idx][key_asso_aug] = input_data['association_info_aug_dict'][key_asso_aug][0]



    # Flatten single prediction results and add to res_dict
    for key in pred_single_emb:
        pred_single_emb[key] = [x for x in pred_single_emb[key] if x is not None]
    for key in pred_emb_pairs:
        pred_emb_pairs[key] = [x for x in pred_emb_pairs[key] if x is not None]

    save_z_list = False     # only needed when intended to associate embeddings to scenes
    pred_single_emb_flat_list = {}
    for key in pred_single_emb:
        if pred_single_emb==[]:
            debug_here = 0
        z_flat, z_list = flat_data(data_nested=pred_single_emb[key])
        if save_z_list:
            pred_single_emb_flat_list[key] = {'z_list': z_list,
                                              'z_flat': z_flat,}
        else:
            pred_single_emb_flat_list[key] = {'z_flat': z_flat,}

    pred_emb_pairs_flat_list = {}
    for key in pred_emb_pairs:
        if pred_single_emb==[]:
            debug_here = 0
        z_flat, z_list = flat_data(data_nested=pred_emb_pairs[key])
        if save_z_list:
            pred_emb_pairs_flat_list[key] = {'z_list': z_list,
                                            'z_flat': z_flat,}
        else:
            pred_emb_pairs_flat_list[key] = {'z_flat': z_flat,}


    res_dict = {'model_preds': {'pred_single_emb':  pred_single_emb_flat_list,
                                'pred_emb_pairs':   pred_emb_pairs_flat_list,
                                'distance_matrix':  preds_dist_matrix},
                'model_input':  input_data_scenes,
                'general_info': input_data['general_info'],
                }

    return res_dict 


def run_dataset_and_return_results_subclasses(model, dataloader, emb_pairs_params, subclass_settings, framework_task="single_object", epoch=1, use_sample_data=False):
    n_scenes = len(dataloader)
        
    obj_keys_single     = ['z_obj_list_camera_subclass_1', 'z_obj_list_lidar_subclass_1', 'z_obj_list_camera_subclass_2', 'z_obj_list_lidar_subclass_2']
    pred_single_emb     = {k: [None]*n_scenes for k in obj_keys_single}

    emb_pairs_params['emb_pair_keys_to_eval'] = ['z_object_pairs_sample_subclass_1', 'z_object_pairs_sample_subclass_2']
    pred_emb_pairs      = {k: [None] * n_scenes for k in emb_pairs_params['emb_pair_keys_to_eval']}

    ### Load and iterate dataloader
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader(epoch=0)):


            if not use_sample_data:
                if subclass_settings['subclass_mode'] == 'offset':
                    # Add an offset to one value (to simulate sensor malfunction) and treat this as other subclass
                    data_subclasses = create_offset_subclasses(input_data, subclass_params=subclass_settings['subclass_offset_params'])
                    model_input_subclass_1 = data_subclasses['batch_subclass_1']
                    model_input_subclass_2 = data_subclasses['batch_subclass_2']
            else:
                # Use sample data for training (randomly created)
                # Create sample data for the subclass 1
                model_input_subclass_1 = create_sample_data_subclasses(n_samples=100, i=batch_idx, subclass=1)
                model_input_subclass_1['general_info'] = input_data['general_info']
                model_input_subclass_1['obj_ego']      = input_data['obj_ego']
                # Create sample data for the subclass 2
                model_input_subclass_2 = create_sample_data_subclasses(n_samples=100, i=batch_idx, subclass=2)
                model_input_subclass_2['general_info'] = input_data['general_info']
                model_input_subclass_2['obj_ego']      = input_data['obj_ego']


            #-----------------------------------------------
            #--- Execute Model with data from subclass 1 ---
            if 0 != len(model_input_subclass_1['association_info_camera_lidar'][0]['associated_objects_idx']):

                model_output_subclass_1 = model(model_input_subclass_1, epoch)                    
                
                # Single Embeddings
                if 'z_obj_list_camera_subclass_1' in pred_single_emb:
                    pred_single_emb['z_obj_list_camera_subclass_1'][batch_idx] = model_output_subclass_1['z_objects_camera'].cpu().detach().numpy()
                if 'z_obj_list_lidar_subclass_1' in pred_single_emb:
                    pred_single_emb['z_obj_list_lidar_subclass_1'][batch_idx] = model_output_subclass_1['z_objects_lidar'].cpu().detach().numpy()
                
                if framework_task == "object_pairs":
                    # Get emb pairs
                    pred_emb_pairs['z_object_pairs_sample_subclass_1'][batch_idx] = get_fixed_emb_pairs(obj_list_camera=pred_single_emb['z_obj_list_camera_subclass_1'][batch_idx],
                                                                                                        obj_list_lidar =pred_single_emb['z_obj_list_lidar_subclass_1'][batch_idx])
                
            #-----------------------------------------------
            #--- Execute Model with data from subclass 2 ---
            if 0 != len(model_input_subclass_2['association_info_camera_lidar'][0]['associated_objects_idx']):

                model_output_subclass_2 = model(model_input_subclass_2, epoch)                    
                
                # Single Embeddings
                if 'z_obj_list_camera_subclass_2' in pred_single_emb:
                    pred_single_emb['z_obj_list_camera_subclass_2'][batch_idx] = model_output_subclass_2['z_objects_camera'].cpu().detach().numpy()
                if 'z_obj_list_lidar_subclass_2' in pred_single_emb:
                    pred_single_emb['z_obj_list_lidar_subclass_2'][batch_idx] = model_output_subclass_2['z_objects_lidar'].cpu().detach().numpy()
    
                if framework_task == "object_pairs":
                    # Get emb pairs
                    pred_emb_pairs['z_object_pairs_sample_subclass_2'][batch_idx] = get_fixed_emb_pairs(obj_list_camera=pred_single_emb['z_obj_list_camera_subclass_2'][batch_idx],
                                                                                                        obj_list_lidar =pred_single_emb['z_obj_list_lidar_subclass_2'][batch_idx])               

    # Flatten single prediction results and add to res_dict
    for key in pred_single_emb:
        pred_single_emb[key] = [x for x in pred_single_emb[key] if x is not None]
    for key in pred_emb_pairs:
        pred_emb_pairs[key] = [x for x in pred_emb_pairs[key] if x is not None]

    save_z_list = False     # only needed when intended to associate embeddings to scenes
    pred_single_emb_flat_list = {}
    for key in pred_single_emb:
        if pred_single_emb==[]:
            debug_here = 0
        z_flat, z_list = flat_data(data_nested=pred_single_emb[key])
        if save_z_list:
            pred_single_emb_flat_list[key] = {'z_list': z_list,
                                              'z_flat': z_flat,}
        else:
            pred_single_emb_flat_list[key] = {'z_flat': z_flat,}

    pred_emb_pairs_flat_list = {}
    for key in pred_emb_pairs:
        if pred_single_emb==[]:
            debug_here = 0
        z_flat, z_list = flat_data(data_nested=pred_emb_pairs[key])
        if save_z_list:
            pred_emb_pairs_flat_list[key] = {'z_list': z_list,
                                             'z_flat': z_flat,}
        else:
            pred_emb_pairs_flat_list[key] = {'z_flat': z_flat,}


    res_dict = {'model_preds': {'pred_single_emb':  pred_single_emb_flat_list,
                                'pred_emb_pairs':   pred_emb_pairs_flat_list,
                                },
                'model_input':  None,
                'general_info': input_data['general_info'],
                }

    return res_dict 


def copy_metrics_to_wandb_dict(epoch, res_dict_association_metrics, metric_results, res_dict_ad):

    log_dict = {"Epoch": epoch}

    # Association Info
    if type(res_dict_association_metrics) == dict:
        for data_split_key in res_dict_association_metrics:
            for augmentation_key in res_dict_association_metrics[data_split_key]: 
                for metric_key in res_dict_association_metrics[data_split_key][augmentation_key]:
                    entry_key = "Object Asso ("+augmentation_key[15:]+"): " + metric_key + "("+data_split_key+")"
                    log_dict[entry_key] = res_dict_association_metrics[data_split_key][augmentation_key][metric_key]
    


    # Anomaly Detection Info
    for emb_type in metric_results['res_dict_ad']:
        for key in res_dict_ad[emb_type]:
            # GMM class prob
            if 'GMM_class_prob' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:
                log_dict["AD z: GMM class prob - f1 (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_class_prob']['metrics_test']['f1_score']
                log_dict["AD z: GMM class prob - f1 (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_class_prob']['metrics_val']['f1_score']
            # GMM log-likelihood
            if 'GMM_log_likelihood_score' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:
                log_dict["AD z: GMM log-likelihood - accuracy (test) " + key + " " + emb_type]       = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['accuracy_overalll']                
                log_dict["AD z: GMM log-likelihood - accuracy (val) " + key + " " + emb_type]        = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['accuracy_overalll']
                log_dict["AD z: GMM log-likelihood - f1 (test) " + key + " " + emb_type]             = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['f1_score']
                log_dict["AD z: GMM log-likelihood - f1 (val) " + key + " " + emb_type]                         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['f1_score']
                log_dict["AD z: GMM log-likelihood - roc_auc_score (test) " + key + " " + emb_type]  = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['roc_auc_score']
                log_dict["AD z: GMM log-likelihood - roc_auc_score (val) " + key + " " + emb_type]   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['roc_auc_score']
                log_dict["AD z: GMM log-likelihood - mcc (test) " + key + " " + emb_type]            = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['mcc']
                log_dict["AD z: GMM log-likelihood - mcc (val) " + key + " " + emb_type]             = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['mcc']
                log_dict["AD z: GMM log-likelihood - far (test) " + key + " " + emb_type]            = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['fpr']
                log_dict["AD z: GMM log-likelihood - far (val) " + key + " " + emb_type]             = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['fpr']
                log_dict["AD z: GMM log-likelihood - tpr (test) " + key + " " + emb_type]            = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_test']['tpr']
                log_dict["AD z: GMM log-likelihood - tpr (val) " + key + " " + emb_type]             = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['GMM_log_likelihood_score']['metrics_val']['tpr']
            # COPOD      
            if 'COPOD' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:          
                log_dict["AD z: COPOD - accuracy (test) " + key + " " + emb_type]              = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_test']['accuracy_overalll']
                log_dict["AD z: COPOD - accuracy (val) " + key + " " + emb_type]               = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_val']['accuracy_overalll']
                log_dict["AD z: COPOD - f1 (test) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_test']['f1_score']
                log_dict["AD z: COPOD - f1 (val) " + key + " " + emb_type]                     = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_val']['f1_score']
                log_dict["AD z: COPOD - roc_auc_score (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_test']['roc_auc_score']
                log_dict["AD z: COPOD - roc_auc_score (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_val']['roc_auc_score']
                log_dict["AD z: COPOD - far (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_test']['fpr']
                log_dict["AD z: COPOD - far (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_val']['fpr']
                log_dict["AD z: COPOD - tpr (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_test']['tpr']
                log_dict["AD z: COPOD - tpr (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['COPOD']['metrics_val']['tpr']
            # ABOD    
            if 'ABOD' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:            
                log_dict["AD z: ABOD - accuracy (test) " + key + " " + emb_type]              = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_test']['accuracy_overalll']
                log_dict["AD z: ABOD - accuracy (val) " + key + " " + emb_type]               = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_val']['accuracy_overalll']
                log_dict["AD z: ABOD - f1 (test) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_test']['f1_score']
                log_dict["AD z: ABOD - f1 (val) " + key + " " + emb_type]                     = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_val']['f1_score']
                log_dict["AD z: ABOD - roc_auc_score (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_test']['roc_auc_score']
                log_dict["AD z: ABOD - roc_auc_score (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_val']['roc_auc_score']
                log_dict["AD z: ABOD - far (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_test']['fpr']
                log_dict["AD z: ABOD - far (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_val']['fpr']
                log_dict["AD z: ABOD - tpr (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_test']['tpr']
                log_dict["AD z: ABOD - tpr (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['ABOD']['metrics_val']['tpr']
            # LOF   
            if 'LOF' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:             
                log_dict["AD z: LOF - accuracy (test) " + key + " " + emb_type]              = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_test']['accuracy_overalll']
                log_dict["AD z: LOF - accuracy (val) " + key + " " + emb_type]               = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_val']['accuracy_overalll']
                log_dict["AD z: LOF - f1 (test) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_test']['f1_score']
                log_dict["AD z: LOF - f1 (val) " + key + " " + emb_type]                     = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_val']['f1_score']
                log_dict["AD z: LOF - roc_auc_score (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_test']['roc_auc_score']
                log_dict["AD z: LOF - roc_auc_score (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_val']['roc_auc_score']
                log_dict["AD z: LOF - far (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_test']['fpr']
                log_dict["AD z: LOF - far (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_val']['fpr']
                log_dict["AD z: LOF - tpr (test) " + key + " " + emb_type]                   = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_test']['tpr']
                log_dict["AD z: LOF - tpr (val) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['LOF']['metrics_val']['tpr']
            # IsolationForest  
            if 'IsolationForest' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:              
                log_dict["AD z: IF - accuracy (test) " + key + " " + emb_type]              = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_test']['accuracy_overalll']
                log_dict["AD z: IF - accuracy (val) " + key + " " + emb_type]               = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_val']['accuracy_overalll']
                log_dict["AD z: IF - f1 (test) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_test']['f1_score']
                log_dict["AD z: IF - f1 (val) " + key + " " + emb_type]                     = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_val']['f1_score']
                log_dict["AD z: IF - roc_auc_score (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_test']['roc_auc_score']
                log_dict["AD z: IF - roc_auc_score (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['IsolationForest']['metrics_val']['roc_auc_score']
            # DBSCAN    
            if 'DBSCAN' in metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']:            
                log_dict["AD z: DBSCAN - accuracy (test) " + key + " " + emb_type]              = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_test']['accuracy_overalll']
                log_dict["AD z: DBSCAN - accuracy (val) " + key + " " + emb_type]               = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_val']['accuracy_overalll']
                log_dict["AD z: DBSCAN - f1 (test) " + key + " " + emb_type]                    = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_test']['f1_score']
                log_dict["AD z: DBSCAN - f1 (val) " + key + " " + emb_type]                     = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_val']['f1_score']
                log_dict["AD z: DBSCAN - roc_auc_score (test) " + key + " " + emb_type]         = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_test']['roc_auc_score']
                log_dict["AD z: DBSCAN - roc_auc_score (val) " + key + " " + emb_type]          = metric_results['res_dict_ad'][emb_type][key]['anomaly_detection_results']['DBSCAN']['metrics_val']['roc_auc_score']


    return log_dict

def eval_embedding_distribution(z_train, z_val, z_test, train_unequals_val):
    # Try to obtain some insights about the distribution within the embedding space
    # Within a class (z_val_normal or z_val_anomaly):  
    #       - Mean: Can be close to 0, but not necessarily — depends on the center of the learned cluster.
    #       - Variance: Should be low — tight, consistent embeddings within a class.
    #       - Cosine similarity: Should be high (⟶ close to 1) — embeddings lie close in direction.
    # Across classes (z_val_all = z_val_normal + z_val_anomaly):
    #       - Mean: Might still be around 0 (if the embeddings are symmetric or centered around origin), but this is less important than variance and similarity.
    #       - Variance: Should be higher — different classes occupy different regions in the space.
    #       - Cosine similarity: Should be low (⟶ ideally < 0.5) — dissimilar classes have distinct directions.

    res_dict = {}
    normal_key  = list(z_val['pred_emb_pairs'].keys())[0]
    anomaly_key = list(z_val['pred_emb_pairs'].keys())[1]


    ### Val-Set
    z_val_normal = z_val['pred_emb_pairs'][normal_key]
    res_dict['z_val_normal mean']     = np.mean(z_val_normal)
    res_dict['z_val_normal variance'] = np.var(z_val_normal)
    res_dict['z_val_normal cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_val_normal).unsqueeze(1), torch.tensor(z_val_normal).unsqueeze(0), dim=2).mean().item()
    z_val_anomaly = z_val['pred_emb_pairs'][anomaly_key]
    res_dict['z_val_anomaly mean']     = np.mean(z_val_anomaly)
    res_dict['z_val_anomaly variance'] = np.var(z_val_anomaly)
    res_dict['z_val_anomaly cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_val_anomaly).unsqueeze(1), torch.tensor(z_val_anomaly).unsqueeze(0), dim=2).mean().item()
    z_val_all = torch.cat((torch.tensor(z_val_normal), torch.tensor(z_val_anomaly)),dim=0)
    res_dict['z_val_all mean']     = np.mean(z_val_all.numpy())
    res_dict['z_val_all variance'] = np.var(z_val_all.numpy())
    res_dict['z_val_all cosine_sim ideally < 0.5)'] = F.cosine_similarity(z_val_all.unsqueeze(1), z_val_all.unsqueeze(0), dim=2).mean().item()

    ### Test-Set
    z_test_normal = z_test['pred_emb_pairs'][normal_key]
    res_dict['z_test_normal mean']     = np.mean(z_test_normal)
    res_dict['z_test_normal variance'] = np.var(z_test_normal)
    res_dict['z_test_normal cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_test_normal).unsqueeze(1), torch.tensor(z_test_normal).unsqueeze(0), dim=2).mean().item()
    z_test_anomaly = z_test['pred_emb_pairs'][anomaly_key]
    res_dict['z_test_anomaly mean']     = np.mean(z_test_anomaly)
    res_dict['z_test_anomaly variance'] = np.var(z_test_anomaly)
    res_dict['z_test_anomaly cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_test_anomaly).unsqueeze(1), torch.tensor(z_test_anomaly).unsqueeze(0), dim=2).mean().item()
    z_test_all = torch.cat((torch.tensor(z_test_normal), torch.tensor(z_test_anomaly)),dim=0)
    res_dict['z_test_all mean']     = np.mean(z_test_all.numpy())
    res_dict['z_test_all variance'] = np.var(z_test_all.numpy())
    res_dict['z_test_all cosine_sim ideally < 0.5)'] = F.cosine_similarity(z_test_all.unsqueeze(1), z_test_all.unsqueeze(0), dim=2).mean().item()

    ### Train-Set
    res_dict['train-set == val-set'] = not train_unequals_val
    z_train_normal = z_train['pred_emb_pairs'][normal_key]
    res_dict['z_train_normal mean']     = np.mean(z_train_normal)
    res_dict['z_train_normal variance'] = np.var(z_train_normal)
    res_dict['z_train_normal cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_train_normal).unsqueeze(1), torch.tensor(z_train_normal).unsqueeze(0), dim=2).mean().item()
    z_train_anomaly = z_train['pred_emb_pairs'][anomaly_key]
    res_dict['z_train_anomaly mean']     = np.mean(z_train_anomaly)
    res_dict['z_train_anomaly variance'] = np.var(z_train_anomaly)
    res_dict['z_train_anomaly cosine_sim (ideal 0.9-1)'] = F.cosine_similarity(torch.tensor(z_train_anomaly).unsqueeze(1), torch.tensor(z_train_anomaly).unsqueeze(0), dim=2).mean().item()
    z_train_all = torch.cat((torch.tensor(z_train_normal), torch.tensor(z_train_anomaly)),dim=0)
    res_dict['z_train_all mean']     = np.mean(z_train_all.numpy())
    res_dict['z_train_all variance'] = np.var(z_train_all.numpy())
    res_dict['z_train_all cosine_sim ideally < 0.5)'] = F.cosine_similarity(z_train_all.unsqueeze(1), z_train_all.unsqueeze(0), dim=2).mean().item()

    return res_dict


class eval_153(eval):
    def __init__(self,
                 idx=153,
                 framework_task='single_object',
                 output_dir=None,
                 z_dim_m = 16,
                 partition_z_space = False,
                 z_dim_ad = 8,
                 fit_ad_method_with_train_set = False,
                 augmentations_to_load = [],
                 augmentations_to_load_associations = [],
                 add_noise_to_data = {},
                 smart_diffs_min_dist_threshold = 0.75,
                 ad_methods_settings = {},
                 emb_pairs_params = {},
                 name  = "save_embeddings",
                 input_="encodings",
                 output="embeddings as json file",
                 description="Save embeddings after the encoder (latent-space) for the umap visualization.",
                 pred_=50,
                 use_sample_data = False,
                 subclass_settings = {'subclasses_enabled': False},
                ):
        super().__init__(idx,
                         name,
                         input_,
                         output,
                         description)
        self.framework_task = framework_task
        self.output_dir = output_dir
        self.pred_frames = pred_
        self.partition_z_space = partition_z_space
        self.z_dim_m = z_dim_m
        self.z_dim_ad = z_dim_ad
        self.fit_ad_method_with_train_set = fit_ad_method_with_train_set
        self.augmentations_to_load = augmentations_to_load
        self.augmentations_to_load_associations = augmentations_to_load_associations
        self.add_noise_to_data = add_noise_to_data
        self.ad_methods_settings   = ad_methods_settings
        self.emb_pairs_params      = emb_pairs_params
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.test_file_order = False
        self.use_sample_data = use_sample_data
        self.subclass_settings  = subclass_settings

        
                          
        
    def __call__(self, model, dataloader_train, dataloader_val, dataloader_test, run_name, epoch=0):
        self._evaluate(model, dataloader_train, dataloader_val, dataloader_test, run_name, epoch)


    def _evaluate(self, model, dataloader_train, dataloader_val, dataloader_test, run_name, epoch=0):
        '''
        This functions receives the model (at current training state), the dataloader and dataset reference. 
        For the whole dataset the embeddings (z-space) are created: z = model(traj_pair).
        These embeddings are saved as json file, to be used for the umap visualization.
        '''
        model.to(self.device)
        model.eval()

        ########################################################################################
        ### Run data splits and obtain data
        ########################################################################################
        if self.subclass_settings['subclasses_enabled']:         
            # Create results for subclasses 
            results_val   = run_dataset_and_return_results_subclasses(model, dataloader=dataloader_val, emb_pairs_params=self.emb_pairs_params, subclass_settings=self.subclass_settings, 
                                                                      framework_task=self.framework_task, use_sample_data=self.use_sample_data)
            results_test  = run_dataset_and_return_results_subclasses(model, dataloader=dataloader_test, emb_pairs_params=self.emb_pairs_params, subclass_settings=self.subclass_settings, 
                                                                      framework_task=self.framework_task, use_sample_data=self.use_sample_data)
            if self.fit_ad_method_with_train_set:
                results_train = run_dataset_and_return_results_subclasses(model, dataloader=dataloader_train, emb_pairs_params=self.emb_pairs_params, subclass_settings=self.subclass_settings, 
                                                                          framework_task=self.framework_task, use_sample_data=self.use_sample_data)
            else: 
                results_train = results_val

        else:
            # Create results for augmentations
            results_val   = run_dataset_and_return_results(model, dataloader=dataloader_val, emb_pairs_params=self.emb_pairs_params, obj_keys_aug=self.augmentations_to_load, keys_association_aug=self.augmentations_to_load_associations, epoch=1, framework_task = self.framework_task, use_sample_data = self.use_sample_data , train_set=False, gmm_train_set=True, add_noise_to_data=self.add_noise_to_data)
            results_test  = run_dataset_and_return_results(model, dataloader=dataloader_test,  emb_pairs_params=self.emb_pairs_params, obj_keys_aug=self.augmentations_to_load, keys_association_aug=self.augmentations_to_load_associations, epoch=1, framework_task = self.framework_task, use_sample_data = self.use_sample_data , train_set=False, gmm_train_set=False, add_noise_to_data=self.add_noise_to_data)
            if self.fit_ad_method_with_train_set:
                results_train = run_dataset_and_return_results(model, dataloader=dataloader_train, emb_pairs_params=self.emb_pairs_params, epoch=1, train_set=True, gmm_train_set=False, framework_task = self.framework_task, use_sample_data = self.use_sample_data )
            else: 
                results_train = results_val
        
        ########################################################################################
        ### Evaluate Results / Apply Metrics
        ########################################################################################
        ### Association Metrics
        if self.framework_task == "object_asso_object_pairs":
            association_metrics_val   = calculate_asso_metrics(data=results_val)
            association_metrics_test  = calculate_asso_metrics(data=results_test)
            association_metrics_train = calculate_asso_metrics(data=results_train)
            res_dict_association_metrics = {'val':   association_metrics_val,
                                            'test':  association_metrics_test,
                                            'train': association_metrics_train}
        else:
            res_dict_association_metrics = "not available for the task: " + self.framework_task

        
        ### Prepare data for anomaly detection (partition latent space / arrange data)
        params_arrange_embeddings = {'partition_space':       self.partition_z_space,
                                     'z_dim_ad':              self.z_dim_ad,
                                     'arrangement_method':    self.emb_pairs_params['arrangement_method'],}
        results_train_ad = arrange_embeddings(results_train, params_arrange_embeddings)
        results_val_ad   = arrange_embeddings(results_val,   params_arrange_embeddings)
        results_test_ad  = arrange_embeddings(results_test,  params_arrange_embeddings)
        
        ### Evaluate the embedding distribution
        # res_emb_space_eval = eval_embedding_distribution(z_train = results_train_ad, z_val = results_val_ad, z_test= results_test_ad, train_unequals_val=self.fit_ad_method_with_train_set)
        res_emb_space_eval = {}

        ### Anomaly Detection for Smart Diffs and Single Embeddings
        res_dict_ad = perform_anomaly_detection(data_val          = results_val_ad,  
                                                data_test         = results_test_ad,
                                                data_train        = results_train_ad,
                                                ad_methods_params = self.ad_methods_settings,
                                                epoch             = epoch,
                                                partition_z_space = self.partition_z_space,
                                                use_subclasses    = self.subclass_settings['subclasses_enabled'])

        ########################################################################################
        ### Save results
        ########################################################################################
        ### Save Embeddings and meta info
        metric_results = {'res_dict_association_metrics':   res_dict_association_metrics,
                          'res_dict_ad':                    res_dict_ad,
                          'res_emb_space_eval':             res_emb_space_eval,}
        save_metrics(metric_results  = metric_results,
                     run_name        = run_name,
                     epoch           = epoch,
                     settings        = {'emb_pairs_params':  self.emb_pairs_params, 
                                        'obj_keys_aug':      self.augmentations_to_load,
                                        'z_dim_m':           self.z_dim_m,},
                     output_dir      = self.output_dir,)
        save_latent_embeddings_per_scene(results_val        = results_val, 
                                         results_test       = results_test,
                                         results_train      = results_train,
                                         metric_results     = metric_results,
                                         run_name           = run_name,
                                         epoch              = epoch,
                                         settings           = {'emb_pairs_params':  self.emb_pairs_params, 
                                                               'obj_keys_aug':      self.augmentations_to_load,
                                                               'z_dim_m':           self.z_dim_m,},
                                         output_dir         = self.output_dir,
                                         res_keys_2_save    = ['meta_info'])
        
        ### Log Metrics in wandb
        log_dict = copy_metrics_to_wandb_dict(epoch=epoch, 
                                              res_dict_association_metrics=res_dict_association_metrics, 
                                              metric_results=metric_results, 
                                              res_dict_ad=res_dict_ad)
        wandb.log(log_dict)  

        # Reset Model to train-mode
        model.train()
