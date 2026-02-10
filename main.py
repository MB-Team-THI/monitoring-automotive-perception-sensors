import os
import math
import wandb
from datetime import datetime
from fileinput import filename

from experiment import Experiment
from src.utils.json_files import check_dir_and_save_json


save_config = True
meta_info = {"name":"contrastive learning approach", 
             "description": "The independently processed object lists from camera and LiDAR sensors provide two representations of the same traffic scene. \
                             These different views are incooperated within a contrastive learning architecture to train a object encoder and shape a latent representation space. \
                             The created latent representation space is utilized to finde discrepancies (anomalies) between the camera and LiDAR object representations.",
            }


wandb_project_name =  "TBD"  

out_dir = 'result_folder'
a = datetime.now()  
run_name = (datetime.today().strftime('%Y%m%d')  + '_' + "%s_%s" % (a.hour, a.minute) + "_contrastive_learning_approach")
output_dir = os.path.join(out_dir, run_name)



# Dataloader settings
use_sample_data = False
representation_types = ["object_pair", "obj_lists_scene"]
representation_type  = representation_types[1]
useEgoCentricCoord   = False

subclasses_offset = {'velocity': {'feature_name':       'v',
                                  'offset_val':         5,
                                  'obj2alter':          'lidar',
                                  'add_sample_value':   False,},
                     'heading':  {'feature_name':       'heading',
                                  'offset_val':         (math.pi / 4),
                                  'obj2alter':          'lidar',
                                  'add_sample_value':   False,},
                    }

subclass_modes = ['offset']
subclass_settings = {'subclasses_enabled':      True,
                     'subclass_mode':           subclass_modes[0],
                     'subclass_offset_params':  subclasses_offset['velocity'],}

# train idx + hypers                                                                
training = {'idx': 153, 'traintest': [70, 30], 'num_gpus': 1, 'eval_epochs': [0, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400],
            'enable_grad_clip': False, 'clip_value': 5, 'use_sample_data': use_sample_data, 'subclass_settings': subclass_settings,
            'save_embeddings': ['train-set', 'test-set'],
            }

# dataset idx + hypers
dataset_train = [{'name':                   'nuscenes',
                  'augmentation_type':      ["no"],
                  'mode':                   'train',
                  'bbox_meter':             [200, 200],
                  'bbox_pixel':             [100,100],
                  'center_meter':           [100.0,100.0],
                  'hist_seq_first':         0,
                  'hist_seq_last':          49,
                  'representation_type':    representation_type,
                  'useEgoCentricCoord':     useEgoCentricCoord,
                  'subdivisions_params' :   {'distance_threshold': 30},
                  'rotation_type':          'ego',
                  'orientation':            'north',
                  'augmentations_to_load':  [],
                  'smooth_heading_values':  True,
                  'min_obj_length':         0,      
                  }]

dataset_val   = [{'name':                   'nuscenes', 
                  'augmentation_type':      ["no"], 
                  'mode':                   'val', 
                  'bbox_meter':             [200, 200], 
                  'bbox_pixel':             [100, 100], 
                  'center_meter':           [100.0, 100.0],
                  'hist_seq_first':         0, 
                  'hist_seq_last':          49,
                  'representation_type':    representation_type,
                  'useEgoCentricCoord':     useEgoCentricCoord,
                  'subdivisions_params':    {'distance_threshold': 30},
                  'rotation_type':          'ego',
                  'orientation':            'north',
                  'augmentations_to_load':  [],
                  'augmentations_to_load_associations': [],
                  'smooth_heading_values':  True,
                  'min_obj_length':         0,  
                  }]

dataset_test  = [{'name':                   'nuscenes', 
                  'augmentation_type':      ["no"], 
                  'mode':                   'test', 
                  'bbox_meter':             [200, 200], 
                  'bbox_pixel':             [100, 100], 
                  'center_meter':           [100.0, 100.0],
                  'hist_seq_first':         0, 
                  'hist_seq_last':          49,
                  'representation_type':    representation_type,
                  'useEgoCentricCoord':     useEgoCentricCoord,
                  'subdivisions_params':    {'distance_threshold': 30},
                  'rotation_type':          'ego',
                  'orientation':            'north',
                  'augmentations_to_load':  dataset_val[0]['augmentations_to_load'],
                  'augmentations_to_load_associations':  dataset_val[0]['augmentations_to_load_associations'],
                  'smooth_heading_values':  True,
                  'min_obj_length':         0,
                  }]


# dataloader idx + hypers
train_dataloader = {'idx':153, 'batch_size': 1, 'epochs': 400, 
                    'num_workers':1, 'shuffle': True, 'representation':'trajectory'}
val_dataloader   = {'idx':153, 'batch_size': 1, 'num_workers':8,
                    'shuffle': False, 'representation':'trajectory'}
test_dataloader  = {'idx':153, 'batch_size': 1, 'num_workers':8, 
                    'shuffle': False, 'representation':'trajectory'}

# model idx + hypers
encoderT_types = ["LSTM-Encoder", "CNN-Encoder", "Transformer-Encoder"]
merger_types   = ["None", "Merger-Transformer-v1", "Merger-Transformer-v2"]
N_z = 64
N_h = 32
architecture_types = ["scene_encoder_transformer"]

model =  {'idx':                          153, 
          'framework_task':               'object_pairs',     
          'architecture_type':            architecture_types[0],
          'encoderT_type':                encoderT_types[2],
          'encoderT_args':                {'joint_encoder':   True,
                                           'depth':           6,
                                           'heads':           8,
                                           'dim_trans':       128, 
                                           'dim_mlp':         128,
                                           'dim_out':         N_z,
                                           'dropout_rate':    0.2,
                                           'use_scaled_sinu_pos_emb': True,
                                           'output_cls_token': True,
                                           'output_mean_var':  False,},
          'contrastive_args':             {'contrastive_loss_enabled': True,
                                           'contrastive_loss_tpye':    "nt_xent_loss",
                                           'projection_head_enabled':  True,
                                           'type':        'MLP',
                                           'n_layer':     3,
                                           'dim_in':      N_z,
                                           'dim_hidden':  48,
                                           'dim_out':     N_h,
                                           'temp':        0.5,
                                           'similarity_m': 'cosine',
                                           'hard_neg_params':          {'enabled': True,
                                                                        'realistic_negatives': False,
                                                                        'normal_dist_var': 1,
                                                                        'add_gaussian_noise_enabled': True,
                                                                         'delete_random_timesteps_enabled': False,},
                                            },
          'merger_type':                  merger_types[0],
          'merger_args':                  {'dim_in':      N_h,      # same as 'dim_out' of 'projection_n_args' or 'encoderT_args'
                                           'dim_trans':   128,
                                           'dim_mlp':     128, 
                                           'depth':       6,
                                           'heads':       8,}, 
          'partition_z_space':            False,
          'z_dim_m':                      N_z,                      # 'dim_out' of 'encoderT_args' or for "Merger-Transformer-v1" of 'merger_args'
          'z_dim_ad':                     int(N_h/2),
          'input_obj_features':           ['time_idx_in_scenario_frame', 'x', 'y', 'heading', 'v'],
          'obj_pair_starts_at_0':         True,      
          'matrix_distance_measure':     'cosine',
         }

framework_task = model['framework_task']

types_arrangement_emb_pairs = ['diff', 'concat', 'concat+diff', 'camera+diff']
evaluation = {"idx":                        153,
              "use_sample_data":            use_sample_data,
              "subclass_settings":          subclass_settings,
              "framework_task":             framework_task,
              "output_dir":                 output_dir,
              "z_dim_m":                    model['z_dim_m'],
              "partition_z_space":          model['partition_z_space'],
              "z_dim_ad":                   model['z_dim_ad'],     
              "fit_ad_method_with_train_set": False,
              "augmentations_to_load":      dataset_val[0]['augmentations_to_load'],
              "augmentations_to_load_associations": dataset_val[0]['augmentations_to_load_associations'],
              "add_noise_to_data":         {'enabled':          False,
                                            'normal_dist_var':  50,
                                            'jitter_max':       5,
                                            'jitter_weight':    3,},
              "emb_pairs_params":          {'emb_pairs_similarity_min_threshold':  0.75,
                                            'use_sim_threshold':                   False,
                                            'arrangement_method':                  types_arrangement_emb_pairs[3], 
                                            'emb_pair_keys_to_eval':               []},
              "ad_methods_settings":       {'GMM':    {'enabled':               True,
                                                        'n_comp':               25,
                                                        'scale_data':           False,
                                                        'scale_log_scores':     False,
                                                        'analyse_gmm':          True,
                                                        'output_dir':           output_dir,
                                                        'use_log_likelihood':   True,
                                                        'use_max_cluster_probas': True,
                                                        'log_likelihood_threshold_percentil': 5},
                                            'DBSCAN':  {'enabled':              False,
                                                        'db_eps':               0.05,       # depends on the range of the features / size of the latent space 
                                                        'db_min_samples':       15,
                                                        'db_metric':            'cosine'},
                                            'LOF':     {'enabled':              True,
                                                        'n_neighbors':          20,
                                                        'novelty':              True,},
                                            'ABOD':    {'enabled':              False,},
                                            'IsolationForest': {'enabled':      False,
                                                                'n_estimators': 200,
                                                                'contamination': 0.00000002,},
                                            'COPOD':   {'enabled':              True,
                                                        'contamination':        0.00000002,},
                                            'SVM':     {'enabled':              False,},
                                            'KNN':     {'enabled':              False,
                                                        'k_neighbors':          5,
                                                        'anomaly_percentile':   5,
                                                        'metric':               'cosine',} ,
                                            'RBF':     {'enabled':              False,
                                                        'model_type':           "Nystroem", },
                                            }
              }


# optimizer idx + hypers
optimizer = {'idx':          1, 
             'lr':           2e-4, 
             'weight_decay': 0, 
             'betas':        (0.9, 0.999)}

# scheduler idx + hypers
scheduler = None

# loss idx + hypers
loss = {'idx':153,
        'framework_task':           framework_task,
        'architecture_type':        model['architecture_type'],
        'contrastive_args':         model['contrastive_args'],
        'contrastive_loss_weight':  1.0,
		'z_dim_m':                  model['z_dim_m'],
        'partition_z_space':        model['partition_z_space'],
        'activate_threshold':       True,
        'contr_l_denominator':      "2*(N-1)",
        }      
            


if __name__ == '__main__':

    run = wandb.init(
            project = wandb_project_name,
            notes   = "",
            tags    = ["baseline", "paper1"],
            config  = {
                "run_name":         run_name,
                "train_dataloader": train_dataloader,
                "val_dataloader":   val_dataloader,
                "test_dataloader":  test_dataloader,
                "dataset_train":    dataset_train,
                "model":            model,
                "training":         training,
                "optimizer":        optimizer,
                "loss":             loss,
                "meta_info":        meta_info,
            }
    )

    experiment = Experiment(meta_info           = meta_info,
                            dataset_param_train = dataset_train,
                            dataset_param_val   = dataset_val,
                            dataset_param_test  = dataset_test,
                            trdataloader_param  = train_dataloader,
                            vadataloader_param  = val_dataloader,
                            tedataloader_param  = test_dataloader,
                            model_param         = model,
                            training_param      = training,
                            evaluation_param    = evaluation,
                            optimizer_param     = optimizer,
                            scheduler_param     = scheduler,
                            loss_param          = loss,
                            run_name            = run_name)
    

    if save_config:
        # Get number of parameters
        num_params_total_model = sum(p.numel() for p in experiment.model.parameters() if p.requires_grad)
        model_parameter_size = {'num_params_total_model': num_params_total_model, }
        wandb_settings = {'project_name':   wandb_project_name,
                          'run_name_orig':  run.name,
                          'run_id':         run.id,
                          'link':           'https://wandb.ai/' + run.path}

        parameter_settings = {
                'meta_info':                meta_info,
                'dataset_param_train':      dataset_train,
                'dataset_param_val':        dataset_val,
                'dataset_param_test':       dataset_test,
                'train_dataloader_param':   train_dataloader,
                'val_dataloader_param':     val_dataloader,
                'test_dataloader_param':    test_dataloader,
                'model_param':              model,
                'training_param':           training,
                'evaluation_param':         evaluation,
                'optimizer_param':          optimizer,
                'scheduler_param':          scheduler,
                'loss_param':               loss,
                'num_model_parameters':     model_parameter_size,
                'run_name':                 run_name,
                'wandb':                    wandb_settings,
        }

        check_dir_and_save_json(output_dir  = output_dir, 
                                filename    ='training_settings.json', 
                                output_data = parameter_settings)
        print('Experiment config saved')
        

    # Start training process
    experiment.train()



    # Save model after training
    filename = output_dir + '\\model_' + str(training['idx']) + '.pt'
    experiment.save_checkpoint(filename)
    print('Training complete - Model saved')
    print('Evaluation started...')
    experiment.evaluate()
    print('Evaluation complete - Embeddings saved')

   
