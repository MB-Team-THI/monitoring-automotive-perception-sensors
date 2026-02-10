import math
import torch
import random
import numpy as np
from sklearn.datasets import make_spd_matrix


def alter_object_list_train(obj_list_in, params, data_order_tracking_res, sensor_name):
    ### Fine and detailed alteration of object lists data
    # Alteration:  Add feature specific noise to the object lists
    obj_list_out    = []
    noise_var       = params['normal_dist_var']
    no_aug_features = ['tracking_id', 'time_idx_in_scenario_frame', 'timestamp', 'pred_class']
    dict_aug_factor = {'x': 5,
                       'y': 5,
                       'z': 0.5,
                       'size_x': 1,
                       'size_y': 0.5,
                       'size_z': 0.5,
                       'heading': (math.pi / 4),
                       'v':     5,
                       'v_x':   1,
                       'v_y':   1,
                       'tracking_score': 0.3,
                       'subclass_value': 0.0}

    for obj in obj_list_in[0]:
        
        if params['add_gaussian_noise_enabled']:
            # Add the noise for each feature specifically
            for idx, feature in enumerate(data_order_tracking_res):
                if feature in no_aug_features:
                    continue
                # Create feature mask
                mask = (obj != 0)
                mask[:,:idx]   = False
                mask[:,idx+1:] = False

                # Create and add noise to the specific feature
                noise = np.random.normal(0, noise_var, size=mask.shape)
                noise_weighted = dict_aug_factor[feature] + noise * dict_aug_factor[feature]              
                obj = obj + (mask * noise_weighted).to(torch.float32)

        if params['delete_random_timesteps_enabled']:
            obj_len = torch.count_nonzero(obj[:,0]).item()
            rand_int = random.randint(0, 100)
            if rand_int > 95 and obj_len > 4:
                # 5% chance: delete two consecutive timesteps
                idx_to_delete = random.randint(2, obj_len-3) # Do no delete the first nor the last element               
                mask = torch.ones(obj.shape, dtype=torch.bool, device=obj.device)
                mask[idx_to_delete] = False
                mask[idx_to_delete+1] = False
                obj = obj[mask].view((obj.shape[0]-2, obj.shape[1]))
                zero_row = torch.zeros((2, 15), dtype=torch.float32, device=obj.device)
                obj = torch.cat((obj, zero_row), dim=0)
            elif rand_int > 75 and obj_len > 3:
                # 20% chance: delete one random timestamp
                idx_to_delete = random.randint(1, obj_len-2) # Do no delete the first nor the last element                
                mask = torch.ones(obj.shape, dtype=torch.bool, device=obj.device)
                mask[idx_to_delete] = False
                obj = obj[mask].view((obj.shape[0]-1, obj.shape[1]))
                zero_row = torch.zeros((1, 15), dtype=torch.float32, device=obj.device)
                obj = torch.cat((obj, zero_row), dim=0)
            else: 
                # 75% chance: do not delete any timesteps
                pass         
        
        obj_list_out.append(obj)

    obj_list_out = torch.stack(obj_list_out).unsqueeze(axis=0)
    
    assert obj_list_in.shape == obj_list_out.shape, sensor_name + ": altered and original object list must have same dimension"
    assert not (obj_list_in == obj_list_out).all(), sensor_name + ": the augmented and the original object list are identical"

    return obj_list_out


def alter_object_list_eval(obj_list_in, params):
    # Broad and very general alteration of object lists data

    noise_var      = params['normal_dist_var']

    obj_list_out = []


    for obj in obj_list_in[0]:
        mask = (obj != 0)
        mask[:,0:2] = False

        noise = np.random.normal(0, noise_var, size=mask.shape)
        # Does this jitter makes sense
        # jitter = np.roll(x, shift=np.random.randint(1, 5), axis=0)

        obj_altered = obj + (mask * noise).to(torch.float32)
        obj_list_out.append(obj_altered)

    obj_list_out = torch.stack(obj_list_out).unsqueeze(axis=0)

    return obj_list_out


def create_sample_data_subclasses(n_samples, i=-1, n_features=12, subclass=-1):

    if subclass == 1:
        # Data to train the encoder and fit the GMM (at least 10 samples in the train-split)
        # sample_seq_lengths = [5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
        sample_values =  [5,  6,  7,  8,  9, 10, 11, 12, 13, 14] 
    elif subclass == 2:
        # Data used in inference to test the performance (at least 10 samples in the test and val-split)
        # sample_seq_lengths = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        sample_values = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    else:
        assert False, "Subclass not defined"

    if i != -1:
        idx = i % len(sample_values)
        # seq_len = sample_seq_lengths[idx]
        seq_len = np.random.randint(3,40)
        val = sample_values[idx]
    
    def sample_normal_sequence():
        #return np.random.multivariate_normal(mean, cov, size=seq_len)
        return np.ones((seq_len, n_features)) * val
    
    # Light augmentation for similar pairs (train and test normal)
    def light_augment(x):
        noise = np.random.normal(0, 0.1, size=x.shape)
        return x + noise
    
    
    # sample_len = np.random.randint(2, 40)
    ### Create Sample Data
    objects_camera = []
    objects_lidar  = []
    object_indices = []
    for i in range(n_samples):
        base = sample_normal_sequence()
        obj_camera = light_augment(base)
        obj_lidar  = light_augment(base)

        time_idx = np.array([i for i in range(seq_len)]).reshape(seq_len,1)
        obj_camera_new = np.concatenate((time_idx, time_idx, time_idx, obj_camera), axis=1)
        obj_lidar_new = np.concatenate((time_idx, time_idx, time_idx, obj_lidar), axis=1)

        objects_camera.append(obj_camera_new) 
        objects_lidar.append(obj_lidar_new)
        object_indices.append([i, i])

    objects_camera = np.array(objects_camera)  # shape: (n_train, seq_len, n_features)
    objects_lidar  = np.array(objects_lidar)
    objects_camera = torch.tensor(objects_camera, dtype=torch.float32).unsqueeze(dim=0)     # shape: (1, n_train, seq_len, n_features)
    objects_lidar  = torch.tensor(objects_lidar, dtype=torch.float32).unsqueeze(dim=0) 


    sample_data = {'association_info_camera_lidar': [ {'associated_objects_idx': np.array(object_indices)}],
                   'obj_list_camera': objects_camera,
                   'obj_list_lidar':  objects_lidar,}

    return sample_data


def create_sample_data_augmentations(n_samples=-1, seq_range=[4, 20], n_features=15, aug_camera=False, aug_lidar=False, i=-1, random_seq_length=False, test_set=False):
    np.random.seed(42) 

    if not test_set:
        sample_seq_lengths = [5, 10, 15, 20, 25] # , 5, 10, 35, 20, 25] # , 30, 15]
    else:
        sample_seq_lengths = [7,  9, 12, 14, 16, 17, 21, 24, 26, 27]  # all within the range [5, 25]
        
    if i == -1:
        # Use fixed sequence length
        seq_len = 35 
    else:
        # Use defined sequence length
        idx = i % len(sample_seq_lengths)
        seq_len = sample_seq_lengths[idx]
        
    if random_seq_length:
        # Use random sequence length
        seq_len = random.randint(seq_range[0], seq_range[1])
    
    # Create a base multivariate Gaussian distribution
    cov = make_spd_matrix(n_features)
    mean = np.zeros(n_features)
    
    def sample_normal_sequence():
        return np.random.multivariate_normal(mean, cov, size=seq_len)

    # Light augmentation for similar pairs (train and test normal)
    def light_augment(x):
        noise = np.random.normal(0, 0.1, size=x.shape)
        return x + noise

    # Heavy augmentation for anomaly
    def heavy_augment(x):
        noise = np.random.normal(0, 1, size=x.shape)
        jitter = np.roll(x, shift=np.random.randint(1, 5), axis=0)
        return x + noise + 0.1 * jitter # 0.1 * jitter

    ### Create Sample Data
    objects_camera = []
    objects_lidar  = []
    objects_camera_anomaly = []
    objects_lidar_anomaly  = []
    object_indices = []
    for i in range(n_samples):
        base = sample_normal_sequence()
        obj_camera = light_augment(base)
        obj_lidar  = light_augment(base)
        obj_camera_anomaly = heavy_augment(obj_camera)
        obj_lidar_anomaly  = heavy_augment(obj_lidar)

        objects_camera.append(obj_camera) 
        objects_lidar.append(obj_lidar)
        objects_camera_anomaly.append(obj_camera_anomaly)
        objects_lidar_anomaly.append(obj_lidar_anomaly)
            
        object_indices.append([i, i])

    objects_camera = np.array(objects_camera)  # shape: (n_train, seq_len, n_features)
    objects_lidar  = np.array(objects_lidar)
    objects_camera_anomaly = np.array(objects_camera_anomaly)
    objects_lidar_anomaly  = np.array(objects_lidar_anomaly)

    objects_camera = torch.tensor(objects_camera, dtype=torch.float32).unsqueeze(dim=0)     # shape: (1, n_train, seq_len, n_features)
    objects_lidar  = torch.tensor(objects_lidar, dtype=torch.float32).unsqueeze(dim=0)      
    objects_camera_anomaly = torch.tensor(objects_camera_anomaly, dtype=torch.float32).unsqueeze(dim=0) 
    objects_lidar_anomaly  = torch.tensor(objects_lidar_anomaly, dtype=torch.float32).unsqueeze(dim=0) 


    sample_data = {'association_info_camera_lidar': [ {'associated_objects_idx': np.array(object_indices)}]}
    if aug_camera:
        sample_data['obj_list_camera'] = objects_camera_anomaly
    else:
        sample_data['obj_list_camera'] = objects_camera
    if aug_lidar:
        sample_data['obj_list_lidar']  = objects_lidar_anomaly
    else:         
        sample_data['obj_list_lidar'] = objects_lidar 

    return sample_data
