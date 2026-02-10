import torch
import numpy as np
from scipy.spatial.distance import euclidean


def get_embeddings_associated_objects_from_scene(batch_data, model_output, key='association_info_camera_lidar'):

    associated_object_idx = batch_data[key][0]['associated_objects_idx']
    camera_idx = associated_object_idx[:,0]
    lidar_idx  = associated_object_idx[:,1]

    batch_data_new = {}

    ### Filter the z-space
    z_camera = model_output['z_objects_camera']
    z_lidar  = model_output['z_objects_lidar']
    
    z_camera_filtered = z_camera[camera_idx]
    z_lidar_filtered  = z_lidar[lidar_idx]
    assert z_camera_filtered.shape == z_lidar_filtered.shape, "Must have same shape"
    
    batch_data_new['objects_camera'] = z_camera_filtered
    batch_data_new['objects_lidar']  = z_lidar_filtered

    ### Filter on the h-space
    if 'h_objects_camera' in model_output and 'h_objects_lidar' in model_output:
        h_camera = model_output['h_objects_camera']
        h_lidar  = model_output['h_objects_lidar']

        h_camera_filtered = h_camera[camera_idx]
        h_lidar_filtered  = h_lidar[lidar_idx]
        assert h_camera_filtered.shape == h_lidar_filtered.shape, "Must have same shape"
        
        batch_data_new['h_objects_camera'] = h_camera_filtered
        batch_data_new['h_objects_lidar']  = h_lidar_filtered


    return batch_data_new


def obj_pair_reset(objects_c_in, objects_l_in):

    objects_c_out = []
    objects_l_out = []
    assert len(objects_c_in[0]) == len(objects_l_in[0]), "Must be same length"
    for idx in range(len(objects_c_in[0])):
        obj_c = objects_c_in[0][idx]
        obj_l = objects_l_in[0][idx]
    
        first_time = min(obj_c[0,1],  obj_l[0,1]).clone().item()
        mask_c = (obj_c[:,1] != 0)
        mask_l = (obj_l[:,1] != 0)
        obj_c_new= obj_c.clone()
        obj_l_new= obj_l.clone()
        obj_c_new[:,1] = (obj_c[:,1] - first_time) * mask_c
        obj_l_new[:,1] = (obj_l[:,1] - first_time) * mask_l

        objects_c_out.append(obj_c_new)
        objects_l_out.append(obj_l_new)

        assert obj_c_new[0,1]==0 or obj_l_new[0,1]==0, "at least one object must start at time_idx = 0"

    objects_c_out = torch.stack(objects_c_out).unsqueeze(dim=0)
    objects_l_out = torch.stack(objects_l_out).unsqueeze(dim=0)
    return objects_c_out, objects_l_out


def get_only_associated_objects_from_scene(batch_data, key='association_info_camera_lidar', obj_pair_starts_at_0=False):
    
    associated_object_idx = batch_data[key][0]['associated_objects_idx']
    camera_idx = associated_object_idx[:,0]  # Camera idx at pos 0
    lidar_idx  = associated_object_idx[:,1]  # LiDAR idx at pos 1

    objects_camera_filtered = batch_data['obj_list_camera'][:, camera_idx]
    objects_lidar_filtered  = batch_data['obj_list_lidar'][:, lidar_idx]
    assert objects_camera_filtered.shape[1] == objects_lidar_filtered.shape[1], "Must be the same number of samples"
    associated_object_idx_asso = np.array([[x,x]for x in  list(range(len(camera_idx)))])
    
    check_list=[]
    for obj_c, obj_l in zip(objects_camera_filtered[0], objects_lidar_filtered[0]):
        len_c = torch.count_nonzero(obj_c[:, 2]).item()
        len_l = torch.count_nonzero(obj_l[:, 2]).item()
        indices_c = obj_c[:len_c,1].numpy().tolist()
        indices_l = obj_l[:len_l,1].numpy().tolist()
        overlapping_idx =[idx_c for idx_c in indices_c if idx_c in indices_l]
        assert len(overlapping_idx) > 1, "Not enough overlapping time idx in object pair"
        check_list.append(overlapping_idx)

    if obj_pair_starts_at_0:
        objects_camera_filtered_final, objects_lidar_filtered_final = obj_pair_reset(objects_camera_filtered, objects_lidar_filtered)
    else:
        objects_camera_filtered_final = objects_camera_filtered
        objects_lidar_filtered_final = objects_lidar_filtered

    
    for obj_c, obj_l, i in zip(objects_camera_filtered_final[0], objects_lidar_filtered_final[0], range(len( objects_camera_filtered_final[0]))):
        assert obj_c[0,1]==0 or obj_l[0,1]==0, "at least one object must start at time_idx = 0"
    

    batch_data_new = {'obj_list_camera':    objects_camera_filtered_final,
                      'obj_list_lidar':     objects_lidar_filtered_final,
                      'association_info_camera_lidar': [{'associated_objects_idx': associated_object_idx_asso}], }
    batch_data.update(batch_data_new)

    # if 'scene_info' in batch_data:
    #     batch_data_new['scene_info'] = batch_data['scene_info']
    return batch_data


def calc_distance_between_objects(obj_1, obj_2, data_order, useEgoCentricCoord=False, metric='euclidean'):
    # Return the distance between obj_1 and obj_2 for every common timestep

    if metric == 'euclidean':
        dist_function = euclidean
    else:
        assert False, 'TBD'

    if torch.is_tensor(obj_1):
        obj_1 = obj_1.cpu().detach().numpy()
    if torch.is_tensor(obj_2):
        obj_2 = obj_2.cpu().detach().numpy()

    

    # assert obj_1.shape == obj_2.shape, "the Dimensions should be the same"
    idx_time_idx = data_order.index('time_idx_in_scenario_frame')
    intersection_idx = sorted(set(obj_1[:, idx_time_idx]).intersection(obj_2[:, idx_time_idx]))

    idx_x = data_order.index('x')
    idx_y = data_order.index('y')
    
    if useEgoCentricCoord:
        # obj_1 is EGO which is aleayxs at position (0,0)
        obj_1[:, idx_x] = np.zeros(obj_1[:, idx_x].shape)
        obj_1[:, idx_y] = np.zeros(obj_1[:, idx_y].shape)

    distance_list = []

    for idx in intersection_idx:
        idx_1 = obj_1[:, idx_time_idx].tolist().index(idx)
        idx_2 = obj_2[:, idx_time_idx].tolist().index(idx)
        dist = dist_function([obj_1[idx_1][idx_x], obj_1[idx_1][idx_y]], 
                             [obj_2[idx_2][idx_x], obj_2[idx_2][idx_y]])
        distance_list.append(dist)

    return distance_list


def add_sample_value(obj_list_in, subclass_val=1):
    obj_list_out = []
    for obj in obj_list_in[0]:
        subclass_vector = torch.ones((len(obj), 1)).to(obj.device) * subclass_val
        mask = torch.ones(subclass_vector.shape, dtype=torch.bool, device=obj.device)
        obj_len = torch.count_nonzero(obj[:,0]).item()
        mask[obj_len:] = False
        obj_new = torch.cat((obj, (subclass_vector*mask)), dim=1)
        obj_list_out.append(obj_new)

    if len(obj_list_out) < 0:
        obj_list_out = torch.stack(obj_list_out).unsqueeze(dim=0)

    return obj_list_out


def add_offset_to_objlist(objlist_in, feature_name, offset_value, data_order):
    objlist_out = []
    idx_feature = data_order.index(feature_name)
    idx_timestamp = data_order.index('timestamp')
    for obj in objlist_in[0]:
        obj_len = torch.count_nonzero(obj[:,idx_timestamp])
        obj_new = torch.clone(obj)
        obj_new[:obj_len, idx_feature] = obj_new[:obj_len, idx_feature] + offset_value
        objlist_out.append(obj_new)

    objlist_out = torch.stack(objlist_out).to(objlist_in.device).unsqueeze(dim=0)

    return objlist_out


def create_offset_subclasses(batch_data_in, subclass_params):
    # Input:  batch_data(obj_list_c, obj_list_l, association_list), subclass+setting
    # Output: object    _list_subclass1, object_list_subclass1+2, labels_subclass1+2
    res_dict = {}
    data_order = batch_data_in['general_info'][0]['data_order_tracking_res']
    associated_object_idx = batch_data_in['association_info_camera_lidar'][0]['associated_objects_idx']

    ### Filter based on the object association
    camera_idx = associated_object_idx[:,0]     # Camera idx at pos 0
    lidar_idx  = associated_object_idx[:,1]     # LiDAR idx at pos 1
    obj_list_c_subclass_1 = batch_data_in['obj_list_camera'][:, camera_idx]
    obj_list_l_subclass_1 = batch_data_in['obj_list_lidar'][:, lidar_idx]
    assert obj_list_c_subclass_1.shape[1] == obj_list_l_subclass_1.shape[1], "Must be the same number of samples"
    associated_object_idx_subclass_1 = np.array([[x,x] for x in list(range(len(camera_idx)))])
    
    ### Filter based on the subclasses
    if subclass_params['obj2alter'] == 'camera':
        obj_list_c_subclass_2 = add_offset_to_objlist(objlist_in   = obj_list_c_subclass_1, 
                                                      feature_name = subclass_params['feature_name'], 
                                                      offset_value = subclass_params['offset_val'],
                                                      data_order   = data_order)
        obj_list_l_subclass_2 = obj_list_l_subclass_1
        
    elif subclass_params['obj2alter'] == 'lidar':
        obj_list_l_subclass_2 = add_offset_to_objlist(objlist_in   = obj_list_l_subclass_1, 
                                                      feature_name = subclass_params['feature_name'], 
                                                      offset_value = subclass_params['offset_val'],
                                                      data_order   = data_order)
        obj_list_c_subclass_2 = obj_list_c_subclass_1 

    else:
        assert False, "undefined obj2alter"
    
    assert obj_list_c_subclass_1.shape == obj_list_c_subclass_2.shape, "Camera subclass 1 and 2 must have the same shape"
    assert obj_list_l_subclass_1.shape == obj_list_l_subclass_2.shape, "LiDAR subclass 1 and 2 must have the same shape"


    if subclass_params['add_sample_value']:
        # Add a sample value for all object list data (just for testing p)
        obj_list_c_subclass_1 = add_sample_value(obj_list_c_subclass_1, subclass_val=1)
        obj_list_l_subclass_1 = add_sample_value(obj_list_l_subclass_1, subclass_val=1)
        obj_list_c_subclass_2 = add_sample_value(obj_list_c_subclass_2, subclass_val=2)
        obj_list_l_subclass_2 = add_sample_value(obj_list_l_subclass_2, subclass_val=2)

        batch_data_in['general_info'][0]['data_order_tracking_res'].append('subclass_value')



    # Data just for training and testing
    res_dict['batch_subclass_1'] = {'obj_list_camera':                  obj_list_c_subclass_1,
                                    'obj_list_lidar':                   obj_list_l_subclass_1,
                                    'obj_ego':                          batch_data_in['obj_ego'],
                                    'general_info':                     batch_data_in['general_info'],
                                    'association_info_camera_lidar':    [{'associated_objects_idx': associated_object_idx_subclass_1}], 
                                    } 
    # Data for evaluation  
    res_dict['batch_subclass_2'] = {'obj_list_camera':                  obj_list_c_subclass_2,
                                    'obj_list_lidar':                   obj_list_l_subclass_2,
                                    'obj_ego':                          batch_data_in['obj_ego'],
                                    'general_info':                     batch_data_in['general_info'],
                                    'association_info_camera_lidar':    [{'associated_objects_idx': associated_object_idx_subclass_1}], 
                                    } 
    
    if 'scene_info' in batch_data_in:
        res_dict['batch_subclass_1']['scene_info'] = batch_data_in['scene_info']
        res_dict['batch_subclass_2']['scene_info'] = batch_data_in['scene_info']

    return res_dict