
import numpy as np
from scipy.io import loadmat


# from src.utils.rot_points import rot_points
from src.utils.rotate_and_align_traj import transform_object_lists, convert_to_ego_centric, process_heading_feature


def _dict_to_ndarray2(obj_dict, features_to_load):
    dict_keys = list(obj_dict.keys())

    obj_array = np.ndarray(shape=(len(features_to_load), len(obj_dict[dict_keys[0]])))

    idx = 0
    for key in features_to_load:
        if key in dict_keys:
            obj_array[idx, :] = obj_dict[key]
            idx += 1

    return obj_array


def get_obj_list(data, data_order_key, class_name_dict):
    # convert the imported object data from the mat-files into one dict per object
    # The dict key describes the feature, while the values are strictly numerical (else they cannot be converted into tensors)

    obj_list            = []
    data_order_gt       = [i[0] for i in data['meta'][0][0][0][data_order_key][0][0]]

    if data.size != 0 and 0<len(data['results'][0][0]):
        for obj in data['results'][0][0][0]:
            complete_object = {}
            # Copy the complete obj data based on the data order
            for idx, key in enumerate(data_order_gt):
                complete_object[key] = obj[idx, :]
            # handle the index conversion from matlab to python
            complete_object['time_idx_in_scenario_frame'] = complete_object['time_idx_in_scenario_frame'] - 1        
            obj_list.append(complete_object)
    
    return obj_list


def get_obj_list_associated_format(data_in, object_list_key, feature_keys=None, min_obj_length=1):
    obj_list_out = []
    if object_list_key in data_in:
        if len(data_in[object_list_key]) > 0:            
            # Obtain the data from the nested .mat-file format
            object_list = []
            for obj in data_in[object_list_key][0]:
                if min_obj_length <= len(obj[0][obj[0].dtype.names[0]][0][0]):
                    obj_trans = {k: obj[0][k][0][0] for k in obj[0].dtype.names}
                    object_list.append(obj_trans)

            if len(object_list) > 0:
                # Filter based on feature_keys and convert to nd.array format (leave the intermediate dict-format for interpretability)
                if feature_keys==None:
                    # Take all keys
                    feature_keys = object_list[0].keys()
                for obj in object_list:
                    obj_list_out.append(_dict_to_ndarray2(obj, feature_keys))

    return obj_list_out



def _load_sample_obj_lists_scene_association(self, filename):
    # Load the scene for the object association

    general_names = ["general_info", "scene_info", "obj_list_ego", "obj_list_lidar", "obj_list_camera", "obj_list_gt", "obj_list_lidar_nc", "association_list"]
    camera_augmentations = ["obj_list_camera_aug_br_s3", "obj_list_camera_aug_da_s3", "obj_list_camera_aug_fog_s3", "obj_list_camera_aug_mb_s3", 
                            "obj_list_camera_aug_mc_s3", "obj_list_camera_aug_snow_s3", "obj_list_camera_aug_tm_s3"]
    lidar_augmentations  = ["obj_list_lidar_aug_pr_s1", "obj_list_lidar_aug_pr_s2", "obj_list_lidar_aug_pr_s3", "obj_list_lidar_aug_br_s3", 
                            "obj_list_lidar_aug_fog_s3", "obj_list_lidar_aug_mb_s3", "obj_list_lidar_aug_sm_s3", "obj_list_lidar_aug_tm_s3"]
    all_keys_to_load = general_names + camera_augmentations + lidar_augmentations
    mat_temp = loadmat(filename, variable_names = all_keys_to_load, verify_compressed_data_integrity=False)

    # META===============================================================================================
    # INFO-----------------------------------------------------------------------------------------------

    ### scene_info
    scene_keys      = mat_temp['scene_info']['scene'][0][0][0].__dir__.__self__.dtype.names
    scene_vals      = [mat_temp['scene_info']['scene'][0][0][0][0][i][0][:] for i in range(len(scene_keys))]
    scene_vals[2]   = scene_vals[2][0]
    scene_dict      = {k: v for (k, v) in zip(scene_keys, scene_vals)}

    map_keys        = mat_temp['scene_info']['map'][0][0][0].__dir__.__self__.dtype.names
    map_vals        = [mat_temp['scene_info']['map'][0][0][0][0][i][0][:] for i in range(len(map_keys))]
    map_dict        = {k: v for (k, v ) in zip(map_keys, map_vals)}

    scene_info      = {'scene': scene_dict, 'map': map_dict}

    ### general_info

    class_name_dict = {row[1][0]: row[0][0][0] for row in mat_temp['general_info']['class_list'][0][0][0]}
    data_order  	= [i[0] for i in mat_temp['general_info']['data_order'][0][0][0]]
    gt_attributes   = {str(row[0][3][0][0]):row[0][1][0] for row in mat_temp['obj_list_gt']['meta'][0][0][0]['gt_attributes'][0]}
    gt_categories   = {str(row[0][3][0][0]): row[0][1][0] for row in mat_temp['obj_list_gt']['meta'][0][0][0]['gt_categories'][0]}
    general_info    = {'class_name_dict':           class_name_dict,
                       'data_order_tracking_res':   data_order,
                       'gt_attributes':             gt_attributes,
                       'gt_categories':             gt_categories}

    # DYNAMICS============================================================================================
    # EGO obj list----------------------------------------------------------------------------------------
    ego_temp        = mat_temp['obj_list_ego']
    data_order_ego  = [i[0] for i in ego_temp['meta'][0][0][0]['ego_data_order'][0][0]]
    obj_list_ego    = {}
    # Copy the complete ego data based on the data order
    for idx, key in enumerate(data_order_ego):
        obj_list_ego[key] = ego_temp['results'][0][0][idx, :]
    # handle the index conversion from matlab to python
    obj_list_ego['time_idx_in_scenario_frame'] = obj_list_ego['time_idx_in_scenario_frame'] - 1

    # GT obj list-----------------------------------------------------------------------------------------
    if scene_info['scene']['datasplit'] == 'test':
        obj_list_gt = []
    else:
        obj_list_gt = get_obj_list(data             = mat_temp['obj_list_gt'], 
                                   data_order_key   = "gt_data_order", 
                                   class_name_dict  = class_name_dict)
        
    # camera obj list----------------------------------------------------------------------------------------
    obj_list_camera = get_obj_list(data             = mat_temp['obj_list_camera'], 
                                   data_order_key   = "data_order", 
                                   class_name_dict  = class_name_dict)
    
    # lidar obj list-----------------------------------------------------------------------------------------
    obj_list_lidar = get_obj_list(data              = mat_temp['obj_list_lidar'], 
                                  data_order_key    = "data_order", 
                                  class_name_dict   = class_name_dict)
    
    # lidar obj list nc--------------------------------------------------------------------------------------
    if 'obj_list_lidar_nc' in mat_temp:
        obj_list_lidar_nc = get_obj_list(data           = mat_temp['obj_list_lidar_nc'], 
                                        data_order_key  = "data_order", 
                                        class_name_dict = class_name_dict)
    else:
        obj_list_lidar_nc = []

            
    ### LiDAR obj list augmented --------------------------------------------------------------------
    if 'obj_list_lidar_aug_pr_s1' in mat_temp:
        obj_list_lidar_aug_pr_s1 = get_obj_list(data            = mat_temp['obj_list_lidar_aug_pr_s1'], 
                                                data_order_key  = "data_order", 
                                                class_name_dict = class_name_dict)
    else:
        obj_list_lidar_aug_pr_s1 = []

    if 'obj_list_lidar_aug_pr_s2' in mat_temp:
        obj_list_lidar_aug_pr_s2 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_pr_s2'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_pr_s2 = []

    if 'obj_list_lidar_aug_pr_s3' in mat_temp:
        obj_list_lidar_aug_pr_s3 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_pr_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_pr_s3 = []

    if 'obj_list_lidar_aug_br_s3' in mat_temp:
        obj_list_lidar_aug_br_s3 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_br_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_br_s3 = []
        
    if 'obj_list_lidar_aug_fog_s3' in mat_temp:
        obj_list_lidar_aug_fog_s3 = get_obj_list(data             = mat_temp['obj_list_lidar_aug_fog_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_fog_s3 = []
        
    if 'obj_list_lidar_aug_mb_s3' in mat_temp:
        obj_list_lidar_aug_mb_s3 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_mb_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_mb_s3 = []
        
    if 'obj_list_lidar_aug_sm_s3' in mat_temp:
        obj_list_lidar_aug_sm_s3 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_sm_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_sm_s3 = []
    
    if 'obj_list_lidar_aug_tm_s3' in mat_temp:
        obj_list_lidar_aug_tm_s3 = get_obj_list(data              = mat_temp['obj_list_lidar_aug_tm_s3'], 
                                                data_order_key    = "data_order", 
                                                class_name_dict   = class_name_dict)
    else:
        obj_list_lidar_aug_tm_s3 = []


    ### Camera obj list augmented --------------------------------------------------------------------
    if 'obj_list_camera_aug_br_s3' in mat_temp:
        obj_list_camera_aug_br_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_br_s3'], 
                                                 data_order_key  = "data_order", 
                                                 class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_br_s3 = []

    if 'obj_list_camera_aug_da_s3' in mat_temp:
        obj_list_camera_aug_da_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_da_s3'], 
                                                 data_order_key  = "data_order", 
                                                 class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_da_s3 = []

    if 'obj_list_camera_aug_fog_s3' in mat_temp:
        obj_list_camera_aug_fog_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_fog_s3'], 
                                                  data_order_key  = "data_order",
                                                  class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_fog_s3 = []
    
    if 'obj_list_camera_aug_mb_s3' in mat_temp:
        obj_list_camera_aug_mb_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_mb_s3'], 
                                                 data_order_key  = "data_order", 
                                                 class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_mb_s3 = []
    
    if 'obj_list_camera_aug_mc_s3' in mat_temp:
        obj_list_camera_aug_mc_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_mc_s3'], 
                                                 data_order_key  = "data_order", 
                                                 class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_mc_s3 = []
            
    if 'obj_list_camera_aug_snow_s3' in mat_temp:
        obj_list_camera_aug_snow_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_snow_s3'], 
                                                   data_order_key  = "data_order", 
                                                   class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_snow_s3 = []
                    
    if 'obj_list_camera_aug_tm_s3' in mat_temp:
        obj_list_camera_aug_tm_s3 = get_obj_list(data            = mat_temp['obj_list_camera_aug_tm_s3'], 
                                                 data_order_key  = "data_order", 
                                                 class_name_dict = class_name_dict)
    else:
        obj_list_camera_aug_tm_s3 = []
    camera_augmentations = ["", "", "", "", 
                            "", "", ""]
    # Rotate and align trajectory ========================================================================
    # Convert the object list data within the world coordinate system: EGO-vehicle  starting at (e.g., 100, 100) -> EGO-vehicle starting at (0, 0) aligned to the x-axis to the 
    if True:
        object_list_dict_in = {'obj_list_ego':  obj_list_ego,
                               'obj_list_camera':obj_list_camera,
                               'obj_list_lidar': obj_list_lidar,
                               'obj_list_gt':    obj_list_gt, 
                               'obj_list_lidar_nc': obj_list_lidar_nc,
                               'obj_list_lidar_aug_pr_s1': obj_list_lidar_aug_pr_s1,
                               'obj_list_lidar_aug_pr_s2': obj_list_lidar_aug_pr_s2,
                               'obj_list_lidar_aug_pr_s3': obj_list_lidar_aug_pr_s3,                               
                               'obj_list_lidar_aug_br_s3': obj_list_lidar_aug_br_s3,
                               'obj_list_lidar_aug_mb_s3': obj_list_lidar_aug_mb_s3,
                               'obj_list_lidar_aug_fog_s3': obj_list_lidar_aug_fog_s3,
                               'obj_list_lidar_aug_sm_s3': obj_list_lidar_aug_sm_s3,
                               'obj_list_lidar_aug_tm_s3': obj_list_lidar_aug_tm_s3,
                               'obj_list_camera_aug_br_s3': obj_list_camera_aug_br_s3,
                               'obj_list_camera_aug_da_s3': obj_list_camera_aug_da_s3,
                               'obj_list_camera_aug_fog_s3': obj_list_camera_aug_fog_s3,
                               'obj_list_camera_aug_mb_s3': obj_list_camera_aug_mb_s3,
                               'obj_list_camera_aug_mc_s3': obj_list_camera_aug_mc_s3,
                               'obj_list_camera_aug_snow_s3': obj_list_camera_aug_snow_s3,
                               'obj_list_camera_aug_tm_s3': obj_list_camera_aug_tm_s3,}
        
        transformed_obj_lists = transform_object_lists(object_list_dict_in  = object_list_dict_in,
                                                       rotation_type        = 'basic',
                                                       min_obj_length       = self.min_object_length_object_association)
        
        obj_list_ego      = transformed_obj_lists['obj_list_ego_rot']
        obj_list_camera   = transformed_obj_lists['obj_list_camera_rot']
        obj_list_lidar    = transformed_obj_lists['obj_list_lidar_rot'] 
        obj_list_gt       = transformed_obj_lists['obj_list_gt_rot']
        obj_list_lidar_nc = transformed_obj_lists['obj_list_lidar_nc_rot']
        # Lidar
        obj_list_lidar_aug_pr_s1  = transformed_obj_lists['obj_list_lidar_aug_pr_s1_rot']
        obj_list_lidar_aug_pr_s2  = transformed_obj_lists['obj_list_lidar_aug_pr_s2_rot']
        obj_list_lidar_aug_pr_s3  = transformed_obj_lists['obj_list_lidar_aug_pr_s3_rot']
        obj_list_lidar_aug_br_s3  = transformed_obj_lists['obj_list_lidar_aug_br_s3_rot']
        obj_list_lidar_aug_mb_s3  = transformed_obj_lists['obj_list_lidar_aug_mb_s3_rot']
        obj_list_lidar_aug_fog_s3 = transformed_obj_lists['obj_list_lidar_aug_fog_s3_rot']
        obj_list_lidar_aug_sm_s3  = transformed_obj_lists['obj_list_lidar_aug_sm_s3_rot']
        obj_list_lidar_aug_tm_s3  = transformed_obj_lists['obj_list_lidar_aug_tm_s3_rot']
        # Camera
        obj_list_camera_aug_br_s3   = transformed_obj_lists['obj_list_camera_aug_br_s3_rot']
        obj_list_camera_aug_da_s3   = transformed_obj_lists['obj_list_camera_aug_da_s3_rot']
        obj_list_camera_aug_fog_s3  = transformed_obj_lists['obj_list_camera_aug_fog_s3_rot']
        obj_list_camera_aug_mb_s3   = transformed_obj_lists['obj_list_camera_aug_mb_s3_rot']
        obj_list_camera_aug_mc_s3   = transformed_obj_lists['obj_list_camera_aug_mc_s3_rot']
        obj_list_camera_aug_snow_s3 = transformed_obj_lists['obj_list_camera_aug_snow_s3_rot']
        obj_list_camera_aug_tm_s3   = transformed_obj_lists['obj_list_camera_aug_tm_s3_rot']

    return {
        "general_info":      general_info,
        "scene_info":        scene_info,
        "obj_list_ego":      obj_list_ego,
        "obj_list_gt":       obj_list_gt,
        "obj_list_lidar":    obj_list_lidar,
        "obj_list_camera":   obj_list_camera,
        "obj_list_lidar_nc": obj_list_lidar_nc,
        "obj_list_lidar_aug_pr_s1":     obj_list_lidar_aug_pr_s1,
        "obj_list_lidar_aug_pr_s2":     obj_list_lidar_aug_pr_s2,
        "obj_list_lidar_aug_pr_s3":     obj_list_lidar_aug_pr_s3,
        "obj_list_lidar_aug_br_s3":     obj_list_lidar_aug_br_s3,
        "obj_list_lidar_aug_mb_s3":     obj_list_lidar_aug_mb_s3,
        "obj_list_lidar_aug_fog_s3":    obj_list_lidar_aug_fog_s3,
        "obj_list_lidar_aug_sm_s3":     obj_list_lidar_aug_sm_s3,
        "obj_list_lidar_aug_tm_s3":     obj_list_lidar_aug_tm_s3,
        "obj_list_camera_aug_br_s3":    obj_list_camera_aug_br_s3,
        "obj_list_camera_aug_da_s3":    obj_list_camera_aug_da_s3,
        "obj_list_camera_aug_fog_s3":   obj_list_camera_aug_fog_s3,
        "obj_list_camera_aug_mb_s3":    obj_list_camera_aug_mb_s3,
        "obj_list_camera_aug_mc_s3":    obj_list_camera_aug_mc_s3,
        "obj_list_camera_aug_snow_s3":  obj_list_camera_aug_snow_s3,
        "obj_list_camera_aug_tm_s3":    obj_list_camera_aug_tm_s3,
    }


def _load_sample_obj_lists_scene(self, filename):
    # Load the (already associated) scene for the model
    keys_to_load = ["general_info", "scene_info", "obj_list_ego", "obj_list_lidar", "obj_list_camera", "obj_list_gt",
                    "obj_list_lidar_nc", "association_info_camera_lidar", ]
    keys_to_load = keys_to_load + self.augmentations_to_load + self.augmentations_to_load_associations

    mat_temp = loadmat(
        filename,
        variable_names=keys_to_load,
        verify_compressed_data_integrity=False,
    )

    # META===============================================================================================
    # INFO-----------------------------------------------------------------------------------------------
    ### scene_info
    d = mat_temp["scene_info"]["scene"][0][0]
    scene_dict = {k: d[k][0][0][0] for k in d.dtype.names}
    d = mat_temp["scene_info"]["map"][0][0]
    map_dict = {k: d[k][0][0][0] for k in d.dtype.names}
    scene_info = {"scene": scene_dict, "map": map_dict}
    ### general_info
    data_order      = [x.strip() for x in mat_temp["general_info"][0]['data_order_tracking_res'][0]]
    class_names     = mat_temp["general_info"][0]['class_name_dict'][0]
    class_name_dict = {k: class_names[k][0][0][0][0] for k in class_names.__dir__.__self__.dtype.names}
    general_info = {"data_order_tracking_res":  data_order,
                    "class_name_dict":          class_name_dict}

    # Association information--------------------------------------------------------------------------------
    if "association_info_camera_lidar" in mat_temp:
        data = mat_temp["association_info_camera_lidar"][0]
        association_info_camera_lidar = {
            "associated_objects_idx":       data["associated_objects_idx"][0],
            "associated_objects_track_id":  data["associated_objects_track_id"][0],
            "associated_objects_dists":     data["associated_objects_dists"][0],
            "distance_matrix":              data["distance_matrix"][0],
            "association_params":           {k: data["association_params"][0][k][0][0] for k in data["association_params"][0].dtype.names},
            "key":                          'association_info_camera_lidar',
        }
    else:
        association_info_camera_lidar = []

    # DYNAMICS===============================================================================================
    obj_ego_dict: list = {k: mat_temp["obj_list_ego"][k][0][0][0] for k in mat_temp["obj_list_ego"].dtype.names}
    obj_ego = _dict_to_ndarray2(obj_ego_dict, obj_ego_dict.keys())

    feature_keys = [] 
    obj_list_camera     = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_camera", min_obj_length=self.min_obj_length)
    obj_list_lidar      = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_lidar", min_obj_length=self.min_obj_length)
    obj_list_gt         = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_gt", min_obj_length=self.min_obj_length)
    obj_list_lidar_nc   = get_obj_list_associated_format(mat_temp, object_list_key="obj_list_lidar_nc", min_obj_length=self.min_obj_length)
    obj_lists_aug_dict = {}
    for key in self.augmentations_to_load:
        obj_lists_aug_dict[key] = get_obj_list_associated_format(mat_temp, object_list_key=key)
    
    
    if self.useEgoCentricCoord:
        # Convert the camera and lidar object list
        obj_list_camera = convert_to_ego_centric(obj_list_in=obj_list_camera, ego_obj=obj_ego_dict, data_order_obj=data_order)
        obj_list_lidar  = convert_to_ego_centric(obj_list_in=obj_list_lidar,  ego_obj=obj_ego_dict, data_order_obj=data_order)
        data_order_gt = []

        # Convert the augmented object lists
        for key in self.augmentations_to_load:
            obj_lists_aug_dict[key] = convert_to_ego_centric(obj_list_in=obj_lists_aug_dict[key], ego_obj=obj_ego_dict, data_order_obj=data_order)

    if self.smooth_heading_values:
        obj_list_camera = process_heading_feature(obj_list_in=obj_list_camera, data_order_obj=data_order)
        obj_list_lidar  = process_heading_feature(obj_list_in=obj_list_lidar,  data_order_obj=data_order)

    association_info_aug_dict = {}
    for key in self.augmentations_to_load_associations:
        data = mat_temp[key][0]
        association_info_aug_dict[key] = {
                "associated_objects_idx":       data["associated_objects_idx"][0],
                "associated_objects_track_id":  data["associated_objects_track_id"][0],
                "associated_objects_dists":     data["associated_objects_dists"][0],
                "distance_matrix":              data["distance_matrix"][0],
                "association_params":           {k: data["association_params"][0][k][0][0] for k in data["association_params"][0].dtype.names},
                "key":                          'association_info_camera_lidar',
            }

    
    ###############################################################
    return {
        "general_info":                     general_info,
        "scene_info":                       scene_info,
        "association_info_camera_lidar":    association_info_camera_lidar,
        "obj_ego":                          obj_ego,
        "obj_list_lidar":                   obj_list_lidar,
        "obj_list_camera":                  obj_list_camera,
        "obj_list_gt":                      obj_list_gt,
        "obj_list_lidar_nc":                obj_list_lidar_nc,
        "obj_lists_aug_dict":               obj_lists_aug_dict,
        "obj_lists_aug_dict":               obj_lists_aug_dict,
        "association_info_aug_dict":        association_info_aug_dict,
    }
