from src.utils.rot_points import rot_points
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def transform_object_lists(object_list_dict_in, rotation_type='basic', min_obj_length=0):

    transformed_obj_lists = {}
    if rotation_type == 'basic':
        ### Option 1 
        # Basic transformation based on initial position and heading of EGO-vehicle

        # Take first sample of EGO-vehicle --> EGO starts at (0,0) 
        obj_ego = object_list_dict_in['obj_list_ego']
        starting_point = {'x':       obj_ego['x'][0],
                          'y':       obj_ego['y'][0],
                          'heading': obj_ego['heading'][0]}

        # Transform the ego obj
        ego_rot = center_and_rotate_trajectory(starting_point, obj=obj_ego)    
        transformed_obj_lists['obj_list_ego_rot'] = ego_rot

        # Transform all object lists based on the starting point     
        for key in object_list_dict_in:
            if key=='obj_list_ego':
                continue
            obj_list = object_list_dict_in[key]
            if obj_list == None:
                transformed_obj_lists[key+'_rot'] = None
            else:
                obj_list_rot = []
                for obj in obj_list:
                    if len(obj['x']) >= min_obj_length:
                        obj_rot = center_and_rotate_trajectory(starting_point, obj=obj)    
                        obj_list_rot.append(obj_rot)
                transformed_obj_lists[key+'_rot'] = obj_list_rot

    elif rotation_type == 'transformation_matrix':
        ### Option 2 
        # Transformation Matrix based on Quaternions 
        # Rotation matrix based on Quaternions (nuScenes provides the EGO orientations by 4 values aka. "Quaternions")
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
        pass

    return transformed_obj_lists


def transform_objects(obj_camera, obj_lidar, obj_ego, obj_gt, rotation_type='ego'):
    # Decide based on what the object the rotation and centering should be done
    if rotation_type == 'ego':
        # Starting point is the position and heading of the 'ego'-vehicle (at the time the first object is seen)
        min_idx = int(min(obj_lidar['time_idx_in_scenario_frame'][0], obj_camera['time_idx_in_scenario_frame'][0]))
        starting_point = {'x':       obj_ego['x'][min_idx],
                          'y':       obj_ego['y'][min_idx],
                          'heading': obj_ego['heading'][min_idx]}
    

    elif rotation_type == 'first_obj':
        # Starting point is the position and heading of the first detected obj (camera or lidar)
        if obj_lidar['time_idx_in_scenario_frame'][0] < obj_camera['time_idx_in_scenario_frame'][0]:
            starting_point = {'x':       obj_lidar['x'][0],
                              'y':       obj_lidar['y'][0],
                              'heading': obj_lidar['heading'][0]}
        else:
            starting_point = {'x':       obj_camera['x'][0],
                              'y':       obj_camera['y'][0],
                              'heading': obj_camera['heading'][0]}

    else:
        raise ValueError('rotation_type not supported')
    
    # Center and rotate the trajectories objects of camera, lidar, ego and gt based on the 'starting_point'
    obj_camera = center_and_rotate_trajectory(starting_point, obj=obj_camera)    
    obj_lidar  = center_and_rotate_trajectory(starting_point, obj=obj_lidar)
    obj_ego    = center_and_rotate_trajectory(starting_point, obj=obj_ego)
    if obj_gt != []:
        obj_gt     = center_and_rotate_trajectory(starting_point, obj=obj_gt)

    return obj_camera, obj_lidar, obj_ego, obj_gt


def center_and_rotate_trajectory(starting_point, obj):
    # Idx=0: x, idx=1: y, idx=2: timestamp, idx=3: heading
    # Set the center and rotate the coordinate system based on the starting points
    # Center
    obj['x'] = obj['x'] - starting_point['x']
    obj['y'] = obj['y'] - starting_point['y']  

    # Rotation
    len_obj   = len(obj['y'])
    new_pos_y = np.zeros(len_obj)
    new_pos_x = np.zeros(len_obj)
    new_heading = np.zeros(len_obj)
    # Note: the incoming angle must be negative
    angle = starting_point['heading']
    for idx in range(len_obj):
        new_pos_x[idx], new_pos_y[idx] = rot_points([obj['x'][idx], obj['y'][idx]], angle)
        new_heading[idx] = - obj['heading'][idx] + angle
        
    obj['x']       = new_pos_x
    obj['y']       = new_pos_y
    obj['heading'] = new_heading

    return obj

def center_single_instance(obj_ego, obj_c, obj_l, idx_t=0, idx_x=1, idx_y=2): 
    # Center the camera and lidar objects for each instance of the ego obj
    # Just return the overlapping camera and lidar objects 

    intersections = sorted(set(obj_c[idx_t]).intersection(obj_l[idx_t]))
    intersection_idx = [int(x) for x in intersections]
    obj_c_new = []
    obj_l_new = []
    for idx in intersection_idx:
        idx_l = list(obj_l[idx_t]).index(idx)
        x = obj_l[idx_x][idx_l] - obj_ego[idx_x][idx] 
        y = obj_l[idx_y][idx_l] - obj_ego[idx_y][idx] 
        obj_l_new.append([x,y])
        
        idx_c = list(obj_c[idx_t]).index(idx)
        x = obj_c[idx_x][idx_c] - obj_ego[idx_x][idx] 
        y = obj_c[idx_y][idx_c] - obj_ego[idx_y][idx] 
        obj_c_new.append([x,y])

    return obj_c_new, obj_l_new, intersection_idx
    

def convert_to_ego_centric(obj_list_in, ego_obj, data_order_obj):
    # Convert the input obj list into an EGO-centric coordinate system
    # Here for every timestep the EGO-vehicle is in the center of the coordinate system and aligned to the right
    # Trajectories may only be plotted in a meaningful way with the positional data (x, y, heading) of the EGO 
    obj_list_out  = []
    idx_time_idx = data_order_obj.index('time_idx_in_scenario_frame')
    idx_x = data_order_obj.index('x')
    idx_y = data_order_obj.index('y')
    idx_heading = data_order_obj.index('heading')

    for obj_idx, obj in enumerate(obj_list_in): 
        obj_new = obj.copy()

        # Get EGO data for obj timesteps
        obj_time_idx = obj[idx_time_idx, :].astype(np.int8)
        ego_x = ego_obj['x'][obj_time_idx]
        ego_y = ego_obj['y'][obj_time_idx]
        ego_heading = ego_obj['heading'][obj_time_idx]

        # Center to EGO-position
        x_new = obj[idx_x, :] - ego_x
        y_new = obj[idx_y, :] - ego_y

        # Rotate around EGO heading
        heading_new = []
        for t_idx in range(len(x_new)):
            angle = ego_heading[t_idx]
            pos_t = [x_new[t_idx], y_new[t_idx]]
            x_new[t_idx], y_new[t_idx] = rot_points(pos_t, angle)
            heading_new.append(- obj[idx_heading, t_idx] + angle)
        
        obj_new[idx_x,:] = x_new
        obj_new[idx_y,:] = y_new
        obj_new[idx_heading,:] = np.array(heading_new)

        obj_list_out.append(obj_new)

    for i in range(len(obj_list_in)):
        assert obj_list_in[i].shape == obj_list_out[i].shape, "Must have the same shape"
    return obj_list_out



def smooth_heading(t, psi, mode='savgol', window=2.5, fps=2):
    w_requested = int(window * fps) | 1
    w_max = len(psi) if len(psi) % 2 else len(psi)-1
    # w = min(w_requested, max(w_max, 3))
    w = min(w_requested, max(w_max, 5))  
    if w < 5:
        return psi                          # no filtering possible

    if mode == 'savgol':
        polyorder = min(3, w - 1)  # ≤ 2 when w == 3
        z = np.exp(1j * psi)
        zr_f = savgol_filter(z.real, w, polyorder)
        zi_f = savgol_filter(z.imag, w, polyorder)
        zf   = zr_f + 1j * zi_f
        return np.angle(z)
    
    elif mode == 'unwrap_spline':
        if len(psi) < 4:                    # splrep needs ≥4 pts
            return psi
        from scipy.interpolate import splrep, splev
        psi_u = np.unwrap(psi)
        tck = splrep(t, psi_u, s=w)
        return np.mod(splev(t, tck), 2*np.pi)
    else:
        raise ValueError(mode)
    

def process_heading_feature(obj_list_in, data_order_obj):
    # The heading values of the tracked object lists are quire noisy
    #
    '''
    - The heading values of the tracked object lists are very noise with a unrealistic flow
    - Postprocess the heading values in order to smooth them
    - Input:  obj_list_in  = object list with noisy heading-values
    - Output: obj_list_out = object list with smoothed heading-values
    '''
    idx_heading = data_order_obj.index('heading')

    obj_list_out = []
    for obj in obj_list_in:
        array_headings = obj[idx_heading,:]
        array_steps    = obj[1,:]

        if len(array_headings) >= 5:
            array_headings_smoothed = smooth_heading(t   = array_steps,
                                                    psi = array_headings)


            obj[idx_heading,:] = array_headings_smoothed

        obj_list_out.append(obj)

    return obj_list_out
