'''
Functions to find object pairs in the object list, recorded from two different sensors that view the same scenario.
The object list origins from the tracking object challenge (nuScenes), which origin from various sensor-channels (e.g. lidar, camera, radar).
The association is based on temporal and spacial conditions and thresholds.
The output is a list of the associated object pairs, consisting out of the object ids, the IoU, the distance score and the time index overlap.
'''

import numpy as np
import random



def create_negative_examples(matching_objects):
    # Randomly sample negative examples from the object ids in matching_objects.
    for idx in range(len(matching_objects)):
        # If there is only one matching object pair, don't create a negative example.
        if len(matching_objects) > 1:             
            # How much negative examples should be created for each positive example?
            random_idx = random.choice([x for x in range(len(matching_objects)) if x != idx])
            match = {'camera':              matching_objects[idx]['camera'],
                     'lidar':               matching_objects[random_idx]['lidar'],
                     'IoU':                 -1,
                     'dist_score':          -1,
                     'time_idx_overlap':    -1,
                     'object_pair_idx':     len(matching_objects),
                     'idx_overall':         -1,
                     'label':               0,
                    }
            matching_objects.append(match)
    return matching_objects


def compare(obj_s1, object_list_s2, association_parameters):
    # Compare a single object from sensor 1 (=obj_s1) with all objects from sensor 2 (=object_list_s2).
    # Return the trackID_global of obj_s1 and its single best matching object from sensor 2.   
      
    times_idx_scenario_s1 = obj_s1['time_idx_in_scenario_frame']
    best_match = {
        'sensor_1':         obj_s1['trackID_global'][0],
        'sensor_2':         -1,
        'IoU':              -1,
        'dist_score':       -1,
        'time_idx_overlap': -1,
        'label': 1 }
    res = {}

    # Iterate over each frame of obj_s1 (frame-based).
    # Calculate the Euclidean distance for each data point, which fulfills the temporal and spacial criteria:
    # 1. Temporal criteria: the 'time_idx_in_scenario_frame' must be the same for obj_1 and obj_2.
    # 2. Spacial criteria:  the Euclidean distance between the two data points must be smaller than the threshold_spacial_distance 
    #                       (broad filter, so that not for every data point a 'nearest' - but far off - match is found).
    num_frames_s1 = len(times_idx_scenario_s1)
    for idx_frame_l in range(num_frames_s1):
        time_idx_frame_s1 = times_idx_scenario_s1[idx_frame_l]
        pos_s1 = (obj_s1['x'][idx_frame_l], obj_s1['y'][idx_frame_l])

        # Iterate all objects and frames from sensor 2, filter by temporal and spacial criteria.
        obj_black_list = []
        for obj_s2 in object_list_s2:
            if len(obj_s2['x']) < association_parameters['threshold_detection_length']:
                continue
            
            for idx_frame_s2 in range(len(obj_s2['x'])):                
                # Temporal association: filter based on the same time index.
                if (obj_s2['time_idx_in_scenario_frame'][idx_frame_s2] == time_idx_frame_s1):
                    
                    # Spacial association: take the obj with the min distance as best match for this frame.
                    pos_s2 = (obj_s2['x'][idx_frame_s2], obj_s2['y'][idx_frame_s2])
                    eucl_dist_numpy = np.linalg.norm(np.array(pos_s1) - np.array(pos_s2))
                    globalID = obj_s2['trackID_global'][idx_frame_s2]
                    
                    spacial_distance_filter_option = 1
                    
                    if spacial_distance_filter_option == 1:
                        # Filter option 1: only include the current distance in res if it is smaller than threshold_spacial_distance.
                        if eucl_dist_numpy < association_parameters['threshold_spacial_distance']:
                            if globalID not in res:
                                res[globalID] = [-1] * num_frames_s1
                            # Store the eucl_dist for all frames (of all objects) if they are within the thresholds (spacial and temporal).
                            res[globalID][idx_frame_l] = eucl_dist_numpy
                            
                    elif spacial_distance_filter_option == 2:                                     
                        # Filter option 2: if any distance of an object exceeds the threshold_spacial_distance, the object is not considered at all.
                        if eucl_dist_numpy < association_parameters['threshold_spacial_distance'] and globalID not in obj_black_list:
                            if globalID not in res:
                                res[globalID] = [-1] * num_frames_s1
                            # Store the eucl_dist for all frames (of all objects) if they are within the thresholds (spacial and temporal).
                            res[globalID][idx_frame_l] = eucl_dist_numpy
                        elif globalID not in obj_black_list:
                            obj_black_list.append(globalID)
                            if globalID in res:
                                res.pop(globalID)                    
                    else:
                        print('ERROR: spacial_distance_filter_option not defined.')       
                        
                        
    # Evaluate the distances of obj_s1 to the objects of sensor 2, which passed the temporal and spacial criteria (trackID-based) and find the best matching object.
    # Apply another filter, which requires a minimum of n (threshold_temporal_overlap) time-steps overlap of the objects from sensor 1 and sensor 2. 
    # Filter empty values and get the average distance between the objects.
    # Divide the average distance by the number of overlapping frames (to include the length factor e.g., avg(5x 1m) should be better than avg(1x 1m)).
    key_min, dist_score_min = -1, 1000
    for key in res:
        # Filter out the -1 values (no temporal or spacial alignment).
        distances_list = [x for x in res[key] if x != -1]
        num_temporal_overlap = len(distances_list)
        # Minimal number of overlapping frames: threshold_temporal_overlap.
        if num_temporal_overlap > association_parameters['threshold_temporal_overlap']:
            dist_avg = sum(distances_list) / num_temporal_overlap
            # Create a distance score
            dist_score = dist_avg / num_temporal_overlap
            if dist_score < dist_score_min:
                dist_score_min = dist_score
                key_min = key
    
    if key_min == -1:
        # No sufficient matches found.
        best_match['sensor_1'] = -1
    else:
        # It's a match!
        best_match['sensor_2'] = key_min
        best_match['dist_score'] = dist_score_min
        
        for obj_s2 in object_list_s2:
            if obj_s2['trackID_global'][0] == key_min:
                # Get the overlapping time indices.
                best_match['time_idx_overlap'] = [int(x) for x in obj_s2['time_idx_in_scenario_frame'] 
                                                                          if x in obj_s1['time_idx_in_scenario_frame']]
                # Calculate the IoU of the best match.
                time_idx_concat = np.concatenate((obj_s2['time_idx_in_scenario_frame'], 
                                                  obj_s1['time_idx_in_scenario_frame']), axis=None)
                best_match['IoU'] = len(best_match['time_idx_overlap']) / len(np.unique(time_idx_concat))              
                break

        # Sanity checks  
        assert best_match['IoU'] > 0.0, "IoU must be > 0.0"  
        assert best_match['IoU'] <= 1.0, "IoU must be <= 1.0"
        assert best_match['sensor_1'] != -1, "sensor_1 must not be -1"
        assert best_match['sensor_2'] != -1, "sensor_2 must not be -1"
        assert best_match['dist_score'] != -1, "dist_score must not be -1"
        assert len(best_match['time_idx_overlap']) > association_parameters[
            'threshold_temporal_overlap'], "time_idx_overlap must not be greater than the threshold_temporal_overlap"
        assert len(best_match['time_idx_overlap']) <= 41, "time_idx_overlap cannot be longer than the longest scene"
        if not association_parameters['include_negative_examples']:
            assert best_match['label'] == 1, "label must be 1, when negative examples are not included"

    return best_match


def find_associations(sample, association_parameters):
    relevant_channels = association_parameters['relevant_channels']
    # Process a whole sample (aka scene) and find the best matching objects from the different relevant_channels.
    # Return pairs of the best matching objects from different sensors.
    
    # Extract objects per sensor from sample, which contains all objects merged.
    channel_order = [i[0] for i in sample['general_info']['channel_order'][0][0][0]]
    # sensor1 = camera, sensor2 = lidar.
    idx_sensor1 = channel_order.index(relevant_channels[0])
    idx_sensor2 = channel_order.index(relevant_channels[1])
    
    tracking_obj_list = sample['tracking_obj_list']
    objects_lidar  = [x for x in tracking_obj_list if x['src_channel'][0] == idx_sensor2]
    objects_camera = [x for x in tracking_obj_list if x['src_channel'][0] == idx_sensor1]
    matching_objects = []

    # Compare one lidar object to all camera objects.
    for obj_l in objects_lidar:
        # Filter: only consider objects, which at least have a length of threshold_detection_length.   
        if len(obj_l['x']) < association_parameters['threshold_detection_length']:
            continue
        best_match = compare(obj_l, objects_camera, association_parameters)
        # In the result of compare() (=best_matches), the key 'sensor_1' revers to the first parameter and 'sensor_2' to the second parameter.
        match = {'camera':              best_match['sensor_2'],
                 'lidar':               best_match['sensor_1'],
                 'dist_score':          best_match['dist_score'],
                 'IoU':                 best_match['IoU'],
                 'time_idx_overlap':    best_match['time_idx_overlap'],
                 'object_pair_idx':     -1, 
                 'idx_overall':         -1,
                 'label':               best_match['label']
                 }
        matching_objects.append(match)

    # Compare one camera object to all lidar objects.
    for obj_c in objects_camera:
        # Filter: only consider objects, which at least have a length of threshold_detection_length.  
        if len(obj_c['x']) < association_parameters['threshold_detection_length']:
            continue        
        best_match = compare(obj_c, objects_lidar, association_parameters)
        # In the result of compare() (=best_matches), the key 'sensor_1' revers to the first parameter and 'sensor_2' to the second parameter.
        match = {'camera':              best_match['sensor_1'],
                 'lidar':               best_match['sensor_2'],
                 'dist_score':          best_match['dist_score'],
                 'IoU':                 best_match['IoU'],
                 'time_idx_overlap':    best_match['time_idx_overlap'],
                 'object_pair_idx':     -1,
                 'idx_overall':         -1,
                 'label':               best_match['label']
                 }
        matching_objects.append(match)

    # Remove empty entries (= -1) from matching_objects.
    remove_indices = []
    for idx, obj_pair in enumerate(matching_objects):
        if obj_pair['camera']==-1:
            remove_indices.append(idx)
            continue    
    matching_objects = [i for j, i in enumerate(matching_objects) if j not in remove_indices]

    # Remove duplicates from matching_objects.     
    remove_indices = []
    for idx_1, obj_pair_1 in enumerate(matching_objects):
        for idx_2, obj_pair_2 in enumerate(matching_objects):
            # When the same object pair is listed twice, remove the second one.
            if obj_pair_1['camera'] == obj_pair_2['camera'] and obj_pair_1['lidar'] == obj_pair_2['lidar'] and idx_1 != idx_2:
                remove_indices.append(idx_2)
    matching_objects = [i for j, i in enumerate(matching_objects) if j not in remove_indices]
               
    # Add index of the object pair inside the scenes (later verification with the LUT)
    for idx, obj_pair in enumerate(matching_objects):
        matching_objects[idx]['object_pair_idx'] = idx

    return matching_objects