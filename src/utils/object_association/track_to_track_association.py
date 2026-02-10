'''
Author:         Alexander Fertig
Date:           25.08.2023
Description:    Within a scene, associate independent object lists with each other (similar to track-to-track-association).

Input:          Sample that describes a scene and contains a object list from the lidar, camera and ground truth  
Output:         List of object pairs. An object pair contains the lidar, object, gt_object and ego object and meta information.
'''


import os
import bbox
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment



class ObjectAssociation:
    def __init__(self, association_params=dict):
        self.__association_params = {key: value for key, value in association_params.items()}

        if self.__association_params['dist_measure'] == 'euclidean':
            self.__out_of_reach_threshold = self.__association_params['threshold_distance_eucl']
        elif self.__association_params['dist_measure'] == 'giou':
            self.__out_of_reach_threshold = self.__association_params['threshold_distance_giou']


    def get_3d_union(self, a, b):
        # code from bbox.metrics.iou_3d()
        intersection_points = bbox.geometry.polygon_intersection(a.p[0:4, 0:2], b.p[0:4, 0:2])
        # If intersection_points is empty, means the boxes don't intersect
        if len(intersection_points) == 0:
            inter_vol = 0.0
        else:
            inter_area = bbox.geometry.polygon_area(intersection_points)
            zmax = np.minimum(a.cz, b.cz)
            zmin = np.maximum(a.cz - a.h, b.cz - b.h)
            inter_vol = inter_area * np.maximum(0, zmax - zmin)

        a_vol = a.l * a.w * a.h
        b_vol = b.l * b.w * b.h
        union_vol = (a_vol + b_vol - inter_vol)

        return union_vol
    

    def get_vol_of_c(self, bb1, bb2):

        
        # Calculate the volume of the smallest convex object C, 
        # which is enclosing / contains the boxes bb1 and bb2.
        mesh = trimesh.Trimesh(vertices=[bb1.p1, bb1.p2, bb1.p3, bb1.p4, bb1.p5, bb1.p6, bb1.p7, bb1.p8, 
                                         bb2.p1, bb2.p2, bb2.p3, bb2.p4, bb2.p5, bb2.p6, bb2.p7, bb2.p8])
        # trimesh_c_convex    = mesh.convex_hull.volume
        trimesh_c_bb        = mesh.bounding_box_oriented.volume
        # Disadvantage: small objects which are far away from each other will have a smaller volume of C 
        # than big objects which are close            
        c_volume = trimesh_c_bb

        return c_volume

    
    def calculate_giou(self, obj_1, obj_2, idx_1, idx_2):
        # Range of GIoU:      giou    [-1, 1] (worst, best)
        # Range of GIoU loss: 1-giou  [ 0, 2] (best, worst) -> goal is to minimize
        bb_camera   = bbox.BBox3D(x = obj_1['x'][idx_1], y = obj_1['y'][idx_1], z = obj_1['z'][idx_1], is_center = True,
                                  length = obj_1['size_x'][idx_1], width = obj_1['size_y'][idx_1], height = obj_1['size_z'][idx_1],
                                  euler_angles=[0, 0, obj_1['heading'][idx_1]])
        bb_lidar    = bbox.BBox3D(x = obj_2['x'][idx_2], y = obj_2['y'][idx_2], z = obj_2['z'][idx_2], is_center = True,
                                  length = obj_2['size_x'][idx_2], width = obj_2['size_y'][idx_2], height = obj_2['size_z'][idx_2],
                                  euler_angles=[0, 0, obj_2['heading'][idx_2]])

        iou         = bbox.metrics.iou_3d(bb_camera, bb_lidar)
        union       = self.get_3d_union(bb_camera, bb_lidar)
        c           = self.get_vol_of_c(bb_camera, bb_lidar)
        giou        = iou - ((c - union) / max(c, 1e-10))        
        # Sanity check
        assert -1 <= giou <= 1, "The range of the GIoU is between [-1; 1] the current value is " + str(giou)

        return giou
    
    
    def plot_trajectories_and_dist_metric(self, object_pair, obj_names=['obj_lidar', 'obj_camera']):
        obj_camera  = object_pair['obj_camera']
        obj_lidar   = object_pair['obj_lidar']
        obj_gt      = object_pair['obj_gt']
        obj_ego     = object_pair['obj_ego']
        if 'obj_lidar_nc' in object_pair:
            obj_lidar_nc = object_pair['obj_lidar_nc']
        obj_1       = object_pair[obj_names[0]]
        obj_2       = object_pair[obj_names[1]]       
        dist_hist_eucl  = []
        dist_hist_giou  = []
        overlapping_time_idx = np.intersect1d(obj_1['time_idx_in_scenario_frame'], obj_2['time_idx_in_scenario_frame'])
        for time_idx in overlapping_time_idx:
            idx_1   = obj_1['time_idx_in_scenario_frame'].tolist().index(time_idx)
            idx_2   = obj_2['time_idx_in_scenario_frame'].tolist().index(time_idx)
            ### Euclidean Distance
            pos1    = np.array((obj_1['x'][idx_1], obj_1['y'][idx_1]))
            pos2    = np.array((obj_2['x'][idx_2],  obj_2['y'][idx_2]))
            dist_hist_eucl.append(np.linalg.norm(pos1 - pos2))       
            ### 3D GIOU
            giou = self.calculate_giou(obj_1, obj_2, idx_1, idx_2)
            # Range of GIoU:      giou    [-1, 1] (worst, best)
            # Range of GIoU loss: 1-giou  [ 0, 2] (best, worst) -> goal in to minimize
            giou_loss = 1- giou
            dist_hist_giou.append(giou_loss)   

        mean_factored_giou = (sum(dist_hist_giou) / (len(dist_hist_giou) * 1.1))
        if True:
            plt.rcParams["figure.figsize"] = (20, 10)
            # Plot right top
            plt.subplot(222)
            plt.bar(overlapping_time_idx, dist_hist_eucl)
            plt.ylabel('Euclidean Distance')
            plt.xlabel('time_idx_in_scenario_frame')
            plt.title('Plot of eucl. dist. - mean ' + str(np.round(np.average(dist_hist_eucl), 4)))
            
            # Plot right bottom
            plt.subplot(224)
            plt.bar(overlapping_time_idx, dist_hist_giou)
            plt.ylabel('1 - gIoU')
            plt.xlabel('time_idx_in_scenario_frame')
            plt.title('Plot of (1-gIoU) - mean ' + str(np.round(mean_factored_giou, 4)))

            # Plot left
            plt.subplot(121)
            plt.plot(obj_camera['x'], obj_camera['y'], 'r', label='Camera Object')
            plt.plot(obj_camera['x'][0], obj_camera['y'][0], 'p', color='r')
            plt.plot(obj_lidar['x'], obj_lidar['y'],  'g', label='Lidar Object')  
            plt.plot(obj_lidar['x'][0], obj_lidar['y'][0], 'p', color='g')   
            if obj_gt != []:
                plt.plot(obj_gt['x'], obj_gt['y'], 'b', label='GT Object', alpha=0.5)
                plt.plot(obj_gt['x'][0], obj_gt['y'][0], 'p', color='b', alpha=0.5)
            if 'obj_lidar_nc' in object_pair:
                plt.plot(obj_lidar_nc['x'],    obj_lidar_nc['y'],  'orange', label='Lidar Object Noncausal')  
                plt.plot(obj_lidar_nc['x'][0], obj_lidar_nc['y'][0], 'p', color='orange')   

            plt.plot(obj_ego['x'], obj_ego['y'], 'pink', label='EGO') 
            plt.plot(obj_ego['x'][0], obj_ego['y'][0], 'p', color='pink')    
            plt.axis('equal')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')

            title_text = ('obj_pair_name: ' + object_pair['obj_pair_name'] + ' \n' +
                          'Trajectories of camera, lidar and GT object (global coord. system) \n ' +
                          obj_names[0] + ': id=' + str(int(obj_1['tracking_id'][0])) + ' class=' + obj_1['pred_class'][0] + '  | ' +
                          obj_names[0] + ': id=' + str(int(obj_2['tracking_id'][0])) + ' class=' + obj_2['pred_class'][0])
            if obj_gt != []:
                title_text = title_text + '\n gt: attribute=' + obj_gt['gt_attribute'][0]   + '  category=' + obj_gt['gt_category'][0] 
            plt.title(title_text)
            plt.legend()
            plt.grid()

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            if self.__association_params['show_plots']:
                plt.show()
            
            if self.__association_params['save_plots']:

                if mean_factored_giou > 1.559:
                    output_dir = self.__association_params['output_dir'] + 'association_plots/critical/' 
                else:
                    output_dir = self.__association_params['output_dir'] + 'association_plots/non_critical/' 

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # filename = str(object_pair['obj_pair_global_idx']) + '.png'
                filename = object_pair['obj_pair_name'] + '.png'
                plt.savefig(output_dir + filename)

            plt.close()


    def plot_all_trajectories_in_scene(self, matching_objects):
        ### Create a plot of the whole scene with all trajectories included
        plt.rcParams["figure.figsize"] = (12, 12)
        for object_pair in matching_objects:
            obj_camera  = object_pair['obj_camera']
            obj_lidar   = object_pair['obj_lidar']
            obj_gt      = object_pair['obj_gt']
            obj_ego     = object_pair['obj_ego']
            # Camera
            plt.plot(obj_camera['x'], obj_camera['y'], 'r', alpha=0.5, label='Camera Object')
            plt.plot(obj_camera['x'][0], obj_camera['y'][0], 'p', color='r', alpha=0.5)
            # LiDAR
            plt.plot(obj_lidar['x'], obj_lidar['y'],  'g', alpha=0.5, label='Lidar Object')  
            plt.plot(obj_lidar['x'][0], obj_lidar['y'][0], 'p', color='g', alpha=0.5)
            # Associations
            plt.plot([obj_camera['x'][0], obj_lidar['x'][0]], [obj_camera['y'][0], obj_lidar['y'][0]], color='k', alpha=0.5, linestyle="dashed" )
            # GT
            if obj_gt != []:
                plt.plot(obj_gt['x'], obj_gt['y'], 'b', alpha=0.5, label='GT Object')
                plt.plot(obj_gt['x'][0], obj_gt['y'][0], 'p', color='b', alpha=0.5)
            # LiDAR Non-causal
            if 'obj_lidar_nc' in object_pair:
                obj_lidar_nc = object_pair['obj_lidar_nc']
                plt.plot(obj_lidar_nc['x'],    obj_lidar_nc['y'],  'orange', alpha=0.5, label='Lidar Object Noncausal')  
                plt.plot(obj_lidar_nc['x'][0], obj_lidar_nc['y'][0], 'p', color='orange', alpha=0.5)   
            # EGO
            plt.plot(obj_ego['x'], obj_ego['y'], 'pink', label='EGO') 
            plt.plot(obj_ego['x'][0], obj_ego['y'][0], 'p', color='pink')    
        plt.axis('equal')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.grid()
        scene_name = matching_objects[0]["scene_info"]["scene"]["name"]
        title = scene_name + " with " + str(len(matching_objects)) + " associations"
        plt.title(title)
        # Do not repeat the labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    	# Save plot        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        output_dir = self.__association_params['output_dir'] + 'plot_scenes/' 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = scene_name + '.png'
        plt.savefig(output_dir + filename)
        plt.close()

    def plot_associated_pairs(self, sample, association_info, output_folder='plot_associations/'):
        key_obj_1 = association_info['associated_modalities']['1']
        key_obj_2 = association_info['associated_modalities']['2']

        ### Create a plot of the lidar and camera obejct for each association within a scene
        for idx_c, idx_l in association_info['associated_objects_idx']:
            plt.rcParams["figure.figsize"] = (12, 12)
            # Plot camera object            
            plt.plot(sample[key_obj_1][idx_c]['x'], sample[key_obj_1][idx_c]['y'], 'green', alpha=0.8,label=key_obj_1) 
            plt.scatter(sample[key_obj_1][idx_c]['x'][0], sample[key_obj_1][idx_c]['y'][0], color='green', s=5) 
            # Plot lidar object            
            plt.plot(sample[key_obj_2][idx_l]['x'], sample[key_obj_2][idx_l]['y'], 'red', alpha=0.8, label=key_obj_2) 
            plt.scatter(sample[key_obj_2][idx_l]['x'][0], sample[key_obj_2][idx_l]['y'][0], color='red', s=5) 

            plt.axis('equal')
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.grid()
            scene_name = sample["scene_info"]["scene"]["name"]
            title = scene_name + ": plot of associated camera object " + str(idx_c) + " and lidar object "+ str(idx_l)
            plt.title(title)
            # Do not repeat the labels in the legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            # Save plot        
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            output_dir = self.__association_params['output_dir'] + output_folder  
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = scene_name + '_pair_c'+str(idx_c)+ '_l'+str(idx_l)+'.png'
            plt.savefig(output_dir + filename)
            plt.close()



    def plot_scene(self, sample, association_info, output_folder='plot_scenes/' ):        
        key_obj_1 = association_info['associated_modalities']['1']
        key_obj_2 = association_info['associated_modalities']['2']

        ### Create a plot of the whole scene with all trajectories included and made associations
        plt.rcParams["figure.figsize"] = (12, 12)
        # GT objects
        for obj in sample['obj_list_gt']:
            plt.plot(obj['x'],    obj['y'], 'b', alpha=0.5, label='GT Object')
            plt.scatter(obj['x'][0], obj['y'][0], color='b', alpha=0.5, s=5)
        # Camera objects
        for obj in sample[key_obj_1]:
            plt.plot(obj['x'],    obj['y'], 'r', alpha=0.5, label=key_obj_1)
            plt.scatter(obj['x'][0], obj['y'][0], color='r', alpha=0.5, s=5)
        # Lidar objects
        for obj in sample[key_obj_2]:
            plt.plot(obj['x'],    obj['y'], 'g', alpha=0.5, label=key_obj_2)
            plt.scatter(obj['x'][0], obj['y'][0], color='g', alpha=0.5, s=5)
        # EGO 
        plt.plot(sample['obj_list_ego']['x'], sample['obj_list_ego']['y'], 'pink', label='EGO') 
        plt.scatter(sample['obj_list_ego']['x'][0], sample['obj_list_ego']['y'][0], color='pink', s=5) 
        # Association lines
        for idx_c, idx_l in association_info['associated_objects_idx']:
            pos_x = [sample[key_obj_1][idx_c]['x'][0], sample[key_obj_2][idx_l]['x'][0]]
            pos_y = [sample[key_obj_1][idx_c]['y'][0], sample[key_obj_2][idx_l]['y'][0]]
            plt.plot(pos_x, pos_y, color='lime', alpha=1, linestyle='--', label='Associated Objects')   

        plt.axis('equal')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.grid()
        scene_name = sample["scene_info"]["scene"]["name"]
        title = scene_name + " with " + str(len(association_info['associated_objects_idx'])) + " associations between " + str(len(sample[key_obj_1])) + " camera objects and " + str(len(sample[key_obj_2])) + " lidar objects."
        plt.title(title)
        # Do not repeat the labels in the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    	# Save plot        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        output_dir = self.__association_params['output_dir'] + output_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = scene_name + '.png'
        plt.savefig(output_dir + filename)
        plt.close()


    def calculate_distance(self, obj_1, obj_2):
        # Calculate distance for all shared time instances between obj_1 and obj_2

        # Is there a time-idx overlap between obj_1 and obj_2?
        overlapping_time_idx = np.intersect1d(obj_1['time_idx_in_scenario_frame'], obj_2['time_idx_in_scenario_frame'])

        if self.__association_params['threshold_min_temporal_overlap'] <= len(overlapping_time_idx):
            distances = []
            for time_idx in overlapping_time_idx:
                idx_1   = obj_1['time_idx_in_scenario_frame'].tolist().index(time_idx)
                idx_2   = obj_2['time_idx_in_scenario_frame'].tolist().index(time_idx)

                # Calc dist
                if self.__association_params['dist_measure'] == "euclidean":
                    pos1    = np.array((obj_1['x'][idx_1], obj_1['y'][idx_1]))
                    pos2    = np.array((obj_2['x'][idx_2], obj_2['y'][idx_2]))
                    distances.append(np.linalg.norm(pos1 - pos2))          

                elif self.__association_params['dist_measure'] == "giou":
                    # Range of GIoU:      giou    [-1, 1] (worst, best)
                    # Range of GIoU loss: 1-giou  [ 0, 2] (best, worst) 
                    giou = self.calculate_giou(obj_1, obj_2, idx_1, idx_2)
                    loss_giou = 1 - giou    # 1 - giou
                    distances.append(loss_giou)

            dist = sum(distances) / (len(distances) * self.__association_params['factor_reward_longer_tracks'])
        else:
            # If there is no time-overlap, set dist to 1,000
            # -> goal is to minimize / higher distance is worse
            dist = 1000
    
        return dist


    def calc_dist_mat(self, obj_list_1, obj_list_2):
        threshold_detection_length = self.__association_params['threshold_min_detection_length']
        
        dist_mat = np.zeros([len(obj_list_1), len(obj_list_2)])

        for idx_1, obj_1 in enumerate(obj_list_1):
            for idx_2, obj_2 in enumerate(obj_list_2):

                if (threshold_detection_length <= len(obj_1['x']) and 
                    threshold_detection_length <= len(obj_2['x'])):
                    dist = self.calculate_distance(obj_1, obj_2)
                    dist_mat[idx_1, idx_2] = dist
                else:
                    dist_mat[idx_1, idx_2] = 1000
        return dist_mat


    def association_procedure(self, obj_list_1, obj_list_2):
        # Get distance Matrix
        dist_mat = self.calc_dist_mat(obj_list_1, obj_list_2)

        # Get the best matches
        matched_idx_1, matched_idx_2 = linear_sum_assignment(dist_mat)

        # Match
        matched_idx  = []
        distance_idx = []
        for idx_1, idx_2 in zip(matched_idx_1, matched_idx_2):
            if dist_mat[idx_1, idx_2] < self.__out_of_reach_threshold:
                # valid match
                matched_idx.append((idx_1, idx_2))
                distance_idx.append(dist_mat[idx_1, idx_2])
        
        return matched_idx, distance_idx, dist_mat


    def create_associated_object_pairs(self, sample, obj_lists=['obj_list_lidar', 'obj_list_camera'], obj_names=['obj_lidar', 'obj_camera']):

        # Obtain objects lists for obj_lists
        obj_list_1  = sample[obj_lists[0]]
        obj_list_2  = sample[obj_lists[1]]
        obj_list_gt = sample['obj_list_gt']

        # Match between obj_list_1 and obj_list_2
        matched_idx, distances, _ = self.association_procedure(obj_list_1 = obj_list_1, 
                                                               obj_list_2 = obj_list_2)
        
        # Iterate over matches 
        matched_obj_pairs = []
        for i, match in enumerate(matched_idx):
            obj_1 = obj_list_1[match[0]]
            obj_2 = obj_list_2[match[1]]

            # Match ground truth data
            if obj_list_gt != []:
                if 'obj_list_lidar_nc' in obj_lists:
                    if 'obj_list_lidar_nc' == obj_lists[0]: 
                        obj_to_match_gt = obj_1
                    elif 'obj_list_lidar_nc' == obj_lists[1]: 
                        obj_to_match_gt = obj_2
                    else:
                        assert True, "It must be one of the two options."
                else:
                    obj_to_match_gt = obj_1 if len(obj_1['x']) > len(obj_2['x']) else obj_2
                match_gt, _, _ = self.association_procedure(obj_list_1 = [obj_to_match_gt], 
                                                            obj_list_2 = obj_list_gt)
                if len(match_gt) != 0:
                    # Match found
                    obj_gt  = obj_list_gt[match_gt[0][1]]
                else:
                    # No suitable match found
                    obj_gt  = []
            else:
                obj_gt  = []

            # Match camera data (if not included yet)
            if 'obj_list_camera' not in obj_lists and 'obj_list_camera' in sample:
                obj_list_camera     = sample['obj_list_camera']
                if 'obj_list_lidar_nc' == obj_lists[0]: 
                    obj_to_match_camera = obj_1
                elif 'obj_list_lidar_nc' == obj_lists[1]: 
                    obj_to_match_camera = obj_2
                else:
                    assert True, "It must be one of the two options."
                match_camera, _, _ = self.association_procedure(obj_list_1 = [obj_to_match_camera], 
                                                                obj_list_2 = obj_list_camera)
                if len(match_camera) != 0:
                    # Match found
                    obj_camera  = obj_list_camera[match_camera[0][1]]
                else:
                    # No suitable match found
                    obj_camera  = []

            # Save matched object pair in dict-structure
            obj_pair_name = sample['scene_info']['scene']['name'] + "_op-" + str(i)
            distance = {'measure':                  self.__association_params['dist_measure'],
                        'association_distance:':    distances[i] }
            if obj_lists == ['obj_list_lidar', 'obj_list_camera']:
                obj_lidar    = obj_1
                obj_camera   = obj_2
            elif obj_lists == ['obj_list_lidar', 'obj_list_lidar_nc']: 
                obj_lidar    = obj_1
                obj_lidar_nc = obj_2
            else:
                assert True, "unknown object list setting"

            obj_pair    = {
                'obj_pair_name': obj_pair_name,
                'obj_camera':    obj_camera,
                'obj_lidar':     obj_lidar,
                'obj_gt':        obj_gt,
                'obj_ego':       sample['obj_list_ego'],
                'general_info':  sample['general_info'],
                'scene_info':    sample['scene_info'],
                'distance':      distance,
                'obj_matching_between': obj_lists,
            }
            if 'obj_list_lidar_nc' in obj_lists:
                obj_pair['obj_lidar_nc'] =  obj_lidar_nc,
            if 'obj_list_camera' not in obj_lists        and 'obj_list_camera' in sample:
                obj_pair['obj_camera'] = obj_camera

            matched_obj_pairs.append(obj_pair)


        return matched_obj_pairs
    

    def get_association_info_per_scene(self, sample, key_obj_list_1, key_obj_list_2):

        # Obtain objects lists for obj_lists
        obj_list_1  = sample[key_obj_list_1]
        obj_list_2  = sample[key_obj_list_2]
        obj_list_gt = sample['obj_list_gt']

        # Match between obj_list_1 and obj_list_2
        matched_objects_idx, distances, dist_mat = self.association_procedure(obj_list_1 = obj_list_1, 
                                                                              obj_list_2 = obj_list_2)
        
        matched_objects_tracking_ids = []
        for idx_obj_1, idx_obj_2 in matched_objects_idx:
            track_id_obj_1 = int(obj_list_1[idx_obj_1]['tracking_id'][0])
            track_id_obj_2 = int(obj_list_2[idx_obj_2]['tracking_id'][0])
            matched_objects_tracking_ids.append([track_id_obj_1, track_id_obj_2])
    
        # see code base in create_associated_object_pairs()

        association_info_scene = {
            'associated_modalities':        {'1': key_obj_list_1,
                                             '2': key_obj_list_2},
            'associated_objects_idx':       matched_objects_idx,
            'associated_objects_track_id':  matched_objects_tracking_ids,
            'num_associations':             len(matched_objects_idx),
            'associated_objects_dists':     distances,
            'distance_matrix':              dist_mat,
            'association_params':           self.__association_params,
        }

        return association_info_scene
    