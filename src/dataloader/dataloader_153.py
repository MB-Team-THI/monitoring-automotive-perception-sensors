import torch
import torchnet as tnt
import math



class dataloader_153(object):
    def __init__(
        self,
        idx=152,
        dataset=None,
        batch_size=None,
        epochs=None,
        num_workers=0,
        num_gpus=1,
        shuffle=False,
        epoch_size=None,
        transformation=None,
        transformation3D=None,
        representation='TBD',
        test=False,
        grid_chosen=None,
        name='TBD',
        description='TBD'
    ):
        self.dataset = dataset[0]
        self.epoch_size = epoch_size if epoch_size is not None else len(self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers,
        self.epochs = epochs
        
        self.test = test
        self.num_gpus = num_gpus

    def _load_function(self, idx):
        idx     = idx % len(self.dataset)
        sample  = self.dataset[idx]

        return sample

    def _collate_fun(self, batch):
        tensors_conversion = True

        general_info        = []
        scene_info          = []
        map_info            = []
        association_info_camera_lidar    = []
        obj_list_camera     = []
        obj_list_lidar      = []
        obj_list_lidar_nc   = []
        obj_ego             = []
        obj_list_gt         = []
        obj_ego_org         = []
        obj_gt_org          = []
        subdivision         = []

        obj_lists_aug_dict  = {}
        association_info_aug_dict = {}

        for elems in batch:
            general_info     .append(elems['general_info'])   
            scene_info       .append(elems['scene_info'])
            # For now just do this for lidar and camera - ego, gt, and lidar-nc are not needed right now
            obj_ego          .append(elems["obj_ego"])
            obj_list_gt      .append(elems["obj_list_gt"])
            obj_list_lidar_nc.append(elems["obj_list_lidar_nc"])
            association_info_camera_lidar.append(elems['association_info_camera_lidar'])

            if len(obj_lists_aug_dict) == 0:
                for key in elems['obj_lists_aug_dict']:
                    obj_lists_aug_dict[key] = []
            if len(association_info_aug_dict) == 0:
                for key in elems['association_info_aug_dict']:
                    association_info_aug_dict[key] = []

            for key in elems["association_info_aug_dict"]:
                association_info_aug_dict[key].append(elems["association_info_aug_dict"][key])

            
            if tensors_conversion:
                ### Convert the ndarray to pytorch tensors 
                # Padding is needed as the single objects within the object list are of different length
                # torch.Size([N_num_obj_max, N_time_max, N_features]) - padded objects with the maximum number of objects object length N_time_max
                # Camera
                obj_list_camera_tensors = [torch.from_numpy(obj).to(torch.float).t() for obj in elems['obj_list_camera']]
                obj_list_camera_padded  = torch.nn.utils.rnn.pad_sequence(obj_list_camera_tensors, batch_first=True)
                obj_list_camera.append(obj_list_camera_padded)
                # LiDAR
                obj_list_lidar_tensors = [torch.from_numpy(obj).to(torch.float).t() for obj in elems['obj_list_lidar']]
                obj_list_lidar_padded  = torch.nn.utils.rnn.pad_sequence(obj_list_lidar_tensors, batch_first=True)
                obj_list_lidar.append(obj_list_lidar_padded)

                # Augmented Object Lists
                for key in elems["obj_lists_aug_dict"]:
                    if 0 < len(elems['obj_lists_aug_dict'][key]):
                        obj_list_aug_tensors = [torch.from_numpy(obj).to(torch.float).t() for obj in elems['obj_lists_aug_dict'][key]]
                        obj_list_aug_padded  = torch.nn.utils.rnn.pad_sequence(obj_list_aug_tensors, batch_first=True)
                        obj_lists_aug_dict[key].append(obj_list_aug_padded)

            else:                
                obj_list_camera     .append(elems["obj_list_camera"])
                obj_list_lidar      .append(elems["obj_list_lidar"])
                for key in obj_lists_aug_dict:
                    if 0 < len(obj_lists_aug_dict[key]):
                        obj_lists_aug_dict[key].append(elems['obj_lists_aug_dict'][key])
               
        
        if tensors_conversion:
            # torch.Size([N_batch, N_num_obj_max, N_time_max, N_features]) - padded objects with the maximum N_num_obj_max
            obj_list_camera = torch.nn.utils.rnn.pad_sequence(obj_list_camera, batch_first=True)
            obj_list_lidar  = torch.nn.utils.rnn.pad_sequence(obj_list_lidar, batch_first=True)            
            for key in obj_lists_aug_dict:
                if 0 < len(obj_lists_aug_dict[key]):
                    obj_lists_aug_dict[key] = torch.nn.utils.rnn.pad_sequence(obj_lists_aug_dict[key], batch_first=True)

        out_dict = {
            "general_info":         general_info,
            "scene_info":           scene_info,
            "map_info":             map_info,
            "association_info_camera_lidar": association_info_camera_lidar,
            "obj_list_camera":      obj_list_camera,
            "obj_list_lidar":       obj_list_lidar,
            "obj_ego":              obj_ego,
            "obj_list_gt":          obj_list_gt,
            "obj_lidar_nc":         obj_list_lidar_nc,
            "obj_lists_aug_dict":   obj_lists_aug_dict,
            "association_info_aug_dict": association_info_aug_dict,
            "obj_ego_org":          obj_ego_org,
            "obj_gt_org":           obj_gt_org,
            "subdivision":          subdivision,
        }
        
        return  out_dict
        

    def get_iterator(self, epoch, gpu_idx):
        self.rand_seed1 = epoch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=self._load_function)

        sampler = torch.utils.data.distributed.DistributedSampler(
            tnt_dataset,
            num_replicas=self.num_gpus,
            shuffle=self.shuffle,
            rank=gpu_idx)
        sampler.set_epoch(epoch)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=self._collate_fun,
                                           num_workers=self.num_workers[0],
                                           sampler=sampler)
        return data_loader
    
    def __call__(self, epoch=0, rank=0):
        return self.get_iterator(epoch, rank)
    
    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) / self.num_gpus)