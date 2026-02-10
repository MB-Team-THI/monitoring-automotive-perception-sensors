import wandb
import random
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from src.train.train import Training
from src.utils.average_meter import AverageMeter
from src.utils.sample_data import create_sample_data_augmentations, create_sample_data_subclasses

from src.utils.object_association.process_scene_data import create_offset_subclasses

def prep_data(input_data, cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    input_data_out = dict((k, Variable(v)) for k,v in input_data.items())
    input_data = input_data_out
    
    if cuda:
        input_data_out = dict((k, v.cuda()) for k,v in input_data.items())
    return input_data_out

class train_153(Training):
    def __init__(self,
                 idx                      = 153,
                 crops_for_assignment     = None,
                 nmb_crops                = None,
                 temperature              = 0.1,
                 freeze_prototypes_niters = 313,
                 epsilon                  = 0.05,
                 queue                    = None,
                 sinkhorn_iterations      = 3,
                 eval_epochs              = [1, 10, 50, 100],
                 save_embeddings          = ['test-set'],
                 enable_grad_clip         = False,
                 clip_value               = None,
                 dummy_example            = False,
                 use_sample_data          = False,
                 subclass_settings        = {'subclasses_enabled': False},
                 **kwargs) -> None:
        super().__init__()
        if nmb_crops is None:
            nmb_crops = [2]
        if crops_for_assignment is None:
            crops_for_assignment = [0, 1]
        self.description                = "Advanced-autoencoder phase-1 training"
        self.temperature                = temperature
        self.freeze_prototypes_niters   = freeze_prototypes_niters
        self.epsilon                    = epsilon
        self.sinkhorn_iterations        = sinkhorn_iterations
        self.queue                      = queue
        self.eval_epochs                = eval_epochs
        self.save_embeddings            = save_embeddings
        self.enable_grad_clip           = enable_grad_clip
        self.clip_value                 = clip_value
        self.dummy_example              = dummy_example
        self.dummy_rand_vals            = [random.random() for idx in range(32)]
        self.use_sample_data            = use_sample_data
        self.subclass_settings          = subclass_settings
        torch.autograd.set_detect_anomaly(True)


    def run_training(self, model, dataloader_train, loss_fc, optimizer, _, dataloader_val, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, dataset_param_val, run_name):
        self._train(model, dataloader_train, loss_fc, optimizer, dataloader_val, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, dataset_param_val, run_name)

    def _train(self, model, dataloader_train, loss_fc, optimizer, dataloader_val, dataloader_test, eval_fc, dataset_param_test, dataset_param_train, dataset_param_val, run_name):
        wandb.watch(model, log='all', log_freq=10)
        
        epochs = dataloader_train.epochs
        pbar   = tqdm(total=int(epochs * len(dataloader_train.dataset) /
                                dataloader_train.batch_size),
                      desc="init training...".center(50))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        writer = SummaryWriter()

        for epoch in range(epochs):
            loss_record = AverageMeter()
            batch_pass = 0

            # Setup log-dict to track the losses
            log_dict_losses = {}


            dataloader = dataloader_train(epoch, rank=0)            
            model.train()
            batch_idx = 0
            for batch_idx, input_data in enumerate(dataloader, start=epoch * len(dataloader)):      
                # writer.add_embedding(z_test, metadata=labels_test,global_step=batch_idx)

                # ============ Forward pass and Loss ============
                batch_scene_data = {'obj_list_camera':               input_data['obj_list_camera'], 
                                    'obj_list_lidar':                input_data['obj_list_lidar'],
                                    'obj_ego':                       input_data['obj_ego'],
                                    'general_info':                  input_data['general_info'],
                                    'scene_info':                    input_data['scene_info'],
                                    'association_info_camera_lidar': input_data['association_info_camera_lidar'],} 

                if self.subclass_settings['subclasses_enabled'] and not self.use_sample_data:
                    
                    if self.subclass_settings['subclass_mode'] == 'offset':
                        # Add an offset to one value (to simulate sensor malfunction) and treat this as other subclass
                        data_subclasses = create_offset_subclasses(input_data, subclass_params=self.subclass_settings['subclass_offset_params'])

                    batch_scene_data = data_subclasses['batch_subclass_1']
                    if 0 == len(batch_scene_data['association_info_camera_lidar'][0]['associated_objects_idx']):
                        continue


                elif self.use_sample_data:
                    # Use sample data for training (randomly created)
                    if self.subclass_settings['subclasses_enabled']:
                        # Create sample data for the subclass 1
                        batch_scene_data = create_sample_data_subclasses(n_samples=100, i=batch_idx, subclass=1)
                    else:
                        # Create sample data for normal object lists
                        batch_scene_data = create_sample_data_augmentations(n_samples=100, i=batch_idx, random_seq_length=False)
                    batch_scene_data['general_info'] = input_data['general_info']
                    batch_scene_data['obj_ego']      = input_data['obj_ego']
                

                model_output = model(batch_scene_data, epoch)

                batch_pass   += 1
                loss, log_l  = loss_fc(batch_data   = batch_scene_data,
                                       model_output = model_output,
                                       epoch        = epoch)
 
                # ============ Backward pass and optim step  ============
                optimizer.zero_grad()
                loss.backward()                
                assert not torch.isnan(loss).item(), "loss is NaN"
                assert not torch.stack([torch.isnan(p).any() for p in model.parameters()]).any().item(), "model parameters are NaN"

                if self.enable_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)

                optimizer.step()

                # ============ Progress bar and summary writer ============
                pbar.update(1)
                loss_record.update(loss.item(), loss)
                log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                        epoch + 1, epochs, 
                        batch_pass, len(dataloader),
                        round(loss_record.avg.item(), 3)).center(50)
                pbar.set_description(log_msg)
                if batch_idx%10==0:
                    writer.add_scalar("Train Loss", loss, batch_idx)
                    writer.add_scalar("Train Loss - AVG", loss_record.avg, batch_idx)
                logging.info(log_msg)

                # Add the losses of this batch up together
                for k in log_l:
                    if k in log_dict_losses:
                        log_dict_losses[k] += log_l[k]
                    else:
                        log_dict_losses[k] = log_l[k]

            log_dict = {k: (log_dict_losses[k]/len(dataloader)) for k in log_dict_losses}
            log_dict['Epoch'] = epoch
            log_dict['Contrastive Loss Temp'] = loss_fc.contrastive_loss_temp.item()
            log_dict['Avg Loss:'] = loss_record.avg
            wandb.log(log_dict)    

            if epoch in self.eval_epochs:
                eval_fc(model            = model, 
                        dataloader_train = dataloader_train,
                        dataloader_val   = dataloader_val,
                        dataloader_test  = dataloader_test,
                        run_name         = run_name, 
                        epoch            = epoch)

                    
                    
