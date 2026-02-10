from src.model.model import model 
    
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder
from fvcore.nn import FlopCountAnalysis, parameter_count


from src.utils.sample_data import alter_object_list_train

from src.utils.calculate_embedding_distance import get_matrix_embedding_pairs
from src.utils.calculate_embedding_distance import get_dist_matrix_from_embeddings

from src.utils.object_association.process_scene_data import get_only_associated_objects_from_scene


architecture_types      = ["scene_encoder_transformer", "Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
processed_objects_types = ['obj_pair', 'obj_lidar', 'obj_camera']
encoderI_types          = ["ViT", "ResNet-18"]
encoderT_types          = ["LSTM-Encoder", "CNN-Encoder","Transformer-Encoder", "Transformer-Encoder"]
merge_types             = ["FC"]
projector_types         = ["MLP"]
decoderT_types          = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']



class object_encoder(nn.Module):
    def __init__(self, *, dim_in, dim_out, depth, heads, dim_trans, dim_mlp, dim_emb_tokes=3, dropout_rate=0.2, use_scaled_sinu_pos_emb=False, output_cls_token=True, output_mean_var=False):
        super().__init__()
        self.emb_token = nn.Parameter(torch.randn(1, dim_in))
        self.dim_emb_tokes = dim_emb_tokes
        assert output_cls_token ^ output_mean_var, "only one version can be used" # XOR
        self.output_cls_token = output_cls_token
        self.output_mean_var  = output_mean_var
        
        if True:
            self.model = ContinuousTransformerWrapper(
                dim_in = dim_in,               # your feature size
                # dim_out = dim_out,             # desired latent embedding dimension
                max_seq_len = 70,
                attn_layers = Encoder(
                    dim   = dim_trans,             # internal model dimension
                    depth = depth,                 # number of layers
                    heads = heads,                 # attention heads   
                ),
                emb_dropout = dropout_rate,
                # use_abs_pos_emb = use_abs_pos_emb
                scaled_sinu_pos_emb = use_scaled_sinu_pos_emb,
            )
        if self.output_mean_var:
            dim_trans = dim_trans * 2
        self.mlp_head = nn.Sequential(nn.Linear(dim_trans, dim_mlp),
                                      nn.ReLU(),
                                      nn.Dropout(dropout_rate),
                                      nn.Linear(dim_mlp, dim_out))


    def forward(self, x=None, x_unpacked=None, x_lengths=None):
        device = x_unpacked.device
        seq_unpacked = x_unpacked
        if len(x_lengths[0]) == 1:
            lens_unpacked = torch.tensor(x_lengths)[0]
            # seq_len_embed = torch.full((len(x_lengths), 1), lens_unpacked / max(lens_unpacked))
        else:
            lens_unpacked = torch.tensor(x_lengths).squeeze()

        # target shape of emb_tokens: [B, C, *], with B = batch size, C = number of channels, and * is not relevant for torch.cat()
        emb_tokens = self.emb_token.expand(seq_unpacked.shape[0], -1, -1)
        if self.dim_emb_tokes == 2:
            seq_unpacked = seq_unpacked.unsqueeze(2)

        emb_tokens = emb_tokens.to(seq_unpacked.device)
        # emb_tokens = einops.rearrange(emb_tokens, 'B T C -> B C T')
        seq_unpacked = torch.cat((emb_tokens, seq_unpacked), dim=1)
        mask = (torch.arange(seq_unpacked.shape[1])[None, :] < lens_unpacked[:, None]+1).to(device)
        self.model.to(mask.device)
        self.mlp_head.to(mask.device)

        # Apply the sequence and mask to the transformer
        z_intermediate = self.model.forward(seq_unpacked, mask=mask)
        if self.output_cls_token:
            return self.mlp_head(z_intermediate[:,0,:])
        
            # masked_z = z_intermediate * mask.unsqueeze(-1)  # (25, 6, 64)
            # seq_embeddings = masked_z.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (25, 64)
        elif self.output_mean_var:
            z = z_intermediate[:, 1:, :]                     # remove emb_token
            mask_content = mask[:, 1:]                       # remove emb_token position
            mask_content = mask_content.unsqueeze(-1)        # [B, seq_len, 1]

            # mean pooling
            z_sum = (z * mask_content).sum(dim=1)
            z_len = mask_content.sum(dim=1).clamp(min=1)     # prevent div-by-zero
            z_mean = z_sum / z_len

            # variance pooling (optional but recommended)
            z_sq = ((z - z_mean.unsqueeze(1)) ** 2) * mask_content
            z_var = z_sq.sum(dim=1) / z_len
            z_pooled = torch.cat([z_mean, z_var], dim=1)     # [B, 2 * dim]

            return self.mlp_head(z_pooled)



class lstm_decoder(nn.Module):    
    def __init__(self, input_size=None, hidden_size=None, output_size=3, num_layers=1): 
        super(lstm_decoder, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.vor_linear  = nn.Linear(hidden_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size  = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers  = self.num_layers, 
                            batch_first = True)
        self.nach_linear = nn.Linear(hidden_size, output_size)           


    def forward(self, x_input, encoder_hidden_states, t_steps):
        if t_steps==0:
            encoder_hidden_states = (encoder_hidden_states[0], encoder_hidden_states[0])
        output, self.hidden = self.lstm(x_input, encoder_hidden_states)

        output = self.nach_linear(output)        
        return output, self.hidden


class transformer_merger_v2(nn.Module):
    def __init__(self, dim_in, dim_trans, heads, depth, dropout=0.1):
        super(transformer_merger_v2, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model         = dim_in,
            nhead           = heads,
            dim_feedforward = dim_trans,
            dropout         = dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)

    def forward(self, x, num_objects=None):
        """
        h: Tensor of shape (batch_size, num_objects, dim_model)
        mask: Optional mask of shape (batch_size, num_objects)
        """
        
        if num_objects is not None:        
            # Determine the maximum number of objects in the batch
            max_num_objects = max(num_objects)
            # Create the mask
            object_lengths_tensor = torch.tensor(num_objects)  # Convert to tensor
            mask = torch.arange(max_num_objects)[None, :] < object_lengths_tensor[:, None]
            # [B x N] -> [N x B]
            # mask = mask.T  # Transpose to match (seq_len, batch_size)
            mask = mask.to(x.device)

        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, dim_model)
        z = self.transformer_encoder(x, src_key_padding_mask = ~mask)
        return z.permute(1, 0, 2)  # Back to (batch_size, num_objects, dim_model)


class transformer_merger_v1(nn.Module):
    # Lakshmans Implementation
    def __init__(self, *, dim_in=64, dim_trans=128, depth=6, heads=8):
        super().__init__()
        self.model = ContinuousTransformerWrapper(
            dim_in = dim_in,
            max_seq_len = 500,              # Maximal number of objects the Merger Encoder can habdle
            attn_layers = Encoder(
                dim = dim_trans,
                depth = depth,
                heads = heads
            )
        )

    def forward(self, x_obj, obj_length_sum, obj_length, dim_i=64, x_infra=None):
        x = torch.zeros((len(obj_length), max(obj_length_sum), dim_i)) # batch_size x valid_obj_length x emb_dim

        x = torch.zeros(x_obj.shape)   # batch_size x Num_Objects x emb_dim

        
        scenario_wise_merge = [] 
        # unpack objects and create scenario wise merge
        for i in range(len(obj_length)):
            scenario_wise_merge_local = []
            # if x_infra is not None:
            #     scenario_wise_merge_local.append(x_infra[i].unsqueeze(0))
            scenario_wise_merge_local.append(x_obj[obj_length_sum[i]:obj_length_sum[i+1],:])
            scenario_wise_merge_local = [j for i in scenario_wise_merge_local for j in i]
            scenario_wise_merge_local = torch.stack(scenario_wise_merge_local,dim=0) # (infra + obj) x emb_dim
            scenario_wise_merge.append(scenario_wise_merge_local)
        scenario_wise_merge = torch.nn.utils.rnn.pad_sequence(scenario_wise_merge, batch_first=True) # batch_size x max(infra + obj) x emb_dim
        #scenario_wise_merge = scenario_wise_merge.to(x_infra.device) # write comment here why the hell I did 
        # if x_infra is not None:
        #     obj_length = torch.Tensor(obj_length)+1 # added infra length
        # else: 
        obj_length = torch.Tensor(obj_length)
        mask = (torch.arange(scenario_wise_merge.shape[1])[None, :] < obj_length[:, None]).to(x_obj.device) #,device = scenario_wise_merge.device # mask out irrelevant objects
        x = self.model.forward(scenario_wise_merge, mask=mask) # batch_size x max(infra + obj) x emb_dim

        batch_size = x_obj.shape[0]
        num_objects = x_obj.shape[1]
        num_objects = x_obj.shape[1]
        x = self.model.forward(x_obj)

        return x, obj_length


class FC(nn.Module):
    def __init__(self, n_layers, dims):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Lin_"+str(i), nn.Linear(dims[i][0], dims[i][1]))
            if i == n_layers-1:
                break
            # self.layers.append(nn.BatchNorm1d(sizes[i + 1]))
            self.layers.add_module("Activation_"+str(i), nn.ReLU())

    def forward(self, x_in):    
        x_out = self.layers(x_in)
        return x_out
    

class scene_encoder_transformer(nn.Module):
    def __init__(
        self,
        framework_task,
        encoderT_type,
        encoderT_args,
        contrastive_args,
        merger_type,
        merger_args,
        partition_z_space           =False,
        z_dim_m                     =16,
        z_dim_ad                    =8,
        input_obj_features          = ['time_idx_in_scenario_frame', 'x', 'y', 'z', 'size_x', 'size_y', 'size_z', 'heading', 'v'],
        obj_pair_starts_at_0        = False,
        distance_measure            = 'cosine',
    ):
        super().__init__()
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'

        self.framework_task     = framework_task
        self.encoderT_type      = encoderT_type
        self.encoderT_args      = encoderT_args
        self.contrastive_args   = contrastive_args
        self.merger_type        = merger_type
        self.merger_args        = merger_args
        self.partition_z_space  = partition_z_space
        self.z_dim_m            = z_dim_m
        self.z_dim_ad           = z_dim_ad
        self.input_obj_features = input_obj_features
        self.distance_measure   = distance_measure
        self.num_obj_features   = len(input_obj_features)
        self.obj_pair_starts_at_0 = obj_pair_starts_at_0
        self.hard_neg_params    = contrastive_args['hard_neg_params']
        # self.norm_min_max_vals  = {key: [math.inf, -math.inf] for key in input_obj_features}
        # Total nuscenes dataset (train, val, and test)
        self.norm_min_max_vals = {'time_idx_in_scenario_frame': [0, 39],
                                  'x': [-95.714,  336.674],
                                  'y': [-210.012, 225.342],
                                  'z':  [-5.31, 7.28],
                                  'size_x': [0.0, 8.073],
                                  'size_y': [0.0, 29.882],
                                  'size_z': [0.0, 9.314],
                                  'heading': [-3.141, 3.141],
                                  'v': [0.0, 20.885],
                                  'pred_class': [0, 10],
                                  'tracking_score': [0.0, 1.0], 
                                  'subclass_value': [1.0, 2.0],}
        number_learnable_params = {}

        # Encoder ========================================================================================================
         
        ### Object Encoder
        if encoderT_type == encoderT_types[2]:
            # "Transformer-Encoder": 
             
            if self.encoderT_args["joint_encoder"]:                
                # One joint encoder (same weights) for camera and lidar data     
                self.obj_encoder = object_encoder(
                        dim_in    = self.num_obj_features,
                        dim_trans = self.encoderT_args["dim_trans"],
                        dim_mlp   = self.encoderT_args["dim_mlp"],
                        dim_out   = self.encoderT_args["dim_out"],
                        depth     = self.encoderT_args["depth"],
                        heads     = self.encoderT_args["heads"],
                        dropout_rate     = self.encoderT_args["dropout_rate"],
                        output_cls_token = self.encoderT_args["output_cls_token"],
                        output_mean_var  = self.encoderT_args["output_mean_var"],
                        use_scaled_sinu_pos_emb = self.encoderT_args["use_scaled_sinu_pos_emb"],
                    )
                number_learnable_params['object_encoder'] = sum(p.numel() for p in self.obj_encoder.parameters() if p.requires_grad)

            else:
                # Two separate encoders (different weights) for camera and lidar data               
                self.obj_encoder_camera = object_encoder(
                        dim_in    = self.num_obj_features,
                        dim_trans = self.encoderT_args["dim_trans"],
                        dim_mlp   = self.encoderT_args["dim_mlp"],
                        dim_out   = self.encoderT_args["dim_out"],
                        depth     = self.encoderT_args["depth"],
                        heads     = self.encoderT_args["heads"],
                        dropout_rate = self.encoderT_args["dropout_rate"],
                    )
                self.obj_encoder_lidar = object_encoder(
                        dim_in    = self.num_obj_features,
                        dim_trans = self.encoderT_args["dim_trans"],
                        dim_mlp   = self.encoderT_args["dim_mlp"],
                        dim_out   = self.encoderT_args["dim_out"],
                        depth     = self.encoderT_args["depth"],
                        heads     = self.encoderT_args["heads"],
                        dropout_rate = self.encoderT_args["dropout_rate"],
                    )
                number_learnable_params['object_encoder_camera'] = sum(p.numel() for p in self.obj_encoder_camera.parameters() if p.requires_grad)
                number_learnable_params['object_encoder_lidar']  = sum(p.numel() for p in self.obj_encoder_lidar.parameters() if p.requires_grad)
        else:
            assert False, "encoderT_type undefined: " + encoderT_type

        ### Projection Head
        if self.contrastive_args['contrastive_loss_enabled']:
            if self.contrastive_args['projection_head_enabled']:
                n_layer     = self.contrastive_args["n_layer"]
                dim_in      = self.contrastive_args["dim_in"]
                dim_hidden  = self.contrastive_args["dim_hidden"]
                dim_out     = self.contrastive_args["dim_out"]
                dims_layers = [[dim_hidden, dim_hidden] for i in range(n_layer)]
                dims_layers[0][0]  = dim_in
                dims_layers[-1][1] = dim_out

                if encoderT_type == encoderT_types[2]:
                    # one joint projection network (same weights) for camera and lidar data
                    self.projection_head = FC(n_layers=n_layer, dims=dims_layers)
                    number_learnable_params['projection_head'] = sum(p.numel() for p in self.projection_head.parameters() if p.requires_grad)   
                
                elif encoderT_type == encoderT_types[3]:
                    # two separate encoders (different weights) for camera and lidar data
                    self.projection_head_camera = FC(n_layers=n_layer, dims=dims_layers)
                    self.projection_head_lidar  = FC(n_layers=n_layer, dims=dims_layers)                
                    number_learnable_params['projection_head_camera'] = sum(p.numel() for p in self.projection_head_camera.parameters() if p.requires_grad)
                    number_learnable_params['projection_head_lidar']  = sum(p.numel() for p in self.projection_head_lidar.parameters() if p.requires_grad)


        ### Merger Encoder
        if merger_type == "None":
            self.merger_encoder = None
        elif merger_type == "Merger-Transformer-v1":
            # "Merger-Transformer-v1": implementation from Lakshman with ContinuousTransformerWrapper()
            self.merger_encoder = transformer_merger_v1(
                    dim_in    = self.merger_args["dim_in"],
                    dim_trans = self.merger_args["dim_trans"],
                    depth     = self.merger_args["depth"],
                    heads     = self.merger_args["heads"],
                )            
            number_learnable_params['merger_encoder'] = sum(p.numel() for p in self.merger_encoder.parameters() if p.requires_grad)
        
        elif merger_type == "Merger-Transformer-v2":
            # "Merger-Transformer-v2": implementation with nn.TransformerEncoderLayer()
            self.merger_encoder = transformer_merger_v2(
                    dim_in    = self.merger_args["dim_in"],
                    dim_trans = self.merger_args["dim_trans"],
                    depth     = self.merger_args["depth"],
                    heads     = self.merger_args["heads"],
                )            
            number_learnable_params['merger_encoder'] = sum(p.numel() for p in self.merger_encoder.parameters() if p.requires_grad)

        else:
            assert False, "merger_type undefined: " + merger_type


    def forward(self, batch_data, epoch):

        ### Preprocessing ===================================================================================================
        # World coordinate system with EGO starting at (0,0) orientated on the x-axis to the right

        ### Features
        # Define the relevant input object features which should be encoded
        data_order_tracking_res = list(batch_data['general_info'][0]['data_order_tracking_res']) + ['subclass_value']
        input_obj_features_idx = []
        for obj_feature in self.input_obj_features:
            assert obj_feature in data_order_tracking_res, "Unknown input object feature: " + obj_feature                
            input_obj_features_idx.append(data_order_tracking_res.index(obj_feature))
        input_obj_features_idx.sort()
        #         
        if self.framework_task == "object_pairs":
            # Filter for only associated objects
            batch_data = get_only_associated_objects_from_scene(batch_data=batch_data, obj_pair_starts_at_0=self.obj_pair_starts_at_0)

            for obj_c, obj_l in zip(batch_data["obj_list_camera"][0], batch_data["obj_list_lidar"][0]):
                assert obj_c[0,1]==0 or obj_l[0,1]==0, "at least one object must start at time_idx = 0"


        # Create hard negative samples   
        if self.hard_neg_params['enabled']:
            # Augment the object lists to create hard negative samples
            batch_data_camera_aug_all_F = alter_object_list_train(batch_data["obj_list_camera"], self.hard_neg_params, data_order_tracking_res, "camera")
            batch_data_lidar_aug_all_F  = alter_object_list_train(batch_data["obj_list_lidar"],  self.hard_neg_params, data_order_tracking_res, "lidar")
            batch_obj_list_camera_aug = batch_data_camera_aug_all_F[:,:,:, input_obj_features_idx]
            batch_obj_list_lidar_aug  = batch_data_lidar_aug_all_F[:,:,:, input_obj_features_idx]
        
        # Only the features of interest
        batch_obj_list_camera = batch_data["obj_list_camera"][:,:,:, input_obj_features_idx]
        batch_obj_list_lidar  = batch_data["obj_list_lidar"][:,:,:, input_obj_features_idx]


        ### Normalization
        # Norm input signal? - how to perform normalization here (between 0 and 1 or -1 and +1)? as I do not know the smallest values available?
        # Get min and max 
        if False:
            # Iterate once over the whole dataset and then save the min-max values
            for idx, key in enumerate(self.input_obj_features):
                for obj_c in batch_obj_list_camera[0]:
                    val = obj_c[:, idx].clone()
                    if min(val).item() < self.norm_min_max_vals[key][0]:
                        self.norm_min_max_vals[key][0] = min(val).item()
                    if self.norm_min_max_vals[key][1] < max(val).item():
                        self.norm_min_max_vals[key][1] = max(val).item()

                for obj_l in batch_obj_list_lidar[0]:
                    val = obj_l[:, idx].clone()
                    if min(val).item() < self.norm_min_max_vals[key][0]:
                        self.norm_min_max_vals[key][0] = min(val).item()
                    if self.norm_min_max_vals[key][1] < max(val).item():
                        self.norm_min_max_vals[key][1] = max(val).item()

        # Apply min max normalization
        for idx, key in enumerate(self.input_obj_features):
            min_val = self.norm_min_max_vals[key][0]
            max_val = self.norm_min_max_vals[key][1]
            for obj_c in batch_obj_list_camera[0]:
                # Count for idx=1 ->('x'-value as time_idx starts with 0)
                obj_c_len = torch.count_nonzero(obj_c[:, 0])
                obj_c[range(0, obj_c_len), idx] = obj_c[range(0, obj_c_len), idx].sub_(min_val).div_(max_val - min_val)
            for obj_l in batch_obj_list_lidar[0]:
                # Count for idx=1 ->('x'-value as time_idx starts with 0)
                obj_l_len = torch.count_nonzero(obj_l[:, 0])
                obj_l[range(0, obj_l_len), idx] = obj_l[range(0, obj_l_len), idx].sub_(min_val).div_(max_val - min_val)
            
            if self.hard_neg_params['enabled']:
                for obj_c_aug in batch_obj_list_camera_aug[0]:
                    # Count for idx=1 ->('x'-value as time_idx starts with 0)
                    obj_c_aug_len = torch.count_nonzero(obj_c_aug[:, 0])
                    obj_c_aug[range(0, obj_c_aug_len), idx] = obj_c_aug[range(0, obj_c_aug_len), idx].sub_(min_val).div_(max_val - min_val)
                for obj_l_aug in batch_obj_list_lidar_aug[0]:
                    # Count for idx=1 ->('x'-value as time_idx starts with 0)
                    obj_l_aug_len = torch.count_nonzero(obj_l_aug[:, 0])
                    obj_l_aug[range(0, obj_l_aug_len), idx] = obj_l_aug[range(0, obj_l_aug_len), idx].sub_(min_val).div_(max_val - min_val)

        ### CUDA
        if torch.cuda.is_available():
            batch_obj_list_camera = batch_obj_list_camera.cuda()
            batch_obj_list_lidar  = batch_obj_list_lidar.cuda()
            if self.hard_neg_params['enabled']:
                batch_obj_list_camera_aug = batch_obj_list_camera_aug.cuda()
                batch_obj_list_lidar_aug  = batch_obj_list_lidar_aug.cuda()

        ### Scene-Encoding ==================================================================================================
        # Encode scene with main traj encoder
        batch_obj_lengths_camera = []
        for idx_batch in range(len(batch_data["obj_list_camera"])):
            lengths = [torch.count_nonzero(torch.tensor([x[2] for x in batch_data["obj_list_camera"][idx_batch][idx]])) for idx in range(len(batch_data["obj_list_camera"][idx_batch]))]
            batch_obj_lengths_camera.append(lengths)

        batch_obj_lengths_lidar = []
        for idx_batch in range(len(batch_data["obj_list_lidar"])):
            lengths = [torch.count_nonzero(torch.tensor([x[2] for x in batch_data["obj_list_lidar"][idx_batch][idx]])) for idx in range(len(batch_data["obj_list_lidar"][idx_batch]))]
            batch_obj_lengths_lidar.append(lengths)
                
        if self.hard_neg_params['enabled']:
            batch_obj_lengths_camera_aug = []
            for idx_batch in range(len(batch_data_camera_aug_all_F)):
                lengths = [torch.count_nonzero(torch.tensor([x[2] for x in batch_data_camera_aug_all_F[idx_batch][idx]])) for idx in range(len(batch_data_camera_aug_all_F[idx_batch]))]
                batch_obj_lengths_camera_aug.append(lengths)

            batch_obj_lengths_lidar_aug = []
            for idx_batch in range(len(batch_data_lidar_aug_all_F)):
                lengths = [torch.count_nonzero(torch.tensor([x[2] for x in batch_data_lidar_aug_all_F[idx_batch][idx]])) for idx in range(len(batch_data_lidar_aug_all_F[idx_batch]))]
                batch_obj_lengths_lidar_aug.append(lengths)

        if self.framework_task == "single_object":
            ###------------- Single Objects -------------
            # Encode ALL object from the Object List -->[Object-Encoder]--> z_1...z_N

            ### Object Enocder for Camera and LiDAR
            if self.encoderT_args["joint_encoder"]: 
                # One Joint Encoder     f: x -> z
                z_objects_camera = self.obj_encoder.forward(x_unpacked = batch_obj_list_camera[0], 
                                                            x_lengths  = batch_obj_lengths_camera)
                z_objects_lidar  = self.obj_encoder.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                            x_lengths  = batch_obj_lengths_lidar)
                if self.contrastive_args['contrastive_loss_enabled']:
                    # Create some augmented view to perform the contrastive learning task for a single object
                    # Should acutally be named z here instead of h (but this is easier with the options)
                    h_objects_camera_other_view = self.obj_encoder.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                        x_lengths  = batch_obj_lengths_camera)
                    h_objects_lidar_other_view  = self.obj_encoder.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                        x_lengths  = batch_obj_lengths_lidar)
                    if self.contrastive_args['projection_head_enabled']:
                        # Prediction head   g: z -> h
                        h_objects_camera = self.projection_head(z_objects_camera)
                        h_objects_lidar  = self.projection_head(z_objects_lidar)                    
                        h_objects_camera_other_view = self.projection_head(h_objects_camera_other_view)
                        h_objects_lidar_other_view  = self.projection_head(h_objects_lidar_other_view)
                
            else:
                # Separate Encoders     f_1: x_1 -> z_1     
                #                       f_2: x_2 -> z_2
                z_objects_camera = self.obj_encoder_camera.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                   x_lengths  = batch_obj_lengths_camera)
                z_objects_lidar  = self.obj_encoder_lidar.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                  x_lengths  = batch_obj_lengths_lidar)                
                if self.contrastive_args['contrastive_loss_enabled']:
                    # Create some augmented view to perform the contrastive learning task for a single object
                    # Should acutally be named z here instead of h (but this is easier with the options)
                    h_objects_camera_other_view = self.obj_encoder_camera.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                                  x_lengths  = batch_obj_lengths_camera)
                    h_objects_lidar_other_view  = self.obj_encoder_lidar.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                                 x_lengths  = batch_obj_lengths_lidar)     
                    if self.contrastive_args['projection_head_enabled']:
                        # Prediction head   g: z -> h
                        h_objects_camera = self.projection_head_camera(z_objects_camera)
                        h_objects_lidar  = self.projection_head_lidar(z_objects_lidar)                    
                        h_objects_camera_other_view = self.projection_head_camera(h_objects_camera_other_view)
                        h_objects_lidar_other_view  = self.projection_head_lidar(h_objects_lidar_other_view)

   
        elif self.framework_task == "object_pairs":
            ###------------- Object Pairs -------------

            # Object List -->[Object-Encoder]--> z_1...z_N
            ### Object Enocder for Camera and LiDAR
            if self.encoderT_type == encoderT_types[2]:
                # One Joint Encoder     f: x -> z
                z_objects_camera = self.obj_encoder.forward(x_unpacked = batch_obj_list_camera[0], 
                                                            x_lengths  = batch_obj_lengths_camera)
                z_objects_lidar  = self.obj_encoder.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                            x_lengths  = batch_obj_lengths_lidar) 
                z_objects_camera = F.normalize(z_objects_camera, p=2, dim=1)
                z_objects_lidar  = F.normalize(z_objects_lidar, p=2, dim=1)

                                                                                                
                if self.contrastive_args['contrastive_loss_enabled']:
                    if self.contrastive_args['projection_head_enabled']:
                        # Prediction head   g: z -> h
                        h_objects_camera = self.projection_head(z_objects_camera)
                        h_objects_lidar  = self.projection_head(z_objects_lidar)        


                if self.hard_neg_params['enabled']:
                    # Execute the same encoding procedure here
                    z_objects_camera_aug = self.obj_encoder.forward(x_unpacked = batch_obj_list_camera_aug[0], 
                                                                    x_lengths  = batch_obj_lengths_lidar_aug)
                    z_objects_lidar_aug  = self.obj_encoder.forward(x_unpacked = batch_obj_list_lidar_aug[0], 
                                                                    x_lengths  = batch_obj_lengths_lidar_aug) 
                    z_objects_camera_aug = F.normalize(z_objects_camera_aug, p=2, dim=1)
                    z_objects_lidar_aug  = F.normalize(z_objects_lidar_aug, p=2, dim=1) 

                    if self.contrastive_args['contrastive_loss_enabled']:
                        if self.contrastive_args['projection_head_enabled']:
                            # Prediction head   g: z -> h
                            h_objects_camera_aug = self.projection_head(z_objects_camera_aug)
                            h_objects_lidar_aug  = self.projection_head(z_objects_lidar_aug)     


            elif self.encoderT_type == encoderT_types[3]:
                # Separate Encoders     f_1: x_1 -> z_1     
                #                       f_2: x_2 -> z_2
                z_objects_camera = self.obj_encoder_camera.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                   x_lengths  = batch_obj_lengths_camera)
                z_objects_lidar  = self.obj_encoder_lidar.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                  x_lengths  = batch_obj_lengths_lidar)   
                z_objects_camera = F.normalize(z_objects_camera, p=2, dim=1)
                z_objects_lidar  = F.normalize(z_objects_lidar, p=2, dim=1)             
                if self.contrastive_args['contrastive_loss_enabled']: 
                    # create the h-space for the contrastive learning task
                    if self.contrastive_args['projection_head_enabled']:
                        # Prediction head   g: z -> h
                        h_objects_camera = self.projection_head_camera(z_objects_camera)
                        h_objects_lidar  = self.projection_head_lidar(z_objects_lidar)
            
                if self.hard_neg_params['enabled']:
                    # Execute the same encoding procedure here
                    z_objects_camera_aug = self.obj_encoder_camera.forward(x_unpacked = batch_obj_list_camera_aug[0], 
                                                                           x_lengths  = batch_obj_lengths_lidar_aug)
                    z_objects_lidar_aug  = self.obj_encoder_lidar.forward(x_unpacked = batch_obj_list_lidar_aug[0], 
                                                                          x_lengths  = batch_obj_lengths_lidar_aug) 
                    z_objects_camera_aug = F.normalize(z_objects_camera_aug, p=2, dim=1)
                    z_objects_lidar_aug  = F.normalize(z_objects_lidar_aug, p=2, dim=1) 

                    if self.contrastive_args['contrastive_loss_enabled']:
                        if self.contrastive_args['projection_head_enabled']:
                            # Prediction head   g: z -> h
                            h_objects_camera_aug = self.projection_head(z_objects_camera_aug)
                            h_objects_lidar_aug  = self.projection_head(z_objects_lidar_aug)  


        elif self.framework_task == "object_asso_object_pairs": 
            ###------------- Object Association Object Pairs -------------

            # Object List -->[Object-Encoder]--> h_1...h_N -->[Merger-Transformer]--> z_1...z_N
            ### Object Enocder for Camera and LiDAR
            if self.encoderT_type == encoderT_types[2]:
                # One Joint Encoder
                h_objects_camera = self.obj_encoder.forward(x_unpacked = batch_obj_list_camera[0], 
                                                            x_lengths  = batch_obj_lengths_camera)
                h_objects_lidar  = self.obj_encoder.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                            x_lengths  = batch_obj_lengths_lidar)   
            elif self.encoderT_type == encoderT_types[3]:
                # Separate Encoders
                h_objects_camera = self.obj_encoder_camera.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                   x_lengths  = batch_obj_lengths_camera)
                h_objects_lidar  = self.obj_encoder_lidar.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                  x_lengths  = batch_obj_lengths_lidar)  

            ### Object Merger
            if self.merger_encoder != None:
                z_objects_camera = self.merger_encoder.forward(x           = h_objects_camera.unsqueeze(0),
                                                               num_objects = [h_objects_camera.shape[0]])    
                z_objects_camera = z_objects_camera.squeeze(0)
                z_objects_lidar  = self.merger_encoder.forward(x           = h_objects_lidar.unsqueeze(0),
                                                               num_objects = [h_objects_lidar.shape[0]])    
                z_objects_lidar = z_objects_lidar.squeeze(0)
            else:
                z_objects_camera = h_objects_camera
                z_objects_lidar  = h_objects_lidar



        ### Normalization of embeddings z ======================================================================================

        if True:
            # Attemd to make the embedding more invariant to the sequence length
            z_objects_camera =  z_objects_camera /  torch.sqrt(torch.tensor(batch_obj_lengths_camera[0]).float()).unsqueeze(1).to(z_objects_camera.device)
            z_objects_lidar  =  z_objects_lidar /  torch.sqrt(torch.tensor(batch_obj_lengths_lidar[0]).float()).unsqueeze(1).to(z_objects_lidar.device)

        if self.contrastive_args['projection_head_enabled']:
            h_objects_camera = F.normalize(h_objects_camera, p=2, dim=1)
            h_objects_lidar  = F.normalize(h_objects_lidar, p=2, dim=1)
            if self.hard_neg_params['enabled']:
                h_objects_camera_aug = F.normalize(h_objects_camera_aug, p=2, dim=1)
                h_objects_lidar_aug  = F.normalize(h_objects_lidar_aug, p=2, dim=1)

        
        # Embedding Matrix (of raw embeddings)
        matrix_embedding_pairs = get_matrix_embedding_pairs(z_objects_1 = z_objects_camera, 
                                                            z_objects_2 = z_objects_lidar, 
                                                            z_dim_m     = self.z_dim_m)



        ### Output-Variations ==================================================================================================

        if self.framework_task == "object_asso_object_pairs": 
            if self.partition_z_space:

                # Partition Latent Space
                # z_dim_half = int(self.z_dim_m / 2)
                z_objects_camera_asso = z_objects_camera[:,:self.z_dim_ad]
                z_objects_lidar_asso  = z_objects_lidar[:,:self.z_dim_ad]

                test_this_ad_encoder = True
                if test_this_ad_encoder:
                    z_objects_camera_ad = self.obj_encoder_ad.forward(x_unpacked = batch_obj_list_camera[0], 
                                                                    x_lengths  = batch_obj_lengths_camera)
                    z_objects_lidar_ad  = self.obj_encoder_ad.forward(x_unpacked = batch_obj_list_lidar[0], 
                                                                    x_lengths  = batch_obj_lengths_lidar)    
                    z_objects_camera_ad = F.normalize(z_objects_camera_ad, p=2, dim=1)
                    z_objects_lidar_ad  = F.normalize(z_objects_lidar_ad, p=2, dim=1)

                    z_objects_camera = torch.cat((z_objects_camera_asso, z_objects_camera_ad), dim=1)
                    z_objects_lidar = torch.cat((z_objects_lidar_asso, z_objects_lidar_ad), dim=1)

                else:        
                    z_objects_camera_ad   = z_objects_camera[:,self.z_dim_ad:]
                    z_objects_lidar_ad    = z_objects_lidar[:,self.z_dim_ad:]

                # Distance Matrix
                dist_matrix = get_dist_matrix_from_embeddings(z_objects_1      = z_objects_camera_asso, 
                                                            z_objects_2      = z_objects_lidar_asso, 
                                                            distance_measure = self.distance_measure)
                
                # Embedding Matrix (of raw embeddings)
                matrix_embedding_pairs = get_matrix_embedding_pairs(z_objects_1 = z_objects_camera_ad, 
                                                                    z_objects_2 = z_objects_lidar_ad, 
                                                                    z_dim_m     = self.z_dim_ad)            


            else:
                        
                # Embedding Matrix (of raw embeddings)
                matrix_embedding_pairs = get_matrix_embedding_pairs(z_objects_1 = z_objects_camera, 
                                                                    z_objects_2 = z_objects_lidar, 
                                                                    z_dim_m     = self.z_dim_m)
                # Distance Matrix
                dist_matrix = get_dist_matrix_from_embeddings(z_objects_1 = z_objects_camera, 
                                                            z_objects_2 = z_objects_lidar, 
                                                            distance_measure = self.distance_measure)
                
                

                                                
        ### Model Output ===================================================================================================
        if self.framework_task == "single_object":
            if self.contrastive_args['contrastive_loss_enabled']:
                model_output = {'z_objects_camera':             z_objects_camera,
                                'z_objects_lidar':              z_objects_lidar,
                                'h_objects_camera':             h_objects_camera,
                                'h_objects_lidar':              h_objects_lidar,            
                                'h_objects_camera_other_view':  h_objects_camera_other_view,
                                'h_objects_lidar_other_view':   h_objects_lidar_other_view, }
            else:
                model_output = {'z_objects_camera':             z_objects_camera,
                                'z_objects_lidar':              z_objects_lidar,}

        elif self.framework_task == "object_pairs":            
            if self.contrastive_args['contrastive_loss_enabled']:
                model_output = {'z_objects_camera':             z_objects_camera,
                                'z_objects_lidar':              z_objects_lidar,
                                'h_objects_camera':             h_objects_camera,
                                'h_objects_lidar':              h_objects_lidar,}
                if self.hard_neg_params['enabled']:
                    model_output['neg_h_objects_camera'] = h_objects_camera_aug
                    model_output['neg_h_objects_lidar']  = h_objects_lidar_aug
            else:
                model_output = {'z_objects_camera':        z_objects_camera,
                                'z_objects_lidar':         z_objects_lidar}
            

        elif self.framework_task == "object_asso_object_pairs":
            model_output = {'z_objects_camera':        z_objects_camera,
                           'z_objects_lidar':         z_objects_lidar,
                           'matrix_embedding_pairs':  matrix_embedding_pairs,
                           'dist_matrix':             dist_matrix}

        return model_output



class model_153(model):
    def __init__(self, 
                framework_task      = None,
                architecture_type   = None,
                encoderT_type       = None, 
                merger_type         = None, 
                architecture_args   = None,
                contrastive_args    = None,
                encoderT_args       = None, 
                merger_args         = None,
                input_obj_features  = None,     
                obj_pair_starts_at_0 = False,
                matrix_distance_measure = 'cosine',
                partition_z_space   = False,
                z_dim_m             = 128,
                z_dim_ad            = 64,
                idx                 = 153,
                name                = 'Advanced autoencoder approach',
                size                = None,
                n_params            = None,
                input_              = 'Scene-based: whole object list from camera and lidar', 
                output              = 'Distance-matrix of lidar and camera objects', 
                task                = 'Learn what objects are similar within an scene and detect outliers based on it.', 
                description         = 'Scene embedding: clip like embedding of the object list of camera and lidar\
                                        application of contrastive loss based on some association / distance matrix'
                ):        
        super().__init__(idx,name,size,n_params,input_,output,task,description)

        assert architecture_type in architecture_types, 'Architecture keyword unknown'
        self.architecture_type = architecture_type
        self.z_dim_m = z_dim_m

        if self.architecture_type == architecture_types[0]:
            self.model = scene_encoder_transformer(
                framework_task     = framework_task,
                encoderT_type      = encoderT_type,
                encoderT_args      = encoderT_args,
                contrastive_args   = contrastive_args,
                merger_type        = merger_type,
                merger_args        = merger_args,   
                input_obj_features = input_obj_features,
                obj_pair_starts_at_0 = obj_pair_starts_at_0,
                distance_measure   = matrix_distance_measure,
                partition_z_space  = partition_z_space,
                z_dim_m            = z_dim_m,
                z_dim_ad           = z_dim_ad
            )
        else:
            assert False, "No architecture selected!"

    

    def forward(self, batch_data, epoch):
        model_output = self.model.forward(batch_data, epoch)
        return model_output
    
    
    def normalize_prototypes(self):
        if self.architecture_type == architecture_types[0]:
            self.model.normalize_prototypes()
        elif self.architecture_type == architecture_types[2]:
            self.model.encoder.normalize_prototypes()
        return True
    

def generate_model(**model_params):
    return model_153(**model_params)