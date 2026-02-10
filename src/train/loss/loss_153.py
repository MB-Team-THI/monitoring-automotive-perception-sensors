
import torch

import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon

from src.train.loss.loss import loss
from src.utils.object_association.process_scene_data import get_only_associated_objects_from_scene


architecture_types          = ["scene_encoder_transformer", "Encoder", "AE", "AE_advanced_phase1", "AE_advanced_phase2", "AE_error_signal"]
decoderT_types              = ['LSTM-Decoder', 'CNN-Decoder', 'MLP']
processed_objects_types     = ['obj_pair', 'obj_lidar', 'obj_camera']
loss_target_recon_types     = ['obj_pair', 'obj_lidar', 'obj_camera', 'calc_error_signal', 'recon_error_signal']
error_signal_types          = ['euclidean', 'difference_for_each_dim']
error_signal_feature_types  = ['diff_x', 'diff_y', 'directed_distance', 'euclidean_distance', 'mean_distance_to_ego_x', 'mean_distance_to_ego_y']



def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class loss_153(loss):
    def __init__(self,
                idx                     = 153,
                framework_task          = 'single_object',
                architecture_type       = architecture_types[0],
                contrastive_args        = {},
                contrastive_loss_weight = 1.0,
                # loss_type               = "kl",
                z_dim_m                 = 16,
                partition_z_space       = False,
                threshold_inv_association_score = 0.5,
                activate_threshold      = False,
                contr_l_denominator     = "N",
                name                    = 'Single Trajectory Reconstruction MSE Loss',
                description             = 'Advanced Autoencoder phase-1, the loss is applied on each reconstructed trajectory (lidar and camera) separately.',
                input_                  = 'Original and reconstructed trajectories of lidar and camera objects',
                output                  = 'loss-object'
                ) -> None:
        super().__init__(idx,name, description, input_,output)

        self.framework_task          = framework_task
        self.architecture_type       = architecture_type
        # self.loss_type               = loss_type
        self.contrastive_args        = contrastive_args
        self.contrastive_loss_weight = contrastive_loss_weight
        self.hard_neg_params         = contrastive_args['hard_neg_params']
        self.bn                      = nn.BatchNorm1d(2, affine=False)
        self.verbose                 = False
        self.margin                  = 1.0
        self.z_dim_m                 = z_dim_m
        self.partition_z_space       = partition_z_space
        self.kmeans_loss_initialized = False
        self.batch_count             = 0
        self.threshold_inv_association_score = threshold_inv_association_score
        self.activate_threshold      = activate_threshold
        self.contr_l_denominator     = contr_l_denominator

         
        use_cuda = torch.cuda.is_available()        
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # self.contrastive_loss_temp    = nn.Parameter(torch.ones([])*np.log(1/7), requires_grad=True).to(self.device)
        self.contrastive_loss_temp = torch.tensor(0.07).to(self.device)
        # self.contrastive_loss_temp   = torch.ones([], requires_grad=True).to(self.device)*0.5

    
    def distance_measures(self, z_1, z_2, distance_type="neg_cosine_similarity"):
        distance = 0.0

        if distance_type == "neg_cosine_similarity":
            z_2 = z_2.detach()  # stop gradient

            z_1 = F.normalize(z_1, dim=1)  # l2-normalize
            z_2 = F.normalize(z_2, dim=1)  # l2-normalize

        else:
            raise ValueError("Unknown distance type", distance_type)

        return distance


    def VICReg_loss(self, x, y):
        # Copied from the VICReg Github
        beta_sim_coeff = 25.0
        beta_std_coeff = 25.0
        beta_cov_coeff = 1.0
        batch_size = x.shape[0]
        num_features = x.shape[1]
        x = torch.squeeze(x)
        y = torch.squeeze(y)

        # Invariance regularization loss
        repr_loss = F.mse_loss(x, y)

        # Variance regularization loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        # Covariance regularization loss
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = (off_diagonal(cov_x).pow_(2).sum().div(num_features) + 
                    off_diagonal(cov_y).pow_(2).sum().div(num_features))

        loss = (beta_sim_coeff * repr_loss +
                beta_std_coeff * std_loss +
                beta_cov_coeff * cov_loss)
        
        return loss
    

    def MSE(self, x, x_pred):
        loss_function = nn.MSELoss(reduction="mean")
        return loss_function(x, x_pred)
    

    def MSE_batch(self, batch_x, batch_x_pred):
        batch_x = batch_x.to(batch_x_pred.device)

        # Reduce x and x_pred to the original length to not falsely influence the loss by masked elements
        if batch_x.dim() == 2:
            sample_lengths = [torch.count_nonzero(x) for x in batch_x]
        elif batch_x.dim() == 3:
            sample_lengths = [int(torch.count_nonzero(x) / len(x[0])) for x in batch_x]

        loss = 0.0
        for x, x_pred, mask_length in zip(batch_x, batch_x_pred, sample_lengths):
            x_masked      = x[:mask_length]
            x_pred_masked = x_pred[:mask_length]
            # Apply the MSE loss
            loss += self.MSE(x_masked, x_pred_masked)
        return loss


    def cross_entropy_v1(self, y, y_hat):
        loss = -1 * (y * math.log(y_hat) + (1 - y) * math.log(1-y))
        return loss

    
    def cross_entropy_v2(self, y, y_hat):
        return -torch.sum(y * torch.log(y_hat + 1e-9))


    def contrastive_loss(self, association_matrix, embedding_matrix, association_idx, loss_type="contrastive_basic"):
        # Norm:  If you normalize the values so they fall between 0 and 1, with smaller distances corresponding to bigger probabilities (values closer to 1),
        ### Data Transformation [low value = high similarity and high value = low similarity] --> inverse [small value = low similarity and high value = high similarity]
        inverse_association_matrix = association_matrix.copy()
        inverse_association_matrix = torch.from_numpy(1.0 / (1.0 + inverse_association_matrix))
        inverse_association_matrix = inverse_association_matrix.float().to(embedding_matrix.device)
        # This already does the norming between 0 and 1
        num_rows = inverse_association_matrix.shape[0]
        num_cols = inverse_association_matrix.shape[1]
        
        ### Loss
        if loss_type == "bce":
            ### Binary Cross Entropy
            loss_bce = nn.BCELoss(reduction='none')
            # Compute base loss
            base_loss = loss_bce(embedding_matrix, inverse_association_matrix)
            # Sum up losses in rows (for each lidar object finding the best matching camera object)
            row_loss = torch.mean(base_loss, dim=1)
            # Sum up losses in columns (for each camera object, finding the best matching lidar object)
            column_loss = torch.mean(base_loss, dim=0)
            # Combine the losses
            total_loss = (torch.mean(row_loss) + torch.mean(column_loss)) / 2.0

        elif loss_type == "ce":
            ### Cross Entropy Loss
            # Rows
            loss_rows = torch.empty(num_rows)
            for idx_row in range(num_rows):
                loss_rows[idx_row] = self.cross_entropy_v2(y=inverse_association_matrix[idx_row, :], y_hat=embedding_matrix[idx_row, :])
            # Cols
            loss_cols = torch.empty(num_cols)
            for idx_col in range(num_cols):
                loss_cols[idx_col] = self.cross_entropy_v2(y=inverse_association_matrix[:, idx_col], y_hat=embedding_matrix[:, idx_col])
            
            total_loss = (torch.mean(loss_rows) + torch.mean(loss_cols)) / 2.0

        elif loss_type == "kl":
            ### Kullback-Leibler (KL) divergence loss
            
            # Rows            
            loss_rows = torch.empty(num_rows)
            for idx_row in range(num_rows):                
                target_probs = inverse_association_matrix[idx_row, :] #  F.softmax(inverse_association_matrix[idx_row, :], dim=0)
                input_probs = F.softmax(embedding_matrix[idx_row, :], dim=0)
                input_log_probs = torch.log(input_probs)
                loss_rows[idx_row] = torch.nn.functional.kl_div(target=target_probs, input=input_log_probs, reduction='batchmean')
            # Cols
            loss_cols = torch.empty(num_cols)
            for idx_col in range(num_cols):
                target_probs = inverse_association_matrix[:, idx_col]
                input_probs = F.softmax(embedding_matrix[:, idx_col], dim=0)
                input_log_probs = torch.log(input_probs)
                loss_cols[idx_col] = torch.nn.functional.kl_div(target=target_probs, input=input_log_probs, reduction='batchmean')
            
            total_loss = (torch.mean(loss_rows) + torch.mean(loss_cols)) / 2.0

        elif loss_type == "js":
            ### Jensen-Shannon 

            # Rows            
            loss_rows = torch.empty(num_rows)
            for idx_row in range(num_rows):                
                target_probs = inverse_association_matrix[idx_row, :]
                input_probs = F.softmax(embedding_matrix[idx_row, :], dim=0)
                loss_rows[idx_row] = jensenshannon(target_probs.cpu(), input_probs.cpu().detach().numpy())

        elif loss_type == "contrastive_basic": 
            # https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
            # Basic contrastive loss 
            # Compute the loss for each row/col based on the BEST match (if the match is greater than the threshold)  

            logits = embedding_matrix / self.contrastive_loss_temp
            exp_logits = torch.exp(logits)

            ### Rows
            if 1 < logits.shape[0]:
                # Multiple rows available -> suitable for row-loss
                best_matches_rows = inverse_association_matrix.argmax(dim=1)
                if self.activate_threshold:
                    mask_rows = (inverse_association_matrix.gather(1, best_matches_rows.unsqueeze(1)) > self.threshold_inv_association_score).squeeze()
                else:
                    mask_rows = torch.ones(best_matches_rows.shape).bool()
                masked_best_matches_rows_idx = best_matches_rows[mask_rows]

                # Get numerator / Positive Pair
                exp_logits_best_matched = exp_logits[mask_rows, masked_best_matches_rows_idx]
                # Get denominator
                if self.contr_l_denominator == "N":
                    # N
                    exp_sum_rows = exp_logits[mask_rows].sum(dim=1)
                elif self.contr_l_denominator == "2N":
                    # 2N
                    exp_sum_rows = exp_logits[mask_rows].sum(dim=1) + exp_logits[:,masked_best_matches_rows_idx].sum(dim=0)
                elif self.contr_l_denominator == "2*(N-1)":
                    # 2*(N-1)

                    if True:
                        # Old
                        mask_r = torch.ones(exp_logits.shape, device=exp_logits.device).bool()[mask_rows]
                        if mask_rows.dim() != 0:
                            r = torch.arange(sum(mask_rows))    # mask_rows and masked_best_matches_rows_idx must have the same number of rows
                        else:
                            r = torch.arange(sum(1))            # if only one row
                        mask_r[r, masked_best_matches_rows_idx] = 0
                        exp_sum_r = (exp_logits[mask_rows] * mask_r).sum(dim=1)

                        mask_c = torch.ones(exp_logits.shape, device=exp_logits.device).bool()  # Start with all True
                        mask_c[r, masked_best_matches_rows_idx] = False
                        mask_c_T = mask_c.T
                        column_sums = (exp_logits.T * mask_c_T).sum(dim=1) 

                        exp_sum_rows = exp_sum_r + column_sums[masked_best_matches_rows_idx]

                loss_rows = -torch.log(torch.div(exp_logits_best_matched, exp_sum_rows + 1e-8))
            else:
                # Only one row available there -> not suitable for row-loss -> this does not make sense here
                loss_rows = torch.tensor([0.0], device=logits.device)

            ### Cols
            if 1 < exp_logits.shape[1]:
                # Multiple column available -> suitable for column-loss

                # Cross check if the loss-calc for the rows on transposed matrices is the same as the loss-calc for the columns 
                # T
                inverse_association_matrix_T = inverse_association_matrix.T
                exp_logits_T = exp_logits.T

                best_matches_rowsT = inverse_association_matrix_T.argmax(dim=1)
                if self.activate_threshold:
                    mask_rowsT = (inverse_association_matrix_T.gather(1, best_matches_rowsT.unsqueeze(1)) > self.threshold_inv_association_score).squeeze()
                else:
                    mask_rowsT = torch.ones(best_matches_rowsT.shape).bool()
                masked_best_matches_rows_idxT = best_matches_rowsT[mask_rowsT]

                # Get numerator / Positive Pair
                exp_logits_best_matchedT = exp_logits_T[mask_rowsT, masked_best_matches_rows_idxT]
                # Get denominator                
                if self.contr_l_denominator == "N":
                    # N
                    exp_sum_rowsT = exp_logits_T[mask_rowsT].sum(dim=1)
                elif self.contr_l_denominator == "2N":
                    # 2N
                    exp_sum_rowsT = exp_logits_T[mask_rowsT].sum(dim=1) + exp_logits_T[:,masked_best_matches_rows_idxT].sum(dim=0)
                elif self.contr_l_denominator == "2*(N-1)":
                    # 2*(N-1)       

                    if True:
                        # Old             
                        # Mask for columns (excluding the positive pair for each column) 
                        mask_r = torch.ones(exp_logits_T.shape, device=exp_logits.device).bool()[mask_rowsT]
                        r = torch.arange(sum(mask_rowsT))  # mask_rowsT and masked_best_matches_cols_idx must have the same number of rows
                        mask_r[r, masked_best_matches_rows_idxT] = 0
                        exp_sum_c = (exp_logits_T[mask_rowsT] * mask_r).sum(dim=1)

                        # Mask for rows to exclude contributions for the best match row
                        mask_c = torch.ones(exp_logits_T.shape, device=exp_logits.device).bool()
                        mask_c[r, masked_best_matches_rows_idxT] = False
                        mask_c_T = mask_c.T  # Transpose back for row masking
                        row_sums = (exp_logits * mask_c_T).sum(dim=1)

                        # Combine column sums with row contributions
                        exp_sum_rowsT = exp_sum_c + row_sums[masked_best_matches_rows_idxT]

                loss_cols = -torch.log(torch.div(exp_logits_best_matchedT, exp_sum_rowsT + 1e-8))
            else:
                # Only one column available there -> not suitable for column-loss -> this does not make sense here
                loss_cols = torch.tensor([0.0], device=logits.device)
            
            total_loss = (torch.mean(loss_rows) + torch.mean(loss_cols)) / 2.0
            assert not torch.isinf(total_loss).any(), "total_loss is inf"
        
        elif loss_type == "contrastive_advanced":
            # Advanced contrastive loss (considering when no proper match is there by the threshold and weights multiple matches)
            # Compute the loss for each row/col based on ALL matches (if the matches are greater than the threshold)  
            # temp = 0.07

            # Rows            
            loss_rows = torch.zeros(num_rows)
            for idx_row in range(num_rows):
                association_weights = inverse_association_matrix[idx_row,:]
                best_match_idx = association_weights.argmax()

                logits = embedding_matrix[idx_row, :].clone() / self.contrastive_loss_temp
                exp    = torch.exp(logits)
           
                single_row = []
                for idx_c in range(num_cols):
                    if association_weights[idx_c] > self.threshold_inv_association_score:
                        softmax = exp[idx_c] / (torch.sum(exp) + 1e-8)
                        row_col_loss = association_weights[idx_c] * -torch.log(softmax + 1e-8)
                        single_row.append(row_col_loss)
                if len(single_row) != 0:
                    loss_rows[idx_row] = torch.mean(torch.stack(single_row))

            # Cols
            loss_cols = torch.zeros(num_cols)
            for idx_col in range(num_cols):
                association_weights = inverse_association_matrix[:, idx_col]
                best_match_idx = association_weights.argmax()
                logits = embedding_matrix[:, idx_col].clone() / self.contrastive_loss_temp
                exp    = torch.exp(logits)
                loss_cols[idx_col] = -torch.log(exp[best_match_idx] / torch.sum(exp))
                    
                single_col = []                   
                for idx_r in range(num_rows):
                    if association_weights[idx_r] > self.threshold_inv_association_score:
                        softmax = exp[idx_r] / (torch.sum(exp) + 1e-8)
                        col_row_loss = association_weights[idx_r] * -torch.log(softmax + 1e-8)
                        single_col.append(col_row_loss)

                if len(single_col) != 0:
                    loss_cols[idx_col] = torch.mean(torch.stack(single_col))

            # Remove 0 (init) values
            mask_rows = (loss_rows.abs() > 1e-8)
            loss_rows_nonzero = loss_rows[mask_rows]
            mask_cols = (loss_cols.abs() > 1e-8)
            loss_cols_nonzero = loss_cols[mask_cols]

            # Compute overall loss
            total_loss = (torch.mean(loss_rows_nonzero) + torch.mean(loss_cols_nonzero)) / 2.0
            assert not math.isinf(total_loss), "total_loss is inf"

        elif loss_type == "contrastive_loss_per_pos_sample":
            # For all positive pairs / associations
            #       Numerator    = positive pairs
            #       Denonminator = negative pairs: all values of same row and column except positive pair
            logits = embedding_matrix / self.contrastive_loss_temp
            exp_logits = torch.exp(logits)


            numerators   = []
            denominators = []
            for i, j in association_idx:
                # Numerator
                numerators.append(exp_logits[i, j])
                # Denominator
                mask = torch.zeros_like(embedding_matrix, dtype=torch.bool)
                mask[i,:] = True
                mask[:,j] = True
                mask[i,j] = False
                # denominator = (exp_logits * mask).sum()  
                denominators.append((exp_logits * mask).sum()  )
            
            numerators = torch.stack(numerators)            
            denominators = torch.stack(denominators)

            loss = -torch.log(numerators / (denominators + 1e-8))

            total_loss = loss.mean()
            assert not math.isinf(total_loss), "total_loss is inf"


        elif loss_type == "NT-Xent":
            # Scale the similariy scores (=embedding_matrix) by temperature
            logits = embedding_matrix / self.contrastive_loss_temp
            exp_logits = torch.exp(logits)

            # Construct a mask to exclude positive pairs in the denominator
            mask = torch.ones_like(embedding_matrix, dtype=torch.bool)

            # Collect positive similarities and modify mask
            pos_similarities = []
            # positive_pairs = association_idx
            for i, j in association_idx:
                pos_similarities.append(exp_logits[i, j])
                mask[i, j] = False  # Exclude positive pair from denominator

            # pos_similarities = torch.tensor(pos_similarities, device=embedding_matrix.device)
            if len(pos_similarities) == 0:
                pos_similarities = [torch.tensor(0.0, device=embedding_matrix.device)] 
            pos_similarities = torch.stack(pos_similarities)

            # Compute denominator (sum over all negatives for each row)
            denominator = (exp_logits * mask).sum()  # Sum over J for each i

            # Compute NT-Xent loss
            loss = -torch.log(pos_similarities.sum() / (denominator + 1e-8))

            total_loss = loss.mean()
            assert not math.isinf(total_loss), "total_loss is inf"


        elif loss_type == "mse":
            ### Mean Squared Error
            mse = nn.MSELoss(reduction='none')
            base_loss = mse(inverse_association_matrix, embedding_matrix)
            
            row_loss = torch.mean(base_loss, dim=1)
            column_loss = torch.mean(base_loss, dim=0)

            total_loss = (torch.mean(row_loss) + torch.mean(column_loss)) / 2.0

        assert total_loss.requires_grad, "The gradient was lost during the loss calculation"
        return total_loss
    

    def nt_xent_loss(self, z1, z2):
        ## The normal NT-Xent loss
        temp = self.contrastive_args['temp']

        assert z1.shape == z2.shape, "Must be of equal shape to build pairs"
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        B = z1.size(0)
        # Concatenate the two views: shape [2B, D]
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: shape [2B, 2B]
        similarity_matrix = torch.matmul(z, z.T)
        
        # Create a mask to remove self-similarities (diagonal) (not the positive pairs)
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        # Apply mask and reshape to [2B, 2B-1]
        similarity_matrix_masked = similarity_matrix[mask].view(2 * B, -1)
        
        # Compute the positive indices for the masked matrix
        pos_indices = []
        for i in range(2 * B):
            if i < B:
                # For z1, positive is at index i+B in the original matrix.
                # If i+B > i, then in the masked row, the index becomes (i+B - 1)
                pos_indices.append(i + B - 1)
            else:
                # For z2, positive is at index i-B.
                # Since i-B < i, no shift is needed.
                pos_indices.append(i - B)
        pos_indices = torch.tensor(pos_indices, device=z.device).long()  # Shape [2B]
        
        # Scale logits by temperature
        logits = similarity_matrix_masked / temp  # shape [2B, 2B-1]
        
        # Compute log-softmax for all samples - hereby the denominator contains pos. and neg. pairs (only self-similarity is excluded)
        log_probs = F.log_softmax(logits, dim=1)
        # Gather the results for just the positive logits using the correct indices - hereby only pos. pairs remain in the nominator
        loss = -log_probs.gather(1, pos_indices.unsqueeze(1)).squeeze().mean()
        # The positive pairs are included in the denominator as well
        
        # Control the loss
        assert not math.isinf(loss), "nt_xent_loss is inf"
        assert loss.requires_grad,   "nt_xent_loss: no gradient available"
        assert not torch.isnan(loss).item(), "nt_xent_loss is NaN"

        return loss



    def nt_xent_loss_w_hard_negatives(self, z1, z2, z1_neg, z2_neg):
        ## The NT-Xent loss with the inclusion of hard negative samples
        assert z1.shape == z2.shape, "Must be of equal shape to build pairs"        
        assert z1_neg.shape == z2_neg.shape, "Hard negatives must be of equal shape to build pairs"

        temp = self.contrastive_args['temp']
        B = z1.size(0)
        z_all = torch.cat([z1, z2, z1_neg, z2_neg], dim=0)
        
        # Similarity matrix: [4B, 4B] 
        if self.contrastive_args['similarity_m'] == "cosine":
            # Dot-Product with normalization = Consine Similarity
            z_all = F.normalize(z_all, dim=1) 
            sim_matrix = torch.matmul(z_all, z_all.T)
        else:
            # Dot-Product: not normalized -> includes effects from both angle and magnitude
            sim_matrix = torch.matmul(z_all, z_all.T)

        # Remove self-similarities (diagonal)
        mask = ~torch.eye(4 * B, dtype=torch.bool, device=z1.device)
        sim_matrix_masked = sim_matrix[mask].view(4 * B, -1)

        # Construct positive indices for camera  lidar only
        pos_indices = []
        for i in range(B):  # For z_cam[0 to B-1]            
            if i < B:
                # For z1, positive is at index i+B in the original matrix.
                # If i+B > i, then in the masked row, the index becomes (i+B - 1)
                pos_indices.append(i + B - 1)
            else:
                # For z2, positive is at index i-B.
                # Since i-B < i, no shift is needed.
                pos_indices.append(i - B)
        pos_indices = torch.tensor(pos_indices, device=z1.device).long()

        # Apply temperature scaling
        logits = sim_matrix_masked / temp
        log_probs = F.log_softmax(logits, dim=1)
        # The denominator includes the positives, negatives and hard negatives (z1_neg, z2_neg)

        # Only compute loss for the camera-to-lidar positives
        loss = torch.gather(input = -log_probs[:2*B], 
                            dim   = 1, 
                            index = pos_indices.unsqueeze(1)
                            ).mean()
        
        # Control the loss
        assert not math.isinf(loss), "nt_xent_loss_w_hard_negatives is inf"
        assert loss.requires_grad,   "nt_xent_loss_w_hard_negatives: no gradient available"
        assert not torch.isnan(loss).item(), "nt_xent_loss_w_hard_negatives is NaN"

        return loss



    def contrastive_loss_w_realistic_negatives(self, z1, z2, z1_neg, z2_neg):
        ## The NT-Xent loss with the inclusion of hard negative samples
        assert z1.shape == z2.shape, "Must be of equal shape to build pairs"        
        assert z1_neg.shape == z2_neg.shape, "Hard negatives must be of equal shape to build pairs"

        temp = self.contrastive_args['temp']

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z1_neg = F.normalize(z1_neg, dim=1)
        z2_neg = F.normalize(z2_neg, dim=1)

        ### The Cosine similarity scores of positive and negative pairs
        pos_sim = torch.sum(z1 * z2, dim=1) / temp              # camera normal and lidar normal -> positive pair
        neg_sim_1 = torch.sum(z1 * z2_neg, dim=1) / temp        # camera normal and lidar aug    -> negative pair
        neg_sim_2 = torch.sum(z1_neg * z2, dim=1) / temp        # camera aug and lidar normal    -> negative pair
        neg_sim_3 = torch.sum(z1_neg * z2_neg, dim=1) / temp    # camera aug and lidar aug       -> negative pair

        logits = torch.stack([pos_sim, neg_sim_1, neg_sim_2, neg_sim_3], dim=1)     # shape [B, 4]
        labels = torch.zeros(pos_sim.shape[0], dtype=torch.long, device=z1.device)  # positives at index 0

        loss = F.cross_entropy(logits, labels)
        
        # Control the loss
        assert not math.isinf(loss), "contrastive_loss_w_realistic_negatives is inf"
        assert loss.requires_grad,   "contrastive_loss_w_realistic_negatives: no gradient available"
        assert not torch.isnan(loss).item(), "contrastive_loss_w_realistic_negatives is NaN"

        return loss


    def forward(self, batch_data, model_output, epoch):
        """
        Computes loss for each batch.
        """

        overall_loss = 0.0
        if self.architecture_type == architecture_types[0]:
            batch_idx = 0
            log_dict = {}


            ### Object Pairs ###
            if self.framework_task == "object_pairs":

                # Filter for only associated objects
                batch_data = get_only_associated_objects_from_scene(batch_data)
                overall_loss = torch.tensor(0.0)

                if self.contrastive_args['contrastive_loss_enabled']:
                    temp = self.contrastive_args['temp']

                    if self.contrastive_args['contrastive_loss_tpye'] == "nt_xent_loss" and not self.hard_neg_params['enabled']:
                        # nt_xent_loss  without hard negatives
                        contrastive_loss_obj_pairs  = self.nt_xent_loss(z1=model_output['h_objects_camera'], 
                                                                        z2=model_output['h_objects_lidar'])

                        overall_loss = overall_loss + 0.5 * contrastive_loss_obj_pairs
                        log_dict['contrastive_loss_obj_pairs'] = contrastive_loss_obj_pairs

                    if self.contrastive_args['contrastive_loss_tpye'] == "nt_xent_loss" and self.hard_neg_params['enabled']:
                        if self.hard_neg_params['realistic_negatives']:
                            # contrastive loss with realistic pairs
                            contrastive_loss_obj_pairs = self.contrastive_loss_w_realistic_negatives(z1=model_output['h_objects_camera'], 
                                                                                                     z2=model_output['h_objects_lidar'],
                                                                                                     z1_neg=model_output['neg_h_objects_camera'],
                                                                                                     z2_neg=model_output['neg_h_objects_lidar'])

                            overall_loss = overall_loss + 0.5 * contrastive_loss_obj_pairs
                            log_dict['contrastive_loss_obj_pairs'] = contrastive_loss_obj_pairs
                
                        else:
                            # nt_xent_loss  with hard negatives
                            contrastive_loss_obj_pairs = self.nt_xent_loss_w_hard_negatives(z1=model_output['h_objects_camera'], 
                                                                                            z2=model_output['h_objects_lidar'],
                                                                                            z1_neg=model_output['neg_h_objects_camera'],
                                                                                            z2_neg=model_output['neg_h_objects_lidar'])

                            overall_loss = overall_loss + 0.5 * contrastive_loss_obj_pairs
                            log_dict['contrastive_loss_obj_pairs'] = contrastive_loss_obj_pairs
                
            elif self.framework_task == "object_asso_object_pairs":

                y_pred_distance_matrix      = model_output['dist_matrix']
                if self.partition_z_space:
                    ### Partition Latent Space -> z_asso and z_ad

                    # Separate embeddings
                    association_matrix = batch_data['association_info_camera_lidar'][batch_idx]['distance_matrix']
                    association_idx    = batch_data['association_info_camera_lidar'][batch_idx]['associated_objects_idx'] 
                    # Apply Contrastive Loss               
                    contrastive_loss = self.contrastive_loss(association_matrix = association_matrix,
                                                             association_idx    = association_idx,
                                                             embedding_matrix   = y_pred_distance_matrix,
                                                             loss_type          = self.loss_type['object_asso_object_pairs'] )
                    # Apply AD loss
                    matrix_embedding_pairs = model_output['matrix_embedding_pairs']


                                
                    ### Combine the losses to the overall_loss
                    ad_loss_weight = 1.0
                    overall_loss = contrastive_loss 
                
                else:

                    ### Contrastive Loss 
                    use_contrastive_loss = True
                    if use_contrastive_loss:
                        y_gt_association_matrix = batch_data['association_info_camera_lidar'][batch_idx]['distance_matrix']
                        y_gt_association_idx    = batch_data['association_info_camera_lidar'][batch_idx]['associated_objects_idx']                
                        contrastive_loss = self.contrastive_loss(association_matrix = y_gt_association_matrix,
                                                                association_idx    = y_gt_association_idx,
                                                                embedding_matrix   = y_pred_distance_matrix,
                                                                loss_type          = self.loss_type )
                    else:
                        contrastive_loss = 0.0
                    
                    ### Combine the losses to the overall_loss
                    overall_loss = self.contrastive_loss_weight * contrastive_loss
                log_dict['contrastive_loss'] = contrastive_loss

            log_dict['overall_loss'] = overall_loss


        assert not torch.isnan(overall_loss).item(), "loss is NaN"
        assert not math.isinf(overall_loss), "overall_loss is inf"
        return overall_loss, log_dict