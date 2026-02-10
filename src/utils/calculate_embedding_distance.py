
import torch

def get_matrix_embedding_pairs(z_objects_1, z_objects_2, z_dim_m):

    N = len(z_objects_1)
    M = len(z_objects_2)
    D = z_dim_m

    # Embedding Matrix (of raw embeddings)
    matrix_embedding_pairs = torch.stack((z_objects_1.unsqueeze(1).expand(N, M, D), 
                                          z_objects_2.unsqueeze(0).expand(N, M, D)), dim=2)
    
    return matrix_embedding_pairs


def get_dist_matrix_from_embeddings(z_objects_1, z_objects_2, distance_measure):
    ### Distance Matrix (with similarity metrix)
    
    if distance_measure == 'euclidean':
        # Calculate Euclidean (L2) distance (good for low-dimensional spaces)
        # Range: 0 (best) to infinity (worst)
        dist_matrix = torch.dist(z_objects_1, z_objects_2)
        # Conversion -> new range 0 (worst) to infinity (best)

    elif distance_measure == 'cosine':
        # Cosine Similarity (good for high-dimensional spaces) 
        # Range: -1 (worst) to 1 (best)
        # Cosine Angle between two non-zero vectors
        z_objects_camera_norm = torch.nn.functional.normalize(z_objects_1, p=2, dim=1)
        z_objects_lidar_norm  = torch.nn.functional.normalize(z_objects_2,  p=2, dim=1)
        cos_similarity_matrix = torch.mm(z_objects_camera_norm, z_objects_lidar_norm.t())
        # Cosine Distance 
        # Range: transform and scale orignal range (-1=worst, 1=best) to 0=worst and 1=best             
        dist_matrix = (1 + cos_similarity_matrix) / 2.0
        assert (0.0 <= dist_matrix.all()) or (dist_matrix.all() <= 1.0), "Unvalid value in cosine distance matrix min=" + str(dist_matrix.min().item()) + ' max=' + str(dist_matrix.max().item())

    elif distance_measure == 'dot-product':
        # Dot-Product distance (= scalar product)
        # Sum of products of two vectors of equal length
        # Range: 0/-infinity (worst) to infinity (best)
        dist_matrix = torch.einsum('ik,jk->ij', z_objects_1, z_objects_2)

    elif distance_measure == 'mahalanobis':
        # # Calculate Mahalanobis distance
        # # Range: 0 (best) to infinity (worst)
        # # Compute the mean and subtract it from the combined embeddings
        # combined_embeddings = torch.cat((z_objects_2, z_objects_1))
        # mean = torch.mean(combined_embeddings, dim=0)
        # centered_embeddings = combined_embeddings - mean
        # covariance = torch.mm(centered_embeddings.T, centered_embeddings) / (combined_embeddings.shape[0] - 1)
        # precision = torch.inverse(covariance)
        # diff = vec1 - vec2
        # dist_matrix = torch.sqrt((diff @ precision @ diff.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1))
        pass

    else:
        assert False, "distance_measure not defined: " + distance_measure

    
    return dist_matrix