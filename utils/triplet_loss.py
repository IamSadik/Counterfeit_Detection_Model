# utils/triplet_loss.py

import torch
import torch.nn.functional as F

def hard_triplet_mining(embeddings, labels, margin=1.0):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # Euclidean distance
    mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask_neg = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

    hardest_positive_dist = (mask_pos * dist_matrix).max(dim=1)[0]
    hardest_negative_dist = (mask_neg * dist_matrix).min(dim=1)[0]

    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    
    return triplet_loss.mean()

def triplet_loss_with_mining(model, images, labels, margin=1.0):
    embeddings = model(images)
    loss = hard_triplet_mining(embeddings, labels, margin)
    return loss
