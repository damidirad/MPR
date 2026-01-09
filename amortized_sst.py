import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
class AmortizedSST(nn.Module):
    """
    Amortized model that takes user embeddings and prior belief about sensitive attribute
    and predicts the sensitive attribute using a feedforward neural network.

    Args:
        emb_size (int): Size of the user embedding vector.
    Inputs:
        z_u (torch.Tensor): User embedding of shape (batch_size, emb_size).
        p0 (float): Prior belief about the sensitive attribute (e.g., probability of being in class 1).
    Outputs:
        torch.Tensor: Predicted sensitive attribute probabilities of shape (batch_size, 1).
    """
    def __init__(self, emb_size):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(emb_size + 1, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),          
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z_u, p0):
        # check if p0 is already a tensor matched to our batch size
        if isinstance(p0, torch.Tensor):
            # if tensor of shape (batch,), reshape to (batch, 1)
            if p0.dim() == 1:
                p0_expanded = p0.view(-1, 1)
            else:
                p0_expanded = p0
        else:
            # if single number, use
            p0_expanded = torch.full((z_u.size(0), 1), p0).to(z_u.device)

        x = torch.cat([z_u, p0_expanded], dim=1)

        return self.fc(x)
    
def train_amortized_sst(sst_model, mf_model, s0, s1, epochs=50, device='cuda'):
    # balanced dataset for training
    user_ids = torch.cat([torch.tensor(s0), torch.tensor(s1)]).long()
    labels = torch.cat([torch.zeros(len(s0)), torch.ones(len(s1))]).float()
    
    # calculate weights for balancing classes
    class_sample_count = torch.tensor([(labels == 0).sum(), (labels == 1).sum()])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[int(t)] for t in labels])
    
    # each batch 50/50 sampling
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    dataset = TensorDataset(user_ids, labels)
    loader = DataLoader(dataset, batch_size=1024, sampler=sampler)

    optimizer = torch.optim.Adam(sst_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    
    sst_model.train()
    
    for _ in tqdm(range(epochs), desc="Training Amortized SST"):
        for batch_u, batch_l in loader:
            batch_u, batch_l = batch_u.to(device), batch_l.to(device)
            
            # get mf embeddings
            z_u = mf_model.user_emb(batch_u).detach()
            
            # sample random prior
            p1 = torch.rand(batch_u.size(0), 1).to(device) * 0.9 + 0.05
            
            preds = sst_model(z_u, p1).view(-1)
            
            # Use the prior-weighted loss logic
            # (Labels are 1 or 0, we want the model to be calibrated to the prior p1)
            loss = criterion(preds, batch_l) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
# def train_amortized_sst(sst_model, mf_model, s0, s1, epochs=20, device='cuda'):
#     optimizer = torch.optim.Adam(sst_model.parameters(), lr=1e-3)
#     criterion = nn.BCELoss(reduction='none') # none to apply weights later
    
#     # combined known sensitive attribute data
#     user_ids = torch.cat([torch.tensor(s0), torch.tensor(s1)]).to(device)
#     labels = torch.cat([torch.zeros(len(s0)), torch.ones(len(s1))]).to(device)
    
#     # empirical ratio
#     hat_p1 = len(s1) / (len(s0) + len(s1))
    
#     sst_model.train()
#     for _ in tqdm(range(epochs), desc="Training Amortized SST"):
#         # get embeddings
#         z_u = mf_model.user_emb(user_ids).detach()
        
#         # sample random prior for each batch
#         batch_p1 = torch.rand(len(user_ids), 1).to(device) * 0.9 + 0.05  # in [0.05, 0.95]
        
#         # fwd pass
#         preds = sst_model(z_u, batch_p1).view(-1)

#         # weighting for importance sampling
#         # weight for class 1: p1 / hat_p1
#         # weight for class 0: (1-p1) / (1-hat_p1)
#         p1_sq = batch_p1.squeeze()
#         weights = torch.where(labels == 1, p1_sq / hat_p1, (1 - p1_sq) / (1 - hat_p1))
        
#         loss = (criterion(preds, labels) * weights).mean()
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

def evaluate_amortized_sst(sst_model, mf_model, s0_test, s1_test, device):
    sst_model.eval()
    # Test on a neutral prior (0.5) and extreme priors
    test_priors = [0.1, 0.5, 0.9]
    
    user_ids = torch.cat([torch.tensor(s0_test), torch.tensor(s1_test)]).to(device)
    labels = torch.cat([torch.zeros(len(s0_test)), torch.ones(len(s1_test))]).to(device)
    
    with torch.no_grad():
        z_u = mf_model.user_emb(user_ids)
        
        print("\n--- Amortized Classifier Diagnostic ---")
        for p in test_priors:
            preds = sst_model(z_u, p).view(-1)
            # Binary prediction based on the current prior's threshold
            # (In a fair world, the threshold is 0.5 because weights handled the bias)
            acc = ((preds > 0.5) == labels).float().mean().item()
            print(f"Accuracy at Prior {p:.1f}: {acc:.2%}")