import torch
from torch import nn, threshold
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
        self.embedding_net = nn.Sequential(
            nn.Linear(emb_size, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z_u, p0):
            if not isinstance(p0, torch.Tensor):
                p0 = torch.full((z_u.size(0), 1), p0).to(z_u.device)
            elif p0.dim() == 1:
                p0 = p0.view(-1, 1)

            # Force the model to process embeddings in a prior-blind way first
            features = self.embedding_net(z_u)
            combined = torch.cat([features, p0], dim=1)

            return self.classifier(combined)
    

def train_amortized_sst(
    sst_model,
    mf_model,
    s0,
    s1,
    epochs=50,
    batch_size=1024,
    alpha=0.1,  # weight for prior matching constraint
    device="cuda"
):
    optimizer = torch.optim.Adam(sst_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss(reduction="none")

    # data
    user_ids = torch.cat([torch.tensor(s0), torch.tensor(s1)]).long()
    labels = torch.cat([torch.zeros(len(s0)), torch.ones(len(s1))]).float()
    hat_p1 = len(s1) / (len(s0) + len(s1))

    dataset = TensorDataset(user_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # fixed priors for stable gradient estimates
    anchor_priors = [0.1, 0.3, 0.5, 0.7, 0.9]

    sst_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_u, batch_l in pbar:
            batch_u, batch_l = batch_u.to(device), batch_l.to(device)
            
            # don't update mf weights
            z_u = mf_model.user_emb(batch_u).detach()
            
            batch_total_loss = 0
            
            # train model on multiple worlds
            for p_val in anchor_priors:
                # forward pass
                preds = sst_model(z_u, p_val).view(-1)
                
                # importance weighting for specific prior
                weights = torch.where(
                    batch_l == 1,
                    p_val / hat_p1,
                    (1 - p_val) / (1 - hat_p1)
                )
                
                # data loss
                data_loss = (criterion(preds, batch_l) * weights).mean()
                
                # force the model to respect prior
                prior_match = (preds.mean() - p_val).pow(2)
                
                batch_total_loss += data_loss + alpha * prior_match

            # backward pass for all priors combined
            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_total_loss.item()
            pbar.set_postfix({"loss": f"{batch_total_loss.item():.4f}"})

def evaluate_amortized_sst(sst_model, mf_model, s0_test, s1_test, device):
    sst_model.eval()
    # Test on a neutral prior (0.5) and extreme priors
    test_priors = [0.1, 0.5, 0.9]
    
    user_ids = torch.cat([torch.tensor(s0_test), torch.tensor(s1_test)]).to(device)

    with torch.no_grad():
        z_u = mf_model.user_emb(user_ids)
        
        print("\n--- Amortized Classifier Calibration ---")
        for p in test_priors:
            preds = sst_model(z_u, p).view(-1)
            print(
                f"Prior {p:.1f} | "
                f"mean(pred)={preds.mean().item():.3f}, "
                f"std(pred)={preds.std().item():.3f}"
            )