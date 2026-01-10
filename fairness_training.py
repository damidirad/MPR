import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation import validate_fairness, test_fairness
from tqdm import tqdm

def train_fair_mf_mpr(
    model,
    sst_model,
    df_train,
    epochs,
    lr,
    weight_decay,
    batch_size,
    beta,
    valid_data,
    test_data,
    resample_range,
    oracle_sensitive_attr,
    fair_reg,
    s0_known,
    s1_known,
    device,
    evaluation_epoch=3,
    unsqueeze=False,
    shuffle=True,
    rmse_thresh=None
):
    # binary cross entropy loss for ratings
    criterion = nn.BCELoss()
    # adam optimizer for model parameters
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    # set sst model to eval mode
    sst_model.eval()

    # track best metrics and epoch
    test_rmse_in_that_epoch = float("inf")
    best_val_unfairness = float("inf")
    naive_unfairness_test_in_that_epoch = float("inf")
    val_rmse_in_that_epoch = float("inf")
    best_epoch = 0
    achieve_rmse_thresh = False

    # compute number of batches per epoch
    num_batches = len(df_train) // batch_size

    num_priors = resample_range.size(0)
    eps = 1e-8

    resample_tensor = resample_range.clone().detach().to(device)
    p0_flat = resample_tensor.repeat(batch_size, 1).view(-1, 1)

    # loop over epochs
    for epoch in tqdm(range(epochs), desc="Training Fair MF-MPR"):
        loss_total = 0.0
        fair_reg_total = 0.0

        # shuffle data indices for this epoch
        indices = np.arange(len(df_train))
        if shuffle:
            np.random.shuffle(indices)

        # loop over batches
        for batch_idx in range(num_batches):
            batch_ids = indices[batch_idx*batch_size : (batch_idx+1)*batch_size]
            data_batch = df_train.iloc[batch_ids].reset_index(drop=True)

            # prepare input tensors
            train_ratings = torch.FloatTensor(data_batch["label"].values).to(device)
            train_user_input = torch.LongTensor(data_batch["user_id"].values).to(device)
            train_item_input = torch.LongTensor(data_batch["item_id"].values).to(device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)

            # forward pass through mf model
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat.view(-1), train_ratings.view(-1))

            # get embeddings
            current_user_embs = model.user_emb(train_user_input)

            # evaluate all priors every step 
            user_emb_flat = current_user_embs.repeat_interleave(num_priors, dim=0)

            s_hat_flat = sst_model(user_emb_flat, p0_flat)
            s_hat_all = s_hat_flat.view(batch_size, num_priors)

            y_hat_exp = y_hat.unsqueeze(1).expand(-1, num_priors)

            mu_1 = torch.sum(y_hat_exp * s_hat_all, dim=0) / (torch.sum(s_hat_all, dim=0) + eps)
            mu_0 = torch.sum(y_hat_exp * (1 - s_hat_all), dim=0) / (torch.sum(1 - s_hat_all, dim=0) + eps)

            fair_diffs = torch.abs(mu_1 - mu_0) / beta

            # log-sum-exp (robust objective)
            C = fair_diffs.max().detach()
            fair_regulation = fair_reg * beta * (torch.logsumexp(fair_diffs - C, dim=0) + C)

            # total loss
            total_loss = loss + fair_regulation

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_total += loss.item()
            fair_reg_total += fair_regulation.item()

        # print epoch averages
        print(f"epoch {epoch}: avg loss {loss_total/num_batches:.4f}, avg fair reg {fair_reg_total/num_batches:.4f}")

        # evaluate on validation and test set every few epochs
        if epoch % evaluation_epoch == 0:
            rmse_val, naive_unfairness_val = validate_fairness(
                model, valid_data, oracle_sensitive_attr, s0_known, s1_known, device
            )
            rmse_test, naive_unfairness_test = test_fairness(
                model, test_data, oracle_sensitive_attr, device
            )
            
            print(f"validation rmse: {rmse_val:.4f}, partial valid unfairness: {naive_unfairness_val:.4f}")
            print(f"test rmse: {rmse_test:.4f}, unfairness: {naive_unfairness_test:.4f}")

            # track best model if rmse threshold is reached
            if rmse_thresh is not None and rmse_val < rmse_thresh:
                achieve_rmse_thresh = True
                if naive_unfairness_val < best_val_unfairness:
                    best_epoch = epoch
                    best_val_unfairness = naive_unfairness_val
                    val_rmse_in_that_epoch = rmse_val
                    test_rmse_in_that_epoch = rmse_test
                    naive_unfairness_test_in_that_epoch = naive_unfairness_test
                    best_model = copy.deepcopy(model)

    # fallback in case no model reached rmse threshold
    if not achieve_rmse_thresh:
        best_epoch = -1
        best_model = copy.deepcopy(model)

    return (
        val_rmse_in_that_epoch,
        test_rmse_in_that_epoch,
        best_val_unfairness,
        naive_unfairness_test_in_that_epoch,
        best_epoch,
        best_model
    )