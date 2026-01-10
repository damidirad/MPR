import optuna
import copy
import torch
import mpr
from fairness_training import train_fair_mf_mpr

args = mpr.args
device = mpr.device
train_data = mpr.train_data
valid_data = mpr.valid_data
test_data = mpr.test_data
orig_sensitive_attr = mpr.orig_sensitive_attr
s0_known = mpr.s0_known
s1_known = mpr.s1_known
MF_model = mpr.MF_model
sst_model = mpr.sst_model
rmse_thresh = mpr.rmse_thresh

def objective(trial):
    # ---- hyperparameters to tune ----
    fair_reg = trial.suggest_float("fair_reg", 10, 300, log=True)
    beta = trial.suggest_float("beta", args.beta / 4, args.beta, log=True)
    num_priors = trial.suggest_categorical("num_priors", [10, 15, 20, 25])

    print("Trial with fair_reg: {}, beta: {}, num_priors: {}".format(
        fair_reg, beta, num_priors
    ))
    resample_range = torch.linspace(0.01, 0.99, num_priors).to(device)

    # ---- fresh model copies ----
    model = copy.deepcopy(MF_model).to(device)
    sst = copy.deepcopy(sst_model).to(device)

    # ---- train (short run) ----
    val_rmse, _, val_unfair, _, _, _ = train_fair_mf_mpr(
        model=model,
        sst_model=sst,
        df_train=train_data,
        epochs=50,                     # short but representative
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        beta=beta,
        valid_data=valid_data,
        test_data=test_data,
        resample_range=resample_range,
        oracle_sensitive_attr=orig_sensitive_attr,
        fair_reg=fair_reg,
        s0_known=s0_known,
        s1_known=s1_known,
        device=device,
        rmse_thresh=args.rmse_thresh
    )
    if val_rmse > rmse_thresh + 0.05:  # small tolerance
        raise optuna.TrialPruned()
    
    # ---- penalty for violating RMSE ----
    penalty = max(0.0, val_rmse - rmse_thresh) * 100.0

    objective_value = val_unfair + penalty

    # optional logging
    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("val_unfair", val_unfair)

    return objective_value

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=1)
)

study.optimize(objective, n_trials=30)