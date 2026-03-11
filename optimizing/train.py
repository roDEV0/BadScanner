import torch

from datasetclass.cephalic import HeadScanDataset
from pathlib import Path
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from models.cephalic.cpnet import CRegression
from models.cvai.cvpnet import CVRegression
import torch.nn.functional as F
import optuna
import optunahub

def objective(trial: optuna.trial.Trial):
    torch.cuda.empty_cache()

    BATCH = trial.suggest_categorical("BATCH", [2, 4, 8, 16, 32, 64, 128])
    EPOCHS = trial.suggest_int("EPOCHS", 10, 500)
    LR = trial.suggest_float("LR", 0.00001, 0.1, log=True)
    EVAL_PERC = 0.2
    WEIGHTED = True
    WEIGHT_VALUE = trial.suggest_int("WEIGHT_VALUE", 2, 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = Path(__file__).resolve().parent.parent
    NPZ_DIR = root / "dataset/cephalic"

    dataset = HeadScanDataset(NPZ_DIR, randomize=False)

    indices = list(range(len(dataset)))
    train_indices, eval_indices = train_test_split(indices, test_size=EVAL_PERC, shuffle=True)

    train_data = HeadScanDataset(NPZ_DIR, randomize=True)
    eval_data = HeadScanDataset(NPZ_DIR, randomize=False)

    train_set = Subset(train_data, train_indices)
    eval_set = Subset(eval_data, eval_indices)

    train_weights = []
    if WEIGHTED:
        for _, _, plageo in train_set:
            if plageo:
                train_weights.append(WEIGHT_VALUE)
            else:
                train_weights.append(1)

        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=BATCH, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)

    eval_loader = DataLoader(eval_set, batch_size=BATCH, shuffle=False)

    train_targets = []
    for _, truths, _ in train_loader:
        train_targets.append(truths)
    train_targets = torch.cat(train_targets, dim=0)

    target_mean = train_targets.mean(dim=0).to(device)
    target_std  = train_targets.std(dim=0).to(device)

    print(f"Target mean:\n{target_mean}")
    print(f"Target std:\n{target_std}")

    model = CRegression(1028, trial.suggest_float("drop1", 0.1, 0.5), trial.suggest_float("drop2", 0.1, 0.5), trial.suggest_float("drop3", 0.1, 0.5)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for cloud, truths, _ in train_loader:
            cloud = cloud.to(device)
            truths = truths.to(device)

            optimizer.zero_grad()
            outputs = model(cloud)

            truths_norm = (truths - target_mean) / target_std

            loss = F.l1_loss(outputs, truths_norm)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for cloud, truths, _ in eval_loader:
                cloud = cloud.to(device)
                truths = truths.to(device)

                outputs = model(cloud)

                truths_norm = (truths - target_mean) / target_std
                val_loss += F.l1_loss(outputs, truths_norm)

        train_loss /= len(train_loader)
        val_loss   /= len(eval_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss

module = optunahub.load_module(package="samplers/auto_sampler")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20), sampler=module.AutoSampler())
study.optimize(objective, n_trials=150, n_jobs=3, gc_after_trial=True)

print(f"Best Params: {study.best_params}")