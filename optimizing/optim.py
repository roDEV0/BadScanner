from datasetclass.cephalic import HeadScanDataset
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from models.pointnet2.abstract import SetAbstraction
import torch
import torch.nn.functional as F
import optuna
import optunahub
import os

class PointNetTunable(nn.Module):
    def __init__(self, trial: optuna.Trial):
        super().__init__()

        # --- SetAbstraction hypers ---
        # npoint: how many centroids to keep at each level
        npoint1 = trial.suggest_categorical("npoint1", [256, 512, 1024])
        npoint2 = trial.suggest_categorical("npoint2", [128, 256, 512])
        npoint3 = trial.suggest_categorical("npoint3", [64, 128, 256])

        # radius grows by a factor each level
        radius1 = trial.suggest_float("radius1", 0.05, 0.3, step=0.05)
        radius_factor = trial.suggest_float("radius_factor", 1.5, 4.0, step=0.5)
        radius2 = radius1 * radius_factor
        radius3 = radius2 * radius_factor

        # k: number of neighbours sampled inside each ball
        k1 = trial.suggest_categorical("k1", [16, 32, 64])
        k2 = trial.suggest_categorical("k2", [16, 32, 64])
        k3 = trial.suggest_categorical("k3", [16, 32, 64])

        # feature channel widths
        ch1 = trial.suggest_categorical("ch1", [32, 64, 128])
        ch2 = trial.suggest_categorical("ch2", [64, 128, 256])
        ch3 = trial.suggest_categorical("ch3", [128, 256, 512])

        self.abs1 = SetAbstraction(radius1, k1, npoint1, 3, ch1)
        self.abs2 = SetAbstraction(radius2, k2, npoint2, ch1, ch2)
        self.abs3 = SetAbstraction(radius3, k3, npoint3, ch2, ch3)

        # --- MLP head hypers ---
        fc1_out = trial.suggest_categorical("fc1_out", [128, 256, 512])
        fc2_out = trial.suggest_categorical("fc2_out", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

        self.fc1 = nn.Linear(ch3, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 4 * 3)

        self.bn1 = nn.BatchNorm1d(fc1_out)
        self.bn2 = nn.BatchNorm1d(fc2_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        features = points

        points, features = self.abs1(points, features)
        points, features = self.abs2(points, features)
        points, features = self.abs3(points, features)

        features = features.max(dim=1).values

        x = self.dropout(F.relu(self.bn1(self.fc1(features))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x.view(batch_size, 4, 3)

def objective(trial: optuna.Trial):
    torch.cuda.empty_cache()
    try:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_categorical("epochs", [50, 75, 100, 125, 150, 175, 200])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)

        lr_reduction = trial.suggest_float("lr_reduction", 0.1, 0.9, log=True)
        patience = trial.suggest_int("patience", 5, 20)

        BATCH = batch_size
        EPOCHS = epochs
        LR = learning_rate
        EVAL_PERC = 0.2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        root = Path(__file__).resolve().parent.parent
        NPZ_DIR = root / "dataset/cephalic"
        MODELS_DIR = root / "checkpoint"

        dataset = HeadScanDataset(NPZ_DIR, randomize=False)

        indices = list(range(len(dataset)))
        train_indices, eval_indices = train_test_split(
            indices, test_size=EVAL_PERC, shuffle=True
        )

        train_data = HeadScanDataset(NPZ_DIR, randomize=True)
        eval_data = HeadScanDataset(NPZ_DIR, randomize=False)

        train_set = Subset(train_data, train_indices)
        eval_set = Subset(eval_data, eval_indices)

        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)

        eval_loader = DataLoader(eval_set, batch_size=BATCH, shuffle=False)

        train_targets = []
        for _, truths, _ in train_loader:
            train_targets.append(truths)
        train_targets = torch.cat(train_targets, dim=0)

        target_mean = train_targets.mean(dim=0).to(device)
        target_std = train_targets.std(dim=0).to(device)

        print(f"Target mean:\n{target_mean}")
        print(f"Target std:\n{target_std}")

        model = PointNetTunable(trial).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_reduction, patience=patience
        )

        best_val_loss = float("inf")

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
                    val_loss += F.l1_loss(outputs, truths_norm).item()

            train_loss /= len(train_loader)
            val_loss /= len(eval_loader)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = MODELS_DIR / f"checkpoint.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "target_mean": target_mean,
                        "target_std": target_std,
                    },
                    save_path,
                )
                print(f"  ✓ Saved best model (val loss: {val_loss:.6f})")

                torch.cuda.empty_cache()
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

        return best_val_loss
    finally:
        del model, optimizer, scheduler, train_loader, eval_loader, train_set, eval_set
        torch.cuda.empty_cache()

module = optunahub.load_module(package="samplers/auto_sampler")
sampler = module.AutoSampler()
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)

storage = optuna.storages.RDBStorage(
    url=os.getenv("DB_URL"),
    failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
    heartbeat_interval=30,
)

study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    pruner=pruner,
    study_name="pointnet2_headscan",
    storage=os.getenv("DB_URL"),
    load_if_exists=True,
)

study.optimize(objective, n_trials=100)

print("\n=== Best trial ===")
best = study.best_trial
print(f"  Val loss : {best.value:.6f}")
print("  Params   :")
for k, v in best.params.items():
    print(f"{k}: {v}")

fig = optuna.visualization.plot_param_importances(study)
fig.write_html("param_importances.html")