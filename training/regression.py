import torch

from datasetclass.cephalic import HeadScanDataset
from pathlib import Path
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from models.cephalic.cpnet import CRegression
from models.cvai.cvpnet import CVRegression
import torch.nn.functional as F

BATCH = 32
EPOCHS = 200
LR = 0.05
EVAL_PERC = 0.2
WEIGHTED = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root = Path(__file__).resolve().parent.parent
NPZ_DIR = root / "dataset/cephalic"
MODELS_DIR = root / "checkpoint"

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
            train_weights.append(2)
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

model = CRegression(1028).to(device)
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

            outputs_real = outputs * target_std + target_mean

            truths_norm = (truths - target_mean) / target_std
            val_loss += F.l1_loss(outputs, truths_norm)

    train_loss /= len(train_loader)
    val_loss   /= len(eval_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = MODELS_DIR / f"checkpoint.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "target_mean": target_mean,
            "target_std": target_std,
        }, save_path)
        print(f"  ✓ Saved best model (val loss: {val_loss:.6f})")

print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")