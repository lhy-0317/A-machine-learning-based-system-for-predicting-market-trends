import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 基础路径配置
project_dir = Path(__file__).parent
csv_path = project_dir / "data" / "market_forecast_dataset.csv"
model_dir = project_dir / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# 读入数据
print(f"Loading dataset from {csv_path} ...")
df = pd.read_csv(csv_path)

# 处理列
feature_columns = [
    col
    for col in df.columns
    if col
    not in {
        "date",
        "target_trend",
    }
]
label_map = {"bearish": 0, "sideways": 1, "bullish": 2}
df = df.dropna(subset=["target_trend"]).reset_index(drop=True)
df["label"] = df["target_trend"].map(label_map)

X = df[feature_columns].values.astype(np.float32)
y = df["label"].values

# 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2024,
    stratify=y,
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train_tensor = torch.from_numpy(X_train)
X_val_tensor = torch.from_numpy(X_val)
y_train_tensor = torch.from_numpy(y_train).long()
y_val_tensor = torch.from_numpy(y_val).long()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# 建立模型
input_dim = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, len(label_map)),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 50
best_val_acc = 0.0
best_state = None
history = []

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = epoch_loss / total
    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item() * labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / val_total

    print(
        f"Epoch {epoch:02d}/{num_epochs} | "
        f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
        f"Val Loss {val_loss:.4f} Acc {val_acc:.3f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
    )

if best_state is not None:
    model.load_state_dict(best_state)

# 评估
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        logits = model(features)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\nValidation classification report:")
print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))
print("Confusion matrix:\n", confusion_matrix(all_labels, all_preds))

# 保存模型与预处理信息
model_path = model_dir / "market_trend_model.pth"
scaler_path = model_dir / "feature_scaler.pkl"
metadata_path = model_dir / "model_metadata.json"

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)
metadata = {
    "feature_columns": feature_columns,
    "label_map": label_map,
    "input_dim": input_dim,
    "default_payload": df.iloc[-1][feature_columns].astype(float).round(4).to_dict(),
}
metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
history_path = model_dir / "training_history.json"
history_path.write_text(json.dumps(history, indent=2))

print(f"Saved model state to {model_path}")
print(f"Saved scaler to {scaler_path}")
print(f"Saved metadata to {metadata_path}")
print(f"Saved training history to {history_path}")
