### Chat Transcript:

https://chatgpt.com/share/6937450c-3968-800f-82c4-fe1aa16f16b2


### Prompt I used :

I am doing a machine learning class project where I predict whether a Formula 1 driver will win a race based on mid-race style features that I collected and engineered myself by scraping data from formula1.com.

My dataset is at the granularity of one row per driver per race for the 2025 season (first 15 races). Each row has:

Metadata columns (not used as numeric inputs):

year, round, race_slug, race_name, driver_code, driver_name, team_name

Numeric feature columns (used as model inputs), including things like:

grid_pos, grid_pos_norm (starting grid position and normalized)

pit_stops_int (number of pit stops)

first_pit_frac, last_pit_frac (first/last pit lap divided by total race laps)

pitted_at_all, pit_before_half, total_pit_time_sec (approx mid-race pit strategy information)

fastest_lap_rank_norm, fastest_lap_lap_frac, fastest_lap_delta (pace proxies derived from fastest lap timing within the race)

plus one-hot-encoded team columns like team_Red Bull, team_Mercedes, etc.

The label column:

is_winner ∈ {0,1}, where 1 means that driver ultimately won that race.

I want a strong but not overcomplicated baseline model implemented in PyTorch that satisfies a final project requirement to “ask ChatGPT for a baseline neural network and then go beyond it.”

Please propose a baseline that includes:

How to split the data into train / validation / test, given that I have a round column (race number).

How to select the feature columns and standardize/normalize them.

A simple feed-forward neural network (multi-layer perceptron) architecture that is appropriate for this tabular binary classification task, including layer sizes and activation functions.

The loss function and how to handle class imbalance (only one winner per race), and a reasonable choice of optimizer, learning rate, batch size, and number of epochs.

Basic evaluation metrics I should report (e.g., accuracy, F1 for the positive class, etc.).

Please describe the baseline in a clear, structured way so that I can both (a) implement it directly in code, and (b) paste your answer into my project report as “the baseline suggested by ChatGPT.


### The gpt response : 

Here’s a clean, “good but not crazy” baseline you can literally drop into your report as “the baseline suggested by ChatGPT” and also implement in PyTorch.

1. Data Splitting Strategy

Goal: respect the temporal structure of the season and avoid information leakage by not training on later rounds and evaluating on earlier ones.

Your dataset has one row per (driver, race) with a round column (1–15).

Split by round, making sure that all drivers from a given round go into the same split.

A simple, reasonable split:

Train: rounds 1–10

Validation: rounds 11–13

Test: rounds 14–15

This gives the model enough races to learn on, a few races to tune hyperparameters / early stopping, and a couple of final races as a held-out test set.

Implementation-wise:

train_df = df[df["round"] <= 10]
val_df   = df[(df["round"] >= 11) & (df["round"] <= 13)]
test_df  = df[df["round"] >= 14]

2. Feature Selection and Normalization
2.1. Feature Columns

Drop non-numeric metadata from the model inputs:

Exclude: ["year", "round", "race_slug", "race_name", "driver_code", "driver_name", "team_name"]

Include:

Core numeric race/driver features:

grid_pos, grid_pos_norm

pit_stops_int

first_pit_frac, last_pit_frac

pitted_at_all, pit_before_half, total_pit_time_sec

fastest_lap_rank_norm, fastest_lap_lap_frac, fastest_lap_delta

Plus all team one-hot columns: e.g. team_Red Bull, team_Mercedes, …

Let:

continuous_features = [
    "grid_pos", "grid_pos_norm",
    "pit_stops_int",
    "first_pit_frac", "last_pit_frac",
    "total_pit_time_sec",
    "fastest_lap_rank_norm",
    "fastest_lap_lap_frac",
    "fastest_lap_delta"
]

binary_features = [
    "pitted_at_all", "pit_before_half",
    # plus team one-hot columns, e.g.:
    # "team_Red Bull", "team_Mercedes", ...
]

feature_cols = continuous_features + binary_features


The label is:

label_col = "is_winner"

2.2. Standardization / Normalization

For continuous_features, apply z-score standardization using only the training set:

For each feature 
𝑥
x:

𝑥
′
=
𝑥
−
𝜇
train
𝜎
train
+
𝜀
x
′
=
σ
train
	​

+ε
x−μ
train
	​

	​


For binary features (0/1 columns, including team one-hot and pit flags), leave them as-is (no scaling).

Implementation sketch with sklearn:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_df[continuous_features])  # fit on train only

def transform(df):
    X_cont = scaler.transform(df[continuous_features])
    X_bin = df[binary_features].values.astype("float32")
    import numpy as np
    X = np.concatenate([X_cont, X_bin], axis=1)
    y = df[label_col].values.astype("float32")
    return X, y

X_train, y_train = transform(train_df)
X_val, y_val     = transform(val_df)
X_test, y_test   = transform(test_df)


The input dimensionality for the neural net is:

input_dim = X_train.shape[1]

3. Baseline Neural Network Architecture (MLP)

We use a simple feed-forward fully connected network suitable for tabular binary classification.

3.1. Architecture

Input layer: size = input_dim

Hidden Layer 1:

Linear(input_dim → 64)

BatchNorm1d(64)

ReLU

Dropout(p=0.2)

Hidden Layer 2:

Linear(64 → 32)

BatchNorm1d(32)

ReLU

Dropout(p=0.2)

Output Layer:

Linear(32 → 1)

No activation here; we’ll use BCEWithLogitsLoss, which applies a sigmoid internally.

So the model in PyTorch:

import torch
import torch.nn as nn

class F1WinMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (batch,)


This is intentionally small and straightforward, but strong enough as a baseline for structured data.

4. Loss, Class Imbalance, and Training Configuration
4.1. Handling Class Imbalance

Each race has ~1 winner out of ~20 drivers → strong class imbalance.

Use BCEWithLogitsLoss with a positive class weight.

Let:

𝑁
pos
N
pos
	​

 = number of rows where is_winner = 1 in the training set

𝑁
neg
N
neg
	​

 = number of rows where is_winner = 0 in the training set

Set:

pos_weight_value = N_neg / N_pos
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


This up-weights mistakes on the winner class so the model doesn’t just predict 0 for everyone.

4.2. Optimizer and Learning Rate

Optimizer: Adam

Learning rate: 1e-3 (0.001)

Optional: weight decay (e.g., 1e-4) for slight regularization.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

4.3. Batch Size and Epochs

Batch size: 32 (or 16 if dataset is very small; 32 is a good default).

Number of epochs: up to 100, with early stopping based on validation loss or validation F1.

Simple loop idea:

Track best validation F1 (or lowest validation loss).

If no improvement for, say, 10 epochs, stop training (early stopping).

4.4. Training Loop Sketch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).float().to(device)
X_val_t   = torch.from_numpy(X_val).float().to(device)
y_val_t   = torch.from_numpy(y_val).float().to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True
)

model = F1WinMLP(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1, 101):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    # After each epoch, evaluate on validation set to track metrics / early stopping
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits)
        # compute val metrics (e.g., F1) here


You can then implement an early-stopping mechanism based on validation F1 or validation loss.

5. Evaluation Metrics

Because of the strong imbalance, accuracy alone is not enough. Report multiple metrics on the test set:

Overall accuracy

The fraction of all driver–race rows classified correctly.

Precision, recall, and F1-score for the positive class (is_winner = 1)

Positive = “driver wins race”.

This tells you how good the model is at identifying winners specifically.

ROC–AUC

Measures the tradeoff between true positive rate and false positive rate over different thresholds.

PR–AUC (Precision–Recall AUC) for the positive class

More informative than ROC–AUC for highly imbalanced datasets.

“Per-race winner accuracy” (optional but very F1-relevant)

For each race:

Take all drivers in that race,

Choose the driver with the highest predicted probability 
𝑝
(
win
)
p(win),

Check if that driver is actually the winner.

Report the percentage of races where the model picked the correct winner.

This is a very intuitive metric for this application.

Implementation skeleton for predictions:

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

model.eval()
with torch.no_grad():
    test_logits = model(torch.from_numpy(X_test).float().to(device))
    test_probs = torch.sigmoid(test_logits).cpu().numpy()

# default 0.5 threshold for class labels
y_pred = (test_probs >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, pos_label=1, average="binary"
)
roc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)


For per-race winner accuracy, group by race (e.g., round or race_slug), select the driver with highest probability per race, and check whether that driver is the one with is_winner = 1.
