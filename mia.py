"""
Membership inference in PyTorch: does a trained model leak who was in its training data?

Train a small neural network, then act as an attacker who only sees the model's outputs
and tries to guess whether a given example was in the training set. Run it twice: once
with a model that overfits, once with a regularised model, and compare how much each leaks.

Attack: loss-threshold attack (Yeom et al., 2018). Training examples usually get a lower
loss than unseen ones, so "low loss" becomes a guess of "member".

Dataset: scikit-learn's handwritten digits (1,797 8x8 images, bundled offline).
Runs on a laptop CPU in under a minute.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- 1. Data
# Split into two equal halves: MEMBERS (used for training) and NON-MEMBERS (never seen).
# The attacker's job is to tell these two groups apart.
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)   # pixel values 0-16 -> 0-1
y = torch.tensor(y, dtype=torch.long)
idx = torch.randperm(len(X))
half = len(X) // 2
mem_idx, non_idx = idx[:half], idx[half:2 * half]
X_mem, y_mem = X[mem_idx], y[mem_idx]
X_non, y_non = X[non_idx], y[non_idx]

# ---------------------------------------------------------------- 2. Model
class MLP(nn.Module):
    def __init__(self, hidden=256, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 10)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.out(x)          # raw scores (logits) for the 10 digits


def train(model, epochs, weight_decay, n_train=None):
    """Standard PyTorch training loop on the MEMBER set only."""
    Xt, yt = X_mem[:n_train], y_mem[:n_train]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):                  # mini-batches of 32
            b = perm[i:i + 32]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()                              # backpropagation
            opt.step()
    return Xt, yt


@torch.no_grad()
def per_example_loss(model, Xs, ys):
    model.eval()
    return F.cross_entropy(model(Xs), ys, reduction="none").numpy()


@torch.no_grad()
def accuracy(model, Xs, ys):
    model.eval()
    return (model(Xs).argmax(1) == ys).float().mean().item()

# ---------------------------------------------------------------- 3. Attack
def attack(model, Xt, yt):
    """Loss-threshold membership inference.
    Score = -loss (lower loss => more likely a member). AUC 0.5 = attacker is guessing."""
    n = len(Xt)
    loss_in = per_example_loss(model, Xt, yt)
    loss_out = per_example_loss(model, X_non[:n], y_non[:n])   # same number of non-members
    labels = np.r_[np.ones(n), np.zeros(n)]
    scores = -np.r_[loss_in, loss_out]
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    # best single threshold: balanced attack accuracy
    adv = np.max(tpr - fpr)                  # membership "advantage" (Yeom et al.)
    return auc, adv, fpr, tpr, loss_in, loss_out

# ---------------------------------------------------------------- 4. Experiments
configs = {
    # small training set, long training, no regularisation -> memorises
    "overfit":     dict(model=MLP(dropout=0.0), epochs=300, weight_decay=0.0, n_train=200),
    # same data, dropout + weight decay + fewer epochs -> generalises better
    "regularised": dict(model=MLP(dropout=0.5), epochs=40,  weight_decay=1e-3, n_train=200),
}

results, curves = {}, {}
for name, c in configs.items():
    Xt, yt = train(c["model"], c["epochs"], c["weight_decay"], c["n_train"])
    auc, adv, fpr, tpr, li, lo = attack(c["model"], Xt, yt)
    results[name] = dict(
        train_acc=round(accuracy(c["model"], Xt, yt), 3),
        test_acc=round(accuracy(c["model"], X_non, y_non), 3),
        attack_auc=round(auc, 3),
        attack_advantage=round(adv, 3),
        mean_loss_members=round(float(li.mean()), 4),
        mean_loss_nonmembers=round(float(lo.mean()), 4),
    )
    curves[name] = (fpr, tpr, li, lo)

print(json.dumps(results, indent=2))
json.dump(results, open("results.json", "w"), indent=2)

# ---------------------------------------------------------------- 5. Figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for name, (fpr, tpr, li, lo) in curves.items():
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['attack_auc']})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Attack ROC")
ax[0].legend()
for a, name in zip(ax[1:], curves):
    _, _, li, lo = curves[name]
    bins = np.linspace(0, max(li.max(), lo.max()), 40)
    a.hist(li, bins, alpha=.6, label="members (trained on)")
    a.hist(lo, bins, alpha=.6, label="non-members (unseen)")
    a.set(xlabel="per-example loss", title=f"{name}: loss distributions", yscale="log")
    a.legend()
plt.tight_layout()
plt.savefig("mia_results.png", dpi=130)
print("saved mia_results.png")
