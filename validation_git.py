import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

###############################################################################

# Load CSV data
file_path = "path/to/your/dataset.csv"
df = pd.read_csv(file_path)

print("Data columns:", df.columns.tolist())
print(df['phase'].value_counts())

###############################################################################


# Define which columns map to which conceptual factor
sc_vars = [
    'size', 'Entropy', 'len_statics', 'len_dynamics'
]

tu_vars = [
    'ASM', 'homogeneity', 'hue'
    #, 'RoT1',
]

cvc_vars = [
    'colofulness', 'saturation'
    #, 'RoT1', 'RoT2'
]

# Drop rows with missing data (if necessary)
df.dropna(subset=sc_vars + tu_vars + cvc_vars + ['price_boxcox'], inplace=True)

# Standardize each subset
scaler_sc = StandardScaler()
scaler_tu = StandardScaler()
scaler_cvc = StandardScaler()

df[sc_vars] = scaler_sc.fit_transform(df[sc_vars])
df[tu_vars] = scaler_tu.fit_transform(df[tu_vars])
df[cvc_vars] = scaler_cvc.fit_transform(df[cvc_vars])

# Create composite (factor) scores: SC, TU, CVC
df['SC_score'] = df[sc_vars].mean(axis=1)
df['TU_score'] = df[tu_vars].mean(axis=1)
df['CVC_score'] = df[cvc_vars].mean(axis=1)

# Create interaction terms
df['SCxTU'] = df['SC_score'] * df['TU_score']
df['SCxCVC'] = df['SC_score'] * df['CVC_score']

# Our final feature set: SC, TU, CVC, SC×TU, SC×CVC
feature_cols = ['SC_score', 'TU_score', 'CVC_score', 'SCxTU', 'SCxCVC']
X = df[feature_cols].values
y = df['price_boxcox'].values

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


###############################################################################


# Separate train and test data based on 'phase'
df_train = df[df['phase'] == 'train']
df_test = df[df['phase'] == 'test']

# Build numpy arrays for features (X) and target (y)
X_train = df_train[feature_cols].values
y_train = df_train['price_boxcox'].values

X_test = df_test[feature_cols].values
y_test = df_test['price_boxcox'].values

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"Train size: {X_train_t.size(0)}, Test size: {X_test_t.size(0)}")



###############################################################################
# 1.0 PIMM: Physics-Informed Neural Network Model with Causal Constraints
###############################################################################


class PINNWithCausalConstraints(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        z = self.relu(self.layer1(x))
        z = self.relu(self.layer2(z))
        out = self.layer3(z)
        return out


###############################################################################


def causal_constraint_loss(model, X_batch, y_batch, penalty_weight=10.0):
    """
    X_batch shape: [batch_size, 5] => SC, TU, CVC, SCxTU, SCxCVC
    y_batch shape: [batch_size, 1]
    """
    # Forward pass => predictions
    y_pred = model(X_batch)
    # Base MSE
    mse = torch.mean((y_pred - y_batch)**2)

    # Calculate partial derivatives wrt relevant dimensions:
    # 0 -> SC, 3 -> SCxTU, 4 -> SCxCVC
    grad = torch.autograd.grad(
        outputs=y_pred, 
        inputs=X_batch, 
        grad_outputs=torch.ones_like(y_pred),
        retain_graph=True,
        create_graph=True
    )[0]  # shape [batch_size, 5]

    pd_sc    = grad[:, 0]  # partial derivative wrt SC
    pd_sc_tu = grad[:, 3]  # partial derivative wrt SCxTU
    pd_sc_cvc= grad[:, 4]  # partial derivative wrt SCxCVC

    penalty = 0.0

    # If pd_sc < 0 => violation
    violation_sc = torch.relu(-pd_sc)
    penalty += penalty_weight * torch.mean(violation_sc)

    # If pd_sc_tu > 0 => violation
    violation_sc_tu = torch.relu(pd_sc_tu)
    penalty += penalty_weight * torch.mean(violation_sc_tu)

    # If pd_sc_cvc > 0 => violation
    violation_sc_cvc = torch.relu(pd_sc_cvc)
    penalty += penalty_weight * torch.mean(violation_sc_cvc)

    return mse + penalty


###############################################################################


model = PINNWithCausalConstraints(hidden_size=8)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 300
penalty_w = 1.0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    loss = causal_constraint_loss(model, X_train_t, y_train_t, penalty_weight=10.0)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t)
    y_test_pred  = model(X_test_t)

train_mse = torch.mean((y_train_pred - y_train_t)**2).item()
test_mse  = torch.mean((y_test_pred  - y_test_t)**2).item()

print(f"Final Train MSE: {train_mse:.4f}")
print(f"Final Test MSE:  {test_mse:.4f}")


###############################################################################


model.eval()
y_pred_full = model(X_train_t)

# Recompute partial derivatives wrt SC, SCxTU, SCxCVC on the entire train set
grad_full = torch.autograd.grad(
    outputs=y_pred_full,
    inputs=X_train_t,
    grad_outputs=torch.ones_like(y_pred_full),
    retain_graph=True,
    create_graph=True
)[0]

pd_sc_full    = grad_full[:, 0]
pd_sc_tu_full = grad_full[:, 3]
pd_sc_cvc_full= grad_full[:, 4]

print("Avg partial derivative wrt SC (should be >=0):",  pd_sc_full.mean().item())
print("Avg partial derivative wrt SCxTU (should be <=0):",pd_sc_tu_full.mean().item())
print("Avg partial derivative wrt SCxCVC (should be <=0):",pd_sc_cvc_full.mean().item())

# You could also check min() to see if any sample severely violates the constraint


###############################################################################


print("Min partial derivative wrt SC:", pd_sc_full.min().item())
print("Max partial derivative wrt SCxTU:", pd_sc_tu_full.max().item())
print("Max partial derivative wrt SCxCVC:", pd_sc_cvc_full.max().item())



###############################################################################
# 2. Post-Hoc Explainable AI (XAI) Validation
###############################################################################

##############################
# 2.1 Permutation Importance #
##############################

import random

def predict_np(model, X_np):
    """
    Utility function to get model predictions as a NumPy array.
    X_np: NumPy array of shape [n_samples, n_features].
    """
    X_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        preds_t = model(X_t)
    return preds_t.numpy().flatten()

def mse_score(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def permutation_importance(model, X_np, y_np, baseline_score, n_repeats=5, random_state=42):
    """
    Manual permutation importance for a PyTorch model.
    - model: trained PyTorch model.
    - X_np, y_np: NumPy arrays for the dataset.
    - baseline_score: baseline MSE on (X_np, y_np).
    - n_repeats: how many times to shuffle each column.
    Returns:
       importances: array of shape [n_features] with average MSE increase from shuffling each feature.
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_np.shape
    importances = np.zeros(n_features)

    for col_idx in range(n_features):
        score_diffs = []
        for _ in range(n_repeats):
            # shuffle column col_idx
            X_shuffled = X_np.copy()
            idx_permutation = rng.permutation(n_samples)
            X_shuffled[:, col_idx] = X_shuffled[idx_permutation, col_idx]

            y_shuffled_pred = predict_np(model, X_shuffled)
            score_shuffled = mse_score(y_np, y_shuffled_pred)
            score_diffs.append(score_shuffled - baseline_score)

        importances[col_idx] = np.mean(score_diffs)

    return importances

# Prepare data for permutation importance
X_train_np = X_train  # Already a NumPy array
y_train_np = y_train

# Baseline MSE on the training set
baseline_train_pred = predict_np(model, X_train_np)
baseline_train_mse = mse_score(y_train_np, baseline_train_pred)
print("\n=== Permutation Importance (Train) ===")
print("Baseline Train MSE:", baseline_train_mse)

# Calculate feature importances
perm_importances = permutation_importance(model, X_train_np, y_train_np, baseline_train_mse, n_repeats=5)
for i, col in enumerate(feature_cols):
    print(f"Feature: {col}, Importance (MSE increase): {perm_importances[i]:.5f}")

# You might also do it on the test set:
X_test_np = X_test
y_test_np = y_test
baseline_test_pred = predict_np(model, X_test_np)
baseline_test_mse = mse_score(y_test_np, baseline_test_pred)
print("\n=== Permutation Importance (Test) ===")
print("Baseline Test MSE:", baseline_test_mse)

perm_importances_test = permutation_importance(model, X_test_np, y_test_np, baseline_test_mse, n_repeats=5)
for i, col in enumerate(feature_cols):
    print(f"Feature: {col}, Importance (MSE increase): {perm_importances_test[i]:.5f}")


####################################
# 2.2 SHAP (Global & Local Analysis)
####################################

import shap

# We'll use a wrapper so shap can call model.predict
def model_predict_for_shap(data_np):
    """
    data_np: shape [n, 5]
    Returns: shape [n]
    """
    return predict_np(model, data_np)

# We'll take a small background sample from the training set for KernelExplainer
background_size = min(200, len(X_train_np))
X_background = X_train_np[:background_size, :]

explainer = shap.KernelExplainer(model_predict_for_shap, X_background)

# For demonstration, let's compute SHAP values on a small sample of test data
test_sample_size = min(50, len(X_test_np))
X_test_sample = X_test_np[:test_sample_size, :]

# nsamples: how many evaluations to approximate shap values
shap_values = explainer.shap_values(X_test_sample, nsamples=100)

# shap_values is shape [n, features], in this case features=5
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols)

# Because we have interactions (SCxTU, SCxCVC), you might want to check shap interaction values
# But KernelExplainer can be slow for large data. You can also try shap.TreeExplainer if you use XGBoost, or shap.GradientExplainer for a suitable PyTorch approach.


############################################
# Local SHAP Interpretation (Example NFTs) #
############################################

# Let's pick a few test rows to see local SHAP.
indices_to_explain = [0, 1, 2]  # first three test examples
for idx in indices_to_explain:
    shap_values_single = explainer.shap_values(X_test_np[idx:idx+1,:], nsamples=100)
    print(f"\nLocal SHAP for test sample index {idx} (feature contributions):")
    for col_i, col_name in enumerate(feature_cols):
        print(f"  {col_name}: {shap_values_single[0][col_i]:.4f}")
    print("  Model prediction:", model_predict_for_shap(X_test_np[idx:idx+1,:]))
    print("  True price_boxcox:", y_test_np[idx])


######################################################
# 2.3 Partial Dependence & ICE (via sklearn.inspection)
######################################################

from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt  # Ensure you've imported matplotlib

# NEW: We need these for the 'fitted' check
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

# We can wrap the PyTorch model in a scikit-learn-style "predict" function
# so that PartialDependenceDisplay can be used.

class PyTorchRegressorSK(BaseEstimator, RegressorMixin):
    """
    Minimal sklearn-style wrapper for a trained PyTorch model.
    """
    def __init__(self, model):
        self.model = model
        self._fitted = False  # Will set to True in 'fit()'
    
    def fit(self, X, y):
        # We do not retrain the PyTorch model here.
        # Just mark as 'fitted' and record input feature info for scikit-learn checks.
        self._fitted = True
        self.n_features_in_ = X.shape[1]
        return self
    
    def predict(self, X):
        # Raise if not fitted
        if not self._fitted:
            raise NotFittedError(
                "This PyTorchRegressorSK instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        # Optional shape check
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Number of features in X ({X.shape[1]}) does not "
                             f"match the training set ({self.n_features_in_}).")
        
        # Use your PyTorch model to predict
        return predict_np(self.model, X)

# Wrap your already-trained model
pytorch_model_wrapper = PyTorchRegressorSK(model)

# Call 'fit' just to set _fitted = True (no training happens)
pytorch_model_wrapper.fit(X_train, y_train)  # NEW: ensures partial dependence won't complain

# We'll generate partial dependence for SC_score (index=0) as an example
# NOTE: PartialDependenceDisplay requires scikit-learn >= 1.0
# We'll do a single feature first: SC_score => 0
features_to_plot = [0]  # SC_score is col 0
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    estimator=pytorch_model_wrapper,
    X=X_test_np, 
    features=features_to_plot,
    feature_names=feature_cols,
    ax=ax,
)
plt.title("Partial Dependence Plot for SC_score -> price_boxcox")
plt.show()

# For a 2D partial dependence (interaction), e.g., SC_score and TU_score => [0, 1]
# But note that partial_dependence() with 2 features can be slower to compute.
features_2d = [(0, 1)]  # (SC_score, TU_score)
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    estimator=pytorch_model_wrapper,
    X=X_test_np,
    features=features_2d,
    feature_names=feature_cols,
    ax=ax
)
plt.title("2D Partial Dependence Plot (SC_score x TU_score)")
plt.show()


# Suppose SC_score is col 0 and CVC_score is col 2
features_2d_cvc = [(0, 2)]  # SC_score, CVC_score
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    estimator=pytorch_model_wrapper,
    X=X_test_np,
    features=features_2d_cvc,
    feature_names=feature_cols,
    ax=ax
)
plt.title("2D Partial Dependence Plot (SC_score x CVC_score)")
plt.show()

###########################################
# 2.4 ICE Curves for SC_score
###########################################


fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(
    estimator=pytorch_model_wrapper,
    X=X_test_np,
    features=[0],  # SC_score
    feature_names=feature_cols,
    kind='individual',   # <---- ICE instead of average
    subsample=100,       # limit to 100 points if dataset is large
    n_jobs=1,            # control parallel jobs if needed
    ax=ax
)
plt.title("ICE Curves for SC_score -> price_boxcox (subset of test data)")
plt.show()


print("\n=== Done with XAI steps ===")



###############################################################################
# Robustness Check 1: PCA Factor Scores for SC, TU, CVC
###############################################################################
from sklearn.decomposition import PCA
import copy

# --- 1A. PCA Instead of Averages ---
# We'll create new columns: SC_score_pca, TU_score_pca, CVC_score_pca

# Copy the original df so we don't overwrite main pipeline
df_pca = df.copy()

# We know our original sc_vars, tu_vars, cvc_vars
sc_vars = ['size', 'Entropy', 'len_statics', 'len_dynamics']
tu_vars = ['ASM', 'homogeneity', 'RoT1', 'hue']
cvc_vars = ['colofulness', 'saturation', 'RoT1', 'RoT2']

# 1. SC PCA
pca_sc = PCA(n_components=1)
sc_pca_scores = pca_sc.fit_transform(df_pca[sc_vars])
df_pca['SC_score_pca'] = sc_pca_scores[:, 0]

# 2. TU PCA
pca_tu = PCA(n_components=1)
tu_pca_scores = pca_tu.fit_transform(df_pca[tu_vars])
df_pca['TU_score_pca'] = tu_pca_scores[:, 0]

# 3. CVC PCA
pca_cvc = PCA(n_components=1)
cvc_pca_scores = pca_cvc.fit_transform(df_pca[cvc_vars])
df_pca['CVC_score_pca'] = cvc_pca_scores[:, 0]

# 1B. Create new interaction terms using PCA-based factor scores
df_pca['SCxTU_pca'] = df_pca['SC_score_pca'] * df_pca['TU_score_pca']
df_pca['SCxCVC_pca'] = df_pca['SC_score_pca'] * df_pca['CVC_score_pca']

# 1C. Set up new feature columns
feature_cols_pca = ['SC_score_pca', 'TU_score_pca', 'CVC_score_pca',
                    'SCxTU_pca', 'SCxCVC_pca']

# 1D. Build train/test splits with these PCA features
df_train_pca = df_pca[df_pca['phase'] == 'train']
df_test_pca  = df_pca[df_pca['phase'] == 'test']

X_train_pca = df_train_pca[feature_cols_pca].values
y_train_pca = df_train_pca['price_boxcox'].values

X_test_pca  = df_test_pca[feature_cols_pca].values
y_test_pca  = df_test_pca['price_boxcox'].values

X_train_t_pca = torch.tensor(X_train_pca, dtype=torch.float32, requires_grad=True)
y_train_t_pca = torch.tensor(y_train_pca, dtype=torch.float32).view(-1, 1)

X_test_t_pca = torch.tensor(X_test_pca, dtype=torch.float32, requires_grad=True)
y_test_t_pca = torch.tensor(y_test_pca, dtype=torch.float32).view(-1, 1)

# 1E. Retrain the same model architecture, but with PCA-based inputs
model_pca = PINNWithCausalConstraints(hidden_size=8)  # from your existing class
optimizer_pca = optim.Adam(model_pca.parameters(), lr=0.01)
num_epochs_pca = 300

def causal_constraint_loss_pca(model, X_batch, y_batch, penalty_weight=10.0):
    """
    Same logic but note the input now has columns:
      0->SC_score_pca, 1->TU_score_pca, 2->CVC_score_pca,
      3->SCxTU_pca, 4->SCxCVC_pca
    Indices for partial derivatives remain the same.
    """
    y_pred = model(X_batch)
    mse = torch.mean((y_pred - y_batch)**2)

    grad = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_batch,
        grad_outputs=torch.ones_like(y_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    pd_sc    = grad[:, 0]  # SC_score_pca
    pd_sc_tu = grad[:, 3]  # SCxTU_pca
    pd_sc_cvc= grad[:, 4]  # SCxCVC_pca

    penalty = 0.0
    violation_sc = torch.relu(-pd_sc)
    penalty += penalty_weight * torch.mean(violation_sc)
    violation_sc_tu = torch.relu(pd_sc_tu)
    penalty += penalty_weight * torch.mean(violation_sc_tu)
    violation_sc_cvc = torch.relu(pd_sc_cvc)
    penalty += penalty_weight * torch.mean(violation_sc_cvc)

    return mse + penalty

# Train loop
for epoch in range(num_epochs_pca):
    model_pca.train()
    optimizer_pca.zero_grad()
    loss_pca = causal_constraint_loss_pca(model_pca, X_train_t_pca, y_train_t_pca, penalty_weight=10.0)
    loss_pca.backward()
    optimizer_pca.step()

    if (epoch + 1) % 50 == 0:
        print(f"[PCA] Epoch {epoch+1}/{num_epochs_pca} | Loss: {loss_pca.item():.4f}")

# Evaluate
model_pca.eval()
with torch.no_grad():
    y_train_pred_pca = model_pca(X_train_t_pca)
    y_test_pred_pca  = model_pca(X_test_t_pca)

train_mse_pca = torch.mean((y_train_pred_pca - y_train_t_pca)**2).item()
test_mse_pca  = torch.mean((y_test_pred_pca  - y_test_t_pca)**2).item()

print(f"[PCA] Final Train MSE: {train_mse_pca:.4f}")
print(f"[PCA] Final Test MSE:  {test_mse_pca:.4f}")

# Quick partial derivative check
y_pred_full_pca = model_pca(X_train_t_pca)
grad_full_pca = torch.autograd.grad(
    outputs=y_pred_full_pca,
    inputs=X_train_t_pca,
    grad_outputs=torch.ones_like(y_pred_full_pca),
    retain_graph=True,
    create_graph=True
)[0]

pd_sc_full_pca    = grad_full_pca[:, 0]
pd_sc_tu_full_pca = grad_full_pca[:, 3]
pd_sc_cvc_full_pca= grad_full_pca[:, 4]

print("[PCA] Avg partial derivative wrt SC (>=0):",  pd_sc_full_pca.mean().item())
print("[PCA] Avg partial derivative wrt SCxTU (<=0):",pd_sc_tu_full_pca.mean().item())
print("[PCA] Avg partial derivative wrt SCxCVC (<=0):",pd_sc_cvc_full_pca.mean().item())



###############################################################################
# Robustness Check 2: Constrained vs. Unconstrained
###############################################################################

# (A) Unconstrained Model => standard MSE loss, no partial-derivative penalties
class UnconstrainedModel(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        z = self.relu(self.layer1(x))
        z = self.relu(self.layer2(z))
        out = self.layer3(z)
        return out

def plain_mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# (B) Train unconstrained with MSE only
unconstrained_model = UnconstrainedModel(hidden_size=8)
optimizer_uncon = optim.Adam(unconstrained_model.parameters(), lr=0.01)
num_epochs_uncon = 300

for epoch in range(num_epochs_uncon):
    unconstrained_model.train()
    optimizer_uncon.zero_grad()
    
    y_pred_uncon = unconstrained_model(X_train_t)   # X_train_t from original factor approach
    loss_uncon = plain_mse_loss(y_pred_uncon, y_train_t)
    
    loss_uncon.backward()
    optimizer_uncon.step()
    
    if (epoch+1) % 50 == 0:
        print(f"[Unconstrained] Epoch {epoch+1}/{num_epochs_uncon} | Loss: {loss_uncon.item():.4f}")

# Evaluate unconstrained
unconstrained_model.eval()
with torch.no_grad():
    y_train_pred_uncon = unconstrained_model(X_train_t)
    y_test_pred_uncon  = unconstrained_model(X_test_t)

train_mse_uncon = torch.mean((y_train_pred_uncon - y_train_t)**2).item()
test_mse_uncon  = torch.mean((y_test_pred_uncon  - y_test_t)**2).item()

print(f"[Unconstrained] Final Train MSE: {train_mse_uncon:.4f}")
print(f"[Unconstrained] Final Test MSE:  {test_mse_uncon:.4f}")

# Check partial derivatives => see how many sign violations appear
y_pred_full_uncon = unconstrained_model(X_train_t)
grad_full_uncon = torch.autograd.grad(
    outputs=y_pred_full_uncon,
    inputs=X_train_t,
    grad_outputs=torch.ones_like(y_pred_full_uncon),
    retain_graph=True,
    create_graph=True
)[0]

pd_sc_full_uncon    = grad_full_uncon[:, 0]
pd_sc_tu_full_uncon = grad_full_uncon[:, 3]
pd_sc_cvc_full_uncon= grad_full_uncon[:, 4]

print("[Unconstrained] Avg partial derivative wrt SC:",  pd_sc_full_uncon.mean().item())
print("[Unconstrained] Avg partial derivative wrt SCxTU:", pd_sc_tu_full_uncon.mean().item())
print("[Unconstrained] Avg partial derivative wrt SCxCVC:", pd_sc_cvc_full_uncon.mean().item())

# You can also count how many samples have pd_sc < 0 or pd_sc_tu > 0
n_sc_negative = (pd_sc_full_uncon < 0).sum().item()
n_sc_tu_positive = (pd_sc_tu_full_uncon > 0).sum().item()
n_sc_cvc_positive= (pd_sc_cvc_full_uncon> 0).sum().item()
print(f"[Unconstrained] #samples with SC<0: {n_sc_negative}, SCxTU>0: {n_sc_tu_positive}, SCxCVC>0: {n_sc_cvc_positive}")
