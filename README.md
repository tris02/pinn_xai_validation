# PINN-XAI Validation

## Overview
This repository provides a PINN-XAI machine learning validation script using blended physics-informed neural networks (PINN) with causal constraints, and explanatory AI (XAI) implementation. It includes data preprocessing, feature engineering, PINN-XAI pipeline, and robustness check.

## Features
- Uses **PINN** to validate SEM modeling for NFT pricing.
- Implements **causal constraints** and **explainability techniques (permutation importance, SHAP, PDP, ICE)**.
- Performs **robustness checks** using **PCA-based factor scores** and **constrained-unconstrained NN**.
- Supports **model interpretability**.

## Installation
To set up the environment, install the required dependencies using:

```bash
pip install numpy pandas torch scikit-learn shap matplotlib
```

## Usage
Clone this repository and navigate to the project folder:

```bash
git clone https://github.com/tris02/pinn_xai_validation.git
cd pinn_xai_validation
```

Run the validation script:

```bash
python validation_git.py
```

## File Structure
```
📂 pinn_xai_validation
├── validation_git.py    # Main script implementing PINN and XAI methods
├── README.md            # Documentation
├── .gitignore           # Ignoring unnecessary files (cache, logs, etc.)
├── requirements.txt     # List of dependencies
```

## Explanation of Key Components
### 1️⃣ Physics-Informed Neural Network (PINN)
The model incorporates factor relationships from SEM model by enforcing constraints on partial derivatives, ensuring interpretability and generalizability.

### 2️⃣ Explainable AI (XAI) Techniques
- **Permutation Importance**: Measures feature significance by shuffling inputs.
- **SHAP Values**: Decomposes model predictions into contributions from each input variable.
- **Partial Dependence & ICE**: Visualizes how a feature impacts predictions.

### 3️⃣ Robustness Checks
- PCA-derived factor scores replace manually computed ones to validate the model’s robustness.
- A comparison is made between constrained and unconstrained models to assess the effect of causal constraints.


## Contributing
If you’d like to contribute, fork the repository and submit a pull request with improvements.

## License
This project is licensed under the MIT License.

## Author
Tristan Lim - tris02@gmail.com

