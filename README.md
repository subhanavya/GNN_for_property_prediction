# HLB Prediction using Graph Neural Networks  


## Objective  
This project focuses on predicting the **Hydrophilic–Lipophilic Balance (HLB)** of surfactants directly from their **SMILES molecular representations**. The goal was to leverage modern **Graph Neural Networks (GNNs)** to enable accurate, data-driven prediction of HLB values, which are crucial in applications such as emulsifier design, detergents, and surfactant formulation.

---

## Approach  
- **Modeling Framework:** Implemented the **AttentiveFP Graph Neural Network** using PyTorch Geometric.  
- **Molecular Representation:** Converted **SMILES strings** into graph-based molecular representations via **RDKit**.  
- **Training Strategy:**  
  - Applied **dropout** for regularization.  
  - Used **early stopping** to prevent overfitting.  
  - Performed **k-fold cross-validation (k=10)** for robust evaluation.  
- **Implementation:** Modular training pipeline with configurable hyperparameters via `params.yaml`.

---

## Results  
- **RMSE achieved:** `0.43` on benchmark surfactant dataset.  
- Demonstrated the potential of GNNs to capture molecular features critical for HLB prediction.  
- Outperformed baseline regression approaches.

---

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/subhanavya/GNN_for_property_prediction.git
cd GNN_for_property_prediction
pip install -r requirements.txt
```

---

## Data Preparation  
Prepare your dataset as a CSV file with at least two columns:  
- `smiles` → Molecular representation  
- `hlb` → Target property  

Example: `data/hlb.csv`

---

## Training  
Run the training script:  
```bash
python train_hlb.py     --csv_path data/surfpro.csv     --out_dir results     --params_yaml params.yaml     --n_splits 10
```

---

## Prediction  
Run the prediction script on unseen data:  
```bash
python predict_hlb.py     --input_csv data/test_hlb.csv     --params_yaml params.yaml     --out_dir results/     --ckpt_path results/models/best.ckpt     --target_col hlb
```
## Output 
results are stored in results/predictions_metrics.json
---

## Project Insights  
- This work demonstrates the **practical utility of GNNs** in cheminformatics, specifically in predicting surfactant HLB values.  
- The model can serve as a foundation for **formulation design pipelines** in pharmaceuticals, personal care, and industrial applications.  
