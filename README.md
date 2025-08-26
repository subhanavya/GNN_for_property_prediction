
pip install -r requirements.txt

Prepare your data
data/hlb.csv eg: smiles, hlb


Train

python train_hlb.py `
--csv_path data/surfpro.csv `
--out_dir results  `
--params_yaml params.yaml `
--n_splits 10

Command line for prediction 

python predict_hlb.py `                                                                                           
--input_csv data/test_hlb.csv `
--params_yaml params.yaml `
--out_dir results/ `
--ckpt_path results/models/best.ckpt `
--target_col hlb

