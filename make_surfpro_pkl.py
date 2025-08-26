
import pandas as pd
import pickle
import os
from src.dataloader import SurfProDB

def make_surfpro_pkl(csv_path, save_path, prop_col='hlb', n_splits=5, scale=True, test_size=0.1):
    df = pd.read_csv(csv_path)
    assert prop_col in df.columns, f'Missing column {prop_col} in {csv_path}'
    db = SurfProDB(df, propnames=[prop_col], scale=scale, test_size=test_size)
    db.split('KFOLD', n_splits=n_splits)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(db, f)
    print(f'Saved SurfPro pickle to {save_path}')

if __name__ == '__main__':
    make_surfpro_pkl('data/hlb.csv', 'data/hlb_prediction/surfpro.pkl')
