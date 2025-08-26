import torch
from torch_geometric.data import Data
from sklearn.model_selection import KFold, train_test_split
from rdkit import Chem
import numpy as np

# Map RDKit hybridization types to integers
hybrid_map = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
}

def atom_features(atom):
    """Return a 10-dim feature vector for an atom."""
    return [
        atom.GetAtomicNum(),                     # Z
        atom.GetTotalDegree(),                   # number of bonded neighbors
        atom.GetFormalCharge(),                  # formal charge
        atom.GetTotalNumHs(),                    # total number of Hs
        atom.GetImplicitValence(),               # implicit valence
        int(atom.GetIsAromatic()),               # aromaticity
        hybrid_map.get(atom.GetHybridization(), 0),  # hybridization as int
        atom.GetMass() * 0.01,                   # scaled mass
        0,                                       # placeholder
        0                                        # placeholder
    ]

def mol_to_graph(smiles, y):
    """Convert a SMILES string to a PyTorch Geometric Data object with rich atom features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)

    # Bonds (edges)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected
        edge_attr.append([bond.GetBondTypeAsDouble()])
        edge_attr.append([bond.GetBondTypeAsDouble()])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Labels
    if isinstance(y, (list, np.ndarray)):
        y_tensor = torch.tensor(y, dtype=torch.float)
    else:
        y_tensor = torch.tensor([y], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor, smiles=smiles)


class SurfProDB:
    def __init__(self, df, propnames):
        self.df = df
        self.propnames = propnames
        self.smiles = df["smiles"].tolist() if "smiles" in df.columns else []
        self.train = []
        self.valid = []
        self.test = []
        self._train_full = None  # for retraining

    def _to_graph_dataset(self, df_subset):
        """Convert DataFrame subset into list of Data objects."""
        dataset = []
        for _, row in df_subset.iterrows():
            smiles = row["smiles"]
            y = row[self.propnames].values
            if len(y) == 1:
                y = y[0]
            data = mol_to_graph(smiles, y)
            if data is not None:
                dataset.append(data)
        return dataset

    def split(self, method="KFOLD", n_splits=10, val_size=0.1, seed=42):
        """Split dataset into train/val/test sets using KFOLD or random split"""
        X = self.df.index.values
        y = self.df[self.propnames].values

        if method.upper() == "KFOLD":
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in kf.split(X, y):
                train_data = self.df.iloc[train_idx]
                test_data = self.df.iloc[test_idx]

                # validation split
                train_data, val_data = train_test_split(
                    train_data,
                    test_size=val_size,
                    random_state=seed
                )

                self.train.append(self._to_graph_dataset(train_data))
                self.valid.append(self._to_graph_dataset(val_data))
                self.test.append(self._to_graph_dataset(test_data))

            # save full dataset for retraining
            self._train_full = self._to_graph_dataset(self.df)

        else:
            raise ValueError(f"Unsupported split method: {method}")
