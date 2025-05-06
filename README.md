# Airfoil_Recon_GNN
## Reconstruction of flowfield around airfoils using surface pressure data

### 1. Download training dataset
```
wget "https://zenodo.org/records/14629208/files/train_dataset.zip"
```

### 2. Install anaconda environment
```
conda env create -f pyg_env.yaml
```

### 3. Model training
```
qsub submit.sh
```

### 4. Test & Plot
```
python test.py
```
