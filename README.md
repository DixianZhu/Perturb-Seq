# Perturb-Seq
General Python Repo for Perturb-Seq

## For Akana2024 Dataset:
- Cells with perturb-genes, ['B2M', 'CD274', 'KLF4', 'NTC', 'TRPS1'], are hold out for testing dataset, the 10% of remaining cells are also uniformly randomly selected to testing dataset. The rest of cells are selected for training/validation. Relevant data: <code>/oak/stanford/groups/ljerby/SharedResources/Akana2024/Data/A375_labels_and_extras.npz</code>. Relevant preprocessing code: <code>preprocess_Akana2024_data.ipynb</code>.
- p-values <code>neg.p.value</code> and <code>pos.p.value</code> are the predictive targets for regression model, which also can be evaluated in classification mode. We take p-value <=0.05, 0.05<p-value <0.95, 0.95<=p-value as 3 classes for the 2 predictive targets. It would be also possible to generate 9 classes independently by the 3 classes from 2 targets. But they are equivalent and thereby we keep 6 classes for prediction for the sake of simplicity. 
- Current we have some regression and classification baseline models including: <code>MAE, MSE, Huber, RNC, ConR, FAR</code> (for regression) and <code>CE, CS, WW, SCE, GCE, LDR</code> (for classification).

### How to run:
For regression:

<code> python3 main.py --lr=1e-2 --decay=1e-4 --loss=MAE  --dataset=Akana <code>

<code> python3 main.py --lr=1e-2 --decay=1e-4 --loss=FAR  --dataset=Akana <code>

