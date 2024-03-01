# Perturb-Seq
General Python Repo for Perturb-Seq

## For Akana2024 Dataset:
- Cells with perturb-genes, ['B2M', 'CD274', 'KLF4', 'NTC', 'TRPS1'], are hold out for testing dataset, the 10% of remaining cells are also uniformly randomly selected to testing dataset. The rest of cells are selected for training/validation. Relevant processed data: <code>/oak/stanford/groups/ljerby/SharedResources/Akana2024/Data/A375_labels_and_extras.npz</code>. Relevant preprocessing code: <code>preprocess_Akana2024_data.ipynb</code>.
- p-values <code>neg.p.value</code> and <code>pos.p.value</code> are the predictive targets for regression model, which also can be evaluated in classification mode. We take p-value <=0.05, 0.05<p-value <0.95, 0.95<=p-value as 3 classes for the 2 predictive targets. It would be also possible to generate 9 classes independently by the 3 classes from 2 targets. But they are equivalent and thereby we keep 6 classes for prediction for the sake of simplicity. 
- Current we have some regression and classification baseline models including: <code>MAE, MSE, Huber, RNC, ConR, FAR</code> (for regression) and <code>CE, CS, WW, SCE, GCE, LDR</code> (for classification). <code>CS, WW</code> are two variants for support vector machine (SVM) under multi-class classification case.

### How to run:
Run the automatic 5-fold-validation code.

For regression:

<code> python3 main.py --lr=1e-2 --decay=1e-4 --loss=MAE  --dataset=Akana </code>

<code> python3 main.py --lr=1e-2 --decay=1e-4 --loss=FAR  --dataset=Akana </code>

For classification:

<code> python3 class_main.py --lr=1e-2 --decay=1e-4 --loss=CE  --dataset=Akana </code>

<code> python3 class_main.py --lr=1e-2 --decay=1e-4 --loss=CS  --dataset=Akana </code>

<code> python3 class_main.py --lr=1e-2 --decay=1e-4 --loss=LDR  --dataset=Akana </code>


The validation code automatically run 5-fold-cross-validation and print out evaluation performance for training and validation. The backbone model is a 7-layer Feed Forward Neural Network.
### Initial findings:
Roughly scan through some results on the 1-st fold with raw eyes, for the validation performance: 
- The averaged accuracy for classification performance can be achieved ~ 0.65; averaged AUROC can be achieved ~ 0.7.
- The averaged MAE for regression performance can be achieved ~ 0.28; averaged Pearson correlation can be achieved ~ 0.33.

More detailed analysis and future research will be required for publishing a paper.


:page_with_curl: Citation
---------
If you find FAR and/or LDR are useful in your work, please cite the following paper:
```
@misc{zhu2024function,
      title={Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data}, 
      author={Dixian Zhu and Livnat Jerby-Arnon},
      year={2024},
      eprint={2402.06104},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@inproceedings{zhu2023label,
	title={Label Distributionally Robust Losses for Multi-class Classification: Consistency, Robustness and Adaptivity},
	author={Zhu, Dixian and Ying, Yiming and Yang, Tianbao},
	booktitle={Proceedings of the 40th International Conference on Machine Learning},
	year={2023}
	}  
```

