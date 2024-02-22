import torch 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics import roc_auc_score
from scipy import sparse

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class pair_dataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
    self.X = torch.from_numpy(X.astype(np.float32))
    self.Y = torch.from_numpy(Y.astype(np.float32))
  def __len__(self):
    try:
      L = len(self.X)
    except:
      L = self.X.shape[0]
    return L 
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]

def reg_eval(preds, truths):
# for regression based methods doing regression evaluation
# preds: N x 2
# truths: N x 2
  MAE, RMSE, pearson, spearman = [], [], [], []
  num_targets = truths.shape[1]
  for i in range(num_targets):
    pred, truth = preds[:,i], truths[:,i]
    MAE.append(np.abs(pred-truth).mean())
    RMSE.append(((pred-truth)**2).mean()**0.5)
    pearson.append(np.corrcoef(truth, pred, rowvar=False)[0,1])
    spearman.append(stats.spearmanr(truth, pred).statistic)
  return MAE, RMSE, pearson, spearman


def class_reg_eval(preds, truths): 
# for regression based methods doing classification evaluation
# preds: N x 2
# truths: N x 2
  acc, auc = [], []
  num_targets = truths.shape[1]
  for i in range(num_targets):
    pred, truth = np.expand_dims(preds[:,i], axis=1), np.expand_dims(truths[:,i], axis=1)
    pred_auc = np.concatenate([-pred, -np.abs(pred-0.5) , pred], axis=1)
    truth = np.concatenate([(truth<=0.05).astype(float), (truth>0.05).astype(float)*(truth<0.95).astype(float) ,(truth>=0.95).astype(float)], axis=1)
    auc.append(roc_auc_score(truth, pred_auc, average=None))
    pred_acc = np.concatenate([(pred<=0.05).astype(float), (pred>0.05).astype(float)*(pred<0.95).astype(float) ,(pred>=0.95).astype(float)], axis=1)
    accuracy = (pred_acc == truth).astype(float).mean(axis=0)
    acc.append(accuracy)
  return acc, auc


def class_eval(preds, truths, num_class=3): 
# for classification based methods doing classification evaluation
# preds: N x 6
# truths: N x 6
  acc, auc = [], []
  num_targets = int(truths.shape[1]/num_class)
  one_hot = np.eye(num_class)
  for i in range(num_targets):
    s, e = i*num_class, (i+1)*num_class
    pred, truth = preds[:,s:e], truths[:,s:e]
    auc.append(roc_auc_score(truth, pred, average=None))
    pred_acc = np.argmax(pred, axis=1)
    pred_acc = one_hot[pred_acc]
    accuracy = (pred_acc == truth).astype(float).mean(axis=0)
    acc.append(accuracy)
  return acc, auc


def Akana_data(path='/oak/stanford/groups/ljerby/SharedResources/Akana2024/Data/', class_flag=False):
  X = sparse.load_npz(path+'A375_tpm.npz').toarray()
  #pca = PCA(n_components=1024)
  #pca.fit(X)
  #X = pca.transform(X)
  tmp = np.load(path+'A375_labels_and_extras.npz')
  Y = tmp['Y']
  if class_flag:
    Y = np.concatenate([(np.expand_dims(Y[:,0],axis=1)<=0.05).astype(float), np.expand_dims((Y[:,0]>0.05).astype(float)*(Y[:,0]<0.95).astype(float),axis=1), 
                        (np.expand_dims(Y[:,0],axis=1)>=0.95).astype(float), (np.expand_dims(Y[:,1],axis=1)<=0.05).astype(float), 
                        np.expand_dims((Y[:,1]>0.05).astype(float)*(Y[:,1]<0.95).astype(float),axis=1), (np.expand_dims(Y[:,1],axis=1)>=0.95).astype(float)], axis=1)
  tr_ids = tmp['tr_ids']
  trX = X[tr_ids]
  trY = Y[tr_ids]
  return trX, trY
