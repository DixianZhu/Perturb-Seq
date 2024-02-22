import time
from loss import *
from models import MLP
from utils import *
import torch 
import numpy as np
from sklearn.model_selection import KFold
import argparse
from scipy import stats
parser = argparse.ArgumentParser(description = 'FAR experiments')
parser.add_argument('--loss', default='FAR', type=str, help='loss functions to use ()')
parser.add_argument('--dataset', default='Akana', type=str, help='the name for the dataset to use')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum parameter for SGD optimizer')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay for training the model')
parser.add_argument('--batch_size', default=256, type=int, help='training batch size')

# paramaters
args = parser.parse_args()
SEED = 123
BATCH_SIZE = args.batch_size
lr = args.lr
decay = args.decay
set_all_seeds(SEED)
# dataloader
num_targets = 1
if args.dataset == 'Akana':
  trX, trY = Akana_data()
  num_targets = 2
tr_pair_data = pair_dataset(trX, trY)

epochs = 150
milestones = [50,100]

kf = KFold(n_splits=5)
tmpX = np.zeros((trY.shape[0],1))
part = 0
print ('Start Training')
print ('-'*30)
paraset = [0.1, 0.5, 0.9]
if args.loss in ['Huber', 'focal-MAE', 'focal-MSE']:
  paraset = [0.25,1,4]
elif args.loss in ['MAE', 'MSE']:
  paraset = [0.1,0.5,0.9] # dummy repeats
elif args.loss == 'ranksim':
  paraset = [0.5,1,2]
elif args.loss in ['FAR','FAR-EXP']:
  paraset = [0.2, 2, 5]
elif args.loss in ['RNC']:
  paraset = [1,2,4]
elif args.loss in ['ConR']:
  paraset = [0.2,1,4]

device = 'cpu' 
# can use gpu if it is faster for you:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for train_id, val_id in kf.split(tmpX):
  tmp_trainSet = torch.utils.data.Subset(tr_pair_data, train_id)
  tmp_valSet = torch.utils.data.Subset(tr_pair_data, val_id)
  for para in paraset: 
    trainloader = torch.utils.data.DataLoader(dataset=tmp_trainSet, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(dataset=tmp_valSet, batch_size=BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)
    basic_loss = torch.nn.L1Loss()
    model = MLP(input_dim=trX.shape[-1], hidden_sizes=(1024,512,256,128,64, ), num_classes=num_targets).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    if args.loss in ['FAR','FAR-EXP']:
      basic_loss = FAR(alpha=para,version=args.loss)
    elif args.loss in ['MSE']:
      basic_loss = torch.nn.MSELoss()
    elif args.loss == 'Huber':
      basic_loss = torch.nn.HuberLoss(delta=para)
    elif args.loss == 'RNC':
      add_loss = RnCLoss(temperature=para)


    print('para=%s, part=%s'%(para, part))
    for epoch in range(epochs): # could customize the running epochs
      epoch_loss = 0
      pred = []
      truth = []
      start_time = time.time()
      for idx, data in enumerate(trainloader):
          optimizer.zero_grad()
          tr_X, tr_Y = data[0].to(device), data[1].to(device)
          pred_Y, feat = model(tr_X)
          pred.append(pred_Y.cpu().detach().numpy())
          truth.append(tr_Y.cpu().detach().numpy())
          if args.loss in ['FAR','FAR-EXP']:
            ratio = epoch/float(epochs)
            bloss = basic_loss(pred_Y, tr_Y)
            # Potentially can utilize adaptive alpha for FAR. We didn't use it in our experiments.
            # bloss = basic_loss(pred_Y, tr_Y, alpha = (0.1+ratio)*para)
          else:
            bloss = basic_loss(pred_Y, tr_Y)
          if args.loss in ['MAE', 'MSE', 'Huber', 'ranksim', 'focal-MAE', 'focal-MSE', 'ConR', 'FAR', 'FAR-EXP']:
            loss = bloss
          else:
            if args.loss in ['RNC']:
              aloss = add_loss(feat, tr_Y)
            else:
              aloss = add_loss(pred_Y, tr_Y)
          if args.loss == 'ranksim':
            loss += 100*batchwise_ranking_regularizer(feat, tr_Y, para)
          elif args.loss == 'ConR':
            loss += para*ConR_extend(feat, tr_Y, pred_Y) 
          elif args.loss == 'focal-MAE':
            loss = weighted_focal_mae_loss(pred_Y, tr_Y, beta = para)
          elif args.loss == 'focal-MSE':
            loss = weighted_focal_mse_loss(pred_Y, tr_Y, beta = para)
          elif args.loss == 'RNC':
            if epoch < milestones[0]:
              loss = aloss
            else:
              loss = bloss
          epoch_loss += loss.cpu().detach().numpy()
          loss.backward()
          optimizer.step()
      scheduler.step()
      epoch_loss /= (idx+1)
      print('Epoch=%s, time=%.4f'%(epoch, time.time() - start_time))
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0)
      MAE, RMSE, pearson, spearman = reg_eval(preds,truths)
      auc, acc = class_reg_eval(preds,truths)
      print('Epoch=%s, train_loss=%.4f, lr=%.4f'%(epoch, epoch_loss, scheduler.get_last_lr()[0]))
      for i in range(num_targets):
        print('target=%s, train_MAE=%.4f, train_RMSE=%.4f, train_Pearson=%.4f, train_Spearman=%.4f'%(i, MAE[i], RMSE[i], pearson[i], spearman[i]))
        print('AUCs:', auc[i], np.around(np.mean(auc[i]),3))
        print('ACCs:', acc[i], np.around(np.mean(acc[i]),3))
 
      
      pred = []
      truth = [] 
      model.eval()
      for idx, data in enumerate(validloader):
          te_X, te_Y = data[0].to(device), data[1].to(device)
          pred_Y, feat = model(te_X)
          pred.append(pred_Y.cpu().detach().numpy())
          truth.append(te_Y.cpu().detach().numpy())
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0) 
      MAE, RMSE, pearson, spearman = reg_eval(preds,truths)
      auc, acc = class_reg_eval(preds,truths)
      for i in range(num_targets):
        print('target=%s, valid_MAE=%.4f, valid_RMSE=%.4f, valid_Pearson=%.4f, valid_Spearman=%.4f'%(i, MAE[i], RMSE[i], pearson[i], spearman[i]))
        print('AUCs:', auc[i], np.around(np.mean(auc[i]),3))
        print('ACCs:', acc[i], np.around(np.mean(acc[i]),3))

 
      model.train()

  part += 1 

