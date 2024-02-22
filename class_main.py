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
parser.add_argument('--loss', default='CE', type=str, help='loss functions to use ()')
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
  trX, trY = Akana_data(class_flag=True)
  print(trX.shape)
  print(trY.shape)
  num_targets = 2
  num_class = 3
tr_pair_data = pair_dataset(trX, trY)

epochs = 150
milestones = [50,100]

kf = KFold(n_splits=5)
tmpX = np.zeros((trY.shape[0],1))
part = 0
print ('Start Training')
print ('-'*30)
paraset = [0.1, 1.0, 10.0]
if args.loss in ['GCE']:
  paraset = [0.05, 0.7, 0.95]
elif args.loss in ['SCE']:
  paraset = [0.05, 0.5, 0.95]

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
    model = MLP(input_dim=trX.shape[-1], hidden_sizes=(1024,512,256,128,64, ), num_classes=num_targets*num_class).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if args.loss == 'CE':
      Loss = torch.nn.CrossEntropyLoss()
    elif args.loss == 'LDR':
      Loss = LDRLoss_V1(Lambda = para)
    elif args.loss == 'GCE':
      Loss = GCELoss(q = para)
    elif args.loss == 'CS':
      Loss = CSLoss(threshold = para)
    elif args.loss == 'WW':
      Loss = WWLoss(threshold = para)
    elif args.loss == 'SCE':
      Loss = SCELoss(balance = para)



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
          loss = 0 
          for i in range(num_targets):
            s,e = i*num_class, (i+1)*num_class
            sub_tr_Y, sub_pred_Y = tr_Y[:,s:e], pred_Y[:,s:e]
            #print(sub_tr_Y.shape, sub_pred_Y.shape)
            loss += Loss(sub_pred_Y, sub_tr_Y)
          epoch_loss += loss.cpu().detach().numpy()
          loss.backward()
          optimizer.step()
      scheduler.step()
      epoch_loss /= (idx+1)
      print('Epoch=%s, time=%.4f'%(epoch, time.time() - start_time))
      preds = np.concatenate(pred, axis=0)
      truths = np.concatenate(truth, axis=0)
      auc, acc = class_eval(preds,truths)
      print('Epoch=%s, train_loss=%.4f, lr=%.4f'%(epoch, epoch_loss, scheduler.get_last_lr()[0]))
      for i in range(num_targets):
        print('target=%s'%(i))
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
      auc, acc = class_eval(preds,truths)
      for i in range(num_targets):
        print('target=%s'%(i))
        print('AUCs:', auc[i], np.around(np.mean(auc[i]),3))
        print('ACCs:', acc[i], np.around(np.mean(acc[i]),3))

 
      model.train()

  part += 1 

