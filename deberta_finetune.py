from scipy.sparse import data
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bert_nli import BertNLIModel
from random import shuffle
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int,
                    help="GPU device number",
                    default=0)
parser.add_argument("--lr", type=float,
                    help="learning rate",
                    default=1e-5)
parser.add_argument("--n_epoch", type=int,
                    help="max number of epochs",
                    default=100)
parser.add_argument("--tolerant", type=int,
                    help="tolerant",
                    default=10)
parser.add_argument("--n_layer", type=int,
                    help="train how much layer",
                    default=12)
parser.add_argument("--ratio", type=float,
                    help="train-test ratio",
                    default=0.8)
parser.add_argument("--pos_repeat", type=int,
                    help="repeat how much times for positive data",
                    default=1)

args = parser.parse_args()
print(args)

assert args.pos_repeat >= 1
torch.cuda.set_device(args.device)
ratio = args.ratio
n_layer = args.n_layer
lr = args.lr # 1e-5 originally

prefix = 'deberta_tars/'
checkpoint_file = prefix + 'deberta_'+str(ratio)+'_'+str(n_layer)+'_'+str(lr)+'_'+str(args.pos_repeat)+'_checkpoint.pth.tar'
best_file = prefix + 'deberta_'+str(ratio)+'_'+str(n_layer)+'_'+str(lr)+'_'+str(args.pos_repeat)+'_best.pth.tar'

def checkpoint_save(epoch, mymodel, optimizer, best_pred, is_best):
    state = {'epoch': epoch + 1,
            'state_dict': mymodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred}
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)

def checkpoint_load(model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

model = BertNLIModel(model_path=None,gpu=torch.cuda.is_available(),bert_type='deberta',label_num=3,batch_size=64,reinit_num=-1,freeze_layers=False)

for name, param in model.bert.named_parameters():
    if (len(name.split('.'))>3 and name.split('.')[3].isdigit() and int(name.split('.')[3]) < 12-n_layer) and 'classifier' not in name:
        param.requires_grad = False
optimizer = optim.SGD([p for n,p in model.bert.named_parameters() if p.requires_grad], lr=lr, momentum=0.9)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

with open('deberta_train_corpus.txt', 'r') as f_corpus:
    data_list = list(f_corpus.readlines())
data_list = [a.strip().split('\t') for a in data_list]
shuffle(data_list)
train_num = int(ratio*len(data_list))
train_data = data_list[:train_num]
test_data = data_list[train_num:]
pos_data = [a for a in train_data if a[2]=='1']
if args.pos_repeat > 1:
    print('pos data len', len(pos_data))
    train_data = train_data + pos_data * (args.pos_repeat-1)
    shuffle(train_data)
train_X = [(a[0],a[1]) for a in train_data]
train_Y = torch.LongTensor([int(a[2]) for a in train_data]).cuda()
test_X = [(a[0],a[1]) for a in test_data]
test_Y = torch.LongTensor([int(a[2]) for a in test_data]).cuda()

def train_one_epoch(model, optimizer, train_X, train_Y, test_X, test_Y):
    model.train()
    stt = time.time()
    total_loss = 0
    bs = model.batch_size
    for batch_idx in tqdm(range(0,len(train_X),bs), disable=True,desc='Training'):
        model.zero_grad()
        optimizer.zero_grad()
        probs = model.ff(train_X[batch_idx:batch_idx+bs],False)[1]
        probs = torch.cat([probs[:,:2].sum(dim=1,keepdim=True), probs[:,2].unsqueeze(1)],dim=1)
        if batch_idx == 0:
            p = probs.detach().cpu().numpy()
            for kk in range(batch_idx,batch_idx+bs):
                print(train_X[kk], train_Y[kk], p[kk])
        loss = nn.functional.cross_entropy(probs, train_Y[batch_idx:batch_idx+bs])
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('total loss', total_loss, 'time', time.time()-stt)

    return model_eval(model, test_X, test_Y)

def model_eval(model, test_X, test_Y):
    model.eval()
    with torch.no_grad():
        label, test_probs = model(test_X)
        test_probs = torch.cat([test_probs[:,:2].sum(dim=1,keepdim=True), test_probs[:,2].unsqueeze(1)],dim=1)
        test_pred_Y = test_probs.argmax(dim=1)

        print(test_pred_Y[:10], test_Y[:10])
        TP = ((test_pred_Y == 1) & (test_Y == 1)).cpu().sum().item()
        TN = ((test_pred_Y == 0) & (test_Y == 0)).cpu().sum().item()
        FN = ((test_pred_Y == 0) & (test_Y == 1)).cpu().sum().item()
        FP = ((test_pred_Y == 1) & (test_Y == 0)).cpu().sum().item()
        print(TP, TN, FN, FP)

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        return p, r, F1, acc

n_epoch = args.n_epoch
p, r, F1, acc = model_eval(model, test_X, test_Y)
print('original','p:',p,'r:',r,'F1:',F1,'acc:',acc)
bestF1 = F1
tolerant = args.tolerant
no_inc = 0
for e in range(n_epoch):
    p, r, F1, acc = train_one_epoch(model, optimizer, train_X, train_Y, test_X, test_Y)
    print('epoch:',e,'p:',p,'r:',r,'F1:',F1,'acc:',acc)
    checkpoint_save(e, model, optimizer, F1, (F1 > bestF1))
    if F1 <= bestF1:
        no_inc = no_inc + 1
        if no_inc == tolerant:
            print('Untolerable!')
            break
    else:
        bestF1 = F1
        no_inc = 0
