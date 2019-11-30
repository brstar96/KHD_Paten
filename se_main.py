import cv2, math, time, sys, argparse, os
import numpy as np
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.AdamW import AdamW
from utils.RAdam import RAdam
from torch.optim.lr_scheduler import LambdaLR
from backbones.SENet import senet154, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
from backbones.shufflenet import shufflenet_v2_x2_0
from utils import lr_scheduler
import torchvision
import torchvision.transforms as transforms

import math

import torch
import torch.nn as nn

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data): ## 해당 부분은 data loader의 infer_func을 의미
        print(len(data))
        print(data.shape)
        batch = 20 # batch 사이즈 바꾸기 위해서, 단, 숫자 4 와 200 공약수여야한다.
        print(len(data)//batch)
        # ex ) 200/16 = 12.5 는 총 13번 포문이 돈다.
        pred = []
        for batch_num in range(len(data)//batch):
            print("test_batch")

            batch_data = data[batch_num*batch:batch_num*batch+batch]
            X = preprocessing(batch_data)
            print("batchnum"+str(batch_num))
            with torch.no_grad():
                X = torch.from_numpy(X).float().to(device)
                outputs = model.forward(X)
                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.tolist())

            print('predicted')

        print("list:"+str(pred))
        return pred

    nsml.bind(save=save, load=load, infer=infer)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def data_loader (root_path):
    t = time.time()
    print('Data loading...')
    data = [] # data path 저장을 위한 변수
    labels=[] # 테스트 id 순서 기록
    ## 하위 데이터 path 읽기
    for dir_name,_,_ in os.walk(root_path):
        try: 
            data_id = dir_name.split('/')[-1]
            int(data_id)    
        except: pass
        else: 
            data.append(np.load(dir_name+'/mammo.npz')['arr_0'])            
            labels.append(int(data_id[0]))
    data = np.array(data) ## list to numpy 
    labels = np.array(labels) ## list to numpy 
    print('Dataset Reading Success \n Reading time',time.time()-t,'sec')
    print('Dataset:',data.shape,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, 'each of which 0~2')
    return data, labels

from torch.utils.data import Dataset, DataLoader 
class MammoDataset(Dataset): 
    def __init__(self,X,y): 
        self.len = X.shape[0] 
        self.x_data = torch.from_numpy(X) 
        self.y_data = torch.from_numpy(y) 
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index] 
    def __len__(self): 
        return self.len


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))

def preprocessing(data): 
    print('Preprocessing start')
    # 자유롭게 작성해주시면 됩니다.
    data = np.concatenate([np.concatenate([data[:,0],data[:,1]],axis=2)
                    ,np.concatenate([data[:,2],data[:,3]],axis=2)],axis=1)
    
    X =  np.expand_dims(data, axis=1)
    X = X-X.min()/(X.max()-X.min())
    
    print('Preprocessing complete...')
    print('The shape of X changed',X.shape)
    
    return X

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=20)
    args.add_argument('--num_classes', type=int, default=3)
    args.add_argument('--learning_rate', type=int, default=0.1)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    args.add_argument('--optimizer', type=str, default='RAdam',
                        choices=['SGD', 'ADAM', 'AdamW', 'RAdam'],
                        help='Set optimizer type. (default: RAdam)')
    args.add_argument('--lr_scheduler', type=str, default='WarmupCosineWithHardRestartsSchedule',
                        choices=['StepLR', 'MultiStepLR', 'WarmupCosineSchedule',
                                 'WarmupCosineWithHardRestartsSchedule'],
                        help='Set lr scheduler mode: (default: WarmupCosineSchedule)')
    config = args.parse_args()

    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    learning_rate = config.learning_rate
    optimizer = config.optimizer
    lr_scheduler = config.lr_scheduler

    random_seed = 2019
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = se_resnet101(pretrained=False)
    model = shufflenet_v2_x2_0(pretrained=False)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')
        # train mode 일때, path 설정
        # nsml.load(checkpoint='1', session='team059/KHD2019_MAMMO/48')           # load시 수정 필수!
        # nsml.save(100)
        # print('model_tmp_save')
        img_path = DATASET_PATH + '/train/'
        data, y = data_loader(img_path)
        X = preprocessing(data)

        # Data loader
        batch_loader = DataLoader(dataset=MammoDataset(X,y), ## pytorch data loader 사용
                                    batch_size=batch_size, 
                                    shuffle=True)

        # Define learning rate scheduler
        warmup_steps = int(len(batch_loader) * 0.1)
        lr_scheduler = WarmupCosineWithHardRestartsSchedule(optimizer=optimizer, warmup_steps=warmup_steps,
                                                         t_total=len(batch_loader) , cycles=1.0, last_epoch=-1)
        # Train the model
        total_step = len(batch_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(batch_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item())#, acc=train_acc)
                nsml.save(epoch)
