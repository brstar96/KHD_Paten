import os
import argparse
import sys
import time
import cv2
import numpy as np
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch
import torch.nn.functional as F
import torch.nn as nn
# from utils.AdamW import AdamW
# from utils.RAdam import RAdam
import evaluation.data_loader as testdataloader
import torchvision
import torchvision.transforms as transforms

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
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Xception(nn.Module):
    def __init__(self, input_channel, num_classes=3):
        super(Xception, self).__init__()

        # Entry Flow
        self.entry_flow_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.entry_flow_2 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            depthwise_separable_conv(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.entry_flow_3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(128, 256, 3, 1),
            nn.BatchNorm2d(256),

            nn.ReLU(True),
            depthwise_separable_conv(256, 256, 3, 1),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        self.entry_flow_4 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(256, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)

        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728)
        )

        # Exit Flow
        self.exit_flow_1 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 1024, 3, 1),
            nn.BatchNorm2d(1024),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.exit_flow_1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.exit_flow_2 = nn.Sequential(
            depthwise_separable_conv(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),

            depthwise_separable_conv(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        entry_out1 = self.entry_flow_1(x)
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        entry_out = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)

        middle_out = self.middle_flow(entry_out) + entry_out

        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)

        return output

class ConvNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(3808, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=150)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--num_classes', type=int, default=3)
    args.add_argument('--learning_rate', type=int, default=0.00001)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    learning_rate = config.learning_rate
    validation_ratio = 0.1 #validation
    random_seed = 2019
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Xception(1, 3)
    model.to(device)
    model2 = ConvNet(num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # # Define Optimizer
    # if args.optimizer.lower() == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # elif args.optimizer.lower() == 'adamw':
    #     optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    # elif args.optimizer.lower() == 'radam':
    #     optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    # elif args.optimizer.lower() == 'radam':
    #     optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    # else:
    #     print("Wrong optimizer args input.")
    #     raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')
        # train mode 일때, path 설정
        nsml.load(checkpoint='34', session='team059/KHD2019_MAMMO/244')           # load시 수정 필수!
        nsml.save(100)
        # print('model_tmp_save')
        img_path = DATASET_PATH + '/train/'
        data, y = data_loader(img_path)
        X = preprocessing(data)

        # Data loader
        batch_loader = DataLoader(dataset=MammoDataset(X,y), ## pytorch data loader 사용
                                    batch_size=batch_size, 
                                    shuffle=True)
        
        # Train the model
        total_step = len(batch_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(batch_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                outputs2 = model2(images)
                if i == 0:
                    print("output : "+str(outputs),"outputs : "+str(outputs2))
                loss = criterion((outputs+outputs2)/2, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item())#, acc=train_acc)
            nsml.save(epoch)
