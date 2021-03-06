# 참고용 Stratified CV + 앙상블 코드 : https://www.kaggle.com/janged/3rd-ml-month-xception-stratifiedkfold-ensemble

import argparse, os, torch, warnings, random, time
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.metrics import Evaluator
from utils.dataLoader import KaKR3rdDataset, MammoDataset
from utils.lr_scheduler import defineLRScheduler
from sklearn.model_selection import train_test_split
from utils.AdamW import AdamW
from utils.RAdam import RAdam
import models
from torch.backends import cudnn
from torch import optim
from constants import VIEWS
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

warnings.filterwarnings('ignore')

# 모든 Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_np(t):
    # return t.cpu().detach().numpy()
    return t.detach().numpy()

def soft_voting(probs):
    _arrs = [probs[key] for key in probs]
    return np.mean(np.mean(_arrs, axis=1), axis=0)

def data_loader(data_set_path):
    t = time.time()
    print('Data loading...')
    data_path = []  # data path 저장을 위한 변수
    labels = []  # 테스트 id 순서 기록
    ## 하위 데이터 path 읽기
    for dir_name, _, _ in os.walk(data_set_path):
        try:
            data_id = dir_name.split('/')[-1]
            int(data_id)
        except:
            pass
        else:
            data_path.append(dir_name)
            labels.append(int(data_id[0]))

    ## 데이터만 읽기
    data = []  # img저장을 위한 list
    for d_path in data_path:
        sample = np.load(d_path + '/mammo.npz')['arr_0']
        data.append(sample)
    data = np.array(data)  ## list to numpy

    print('Dataset Reading Success \n Reading time', time.time() - t, 'sec')
    print('Dataset:', data.shape, 'np.array.shape(files, views, width, height)')

    return data, labels, len(data), len(labels)  # Numpy arr과 정답 클래스

def bind_model(model, args):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## 해당 부분은 data loader의 infer_func을 의미
        print(len(data))
        print(data.shape)
        batch = 20  # batch 사이즈 바꾸기 위해서, 단, 숫자 4 와 200 공약수여야한다.
        print(len(data) // batch)
        # ex ) 200/16 = 12.5 는 총 13번 포문이 돈다.
        pred = []
        for batch_num in range(len(data) // batch):
            print("test_batch")

            batch_data = data[batch_num * batch:batch_num * batch + batch]
            X = preprocessing(batch_data)
            print("batchnum" + str(batch_num))
            with torch.no_grad():
                X = torch.from_numpy(X).float().to(device)
                outputs = model.forward(X)
                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.tolist())

            print('predicted')

        print("list:" + str(pred))
        return pred

    nsml.bind(save=save, load=load, infer=infer)

    nsml.bind(save=save, load=load, infer=infer)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Total Epoches:', args.epochs)

        # Define network
        input_channels = 3 if args.use_additional_annotation else 2 # use_additional_annotation = True이면 3
        model = models.ImageBreastModel(args, input_channels) # 4개 모델들의 softmax값 리턴 (총 8개의 softmax)
        model.to(self.device)

        # Print parameters to be optimized/updated.
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)

        # Define Optimizer
        if args.optim.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim.lower() == 'adamw':
            optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        elif args.optim.lower() == 'radam':
            optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            print("Wrong optimizer args input.")
            raise NotImplementedError

        # Define Evaluator (F1, Acc_class 등의 metric 계산을 정의한 클래스)
        self.evaluator = Evaluator(self.args.class_num)

        # Define Criterion
        weight = None # Calculate class weight when dataset is strongly imbalanced. (see pytorch deeplabV3 code's main_local.py)
        self.criterion = nn.CrossEntropyLoss()

        # buildLosses(cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        bind_model(self.model, args)

        if args.pause:  ## test mode 일때는 여기만 접근
            print('Inferring Start...')
            nsml.paused(scope=locals())

        if args.mode == 'train':  ### training mode 일때는 여기만 접근
            print('Training Start...')
            img_path = DATASET_PATH + '/train/'

            # Define Dataloader
            if args.dataset == 'local':
                parent_dir = os.path.join(os.getcwd(), '../')
                dataset_root_path = os.path.join(parent_dir, '/2019-3rd-ml-month-with-kakr/')

                # Pytorch Data loader
                self.train_dataset = KaKR3rdDataset(args, mode='train',
                                                    DATA_PATH=os.path.join(dataset_root_path, 'train/'))
                self.validation_dataset = KaKR3rdDataset(args, mode='val',
                                                         DATA_PATH=os.path.join(dataset_root_path, 'test/'))
                self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
                self.validation_loader = DataLoader(self.validation_dataset, batch_size=args.batch_size, shuffle=True)
                print('Dataset class : ', self.args.class_num)
                print('Train/Val dataloader length : ' + str(len(self.train_dataset)) + ', ' + str(len(self.validation_dataset)))
            elif args.dataset == 'KHD_NSML':
                img_path = DATASET_PATH + '/train/'
                img_path_validaton = DATASET_PATH + '/validation/'  # 만약 validation용 데이터셋을 제공해주지 않을 경우 train_test_split으로 나눠서 넣기

                # Pytorch Data loader
                data, labels, length_data, length_labels = data_loader(img_path)
                X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.1, random_state=2019)

                self.train_dataset = MammoDataset(args, mode='train', data=X_train, labels=X_test, len_data=len(X_train), len_label=len(X_test))
                self.validation_dataset = MammoDataset(args, mode='val', data=Y_train, labels=Y_test, len_data=len(Y_train), len_label=len(Y_test))
                self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
                self.validation_loader = DataLoader(self.validation_dataset, batch_size=args.batch_size, shuffle=True)
                print('Dataset class : ', self.args.class_num)
                print('Train/Val dataset length : ' + str(len(self.train_dataset)) + str(len(self.validation_dataset)))
            else:
                print("Invalid dataset type.")
                raise ValueError('Argument --dataset must be `local` or `KHD_NSML`.')

            # Define learning rate scheduler
            self.scheduler = defineLRScheduler(args, optimizer, len(self.train_dataset))

            # Train the model
            self.training(args.epoch)

    def training(self, epochs):
        self.model.train() # Train모드로 전환
        num_img_tr = len(self.train_dataset)
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                image, target = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(image) # 각 뷰포인트마다 2개의 softmax결과 * 4개 = 8개의 softmax (python set으로 반환됨)
                output = soft_voting(outputs) # voting해서 하나의 클래스만 남기도록 하는 부분 추가 (set의 voting)

                loss_LMLO = self.criterion(outputs[VIEWS.LMLO], labels)
                loss_RMLO = self.criterion(outputs[VIEWS.RMLO], labels)
                loss_LCC = self.criterion(outputs[VIEWS.LCC], labels)
                loss_RCC = self.criterion(outputs[VIEWS.RCC], labels)
                loss_VIEWS = [loss_LMLO, loss_RMLO, loss_LCC, loss_RCC]
                loss = self.criterion(output, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predict_vector = np.argmax(to_np(output), axis=1)
                label_vector = to_np(labels)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                nsml.report(summary=True, step=epoch, epoch_total=epochs, loss=loss.item(), acc = accuracy, loss_VIEWS = loss_VIEWS)
                log_batch = 'Epoch {}  Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f}'.format(
                    int(epoch), int(batch_idx), len(self.train_dataset), float(loss.item()), float(accuracy))
                if batch_idx % 10 == 0: # 10스텝마다 출력
                    print(log_batch)

                total_loss += loss.item()
                total_correct += bool_vector.sum()

            nsml.report(summary=True, step=epoch, epoch_total=epochs, loss=total_loss.item(), acc = total_correct / num_img_tr)
            log_epoch = 'Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f}'.format(
                epoch, epochs, total_loss / num_img_tr, total_correct / num_img_tr)
            if epoch / 2 == 0:
                print(log_epoch)

            if epoch / 5 == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.evaluator.reset()  # metric이 정의되어 있는 evaluator클래스 초기화 (confusion matrix 초기화 수행)
                    length_val_dataloader = len(self.validation_dataset)
                    print("Start epoch validation...")

                    for item in self.validation_loader:
                        images = item['image'].to(device)
                        labels = item['label'].to(device)

                        outputs = self.model(images)  # 각 뷰포인트마다 2개의 softmax결과 * 4개 = 8개의 softmax (python set으로 반환됨)
                        output = soft_voting(outputs)  # voting해서 하나의 클래스만 남기도록 하는 부분 추가 (set의 voting)

                        predict_vector = np.argmax(to_np(output), axis=1)
                        label_vector = to_np(labels)
                        bool_vector = predict_vector == label_vector
                        accuracy = bool_vector.sum() / len(bool_vector)

                        log_validacc = 'Validation Acc of the model on {} images : {}'.format(length_val_dataloader, accuracy)
                        nsml.report(summary=True, step=epoch, epoch_total=epochs, acc=accuracy)
                        print(log_validacc)

    # def validation(self, epoch):
    #     self.model.eval()
    #     self.evaluator.reset() # metric이 정의되어 있는 evaluator클래스 초기화 (confusion matrix 초기화 수행)
    #
    #     tbar = tqdm(self.validation_loader, desc='\r')
    #     test_loss = 0.0
    #     for i, sample in enumerate(tbar):
    #         image, target = sample['image'], sample['label']
    #         if self.args.cuda:
    #             # image, target = image.cuda(), target.cuda()
    #             image, target = image.to(self.device), target.to(self.device)
    #         with torch.no_grad():
    #             output = self.model(image)
    #         loss = self.criterion(output, target)
    #         test_loss += loss.item()
    #         tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
    #         pred = output.data.cpu().numpy()
    #         target = target.cpu().numpy()
    #         pred = np.argmax(pred, axis=1)
    #
    #         # Add batch sample into evaluator
    #         self.evaluator.add_batch(target, pred)
    #
    #     # Print validation log during the training
    #     F1 = self.evaluator.F1_Score()
    #     Acc_class = self.evaluator.Accuracy_Class()
    #     self.writer.add_scalar('val/mIoU', F1, epoch)
    #     self.writer.add_scalar('val/Acc', Acc_class, epoch)
    #     print('Validation:')
    #     print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
    #     print("F1-Score:{}, Acc_class:{}".format(F1, Acc_class))
    #     print('Loss: %.3f' % test_loss)
    #
    #     # F1 metric 성능에 따라 제일 좋은 모델의 checkpoint를 저장
    #     new_pred = F1
    #     if new_pred > self.best_pred:
    #         is_best = True
    #         self.best_pred = new_pred
    #         self.saver.save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': self.model.module.state_dict(),
    #             'optimizer': self.optimizer.state_dict(),
    #             'best_pred': self.best_pred,
    #         }, is_best)

def main():
    # Set base parameters (dataset path, backbone name etc...)
    parser = argparse.ArgumentParser(description="This code is for testing various octConv+ResNet.")
    parser.add_argument('--backbone', type=str, default='oct_resnet26',
                        choices=['resnet101', 'resnet152' # Original ResNet 
                            'resnext50_32x4d', 'resnext101_32x8d', # Modified ResNet
                            'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200', # OctConv + Original ResNet
                            'senet154', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', # Squeeze and excitation module based models
                            'efficientnetb3', 'efficientnetb4', 'efficientnetb5'], # EfficientNet models
                        help='Set backbone name')
    parser.add_argument('--dataset', type=str, default='KHD_NSML',
                        choices=['local', 'KHD_NSML'],
                        help='Set dataset path. `local` is for testing via local device, KHD_NSML is for testing via NSML server. ')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Set CPU threads for pytorch dataloader')
    parser.add_argument('--checkname', type=str, default=None,
                        help='Set the checkpoint name. if None, checkname will be set to current dataset+backbone+time.')

    # Set hyper params for training network.
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Set k_folds params for stratified K-fold cross validation.')
    parser.add_argument('--distributed', type=bool, default=None,
                        help='Whether to use distributed GPUs. (default: None)')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='Whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='Whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='Set loss func type. `ce` is crossentropy, `focal` is focal entropy from DeeplabV3.')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Set max epoch. If None, max epoch will be set to current dataset`s max epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--class_num', type=int, default=None,
                        help='Set class number. If None, class_num will be set according to dataset`s class number.')
    parser.add_argument('--use_pretrained', type=bool, default=False) # ImageNet pre-trained model 사용여부
    parser.add_argument('--use_additional_annotation', type=bool, default=True, help='Whether use additional annotation') # 데이터셋에 악성 종양에 대한 세그먼트 어노테이션이 있는 경우 True

    # Set optimizer params for training network.
    parser.add_argument('--lr', type=float, default=None,
                        help='Set starting learning rate. If None, lr will be set to current dataset`s lr.')
    parser.add_argument('--lr_scheduler', type=str, default='WarmupCosineWithHardRestartsSchedule',
                        choices=['StepLR', 'MultiStepLR', 'WarmupCosineSchedule', 'WarmupCosineWithHardRestartsSchedule'],
                        help='Set lr scheduler mode: (default: WarmupCosineSchedule)')
    parser.add_argument('--optim', type=str, default='RAdam',
                        choices=['SGD', 'ADAM', 'AdamW', 'RAdam'],
                        help='Set optimizer type. (default: RAdam)')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='Set momentum value for pytorch`s SGD optimizer. (default: 0.9)')

    # Set params for CUDA, seed and logging
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=2019, metavar='S', help='random seed (default: 2019)')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    parser.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = parser.parse_args()
    print('cuDNN version : ', torch.backends.cudnn.version())

    # default settings for epochs, lr and class_num of dataset.
    if args.epochs is None:
        epoches = {'local': 150, 'KHD_NSML': 100, }
        args.epochs = epoches[args.dataset]

    if args.lr is None:
        lrs = {'local': 0.1, 'KHD_NSML': 0.1,}
        args.lr = lrs[args.dataset]

    if args.class_num is None:
        # local은 KaKR 3rd 자동차 차종분류 데이터셋인 경우 192개의 차종 클래스
        # KHD_NSML은 정상(normal), 양성(benign), 악성(malignant) 3개의 클래스
        class_nums = {'local': 192, 'KHD_NSML': 3, }
        args.class_num  = class_nums[args.dataset]

    if args.checkname is None:
        now = datetime.now()
        args.checkname = str(args.dataset) + '-' + str(args.backbone) + ('%s-%s-%s' % (now.year, now.month, now.day))

    print(args)
    torch.manual_seed(args.seed)

    # Define trainer. (Define dataloader, model, optimizer etc...)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)

    trainer.training(args.epoch)
    # if not trainer.args.no_val and args.epoch % args.eval_interval == (args.eval_interval - 1):
    #     trainer.validation(args.epoch)


if __name__ == "__main__":
    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
    except ImportError:
        from torch.nn.parallel import DistributedDataParallel as DDP
        has_apex = False
    cudnn.benchmark = True

    SEED = 20191129
    seed_everything(SEED) # 하이퍼파라미터 테스트를 위해 모든 시드 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main()