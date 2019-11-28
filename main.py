import argparse, logging, os, torch, warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import Dataset, DataLoader
from utils.metrics import Evaluator
from utils.dataLoader import KaKR3rdDataset, MammoDataset
from utils.tensorboard_summary import TensorboardSummary
from utils import lr_scheduler
from utils.loss import buildLosses
from utils.model_saver import Saver
from networks import initialize_model
from torch.backends import cudnn
from torch import optim
from AdamW import AdamW
from RAdam import RAdam
# import nsml
# from nsml.constants import DATASET_PATH, GPU_NUM
DATASET_PATH = None # temp

warnings.filterwarnings('ignore')

# def bind_model(model):
#     def save(dir_name):
#         os.makedirs(dir_name, exist_ok=True)
#         torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
#         print('model saved!')
#
#     def load(dir_name):
#         model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
#         model.eval()
#         print('model loaded!')
#
#     def infer(data): ## 해당 부분은 data loader의 infer_func을 의미
#         X = preprocessing(data)
#         with torch.no_grad():
#             X = torch.from_numpy(X).float().to(device)
#             pred = model.forward(X)
#         print('predicted')
#         return pred
#     nsml.bind(save=save, load=load, infer=infer)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        if args.dataset == 'local':
            parent_dir = os.path.join(os.getcwd(), '../')
            dataset_root_path = os.path.join(parent_dir, '/2019-3rd-ml-month-with-kakr/')

            # 작업중
        elif args.dataset == 'KHD_NSML':
            img_path_train = DATASET_PATH + '/train/' #대회 당일날 주석 풀고 사용.
            img_path_validaton = DATASET_PATH + '/validation/' # 만약 validation용 데이터셋을 제공해주지 않을 경우 train_test_split으로 나눠서 넣기

            # Pytorch Data loader
            self.train_dataset = MammoDataset(args, mode = 'train', DATA_PATH=img_path_train)
            self.validation_dataset = MammoDataset(args, mode='val', DATA_PATH=img_path_validaton)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
            self.validation_loader = DataLoader(self.validation_dataset, batch_size=args.batch_size, shuffle=True)
            print('Dataset class : ', self.args.class_num)
            print('Train/Val dataloader length : ' + str(len(self.train_loader)) + ', ' + str(len(self.validation_loader)))
        else:
            print("Invalid dataset type.")
            raise ValueError('Argument --dataset must be `local` or `KHD_NSML`.')

        # Define network
        model = initialize_model(model_name=args.backbone, use_pretrained=True)
        model.to(self.device)

        # Print parameters to be optimized/updated.
        print("Params to learn:")
        if args.feature_extracting:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Define Optimizer
        if args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adamw':
            optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'radam':
            optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            print("Wrong optimizer args input.")
            raise NotImplementedError

        # Define Evaluator (F1, Acc_class 등의 metric 계산을 정의한 클래스)
        self.evaluator = Evaluator(self.args.class_num)

        # Define learning rate scheduler
        self.scheduler = lr_scheduler.defineLRScheduler(args, optimizer, len(self.train_loader))

        # Define Criterion
        weight = None # Calculate class weight when dataset is strongly imbalanced. (see pytorch deeplabV3 code's main.py)
        self.criterion = buildLosses(cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        use_amp = False
        if has_apex and args.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O2')
            use_amp = True
        if args.local_rank == 0:
            logging.info('NVIDIA APEX {}. AMP {}.'.format(
                'installed' if has_apex else 'not installed', 'on' if use_amp else 'off'))

        if args.distributed:
            if args.sync_bn:
                try:
                    if has_apex:
                        self.model = convert_syncbn_model(model)
                    else:
                        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    if args.local_rank == 0:
                        logging.info('Converted model to use Synchronized BatchNorm.')
                except Exception as e:
                    logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
            if has_apex:
                self.model = DDP(model, delay_allreduce=True)
            else:
                if args.local_rank == 0:
                    logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
                self.model = DDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.to(self.device), target.to(self.device)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset() # metric이 정의되어 있는 evaluator클래스 초기화 (confusion matrix 초기화 수행)

        tbar = tqdm(self.validation_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                # image, target = image.cuda(), target.cuda()
                image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Print validation log during the training
        F1 = self.evaluator.F1_Score()
        Acc_class = self.evaluator.Accuracy_Class()
        self.writer.add_scalar('val/mIoU', F1, epoch)
        self.writer.add_scalar('val/Acc', Acc_class, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("F1-Score:{}, Acc_class:{}".format(F1, Acc_class))
        print('Loss: %.3f' % test_loss)

        # F1 metric 성능에 따라 제일 좋은 모델의 checkpoint를 저장
        new_pred = F1
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    # Set base parameters (dataset path, backbone name etc...)
    parser = argparse.ArgumentParser(description="This code is for testing various octConv+ResNet.")
    parser.add_argument('--backbone', type=str, default='oct_resnet50',
                        choices=['resnet101', 'resnet152' # Original ResNet 
                            'resnext50_32x4d', 'resnext101_32x8d', # Modified ResNet
                            'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200', # OctConv + Original ResNet
                            'senet154', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', # Squeeze and excitation module based models
                            'efficientnetb3', 'efficientnetb4', 'efficientnetb5'], # EfficientNet models
                        help='Set backbone name')
    parser.add_argument('--dataset', type=str, default='local',
                        choices=['local', 'KHD_NSML'],
                        help='Set dataset path. `local` is for testing via local device, KHD_NSML is for testing via NSML server. ')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Set CPU threads for pytorch dataloader')
    parser.add_argument('--checkname', type=str, default=None,
                        help='Set the checkpoint name. if None, checkname will be set to current dataset+backbone+time.')

    # Set hyper params for training network.
    parser.add_argument('--base_size', type=int, default=224,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='crop image size')
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
    parser.add_argument('--use_pretrained', type=bool, default=True) # pre-trained model 사용여부(pytorch model zoo에 있는 모델 위주로 사용 권장)
    parser.add_argument('--feature_extracting', type=bool, default=True) #

    # Set optimizer params for training network.
    parser.add_argument('--lr', type=float, default=None,
                        help='Set starting learning rate. If None, lr will be set to current dataset`s lr.')
    parser.add_argument('--lr_scheduler', type=str, default='WarmupCosineSchedule',
                        choices=['StepLR', 'MultiStepLR', 'WarmupCosineSchedule', 'WarmupCosineWithHardRestartsSchedule'],
                        help='Set lr scheduler mode: (default: WarmupCosineSchedule)')
    parser.add_argument('--optim', type=str, default='RAdam',
                        choices=['SGD', 'ADAM', 'AdamW', 'RAdam'],
                        help='Set optimizer type. (default: RAdam)')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='Set momentum value for pytorch`s SGD optimizer. (default: 0.9)')

    # Set params for CUDA, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=2019, metavar='S', help='random seed (default: 2019)')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    parser.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuDNN version : ', torch.backends.cudnn.version())

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.distributed is None and args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.distributed = True
            args.sync_bn = True
        else:
            args.distributed = False
            args.sync_bn = False

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
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

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

    main()