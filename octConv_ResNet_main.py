import argparse
import os, torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils.dataLoader import train_dataloader
from utils.tensorboard_summary import TensorboardSummary
from utils.datasetPath import Path
from utils.loss import buildLosses
from utils.model_saver import Saver
from networks import initialize_model
import torch.optim as optim
from AdamW import AdamW
from RAdam import RAdam

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
        train_DataLoader = train_dataloader(storageType='local', input_size=args.base_size, batch_size=args.batch_size, num_workers=4)
        print('Dataset class : ', self.nclass)

        # Define network

        model = initialize_model(model_name, embedding_dim, feature_extracting, use_pretrained=True)
        model.to(self.device)

        # Define Optimizer
        if args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adam':
            optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'radam':
            optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            print("Wrong optimizer args input.")
            raise NotImplementedError

        # Define learning rate scheduler
        if args.lr_scheduler.lower() == 'steplr':
            self.scheduler = optim.plr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif args.lr_scheduler.lower() == 'multisteplr':
            '''
            Assuming optimizer uses lr = 0.05 for all groups,
            lr = 0.05     if epoch < 30
            lr = 0.005    if 30 <= epoch < 80
            lr = 0.0005   if epoch >= 80. (gamma : Multiplicative factor of learning rate decay. Default: 0.1.)
            '''
            self.scheduler = optim.plr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        elif args.lr_scheduler.lower() == 'warmupcosineschedule': # 11/25 여기서부터 구현하기
            self.scheduler = optim.plr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif args.lr_scheduler.lower() == 'warmupcosinewithhardrestartsschedule':
            self.scheduler = optim.plr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            print("Wrong lr_scheduler args input.")
            raise NotImplementedError

        # Define Criterion
        weight = None # Calculate class weight when dataset is strongly imbalanced. (see pytorch deeplabV3 code's main.py)
        self.criterion = buildLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

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

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            # print('Training function activated, ' + str(i) + 'th step running.')
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                # image, target = image.cuda(), target.cuda()
                image, target = image.to(self.device), target.to(self.device)
                # print(image.tolist())
                # print(target.tolist())

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

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
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
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
            # print(target.tolist())
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
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
                        choices=['oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200'],
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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuDNN version : ', torch.backends.cudnn.version())

    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, lr and class_num of dataset.
    if args.epochs is None:
        epoches = {'local': 150, 'KHD_NSML': 100, }
        args.epochs = epoches[args.dataset]

    if args.lr is None:
        lrs = {'local': 0.1, 'KHD_NSML': 0.1,}

    if args.class_num is None:
        class_nums = {'local': 192, 'KHD_NSML': None, } # change KHD_NSML's class num in 29, september
        args.class_num  = class_nums[args.dataset]

    if args.checkname is None:
        now = datetime.now()
        args.checkname = str(args.dataset) + '-' + str(args.backbone) + ('%s-%s-%s' % (now.year, now.month, now.day))

    print(args)
    torch.manual_seed(args.seed)

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
