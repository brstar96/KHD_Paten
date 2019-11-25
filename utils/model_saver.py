import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves best checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred'] # read best model's state
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs: # runs : 모델이 돌고 있는 회차
                previous_metric = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            metric = float(f.readline()) # 학습 평가에 사용된 지표 (mIoU, ROC 등등)
                            previous_metric.append(metric)
                    else:
                        continue
                max_miou = max(previous_metric)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    # 실험에 사용된 파라미터 정보를 model dict로부터 불러와 parameters.txt로 저장
    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()