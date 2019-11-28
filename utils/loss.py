'''
loss function을 정의하는 .py입니다.
'''
import torch
import torch.nn as nn

class buildLosses(object):
    def __init__(self, cuda=True, batch_average=True, ignore_index=255, ):
        self.cuda = cuda
        self.batch_average = batch_average
        self.ignore_index = ignore_index

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        # 1. loss 추가는 이곳에 elif 추가 후 진행할것.
        # 2. 아래에 추가하고 싶은 loss에 대한 함수를 def할것.
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)

        if alpha is not None:
            logpt *= alpha

        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

# 이 .py에서 standalone으로 loss function이 잘 작동하는지 확인하기 위한 목적
if __name__ == "__main__":
    loss = buildLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




