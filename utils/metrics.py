import numpy as np
from sklearn.metrics import f1_score

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def F1_Score(self, gt_label, predicted_label):
        # class imbalance를 고려하고 싶은 경우 average = 'weighted' 옵션 사용
        return f1_score(gt_label, predicted_label, average='macro')

    def Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def _generate_matrix(self, gt_label, predicted_label):
        mask = (gt_label >= 0) & (gt_label < self.num_class)
        label = self.num_class * gt_label[mask].astype('int') + predicted_label[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_label, predicted_label):
        assert gt_label.shape == predicted_label.shape
        self.confusion_matrix += self._generate_matrix(gt_label, predicted_label)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




