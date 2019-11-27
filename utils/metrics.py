import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.cfMatrix = np.zeros((self.num_class,)*2)

    def F1_Score(self, gt_label, predicted_label):
        # class imbalance를 고려하고 싶은 경우 average = 'weighted' 옵션 사용
        return f1_score(gt_label, predicted_label, average='macro')

    def Accuracy_Class(self):
        # calculation using confusion matrix
        Acc = np.diag(self.cfMatrix) / self.cfMatrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def add_batch(self, gt_label, predicted_label):
        # Data(GT, Predicted) fetch and build confusion matrix.
        # gt_label and predicted_label must be same dimension.
        assert gt_label.shape == predicted_label.shape
        cfMatrix = confusion_matrix(gt_label, predicted_label)
        print(classification_report(gt_label, predicted_label))
        self.cfMatrix = cfMatrix

    def reset(self):
        # reset confusion matrix
        self.cfMatrix = np.zeros((self.num_class,) * 2)