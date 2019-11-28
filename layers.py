# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin,
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh,
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao,
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema,
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy,
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of breast_cancer_classifier.
#
# breast_cancer_classifier is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# breast_cancer_classifier is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with breast_cancer_classifier.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
"""
Defines layers used in models.py.
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3

from constants import VIEWS

class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape)) # np.prod를 통해 (4, 2) 사이즈 배열의 요소들을 곱함
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape) # 256 FC를 입력으로 받아 flattened output (scalar) 반환

    def forward(self, input_image):
        output = self.fc_layer(input_image) # scalar
        if len(self.output_shape) > 1: # output_shape = (4, 2)인경우 len(output_shape)=2
            output = output.view(output.shape[0], *self.output_shape) # output_shape = (4, 2)인경우 output_shape[0]=4. 즉 (scalar, 2),
        output = F.log_softmax(output, dim=-1) # 2채널 softmax 아웃풋 (scalar, 2)
        return output

class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std

    def forward(self, x):
        return {
            VIEWS.L_CC: self.single_add_gaussian_noise(x[VIEWS.L_CC]),
            VIEWS.L_MLO: self.single_add_gaussian_noise(x[VIEWS.L_MLO]),
            VIEWS.R_CC: self.single_add_gaussian_noise(x[VIEWS.R_CC]),
            VIEWS.R_MLO: self.single_add_gaussian_noise(x[VIEWS.R_MLO]),
        }

    def single_add_gaussian_noise(self, single_view):
        if not self.gaussian_noise_std or not self.training:
            return single_view
        return single_view + single_view.new(single_view.shape).normal_(std=self.gaussian_noise_std)


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            view_name: self.single_avg_pool(view_tensor)
            for view_name, view_tensor in x.items()
        }

    @staticmethod
    def single_avg_pool(single_view):
        n, c, _, _ = single_view.size()
        return single_view.view(n, c, -1).mean(-1)
