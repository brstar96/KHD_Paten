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
Defines architectures for breast cancer classification models.
"""
import collections as col

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_networks import initialize_model
import layers as layers
from constants import VIEWS

class ImageBreastModel(nn.Module):
    def __init__(self, args, input_channels):
        super(ImageBreastModel, self).__init__()

        self.args = args
        self.four_view_resnet = FourViewResNet(input_channels, args)

        self.fc1_lcc = nn.Linear(256, 256) # in_feature(256), out_feature(256)
        self.fc1_rcc = nn.Linear(256, 256)
        self.fc1_lmlo = nn.Linear(256, 256)
        self.fc1_rmlo = nn.Linear(256, 256)
        self.output_layer_lcc = layers.OutputLayer(256, (4, 2)) # in_feature(256), out_shape(4, 2)
        self.output_layer_rcc = layers.OutputLayer(256, (4, 2))
        self.output_layer_lmlo = layers.OutputLayer(256, (4, 2))
        self.output_layer_rmlo = layers.OutputLayer(256, (4, 2))

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, input_image):
        h = self.all_views_gaussian_noise_layer(input_image) # 각 인풋 이미지에 가우시안 노이즈 추가
        result = self.four_view_resnet(h) # 모델에 이미지 4장 각각 통과
        h = self.all_views_avg_pool(result) # 4개의 result를 각각 average pooling

        # 활성화함수 적용
        h_lcc = F.relu(self.fc1_lcc(h[VIEWS.L_CC]))
        h_rcc = F.relu(self.fc1_rcc(h[VIEWS.R_CC]))
        h_lmlo = F.relu(self.fc1_lmlo(h[VIEWS.L_MLO]))
        h_rmlo = F.relu(self.fc1_rmlo(h[VIEWS.R_MLO]))

        # 2개의 FC 레이어를 생성한 후 2개의 softmax 결과를 반환
        h_lcc = self.output_layer_lcc(h_lcc) # (scalar, 2)
        h_rcc = self.output_layer_rcc(h_rcc)
        h_lmlo = self.output_layer_lmlo(h_lmlo)
        h_rmlo = self.output_layer_rmlo(h_rmlo)

        h = {
            VIEWS.L_CC: h_lcc, # (scalar, 2)
            VIEWS.R_CC: h_rcc,
            VIEWS.L_MLO: h_lmlo,
            VIEWS.R_MLO: h_rmlo,
        }

        return h

class FourViewResNet(nn.Module):
    def __init__(self, input_channels, args):
        super(FourViewResNet, self).__init__()

        self.args = args
        self.cc = initialize_model(model_name=self.args.backbone, use_pretrained=self.args.use_pretrained, input_channels = input_channels, num_classes=args)
        self.mlo = initialize_model(model_name=self.args.backbone, use_pretrained=self.args.use_pretrained, input_channels=input_channels, num_classes=args)

        self.model_dict = {}
        self.model_dict[VIEWS.L_CC] = self.l_cc = self.cc
        self.model_dict[VIEWS.L_MLO] = self.l_mlo = self.mlo
        self.model_dict[VIEWS.R_CC] = self.r_cc = self.cc
        self.model_dict[VIEWS.R_MLO] = self.r_mlo = self.mlo

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.model_dict[view](single_x)