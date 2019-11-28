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
import src.modeling.layers as layers
from src.constants import VIEWS, VIEWANGLES

class ImageBreastModel(nn.Module):
    def __init__(self, args, input_channels):
        super(ImageBreastModel, self).__init__()

        self.args = args
        self.four_view_resnet = FourViewResNet(input_channels)

        self.fc1_lcc = nn.Linear(256, 256)
        self.fc1_rcc = nn.Linear(256, 256)
        self.fc1_lmlo = nn.Linear(256, 256)
        self.fc1_rmlo = nn.Linear(256, 256)
        self.output_layer_lcc = layers.OutputLayer(256, (4, 2))
        self.output_layer_rcc = layers.OutputLayer(256, (4, 2))
        self.output_layer_lmlo = layers.OutputLayer(256, (4, 2))
        self.output_layer_rmlo = layers.OutputLayer(256, (4, 2))

        self.all_views_avg_pool = layers.AllViewsAvgPool()
        self.all_views_gaussian_noise_layer = layers.AllViewsGaussianNoise(0.01)

    def forward(self, x):
        h = self.all_views_gaussian_noise_layer(x)
        result = self.four_view_resnet(h)
        h = self.all_views_avg_pool(result)

        h_lcc = F.relu(self.fc1_lcc(h[VIEWS.L_CC]))
        h_rcc = F.relu(self.fc1_rcc(h[VIEWS.R_CC]))
        h_lmlo = F.relu(self.fc1_lmlo(h[VIEWS.L_MLO]))
        h_rmlo = F.relu(self.fc1_rmlo(h[VIEWS.R_MLO]))

        h_lcc = self.output_layer_lcc(h_lcc)
        h_rcc = self.output_layer_rcc(h_rcc)
        h_lmlo = self.output_layer_lmlo(h_lmlo)
        h_rmlo = self.output_layer_rmlo(h_rmlo)

        h = {
            VIEWS.L_CC: h_lcc,
            VIEWS.R_CC: h_rcc,
            VIEWS.L_MLO: h_lmlo,
            VIEWS.R_MLO: h_rmlo,
        }

        return h

class FourViewResNet(nn.Module):
    def __init__(self, input_channels):
        super(FourViewResNet, self).__init__()

        self.cc = initialize_model(model_name=self.args.backbone, use_pretrained=self.args.use_pretrained)

        self.cc = resnet22(input_channels)
        self.mlo = resnet22(input_channels)
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