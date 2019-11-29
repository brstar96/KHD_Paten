# Copyright (C) 2019 Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, 
#   Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, 
#   Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, 
#   Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, 
#   Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, 
#   Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of src.
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
Defines constants used in src.
"""


class VIEWS:
    LMLO = "L-CC"
    RMLO = "R-CC"
    LCC = "L-MLO"
    RCC = "R-MLO"

    LIST = [LMLO, RMLO, LCC, RCC]

    @classmethod
    def is_cc(cls, view):
        return view in (cls.LMLO, cls.RMLO)

    @classmethod
    def is_mlo(cls, view):
        return view in (cls.LCC, cls.RCC)

    @classmethod
    def is_left(cls, view):
        return view in (cls.LMLO, cls.LCC)

    @classmethod
    def is_right(cls, view):
        return view in (cls.RMLO, cls.RCC)


class VIEWANGLES:
    CC = "CC"
    MLO = "MLO"

    LIST = [CC, MLO]


class LABELS:
    LEFT_BENIGN = "left_benign"
    RIGHT_BENIGN = "right_benign"
    LEFT_MALIGNANT = "left_malignant"
    RIGHT_MALIGNANT = "right_malignant"

    LIST = [LEFT_BENIGN, RIGHT_BENIGN, LEFT_MALIGNANT, RIGHT_MALIGNANT]


class MODELMODES:
    VIEW_SPLIT = "view_split"
    IMAGE = "image"


INPUT_SIZE_DICT = {
    VIEWS.LMLO: (2677, 1942),
    VIEWS.RMLO: (2677, 1942),
    VIEWS.LCC: (2974, 1748),
    VIEWS.RCC: (2974, 1748),
}
