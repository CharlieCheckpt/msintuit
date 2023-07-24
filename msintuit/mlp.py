# Copyright (c) Owkin, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class MLP(torch.nn.Sequential):
    """
    MLP Module

    Parameters
    ----------
    in_features: int
    out_features: int
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        d_model = in_features
        layers = []

        activation = torch.nn.Sigmoid()
        for h in [128, 64]:
            seq = [torch.nn.Linear(d_model, h, bias=True)]
            d_model = h

            if activation is not None:
                seq.append(activation)

            layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)
