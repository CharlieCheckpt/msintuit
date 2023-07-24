# Copyright (c) Owkin, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from typing import Optional

from msintuit.tile_layers import TilesMLP
from msintuit.mlp import MLP
from msintuit.extreme_layer import ExtremeLayer


class MSIntuitChowder(torch.nn.Module):
    """
    MSIntuit Chowder module.
    See https://arxiv.org/abs/1802.02212.

    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
    out_features: int
        controls the number of scores and, by extension, the number of out_features
    n_extreme: int
        number of extreme tiles
    """

    def __init__(
        self,
        in_features: int,
        n_extreme: int,
    ):
        super(MSIntuitChowder, self).__init__()

        self.score_model = TilesMLP(in_features, out_features=1)
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_extreme, n_bottom=n_extreme)

        self.mlp = MLP(2*n_extreme, 1)

        self.mlp.apply(self.weight_initialization)

    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        """
        scores = self.score_model(x=x, mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, 2*N_EXTREME, OUT_FEATURES)

        # Apply MLP to the 2*N_EXTREME scores
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        return y.squeeze(2), extreme_scores
