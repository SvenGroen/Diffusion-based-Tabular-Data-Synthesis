'''
# adopted from https://github.com/pfnet-research/CSDI_T/blob/d7655578d51b062fefb16656ba635478b458c92d/src/main_model_table_ft.py
# who partially uses work from https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py
'''

import torch
import torch.nn as nn
import torch.nn.init as nn_init

import numpy as np
from torch import Tensor
import math
import typing as ty

class Tokenizer(nn.Module):
    """
    A transformer-based neural network for tabular data with categorical and numerical features. It performs tokenization
    of categorical features and creates embeddings of them to be used as inputs to the model, in combination with the numerical
    features.

    Attributes:
    ----------
    d_numerical: int
        The number of numerical input features
    categories: Optional[List[int]]
        A list containing the number of categories for each categorical feature
    d_token: int
        The embedding dimension
    bias: bool
        If True, a bias term will be added to the embedding layer

    Methods:
    -------
    n_tokens:
        Returns the total number of tokens used in the model
    forward(x_num, x_cat=None):
        Forward pass of the transformer model
    recover(Batch, d_numerical):
        Inverse transformation of the input data to recover the original data before tokenization
    """
    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        """
        Initializes the Tokenizer object.

        Args:
        -----
        d_numerical: int
            The number of numerical input features
        categories: Optional[List[int]]
            A list containing the number of categories for each categorical feature
        d_token: int
            The embedding dimension
        bias: bool
            If True, a bias term will be added to the embedding layer
        """
        super().__init__()

        d_bias = d_numerical + len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.d_token = d_token
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories) + 1, self.d_token)
        self.category_embeddings.weight.requires_grad = False
        nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.weight = nn.Parameter(Tensor(d_numerical, self.d_token))
        self.weight.requires_grad = False # static embedding

        self.bias = nn.Parameter(Tensor(d_bias, self.d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            self.bias.requires_grad = False # static embedding

    @property
    def n_tokens(self) -> int:
        """
        Returns the total number of tokens used in the model.

        Returns:
        -------
        int:
            The number of tokens used in the model
        """
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        """
        Forward pass of the transformer model.

        Args:
        -----
        x_num: Tensor
            The numerical input data
        x_cat: Optional[Tensor]
            The categorical input data (optional)

        Returns:
        -------
        Tensor:
            The transformed input data, ready to be used as inputs to the transformer model
        """
        x_some = x_num if x_cat is None else x_cat
        x_cat = x_cat.type(torch.int32)

        assert x_some is not None
        x = self.weight.T * x_num # x_num maybe very large --> normalization later enough?

        if x_cat is not None:
            x = x[:, np.newaxis, :, :]
            x = x.permute(0, 1, 3, 2)
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=2,
            )
        if self.bias is not None:
            x = x + self.bias[None]

        return x

    def recover(self, Batch: Tensor, d_numerical: int) -> ty.Tuple[Tensor, Tensor]:
        """
        Recover the original data from the tokenized batch.

        Args:
            Batch (Tensor): The tokenized batch.
            d_numerical (int): The number of numerical features.

        Returns:
            A tuple containing:
                - new_Batch_cat (Tensor): The recovered categorical features tensor.
                - Batch_numerical (Tensor): The recovered numerical features tensor.
        """
        B, L, K = Batch.shape
        L_new = int(L / self.d_token)
        Batch = Batch.reshape(B, L_new, self.d_token)
        Batch = Batch - self.bias

        Batch_numerical = Batch[:, :d_numerical, :]
        Batch_numerical = Batch_numerical / self.weight
        Batch_numerical = torch.mean(Batch_numerical, 2, keepdim=False)

        Batch_cat = Batch[:, d_numerical:, :]
        new_Batch_cat = torch.zeros([Batch_cat.shape[0], Batch_cat.shape[1]])
        for i in range(Batch_cat.shape[1]):
            token_start = self.category_offsets[i] + 1
            if i == Batch_cat.shape[1] - 1:
                token_end = self.category_embeddings.weight.shape[0] - 1
            else:
                token_end = self.category_offsets[i + 1]
            emb_vec = self.category_embeddings.weight[token_start : token_end + 1, :]
            for j in range(Batch_cat.shape[0]):
                distance = torch.norm(emb_vec - Batch_cat[j, i, :], dim=1)
                nearest = torch.argmin(distance)
                new_Batch_cat[j, i] = nearest + 1
            new_Batch_cat = new_Batch_cat.to(Batch_numerical.device)
        return  new_Batch_cat, Batch_numerical #changed from concatenate version to separate numerical and categorical