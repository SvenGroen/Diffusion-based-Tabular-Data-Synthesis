import unittest
import pandas as pd
import numpy as np
import os
import torch
from .utils import ProcessorFactory as PF
import pytest
import os


class TestFTProcessor(unittest.TestCase):

    # def test_example(self):
    #     assert True
    
    def setUp(self):
        self.processor = PF.get_instance("ft")
        self.x_cat, self.x_num, self.y = PF.get_data_sample(100)

    @pytest.mark.dependency()
    def test_init(self):
        assert self.processor is not None
        assert self.processor.d_numerical >= 0
    

    @pytest.mark.dependency(name="test_init")
    def test_fit(self):
        assert self.processor._was_fit == False
        self.processor.fit()
        assert self.processor._was_fit == True

    # # @pytest.mark.dependency(depends=["test_fit"])
    def test_transform(self):
        self.processor.fit()
        x_cat, x_num, y = self.processor.transform(
            x_cat=self.x_cat,
            x_num=self.x_num,
            y=self.y
            )
        assert x_cat is None
        assert x_num is not None
        assert y is not None
        embed_dim = self.processor.tokenizer.d_token
        assert x_num.shape[1] == self.x_num.shape[1] * embed_dim + self.x_cat.shape[1] * embed_dim


    # @pytest.mark.dependency(depends=["test_transform"])
    def test_inverse_transform(self):
        self.processor.fit()
        x_cat, x_num, y = self.processor.transform(
            x_cat=self.x_cat,
            x_num=self.x_num,
            y=self.y
            )
        x_cat_new, x_num_new, y_new = self.processor.inverse_transform(
            x_cat=x_cat,
            x_num=x_num,
            y_pred=y
            )
        for old, new in [(self.x_cat, x_cat_new), (self.x_num, x_num_new), (self.y, y_new)]:
            assert new is not None
            assert old.shape == new.shape
            self.assertAlmostEqual((old-new).sum(), 0, places=5)



        



