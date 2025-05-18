import os
import torch
from models import mamba_mtp
import darts.models
from functools import partial

class Exp_Basic():

    def __init__(self, args):

        self.args = args

        self.model_dict = {
            'Mamba_MTP': mamba_mtp,
        }

        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
    
    def train(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass

    def _get_data(self):
        pass