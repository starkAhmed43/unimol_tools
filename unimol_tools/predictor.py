# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import logger
from .data import DataHub
from .models import UniMolModel, UniMolV2Model
from .tasks import Trainer


class MolDataset(Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """

    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class UniMolRepr(object):
    """
    A :class:`UniMolRepr` class is responsible for interface of molecular representation by unimol
    """

    def __init__(
        self,
        data_type='molecule',
        batch_size=32,
        remove_hs=False,
        model_name='unimolv1',
        model_size='84m',
        use_cuda=True,
        use_ddp=False,
        use_gpu='all',
        save_path=None,
        **kwargs,
    ):
        """
        Initialize a :class:`UniMolRepr` class.

        :param data_type: str, default='molecule', currently support molecule, oled.
        :param batch_size: int, default=32, batch size for training.
        :param remove_hs: bool, default=False, whether to remove hydrogens in molecular.
        :param model_name: str, default='unimolv1', currently support unimolv1, unimolv2.
        :param model_size: str, default='84m', model size of unimolv2. Avaliable: 84m, 164m, 310m, 570m, 1.1B.
        :param use_cuda: bool, default=True, whether to use gpu.
        :param use_ddp: bool, default=False, whether to use distributed data parallel.
        :param use_gpu: str, default='all', which gpu to use.
        """
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        logger.info(f"Initializing UniMolRepr with model_name={model_name}, model_size={model_size}, device={self.device}")
        if model_name == 'unimolv1':
            self.model = UniMolModel(
                output_dim=1, data_type=data_type, remove_hs=remove_hs
            ).to(self.device)
        elif model_name == 'unimolv2':
            self.model = UniMolV2Model(output_dim=1, model_size=model_size).to(
                self.device
            )
        else:
            logger.error(f"Unknown model name: {model_name}")
            raise ValueError('Unknown model name: {}'.format(model_name))
        self.model.eval()
        self.params = {
            'data_type': data_type,
            'batch_size': batch_size,
            'remove_hs': remove_hs,
            'model_name': model_name,
            'model_size': model_size,
            'use_cuda': use_cuda,
            'use_ddp': use_ddp,
            'use_gpu': use_gpu,
            'save_path': save_path,
        }
        logger.info(f"UniMolRepr initialized with params: {self.params}")

    def get_repr(self, data=None, return_atomic_reprs=False, return_tensor=False):
        """
        Get molecular representation by unimol.

        :param data: str, dict or list, default=None, input data for unimol.

            - str: smiles string or path to a smiles file.

            - dict: custom conformers, should take atoms and coordinates as input.

            - list: list of smiles strings.

        :param return_atomic_reprs: bool, default=False, whether to return atomic representations.

        :return: dict of molecular representation.
        """

        logger.info(f"get_repr called with data type: {type(data)}")
        if isinstance(data, str):
            if data.endswith('.sdf'):
                logger.info("Input data is an SDF file.")
                # Datahub will process sdf file.
                pass
            elif data.endswith('.csv'):
                logger.info("Input data is a CSV file.")
                data = pd.read_csv(data)
                assert 'SMILES' in data.columns
                data = data['SMILES'].values
            else:
                logger.info("Input data is a single SMILES string.")
                data = [data]
                data = np.array(data)
        elif isinstance(data, dict):
            logger.info("Input data is a custom conformer dict.")
            # custom conformers, should take atoms and coordinates as input.
            assert 'atoms' in data and 'coordinates' in data
        elif isinstance(data, list):
            logger.info("Input data is a list of SMILES strings.")
            # list of smiles strings.
            assert isinstance(data[-1], str)
            data = np.array(data)
        else:
            logger.error(f"Unknown data type: {type(data)}")
            raise ValueError('Unknown data type: {}'.format(type(data)))

        logger.info("Creating DataHub and MolDataset for inference.")
        datahub = DataHub(
            data=data,
            task='repr',
            is_train=False,
            **self.params,
        )
        dataset = MolDataset(datahub.data['unimol_input'])
        self.trainer = Trainer(task='repr', **self.params)
        logger.info("Starting model inference.")
        repr_output = self.trainer.inference(
            model=self.model,
            dataset=dataset,
            model_name=self.params['model_name'],
            return_repr=True,
            return_atomic_reprs=return_atomic_reprs,
            return_tensor=return_tensor,
        )
        logger.info("Inference completed.")
        return repr_output
