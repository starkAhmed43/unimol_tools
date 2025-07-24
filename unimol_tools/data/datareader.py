# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm
from multiprocessing import Pool
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..utils import logger


class MolDataReader(object):
    '''A class to read Mol Data.'''

    def read_data(self, data=None, is_train=True, **params):
        logger.info("Starting read_data")
        # TO DO
        # 1. add anomaly detection & outlier removal.
        # 2. add support for other file format.
        # 3. add support for multi tasks.

        """
        Reads and preprocesses molecular data from various input formats for model training or prediction.
        Parsing target columns
        1. if target_cols is not None, use target_cols as target columns.
        2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        3. use given target_cols as target columns placeholder with value -1.0 for predict

        :param data: The input molecular data. Can be a file path (str), a dictionary, or a list of SMILES strings.
        :param is_train: (bool) A flag indicating if the operation is for training. Determines data processing steps.
        :param params: A dictionary of additional parameters for data processing.

        :return: A dictionary containing processed data and related information for model consumption.
        :raises ValueError: If the input data type is not supported or if any SMILES string is invalid (when strict).
        """
        task = params.get('task', None)
        target_cols = params.get('target_cols', None)
        smiles_col = params.get('smiles_col', 'SMILES')
        target_col_prefix = params.get('target_col_prefix', 'TARGET')
        anomaly_clean = params.get('anomaly_clean', False)
        smi_strict = params.get('smi_strict', False)
        split_group_col = params.get('split_group_col', 'scaffold')

        if isinstance(data, str):
            logger.info(f"Loading data from file: {data}")
            # load from file
            self.data_path = data
            if data.endswith('.sdf'):
                logger.info("Detected SDF file format")
                # load sdf file
                data = PandasTools.LoadSDF(data)
                data = self._convert_numeric_columns(data)
            elif data.endswith('.csv'):
                logger.info("Detected CSV file format")
                data = pd.read_csv(self.data_path)
            else:
                logger.error(f"Unknown file type: {data}")
                raise ValueError('Unknown file type: {}'.format(data))
        elif isinstance(data, dict):
            logger.info("Loading data from dictionary")
            # load from dict
            if 'target' in data:
                label = np.array(data['target'])
                if len(label.shape) == 1 or label.shape[1] == 1:
                    data[target_col_prefix] = label.reshape(-1)
                else:
                    for i in range(label.shape[1]):
                        data[target_col_prefix + str(i)] = label[:, i]

            _ = data.pop('target', None)
            data = pd.DataFrame(data).rename(columns={smiles_col: 'SMILES'})

        elif isinstance(data, list) or isinstance(data, np.ndarray):
            logger.info("Loading data from SMILES list or numpy array")
            # load from smiles list
            data = pd.DataFrame(data, columns=['SMILES'])
        else:
            logger.error(f"Unknown data type: {type(data)}")
            raise ValueError('Unknown data type: {}'.format(type(data)))

        #### parsing target columns
        #### 1. if target_cols is not None, use target_cols as target columns.
        #### 2. if target_cols is None, use all columns with prefix 'target_col_prefix' as target columns.
        #### 3. use given target_cols as target columns placeholder with value -1.0 for predict
        if task == 'repr':
            logger.info("Task is 'repr', skipping target column parsing")
            # placeholder for repr task
            targets = None
            target_cols = None
            num_classes = None
            multiclass_cnt = None
        else:
            logger.info(f"Parsing target columns for task: {task}")
            if target_cols is None:
                target_cols = [
                    item for item in data.columns if item.startswith(target_col_prefix)
                ]
            elif isinstance(target_cols, str):
                target_cols = [target_col.strip() for target_col in target_cols.split(',')]
            elif isinstance(target_cols, list):
                pass
            else:
                raise ValueError(
                    'Unknown target_cols type: {}'.format(type(target_cols))
                )

            if is_train:
                logger.info("Training mode: anomaly_clean=%s", anomaly_clean)
                if anomaly_clean:
                    data = self.anomaly_clean(data, task, target_cols)
                if task == 'multiclass':
                    logger.info("Multiclass task detected")
                    multiclass_cnt = int(data[target_cols].max() + 1)
            else:
                logger.info("Prediction mode: filling missing target columns with -1.0")
                for col in target_cols:
                    if col not in data.columns or data[col].isnull().any():
                        data[col] = -1.0

            targets = data[target_cols].values.tolist()
            num_classes = len(target_cols)

        logger.info("Preparing output dictionary")
        dd = {
            'raw_data': data,
            'raw_target': targets,
            'num_classes': num_classes,
            'target_cols': target_cols,
            'multiclass_cnt': (
                multiclass_cnt if task == 'multiclass' and is_train else None
            ),
        }
        if smiles_col in data.columns:
            logger.info(f"Checking SMILES validity in column: {smiles_col}")
            """mask = data[smiles_col].apply(
                lambda smi: self.check_smiles(smi, is_train, smi_strict)
            )
            data = data[mask]"""
            dd['smiles'] = data[smiles_col].tolist()
            with Pool(processes=os.cpu_count()) as pool:
                smiles_list = data[smiles_col].tolist()
                logger.info(f"Scaffolding {len(smiles_list)} SMILES strings")
                dd['scaffolds'] = list(
                    tqdm(pool.imap(self.smi2scaffold, smiles_list), total=len(smiles_list), desc="Generating scaffolds")
                )
        else:
            logger.warning(f"SMILES column '{smiles_col}' not found in data")
            dd['smiles'] = None
            dd['scaffolds'] = None

        if split_group_col in data.columns:
            logger.info(f"Using split group column: {split_group_col}")
            dd['group'] = data[split_group_col].tolist()
        elif split_group_col == 'scaffold':
            logger.info("Using scaffolds as group")
            dd['group'] = dd['scaffolds']
        else:
            logger.warning(f"Split group column '{split_group_col}' not found")
            dd['group'] = None

        if 'atoms' in data.columns and 'coordinates' in data.columns:
            logger.info("Found 'atoms' and 'coordinates' columns")
            dd['atoms'] = data['atoms'].tolist()
            dd['coordinates'] = data['coordinates'].tolist()

        if 'ROMol' in data.columns:
            logger.info("Found 'ROMol' column")
            dd['mols'] = data['ROMol'].tolist()

        logger.info("Finished read_data")
        return dd

    def check_smiles(self, smi, is_train, smi_strict):
        logger.debug(f"Checking SMILES: {smi}")
        """
        Validates a SMILES string and decides whether it should be included based on training mode and strictness.

        :param smi: (str) The SMILES string to check.
        :param is_train: (bool) Indicates if this check is happening during training.
        :param smi_strict: (bool) If true, invalid SMILES strings raise an error, otherwise they're logged and skipped.

        :return: (bool) True if the SMILES string is valid, False otherwise.
        :raises ValueError: If the SMILES string is invalid and strict mode is on.
        """
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                logger.info(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True

    def smi2scaffold(self, smi):
        logger.debug(f"Converting SMILES to scaffold: {smi}")
        """
        Converts a SMILES string to its corresponding scaffold.

        :param smi: (str) The SMILES string to convert.

        :return: (str) The scaffold of the SMILES string, or the original SMILES if conversion fails.
        """
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=True
            )
        except:
            return smi

    def anomaly_clean(self, data, task, target_cols):
        logger.info(f"Performing anomaly cleaning for task: {task}")
        """
        Performs anomaly cleaning on the data based on the specified task.

        :param data: (DataFrame) The dataset to be cleaned.
        :param task: (str) The type of task which determines the cleaning strategy.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset.
        :raises ValueError: If the provided task is not recognized.
        """
        if task in [
            'classification',
            'multiclass',
            'multilabel_classification',
            'multilabel_regression',
        ]:
            return data
        if task == 'regression':
            return self.anomaly_clean_regression(data, target_cols)
        else:
            raise ValueError('Unknown task: {}'.format(task))

    def anomaly_clean_regression(self, data, target_cols):
        logger.info(f"Performing regression anomaly cleaning for columns: {target_cols}")
        """
        Performs anomaly cleaning specifically for regression tasks using a 3-sigma threshold.

        :param data: (DataFrame) The dataset to be cleaned.
        :param target_cols: (list) The list of target columns to consider for cleaning.

        :return: (DataFrame) The cleaned dataset after applying the 3-sigma rule.
        """
        sz = data.shape[0]
        target_col = target_cols[0]
        _mean, _std = data[target_col].mean(), data[target_col].std()
        data = data[
            (data[target_col] > _mean - 3 * _std)
            & (data[target_col] < _mean + 3 * _std)
        ]
        logger.info(
            'Anomaly clean with 3 sigma threshold: {} -> {}'.format(sz, data.shape[0])
        )
        return data

    def _convert_numeric_columns(self, df):
        logger.info("Converting DataFrame columns to numeric types")
        """
        Try to convert all columns in the DataFrame to numeric types, except for the 'ROMol' column.
        
        :param df: DataFrame to be converted.
        :return: DataFrame with numeric columns.
        """
        for col in df.columns:
            if col == 'ROMol': 
                continue
                
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.isna().sum() / len(df) < 0.1:  # Allow up to 10% NaN values
                    df[col] = numeric_series
                    logger.debug(f"Column '{col}' converted to numeric type")
            except:
                pass
                
        return df