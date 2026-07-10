import math
import os
from typing import Any, Dict, Optional, Tuple
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

class H5Dataset(Dataset):
    def __init__(self, 
                root_dir,
                mode,
                file_list,
                start_idx,
                end_idx,
                variables,
                out_variables,
                max_predict_ranges,
                transforms,
                output_transforms,
        ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.max_predict_ranges = max_predict_ranges
        self._get_files_stats()
        for i in range(self.n_shard):
            self.h5dataset[i] = self._open_file(i)
        self.transforms = transforms
        self.output_transforms = output_transforms

    def _get_files_stats(self):
        self.n_shard = len(self.file_list)
        with h5py.File(self.file_list[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.file_list[0]))
            self.n_samples_per_shard = _f[self.variables[0]].shape[0]
            self.in_chans = len(self.variables)
            self.shape_x = _f[self.variables[0]].shape[2]
            self.shape_y = _f[self.variables[0]].shape[3]

        self.n_samples_total = self.n_shard * self.n_samples_per_shard
        self.h5dataset = [None for _ in range(self.n_shard)]
        logging.info("Number of samples per shard: {}".format(self.n_samples_per_shard))
        logging.info("Number of examples: {}. Fields Shape: {} x {} x {}".format(self.n_samples_total, self.in_chans, self.shape_x, self.shape_y))

    def _open_file(self, shard_idx):
        _file = h5py.File(self.file_list[shard_idx], 'r')
        return _file

    def __len__(self):
        return self.n_samples_total - self.max_predict_ranges

    def __getitem__(self, global_idx):
        shard_idx = int(global_idx / self.n_samples_per_shard)
        local_idx = int(global_idx % self.n_samples_per_shard)

        if self.h5dataset[shard_idx] is None:
            self.h5dataset[shard_idx] = self._open_file(shard_idx)

        lead_times = 6 * np.ones(shape=1)

        if local_idx >= self.n_samples_per_shard - int(lead_times[0]) - 1:
            local_idx = self.n_samples_per_shard - int(lead_times[0]) - 1

        output_shard_idx = shard_idx
        output_local_idx = local_idx + int(lead_times[0])

        if self.h5dataset[output_shard_idx] is None:
            self.h5dataset[output_shard_idx] = self._open_file(output_shard_idx)

        inputs = torch.from_numpy(np.concatenate([self.h5dataset[shard_idx][k][local_idx] for k in self.variables], axis=0).astype(np.float32))
        targets = torch.from_numpy(np.concatenate([self.h5dataset[output_shard_idx][k][output_local_idx] for k in self.out_variables], axis=0).astype(np.float32))
        
        return self.transforms(inputs), self.output_transforms(targets), self.variables, self.out_variables