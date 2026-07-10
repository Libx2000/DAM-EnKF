import torch
import numpy as np
import time
import xarray as xr
import torch
from src.utils.weighted_acc_rmse import weighted_acc_torch, weighted_rmse_torch

def autoregressive_inference(ic, val_dataset, module, prediction_length, clim, mult, variabels, device):
    ic = int(ic)
    prediction_length = int(prediction_length) // 6
    clim = clim
    mult = mult

    seq_pred = torch.zeros((1 + prediction_length, 69, 64, 128)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((1 + prediction_length, 69, 64, 128)).to(device, dtype=torch.float32)
    seq_rmse = torch.zeros((1 + prediction_length, 69)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((1 + prediction_length, 69)).to(device, dtype=torch.float32)
    n_samples_per_shard = val_dataset.n_samples_per_shard

    #np.save(f"/HOME/scz0rca/run/output/forcast_"+str(ia)+".npy", seq_pred)
    
    if (ic % n_samples_per_shard + prediction_length) >= n_samples_per_shard:
        return None, None, None, None
    else:
        valid_data_all = [val_dataset.__getitem__(ic+i) for i in range(0, prediction_length+1)]
        # standardize
        init_data = torch.as_tensor(valid_data_all[0][0]).to(device, dtype=torch.float32)

        with torch.no_grad():
            for i in range(1 + prediction_length):
                # 从ic开始
                if i == 0:  # start of sequence
                    seq_real[i:i + 1] = init_data
                    seq_pred[i:i + 1] = init_data
                else:
                    seq_real[i:i + 1] = torch.as_tensor(valid_data_all[i][0])
                    # Switch the input back to the stored input
                    seq_pred[i:i+1] = module(seq_pred[i-1:i],
                                             variabels,
                                            variabels)

                seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
                seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

        seq_pred = seq_pred.cpu().detach().numpy()
        seq_real = seq_real.cpu().detach().numpy()
        seq_rmse = seq_rmse.cpu().detach().numpy()
        seq_acc = seq_acc.cpu().detach().numpy()

        return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)