import sys
sys.path.append(".")
import os
# from pathlib import Path
# import pickle
import numpy as np
import torch
from src.datamodules.forecast.h5dataset import H5Dataset
# from h5datamodule import ForecastDataModule
from src.models.forecast.forecast_module import ForecastLitModule
from src.evaluate.inference import autoregressive_inference
# import matplotlib as plt
import argparse
import torchdata.datapipes as dp
from torchvision.transforms import transforms

VARIABLES = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation",
            "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200",
            "geopotential_250", "geopotential_300", "geopotential_400", "geopotential_500",
            "geopotential_600", "geopotential_700", "geopotential_850", "geopotential_925", "geopotential_1000",
            "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200",
            "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500",
            "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", "u_component_of_wind_1000",
            "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200",
            "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500",
            "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", "v_component_of_wind_1000",
            "temperature_50", "temperature_100", "temperature_150", "temperature_200",
            "temperature_250", "temperature_300", "temperature_400", "temperature_500",
            "temperature_600", "temperature_700", "temperature_850", "temperature_925", "temperature_1000",
            "relative_humidity_50", "relative_humidity_100", "relative_humidity_150", "relative_humidity_200",
            "relative_humidity_250", "relative_humidity_300", "relative_humidity_400", "relative_humidity_500",
            "relative_humidity_600", "relative_humidity_700", "relative_humidity_850", "relative_humidity_925", "relative_humidity_1000",]

def get_normalize(data_dir, variables):
    normalize_mean = dict(np.load(os.path.join(data_dir, "normalize_mean.npz")))
    mean = []
    for var in variables:
        if var != "total_precipitation":
            mean.append(normalize_mean[var])
        else:
            mean.append(np.array([0.0]))
    normalize_mean = np.concatenate(mean)
    normalize_std = dict(np.load(os.path.join(data_dir, "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[var] for var in variables])
    data_transforms = transforms.Normalize(normalize_mean, normalize_std)
    return data_transforms

def forecast_model_inference(data_dir,
                            start_idx,
                            end_idx,
                            pretrain_ckpt,
                            output_dir,
                            forecast_hours,
                            decorrelation_hours,
                            mode,
                            model_name,
                            device,
                            ia):

    decorrelation_hours = decorrelation_hours

    forecast_model = ForecastLitModule.load_from_checkpoint(f"{pretrain_ckpt}/1.ckpt")
    forecast_net = forecast_model.net.to(device).eval()

    mult = forecast_model.mult
    clim = forecast_model.clims[2]

    eval_dataset = H5Dataset(root_dir=data_dir,
                            mode=mode,
                            file_list=list(dp.iter.FileLister(os.path.join(data_dir, mode))),
                            start_idx=start_idx,
                            end_idx=end_idx,
                            variables=VARIABLES,
                            out_variables=VARIABLES,
                            max_predict_ranges=forecast_hours,
                            transforms=get_normalize(data_dir, VARIABLES),
                            output_transforms=get_normalize(data_dir, VARIABLES)
                            )
#    print(eval_dataset) 
    
                        
 #   np.save(f"{output_dir}/evaldata.npy", np.asarray(eval_dataset, dtype = object))

    # 取初始场
    n_samples = eval_dataset.__len__()
    stop = n_samples
    ics = np.arange(0, stop, decorrelation_hours)

    val_forecsat, val_real, val_rmse, val_acc = [], [], [], []

    for i, ic in enumerate(ics):
        seq_real, seq_forecsat, seq_rmse, seq_acc = autoregressive_inference(ic,
                                                                          eval_dataset,
                                                                          forecast_net,
                                                                          forecast_hours,
                                                                          clim.to(device),
                                                                          mult.to(device),
                                                                          VARIABLES,
                                                                          device)
        if i == 0:
            val_real = seq_real
            val_forecsat = seq_forecsat
            val_rmse = seq_rmse
            val_acc = seq_acc
        else:
            if seq_real is not None:
                val_real = np.concatenate((val_real, seq_real), axis=0)
                val_forecsat = np.concatenate((val_forecsat, seq_forecsat), axis=0)
                val_rmse = np.concatenate((val_rmse, seq_rmse), axis=0)
                val_acc = np.concatenate((val_acc, seq_acc), axis=0)

    for i in range(val_rmse.shape[-1]):
        print(f"RMSE of {VARIABLES[i]} is： {np.mean(val_rmse, axis=0)[:, i]}")
        print(f"ACC of {VARIABLES[i]} is： {np.mean(val_acc, axis=0)[:, i]}")

    np.save(f"{output_dir}/rmse_{model_name}.npy", val_rmse)
    np.save(f"{output_dir}/acc_{model_name}.npy", val_acc)
    np.save(f"{output_dir}/forcast_{model_name}_"+str(ia)+".npy", val_forecsat)

def prepare_parser():
    parser = argparse.ArgumentParser(description='Inference for prediction and assimilation loop!')

    parser.add_argument(
        '--data_dir',
        type=str,
        help='path of the validation data',
        default="/HOME/scz0rca/run/1/1"
    )

    parser.add_argument(
        '--start_idx',
        type=float,
        help='start index of the dataset',
        default=0
    )

    parser.add_argument(
        '--end_idx',
        type=float,
        help='end index of the dataset',
        default=1
    )

    parser.add_argument(
        '--pretrain_dir',
        type=str,
        help='path for pretrain prediction models',
        default='/HOME/scz0rca/run/AIEnKF/AIEnKF'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='path for output',
        default='/HOME/scz0rca/run/output'
    )

    parser.add_argument(
        '--forecast_hours',
        type=int,
        help='length of the assimilation window [d]',
        default=48
    )

    parser.add_argument(
        '--decorrelation_hours',
        type=int,
        help='decoorelation between each initial time [d]',
        default=100000
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='mode of data',
        default='test'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='method used to do assimilation',
        default='afnonet'
    )

    return parser


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    start_idx = args.start_idx
    end_idx = args.end_idx
    pretrain_ckpt = args.pretrain_dir
    output_dir = args.output_dir
    forecast_hours = args.forecast_hours
    decorrelation_hours = args.decorrelation_hours
    mode = args.mode
    model_name = args.model_name
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    for ia in range(1):
        forecast_model_inference(data_dir,
                                start_idx,
                                end_idx,
                                pretrain_ckpt,
                                output_dir,
                                forecast_hours,
                                decorrelation_hours,
                                mode,
                                model_name,
                                device,
                                ia)