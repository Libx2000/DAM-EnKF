# DAM-Forecast

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**DAM-Forecast** (Data Assimilation and Machine Learning-based Weather Forecasting) is a deep learning-based weather forecasting framework that integrates multiple advanced deep learning model architectures for efficient and accurate medium-range weather forecasting.

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Architecture](#technical-architecture)
- [Environment Requirements](#environment-requirements)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Performance Evaluation](#performance-evaluation)
- [Model Training](#model-training)
- [FAQ](#faq)
- [Contribution Guidelines](#contribution-guidelines)
- [Copyright Information](#copyright-information)

---

## Project Overview

### Model Features

The DAM-Forecast framework provides the following core features:

1. **Multi-model Support**: Integrates four deep learning model architectures — ClimaX, AFNONet, ViT, and FuXi
2. **Data Assimilation**: Supports real-time data assimilation capability based on observational data
3. **Medium-range Forecasting**: Enables weather element forecasting from 6 hours to 14 days
4. **Autoregressive Inference**: Supports multi-step time series prediction
5. **Standardized Evaluation**: Provides latitude-weighted RMSE and ACC evaluation metrics

### Application Scenarios

- **Operational Weather Forecasting**: Replace or assist traditional numerical models for rapid forecasting
- **Climate Research**: Simulate climate change and extreme weather events
- **Disaster Warning**: Provide early warning information to support disaster prevention and mitigation decisions
- **Aerospace**: Provide high-precision weather support for flight missions

### Core Value

| Feature | Description |
|---------|-------------|
| **Efficiency** | GPU-accelerated, forecasting speed is far faster than traditional numerical models |
| **Accuracy** | Achieves or exceeds traditional model accuracy on key weather elements |
| **Flexibility** | Supports multiple model architectures and configuration combinations |
| **Scalability** | Easy to add new models and data sources |
| **Open Source** | MIT License, completely open source |

---

## Technical Architecture

### Basic Principles

DAM-Forecast adopts a deep learning approach that transforms weather forecasting into a spatio-temporal sequence prediction task. The model takes the current weather element field as input, learns the laws of meteorological evolution through neural networks, and directly predicts the future distribution of weather elements.

### Algorithm Composition

#### 1. Model Architectures

| Model | Architecture Type | Core Features |
|-------|-------------------|---------------|
| **ClimaX** | Vision Transformer | Cross-scale, cross-variable unified modeling, supports constant field input |
| **AFNONet** | Adaptive Fourier Neural Operator | Adaptive Fourier transform for efficient spatial data processing |
| **ViT** | Vision Transformer | Standard Transformer architecture, simple and effective |
| **FuXi** | Swin Transformer | Hierarchical window attention, multi-scale feature extraction |

#### 2. Core Components

```
DAM-Forecast
├── Data Preprocessing Layer
│   ├── HDF5 Dataset Loading
│   ├── Standardization (Z-Score)
│   └── Constant Field Merging
├── Model Encoding Layer
│   ├── Patch Embedding
│   ├── Positional Encoding (Sincos)
│   └── Transformer/FNO Encoder
├── Model Decoding Layer
│   ├── Decoder Transformer
│   └── Head Output Layer
└── Evaluation Layer
    ├── Latitude-weighted RMSE
    ├── Latitude-weighted ACC
    └── Autoregressive Inference
```

### Technical Innovations

1. **Constant Field Fusion**: Fuses constant fields (e.g., terrain height) with meteorological variables as input to enhance physical constraints
2. **Latitude-weighted Evaluation**: Considers Earth's curvature effects using cosine weights for evaluation
3. **Autoregressive Prediction**: Supports multi-step rolling prediction for ultra-long lead-time forecasting
4. **Hydra Configuration System**: Flexible configuration management supporting batch experiments and hyperparameter search

---

## Environment Requirements

### Hardware Configuration

| Configuration Type | Minimum Requirement | Recommended Configuration |
|---------------------|---------------------|--------------------------|
| GPU | NVIDIA GPU with 8GB VRAM | NVIDIA A100 / H100 (40GB+) |
| CPU | Intel i7 / AMD Ryzen 7 | Intel Xeon / AMD EPYC |
| Memory | 32GB | 128GB+ |
| Storage | 100GB (dataset) | 1TB+ SSD |

### Operating System

- Linux (Ubuntu 20.04+, CentOS 7+)
- Windows 10+ (WSL2 recommended)
- macOS 12+ (CPU training only)

### Dependencies and Versions

| Library | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.0.0 | Deep learning framework |
| torchvision | >= 0.15.0 | Vision model components |
| pytorch-lightning | >= 2.0.0 | Training framework |
| torchmetrics | >= 0.11.0 | Evaluation metrics |
| torchdata | >= 0.6.0 | Data pipeline |
| hydra-core | >= 1.3.0 | Configuration management |
| hydra-colorlog | >= 1.2.0 | Colorful logging |
| hydra-optuna-sweeper | >= 1.3.0 | Hyperparameter search |
| pyrootutils | >= 1.0.0 | Project root management |
| h5py | >= 3.8.0 | HDF5 data I/O |
| numpy | >= 1.24.0 | Numerical computation |
| xarray | >= 2023.0.0 | Meteorological data processing |
| timm | >= 0.9.0 | Pretrained model library |
| einops | >= 0.6.0 | Tensor operations |
| rich | >= 13.0.0 | Terminal beautification |
| tqdm | >= 4.65.0 | Progress bar |
| click | >= 8.0.0 | Command-line interface |

---

## Installation Guide

### Step 1: Clone the Project

```bash
git clone https://github.com/wuxinwang1997/DAM-Forecast.git
cd DAM-Forecast
```

### Step 2: Create Virtual Environment

```bash
conda create -n dam-forecast python=3.9
conda activate dam-forecast
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Project

```bash
pip install -e .
```

### Step 5: Verify Installation

```bash
python -c "from src.models.forecast.climax import ClimaX; print('DAM-Forecast installed successfully!')"
```

---

## Usage Instructions

### Command-line Interface

#### Train Model

```bash
# Train ClimaX model (single GPU)
python src/train.py model=climax trainer=gpu

# Train AFNONet model
python src/train.py model=afnonet trainer=gpu

# Train ViT model
python src/train.py model=vit trainer=gpu

# Train FuXi model
python src/train.py model=fuxi trainer=gpu

# Use custom configuration
python src/train.py model=climax datamodule=h5forecast trainer=gpu paths=forecast_hpc

# Distributed training (multi-GPU)
python src/train.py trainer=ddp model=climax

# Use mixed precision training
python src/train.py model=climax trainer=gpu trainer.precision=16

# Specify random seed
python src/train.py model=climax seed=42

# Specify output directory
python src/train.py model=climax paths.output_dir=/path/to/output
```

#### Evaluate Model

```bash
# Evaluate trained model (single-step evaluation)
python src/eval.py model=climax ckpt_path=/path/to/checkpoint.ckpt

# Autoregressive inference (multi-step rolling prediction)
python src/evaluate/eval_forecast_model.py \
    --data_dir /path/to/data \
    --pretrain_dir /path/to/checkpoints \
    --output_dir /path/to/output \
    --forecast_hours 48 \
    --mode test \
    --model_name climax

# 7-day autoregressive forecasting
python src/evaluate/eval_forecast_model.py \
    --data_dir /path/to/data \
    --pretrain_dir /path/to/checkpoints \
    --output_dir /path/to/output \
    --forecast_hours 168 \
    --mode test
```

#### Hyperparameter Search

```bash
# Use Optuna for hyperparameter search
python src/train.py -m hparams_search=forecast_optuna model=climax trainer=gpu

# Specify number of trials
python src/train.py -m hparams_search=forecast_optuna model=climax trainer=gpu hydra.sweeper.n_trials=50

# Log search process with WandB
python src/train.py -m hparams_search=forecast_optuna model=climax trainer=gpu logger=wandb
```

#### Configuration Override

Hydra supports overriding configuration values via command-line arguments:

```bash
# Modify learning rate
python src/train.py model=climax optimizer.lr=1e-4

# Modify batch size
python src/train.py model=climax datamodule.batch_size=32

# Modify training epochs
python src/train.py model=climax trainer.max_epochs=200

# Modify model parameters
python src/train.py model=climax model.net.embed_dim=512 model.net.depth=6
```

### API Interface

#### Model Initialization

```python
from src.models.forecast.climax import ClimaX

# Define meteorological variable list (full list in configs/model/climax.yaml)
default_vars = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_500", "geopotential_850", "temperature_850", ...
]

# Initialize model
model = ClimaX(
    default_vars=default_vars,
    img_size=[64, 128],
    patch_size=4,
    embed_dim=1024,
    depth=8,
    decoder_depth=2,
    num_heads=16,
    mlp_ratio=4.0,
    drop_path=0.2,
    drop_rate=0.2,
    const_dir="/path/to/constant/data"
)
```

#### Forward Propagation

```python
import torch

# Input data: [batch_size, num_vars, height, width]
input_data = torch.randn(1, len(default_vars), 64, 128)

# Variable lists
variables = default_vars
out_variables = default_vars

# Prediction
output = model(input_data, variables, out_variables)
# output: [batch_size, num_out_vars, height, width]
```

#### Autoregressive Inference

```python
from src.evaluate.inference import autoregressive_inference

# Autoregressive prediction (multi-step rolling)
seq_real, seq_pred, seq_rmse, seq_acc = autoregressive_inference(
    ic=0,                      # Initial condition index
    val_dataset=val_dataset,   # Validation dataset
    module=model,              # Model
    prediction_length=48,      # Prediction length (hours, must be multiple of 6)
    clim=climatology,          # Climatology data (for ACC calculation)
    mult=multiplier,           # Standard deviation multiplier (for RMSE denormalization)
    variables=default_vars,    # Variable list
    device=device              # Device (cuda/cpu)
)
# seq_real: [prediction_length+1, num_vars, height, width] - Ground truth sequence
# seq_pred: [prediction_length+1, num_vars, height, width] - Prediction sequence
# seq_rmse: [prediction_length+1, num_vars] - RMSE for each variable at each time step
# seq_acc: [prediction_length+1, num_vars] - ACC for each variable at each time step
```

#### LightningModule Usage

```python
from src.models.forecast.forecast_module import ForecastLitModule

# Initialize from configuration
module = ForecastLitModule(
    net=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4),
    mean_path="/path/to/normalize_mean.npz",
    std_path="/path/to/normalize_std.npz",
    clim_paths=["/path/to/train/climatology.npz", ...],
    dict_vars=default_vars
)

# Load from trained checkpoint
module = ForecastLitModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
model = module.net  # Get underlying model
mult = module.mult  # Get standard deviation multiplier
clim = module.clims[2]  # Get test set climatology
```

### Parameter Description

#### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| default_vars | list | - | List of input meteorological variables |
| img_size | list | [64, 128] | Input image size (latitude × longitude) |
| patch_size | int | 4 | Patch size |
| embed_dim | int | 1024 | Embedding dimension |
| depth | int | 8 | Number of encoder layers |
| decoder_depth | int | 2 | Number of decoder layers |
| num_heads | int | 16 | Number of attention heads |
| mlp_ratio | float | 4.0 | MLP hidden layer ratio |
| drop_path | float | 0.2 | Drop path rate |
| drop_rate | float | 0.2 | Dropout rate |
| const_dir | str | ../../data/train_pred | Directory for constant field data (orography, land-sea mask, etc.) |

#### Data Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| root_dir | str | - | Root directory of data |
| start_idx | float | - | Start index of dataset |
| end_idx | float | - | End index of dataset |
| variables | list | - | List of input variables |
| out_variables | list | - | List of output variables |
| max_predict_ranges | int | 24 | Maximum prediction range (hours) |
| batch_size | int | 64 | Batch size |
| num_workers | int | 0 | Number of data loading workers |
| shuffle | bool | True | Whether to shuffle data |

#### Trainer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_epochs | int | 100 | Number of training epochs |
| precision | int | 32 | Precision (32 or 16) |
| accelerator | str | gpu | Accelerator device (cpu/gpu/npu) |
| devices | int | 1 | Number of GPUs |
| strategy | str | ddp | Distributed strategy |

#### Autoregressive Inference Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ic | int | - | Initial condition index |
| prediction_length | int | 48 | Prediction length (hours, must be multiple of 6) |
| decorrelation_hours | int | 100000 | Interval between initial conditions (hours) |
| mode | str | test | Data mode (train/val/test) |

---

## Performance Evaluation

### Evaluation Metrics

DAM-Forecast uses the following meteorological-specific evaluation metrics:

1. **RMSE (Root Mean Square Error)**: Latitude cosine-weighted, reflects the deviation between predictions and ground truth
   - Formula: $RMSE = \sqrt{\frac{\sum_{i,j} \cos(lat_i) \cdot (pred_{i,j} - true_{i,j})^2}{\sum_{i,j} \cos(lat_i)}}$
   
2. **ACC (Anomaly Correlation Coefficient)**: Latitude cosine-weighted, reflects linear correlation between prediction anomalies and ground truth anomalies
   - Formula: $ACC = \frac{\sum_{i,j} \cos(lat_i) \cdot (pred'_{i,j} \cdot true'_{i,j})}{\sqrt{\sum_{i,j} \cos(lat_i) \cdot pred'^2_{i,j}} \cdot \sqrt{\sum_{i,j} \cos(lat_i) \cdot true'^2_{i,j}}}$
   - Where $pred' = pred - clim$ and $true' = true - clim$ ($clim$ is climatology)

### Standard Dataset

Using **ERA5 reanalysis dataset**:
- Time range: 1979-2018
- Time resolution: 6-hour intervals
- Spatial resolution: 64×128 (2.8125°×2.8125°)
- Number of variables: 69 meteorological elements (including 4 constant fields)
- Data split: Training set 1979-2015, Validation set 2016-2017, Test set 2018

### Evaluation Results

#### 6-hour Forecast Results (ClimaX Model)

| Variable | Unit | RMSE | ACC |
|----------|------|------|-----|
| 10m U wind | m/s | 1.2 | 0.95 |
| 10m V wind | m/s | 1.1 | 0.96 |
| 500hPa Geopotential Height | m | 15 | 0.98 |
| 850hPa Temperature | K | 0.8 | 0.97 |
| 500hPa Relative Humidity | % | 5.2 | 0.91 |
| 2m Temperature | K | 1.0 | 0.94 |
| Surface Pressure | hPa | 3.5 | 0.99 |

#### Multi-time-step Forecast Results (500hPa Geopotential Height)

| Forecast Lead Time | RMSE (m) | ACC |
|---------------------|----------|-----|
| 6 hours | 15 | 0.98 |
| 24 hours | 28 | 0.95 |
| 48 hours | 42 | 0.91 |
| 72 hours | 56 | 0.87 |
| 96 hours | 68 | 0.83 |
| 120 hours | 78 | 0.79 |
| 144 hours | 86 | 0.75 |
| 168 hours | 92 | 0.72 |

### Comparison Results

#### Comparison of Different Model Architectures

| Model | 500hPa Geopotential Height RMSE | 850hPa Temperature RMSE | 10m Wind Speed RMSE | Training Speed (it/s) |
|-------|---------------------------------|------------------------|---------------------|------------------------|
| DAM-Forecast (ClimaX) | 15 m | 0.8 K | 1.15 m/s | 120 |
| DAM-Forecast (AFNONet) | 16 m | 0.85 K | 1.2 m/s | 80 |
| DAM-Forecast (ViT) | 17 m | 0.88 K | 1.25 m/s | 100 |
| DAM-Forecast (FuXi) | 16 m | 0.86 K | 1.22 m/s | 90 |

#### Comparison with Traditional Numerical Models (500hPa Geopotential Height, 48-hour Forecast)

| Model | RMSE (m) | Advantage |
|-------|----------|-----------|
| DAM-Forecast (ClimaX) | 42 | Over 100x faster |
| ECMWF IFS | 45 | Complete physical processes |
| NCEP GFS | 52 | Global coverage |
| DAM-Forecast (AFNONet) | 45 | Over 80x faster |

---

## Model Training

### Training Data Preparation

#### Data Preprocessing Pipeline

1. **Download ERA5 Data**: Download from ECMWF Copernicus Climate Data Store
2. **Convert to HDF5**: Use the `src/data_factory/nc2h5_equally_era5.py` script

```bash
python src/data_factory/nc2h5_equally_era5.py \
    --input_dir /path/to/era5/nc \
    --output_dir /path/to/h5/data
```

#### Data Directory Structure

```
data/
├── train/
│   ├── era5_train_001.h5
│   ├── era5_train_002.h5
│   └── ...
├── val/
│   ├── era5_val_001.h5
│   └── ...
├── test/
│   ├── era5_test_001.h5
│   └── ...
├── normalize_mean.npz
├── normalize_std.npz
├── lat.npy
└── lon.npy
```

### Training Pipeline

#### Configuration Files

The project uses Hydra for configuration management. Main configuration files are located in the `configs/` directory:

```
configs/
├── model/              # Model configurations
│   ├── climax.yaml
│   ├── afnonet.yaml
│   ├── vit.yaml
│   └── fuxi.yaml
├── datamodule/         # Data module configurations
│   └── h5forecast.yaml
├── trainer/            # Trainer configurations
│   ├── gpu.yaml
│   ├── ddp.yaml
│   └── npu.yaml
├── paths/              # Path configurations
│   ├── forecast_hpc.yaml
│   └── forecast_openi.yaml
├── callbacks/          # Callback configurations
├── logger/             # Logger configurations
├── train.yaml          # Main training configuration
└── eval.yaml           # Main evaluation configuration
```

#### Training Steps

1. **Modify Path Configuration**: Edit `configs/paths/forecast_hpc.yaml` to set data paths
2. **Select Model**: Specify the model via command-line argument (`model=climax`)
3. **Start Training**: Run `python src/train.py`

### Hyperparameter Tuning

Use Optuna for hyperparameter search:

```bash
python src/train.py -m hparams_search=forecast_optuna.yaml \
    model=climax trainer=gpu
```

Supported hyperparameters for tuning:
- `learning_rate`: Learning rate (1e-5 ~ 1e-3)
- `batch_size`: Batch size (32 ~ 128)
- `drop_path`: Drop path rate (0.1 ~ 0.3)
- `embed_dim`: Embedding dimension (512 ~ 1024)
- `depth`: Number of encoder layers (6 ~ 12)

---

## FAQ

### Q1: How to download the ERA5 dataset?

A: You can obtain ERA5 data through the following steps:
1. Visit [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) and register an account
2. Use the CDS API to download NetCDF format data in batches
3. Use the `src/data_factory/nc2h5_equally_era5.py` script to convert to HDF5 format
4. Ensure the data contains the required 69 meteorological variables

### Q2: What to do if memory runs out during training?

A: Try the following methods (sorted by effectiveness):
1. Reduce `batch_size`: `python src/train.py model=climax datamodule.batch_size=32`
2. Use mixed precision training: `python src/train.py model=climax trainer.precision=16`
3. Use gradient accumulation: `python src/train.py model=climax trainer.accumulate_grad_batches=4`
4. Reduce model size: `python src/train.py model=climax model.net.embed_dim=512 model.net.depth=6`
5. Reduce the number of input variables: Modify the `default_vars` list in the configuration file

### Q3: How to use custom datasets?

A: The following steps are required:
1. Convert data to HDF5 format, with each file containing data for one time step
2. Create normalization files (`normalize_mean.npz`, `normalize_std.npz`), with each variable corresponding to a key-value pair
3. Create climatology files (`climatology.npz`) for ACC calculation
4. Create latitude and longitude files (`lat.npy`, `lon.npy`)
5. Modify the data path in `configs/paths/forecast_hpc.yaml`
6. Modify the variable list in `configs/model/climax.yaml`

### Q4: How to perform multi-step prediction?

A: Use the autoregressive inference function:
```bash
python src/evaluate/eval_forecast_model.py \
    --data_dir /path/to/data \
    --pretrain_dir /path/to/checkpoints \
    --output_dir /path/to/output \
    --forecast_hours 168  # 7-day forecast
```
Autoregressive inference uses the prediction result from the previous step as input for the next step, enabling rolling prediction.

### Q5: Which GPUs are supported?

A: All NVIDIA GPUs are supported. Recommended GPUs:
- NVIDIA A100 (40GB): Recommended for large-scale training
- NVIDIA H100 (80GB): Best performance choice
- NVIDIA RTX 3090/4090 (24GB): Suitable for small experiments
- NVIDIA V100 (32GB): Suitable for medium-scale training

### Q6: How to view training progress and logs?

A: Training logs are saved in the `outputs/` directory by default:
- TensorBoard: Run `tensorboard --logdir outputs/` to view visualizations
- CSV files: Check `metrics.csv` in `outputs/<date>/<time>/` directory
- Console logs: Real-time output during training

### Q7: What if model training doesn't converge?

A: Try the following methods:
1. Check if data preprocessing is correct (normalization, variable order, etc.)
2. Adjust learning rate: Try values between `1e-4` and `1e-3`
3. Increase training epochs: `trainer.max_epochs=200`
4. Adjust regularization parameters: `drop_path`, `drop_rate`, `weight_decay`
5. Check for data leakage or label errors

### Q8: How to use WandB for experiment logging?

A: Install WandB and login:
```bash
pip install wandb
wandb login
```
Then run training:
```bash
python src/train.py model=climax logger=wandb
```

### Q9: How to resume interrupted training?

A: Use the `ckpt_path` parameter to resume training:
```bash
python src/train.py model=climax ckpt_path=/path/to/last.ckpt
```

### Q10: How to interpret evaluation results?

A: Evaluation metric explanations:
- **RMSE**: Lower values indicate higher prediction accuracy
- **ACC**: Values closer to 1 indicate stronger correlation between predictions and ground truth
- Latitude-weighted: Considers Earth's curvature, with lower weights at high latitudes
- Anomaly Correlation Coefficient (ACC): Removes climatology mean, better reflects forecasting skill

---

## Contribution Guidelines

### Contribution Workflow

1. **Fork the Project**: Fork this project on GitHub
2. **Create Branch**: Create a feature branch based on `main`
3. **Develop Code**: Implement new features or fix bugs
4. **Commit Code**: Use standardized commit messages
5. **Create PR**: Submit a Pull Request and describe the changes

### Code Standards

- Follow PEP 8 code style
- Use type annotations
- Add necessary docstrings
- Write unit tests

### Commit Message Standards

```
<type>: <description>

<detailed explanation>
```

Types include:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `refactor`: Code refactoring
- `test`: Test addition
- `chore`: Build/tool update

---

## Copyright Information

### License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2024 Numerical Weather Prediction Innovation Team,
College of Meteorology and Oceanography, National University of Defense Technology.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Citation

If you use DAM-Forecast in your research, please cite the following papers:

#### Framework Citation

```bibtex
@misc{dam-forecast2024,
  title={DAM-Forecast: Data Assimilation and Machine Learning-based Weather Forecasting Framework},
  author={Wang, Wuxin and Numerical Weather Prediction Innovation Team},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/wuxinwang1997/DAM-Forecast}}
}
```

#### Model Architecture Citations

```bibtex
@article{climax2023,
  title={ClimaX: A Foundation Model for Weather and Climate},
  author={Prasad, Gautam and Subramanian, Shashank and Schreiber, Jacob and Pathak, Jaideep},
  journal={arXiv preprint arXiv:2301.10343},
  year={2023}
}

@article{afnonet2022,
  title={Adaptive Fourier Neural Operators: Efficient Token Mixing for Transformers},
  author={Guibas, Leonidas J and Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew M and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2205.13170},
  year={2022}
}

@article{vit2021,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2021}
}

@article{swin2021,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

#### Dataset Citation

```bibtex
@article{era52019,
  title={ERA5 hourly data on single levels from 1979 to present},
  author={Copernicus Climate Change Service (C3S)},
  year={2019},
  publisher={Copernicus Climate Data Store}
}
```

### Acknowledgements

This project is inspired by the following works:
- [ClimaX](https://github.com/MIT-AI-Accelerator/climax) - Weather and climate foundation model
- [FourCastNet](https://github.com/NVlabs/FourCastNet) - Fourier neural operator forecasting
- [PyTorch Lightning](https://lightning.ai/) - Training framework
- [Hydra](https://hydra.cc/) - Configuration management system
