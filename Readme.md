# SpatSite

SpatSite is a geometric deep learning model for binding site prediction by introducing spatial features to capture protein shape and integrating multi-source features for learning latent residue binding patterns.

This repository provides the implementation of SpatSite, including model code, training scripts, predicting scripts, and testing scripts.

## Repository Structure

```text
SpatSite/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── scripts/
│   ├── Extract_features.py             # feature extraction script
│   ├── train.py                        # train script
│   ├── test.py                         # test script
│   ├── predict.py                      # predict script
│   ├── SpatSite.py
│   ├── topological_features.py
│   └── metrics_calculation.py
├── data/
│   └── example                         # example data
├── checkpoints
└── results/
    └── PMN_spatsite_model.pth
```

## Installation

Clone this repository:

```bash
git clone https://github.com/Lowe32917/SpatSite.git
cd SpatSite
```

Create and activate a Python environment. For example:

```bash
conda create -n spatsite python=3.9
conda activate spatsite
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The original experiments were conducted with PyTorch 1.13.1 and CUDA 11.7. Some PyTorch Geometric dependencies, such as `torch-scatter` and `torch-cluster`, are CUDA-dependent. If they cannot be installed directly from `requirements.txt`, please install the versions matching your PyTorch and CUDA environment.

For the original experimental environment, the following commands can be used:

```bash
python -m pip install torch-scatter==2.1.1+pt113cu117 torch-cluster==1.6.1+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install torch-geometric==2.6.1
```

## Requirements

The main dependencies include:

```text
freesasa==2.2.1
h5py==3.14.0
networkx==3.2.1
numpy==1.24.3
pandas==1.3.5
prettytable==3.16.0
requests==2.32.5
scikit-learn==1.0.2
scipy==1.13.1
torch==1.13.1
torch-geometric==2.6.1
torch-cluster==1.6.1+pt113cu117
torch-scatter==2.1.1+pt113cu117
torchnet==0.0.4
tqdm==4.67.1
visdom==0.2.4
```

Please refer to `requirements.txt` for the complete dependency list.

## Data

The original datasets and processed feature files are not included in this repository because of file size and/or data license restrictions.

Please place the required data files under the `data/` directory before running the scripts.

Example:

```text
data/
└── example/
    ├── feature/
    │   ├── HMM
    │   ├── PSSM
    │   ├── SS
    │   └── LLM
    ├── PDB
    ├── train.txt
    ├── test.txt
    └── predict.txt
```

## Training

Example command for training:

```bash
python scripts/Extract_features.py --dataset_name example --dataname train
python scripts/Extract_features.py --dataset_name example --dataname test

python scripts/train.py --dataset_name example
```

Please modify the input paths and parameters according to your own dataset.
Note: Due to the large scale of the complete datasets, only a very small protein–MN example dataset is provided to test the functionality of the training script. The results obtained from this example do not reflect the actual performance of SpatSite.
We provide trained model weights at `checkpoints/PMN_spatsite_model.pth` for testing the downstream evaluation and prediction scripts.

## Predicting

Example command for testing:

```bash
python scripts/Extract_features.py --dataset_name example --dataname predict

python scripts/predict.py --dataset_name example
```

By default, the prediction script uses the model checkpoint generated after training with the corresponding dataset. To specify a different checkpoint, please use the `--model_path` argument.

```bash
SPATSITE_DIR=/your_path/SpatSite
python scripts/predict.py --dataset_name example --model_path ${SPATSITE_DIR}/checkpoints/models/PMN_spatsite_model.pth
```


## Testing

Example command for testing:

```bash
python scripts/test.py --dataset_name example
```

By default, the testing script uses the model checkpoint generated after training with the corresponding dataset. To specify a different checkpoint, please use the `--model_path` argument.

```bash
SPATSITE_DIR=/your_path/SpatSite
python scripts/test.py --dataset_name example --model_path ${SPATSITE_DIR}/checkpoints/models/PMN_spatsite_model.pth
```

## Output

Trained model checkpoints and evaluation outputs will be saved under:

```text
checkpoints/
```

Prediction results will be saved under:

```text
results/
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{!!!!!!},
  title={!!!!!!},
  author={!!!!!!},
  journal={!!!!!!},
  year={!!!!!!}
}
```

## License

The source code in this repository is released under the MIT License. The example data and trained model weights are provided for research and testing purposes only.


