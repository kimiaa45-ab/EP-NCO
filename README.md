# EP-NCO
# EP-NCO: Latency-Aware Service Placement using Neural Combinatorial Optimisers

EP-NCO is a learning-based framework for latency-aware microservice placement in edge–cloud systems. It combines Graph Neural Networks (GNNs) with Reinforcement Learning (RL) to learn scalable placement policies for heterogeneous infrastructures.

This repository contains implementations of:
- EP-NCO with hard decoder (`nco_rl_hard_dec.py`)
- EP-NCO with soft decoder (`nco_rl_soft_dec.py`)
- RL hard-decoder baseline (`rl_hard_dec.py`)
- RL Soft-decoder baseline (`rl_soft_dec.py`)

The models load their training and testing settings from `configs/config.yaml`.

## Project Structure
```text
├── configs/
│ └── config.yaml # Main experiment configuration
├── data/
│ └── generated/ # (Downloaded externally)
├── img/ # Output plots
├── nets/
│ ├── encoder/
│ │ ├── mlp_encode.py
│ │ ├── nodegnn.py
│ │ └── servicegnn.py
│ ├── hard_decoder.py
│ └── soft_decoder.py
├── problem/
│ └── Dataset.py # Data loader
├── src/
│ ├── state.py
├── utils/
│ └── costfunction1.py
├── nco_rl_hard_dec.py # EP-NCO (hard decoder)
├── nco_rl_soft_dec.py # EP-NCO (soft decoder)
├── rl_hard_dec.py # RL baseline
├── rl_soft_dec.py # RL baseline (soft)
├── Test_GNN.py
├── testrl.py
├── requirements.txt
└── README.md
```

## Dataset

The generated datasets are not included directly in this repository because of their size.

Download the dataset from Google Drive:

```text
https://drive.google.com/drive/folders/1EBjEOiklv3lFB4W6p8aBOZsQx5tJC7bx
```

After downloading, place the folders under:

```text
data/generated/
```

The expected dataset folders are:

```text
small
medium
large
test_small
test_medium
test_large
test_xlarge
```

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include PyTorch, PyTorch Geometric, NumPy, Pandas, Matplotlib, PyYAML, and Scikit-learn. The full dependency list is provided in `requirements.txt`.

## Configuration

All experiments are controlled using:

```text
configs/config.yaml
```

To select the training scale, modify:

```yaml
data:
  type: small        # options: small, medium, large
  base_dir: data/generated
  test_type: test_xlarge
```

For example, to train on the medium-scale dataset:

```yaml
data:
  type: medium
```

To test on a specific test set, change:

```yaml
test_type: test_small     # options: test_small, test_medium, test_large, test_xlarge
```

Model-specific hyperparameters are defined under:

```yaml
model:
  small:
  medium:
  large:
  xlarge:
  xxlarge:
```

## Running EP-NCO

### EP-NCO with hard decoder

```bash
python nco_rl_hard_dec.py
```

### EP-NCO with soft decoder

```bash
python nco_rl_soft_dec.py
```

### RL hard-decoder baseline

```bash
python rl_hard_dec.py
```

Before running, make sure that `configs/config.yaml` points to the correct dataset scale.

## Example Workflow

Train EP-NCO on the small-scale dataset and evaluate on the XLarge test set:

```yaml
data:
  type: small
  base_dir: data/generated
  test_type: test_xlarge
```

Then run:

```bash
python nco_rl_hard_dec.py
```

## Paper

This repository accompanies the paper:

**EP-NCO: Latency-Aware Service Placement using Neural Combinatorial Optimisers for Edge–Cloud Systems**

## Citation

If you use this code, please cite:

```bibtex
?
```

## License

This project is released under the MIT License.

