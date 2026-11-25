# Chronic Kidney Disease Classification with a From-Scratch MLP

Artificial Intelligence course, Final Project, Sofía Salas and Hernán Silva.

For this assignment we picked the [UCI Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
(400 patients, 24 clinical features, ckd vs notckd) and a multilayer
perceptron (neural network) as our algorithm. The network is implemented
from scratch in NumPy: no sklearn, no PyTorch, nothing like that for the
actual learning algorithm, as required by the assignment.

## Files

```
chronic-kidney-disease-nn/
├── data/
│   └── kidney_disease.csv    # raw dataset
├── preprocess.py              # cleans the data, splits it, scales it
├── mlp.py                     # the neural network itself (forward/backward/training)
├── experiments.py             # hyperparameter search + final test results
├── make_plots.py              # makes the plots for the report
├── results/                   # json output from experiments.py
├── plots/                     # the generated plots used in the report
├── requirements.txt
└── README.md
```

## What each script does

**preprocess.py**: loads the raw csv, fixes some messy formatting, splits
it into train/val/test (60/20/20, keeping the same proportion of each
class in every split), and then fills in missing values and scales the
numeric columns. We split before doing any of that filling/scaling so we
don't accidentally leak information from val/test into training.

**mlp.py**: the actual network. He-initialized weights, ReLU in the
hidden layers, sigmoid at the end, binary cross-entropy loss, L2
regularization, and three optimizers we wrote ourselves (plain SGD,
momentum, Adam). Training stops early once the validation loss stops
improving.

**experiments.py**: tries out different architectures, learning rates, L2
values and optimizers (averaging a few seeds each time so we're not just
getting lucky/unlucky), picks the best one, retrains it a few more times
to pick the best seed, and only then checks the result on the test set
(once, at the very end).

**make_plots.py**: generates the loss/accuracy curves, the confusion
matrix, and the architecture comparison chart used in the report.

## How to run it

```bash
pip install -r requirements.txt
python3 preprocess.py
python3 experiments.py
python3 make_plots.py
```

## Report

The full report (dataset, methodology, results, discussion), written in
English as the assignment asks, is here:
https://docs.google.com/document/d/1e0ZKxRLE4AnjskrtXZ1kGI1MRVwcSLK47SOIhbF0ils/edit?tab=t.0
