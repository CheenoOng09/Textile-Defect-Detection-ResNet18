# Textile Defect Detection (Smart Manufacturing Project)

This repository contains my **Smart Manufacturing** class project: detecting textile defects using a **ResNet** classifier trained on grayscale patch datasets (32×32 and 64×64). Code supports building datasets from CSV+HDF5 pairs, training/evaluation with PyTorch, and running inference on new images.

- Slides: `docs/Smart Manufacturing Presentation.pdf` (overview, baselines, metrics) fileciteturn1file0
- Report: `docs/Smart Manufacturing Project Report.pdf` (training results, overfitting discussion, datasets used) fileciteturn1file1

## 🧠 Approach (summary)
- **Model:** ResNet18 adapted for 1‑channel input; tested cross‑entropy vs focal loss; Adam and SGD optimizers; LR scheduler (ReduceLROnPlateau) in the optimized loop. fileciteturn1file2 fileciteturn1file3
- **Data:** CSV files provide `index`, `indication_value` (0=good, 1=defect after binarization), and `angle` metadata; HDF5 files store image tensors. The loader builds `x_train/x_val/x_test` by mapping CSV indices to HDF5 datasets. fileciteturn1file2 fileciteturn1file3
- **Results:** Validation accuracy ~**87%** with Adam @ 100 epochs on 32×32 patches; signs of **overfitting** on out‑of‑distribution images; suggestions include stronger regularization/augmentation and checking preprocessing/labels.

## 📂 Repo structure
```
.
├─ src/
│  ├─ training_script.py          # Optimized ResNet training loop (PyTorch) fileciteturn1file3
│  └─ smartman_1.py               # Earlier end‑to‑end notebook export (CSV+H5 to train/test) fileciteturn1file2
├─ docs/
│  ├─ Smart Manufacturing Presentation.pdf
│  └─ Smart Manufacturing Project Report.pdf
├─ data/                          # (gitignored) place CSV/HDF5 here if you keep them locally
│  ├─ train32.csv / test32.csv
│  ├─ train32.h5 / test32.h5
│  ├─ train64.csv / test64.csv
│  └─ train64.h5 / test64.h5
├─ models/                        # (LFS) trained weights & metrics
│  ├─ best_model.pth
│  └─ metrics.pth
└─ README.md
```

> **Large files:** The raw datasets and weights are big. This repo uses **Git LFS** for `*.pth` and ignores `data/` by default. If you need to share datasets, link to their source (e.g., Kaggle/ZJU Leaper) rather than committing binaries. 

## ⚙️ Environment (Windows + Anaconda)
From the project report/setup notes: create an environment, install PyTorch (with CUDA if available), and common libs. fileciteturn1file4

```bash
# Create env
conda create -n sm-textiles python=3.9 -y
conda activate sm-textiles

# Install PyTorch (CUDA 11.8 example; adjust to your GPU/CPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Others
conda install pandas h5py matplotlib scikit-learn pillow -y
pip install keras  # needed for one-hot utilities used in scripts
```

## ▶️ Training / Evaluation
Update the paths in `src/training_script.py` to point to your CSV/H5 files (see comments in the file). It expects CSVs with columns including `index`, `indication_value`, and `angle`, and HDF5 with an `'images'` dataset (verify your key per notes). 

```bash
# Train (saves best_model.pth and metrics.pth to models/)
python src/training_script.py
```

To run the earlier end‑to‑end pipeline with focal loss/Adam and quick inference examples, see `src/smartman_1.py`. fileciteturn1file2

## 🗃️ About the large files (what they are & how they affect the project)
- **`matchingtDATASET_train_32.h5`, `matchingtDATASET_test_32.h5`, `..._64.h5`**  
  HDF5 containers holding grayscale image patches (32×32 or 64×64). The scripts load arrays from these files using a dataset key (often `'images'`) and pair them to rows in the CSV via the `index` column. They are required to actually build the `x_train/x_val/x_test` tensors. 
- **`train32.csv`, `test32.csv`, `train64.csv`, `test64.csv`**  
  Metadata tables used to **select rows** from HDF5 and provide labels: `indication_value` is binarized (0→good, ≠0→defect), and `angle` is used to filter/concatenate subsets (e.g., only 20°/120° angles in some experiments). Uploading CSVs alone is not enough to run training, but they are small and **useful to include** for transparency and to document the schema. fileciteturn1file2 fileciteturn1file3
- **`best_model.pth`**  
  PyTorch model weights saved when validation accuracy improves. Needed for inference without retraining; store via **Git LFS** (binary). fileciteturn1file3
- **`metrics.pth`**  
  A small PyTorch checkpoint containing lists of `train_losses`, `val_losses`, `train_accuracies`, `val_accuracies` so you can re‑plot curves without re‑training. Optional but nice to keep under `models/` (LFS is OK but not strictly required if it’s small). 

### Should you upload the CSVs?
**Yes**, include the CSVs (they’re small) so others can inspect the splits/labels and replicate indexing. Keep all the `*.h5` datasets **out** of the repo; link to their sources or a cloud download instead. Your code will still run if the user downloads the H5 files locally and sets `file_path` accordingly. fileciteturn1file1

## ⬆️ Git LFS & .gitignore
This repo is configured to:
- track large binaries (`*.pth`) with **Git LFS**,
- ignore datasets (`data/**`) to keep the repo light.

### LFS (first time per machine)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## 📌 Repro tips
- Verify HDF5 keys before training (print `list(f.keys())` and use that key in the loader). fileciteturn1file4
- Ensure GPU is detected (`torch.cuda.is_available()`); training speed and batch size depend on it. fileciteturn1file4
- If you see overfitting, try stronger augmentation, `weight_decay`, `dropout`, and review labeling/normalization. fileciteturn1file1

## 📄 License
Educational use.
