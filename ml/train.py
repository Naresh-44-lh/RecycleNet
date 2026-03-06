# train.py — Training is done on Google Colab with GPU (T4).
# See the Colab notebook cells documented in the project README/docs.
# This file is a placeholder so the ml/ package is complete.
#
# Summary of training approach:
#   - Dataset: TrashNet (garythung/trashnet on Kaggle), 6 classes
#   - Model:   EfficientNetB4 (ImageNet pretrained, 380x380 input)
#   - Phase 1: Frozen base, train classifier head — Adam lr=1e-3, 10 epochs
#   - Phase 2: Unfreeze top 20 layers, fine-tune — Adam lr=1e-4, 15 epochs
#   - Class imbalance handled via sklearn compute_class_weight("balanced")
#   - Output:  model_final.h5, class_names.json  → place in ml/
