
# TasForester 🌲

**TasForester** is a deep learning-based system to classify forest types (non-woody, sparse woody, and dense forest) using satellite data from Tasmania.

## 🔧 How it works
- Preprocess `.tif` Landsat tiles into 64x64 patches
- Train a CNN to classify each patch
- Handle class imbalance using weighted loss
- Evaluate and visualize predictions with pixel-level maps

## 📦 File Structure
- `tasforester_pipeline.ipynb`: Main notebook
- `requirements.txt`: Dependencies
- `forest_classifier_final.pth`: [Download Model (Google Drive)](https://drive.google.com/)  ← Replace with real link
- `tasmania_split/`: [Download Dataset (Google Drive)](https://drive.google.com/) ← Replace with real link

## 📊 Results
- Test Accuracy: 99.18%
- Sparse Woody Recall: 93%
- Visual output: Confusion matrix, heatmaps, prediction overlays

## 📜 License
Open source for academic and non-commercial use.

## 📬 Contact
For questions, please contact [rugvednivrutti.badhe@student.uts.edu] and [Anjali.k.ledade@student.uts.edu.au]
