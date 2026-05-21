# model/

This folder stores the trained machine learning model.

## Files
- `plant_disease_model.h5` — Generated after running `train_model.py`
- `training_history.png` — Accuracy/loss plots (generated after training)

## How to generate the model

1. Download the PlantVillage dataset from Kaggle:
   https://www.kaggle.com/datasets/emmarex/plantdisease

2. Organize the dataset like this:
   ```
   dataset/
     train/
       Healthy/
       Early_Blight/
       Late_Blight/
       Leaf_Mold/
     val/
       Healthy/
       Early_Blight/
       Late_Blight/
       Leaf_Mold/
   ```

3. From the project root, run:
   ```bash
   python train_model.py
   ```

4. Training takes ~10–30 minutes depending on your hardware.
   The model will be saved here automatically.

## Running without a trained model (Demo Mode)
If no model file exists, the app runs in **Demo Mode**:
- Random predictions are returned for testing
- A yellow banner appears in the result section
- All UI features still work normally
