# Delivery Duration Prediction

Predict total delivery time (`created_at` to `actual_delivery_time`, in seconds) for about 197k DoorDash orders.

**Files:** `delivery_duration_prediction.ipynb` (full project, already executed), `utility.py` (helper functions), `requirements.txt`.
**To run:** put `historical_data.csv` in the same folder and run the notebook top to bottom (about 13 minutes; most of it is the stacking and tuning steps).

## Results (time-based split: train to Feb 13, test Feb 13-18)
| Model | RMSE (s) | MAE (s) | R2 | Within 10 min |
|---|---|---|---|---|
| Baseline: mean | 1123.7 | 834.8 | -0.015 | 45.2% |
| Baseline: linear on estimates | 1089.5 | 806.4 | 0.045 | 46.8% |
| LightGBM (no store features) | 909.2 | 653.6 | 0.335 | 58.0% |
| LightGBM | 879.9 | 630.0 | 0.377 | 59.8% |
| LightGBM, prep-time decomposition | 879.0 | 629.4 | 0.379 | 60.1% |
| Ridge | 917.6 | 666.3 | 0.323 | 56.3% |
| MLP | 891.3 | 645.7 | 0.361 | 58.0% |
| LightGBM (tuned) | 882.5 | 632.3 | 0.374 | 59.6% |
| **Stacked ensemble** | **875.2** | 632.9 | 0.384 | 58.9% |

The stack and the decomposition model are within a few seconds of each other, so the simpler LightGBM decomposition model is the practical choice.
