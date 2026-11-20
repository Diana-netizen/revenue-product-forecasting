"""Structural constants shared by every module (change them and all results change)."""
import numpy as np

# For reproducibility
SEED = 42

# Forecast configurations
LOOKBACK = 90
HORIZON = 365

# Forecast Targets
TARGETS = ["Revenue", "Quantity"]

# Probability Forecasting
QUANTILES = np.array([0.10, 0.50, 0.90])


# Seasonal naive benchmark 52 weeks: keeps the weekday aligned (365 shifts it by one day)
SEASONAL_LAG = 364          

# one training window per week, anchored on the most recent data
STRIDE = 7                  

# LightGBM: every 2nd publication day of each window
ROW_STRIDE = 2  

# neural "epoch" = 25 random mini-batches
STEPS_PER_EPOCH = 25        
TFT_BATCH = 64
HORIZON_ORDER = ["First_Month", "First_Quarter", "Full_Year"]