import pandas as pd
import matplotlib.pyplot as plt

from data.tickers import ASSETS
from src.data_loader import load_data
from src.correlation import rolling_correlation
from src.systemic_metrics import eigenvalue_concentration
from src.structural_analysis import detect_structural_breaks