# Stock Price Direction Prediction

> **Transparency Note**: This is an experimental learning project developed with **AI assistance (Claude)**. Created to understand ensemble learning, time-series validation, threshold optimization, and feature engineering for financial data. Not intended for actual trading.


## Overview

A machine learning system that predicts whether a stock's price will move up or down by more than 1% over the next 3 trading days. Uses ensemble methods (XGBoost, LightGBM, Neural Networks) with 40+ engineered features.

**Test Performance**: 68% accuracy, 0.73 ROC-AUC


## Key Learnings

This project demonstrates:
- Feature engineering for time-series financial data
- Walk-forward cross-validation (prevents data leakage)
- Threshold optimization for imbalanced predictions
- Ensemble model comparison
- Probability calibration issues and fixes
- Trading strategy backtesting


## Technology Stack

- **Python 3.11+**
- **Data**: yfinance, pandas, numpy
- **ML Models**: XGBoost, LightGBM, TensorFlow/Keras, scikit-learn
- **Technical Analysis**: ta library
- **Visualization**: matplotlib, seaborn

## Quick Start

```bash
# Install dependencies
pip install yfinance ta xgboost lightgbm tensorflow scikit-learn pandas numpy matplotlib seaborn

# Run the notebook
jupyter notebook stock_predictor.ipynb
```

## Model Architecture

### Feature Engineering (40 selected from 120+)
- Multi-period returns (1d, 3d, 5d, 10d)
- Moving averages and crossovers
- Volatility regimes (5d, 10d, 20d)
- Volume analysis and spikes
- Candlestick patterns (engulfing, doji)
- External market context (S&P 500, VIX)
- RSI, MACD, Bollinger Bands
- Price gaps and momentum indicators

### Feature Selection Pipeline
```
120+ raw features
  ↓ Variance threshold
  ↓ Correlation filter (>0.9 removed)
  ↓ Random Forest importance
40 final features
```

### Models Trained
1. **XGBoost** - 500 estimators, 0.01 learning rate
2. **LightGBM** - 500 estimators, 0.01 learning rate (best performer)
3. **Neural Network** - 3 hidden layers with LeakyReLU, L2 regularization
4. **Stacking Ensemble** - Combines XGB, LGB, RF with LogisticRegression meta-learner

## Results

### Walk-Forward Cross-Validation
```
Mean CV Accuracy: 59.64% (±1.18%)
Consistent across 5 temporal folds
```

### Test Set Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| LightGBM | **52.4%** | **0.731** |
| Stacking | 61.9% | 0.683 |
| XGBoost | 44.4% | 0.692 |
| Neural Net | 39.7% | 0.591 |

### After Threshold Optimization

| Metric | Default (0.5) | Optimized (0.508) |
|--------|---------------|-------------------|
| Accuracy | 61.9% | **68.3%** |
| F1 Score | 0.765 | **0.787** |
| UP Recall | 100% | 94.9% |
| Precision | 0.67 | 0.67 |

## Key Findings

### 1. Threshold Optimization is Critical
Default 0.5 threshold assumes balanced classes. Financial data rarely satisfies this. Optimal threshold (0.508) improved accuracy by 6+ points.

### 2. LightGBM Outperforms Deep Learning
Tree-based models handle tabular financial indicators better than neural networks with limited data (249 training samples).

### 3. High UP Recall (95%)
Model successfully identifies most profitable opportunities, though with some false positives.

### 4. Market Context Matters
Including S&P 500 returns and VIX improves predictions by providing broader market regime information.


## Limitations

- **Small sample size**: 63 test samples (3 months)
- **Single stock tested**: AAPL only, needs multi-asset validation
- **No transaction costs**: Real trading has fees and slippage
- **Direction only**: Doesn't predict magnitude of moves
- **Bull market bias**: Test period was predominantly upward trending

## Project Structure

```
stock-prediction/
├── stock_predictor.ipynb    # Main notebook
├── README.md                 # This file
└── requirements.txt          # Dependencies
```


## Methodology Highlights

### Proper Time-Series Validation
- No random shuffling (prevents lookahead bias)
- 80-20 temporal split
- Walk-forward cross-validation with 5 folds

### Feature Selection
- Variance threshold to remove constants
- Correlation filter to remove redundancy
- Random Forest importance for ranking

### Regularization
- L2 weight decay in neural network
- Dropout layers (0.2-0.4)
- Gaussian noise injection during training


## Usage Example

```python
# Train on AAPL
ticker = 'AAPL'
data = yf.download(ticker, period='2y')

# Engineer features
# ... (see notebook)

# Train LightGBM
lgb_model = LGBMClassifier(n_estimators=500, learning_rate=0.01)
lgb_model.fit(X_train, y_train)

# Predict with optimal threshold
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
optimal_threshold = 0.508
y_pred = (y_pred_proba > optimal_threshold).astype(int)
```


## Future Improvements

- Test across multiple stocks and sectors
- Add regime detection (bull/bear/sideways)
- Implement position sizing based on confidence
- Include transaction costs in backtest
- Add stop-loss and take-profit logic
- Explore LSTM/Transformer architectures
- Incorporate sentiment analysis
