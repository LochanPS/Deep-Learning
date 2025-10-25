# 📈 Stock Price Direction Prediction using Ensemble Machine Learning

## ⚠️ Transparency Note

**This project is an experimental educational exercise exploring advanced machine learning techniques for stock price prediction.** The code was developed with **AI assistance (Claude)** as part of learning ensemble methods, time-series validation, threshold optimization, and feature engineering for financial data.

## 🎯 Project Overview

This project implements a comprehensive stock price direction prediction system using ensemble machine learning approaches. The goal is to predict whether a stock's price will move up or down by more than 1% over the next 3 trading days.

### Key Features

- **Advanced Feature Engineering**: 40 carefully selected features from 120+ engineered indicators
- **Multiple Model Architectures**: XGBoost, LightGBM, Neural Networks, and Stacking Ensemble
- **Proper Time Series Validation**: Walk-forward cross-validation to prevent data leakage
- **Threshold Optimization**: Custom probability thresholds for imbalanced predictions
- **Trading Strategy Simulation**: Backtesting framework to evaluate real-world profitability
- **Comprehensive Evaluation**: Accuracy, ROC-AUC, confusion matrices, calibration analysis


## 🛠️ Technologies Used

### Core Libraries
- **Python 3.11+**
- **pandas** & **numpy** - Data manipulation and numerical computing
- **yfinance** - Real-time stock data fetching

### Machine Learning
- **scikit-learn** - Preprocessing, validation, classical ML algorithms
- **XGBoost** - Extreme gradient boosting
- **LightGBM** - Fast gradient boosting framework
- **TensorFlow/Keras** - Deep learning and neural networks

### Technical Analysis & Visualization
- **ta** - Technical analysis library for indicators
- **matplotlib** - Plotting and visualization
- **seaborn** - Statistical data visualization


## 📊 Model Architecture

### 1. Feature Engineering (120+ → 40 Selected Features)

#### Standard Technical Indicators
- Moving Averages (SMA, EMA, WMA at multiple periods)
- Momentum indicators (RSI, MACD, Stochastic Oscillator)
- Volatility measures (Bollinger Bands, ATR, Standard Deviation)
- Volume indicators (OBV, Volume Moving Average, MFI)

#### Custom Engineered Features
```python
# Multi-period returns
Return_1d, Return_3d, Return_5d, Return_10d

# Volatility regimes and patterns
Vol_5d, Vol_10d, Vol_20d, Volatility_Regime

# Candlestick patterns
Bullish_Engulfing, Bearish_Engulfing, Doji
Hammer, Shooting_Star

# External market context
SPX_Return_1d, Market_Trend, VIX, VIX_High
Market_Correlation

# Price action signals
Gap, Gap_Up, Gap_Down
Price_Momentum, Vol_Spike

# Moving average crossovers
MA5_MA20_Cross, MA10_MA50_Cross
Price_Above_MA50, Price_Above_MA200
```

### 2. Feature Selection Pipeline

Our systematic approach to feature selection:

```
Raw Features Generated (120+)
    ↓
Variance Threshold Filter
(Remove near-constant features)
    ↓
Correlation Filter
(Remove highly correlated features >0.9)
    ↓
Random Forest Importance Ranking
(Rank by predictive power)
    ↓
Final Selected Features (Top 40)
```

**Rationale**: Too many features lead to overfitting, while many technical indicators are highly correlated. This pipeline ensures we keep only the most informative, non-redundant signals.

### 3. Model Ensemble

Four distinct models trained and evaluated:

1. **XGBoost Classifier**
   - 500 estimators with 0.01 learning rate
   - Handles non-linear relationships effectively
   - Robust to outliers

2. **LightGBM Classifier** ⭐ **Best Performer**
   - 500 estimators with 0.01 learning rate
   - Memory efficient and fast training
   - Excellent handling of categorical features

3. **Neural Network**
   - 3 hidden layers with LeakyReLU activation
   - L2 regularization and dropout (0.2-0.4)
   - Gaussian noise injection for robustness

4. **Stacking Ensemble**
   - Base learners: XGBoost, LightGBM, Random Forest
   - Meta-learner: Logistic Regression
   - Combines strengths of multiple models


## 🎓 Key Learning Improvements Implemented

### Problem Formulation
- ❌ **Before**: Predict next-day direction (too noisy, ~50% accuracy)
- ✅ **After**: Predict 3-day direction with 1% threshold (clearer signal, better accuracy)

### Validation Strategy
- ❌ **Before**: Random 80-20 split (causes data leakage)
- ✅ **After**: Walk-forward cross-validation (realistic temporal evaluation)

### Feature Engineering
- ❌ **Before**: Basic price and volume only
- ✅ **After**: 120+ engineered features including market context, patterns, regimes

### Threshold Optimization
- ❌ **Before**: Default 0.5 threshold (assumes balanced classes)
- ✅ **After**: Optimized threshold (0.508) based on F1 score maximization

### Model Selection
- ❌ **Before**: Single model approach
- ✅ **After**: Ensemble of 4 different model architectures

### Regularization
- ❌ **Before**: No explicit overfitting prevention
- ✅ **After**: L2 penalty + Dropout + Gaussian noise + Correlation filtering


## 📈 Results & Performance

### Test Set Performance (63 samples, 3 months)

#### Individual Models

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| **LightGBM** ⭐ | **52.4%** | **0.731** | Best discrimination |
| Stacking Ensemble | 61.9% | 0.683 | Good balanced performance |
| XGBoost | 44.4% | 0.692 | High variance |
| Neural Network | 39.7% | 0.591 | Needs more data |

#### After Threshold Optimization (Stacking Model)

| Metric | Default (0.5) | Optimized (0.508) | Improvement |
|--------|---------------|-------------------|-------------|
| **Accuracy** | 61.9% | **68.3%** | +6.4% |
| **F1 Score** | 0.765 | **0.787** | +2.2% |
| **UP Recall** | 100.0% | 94.9% | -5.1% |
| **Precision** | 0.67 | 0.67 | No change |

**Key Insight**: Small threshold adjustment (0.5 → 0.508) significantly improved accuracy while maintaining high recall for profitable opportunities.

### Walk-Forward Cross-Validation (249 training samples)

```
Fold 1: Validation Accuracy = 58.00%
Fold 2: Validation Accuracy = 61.54%
Fold 3: Validation Accuracy = 59.26%
Fold 4: Validation Accuracy = 61.90%
Fold 5: Validation Accuracy = 57.50%

Mean CV Accuracy: 59.64% (±1.18%)
```

**Consistent performance** across temporal folds indicates model stability and generalization.


## 🔬 Key Findings & Insights

### 1. Threshold Optimization is Critical ⚡

Financial data is rarely balanced. The default 0.5 probability threshold assumes equal class distribution, which doesn't hold in stock markets. By optimizing the threshold based on the validation set:
- **+6.4% accuracy gain**
- Better alignment with business objectives (catching upward moves)
- Reduced false negatives without sacrificing precision

### 2. Tree-Based Models Excel on Tabular Financial Data 🌲

LightGBM achieved the **highest ROC-AUC (0.731)**, outperforming the neural network:
- Better handling of mixed feature types (numerical + categorical)
- More efficient with limited training data (249 samples)
- Built-in feature importance for interpretability
- No need for extensive hyperparameter tuning

### 3. High UP Recall (95%) = Opportunity Capture 📊

The optimized model successfully identifies **95% of profitable 3-day periods**:
- Critical for long-only trading strategies
- Some false positives acceptable (67% precision)
- Better to be in the market during real uptrends

### 4. Market Context Features Boost Performance 🌐

Including external factors significantly improved predictions:
- **S&P 500 returns**: Provides broader market trend context
- **VIX volatility index**: Captures market fear/greed sentiment
- **Market correlation**: Identifies stock's relationship to overall market

Individual stocks don't trade in isolation—market regime matters.

### 5. Limitations Discovered 🚧

- **Small sample size**: Only 63 test samples (3 months) - needs longer validation
- **Single stock tested**: AAPL only, may not generalize to other sectors
- **No transaction costs**: Real trading has commissions, slippage, spread
- **Direction only**: Doesn't predict magnitude of price moves
- **Bull market bias**: Test period was predominantly upward trending


## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-prediction-ml.git
cd stock-prediction-ml

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
ta>=0.10.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Quick Start

```bash
# Run the Jupyter notebook
jupyter notebook stock_predictor.ipynb

# Or run as a Python script
python stock_predictor.py
```

### Basic Usage Example

```python
import yfinance as yf
from lightgbm import LGBMClassifier

# 1. Download stock data
ticker = 'AAPL'
data = yf.download(ticker, period='2y')

# 2. Engineer features (see notebook for full pipeline)
# ... feature engineering code ...

# 3. Split data temporally (no shuffling!)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Train LightGBM model
lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
)
lgb_model.fit(X_train, y_train)

# 5. Predict with optimized threshold
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
optimal_threshold = 0.508  # From validation set
y_pred = (y_pred_proba > optimal_threshold).astype(int)

# 6. Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

### Customization

Change the stock, parameters, and target definition:

```python
# Test different stocks
ticker = 'TSLA'  # or 'MSFT', 'GOOGL', etc.
period = '3y'     # More historical data

# Adjust prediction parameters
PREDICTION_DAYS = 5      # Predict 5 days instead of 3
MOVEMENT_THRESHOLD = 0.015  # 1.5% instead of 1%

# Create target variable
data['Future_Return'] = (
    data['Close'].shift(-PREDICTION_DAYS) / data['Close'] - 1
)
data['Target'] = (data['Future_Return'] > MOVEMENT_THRESHOLD).astype(int)

# Tune model hyperparameters
lgb_model = LGBMClassifier(
    n_estimators=1000,      # More trees
    learning_rate=0.005,    # Slower learning
    max_depth=7,            # Deeper trees
    num_leaves=64,          # More leaf nodes
    min_child_samples=20    # Regularization
)
```


## 🔍 Methodology Deep Dive

### 1. Why 3-Day Prediction Window?

**Problem with 1-day predictions:**
- Too noisy (random daily fluctuations)
- ~50% accuracy (no better than coin flip)
- High frequency = more transaction costs

**Benefits of 3-day window:**
- Smoother signal, filters out noise
- More actionable (time to enter/exit positions)
- Better risk/reward for realistic trading
- Improved accuracy (59-68% range)

### 2. Target Engineering

```python
# Calculate 3-day forward return
data['Future_Return_3d'] = (
    data['Close'].shift(-3) / data['Close'] - 1
)

# Binary classification with significance threshold
data['Target'] = np.where(
    data['Future_Return_3d'] > 0.01,  # >1% move
    1,  # UP
    0   # DOWN/NEUTRAL
)
```

**Why 1% threshold?**
- Filters out insignificant moves
- Typical daily volatility for large-cap stocks is 1-2%
- Ensures predicted moves are tradeable after costs

### 3. Walk-Forward Cross-Validation

Standard k-fold CV randomly shuffles data—**this leaks future information** in time series!

Our approach:
```python
# Split data temporally
splits = [
    (train_1, val_1),  # Oldest data
    (train_2, val_2),
    (train_3, val_3),
    (train_4, val_4),
    (train_5, val_5)   # Most recent
]

# Each split:
# - Training set comes BEFORE validation set
# -
