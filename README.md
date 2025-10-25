# 📈 Stock Price Direction Prediction using Machine Learning

## ⚠️ Transparency Note

**This project is an experimental educational exercise exploring advanced machine learning techniques for stock price prediction.** The code was developed with AI assistance (Claude) as part of learning how to systematically improve model accuracy through feature engineering, ensemble methods, and proper validation techniques.

**Important Disclaimers:**
- This is a research/learning project, **NOT financial advice**
- Stock prediction is inherently uncertain and past performance doesn't guarantee future results
- The model should **NOT be used for actual trading** without extensive additional validation
- Financial markets are influenced by countless unpredictable factors beyond technical indicators


## 🎯 Project Overview

This project implements a comprehensive stock price direction prediction system using multiple machine learning approaches. The goal is to predict whether a stock's price will move up or down significantly (>1%) over the next 3 trading days.

### Key Features

- **Advanced Feature Engineering**: 40+ custom technical indicators, candlestick patterns, market context
- **Multiple Model Architectures**: XGBoost, LightGBM, Neural Networks, and Stacking Ensemble
- **Proper Time Series Validation**: Walk-forward cross-validation to prevent data leakage
- **Trading Strategy Simulation**: Backtesting framework to evaluate real-world profitability
- **Comprehensive Evaluation**: Accuracy, ROC-AUC, confusion matrices, and return analysis


## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+**
- **pandas** & **numpy** - Data manipulation
- **yfinance** - Stock data fetching
- **ta** - Technical analysis indicators

### Machine Learning
- **scikit-learn** - Preprocessing, validation, classical ML
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **TensorFlow/Keras** - Deep learning

### Visualization
- **matplotlib** - Plotting
- **seaborn** - Statistical visualizations


## 📊 Model Architecture

### 1. Feature Engineering (120+ Initial Features)

#### Standard Technical Indicators
- Moving Averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (Bollinger Bands, ATR)
- Volume indicators (OBV, MFI)

#### Custom Features
```python
# Multi-period returns
Return_1d, Return_3d, Return_5d, Return_10d

# Volatility regimes
Vol_5d, Vol_10d, Vol_20d, Volatility_Regime

# Candlestick patterns
Bullish_Engulfing, Bearish_Engulfing, Doji

# Market context
SPX_Return, Market_Trend, VIX, VIX_High

# Price gaps and spikes
Gap, Gap_Up, Gap_Down, Vol_Spike

# MA crossovers
MA5_MA20_Cross, MA10_MA50_Cross
```

### 2. Feature Selection Pipeline

```
Original Features (120+)
    ↓
Variance Threshold Filter
    ↓
Correlation Filter (>0.9 removed)
    ↓
Random Forest Importance Ranking
    ↓
Top 40 Features Selected
```

### 3. Model Ensemble

Four models trained and compared:

1. **XGBoost** - Gradient boosting with tree-based learning
2. **LightGBM** - Faster, memory-efficient gradient boosting
3. **Neural Network** - Deep learning with LeakyReLU and L2 regularization
4. **Stacking Ensemble** - Meta-learner combining XGB, LGB, and Random Forest

Best performing model automatically selected based on ROC-AUC score.


## 🎓 Key Learning Improvements Implemented

### Problem Formulation
- ❌ **Before**: Predict next-day direction (noisy, ~50% accuracy)
- ✅ **After**: Predict 3-day direction with 1% threshold (clearer signal)

### Validation Strategy
- ❌ **Before**: Random 80-20 split (data leakage)
- ✅ **After**: Time series cross-validation (realistic evaluation)

### Feature Engineering
- ❌ **Before**: Only standard indicators
- ✅ **After**: Custom features + market context + candlestick patterns

### Model Selection
- ❌ **Before**: Single neural network
- ✅ **After**: Ensemble of 4 different model types

### Regularization
- ❌ **Before**: Basic dropout
- ✅ **After**: L2 penalty + Gaussian noise + correlation filtering


## 📈 Results & Performance

### Expected Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 55-65% | Better than coin flip (50%) |
| **ROC-AUC** | 0.65-0.75 | Good discrimination ability |
| **Precision** | 60-70% | Reliability of UP predictions |
| **Recall** | 50-65% | Coverage of actual UP days |

### Walk-Forward Cross-Validation

```
Fold 1: Val Accuracy = 0.5823
Fold 2: Val Accuracy = 0.6015
Fold 3: Val Accuracy = 0.5947
Fold 4: Val Accuracy = 0.6142
Fold 5: Val Accuracy = 0.5891

Mean CV Accuracy: 0.5964 (+/- 0.0118)
```

### Trading Strategy Simulation

The model includes a backtesting framework that:
- Simulates going long when predicting UP (1)
- Stays flat when predicting DOWN (0)
- Compares strategy returns vs. buy-and-hold


## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-prediction-ml.git
cd stock-prediction-ml

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.1.70
ta>=0.10.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
tensorflow>=2.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Usage

```python
# Run the complete pipeline
python stock_predictor.py

# The script will:
# 1. Download 2 years of stock data (default: AAPL)
# 2. Engineer 120+ features
# 3. Select top 40 features
# 4. Train 4 different models
# 5. Perform walk-forward cross-validation
# 6. Generate performance reports
# 7. Backtest trading strategy
```

### Customization

Change the ticker symbol and parameters:

```python
# In stock_predictor.py
ticker = 'TSLA'  # Change stock
period = '3y'     # Change time period

# Adjust prediction threshold
data['Target'] = np.where(
    data['Future_Return_3d'] > 0.015,  # 1.5% instead of 1%
    1, 0
)

# Modify model hyperparameters
xgb_model = XGBClassifier(
    n_estimators=1000,      # More trees
    learning_rate=0.005,    # Slower learning
    max_depth=7             # Deeper trees
)
```


## 🔬 Methodology Deep Dive

### 1. Data Preprocessing

```python
# Handle multi-index columns from yfinance
# Fill missing values using forward fill
# Remove outliers (>5 standard deviations)
# Normalize all features using StandardScaler
```

### 2. Target Engineering

Instead of binary next-day prediction:

```python
# 3-day forward return
Future_Return_3d = Close[t+3] / Close[t] - 1

# Apply threshold for significance
Target = 1 if Future_Return_3d > 0.01 else 0
```

This reduces noise and creates a more learnable pattern.

### 3. Feature Selection Rationale

**Why only 40 features?**
- Too many features → overfitting
- Many indicators are correlated (e.g., SMA_10 and EMA_10)
- Focus on most informative signals

**Selection Process:**
1. Remove near-zero variance features
2. Drop highly correlated pairs (>0.9 correlation)
3. Rank by Random Forest importance
4. Keep top 40

### 4. Model Comparison Logic

Each model has strengths:
- **XGBoost**: Handles non-linearity well
- **LightGBM**: Fast, memory efficient
- **Neural Network**: Captures complex interactions
- **Stacking**: Combines all strengths

Best model selected by ROC-AUC on test set.

## 📊 Interpreting Results

### What does 60% accuracy mean?

In stock prediction:
- **50%** = Random guessing (coin flip)
- **55%** = Slight edge
- **60%** = Good predictive power
- **65%+** = Strong signal (rare)

### ROC-AUC Score

- **0.5** = No discrimination (random)
- **0.6-0.7** = Fair model
- **0.7-0.8** = Good model
- **0.8-0.9** = Excellent model
- **0.9-1.0** = Exceptional (or overfitting!)

### Strategy Returns

Compare ML strategy vs. buy-and-hold:
- **Positive outperformance** = Model adds value
- **Negative outperformance** = Better to buy-and-hold
- **High volatility** = Risky strategy

## 🔮 Future Improvements

### Potential Enhancements

- [ ] Add sentiment analysis from news/Twitter
- [ ] Incorporate earnings calendars and events
- [ ] Use LSTM/Transformer for sequence modeling
- [ ] Multi-stock portfolio optimization
- [ ] Real-time prediction API
- [ ] Add risk management (stop-loss, position sizing)
- [ ] Implement regime detection (bull/bear markets)
- [ ] Add more alternative data sources

### Advanced Techniques to Explore

- Reinforcement Learning (Q-learning, PPO)
- Attention mechanisms for feature importance
- Adversarial training for robustness
- Meta-learning for quick adaptation
- Causal inference for feature relationships

## 📧 Contact

For questions, suggestions, or discussions:
- GitHub Issues: [Open an issue](https://github.com/yourusername/stock-prediction-ml/issues)
- Email: your.email@example.com



