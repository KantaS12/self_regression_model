# Linear Regression From Scratch

A Python implementation of linear regression without using scikit-learn's built-in regression models. This project demonstrates the fundamental concepts of linear regression through a stock price prediction model for Apple Inc. (AAPL).

## Project Overview

This implementation includes:
- Data preprocessing with Principal Component Analysis (PCA)
- Linear regression using normal equation method
- Gradient descent optimization with L1 and L2 regularization
- Model evaluation and comparison with scikit-learn's implementation
- Visualization of predicted vs actual values

## Features

- **Data Acquisition**: Download stock data using yfinance
- **Preprocessing**: Standardization and dimensionality reduction using PCA
- **Model Implementation**:
  - Normal equation method (closed-form solution)
  - Gradient descent with customizable learning rate
  - L1 (Lasso) and L2 (Ridge) regularization
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
  - Relative error calculations

## Requirements

```
numpy
pandas
matplotlib
yfinance
scikit-learn
```

## Installation

```bash
pip install numpy pandas matplotlib yfinance scikit-learn
```

## Usage Example

```python
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# Download data
stock = 'AAPL'
data = yf.download(stock, start='2020-01-01', end='2025-01-01')

# Prepare data
data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
data['Returns_AAPL'] = data['Close_AAPL'].pct_change()
data.dropna(inplace=True)

# Define features and target
X = data[['Open_AAPL', 'High_AAPL', 'Low_AAPL', 'Volume_AAPL']]
y = data['Close_AAPL']

# Train/test split
X = np.array(X)
indices = np.random.permutation(len(X))
train_ratio = 0.8
split_index = int(len(X) * train_ratio)
x_train = X[indices[:split_index]]
x_test = X[indices[split_index:]]
y_train = np.array(y)[indices[:split_index]]
y_test = np.array(y)[indices[split_index:]]

# Apply your custom model (see full implementation in code)
# ...

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Closing Price')
plt.ylabel('Predicted Closing Price')
plt.title('Actual vs Predicted Closing Price')
plt.show()
```

## Implementation Details

### Preprocessing with PCA
This project applies Principal Component Analysis to reduce dimensionality while preserving the most important information in the data. The implementation follows these steps:
1. Standardization (z-score normalization)
2. Calculation of covariance matrix
3. Eigenvalue decomposition
4. Feature projection onto principal components

### Linear Regression Methods

#### Normal Equation
The closed-form solution:
```python
X = np.hstack([np.ones((x_train_reduced.shape[0], 1)), x_train_reduced])
w = np.linalg.inv(X.T @ X) @ X.T @ y_train
```

#### Gradient Descent with Regularization
Iterative optimization with L1 and L2 penalties:
```python
for _ in range(iterations):
    y_pred = x_train_reduced @ w + b
    
    dw = (-2/n) * x_train_reduced.T @ (y_train - y_pred) + lambda_l1 * np.sign(w) + 2 * lambda_l2 * w
    db = (-2/n) * np.sum(y_train - y_pred)
    
    w -= LR * dw
    b -= LR * db
```

## Results and Findings

- PCA successfully reduced the feature space while preserving important information
- Gradient descent with regularization improved model performance over the normal equation
- The custom implementation achieved different results compared to scikit-learn due to:
  - Feature scaling differences
  - Dimensionality reduction with PCA
  - Regularization techniques

## Future Improvements

- Apply consistent preprocessing for both custom and scikit-learn implementations
- Experiment with different regularization strengths
- Implement cross-validation to better evaluate model performance
- Explore additional financial features for stock price prediction

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

- **Yahoo Finance API**: For providing access to historical stock data
- **NumPy and pandas**: For data manipulation and numerical operations
- **Matplotlib**: For data visualization
- **scikit-learn**: For comparison against industry-standard implementations
- **Financial data providers**: For making stock market data accessible for educational purposes

## Author

Kanta Saito
