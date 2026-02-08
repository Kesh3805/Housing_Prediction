# 🏘️ California Housing Price Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

*A comprehensive machine learning project for predicting California housing prices using multiple regression models with hyperparameter tuning.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models Used](#-models-used)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Visualizations](#-visualizations)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Overview

This project aims to predict housing prices in California using machine learning regression models. The project includes:

- 📊 **Data Preprocessing**: Handling missing values and feature scaling
- 🤖 **Multiple ML Models**: Linear Regression, Decision Tree, Random Forest, and XGBoost
- 🔧 **Hyperparameter Tuning**: GridSearchCV for optimal model parameters
- 📈 **Model Evaluation**: Comprehensive comparison using MSE, MAE, and R² metrics
- 🎨 **Rich Visualizations**: Feature importance, correlation heatmaps, and prediction plots
- 💻 **Interactive Prediction**: Widget-based user interface for price predictions

---

## 📊 Dataset

The project uses the **California Housing Dataset** from `sklearn.datasets`, which contains:

| Feature | Description |
|---------|-------------|
| **MedInc** | Median income in block group |
| **HouseAge** | Median house age in block group |
| **AveRooms** | Average number of rooms per household |
| **AveBedrms** | Average number of bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average number of household members |
| **Latitude** | Block group latitude |
| **Longitude** | Block group longitude |

**Dataset Statistics:**
- Total Samples: **20,640**
- Features: **8**
- Target: Median house value (in $100,000s)

---

## 🤖 Models Used

| Model | Description | Hyperparameter Tuning |
|-------|-------------|----------------------|
| **Linear Regression** | Baseline model | No tuning |
| **Decision Tree** | Non-linear regression | Max depth, Min samples split |
| **Random Forest** | Ensemble of decision trees | N estimators, Max depth |
| **XGBoost** | Gradient boosting algorithm | N estimators, Learning rate, Max depth |

---

## 📈 Results

### Model Performance Comparison

| Model | MSE | R² Score | MAE | Accuracy |
|-------|-----|----------|-----|----------|
| **Linear Regression** | 0.5559 | 0.5758 | 0.5332 | 57.58% |
| **Decision Tree** | 0.4174 | 0.6815 | 0.4341 | 68.15% |
| **Random Forest** | 0.2545 | 0.8058 | 0.3276 | 80.58% |
| **XGBoost** | 0.2152 | 0.8358 | 0.3026 | **83.58%** ⭐ |

### 🏆 Best Model: XGBoost
- **R² Score**: 83.58%
- **Mean Absolute Error**: $0.3026 (scaled units)
- **Mean Squared Error**: 0.2152

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Housing-Price-Prediction.git
   cd Housing-Price-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook Manoo_Housing_Price_Prediction.ipynb
   ```

---

## 💻 Usage

1. Open the notebook `Manoo_Housing_Price_Prediction.ipynb`
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Explore the visualizations and model comparisons
4. Use the interactive widget at the end to predict housing prices with custom inputs

### Quick Prediction Example

```python
# Example input features
MedInc = 8.3252
HouseAge = 41.0
AveRooms = 6.984
AveBedrms = 1.024
Population = 322.0
AveOccup = 2.555
Latitude = 37.88
Longitude = -122.23

# Predict using XGBoost model
prediction = best_models["XGBoost"].predict(user_input_scaled)
print(f"Predicted Price: ${prediction[0]:.2f}")
```

---

## 📁 Project Structure

```
Housing-Price-Prediction/
│
│── data/                                    # Data directory
│   └── housing.csv                          # (Optional) Saved dataset
│
│── Manoo_Housing_Price_Prediction.ipynb     # Main Jupyter notebook
│── requirements.txt                         # Python dependencies
│── README.md                                # Project documentation
│
└── .gitignore                               # Git ignore file
```

---

## ✨ Features

- ✅ **Automated Data Preprocessing**: Missing value handling and feature scaling
- ✅ **GridSearchCV Optimization**: Automatic hyperparameter tuning
- ✅ **Multiple Model Comparison**: Side-by-side performance evaluation
- ✅ **Rich Visualizations**: 
  - Model performance comparison charts
  - Feature importance plots
  - Correlation heatmaps
  - Actual vs Predicted scatter plots
- ✅ **Interactive Prediction Tool**: User-friendly widgets for custom predictions
- ✅ **Clean Code Structure**: Well-documented and organized

---

## 📊 Visualizations

The notebook includes several professional visualizations:

1. **Model Performance Comparison**
   - Side-by-side MSE, R², and MAE bar charts

2. **Feature Importance**
   - Random Forest and XGBoost feature importance rankings

3. **Correlation Heatmap**
   - Visual representation of feature correlations

4. **Actual vs Predicted Prices**
   - Scatter plot showing model prediction accuracy

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Manoranjini P**

- 📧 Email: [manoranjiniperiyasamy2005@gmail.com](mailto:manoranjiniperiyasamy2005@gmail.com])
- 💼 LinkedIn: [https://linkedin.com/in/manoranjini-periyasamy-52839626a](https://linkedin.com/in/manoranjini-periyasamy-52839626a)
- 🐙 GitHub: [@Manoranjini6268](https://github.com/Manoranjini6268)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- California Housing Dataset from `sklearn.datasets`
- scikit-learn documentation and community
- XGBoost developers and contributors
- Matplotlib and Seaborn for visualization tools

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ and Python**

</div>
