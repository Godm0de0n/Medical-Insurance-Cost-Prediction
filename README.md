# 💸 Medical Insurance Cost Prediction

## 📌 Project Overview
This project aims to predict **medical insurance costs** based on various personal and medical attributes using machine learning techniques. The dataset includes key health indicators and demographic factors that influence insurance charges, and a **Linear Regression** model is trained to estimate these costs.

## 📊 Dataset
The dataset contains multiple features, including:
- **👤 Age**
- **⚧️ Gender**
- **🏠 Region**
- **📏 BMI (Body Mass Index)**
- **👶 Number of Children**
- **🚬 Smoker Status**
- **💲 Insurance Charges (Target Variable)**

## 🛠 Dependencies
The following Python libraries are required to run this project:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

## 🚀 Installation
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install pandas numpy matplotlib scikit-learn
```

## 🔄 Data Processing
1️⃣ Load the dataset using Pandas.  
2️⃣ Perform exploratory data analysis (EDA) to understand relationships.  
3️⃣ Handle missing values if any.  
4️⃣ Encode categorical features like **Gender** and **Smoker Status**.  
5️⃣ Split the dataset into training and testing sets.  

## 💡 Model Training
- The **Linear Regression** model is used to predict insurance charges.
- The dataset is split into **training (80%) and testing (20%)** sets.
- The model is trained on the training set to learn the relationship between features and insurance costs.

## 📈 Model Evaluation
The performance of the trained model is evaluated using:
- **✔️ R-squared (R²) Score**
- **📉 Mean Absolute Error (MAE)**
- **📊 Mean Squared Error (MSE)**

## ▶️ How to Run
1️⃣ Open the `Medical_Insurance_Cost_Prediction.ipynb` file in **Jupyter Notebook** or **Google Colab**.  
2️⃣ Run all the cells sequentially to preprocess data, train the model, and visualize results.  
3️⃣ Observe the predicted insurance charges and model evaluation metrics.  

## 🔥 Future Enhancements
- Implement **polynomial regression** for better accuracy.
- Explore **hyperparameter tuning** for the regression model.
- Deploy the model as a **web application** for user interaction.

## 🤝 Contribution
Feel free to contribute by **opening an issue** or **submitting a pull request**.  

## 📜 License
This project is licensed under the **MIT License**.

