import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Loading the dataset
data = pd.read_csv('ECG_dataset.csv')

# Listing of column names containing numeric values with commas
numeric_columns = ['Mean HR (bpm)', 'AVNN (ms)', 'SDNN (ms)', 'NN50 (beats)', 'pNN50 (%)', 'RMSSD (ms)', 'LF (ms2)', 'LF Norm (n.u.)', 'HF (ms2)', 'HF Norm (n.u.)', 'LF/HF Ratio']

# Iterating through numeric columns and replace commas with periods and convert to float if not already numeric
for col in numeric_columns:
    if data[col].dtype == 'object':  # Check if the column is of object type (string)
        data[col] = data[col].str.replace(',', '.').astype(float)

# Defining input features and the target variable
X = data[['Mean HR (bpm)', 'AVNN (ms)', 'SDNN (ms)', 'NN50 (beats)', 'pNN50 (%)', 'RMSSD (ms)', 'LF (ms2)', 'LF Norm (n.u.)', 'HF (ms2)', 'HF Norm (n.u.)']]
y = data['LF/HF Ratio']

# Spliting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Makeing predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Ploting the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual LF/HF Ratio")
plt.ylabel("Predicted LF/HF Ratio")
plt.title("Actual vs. Predicted LF/HF Ratio")
plt.show()


# # Saving the trained model to a file
# model_filename = 'linear_regression_model.pkl'
# joblib.dump(model, model_filename)