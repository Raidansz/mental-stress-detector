import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv('synthetics-ecg.csv')

numeric_columns = ['Mean HR (BPM)', 'AVNN (ms)', 'SDNN (ms)', 'NN50 (beats)', 'pNN50 (%)', 'RMSSD (ms)', 'LF (ms2)', 'LF Norm (n.u.)', 'HF (ms2)', 'HF Norm (n.u.)', 'LF/HF Ratio']

for col in numeric_columns:
    if data[col].dtype == 'object':  
        data[col] = data[col].str.replace(',', '.').astype(float)

X = data[['Mean HR (BPM)', 'AVNN (ms)', 'SDNN (ms)', 'NN50 (beats)', 'pNN50 (%)', 'RMSSD (ms)', 'LF (ms2)', 'LF Norm (n.u.)', 'HF (ms2)', 'HF Norm (n.u.)','LF/HF Ratio']]
y = data['stress level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Ploting the actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual stress level Ratio")
plt.ylabel("Predicted stress level Ratio")
plt.title("Actual vs. Predicted stress level Ratio")
plt.show()


# Saving the trained model to a file
# model_filename = 'linear_regression_model.pkl'
# joblib.dump(model, model_filename)