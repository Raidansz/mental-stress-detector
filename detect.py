import joblib

# Load the trained model from the file
model_filename = 'linear_regression_model.pkl'
loaded_model = joblib.load(model_filename)

# Now you can use loaded_model for predictions without retraining
y_pred = loaded_model.predict(X_test)