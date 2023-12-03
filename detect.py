import joblib
import numpy as np
import pandas as pd


model_filename = 'linear_regression_model.pkl'


model = joblib.load(model_filename)



def generate_mock_ecg_data():
    mean_hr = np.random.uniform(60, 100)
    avnn = np.random.uniform(600, 1000)
    sdnn = np.random.uniform(20, 100)
    nn50 = np.random.uniform(0, 50)
    pnn50 = np.random.uniform(0, 5)
    rmssd = np.random.uniform(10, 50)
    lf = np.random.uniform(0, 100)
    lf_norm = np.random.uniform(0, 1)
    hf = np.random.uniform(0, 100)
    hf_norm = np.random.uniform(0, 1)
    lf_hf_ratio = np.random.uniform(0.1, 2)

    mock_data = pd.DataFrame({
        'Mean HR (BPM)': [mean_hr],
        'AVNN (ms)': [avnn],
        'SDNN (ms)': [sdnn],
        'NN50 (beats)': [nn50],
        'pNN50 (%)': [pnn50],
        'RMSSD (ms)': [rmssd],
        'LF (ms2)': [lf],
        'LF Norm (n.u.)': [lf_norm],
        'HF (ms2)': [hf],
        'HF Norm (n.u.)': [hf_norm],
        'LF/HF Ratio': [lf_hf_ratio]
    })

    return mock_data






predictions = model.predict(generate_mock_ecg_data())

print("Predicted Stress Level:", predictions[0])




