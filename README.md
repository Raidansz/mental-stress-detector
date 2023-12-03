
# ECG Stress Level Prediction

## Overview

This project aims to predict stress levels based on Electrocardiogram (ECG) data using a linear regression model. The model is trained on real ECG data obtained from [Zenodo](https://zenodo.org/records/7782558). Before training the model, the ECG data was cleaned and refined using [OpenRefine](https://openrefine.org/). 

Before training the model, the dataset was split into 80% training and 20% testing. The model's accuracy was observed to be around 0.6 on real data. Additionally, synthetic ECG data was used to further train the model, with accuracy tested around 0.4. More information about the synthetic data can be found in the `Gretel synthetics report.pdf` included in the repository.

**Note:** Do not run `train.py` unless you provide a new dataset and intend to retrain the model.

## Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Making Predictions](#making-predictions)
- [Mock ECG Data Generator](#mock-ecg-data-generator)
- [License](#license)

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ecg-stress-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ecg-stress-prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Making Predictions

To use the trained linear regression model for making predictions on new ECG data:

1. Load the trained model:

   ```python
   import joblib

   model_filename = 'linear_regression_model.pkl'
   model = joblib.load(model_filename)
   ```

2. Prepare input data using the provided function or your own data:

   ```python
   new_data = generate_mock_ecg_data()  # or provide your own DataFrame
   Check the sample structure in `detect.py`.
   ```

3. Make predictions:

   ```python
   predictions = model.predict(new_data)
   print("Predicted Stress Level:", predictions[0])
   ```

## Mock ECG Data Generator

To facilitate testing and demonstration, a function `generate_mock_ecg_data` is provided to generate mock ECG data with random but reasonable values.

Example usage:

```python
generated_data = generate_mock_ecg_data()
print(generated_data)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

