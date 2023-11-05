import numpy as np
import matplotlib.pyplot as plt

def generate_ecg_data(points, stress=False):
    x = np.linspace(0, 4 * np.pi, points)  # generates evenly spaced values
    if stress:
        return np.sin(x) + np.random.normal(0, 0.5, points)  # increased "noise" for stressed
    else:
        return np.sin(x) + np.random.normal(0, 0.2, points)

if __name__ == "__main__":
    not_stressed_sample = generate_ecg_data(100)
    stressed_sample = generate_ecg_data(100, stress=True)
    
    plt.plot(not_stressed_sample, label="Not Stressed")
    plt.plot(stressed_sample, label="Stressed", linestyle='dashed')
    plt.legend()
    plt.show()
