import numpy as np

from lorenz import simulate_lorenz, add_white_noise


print("Creating time series...")
ground_truth = simulate_lorenz()
time_series = add_white_noise(ground_truth, signal_to_noise_ratio=10)

np.savetxt("lorenz.csv", time_series, delimiter=",")
