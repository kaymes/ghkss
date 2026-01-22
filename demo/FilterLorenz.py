import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# Add the package to the path in case we're still in the source tree instead of using an installed package.
SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '../src'))

import ghkss
from lorenz import simulate_lorenz, add_white_noise

def print_statistics(time_series, ground_truth):
    noise = time_series - ground_truth
    signal_power = np.mean(np.square(ground_truth - np.mean(ground_truth, axis=0)))
    noise_power = np.mean(np.square(noise - np.mean(noise, axis=0)))
    signal_to_noise_ratio = 10 * math.log10(signal_power / noise_power)
    rmse = np.sqrt(np.mean(np.square(noise), axis=0))

    print("Standard deviation of noise:", np.std(noise, axis=0))
    print("RMSE:", rmse)
    print(f"SNR: {signal_to_noise_ratio:.2f} dB")
    print()



print("Creating time series...")
ground_truth = simulate_lorenz()
time_series = add_white_noise(ground_truth, signal_to_noise_ratio=10)

# Configure the filter
filter_config = ghkss.FilterConfig()
filter_config.set_delay_vector_pattern(signal_components=3, delay_vector_timesteps=5)
filter_config.projection_dimension = 3
filter_config.iterations = 3
filter_config.neighbour_epsilon = 10
filter_config.verbosity = ghkss.verbosity_info

print("Noisy time series metrics:")
print_statistics(time_series, ground_truth)

print("Filtering time series...")
filtered_time_series, neighbour_statistics = ghkss.filter_ghkss(time_series, filter_config, return_neighbour_statistics=True)

for index, statistics in enumerate(neighbour_statistics):
    print(f"The average neighbour count in round {index+1} was {statistics.average_neighbour_count:.1f}.")

print("Filtered time series metrics:")
print_statistics(filtered_time_series, ground_truth)

# Plot the results for visual comparison
figure = plt.figure()
axis = figure.add_subplot(131, projection='3d')
axis.scatter(time_series[:,0], time_series[:,1], time_series[:,2], s=0.1)
axis.set_title("Noisy time series")
axis = figure.add_subplot(132, projection='3d')
axis.scatter(filtered_time_series[:,0], filtered_time_series[:,1], filtered_time_series[:,2], s=0.1)
axis.set_title("Filtered time series")
axis = figure.add_subplot(133, projection='3d')
axis.scatter(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], s=0.1)
axis.set_title("Ground truth")

print("Done!")
plt.show()