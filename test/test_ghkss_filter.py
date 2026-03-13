import math
import os
import sys
import numpy as np

import ghkss

try:
    import lorenz
except ImportError:
    SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(SCRIPT_DIR, '../demo'))
    import lorenz


def filter_ghkss_python(time_series, config):

    filtered_time_series_cpp = ghkss.ghkss_cpp.filter_ghkss(time_series, config._as_base())

    if not hasattr(ghkss.ghkss_cpp, '_debug_info'):
        print("Warning: the C++ extension is not a debug build. Only testing the final result, not verifying intermediate steps.")

    # Check the the C++ code got the time series correctly.
    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        assert np.allclose(ghkss.ghkss_cpp._debug_info.time_series, time_series), f"original time series differ by {np.max(np.abs(filtered_time_series_cpp - time_series))}"

    # Work out the limit on the delay vector index.
    max_delay_index = len(time_series) - 1 - max(config.delay_vector_pattern)

    # Verify that the C++ code got the number of delay vectors correct.
    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        delay_vector_count = len(list(range(0,max_delay_index+1,config.delay_vector_alignment)))
        assert ghkss.ghkss_cpp._debug_info.delay_vector_count == delay_vector_count, f"The number of delay vectors doesn't match. {ghkss.ghkss_cpp._debug_info.delay_vector_count=} != {delay_vector_count=}"

    # The neighbour-finder is tested separately, so we can use it here without validating its results.
    knn_finder = ghkss.ghkss_cpp.KNearestNeighbourFinder(time_series, delay_vector_pattern=config.delay_vector_pattern, delay_vector_alignment=config.delay_vector_alignment)

    def get_neighbour_indices(index):
        if config.tisean_epsilon_widening:
            neighbours = []
            epsilon = config.neighbour_epsilon
            while len(neighbours) < config.minimum_neighbour_count:
                neighbours = knn_finder.find_nearest_neighbours(
                    index=index,
                    minimum_neighbour_count=0,
                    neighbour_epsilon=epsilon,
                    maximum_neighbour_count=config.maximum_neighbour_count,
                    euclidean_norm=config.euclidean_norm
                )
                epsilon *= math.sqrt(2)
        else:
            neighbours = knn_finder.find_nearest_neighbours(
                index=index,
                minimum_neighbour_count=config.minimum_neighbour_count,
                neighbour_epsilon=config.neighbour_epsilon,
                maximum_neighbour_count=config.maximum_neighbour_count,
                euclidean_norm=config.euclidean_norm
            )

        return neighbours


    # Build the list of all neighbour indices.
    # We get it as dict because the index is a multiple of the alignment.
    all_neighbours = {index: get_neighbour_indices(index) for index in range(0,max_delay_index+1,config.delay_vector_alignment)}

    # Check that the neighbour indices are correctly aligned
    for neighbours in all_neighbours.values():
        for neighbour in neighbours:
            assert neighbour % config.delay_vector_alignment == 0

    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        # The C++ code stores the neighbour indices in a flat array, so we need to adjust for the alignment.
        assert {i*config.delay_vector_alignment:n for i,n in enumerate(ghkss.ghkss_cpp._debug_info.all_neighbours)} == all_neighbours, "The neighbour indices don't match."



    # Create the weights
    delay_vector_weights = [1] * len(config.delay_vector_pattern)
    for offset in range(config.delay_vector_alignment):
        relevant_indices = [i for i,v in enumerate(config.delay_vector_pattern) if v % config.delay_vector_alignment == offset]
        if relevant_indices:
            delay_vector_weights[relevant_indices[0]] = 1000
            delay_vector_weights[relevant_indices[-1]] = 1000

    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        assert np.allclose(ghkss.ghkss_cpp._debug_info.delay_vector_weights, delay_vector_weights), f"The weights don't match. {delay_vector_weights=} != {ghkss.ghkss_cpp._debug_info.delay_vector_weights=}"

    # Create the traces
    traces = [0.] * config.delay_vector_alignment
    for index, weight in enumerate(delay_vector_weights):
        traces[config.delay_vector_pattern[index] % config.delay_vector_alignment] += 1. / weight

    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        assert np.allclose(ghkss.ghkss_cpp._debug_info.traces, traces), f"The traces don't match. {traces=} != {ghkss.ghkss_cpp._debug_info.traces=}"


    def get_delay_vector(index):
        assert index % config.delay_vector_alignment == 0
        return [time_series[index + i] for i in config.delay_vector_pattern]

    def calculate_initial_correction_deltas(index):
        assert index % config.delay_vector_alignment == 0

        neighbours = [get_delay_vector(i) for i in all_neighbours[index]]
        neighbours = np.array(neighbours)
        covariance_matrix = np.cov(neighbours, rowvar=False, bias=True)

        # apply the weights to the covariance matrix
        covariance_matrix *= np.array(delay_vector_weights).reshape(-1, 1) * np.array(delay_vector_weights).reshape(1, -1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # np.linalg.eigh returns eigenvectors in ascending order of eigenvalues.
        null_space_dimension = len(config.delay_vector_pattern) - config.projection_dimension
        projection_vectors = eigenvectors.T[:null_space_dimension]

        to_project = np.array(get_delay_vector(index)) - np.mean(neighbours, axis=0)
        to_project *= delay_vector_weights # appy the weights to the delay vector
        projected = np.zeros(to_project.shape)
        for vector in projection_vectors:
            projected += vector * np.dot(to_project, vector)

        return projected / delay_vector_weights


    # Calculate the initial correction deltas.
    initial_correction_deltas = []
    for index in range(0,max_delay_index+1,config.delay_vector_alignment):
        initial_correction_deltas.append(calculate_initial_correction_deltas(index))

    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        assert np.allclose(ghkss.ghkss_cpp._debug_info.initial_correction_deltas, initial_correction_deltas), f"The correction deltas don't match. {initial_correction_deltas=} != {ghkss.ghkss_cpp._debug_info.initial_correction_deltas=}"

    # We handle the trend here. (handle_trend() in the C++ code)
    final_deltas = np.zeros(len(time_series))
    for index in range(0,max_delay_index+1,config.delay_vector_alignment):
        average_correction = np.mean(np.array([initial_correction_deltas[i//config.delay_vector_alignment] for i in all_neighbours[index]]), axis=0)

        for dimension, offset_index in enumerate(config.delay_vector_pattern):
            final_deltas[index+offset_index] += ((initial_correction_deltas[index//config.delay_vector_alignment][dimension] - average_correction[dimension]) /
                                                 (traces[offset_index % config.delay_vector_alignment] * delay_vector_weights[dimension]))

    if hasattr(ghkss.ghkss_cpp, '_debug_info'):
        assert np.allclose(ghkss.ghkss_cpp._debug_info.final_deltas, final_deltas), f"The final correction deltas don't match. {final_deltas=} != {ghkss.ghkss_cpp._debug_info.final_deltas=}"

    # Now we can apply the correction deltas to the original time series.
    filtered_time_series = time_series - final_deltas
    assert np.allclose(filtered_time_series, filtered_time_series_cpp), f"The time series differ by {np.max(np.abs(filtered_time_series - filtered_time_series_cpp))}."




def test_ghkss_filter_single_component():
    # we test the C++ module for a single component.
    time_series = lorenz.simulate_lorenz(num_steps=5000)[:,0]
    time_series = lorenz.add_white_noise(time_series, signal_to_noise_ratio=10)
    time_series = time_series.flatten(order='C')

    filter_config = ghkss.FilterConfig()
    filter_config.set_delay_vector_pattern(signal_components=1, delay_vector_timesteps=5)
    filter_config.projection_dimension = 3
    filter_config.iterations = 1
    filter_config.neighbour_epsilon = 10

    filter_ghkss_python(time_series=time_series, config=filter_config)


def test_ghkss_filter_multiple_components():
    # we test the C++ module for a multiple components.
    time_series = lorenz.simulate_lorenz(num_steps=5000)
    time_series = lorenz.add_white_noise(time_series, signal_to_noise_ratio=10)
    assert time_series.shape[1] == 3
    time_series = time_series.flatten(order='C')

    filter_config = ghkss.FilterConfig()
    filter_config.set_delay_vector_pattern(signal_components=3, delay_vector_timesteps=5)
    filter_config.projection_dimension = 3
    filter_config.iterations = 1
    filter_config.neighbour_epsilon = 10

    filter_ghkss_python(time_series=time_series, config=filter_config)

def test_ghkss_filter_iterations():
    # We test whether the python implementation handles batches and iterations correctly
    time_series = lorenz.simulate_lorenz(num_steps=5000)
    time_series = lorenz.add_white_noise(time_series, signal_to_noise_ratio=10)
    assert time_series.shape[1] == 3

    filter_config = ghkss.FilterConfig()
    filter_config.set_delay_vector_pattern(signal_components=3, delay_vector_timesteps=5)
    filter_config.projection_dimension = 3
    filter_config.iterations = 3
    filter_config.neighbour_epsilon = 10

    filtered_time_series = ghkss.filter_ghkss(time_series, filter_config)

    filter_config.iterations = 1
    iteratively_filtered_time_series = time_series
    for _ in range(3):
        iteratively_filtered_time_series = ghkss.filter_ghkss(iteratively_filtered_time_series, filter_config)

    assert np.allclose(filtered_time_series, iteratively_filtered_time_series)

def test_ghkss_filter_batch():
    # We test whether the python implementation handles batches and iterations correctly
    time_series = lorenz.simulate_lorenz(num_steps=5000)
    time_series = lorenz.add_white_noise(time_series, signal_to_noise_ratio=10)
    assert time_series.shape[1] == 3

    batch_size = 2000

    filter_config = ghkss.FilterConfig()
    filter_config.set_delay_vector_pattern(signal_components=3, delay_vector_timesteps=5)
    filter_config.projection_dimension = 3
    filter_config.iterations = 3
    filter_config.neighbour_epsilon = 10
    filter_config.batch_size = batch_size

    filtered_time_series = ghkss.filter_ghkss(time_series, filter_config)

    filter_config.batch_size = float('inf')
    for batch_start in range(0, len(time_series), batch_size):
        assert np.allclose(filtered_time_series[batch_start:batch_start+batch_size,:], ghkss.filter_ghkss(time_series[batch_start:batch_start+batch_size,:], filter_config))


