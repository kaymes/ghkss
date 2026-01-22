import os
import sys
import numpy as np

try:
    import ghkss
except ImportError:
    SCRIPT_DIR=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(SCRIPT_DIR, '../src'))
    import ghkss



def test_knn_finder():

    np.random.seed(0)

    for components, sequence_length, delay_pattern in [
        (1, 10000, [0,1,2,3,4]),
        (1, 10000, [4,2,0,7,3]),
        (3, 10000, [1,2,3,11,12,13,21,22,23]),
    ]:
        for minimum_neighbour_count, epsilon, maximum_neighbour_count in [
            (0, -1, 10),
            (1, -1, 10),
            (0, 0.5, 50),
        ]:
            for euclidean_norm in [True, False]:

                print(f"Running test for {components=}, {sequence_length=}, {delay_pattern=}, {minimum_neighbour_count=}, {epsilon=}, {maximum_neighbour_count=}, {euclidean_norm=}.")

                time_series = np.random.normal(0, 1, (sequence_length, components)).flatten()

                knn_finder = ghkss.ghkss_cpp.KNearestNeighbourFinder(time_series, delay_vector_pattern=delay_pattern, delay_vector_alignment=components)

                for index in range(0, sequence_length-max(delay_pattern), components):
                    neighbour_indices = knn_finder.find_nearest_neighbours(index=index, minimum_neighbour_count=minimum_neighbour_count, neighbour_epsilon=epsilon, maximum_neighbour_count=maximum_neighbour_count, euclidean_norm=euclidean_norm)

                    try:
                        knn_finder.assert_knn_result(index, neighbour_indices, minimum_neighbour_count=minimum_neighbour_count, neighbour_epsilon=epsilon, maximum_neighbour_count=maximum_neighbour_count, euclidean_norm=euclidean_norm)
                    except:
                        print(f"Failed for index {index}. The neighbour set was {neighbour_indices}.")
                        raise

