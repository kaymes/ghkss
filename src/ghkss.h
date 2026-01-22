#pragma once

#include <vector>
#include <optional>
#include <Eigen/Dense>

#ifdef WITH_PYTHON
#include <pybind11/pybind11.h>
#endif


namespace ghkss {

    typedef uint_fast32_t fast_uint;

    typedef std::vector<double> TimeSeries;

    inline constexpr unsigned int verbosity_none = 0;
    inline constexpr unsigned int verbosity_info = 1;
    inline constexpr unsigned int verbosity_high = 2;
    inline constexpr unsigned int verbosity_debug = 3;
    inline constexpr unsigned int verbosity_trace = 4;

    struct GhkssConfig {
        // The relative offsets of indices in the time series that form a delay vector.
        // The default pattern means that five consecutive indices are used.
        std::vector<fast_uint> delay_vector_pattern = {0,1,2,3,4};

        // The alignment of the delay vectors. If set to a value greater than 1, the delay vectors will only start at
        // indices that are multiples of the alignment. This is useful if the time series contains data from multiple
        // merged time series/
        fast_uint delay_vector_alignment = 1;

        // The dimension of the manifold onto which the data is projected.
        fast_uint projection_dimension = 2;

        // The minumum number of neighbours that should be found for each delay vector.
        size_t minimum_neighbour_count = 50;

        // If set to a positive value, then the algorithm will consider all neighbours within the given epsilon distance,
        // even if the number of neighbours is more than minimum_neighbour_count.
        double neighbour_epsilon = -1;

        // If set to true, the neighbour search will simulate the behaviour of the TISEAN algorithm:
        // In each round, all neighbours within the neighbour_epsilon radius are collected. If they are less
        // than minimum_neighbour_count, the epsilon radius is increased by a factor of sqrt(2) and the search is repeated
        // until at least minimum_neighbour_count neighbours are found. All neighbours found in the final round are returned.
        // If set to true, maximum_neighbour_count is ignored.
        bool tisean_epsilon_widening = false;

        // If neighbour_epsilon is positive, then this is the maximum number of neighbours that will be considered,
        // even if there are more within the epsilon distance. It is not guaranteed that the chosen neighbours are the closest ones.
        size_t maximum_neighbour_count = std::numeric_limits<size_t>::max();

        // If set to true, the distance between delay vectors will be calculated using the Euclidean norm, otherwise
        // the maximum norm is used.
        bool euclidean_norm = false;

        // Controls the verbosity of the messages printed to the console.
        // Zero means no output.
        unsigned int verbosity = verbosity_none;

    };

    struct NeighbourStatistics {
        size_t minimum_neighbour_count;
        size_t maximum_neighbour_count;
        double average_neighbour_count;
    };

    TimeSeries filter_ghkss(const TimeSeries& time_series, const GhkssConfig& config = GhkssConfig(), NeighbourStatistics* neighbour_statistics=nullptr);

#ifdef WITH_PYTHON
    void register_with_python(pybind11::module_& module);
#endif

} //namespace