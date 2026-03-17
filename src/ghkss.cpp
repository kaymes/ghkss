#include "ghkss.h"

#include <algorithm>
#include <ranges>
#include <Eigen/Dense>
#include "knn_kdtree.h"

#ifdef WITH_PYTHON
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#endif

#ifdef WITH_DEBUG_INFO
#include <mutex>
#endif


namespace ghkss {

    typedef decltype(((KNearestNeighbourFinder*) 1)->find_nearest_neighbours(0, 0))::value_type SequenceIndex;

    typedef Eigen::VectorXd DelayVector;

#ifdef WITH_DEBUG_INFO
    class DebugInfo {
    private:
        DebugInfo() = default;
        DebugInfo(const DebugInfo&) = delete;
        DebugInfo& operator=(const DebugInfo&) = delete;
    public:
        // The debug interface is only usable for single-threaded applications
        // due to the use of a global singleton. However, to avoid undefined behaviour
        // if a debug build is used multithreaded, we protect access with a mutex.
        class Accessor {
        public:
            DebugInfo& data;
            Accessor(DebugInfo& _data, std::mutex& mutex) : data(_data), lock(mutex) {}
        private:
            std::lock_guard<std::mutex> lock;
        };

        static Accessor instance() {
            static DebugInfo instance;
            static std::mutex mutex;
            return Accessor(instance, mutex);
        }

    public:
        TimeSeries time_series;
        size_t delay_vector_count;
        DelayVector delay_vector_weights;
        std::vector<std::vector<SequenceIndex>> all_neighbours;
        std::vector<DelayVector> initial_correction_deltas;
        Eigen::VectorXd traces;
        Eigen::VectorXd final_deltas;
        TimeSeries filtered_time_series;

    public:
#ifdef WITH_PYTHON
        static void register_with_python(pybind11::module_& module) {
            static bool first_call = true;
            if (!first_call) {
                return;
            }
            first_call = false;

            pybind11::class_<DebugInfo>(module, "_debug_info")
                .def_property_readonly_static("time_series", [](const pybind11::object&) { return instance().data.time_series; })
                .def_property_readonly_static("delay_vector_count", [](const pybind11::object&) { return instance().data.delay_vector_count; })
                .def_property_readonly_static("delay_vector_weights", [](const pybind11::object&) { return instance().data.delay_vector_weights; })
                .def_property_readonly_static("all_neighbours", [](const pybind11::object&) { return instance().data.all_neighbours; })
                .def_property_readonly_static("initial_correction_deltas", [](const pybind11::object&) { return instance().data.initial_correction_deltas; })
                .def_property_readonly_static("traces", [](const pybind11::object&) { return instance().data.traces; })
                .def_property_readonly_static("final_deltas", [](const pybind11::object&) { return instance().data.final_deltas; })
                .def_property_readonly_static("filtered_time_series", [](const pybind11::object&) { return instance().data.filtered_time_series; })
            ;
        };
#endif
    };
#endif


    DelayVector get_delay_vector_weights(const GhkssConfig& config) {
        // Set the delay vector weights. We just use the suggested values from the book / TISEAN here.
        // Maybe this should be configurable later on.
        auto embedding_dimension = config.delay_vector_pattern.size();
        DelayVector delay_vector_weights = DelayVector::Ones(embedding_dimension);
        for (size_t alignment_offset = 0; alignment_offset < config.delay_vector_alignment; alignment_offset++) {
            for (size_t delay_index=0; delay_index < embedding_dimension; delay_index++) {
                if (config.delay_vector_pattern[delay_index] % config.delay_vector_alignment == alignment_offset) {
                    delay_vector_weights[delay_index] = 1000;
                    break;
                }
            }
            for (int delay_index = embedding_dimension-1; delay_index>=0; delay_index--) {
                if (config.delay_vector_pattern[delay_index] % config.delay_vector_alignment == alignment_offset) {
                    delay_vector_weights[delay_index] = 1000;
                    break;
                }
            }
        }
        return delay_vector_weights;
    }


    // This function doesn't actually perform a correction. It calculates correction delta vectors.
    // The resulting vectors are the \delta s_n shown in Equation 10.13 on page 182 of the Kantz book.
    DelayVector calculate_correction_deltas(
            const size_t index_to_correct,          // The index for which correction is applied. Given as index of the time series.
            const std::vector<SequenceIndex>& neighbours,
            const std::vector<double>& time_series,
            const DelayVector& weights,           // Array of weights that specify how much each dimension of the embedding should be considered.
            const GhkssConfig& config
       ) {

        auto embedding_dimension = config.delay_vector_pattern.size();

        // First we calculate the average of all our neighbours.
        DelayVector average_neighbour(embedding_dimension);
        for (size_t dimension = 0; dimension < embedding_dimension; dimension++) {
            double sum = 0;
            for (auto neighbour: neighbours) {
                sum += time_series[neighbour + config.delay_vector_pattern[dimension]];
            }
            average_neighbour[dimension] = sum / neighbours.size();
        }


        // allocate some memory for the covariance matrix.
        Eigen::MatrixXd covariance_matrix(embedding_dimension, embedding_dimension);

        // Now we calculate the covariance matrix between different dimensions in the neighbouring vectors.
        for (long dimension1 = 0; dimension1 < embedding_dimension; dimension1++) {
            for (long dimension2 = dimension1; dimension2 < embedding_dimension; dimension2++) { // only iterate over part of the range because the resulting matrix is summetrical.

                // Let's calculate E[XY] where X and Y are the values the neighbours have at index "dimension1" and "dimension2".
                double sum = 0;
                for (auto neighbour: neighbours) {
                    sum += time_series[neighbour + config.delay_vector_pattern[dimension1]] * time_series[neighbour + config.delay_vector_pattern[dimension2]];
                }

                // The following line uses Cov(X,Y) = E[XY]-E[X]E[X] which is susceptible to catastrophic cancellation.
                // Generally, it should be avoided if data hasn't been centered before (i.e., E[X]=E[Y]=0).
                covariance_matrix(dimension1, dimension2) =
                        (sum / neighbours.size() - average_neighbour[dimension1] * average_neighbour[dimension2]) *
                        weights[dimension1] * weights[dimension2];

                // The covariance matrix is symmetric, so we can save some work and store the result in the opposite entry as well.
                covariance_matrix(dimension2, dimension1) = covariance_matrix(dimension1, dimension2);
            }
        }

        // Calculate the eigenvectors of the covariance matrix.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
        auto eigen_vector_matrix = eigen_solver.eigenvectors();
        auto eigen_values = eigen_solver.eigenvalues();


        // Sort the eigenvalues in descending order.
        // Note, the book says "increaing order", but it is decreasing here (and therefore we will take the last instead of the first vectors).
        // Afterwards, eigenvalue_order contains the indices of the eigenvalues in descending order.
        // Thus, eigen_vector_matrix.cols(eigenvalue_order[0]) is the eigenvector to the largest eigenvalue and so on.
        std::vector<size_t> eigenvalue_order(embedding_dimension);
        for (size_t index = 0; index < embedding_dimension; index++) {
            eigenvalue_order[index] = index;
        }
        std::sort(eigenvalue_order.begin(),
                  eigenvalue_order.end(),
                  [&eigen_values](size_t a, size_t b) { return eigen_values[a] > eigen_values[b]; });


        // Now we calculate the correction delta.
        // We take the difference between the delay vector and the average of the neighbours and project it onto an eigen vector.
        // In formulas R^{-1} v(v \dot R(s-\bar{s})) where v is the eigenvector.
        // R denotes the weighting function.
        // This is equation 10.13 on page 182 of the Kantz book.
        // We do this projection for all but the qdim largest eigenvectors and sum up the result.
        DelayVector result(embedding_dimension);
        for (long dimension1 = 0; dimension1 < embedding_dimension; dimension1++) {
            double sum = 0;
            for (long projection_dimension = config.projection_dimension;
                    projection_dimension < embedding_dimension; projection_dimension++) {

                long eigen_vector_index = eigenvalue_order[projection_dimension];

                for (long dimension2 = 0; dimension2 < embedding_dimension; dimension2++) {

                    sum += (time_series[index_to_correct + config.delay_vector_pattern[dimension2]] - average_neighbour[dimension2])
                           * eigen_vector_matrix(dimension2, eigen_vector_index) *
                           eigen_vector_matrix(dimension1, eigen_vector_index)
                           * weights[dimension2];
                }
            }
            result[dimension1] = sum / weights[dimension1];
        }

        return result;
    }


    // This function deals with the issue of a shift due to curvature that is illustrated in Figure 10.3 on page 184 of the Kantz book.
    // The method used here is NOT the one from the Kantz book but the one from the Grassberger et al paper.
    // This is the point where the Kantz book ("project" binary in TISEAN) and Grassberger et al paper ("ghkss" binary in TISEAN) differ.
    void handle_trend(
            const size_t index_to_correct,          // The index for which correction is applied. Given as index of the time series.
            const std::vector<SequenceIndex>& neighbours,
            const DelayVector& weights,           // Array of weights that specify how much each dimension of the embedding should be considered.
            const Eigen::VectorXd& traces, // should contain the sum of the reciprocal values of weights.
            const std::vector<DelayVector>& correction_deltas, // the output of the make_corrections function. It's the \delta s_n of equation 10.13 (page 182 in Kantz book).
            const GhkssConfig& config,
            Eigen::VectorXd& final_deltas // the accumulation of the output. This
                     ) {

        for (long dimension = 0; dimension < config.delay_vector_pattern.size(); dimension++) {

            // First we calculate the average correction that is applied across all of our neighbours in the given dimension.
            double average_correction = 0;
            for (auto neighbour: neighbours) {
                average_correction += correction_deltas[neighbour / config.delay_vector_alignment][dimension];
            }
            average_correction = average_correction / neighbours.size();

            // Now we subtract the average correction from our own correction.
            final_deltas[index_to_correct + config.delay_vector_pattern[dimension]] +=
                    (correction_deltas[index_to_correct / config.delay_vector_alignment][dimension] - average_correction) /
                    (traces[config.delay_vector_pattern[dimension]%config.delay_vector_alignment] * weights[dimension]);
        }
    }


    TimeSeries filter_ghkss(const TimeSeries& time_series, const GhkssConfig& config, NeighbourStatistics* neighbour_statistics) {

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.time_series = time_series;
#endif

        if (neighbour_statistics != nullptr) {
            neighbour_statistics->minimum_neighbour_count = std::numeric_limits<size_t>::max();
            neighbour_statistics->maximum_neighbour_count = 0;
        }

        if (config.delay_vector_pattern.empty()) {
            throw std::runtime_error("The delay vector pattern must be set.");
        }

        if (time_series.size() <= std::ranges::max(config.delay_vector_pattern)) {
            throw std::runtime_error("The time series must be longer than the maximum entry in the delay vector pattern.");
        }

        // Set the delay vector weights. We just use the suggested values from the book / TISEAN here.
        // Maybe this should be configurable later on.
        DelayVector delay_vector_weights = get_delay_vector_weights(config);

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.delay_vector_weights = delay_vector_weights;
#endif

        // Look for the nearest neighbours.
        if ( config.verbosity >= verbosity_high) {
            std::cerr << "Finding nearest neighbours..." << std::endl;
        }
        if (config.verbosity >= verbosity_debug) {
            std::cerr << "Building KD-tree..." << std::endl;
        }
        KNearestNeighbourFinder knn_finder(time_series.data(), time_series.size(), config.delay_vector_pattern, config.delay_vector_alignment);
        size_t delay_vector_count;
        if (time_series.size() + 1 < std::ranges::max(config.delay_vector_pattern)) {
            delay_vector_count = 0;
        } else {
            delay_vector_count = (time_series.size() - std::ranges::max(config.delay_vector_pattern) - 1) / config.delay_vector_alignment + 1;
        }
#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.delay_vector_count = delay_vector_count;
#endif
        if (config.verbosity >= verbosity_debug) {
            std::cerr << "Finding neighbours for " << delay_vector_count << " delay vectors" << std::endl;
        }
        std::vector<std::vector<SequenceIndex>> all_neighbours;
        all_neighbours.reserve(delay_vector_count);
        for (size_t delay_index = 0; delay_index < delay_vector_count; delay_index++) {
            std::vector<SequenceIndex> neighbours;
            if (config.tisean_epsilon_widening) {
                auto epsilon = config.neighbour_epsilon;
                if (epsilon <= 0) {
                    throw std::runtime_error("When TISEAN style epsilon wideing is used, the initial epsilon value must be positive.");
                }
                if (knn_finder.size() < config.minimum_neighbour_count) {
                    throw std::runtime_error("The time series is too short to form the desired number of delay vectors.");
                }
                while (neighbours.size() < config.minimum_neighbour_count) {
                    if (config.verbosity >= verbosity_trace) {
                        std::cerr << "Looking for " << config.minimum_neighbour_count << " neighbours for delay vector " << delay_index << " within epsilon radius " << epsilon << "..." << std::endl;
                    }
                    neighbours = knn_finder.find_nearest_neighbours(delay_index * config.delay_vector_alignment,
                                                                    0,
                                                                    epsilon,
                                                                    -1,
                                                                    config.euclidean_norm);
                    epsilon *= sqrt(2);
                }
            } else {
                neighbours = knn_finder.find_nearest_neighbours(delay_index * config.delay_vector_alignment,
                                                                config.minimum_neighbour_count,
                                                                config.neighbour_epsilon,
                                                                config.maximum_neighbour_count,
                                                                config.euclidean_norm);
            }
            if (config.verbosity >= verbosity_trace) {
                std::cerr << "Found " << neighbours.size() << " neighbours for delay vector " << delay_index << std::endl;
            }

            all_neighbours.push_back(std::move(neighbours));
        }

        // Collect some statistics if requested.
        if (neighbour_statistics != nullptr) {
            size_t minimum_neighbour_count = std::numeric_limits<size_t>::max();
            size_t maximum_neighbour_count = 0;
            size_t total_neighbour_count = 0;
            for (auto& neighbours: all_neighbours) {
                minimum_neighbour_count = std::min(minimum_neighbour_count, neighbours.size());
                maximum_neighbour_count = std::max(maximum_neighbour_count, neighbours.size());
                total_neighbour_count += neighbours.size();
            }
            neighbour_statistics->minimum_neighbour_count = minimum_neighbour_count;
            neighbour_statistics->maximum_neighbour_count = maximum_neighbour_count;
            neighbour_statistics->average_neighbour_count = double(total_neighbour_count) / double(all_neighbours.size());
            if (config.verbosity >= verbosity_high) {
                std::cerr << "Neighbour statistics: Minimum neighbour count: " << neighbour_statistics->minimum_neighbour_count << ", Maximum neighbour count: " << neighbour_statistics->maximum_neighbour_count << ", Average neighbour count: " << neighbour_statistics->average_neighbour_count << std::endl;
            }
        }

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.all_neighbours = all_neighbours;
#endif

        // Calculate the preliminary correction deltas
        if ( config.verbosity >= verbosity_high) {
            std::cerr << "Calculating preliminary correction deltas..." << std::endl;
        }
        std::vector<DelayVector> initial_correction_deltas;
        initial_correction_deltas.reserve(delay_vector_count);
        for (size_t delay_index = 0; delay_index < delay_vector_count; delay_index++) {
            initial_correction_deltas.push_back(calculate_correction_deltas(delay_index*config.delay_vector_alignment,
                                                                            all_neighbours[delay_index],
                                                                            time_series,
                                                                            delay_vector_weights,
                                                                            config));
        }

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.initial_correction_deltas = initial_correction_deltas;
#endif

        // Handle the trend and calculate the final correction deltas.
        if ( config.verbosity >= verbosity_high) {
            std::cerr << "Handling trend and calculating final correction deltas..." << std::endl;
        }
        Eigen::VectorXd traces = Eigen::VectorXd::Zero(config.delay_vector_alignment);
        for (auto index=0; index<delay_vector_weights.size(); index++) {
            traces[config.delay_vector_pattern[index] % config.delay_vector_alignment] += 1. / delay_vector_weights[index];
        }

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.traces = traces;
#endif

        Eigen::VectorXd final_deltas = Eigen::VectorXd::Zero(time_series.size());
        for (size_t delay_index = 0; delay_index < delay_vector_count; delay_index++) {
            handle_trend(delay_index*config.delay_vector_alignment,
                         all_neighbours[delay_index],
                         delay_vector_weights,
                         traces,
                         initial_correction_deltas,
                         config,
                         final_deltas);
        }

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.final_deltas = final_deltas;
#endif

        // Now we do the actual correction.
        if ( config.verbosity >= verbosity_high) {
            std::cerr << "Applying final correction deltas..." << std::endl;
        }
        TimeSeries filtered_time_series;
        filtered_time_series.reserve(time_series.size());
        for (long index = 0; index < time_series.size(); index++) {
            filtered_time_series.push_back(time_series[index] - final_deltas[index]);
        }

#ifdef WITH_DEBUG_INFO
        DebugInfo::instance().data.filtered_time_series = filtered_time_series;
#endif

        if ( config.verbosity >= verbosity_high) {
            std::cerr << "One round of filtering completed." << std::endl;
        }
        return filtered_time_series;
    }


#ifdef WITH_PYTHON
    void register_with_python(pybind11::module_& module) {

        static bool first_call = true;
        if (!first_call) {
            return;
        }
        first_call = false;

        using namespace pybind11::literals;

        module.attr("verbosity_none") = verbosity_none;
        module.attr("verbosity_info") = verbosity_info;
        module.attr("verbosity_high") = verbosity_high;
        module.attr("verbosity_debug") = verbosity_debug;
        module.attr("verbosity_trace") = verbosity_trace;

        pybind11::class_<GhkssConfig>(module, "GhkssConfig")
                .def(pybind11::init())
                .def_readwrite("delay_vector_pattern", &GhkssConfig::delay_vector_pattern)
                .def_readwrite("delay_vector_alignment", &GhkssConfig::delay_vector_alignment)
                .def_readwrite("projection_dimension", &GhkssConfig::projection_dimension)
                .def_readwrite("minimum_neighbour_count", &GhkssConfig::minimum_neighbour_count)
                .def_readwrite("neighbour_epsilon", &GhkssConfig::neighbour_epsilon)
                .def_readwrite("tisean_epsilon_widening", &GhkssConfig::tisean_epsilon_widening)
                .def_readwrite("maximum_neighbour_count", &GhkssConfig::maximum_neighbour_count)
                .def_readwrite("euclidean_norm", &GhkssConfig::euclidean_norm)
                .def_readwrite("verbosity", &GhkssConfig::verbosity)
                ;

        pybind11::class_<NeighbourStatistics>(module, "NeighbourStatistics")
                .def(pybind11::init())
                .def_readwrite("minimum_neighbour_count", &NeighbourStatistics::minimum_neighbour_count)
                .def_readwrite("maximum_neighbour_count", &NeighbourStatistics::maximum_neighbour_count)
                .def_readwrite("average_neighbour_count", &NeighbourStatistics::average_neighbour_count)
                ;

        module.def("filter_ghkss", [](const TimeSeries& time_series, const GhkssConfig& config, const bool return_neighbour_statistics) -> std::variant<TimeSeries, std::pair<TimeSeries,NeighbourStatistics>> {
                       if (return_neighbour_statistics) {
                           NeighbourStatistics neighbour_statistics;
                           auto result = filter_ghkss(time_series, config, &neighbour_statistics);
                           return std::make_pair(std::move(result), neighbour_statistics);
                       } else {
                           return filter_ghkss(time_series, config);
                       }
                   },
                   "Filter the time series using the GHKSS method.", "time_series"_a, "config"_a = GhkssConfig(), "return_neighbour_statistics"_a = false);

        // Expose some internal functions for debugging and testing purposes.

        module.def("_get_delay_vector_weights", [](const GhkssConfig& config) {return get_delay_vector_weights(config);}, "config"_a);

        module.def("_calculate_correction_deltas", [](
                const size_t index_to_correct,          // The index for which correction is applied
                const std::vector<SequenceIndex>& neighbours,
                const std::vector<double>& time_series,
                const DelayVector& weights,           // Array of weights that specify how much each dimension of the embedding should be considered.
                const GhkssConfig& config
                                                     ) {
            if (weights.size() != config.delay_vector_pattern.size()) {
                throw std::runtime_error("The weights vector must have the same length as the delay vector pattern.");
            }
            if (neighbours.empty()) {
                throw std::runtime_error("The neighbours vector must not be empty.");
            }
            for (auto offset : config.delay_vector_pattern) {
                if (index_to_correct + offset >= time_series.size()) {
                    throw std::runtime_error("The index to correct is out of bounds.");
                }
                for (auto neighbour : neighbours) {
                    if (neighbour + offset >= time_series.size()) {
                        throw std::runtime_error("A neighbour index is out of bounds.");
                    }
                }
            }
            return calculate_correction_deltas(index_to_correct, neighbours, time_series, weights, config);
        },
        "index_to_correct"_a, "neighbours"_a, "time_series"_a, "weights"_a, "config"_a);

        module.def("_handle_trend", [](
                const size_t index_to_correct,          // The index for which correction is applied
                const std::vector<SequenceIndex>& neighbours,
                const DelayVector& weights,           // Array of weights that specify how much each dimension of the embedding should be considered.
                const Eigen::VectorXd& traces, // should contain the sum of the reciprocal values of weights.
                const std::vector<DelayVector>& correction_deltas, // the output of the make_corrections function. It's the \delta s_n of equation 10.13 (page 182 in Kantz book).
                const GhkssConfig& config,
                Eigen::VectorXd final_deltas // Pybind creates a copy on invocation. So we return the modified results. Inefficient, but it is only for unit testing anyway.
                ) {

            fast_uint max_offset = 0;
            for (auto offset : config.delay_vector_pattern) {
                max_offset = std::max(max_offset, offset);
            }
            if (index_to_correct + max_offset >= final_deltas.size()) {
                throw std::runtime_error("The index to correct is out of bounds.");
            }
            for (auto neighbour : neighbours) {
                if (neighbour + max_offset >= final_deltas.size()) {
                    throw std::runtime_error("A neighbour index is out of bounds.");
                }
            }
            if (weights.size() != config.delay_vector_pattern.size()) {
                throw std::runtime_error("The weights vector must have the same length as the delay vector pattern.");
            }
            if (traces.size() != config.delay_vector_alignment) {
                throw std::runtime_error("The traces vector must have the same length as the delay vector alignment.");
            }
            if (final_deltas.size() < max_offset+1){
                if (!correction_deltas.empty()) {
                    throw std::runtime_error("There are too few elements for corrections to happen but the correction deltas vector is not empty.");
                }
            } else if (correction_deltas.size() != (final_deltas.size()-max_offset-1)/config.delay_vector_alignment+1) {
                throw std::runtime_error("There must be a correction delta for each delay vector.");
            }
            for (const auto& correction_delta : correction_deltas) {
                if (correction_delta.size() != config.delay_vector_pattern.size()) {
                    throw std::runtime_error("Each correction deltas vector must have the same length as the delay vector pattern.");
                }
            }

            handle_trend(
                    index_to_correct,
                    neighbours,
                    weights,
                    traces,
                    correction_deltas,
                    config,
                    final_deltas
            );
            return final_deltas;
        },
        "index_to_correct"_a, "neighbours"_a, "weights"_a, "traces"_a, "correction_deltas"_a, "config"_a, "final_deltas"_a);

        // Make the KNearestNeighbourFinder class visible to Python. Not really necessary
        // but it may be useful and also helps debugging.
        KNearestNeighbourFinder::register_with_python(module);

#ifdef WITH_DEBUG_INFO
        DebugInfo::register_with_python(module);
#endif
    }
#endif




} // namespace