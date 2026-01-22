#include "knn_kdtree.h"

#include <queue>
#include <set>
#include <Eigen/Dense>


#ifdef WITH_PYTHON
#include <pybind11/stl.h>
#endif

namespace ghkss {

#ifdef WITH_PYTHON
    void KNearestNeighbourFinder::register_with_python(pybind11::module_& module)
    {
        namespace py = pybind11;
        using namespace pybind11::literals;

        py::class_<KNearestNeighbourFinder>(module, "KNearestNeighbourFinder")

            .def(py::init([](const std::vector<double>& sequence, const std::vector<fast_uint>& delay_vector_pattern, const int delay_vector_alignment){
                return std::make_unique<KNearestNeighbourFinder>(sequence.data(), sequence.size(), delay_vector_pattern, delay_vector_alignment);
            }), "sequence"_a, "delay_vector_pattern"_a, "delay_vector_alignment"_a=1)


            .def("find_nearest_neighbours", &KNearestNeighbourFinder::find_nearest_neighbours, "index"_a, "minimum_neighbour_count"_a, "neighbour_epsilon"_a = -1, "maximum_neighbour_count"_a = std::numeric_limits<fast_uint>::max(), "euclidean_norm"_a = false)

            .def("find_all_nearest_neighbours", [](const KNearestNeighbourFinder& self, const int minimum_neighbour_count, bool euclidean_norm){
                std::vector<std::vector<fast_uint>> result;
                for (long index=0; index+self.max_delay_vector_offset<self.sequence.size(); index++) {
                    result.push_back(self.find_nearest_neighbours(index, minimum_neighbour_count, -1, std::numeric_limits<fast_uint>::max(), euclidean_norm));
                }
                return result;

            }, "minimum_neighbour_count"_a, "euclidean_norm"_a = false)

            .def("assert_knn_result", [](const KNearestNeighbourFinder& self, const fast_uint query, const std::vector<fast_uint>& result, const fast_uint minimum_neighbour_count, const SampleType neighbour_epsilon, const fast_uint maximum_neighbour_count, bool euclidean_norm) {
                try {
                    self.assert_knn_result(query, result, minimum_neighbour_count, neighbour_epsilon, maximum_neighbour_count, euclidean_norm);
                } catch (const std::runtime_error& e) {
                    // Convert the exception to a proper AssertionError in Python.
                    PyErr_SetString(PyExc_AssertionError, e.what());
                    throw py::error_already_set();
                }
            }, "query"_a, "result"_a, "minimum_neighbour_count"_a, "neighbour_epsilon"_a = -1, "maximum_neighbour_count"_a = std::numeric_limits<fast_uint>::max(), "euclidean_norm"_a = false);

#ifdef CREATE_DEBUG_STATS
            .def_readwrite("visited_nodes", &KNearestNeighbourFinder::visited_nodes)
            .def_readwrite("visited_leaves", &KNearestNeighbourFinder::visited_leaves)
            .def_readwrite("accepted_neighbour_candidates", &KNearestNeighbourFinder::accepted_neighbour_candidates)
            .def("reset_debug_stats", [](KNearestNeighbourFinder& self) {
                self.visited_nodes = 0;
                self.visited_leaves = 0;
                self.accepted_neighbour_candidates = 0;
            })
            .def("print_debug_stats", [](KNearestNeighbourFinder& self) {
                std::cout << "visited_nodes: " << self.visited_nodes << std::endl;
                std::cout << "visited_leaves: " << self.visited_leaves << std::endl;
                std::cout << "accepted_neighbour_candidates: " << self.accepted_neighbour_candidates << std::endl;
            })
#endif

            ;
    }
#endif

    KNearestNeighbourFinder::KNearestNeighbourFinder(const double* sequence_, fast_uint sequence_length, const std::vector<fast_uint>& delay_vector_pattern_, fast_uint delay_vector_alignment_):
        delay_vector_pattern(delay_vector_pattern_),
        delay_vector_alignment(delay_vector_alignment_),
        max_delay_vector_offset(std::ranges::max(delay_vector_pattern)),
        delay_vector_count((sequence_length-std::ranges::max(delay_vector_pattern)) / delay_vector_alignment_),
        sequence(sequence_, sequence_+sequence_length) {

        if (sequence_length <= max_delay_vector_offset) {
            throw std::runtime_error("The sequence length must be larger than the delay vector dimension.");
        }
        if (delay_vector_pattern.size() < 1) {
            throw std::runtime_error("The delay vector dimension must be at least 1.");
        }

        build_kdtree();
    }

    void CandidateSorter::add(const SampleType distance, const fast_uint index) {
        if (max_candidates == 0 || candidates.front().distance < minimum_epsilon_distance) {
            // We have filled our desired number of candidates with ones that are within the epsilon acceptance bounds.
            // We abandon the heap structure and just append whatever we get that is also within the epsilon bounds.
            if (distance < minimum_epsilon_distance) {
                candidates.push_back(Candidate(distance, index));
            }

        } else if ( distance < candidates.front().distance ) {
            // The candidate should replace the largest existing candidate
            candidates.front().distance = distance;
            candidates.front().index = index;

            // Perform sift-down to maintain max-heap order
            fast_uint current = 0;
            while (true) {
                fast_uint left_child = 2 * current + 1;
                fast_uint right_child = 2 * current + 2;
                fast_uint largest = current;

                if ( left_child < max_candidates && candidates[left_child] > candidates[largest] ) {
                    largest = left_child;
                }
                if ( right_child < max_candidates && candidates[right_child] > candidates[largest] ) {
                    largest = right_child;
                }

                if ( largest == current ) {
                    break;
                }

                std::swap(candidates[current], candidates[largest]);
                current = largest;
            }
        }
    }

    void KNearestNeighbourFinder::search_nodes_recursive(fast_uint node_index,
                                                         const SampleType* const query,
                                                         CandidateSorter& candidates,
                                                         SampleType box_distance) const {

        //std::cout << "Searching node " << node_index << " with a box distance of " << box_distance << " and candidate distance of " << candidates.max_distance() << std::endl;

#ifdef CREATE_DEBUG_STATS
        visited_nodes++;
#endif

        const auto& the_node = tree[node_index];
        if (std::holds_alternative<LeafNode>(tree[node_index])) {

            // We are in a leaf node. Check the contained delay vector and add it to the candidates if applicable.
#ifdef CREATE_DEBUG_STATS
            visited_leaves++;
#endif
            const LeafNode& node = std::get<LeafNode>(tree[node_index]);
            SampleType distance = 0;
            for (auto dimension : delay_vector_pattern) {
                auto dx = std::fabs(sequence[node.index+dimension] - query[dimension]);
                if (candidates.max_distance() < dx) {
                    //std::cout << "Rejecting candidate " << node.index << " (node " << node_index << ") with distance at least " << dx << std::endl;
                    return;
                }
                if (distance < dx) {
                    distance = dx;
                }
            }
            //std::cout << "Adding candidate " << node.index << " (node " << node_index << ") with distance " << distance << std::endl;
            candidates.add(distance, node.index);
#ifdef CREATE_DEBUG_STATS
            accepted_neighbour_candidates++;
#endif

        } else {

            // We are in a split node. Recurse into its children.
            const SplitNode& node = std::get<SplitNode>(tree[node_index]);

            auto query_value = query[delay_vector_pattern[node.dimension]];
            auto cut_distance = query_value - node.split_value;
            if (cut_distance > 0) {
                // We want to process the larger child first.
                auto child_distance = std::max(box_distance, query_value - node.max_value);
                if (child_distance < candidates.max_distance()) {
                    search_nodes_recursive(node.upper_child, query, candidates, child_distance);
                }
                child_distance = std::max(box_distance, cut_distance);
                if (child_distance < candidates.max_distance()) {
                    search_nodes_recursive(node.lower_child, query, candidates, child_distance);
                }

            } else {
                // We want to process the smaller child first
                auto child_distance = std::max(box_distance, node.min_value - query_value);
                if (child_distance < candidates.max_distance()) {
                    search_nodes_recursive(node.lower_child, query, candidates, child_distance);
                }
                child_distance = std::max(box_distance, -cut_distance);
                if (child_distance < candidates.max_distance()) {
                    search_nodes_recursive(node.upper_child, query, candidates, child_distance);
                }
            }
            //std::cout << "Finished searching node " << node_index << " with a box distance of " << box_distance << " and candidate distance of " << candidates.max_distance() << std::endl;
        }

    }

    namespace {
        template<typename T>
        T sqr(T x) {
            return x*x;
        }
    }


    void KNearestNeighbourFinder::search_nodes_recursive_euclidean(fast_uint node_index,
                                                         const SampleType* const query,
                                                         CandidateSorter& candidates,
                                                         SampleType box_distance,
                                                         SampleType* box_distance_per_dimension) const {

        //std::cout << "Searching node " << node_index << " with a box distance of " << box_distance << " and candidate distance of " << candidates.max_distance() << std::endl;

#ifdef CREATE_DEBUG_STATS
        visited_nodes++;
#endif

        const auto& the_node = tree[node_index];
        if (std::holds_alternative<LeafNode>(tree[node_index])) {

            // We are in a leaf node. Check the contained delay vector and add it to the candidates if applicable.
#ifdef CREATE_DEBUG_STATS
            visited_leaves++;
#endif
            const LeafNode& node = std::get<LeafNode>(tree[node_index]);
            SampleType distance = 0;
            for (auto dimension : delay_vector_pattern) {
                distance += sqr(sequence[node.index+dimension] - query[dimension]);
                if (candidates.max_distance() < distance) {
                    //std::cout << "Rejecting candidate " << node.index << " (node " << node_index << ") with distance at least " << dx << std::endl;
                    return;
                }
            }
            //std::cout << "Adding candidate " << node.index << " (node " << node_index << ") with distance " << distance << std::endl;
            candidates.add(distance, node.index);
#ifdef CREATE_DEBUG_STATS
            accepted_neighbour_candidates++;
#endif

        } else {

            // We are in a split node. Recurse into its children.
            const SplitNode& node = std::get<SplitNode>(tree[node_index]);

            auto query_value = query[delay_vector_pattern[node.dimension]];
            auto cut_distance = query_value - node.split_value;
            auto our_dimension_distance_to_restore = box_distance_per_dimension[node.dimension];
            auto other_dimension_distances = box_distance - our_dimension_distance_to_restore;
            if (cut_distance > 0) {
                // We want to process the larger child first.
                decltype(query_value) our_dimension_distance;
                if (query_value <= node.max_value) {
                    our_dimension_distance = 0;
                } else {
                    our_dimension_distance = sqr(query_value - node.max_value);
                }
                auto child_distance = other_dimension_distances + our_dimension_distance;
                if (child_distance < candidates.max_distance()) {
                    box_distance_per_dimension[node.dimension] = our_dimension_distance;
                    search_nodes_recursive_euclidean(node.upper_child, query, candidates, child_distance, box_distance_per_dimension);
                }
                our_dimension_distance = sqr(cut_distance);
                child_distance = other_dimension_distances + our_dimension_distance;
                if (child_distance < candidates.max_distance()) {
                    box_distance_per_dimension[node.dimension] = our_dimension_distance;
                    search_nodes_recursive_euclidean(node.lower_child, query, candidates, child_distance, box_distance_per_dimension);
                }

            } else {
                // We want to process the smaller child first
                decltype(query_value) our_dimension_distance;
                if (query_value < node.min_value) {
                    our_dimension_distance = sqr(node.min_value - query_value);
                } else {
                    our_dimension_distance = 0;
                }
                auto child_distance = other_dimension_distances + our_dimension_distance;
                if (child_distance < candidates.max_distance()) {
                    box_distance_per_dimension[node.dimension] = our_dimension_distance;
                    search_nodes_recursive_euclidean(node.lower_child, query, candidates, child_distance, box_distance_per_dimension);
                }
                our_dimension_distance = sqr(cut_distance);
                child_distance = other_dimension_distances + our_dimension_distance;
                if (child_distance < candidates.max_distance()) {
                    box_distance_per_dimension[node.dimension] = our_dimension_distance;
                    search_nodes_recursive_euclidean(node.upper_child, query, candidates, child_distance, box_distance_per_dimension);
                }
            }
            box_distance_per_dimension[node.dimension] = our_dimension_distance_to_restore; // We must restore this to its previous value for the caller
            //std::cout << "Finished searching node " << node_index << " with a box distance of " << box_distance << " and candidate distance of " << candidates.max_distance() << std::endl;
        }

    }



    std::vector<fast_uint> KNearestNeighbourFinder::find_nearest_neighbours(const fast_uint delay_vector, const fast_uint minimum_neighbour_count, const SampleType neighbour_epsilon, const fast_uint maximum_neighbour_count, bool euclidean_norm) const {

        if (delay_vector + max_delay_vector_offset >= sequence.size()) {
            throw std::runtime_error("The delay vector index is out of bounds.");
        }

        if (delay_vector % delay_vector_alignment != 0) {
            throw std::runtime_error("The delay vector index is not correctly aligned.");
        }

        auto effective_neighbour_epsilon = neighbour_epsilon;
        if (euclidean_norm) {
            effective_neighbour_epsilon *= abs(effective_neighbour_epsilon);
        }
        CandidateSorter candidates(minimum_neighbour_count, effective_neighbour_epsilon);

        if (euclidean_norm) {
            std::vector<SampleType> box_distance_per_dimension(delay_vector_pattern.size(),0);
            search_nodes_recursive_euclidean(root, &sequence[delay_vector], candidates, 0, box_distance_per_dimension.data());
        } else {
           search_nodes_recursive(root, &sequence[delay_vector], candidates, 0);
        }

        std::vector<fast_uint> result = candidates.get_candidates();

        if (result.size() > maximum_neighbour_count) {
            // We have more than the maximum number of candidates.
            // We keep the ones that are closest in time to the query delay vector.
            // Close in time means the indices are close to each other.
            std::sort(result.begin(), result.end());
            auto begin = result.begin();
            auto end = result.end();
            while (end-begin > maximum_neighbour_count) {
                if (abs(int(delay_vector)-int(*begin)) < abs(int(delay_vector)-int(*(end-1)))) {
                    end--;
                } else {
                    begin++;
                }
            }
            std::vector<fast_uint> culled_result(begin, end);
            result = std::move(culled_result);
        }

        return result;
    }


    void KNearestNeighbourFinder::build_kdtree() {
        tree.clear();

        // Work out the bounding box.
        bounding_box.min.resize(delay_vector_pattern.size());
        bounding_box.min.setConstant(std::numeric_limits<SampleType>::infinity());
        bounding_box.max.resize(delay_vector_pattern.size());
        bounding_box.max.setConstant(-std::numeric_limits<SampleType>::infinity());

        for (fast_uint index = 0; index < delay_vector_count; index++) {
            for (fast_uint dimension = 0; dimension < delay_vector_pattern.size(); dimension++) {
                bounding_box.min[dimension] = std::min(bounding_box.min[dimension], sequence[index*delay_vector_alignment + delay_vector_pattern[dimension]]);
                bounding_box.max[dimension] = std::max(bounding_box.max[dimension], sequence[index*delay_vector_alignment + delay_vector_pattern[dimension]]);
            }
        }

        std::vector<fast_uint> indices;
        for (fast_uint index = 0; index < delay_vector_count; index++) {
            indices.push_back(index*delay_vector_alignment);
        }

        root = create_nodes_recursive(bounding_box, indices.data(), indices.size());
    }


    fast_uint KNearestNeighbourFinder::create_nodes_recursive(const KNearestNeighbourFinder::BoundingBox& bounds,
                                                              fast_uint* indices,
                                                              const fast_uint number_of_points) {


        if (number_of_points == 0) {
            throw std::runtime_error("The number of points is zero. This should not happen.");
        }

        if (number_of_points == 1) {
            // Create a leaf node.
            LeafNode node;
            node.index = indices[0];
            tree.push_back(node);
            return tree.size()-1;
        }

        // get the cutting point
        fast_uint cut_dimension = 0;
        SampleType largest_dimension_size = 0;
        for (fast_uint dimension = 0; dimension < delay_vector_pattern.size(); dimension++) {
            auto size = bounds.max[dimension] - bounds.min[dimension];
            if (size > largest_dimension_size) {
                largest_dimension_size = size;
                cut_dimension = dimension;
            }
        }


        // local helper function to partition the array to a pivot
        auto partition = [&indices,this](fast_uint low, fast_uint high, fast_uint cut_dimension) {
            fast_uint pivot = low + (high - low) / 2;
            std::swap(indices[pivot], indices[high]);

            auto end_of_low = low;
            for (auto current = end_of_low; current < high; current++) {
                if (sequence[indices[current]+delay_vector_pattern[cut_dimension]] < sequence[indices[high]+delay_vector_pattern[cut_dimension]]) {
                    std::swap(indices[current], indices[end_of_low]);
                    end_of_low++;
                }
            }
            std::swap(indices[end_of_low], indices[high]);
            return end_of_low;
        };

        fast_uint low_node_count = (number_of_points + 1) / 2; // we always want to round up
        fast_uint high_node_count = number_of_points - low_node_count;

        {
            fast_uint low = 0;
            fast_uint high = number_of_points - 1;
            while (true) {
                auto pivot_index = partition(low, high, cut_dimension);
                if ( pivot_index == low_node_count || pivot_index + 1 == low_node_count ) {
                    break;
                } else if ( pivot_index < low_node_count ) {
                    low = pivot_index + 1;
                } else {
                    high = pivot_index - 1;
                }
            }
        }

        // Create the child nodes
        auto child_bounds = bounds;

        SampleType cut_value;
        SampleType global_min_value;
        SampleType global_max_value;

        {
            SampleType min_value = std::numeric_limits<SampleType>::infinity();
            SampleType max_value = -std::numeric_limits<SampleType>::infinity();
            for (fast_uint index = 0; index < low_node_count; index++) {
                auto value = sequence[indices[index] + delay_vector_pattern[cut_dimension]];
                if ( value < min_value ) {
                    min_value = value;
                }
                if ( value > max_value ) {
                    max_value = value;
                }
            }
            child_bounds.max[cut_dimension] = max_value;
            child_bounds.min[cut_dimension] = min_value;
            cut_value = max_value;
            global_min_value = min_value;
        }

        auto low_child = create_nodes_recursive(child_bounds, indices, low_node_count);

        {
            SampleType min_value = std::numeric_limits<SampleType>::infinity();
            SampleType max_value = -std::numeric_limits<SampleType>::infinity();
            for (fast_uint index = low_node_count; index < number_of_points; index++) {
                auto value = sequence[indices[index] + delay_vector_pattern[cut_dimension]];
                if ( value < min_value ) {
                    min_value = value;
                }
                if ( value > max_value ) {
                    max_value = value;
                }
            }
            child_bounds.max[cut_dimension] = max_value;
            child_bounds.min[cut_dimension] = min_value;
            cut_value += (min_value -cut_value) / 2;
            global_max_value = max_value;
        }

        auto high_child = create_nodes_recursive(child_bounds, indices+low_node_count, high_node_count);

        // Create the node
        SplitNode node;
        node.lower_child = low_child;
        node.upper_child = high_child;
        node.dimension = cut_dimension;
        node.split_value = cut_value;
        node.min_value = global_min_value;
        node.max_value = global_max_value;
        tree.push_back(node);

        return tree.size()-1;

    }

    void KNearestNeighbourFinder::assert_knn_result(const fast_uint query, const std::vector<fast_uint>& result, const fast_uint minimum_neighbour_count, const SampleType neighbour_epsilon, const fast_uint maximum_neighbour_count, bool euclidean_norm) const {
        std::set<fast_uint> result_set(result.begin(), result.end());

        if (result_set.size() < minimum_neighbour_count && result_set.size() < delay_vector_count) {
            throw std::runtime_error("The result set is too small. It only contains " + std::to_string(result_set.size()) + " elements.");
        }

        if (result_set.size() > maximum_neighbour_count) {
            throw std::runtime_error("The result set is too large. It contains " + std::to_string(result_set.size()) + " elements.");
        }

        SampleType max_result_distance = 0;
        SampleType min_non_result_distance = std::numeric_limits<SampleType>::infinity();

        for (fast_uint index = 0; index + max_delay_vector_offset < sequence.size(); index += delay_vector_alignment) {
            SampleType distance = 0;
            for (fast_uint dimension : delay_vector_pattern) {
                if (euclidean_norm) {
                    distance += sqr(sequence[index + dimension] - sequence[query + dimension]);
                } else {
                    distance = std::max(distance, std::fabs(sequence[index + dimension] - sequence[query + dimension]));
                }
            }
            if (euclidean_norm) {
                distance = std::sqrt(distance);
            }
            if (result_set.contains(index)) {
                max_result_distance = std::max(max_result_distance, distance);
            } else {
                min_non_result_distance = std::min(min_non_result_distance, distance);
            }
        }

        if (min_non_result_distance < neighbour_epsilon && result_set.size() < maximum_neighbour_count) {
            throw std::runtime_error("There are vectors within the epsilon radius that are not in the result set (and the result set is smaller than the maximum number of neighbours).");
        }

        if (result_set.size() > minimum_neighbour_count && max_result_distance > neighbour_epsilon) {
            throw std::runtime_error("There are vectors in the result set that are further away than the epsilon radius and the result set is larger than the minimum number of neighbours.");
        }

        if (max_result_distance > min_non_result_distance && result_set.size() < maximum_neighbour_count) {
            throw std::runtime_error("The results contain " + std::to_string(result.size()) + " neighbours with a maximum distance of "
                + std::to_string(max_result_distance) + ". But there are other vectors with a minimum distance of "
                + std::to_string(min_non_result_distance) + ".");
        }

    }

} // namespace