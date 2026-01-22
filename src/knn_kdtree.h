#pragma once

#include <algorithm>
#include <stdint.h>
#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <memory>
#include <ostream>
#include <variant>

#ifdef WITH_PYTHON
#include <pybind11/pybind11.h>
#endif

#include <Eigen/Core>

//#define CREATE_DEBUG_STATS


namespace ghkss {

typedef uint_fast32_t fast_uint;
typedef double SampleType;


class CandidateSorter {
private:
    struct Candidate {
        SampleType distance;
        fast_uint index;
        bool operator>(const Candidate& other) const { return distance > other.distance; }
    };
    constexpr static fast_uint no_candidate = std::numeric_limits<fast_uint>::max();
public:
    CandidateSorter(const fast_uint size, const SampleType minimum_epsilon_distance_ = -1) :
        max_candidates(size),
        minimum_epsilon_distance(minimum_epsilon_distance_),
        candidates(size, Candidate{std::numeric_limits<SampleType>::infinity(), no_candidate})
        {}

    SampleType max_distance() const { return max_candidates == 0 ? minimum_epsilon_distance : std::max(candidates.front().distance, minimum_epsilon_distance); }

    void add(const SampleType distance, const fast_uint index);

    std::vector<fast_uint> get_candidates() const {
        std::vector<fast_uint> result;
        for (const auto& candidate : candidates) {
            if (candidate.index != no_candidate) {
                result.push_back(candidate.index);
            }
        }
        return result;
    }
private:
    fast_uint max_candidates = 0;
    SampleType minimum_epsilon_distance = -1;
    std::vector<Candidate> candidates;

};

class KNearestNeighbourFinder {
private:
    struct SplitNode
    {
        fast_uint lower_child;
        fast_uint upper_child;
        fast_uint dimension;
        SampleType split_value;
        SampleType min_value;
        SampleType max_value;
    };
    struct LeafNode {
        fast_uint index;
    };
    typedef std::variant<SplitNode, LeafNode> TreeNode;
    constexpr static fast_uint no_child = std::numeric_limits<fast_uint>::max();

    struct BoundingBox {
        Eigen::VectorXd min;
        Eigen::VectorXd max;
    };

public:
    KNearestNeighbourFinder(const double* sequence, fast_uint sequence_length, const std::vector<fast_uint>& delay_vector_pattern_, fast_uint delay_vector_alignment_=1);
    fast_uint size() const { return delay_vector_count; }
    std::vector<fast_uint> find_nearest_neighbours(const fast_uint delay_vector, const fast_uint minimum_neighbour_count, const SampleType neighbour_epsilon = -1, const fast_uint maximum_neighbour_count = std::numeric_limits<fast_uint>::max(), bool euclidean_norm=false) const;

#ifdef WITH_PYTHON
    static void register_with_python(pybind11::module_& module);
#endif

private:
    void build_kdtree();
    fast_uint create_nodes_recursive(const BoundingBox& bounds, fast_uint* indices, const fast_uint number_of_points);

    void assert_knn_result(const fast_uint query, const std::vector<fast_uint>& result, const fast_uint minimum_neighbour_count, const SampleType neighbour_epsilon, const fast_uint maximum_neighbour_count, bool euclidean_norm=false) const;

    void search_nodes_recursive(fast_uint node, const SampleType* const query, CandidateSorter& candidates, SampleType box_distance) const;
    void search_nodes_recursive_euclidean(fast_uint node, const SampleType* const query, CandidateSorter& candidates, SampleType box_distance, SampleType* box_distance_per_dimension) const;

private:
    std::vector<fast_uint> delay_vector_pattern;
    fast_uint delay_vector_alignment = 1;
    fast_uint max_delay_vector_offset;
    fast_uint delay_vector_count;
    std::vector<SampleType> sequence;
    std::vector<TreeNode> tree;
    fast_uint root = no_child;
    BoundingBox bounding_box;

#ifdef CREATE_DEBUG_STATS
    mutable fast_uint visited_nodes = 0;
    mutable fast_uint visited_leaves = 0;
    mutable fast_uint accepted_neighbour_candidates = 0;
#endif
};

} //namespace