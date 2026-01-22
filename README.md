# Local geometric projection filter (GHKSS)

This package contains an implementation of the local geometric projection filter originally described in [[1]](#References). A textbook description of a variant of the filter can be found in [[2]](#References) (chapter 10.3), but details differ. A C implementation of the filter was previously published as part of the [TISEAN](https://www.pks.mpg.de/tisean/Tisean_3.0.1/) package. This package contains a re-implementation from scratch in C++ as well as Python bindings. A command line interface is provided that is designed to act as a drop-in replacement for the `ghkss` binary from the TISEAN package. 

The algorithm is mostly kept equivalent to the original implemntation without striving for numerically identical behaviour in all cases. A major change is that we replaced the box counting method for the k-nearest neighbor search with a kd-tree for a better run time performance which leads to different neighbour sets being used (see the description of options below for more details).

## Installation

The package can be installed from PyPi using pip:

`pip install ghkss`

### Building from source

A prerequisite for compilation and installing from source is the [Eigen](http://eigen.tuxfamily.org/) library. It is recommended to install it using your system package manager before building the package (e.g., `apt-get install eigen3-dev`). Hoever, if it is missing, the build script attempts to download it directly,

For the Python interface, the [pybind11](https://github.com/pybind/pybind11) library is required. It can be installed using pip: `pip install pybind11`.

The command line iterface also uses the [CLI11](https://github.com/CLIUtils/CLI11) library which is shipped together with this package in the `third_party` folder and doesn't have to be installed separately.

The package can be built by issuing the following command in the root directory of the project:

`pip install .`

The command line interface will be instaled in the `bin` directory.

To build a redistributable wheel, run the following command:

`python -m build`  

If the `build` command is missing, install it using `pip install build`. 

The command line interface is built as part of the python built process. To build it separately, run the following commands:

`mkdir build`  
`cd build`  
`cmake ..`  
`make ghkss_cli`

A debug build of can be created using `python -m build --wheel --config-setting="cmake.build-type=Debug"`. Such a build exposes internal intermediate results of the C++ component to Python.

## The Python interface

### Loading

The module is loaded using `import ghkss`.

### Functions

**filter_ghkss**(time_series, filter_config,return_neighbour_statistics=False)

The main filter function which applies the GHKSS filter to a time series. 

* *time_series*: a numpy array of shape `(n_samples, n_components)` containing the time series.  
* *filter_config*: a parameter object of type `FilterConfig`  
* *return_neighbour_statistics*: a boolean flag indicating whether to return statistics about the neighbour search of the filter.


*Return value:* a numpy array of shape `(n_samples, n_components)` containing the filtered time series. If `return_neighbour_statistics` is `True`, a tuple `(filtered_time_series, neighbour_statistics)` is returned instead with the filtered_time_series a numpy array as before and `neighbour_statistics` is a list with elements of type `NeighbourStatistics`, one for each application of the filter (iteration or batch).

### GhkssConfig

Configuration structure for the GHKSS local projection filter.  
It controls how delay vectors are constructed, how neighbours are selected, and how distances are measured.

#### Member variables
**delay_vector_pattern** (*list*, default: `[0,1,2,3,4]`):
  Relative offsets (indices) in the time series that form a single delay vector.
  For a time index `i`, the corresponding delay vector uses samples at indices
  `i + delay_vector_pattern[0]`, `i + delay_vector_pattern[1]`, …
  The default pattern `{0,1,2,3,4}` corresponds to five consecutive samples for a single-variable time series.
  
  If multiple signal components are processed, they are internally flattened such that components $x_0, x_1, \ldots, y_0, y_1, \ldots, z_0, z_1, \ldots$ are represented as $x_0, y_0, z_0, x_1, y_1, z_1, \ldots$.

  The order of the indices in the pattern is important due to a subtlety of the filter algorithm: when averaging correction factors, weights are applied which give less weight to the first and last elements of the delay vector. For single component time series (`delay_vector_alignment` = 1), the first and last elements of the `delay_vector_pattern` are treated with reduced weight. For multivariate time series (`delay_vector_alignment` > 1), the first and last elements belonging to each component are treated with reduced weight.

  Note: the convenience method `set_delay_vector_pattern(…)` is provided to construct delay vector patterns for multivariate time series.

**delay_vector_alignment** (*int*, default: `1`):
  Alignment constraint for the starting indices of delay vectors.
  If set to a value greater than `1`, delay vectors are only constructed at indices
  that are multiples of this alignment (e.g., `i = 0, alignment, 2*alignment, …`).
  
  If multiple components are being filtered at once, this should be set to the 
  number of components to ensure delay vectors always start at valid index of
  the flattened representation.

**projection_dimension** (*int*, default: `2`):
  Dimensionality of the manifold onto which the local neighbourhood of each delay vector is projected. Typically this corresponds to the intrinsic dimension assumed for the underlying dynamics. In doubt, choose a slightly larger value.

**minimum_neighbour_count** (*int*, default: `50`):
  Minimum number of neighbours that should be found for each delay vector. If the epsilon neighbourhood (see `neighbour_epsilon`) contains less than the specified number of neighbours, the `minimum_neighbour_count` closest neighbours are used regardless of their distance. (The behaviour changes slightly if `tisean_epsilon_widening` is set to `True`).

**neighbour_epsilon** (*float*, default: `-1`):
  If set to a positive value, the nearest neighbour search will return all delay vectors within this distance (epsilon-ball) around the query delay vector. If set to a negative value, a fixed neighbour count is used and the `minimum_neighbour_count` closest neighbours are used regardless of their distance.

**tisean_epsilon_widening** (*bool*, default: `false`):
  This option mimics the behaviour of the TISEAN implementation of the GHKSS filter. If set to `true`, the neighbour search procedure is as follows:
  * Start with `neighbour_epsilon` as the radius.
  * Collect all neighbours within that radius.
  * If their count is less than `minimum_neighbour_count`, increase the radius by a
      factor of √2 and repeat.
  * Repeat until at least `minimum_neighbour_count` neighbours are found.
  * All neighbours found in the final round are returned.
 
If `tisean_epsilon_widening` is `true`, `maximum_neighbour_count` is ignored.

Note: this method may perform multiple nearest neighbour searches for each point, leading to a significant performance penalty.

**maximum_neighbour_count** (*int*, default: *a very large number*):
  Only used when `neighbour_epsilon` is non-negative and `tisean_epsilon_widening` is false. Caps the number of neighbours that will be considered, even if more delay vectors lie within the epsilon radius. This option is predominantly intended as safeguard to limit mempory usage and computation time. It is not guaranteed that the selected neighbours are the closest ones. Instead, the first `maximum_neighbour_count` neighbours that are within the `neighbour_epsilon` radius are used.

**euclidean_norm** (*bool*, default: `false`):
  Controls how distances between delay vectors are computed:
  * If `true`: use the Euclidean norm ($\ell^2$).
  * If `false`: use the maximum norm ($\ell^\infty$, i.e. the maximum absolute component-wise difference).

**verbosity** (*int*, default: `verbosity_none`):

  Controls how much diagnostic information is printed to the console. `0` (i.e. `verbosity_none`) means no output., higher values enable more detailed logging.
  The following constants are defined in the `ghkss` module:
  * `verbosity_none` = 0
  * `verbosity_info` = 1
  * `verbosity_high` = 2
  * `verbosity_debug` = 3
  * `verbosity_trace` = 4

**batch_size** (*float*, default: `inf`):
  If set to a finite value, the input sequence will be split into batches of this size and processed independently. This may be useful for datasets with drifting system dynamics. 

  If multiple components are present, the batch size refers to the number of timesteps (rows of the two-dimensional time series array).

  *Note: this option is part of the Python interface and not available in the C++ API.*

**iterations** (*int*, default: `10`):
  Number of iterations of the GHKSS filter to perform.

*Note: this option is part of the Python interface and not available in the C++ API.*

#### Methods

**set_delay_vector_pattern**(delay_vector_timesteps=5, delay_vector_delta=1, signal_components=1)

A convenience method to construct delay vector patterns for multivariate time series. It sets the correct values for the `delay_vector_pattern` and `delay_vector_alignment` parameters.

* *delay_vector_timesteps* (*int*, default: `5`):
Number of time steps to include in each delay vector.
* *delay_vector_delta* (*int*, default: `1`):
Time step increment between consecutive delay vector components.
* *signal_components* (*int*, default: `1`):
Number of components in the input signal.

**replace**(**kwargs)

Returns a copy of the filter configuration with the specified parameters replaced.

**as_dict**()

Return a dictionary representation of the filter configuration.


### NeighbourStatistics
A simple struct holding statistics about the neighbour search of the filter.

#### Member variables

**minimum_neighbour_count**: the smallest number of neighbours found for any delay vector during the filter iteration.

**maximum_neighbour_count**: the largest number of neighbours found for any delay vector during the filter iteration.

**average_neighbour_count**: the average number of neighbours found for the delay vectors during the filter iteration.

## The command line interface

The command line interface has been designed to be a drop-in replacement for the `ghkss` binary from the TISEAN package. It accepts the same parameters as the original binary. The usage information is as follows:

```
./ghkss [OPTIONS] [datafiles...]


POSITIONALS:
  datafiles TEXT [-]  ...     Data files ("-" for stdin). For compatibility with TISEAN, by 
                              default the last valid file is being used. Use -a to filter all 
                              files. 

OPTIONS:
  -h,     --help              Print this help message and exit 
  -a,     --all Excludes: --output 
                              Filter all files (independently). 
  -l,     --length UINT [whole file] 
                              # of data to use 
  -x,     --skip-lines UINT:NONNEGATIVE [0]  
                              # of lines to be ignored 
  -c,     --columns UINT[,UINT...] [1,..,# of components] 
                              column(s) to read 
  -C,     --components UINT:POSITIVE [1]  Excludes: -m 
                              # of components 
  -e,     --embedding-dimension UINT:POSITIVE [5]  Excludes: -m 
                              embedding dimension 
  -m INT,INT [1,5]  Excludes: --components --embedding-dimension 
                              # of components,embedding dimension. Same as using -C and -e, for 
                              compatibility with TISEAN. 
  -d,     --delay UINT:POSITIVE [1]  
                              delay 
  -q,     --project-dim UINT:POSITIVE [2]  
                              dimension to project to 
  -k,     --kmin UINT:POSITIVE [50]  
                              minimal number of neighbours 
  -r,     --radius FLOAT [(interval of data)/1000] 
                              minimal neighbourhood size 
  -i,     --iterations UINT:POSITIVE [1]  
                              # of iterations 
  -2,     --euclidean         use the Euclidean metric instead of the maximum norm 
  -t,     --tisean-epsilon    use TISEAN style epsilon widening 
  -o,     --output TEXT Excludes: --all 
                              name of output file [Default: 'datafile'.opt.n, where n is the 
                              iteration. If no -o or -a is given, the last iteration is also 
                              written to stdout] 
  -v,     --verbose [0]       increase verbosity. Can be repeated multiple times to increase 
                              verbosity further. 

```

## Licensing
This package is licenced under the MIT license which can be found in the file [LICENSE.txt](LICENSE.txt).

The CLI11 library contained in the third party folder is licensed under a BSD style license which can be found in the file [third_party/CLI11/LICENSE.txt](third_party/CLI11/LICENSE.txt).

The Eigen library is licensed under the [MPL 2.0 license](third_party/Eigen/LICENSE.txt).



## References

[1] P. Grassberger, R. Hegger, H. Kantz, C. Schaffrath, and T. Schreiber, “On noise reduction methods for chaotic data,” Chaos: An Interdisciplinary Journal of Nonlinear Science, vol. 3, no. 2, pp. 127–141, Apr. 1993, doi: 10.1063/1.165979.

[2] H. Kantz and T. Schreiber, "Nonlinear Time Series Analysis", 2nd ed. Cambridge: Cambridge University Press, 2003. doi: 10.1017/CBO9780511755798.
