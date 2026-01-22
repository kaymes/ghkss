import math
import numpy as np

try:
    from . import ghkss_cpp
    from .ghkss_cpp import verbosity_none, verbosity_info, verbosity_debug, verbosity_trace
except ImportError:
    print("Could not import C++ extension. Please make sure that the ghkss_cpp module is compiled.")
    raise


class FilterConfig(ghkss_cpp.GhkssConfig):
    _ghkss_config_field_names = [f for f in ghkss_cpp.GhkssConfig().__dir__() if not f.startswith('_')]
    _config_fields = dict(
        batch_size=float('inf'),
        iterations=10,
    )
    __slots__ = tuple(_config_fields.keys())

    def __init__(self, **kwargs):
        super().__init__()
        for key, default in self._config_fields.items():
            setattr(self, key, default)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _as_base(self):
        result = ghkss_cpp.GhkssConfig()
        for key in self._ghkss_config_field_names:
            setattr(result, key, getattr(self, key))
        return result

    def as_dict(self):
        return {**{k:getattr(self, k) for k in self._config_fields}, **{k:getattr(self, k) for k in self._ghkss_config_field_names}}

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.__init__(**state)

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def replace(self, **kwargs):
        return self.__class__(**{**self.as_dict(), **kwargs})

    def set_delay_vector_pattern(self, delay_vector_timesteps=5, delay_vector_delta=1, signal_components=1):
        delay_vector_pattern = []
        for time_step in range(delay_vector_timesteps):
            for component in range(signal_components):
                delay_vector_pattern.append(time_step * signal_components * delay_vector_delta + component)
        self.delay_vector_pattern = delay_vector_pattern
        self.delay_vector_alignment = signal_components


def filter_ghkss(time_series, filter_config, return_neighbour_statistics=False):

    time_series = np.array(time_series)
    if len(time_series.shape) == 1:
        time_series = time_series.reshape(-1, 1)
    elif len(time_series.shape) != 2:
        raise ValueError("Only 1D or 2D time series are supported.")

    assert len(time_series) > 1, "There is only one time step in the time series. Maybe you need to transpose the input?"


    ghkss_config = filter_config._as_base()
    neighbour_statistics = []
    for iteration in range(filter_config.iterations):
        if filter_config.verbosity >= verbosity_info:
            print(f"Starting filter iteration {iteration+1} of {filter_config.iterations}.")
        batch_start = 0
        while batch_start < len(time_series):
            batch_end = min(batch_start + filter_config.batch_size, len(time_series))
            if filter_config.verbosity >= verbosity_info and filter_config.batch_size < float('inf'):
                print(f"Processing batch {batch_start+1} of {math.ceil(len(time_series)/filter_config.batch_size)}.")
            batch = time_series[batch_start:batch_end,:]
            filtered_batch = ghkss_cpp.filter_ghkss(batch.flatten(order='C'), ghkss_config, return_neighbour_statistics)
            if return_neighbour_statistics:
                filtered_batch, statistics = filtered_batch
                neighbour_statistics.append(statistics)
                if filter_config.verbosity >= verbosity_info:
                    print(f"Average neighbour count: {statistics.average_neighbour_count:.1f} (minimum: {statistics.minimum_neighbour_count}, maximum: {statistics.maximum_neighbour_count}).")
            filtered_batch = np.array(filtered_batch).reshape(batch.shape, order='C')
            time_series[batch_start:batch_end,:] = filtered_batch
            batch_start = batch_end

    if filter_config.verbosity >= verbosity_info:
        print(f"Filtering complete.")

    if return_neighbour_statistics:
        return time_series, neighbour_statistics
    else:
        return time_series

