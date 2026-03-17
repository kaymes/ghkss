import os
import subprocess
import sys
import tempfile
import numpy as np

import ghkss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo')))
from lorenz import simulate_lorenz, add_white_noise


test_data_file = None

def find_ghkss_cli():
    # try to find in the current directory if we compiled things locally
    candidate = os.path.join(os.path.dirname(__file__), 'ghkss')
    if os.path.exists(candidate):
        return candidate

    # Try to find in site-packages bin folder
    for path in sys.path:
        if 'site-packages' in path:
            # Check for bin folder in the site-packages directory
            candidate = os.path.join(path, 'bin', 'ghkss')
            if os.path.exists(candidate):
                return candidate
            # Also check Scripts folder on Windows
            candidate = os.path.join(path, 'Scripts', 'ghkss.exe')
            if os.path.exists(candidate):
                return candidate

    raise RuntimeError("Could not find the ghkss CLI.")

def create_test_data():
    global test_data_file
    if test_data_file is None:
        ground_truth = simulate_lorenz()
        time_series = add_white_noise(ground_truth, signal_to_noise_ratio=10)
        file_name = os.path.abspath(os.path.join(tempfile.mkdtemp(), 'lorenz.csv'))
        np.savetxt(file_name, time_series, delimiter=" ")
        test_data_file = file_name
    return test_data_file

def test_cli_simple():
    file_name = create_test_data()
    options = ""
    command = f"{find_ghkss_cli()} {options} {file_name}"
    result = subprocess.run(command.split(), shell=False, capture_output=True)
    if result.returncode != 0:
        print(result.stdout.decode())
        print(result.stderr.decode())
        result.check_returncode()


def compare_cli_and_python(cli_options, py_config, py_columns):
    file_name = create_test_data()
    input_data = np.loadtxt(file_name, delimiter=" ")
    command = f"{find_ghkss_cli()} {cli_options} {file_name}"
    result = subprocess.run(command.split(), shell=False, capture_output=True)
    if result.returncode != 0:
        print(result.stdout.decode())
        print(result.stderr.decode())
        result.check_returncode()
    out_file = f"{file_name}.opt.1"
    output_data_cli = np.loadtxt(out_file, delimiter=" ")
    if len(output_data_cli.shape) == 1:
        output_data_cli = output_data_cli.reshape(-1, 1)

    input_data_py = input_data[:,py_columns]
    output_data_py = ghkss.filter_ghkss(input_data_py, py_config)

    try:
        assert np.allclose(output_data_cli, output_data_py)
    except:
        print(f"Filename: {file_name}")
        print("======= CLI stderr =======")
        print(result.stderr.decode())
        print("==========================")
        print(f"input_data_py: shape={input_data_py.shape}, data={input_data_py[:20,:]}...")
        print(f"output_data_py: shape={output_data_py.shape}, data={output_data_py[:20,:]}...")
        print(f"output_data_cli: shape={output_data_cli.shape}, data={output_data_cli[:20,:]}...")
        raise

def test_cli_vs_python():
    py_config = ghkss.FilterConfig()
    py_config.batch_size = float('inf')
    py_config.iterations = 1
    py_config.delay_vector_pattern = [0,1,2,3,4]
    py_config.delay_vector_alignment = 1
    py_config.projection_dimension = 2
    py_config.minimum_neighbour_count = 50
    py_config.neighbour_epsilon = 0.1
    py_config.tisean_epsilon_widening = False
    py_config.maximum_neighbour_count = 2**30
    py_config.euclidean_norm = False
    py_config.verbosity = ghkss.verbosity_none

    compare_cli_and_python("-vvv -m 1,5 --radius 0.1", py_config, [0])

    py_config.set_delay_vector_pattern(signal_components=2, delay_vector_timesteps=4)
    compare_cli_and_python("-vvv -m 2,4 --radius 0.1", py_config, [0,1])

    py_config.set_delay_vector_pattern(signal_components=2, delay_vector_timesteps=4)
    compare_cli_and_python("-vvv -m 2,4 -c 1,3 --radius 0.1", py_config, [0,2])




def test_cli():
    test_cli_simple()
    test_cli_vs_python()


