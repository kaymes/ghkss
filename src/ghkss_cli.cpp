#include <vector>
#include <string>
#include <stdexcept>
#include <variant>
#include <fstream>
#include <sstream>
#include <iostream>
#include <optional>

#include <CLI11.hpp>
#include "ghkss.h"


struct Config {
    std::vector<std::string> datafiles;
    bool process_all_files = false;
    std::string output_file;
    std::optional<size_t> length;
    size_t skip_lines = 0;
    size_t iterations = 1;
    std::vector<size_t> columns; // starting with 1
    char line_delimiter = '\n';
    char column_delimiter = ',';
    bool calculate_default_neighbour_epsilon = true;
    ghkss::GhkssConfig ghkss;
};

class SystemExit: public std::exception {
public:
    SystemExit(char exit_code=0): exit_code_(exit_code) {}
    char exit_code() const { return exit_code_; }
private:
    char exit_code_ = 0;
};

Config get_config(int argc, char** argv) {
    Config config;

    CLI::App app{"GHKSS drop-in replacement"};

    // Positional datafiles (0..N). "-" means stdin; if none, read stdin.
    app.add_option("datafiles", config.datafiles, "Data files (\"-\" for stdin). For compatibility with TISEAN, by default the last valid file is being used. Use -a to filter all files.")->default_str("-");
    app.add_flag("-a,--all", config.process_all_files, "Filter all files (independently).");

    app.add_option("-l,--length", config.length, "# of data to use")->option_text("UINT [whole file]")->check(CLI::PositiveNumber);
    app.add_option("-x,--skip-lines", config.skip_lines, "# of lines to be ignored")->check(CLI::NonNegativeNumber)->capture_default_str();

    app.add_option("-c,--columns", config.columns, "column(s) to read")
            ->option_text("UINT[,UINT...] [1,..,# of components]")
            ->check(CLI::PositiveNumber)
            ->delimiter(',');

    size_t components = 1;
    size_t embedding_dimension = 5;

    app.add_option("-C,--components", components, "# of components")->check(CLI::PositiveNumber)->capture_default_str();
    app.add_option("-e,--embedding-dimension", embedding_dimension, "embedding dimension")->check(CLI::PositiveNumber)->capture_default_str();

    app.add_option_function<std::string>(
                    "-m",
                    [&components, &embedding_dimension](const std::string& value) {
                        std::vector<std::string> parts = CLI::detail::split(value, ',');
                        if (parts.size() != 2) {
                            throw CLI::ValidationError("Must be provided as two integers separated by a comma: components,embedding dimension");
                        }
                        try {
                            components = std::stoi(parts[0]);
                            embedding_dimension = std::stoi(parts[1]);
                        } catch(...) {
                            throw CLI::ValidationError("Invalid number");
                        }
                        if (components < 1) {
                            throw CLI::ValidationError("Number of components must be at least 1");
                        }
                        if (embedding_dimension < 1) {
                            throw CLI::ValidationError("Embedding dimension must be at least 1");
                        }
                    },
                    "# of components,embedding dimension. Same as using -C and -e, for compatibility with TISEAN.")->type_name("INT,INT")
            ->default_str(std::to_string(components) + "," + std::to_string(embedding_dimension))
            ->excludes("-C")
            ->excludes("-e");


    size_t delay_delta = 1;
    app.add_option("-d,--delay", delay_delta, "delay")->check(CLI::PositiveNumber)->capture_default_str();

    config.ghkss.projection_dimension = 2;
    app.add_option("-q,--project-dim", config.ghkss.projection_dimension, "dimension to project to")->check(CLI::PositiveNumber)->capture_default_str();

    config.ghkss.minimum_neighbour_count = 50;
    app.add_option("-k,--kmin", config.ghkss.minimum_neighbour_count, "minimal number of neighbours")->check(CLI::PositiveNumber)->capture_default_str();

    config.ghkss.neighbour_epsilon = std::numeric_limits<double>::signaling_NaN();
    config.calculate_default_neighbour_epsilon = true;
    app.add_option_function<double>("-r,--radius",
                                    [&config](const double value) {
                                        config.calculate_default_neighbour_epsilon = false;
                                        config.ghkss.neighbour_epsilon = value;
                                    },
                                    "minimal neighbourhood size")
            ->option_text("FLOAT [(interval of data)/1000]")
            ->check(CLI::NonNegativeNumber);

    app.add_option("-i,--iterations", config.iterations, "# of iterations")->check(CLI::PositiveNumber)->capture_default_str();

    app.add_flag("-2,--euclidean", config.ghkss.euclidean_norm, "use the Euclidean metric instead of the maximum norm")->capture_default_str();

    app.add_flag("-t,--tisean-epsilon", config.ghkss.tisean_epsilon_widening, "use TISEAN style epsilon widening")->capture_default_str();

    // -o name of output file [Default: 'datafile'.opt.n ...]
    app.add_option("-o,--output", config.output_file,
                   "name of output file [Default: 'datafile'.opt.n, where n is the iteration. "
                   "If no -o or -a is given, the last iteration is also written to stdout]")
                   ->excludes("-a");

    app.add_flag("-v,--verbose", config.ghkss.verbosity, "increase verbosity. Can be repeated multiple times to increase verbosity further.");



    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        throw SystemExit(app.exit(e));
    }

    if (components < 1) {
        throw CLI::ValidationError("Number of components must be at least 1");
    }
    if (components < config.columns.size()) {
        config.columns.resize(components);
    }
    size_t max_column = 0;
    for (auto column : config.columns) {
        max_column = std::max(max_column, column);
    }
    while (config.columns.size() < components) {
        max_column++;
        config.columns.push_back(max_column);
    }

    // Build the delay vector pattern and alignment
    config.ghkss.delay_vector_pattern.clear();
    for (int delay_index = 0; delay_index < embedding_dimension; delay_index++) {
        for (int component_index = 0; component_index < config.columns.size(); component_index++) {
            config.ghkss.delay_vector_pattern.push_back(delay_index * delay_delta * config.columns.size() + component_index);
        }
    }
    config.ghkss.delay_vector_alignment = components;

    return config;

}

std::vector<double> read_data(const std::string& filename, const Config& config) {
    std::vector<double> result;
    
    std::istream* input;
    std::ifstream file;

    if ( filename == "-" ) {
        input = &std::cin;
    } else {
        file.open(filename);
        if ( !file ) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        input = &file;
    }

    std::string line;
    long line_count = 0;

    // Skip requested lines
    while (line_count < config.skip_lines && std::getline(*input, line, config.line_delimiter)) {
        line_count++;
    }

    // Read data
    size_t max_column = *std::max_element(config.columns.begin(), config.columns.end());
    std::vector<double> row;
    while (std::getline(*input, line, config.line_delimiter)) {
        line_count++;
        row.clear();

        // Skip empty lines all together without causing a warning
        if (line.empty()) {
            continue;
        }

        if ( config.length && result.size() >= *config.length * config.columns.size()) {
            break;
        }

        std::stringstream ss(line);
        std::string value;

        // Parse CSV line
        while (std::getline(ss, value, config.column_delimiter)) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid number in data: " + value);
            }
        }

        // Check if we have enough columns
        if ( row.size() < max_column) {
            std::cerr << "Skipping line " << line_count << " because there are not enough columns in the data." << std::endl;
        } else {
            // Extract requested columns
            for (int col: config.columns) {
                result.push_back(row[col - 1]); // Convert 1-based to 0-based indexing
            }
        }
    }

    if ( filename != "-" ) {
        file.close();
    }

    return result;
}

void write_data(const std::string& filename, const std::vector<double>& data, const Config& config) {
    std::ofstream output_file;
    bool use_stdout = filename == "-";
    if (!use_stdout) {
        output_file.open(filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
    }
    std::ostream& out = use_stdout ? std::cout : output_file;
    for (size_t index = 0; index < data.size(); index++) {
        out << data[index];
        if ((index+1) % config.columns.size() == 0) {
            out << config.line_delimiter;
        } else {
            out << config.column_delimiter;
        }
    }
}


int main(int argc, char** argv) {

    try {

        Config config = get_config(argc, argv);

        if ( !config.process_all_files ) {
            if ( config.ghkss.verbosity >= ghkss::verbosity_trace) {
                std::cerr << "Selecting the data file to use." << std::endl;
            }
            while (!config.datafiles.empty()) {
                if ( config.datafiles.back() == "-" || std::ifstream(config.datafiles.back()).good()) {
                    config.datafiles[0] = config.datafiles.back();
                    config.datafiles.resize(1);
                    if ( config.ghkss.verbosity >= ghkss::verbosity_debug) {
                        std::cerr << "Selected data file " << config.datafiles[0] << "." << std::endl;
                    }
                    break;
                } else {
                    std::cerr << "File " << config.datafiles.back() << " does not exist. Trying next file." << std::endl;
                    config.datafiles.pop_back();
                }
            }
        }

        for (const auto& filename: config.datafiles) {
            if ( config.ghkss.verbosity >= ghkss::verbosity_info) {
                std::cerr << "Processing file " << filename << "." << std::endl;
            }
            std::vector<double> data = read_data(filename, config);

            if ( config.calculate_default_neighbour_epsilon ) {
                const auto [min_element, max_element] = std::minmax_element(data.begin(), data.end());
                config.ghkss.neighbour_epsilon = (*max_element - *min_element) / 1000;
            }

            for (size_t iteration = 0; iteration < config.iterations; iteration++) {
                if ( config.ghkss.verbosity >= ghkss::verbosity_info) {
                    std::cerr << "Running filter iteration " << iteration+1 << "." << std::endl;
                }
                data = ghkss::filter_ghkss(data, config.ghkss);

                std::string output_file;
                if ( config.output_file.empty()) {
                    if ( filename == "-" ) {
                        output_file = "stdin.opt";
                    } else {
                        output_file = filename + ".opt";
                    }
                } else {
                    output_file = config.output_file;
                }
                output_file += "." + std::to_string(iteration + 1);
                if ( config.ghkss.verbosity >= ghkss::verbosity_info) {
                    std::cerr << "Writing result of iteration " << iteration+1 << " to file " << output_file << "." << std::endl;
                }
                write_data(output_file, data, config);
            }

            if ( config.output_file.empty() && filename == "-" ) {
                write_data("-", data, config);
            }
        }

    } catch (const SystemExit& e) {
        return e.exit_code();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}