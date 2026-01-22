# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Set the project name and version
project(GHKSS VERSION 1.0.0)

# set global compiler options
# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# We need pybind11 to create Python bindings.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# We need the Eigen library for linear algebra.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# We need CLI11 for the command line interface.
# It is shipped as a single header file with this code base.
add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CLI11")

# Build the Python module to drive GHKSS filter search from Python.
pybind11_add_module(ghkss_python MODULE src/ghkss_python.cpp src/ghkss.cpp src/knn_kdtree.cpp)
set_target_properties(ghkss_python PROPERTIES OUTPUT_NAME ghkss_cpp)
target_compile_definitions(ghkss_python PRIVATE WITH_PYTHON)
target_link_libraries (ghkss_python PRIVATE Eigen3::Eigen)
install(TARGETS ghkss_python LIBRARY DESTINATION ghkss)

# During development it may be handy to copy the compiled Python extension into the source tree.
# Uncomment the following line for cmake to do so.
add_custom_command(TARGET ghkss_python POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:ghkss_python>" "${CMAKE_CURRENT_SOURCE_DIR}/src/ghkss/$<TARGET_FILE_NAME:ghkss_python>")

# Compile refactored TISEAN code as C++
add_executable(ghkss_cli src/ghkss_cli.cpp src/ghkss.cpp src/knn_kdtree.cpp)
set_target_properties(ghkss_cli PROPERTIES OUTPUT_NAME ghkss)
target_link_libraries (ghkss_cli PRIVATE Eigen3::Eigen CLI11)
install(TARGETS ghkss_cli DESTINATION bin)
