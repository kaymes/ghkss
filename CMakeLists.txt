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
find_package(Eigen3 NO_MODULE)

if(Eigen3_FOUND)
    if(NOT (Eigen3_VERSION VERSION_GREATER_EQUAL "3.3" AND Eigen3_VERSION VERSION_LESS "6.0"))
        message(WARNING "Found Eigen3 version ${Eigen3_VERSION} which may or may not not be compatible with this package.")
    endif()
else()
    message(STATUS "Eigen3 not found. Downloading...")
    include(FetchContent)
    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
    # FetchContent might not create the Eigen3::Eigen alias exactly as find_package does
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
    endif()
endif()


# We need CLI11 for the command line interface.
# It is shipped as a single header file with this code base.
add_library(CLI11 INTERFACE)
target_include_directories(CLI11 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CLI11")

# Build the Python module to drive GHKSS filter search from Python.
pybind11_add_module(ghkss_python MODULE src/ghkss_python.cpp src/ghkss.cpp src/knn_kdtree.cpp)
set_target_properties(ghkss_python PROPERTIES OUTPUT_NAME ghkss_cpp)
target_compile_definitions(ghkss_python PRIVATE WITH_PYTHON $<$<CONFIG:Debug>:WITH_DEBUG_INFO>)
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
