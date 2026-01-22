#include <pybind11/pybind11.h>
#include "ghkss.h"

PYBIND11_MODULE(ghkss_cpp, module) {
    module.doc() = "C++ implementation for of a local projective filter (GHKSS).";
    ghkss::register_with_python(module);
}