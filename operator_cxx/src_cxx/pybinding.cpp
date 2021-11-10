#include <pybind11/pybind11.h>
#include "nms.h"
namespace py = pybind11;

PYBIND11_MODULE(processing_cxx, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("wnms_4c", &point4_wnms_4c<float>);
}
