/**
 * CASAS – pybind11 Bindings (FULL v2)
 * bindings.cpp
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "casas_da.hpp"

namespace py = pybind11;
using namespace casas;

// Helper functions
static py::list stm_to_pylist(const std::array<std::array<double,6>,6>& stm) {
    py::list result;
    for (int i = 0; i < 6; ++i) {
        py::list row;
        for (int j = 0; j < 6; ++j) row.append(stm[i][j]);
        result.append(row);
    }
    return result;
}

static std::array<std::array<double,6>,6> pylist_to_stm(py::list py_stm) {
    std::array<std::array<double,6>,6> stm{};
    for (int i = 0; i < 6; ++i) {
        py::list row = py_stm[i].cast<py::list>();
        for (int j = 0; j < 6; ++j) {
            stm[i][j] = row[j].cast<double>();
        }
    }
    return stm;
}

// ========================================================================
PYBIND11_MODULE(casas_cpp, m) {
    m.doc() = "CASAS C++ DA Propagator v2";

    // PropConfig
    py::class_<PropConfig>(m, "PropConfig")
        .def(py::init<>())
        .def_readwrite("dt_step_s", &PropConfig::dt_step_s)
        .def_readwrite("use_j2",    &PropConfig::use_j2)
        .def_readwrite("use_drag",  &PropConfig::use_drag)
        .def_readwrite("bstar",     &PropConfig::bstar)
        .def_readwrite("f107",      &PropConfig::f107)
        .def_readwrite("ap",        &PropConfig::ap)
        .def_readwrite("use_rk78",  &PropConfig::use_rk78)
        .def_readwrite("adaptive",  &PropConfig::adaptive)
        .def_readwrite("abs_tol",   &PropConfig::abs_tol)
        .def_readwrite("rel_tol",   &PropConfig::rel_tol)
        .def_readwrite("use_srp",   &PropConfig::use_srp)
        .def_readwrite("cr_am",     &PropConfig::cr_am)
        .def_readwrite("order2",    &PropConfig::order2);

    // PropResult
    py::class_<PropResult>(m, "PropResult")
        .def_readonly("state", &PropResult::state)
        .def_readonly("stm", &PropResult::stm);

// PropResult2 (for order-2 propagation)
py::class_<PropResult2>(m, "PropResult2")
    .def_readonly("state", &PropResult2::state)
    .def_readonly("stm", &PropResult2::stm)
    .def_property_readonly("stt", [](const PropResult2& r) {
    py::list outer;
    for (int i = 0; i < 6; ++i) {
        py::list mat;
        for (int j = 0; j < 6; ++j) {
            py::list row;
            for (int k = 0; k < 6; ++k) row.append(r.stt[i][j][k]);
            mat.append(row);
        }
        outer.append(mat);
    }
    return outer;
        });

// PropResult2Full (for full order-2 with mean correction)
py::class_<PropResult2Full>(m, "PropResult2Full")
    .def_readonly("state", &PropResult2Full::state)
    .def_readonly("mean", &PropResult2Full::mean)
    .def_readonly("stm", &PropResult2Full::stm)
    .def_property_readonly("stt", [](const PropResult2Full& r) {
    py::list outer;
    for (int i = 0; i < 6; ++i) {
        py::list mat;
        for (int j = 0; j < 6; ++j) {
            py::list row;
            for (int k = 0; k < 6; ++k) row.append(r.stt[i][j][k]);
            mat.append(row);
        }
        outer.append(mat);
    }
    return outer;
        });


    // EKFResult
    using EKF3 = EKFResult<3>;
    py::class_<EKF3>(m, "EKFResult")
        .def_readonly("xp", &EKF3::xp)
        .def_property_readonly("Pp", [](const EKF3& r){ return stm_to_pylist(r.Pp); })
        .def_readonly("innovation", &EKF3::innovation)
        .def_readonly("nis", &EKF3::nis);

    // ====================== FUNCTIONS ======================

    m.def("da_propagate", &da_propagate,
          py::arg("x0_km"), py::arg("dt_total_s"), py::arg("cfg") = PropConfig{});

    m.def("da_propagate_rk78", &da_propagate_rk78,
          py::arg("x0_km"), py::arg("dt_total_s"), py::arg("cfg") = PropConfig{});

    m.def("da_propagate_order2", &da_propagate_order2,
          py::arg("x0_km"), py::arg("dt_total_s"), py::arg("cfg") = PropConfig{});
    m.def("da_propagate_order2_full", &da_propagate_order2_full,
        py::arg("x0_km"), py::arg("dt_total_s"), py::arg("P0"), py::arg("cfg") = PropConfig{});
    m.def(
        "da_propagate_order2_full",
        &da_propagate_order2_full,
        py::arg("x0_km"),
        py::arg("dt_total_s"),
        py::arg("P0"),
        py::arg("cfg") = PropConfig(),
        "Order-2 DA propagation with second-order mean correction"
    );

    m.def("propagate_covariance",
        [](py::list P0_py, py::list stm_py) -> py::list {
            auto P0 = pylist_to_stm(P0_py);
            auto stm = pylist_to_stm(stm_py);
            return stm_to_pylist(propagate_covariance(P0, stm));
        });

    m.def("ekf_update",
        [](py::list z_py, py::list H_py, py::list R_py,
           const std::array<double,6>& xm, py::list Pm_py) -> EKF3 {
            std::array<double,3> z{};
            for (int i=0; i<3; i++) z[i] = z_py[i].cast<double>();

            double H[3][6] = {};
            for (int i=0; i<3; i++) {
                auto row = H_py[i].cast<py::list>();
                for (int j=0; j<6; j++) H[i][j] = row[j].cast<double>();
            }

            double R[3][3] = {};
            for (int i=0; i<3; i++) {
                auto row = R_py[i].cast<py::list>();
                for (int j=0; j<3; j++) R[i][j] = row[j].cast<double>();
            }

            auto Pm = pylist_to_stm(Pm_py);
            return ekf_update<3>(z, H, R, xm, Pm);
        });

    m.attr("__version__") = "2.0.0";
    m.attr("__backend__") = "C++ DA Propagator v2";
}