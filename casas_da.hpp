/**
 * CASAS – Differential Algebra Propagator (RK78-Focused)
 * ========================================================
 * casas_da.hpp v2 (Simplified)
 *
 * Architecture:
 *   • DA Algebra (Order-1 & Order-2 variants) – core math
 *   • RK7(8) Dormand-Prince – primary integrator (high-accuracy)
 *   • Adaptive step control – for tight tolerances
 *   • Order-2 propagation – non-linear uncertainty quantification
 *   • EKF measurement update – measurement fusion
 *
 * Removed:
 *   • RK4 fixed-step (legacy, lower accuracy)
 *   • Excessive comment bloat
 *   • Redundant helper functions
 *
 * Usage:
 *   #include "casas_da.hpp"
 *   casas::PropConfig cfg;
 *   cfg.use_rk78 = true;       // Now default
 *   cfg.adaptive = true;        // Enable PI step control
 *   auto res = casas::da_propagate_rk78(x0, dt, cfg);
 *
 * References:
 *   Berz (1987)  – Differential Algebra
 *   Dormand & Prince (1980)  – RK7(8) Coefficients
 *   Valli et al. (2013)  – DA-based Pc computation
 *   Simon (2006)  – EKF Joseph Form Stabilization
 */

#pragma once

#ifndef CASAS_DA_HPP
#define CASAS_DA_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace casas {

    // ========================================================================
    // SECTION 1: DA ALGEBRA (Order-1 & Order-2)
    // ========================================================================

    namespace da {

        // ────────────────────────────────────────────────────────────────
        // Order-1 DAVar<N> – First derivatives (Jacobian)
        // ────────────────────────────────────────────────────────────────

        template <int N>
        struct DAVar {
            double val;
            std::array<double, N> grad;

            DAVar() : val(0.0), grad{} {}
            explicit DAVar(double v) : val(v), grad{} {}

            static DAVar identity(int idx, double center = 0.0) {
                DAVar x;
                x.val = center;
                x.grad[idx] = 1.0;
                return x;
            }

            DAVar operator+(const DAVar& o) const {
                DAVar r;
                r.val = val + o.val;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] + o.grad[i];
                return r;
            }

            DAVar operator-(const DAVar& o) const {
                DAVar r;
                r.val = val - o.val;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] - o.grad[i];
                return r;
            }

            DAVar operator*(const DAVar& o) const {
                DAVar r;
                r.val = val * o.val;
                for (int i = 0; i < N; ++i)
                    r.grad[i] = val * o.grad[i] + grad[i] * o.val;
                return r;
            }

            DAVar operator/(const DAVar& o) const {
                if (std::abs(o.val) < 1e-30)
                    throw std::runtime_error("DAVar division by near-zero");
                DAVar r;
                r.val = val / o.val;
                for (int i = 0; i < N; ++i)
                    r.grad[i] = (grad[i] - r.val * o.grad[i]) / o.val;
                return r;
            }

            DAVar operator+(double s) const {
                DAVar r = *this;
                r.val += s;
                return r;
            }

            DAVar operator-(double s) const {
                DAVar r = *this;
                r.val -= s;
                return r;
            }

            DAVar operator*(double s) const {
                DAVar r;
                r.val = val * s;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] * s;
                return r;
            }

            DAVar operator/(double s) const { return *this * (1.0 / s); }

            friend DAVar operator+(double s, const DAVar& x) { return x + s; }
            friend DAVar operator*(double s, const DAVar& x) { return x * s; }

            DAVar& operator+=(const DAVar& o) {
                *this = *this + o;
                return *this;
            }

            DAVar& operator-=(const DAVar& o) {
                *this = *this - o;
                return *this;
            }

            DAVar& operator*=(double s) {
                *this = *this * s;
                return *this;
            }
        };

        // Elementary functions for DAVar
        template <int N>
        DAVar<N> da_sqrt(const DAVar<N>& x) {
            double sq = std::sqrt(x.val);
            DAVar<N> r;
            r.val = sq;
            double c = 0.5 / (sq + 1e-300);
            for (int i = 0; i < N; ++i) r.grad[i] = c * x.grad[i];
            return r;
        }

        template <int N>
        DAVar<N> da_sin(const DAVar<N>& x) {
            DAVar<N> r;
            r.val = std::sin(x.val);
            double c = std::cos(x.val);
            for (int i = 0; i < N; ++i) r.grad[i] = c * x.grad[i];
            return r;
        }

        template <int N>
        DAVar<N> da_cos(const DAVar<N>& x) {
            DAVar<N> r;
            r.val = std::cos(x.val);
            double c = -std::sin(x.val);
            for (int i = 0; i < N; ++i) r.grad[i] = c * x.grad[i];
            return r;
        }

        template <int N>
        DAVar<N> da_exp(const DAVar<N>& x) {
            double e = std::exp(x.val);
            DAVar<N> r;
            r.val = e;
            for (int i = 0; i < N; ++i) r.grad[i] = e * x.grad[i];
            return r;
        }

        template <int N>
        DAVar<N> da_pow(const DAVar<N>& x, double p) {
            double base = std::pow(x.val, p);
            DAVar<N> r;
            r.val = base;
            double c = p * std::pow(x.val, p - 1.0);
            for (int i = 0; i < N; ++i) r.grad[i] = c * x.grad[i];
            return r;
        }

        using DA6 = DAVar<6>;

        // ────────────────────────────────────────────────────────────────
        // Order-2 DA2Var<N> – First + Second derivatives (Hessian)
        // ────────────────────────────────────────────────────────────────

        template <int N>
        struct DA2Var {
            static constexpr int NHESS = N * (N + 1) / 2;

            double val;
            std::array<double, N> grad;
            std::array<double, NHESS> hess;

            DA2Var() : val(0.0), grad{}, hess{} {}
            explicit DA2Var(double v) : val(v), grad{}, hess{} {}

            static constexpr int hidx(int i, int j) {
                return (i >= j) ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
            }

            static DA2Var identity(int idx, double center = 0.0) {
                DA2Var x;
                x.val = center;
                x.grad[idx] = 1.0;
                return x;
            }

            DA2Var operator+(const DA2Var& o) const {
                DA2Var r;
                r.val = val + o.val;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] + o.grad[i];
                for (int k = 0; k < NHESS; ++k) r.hess[k] = hess[k] + o.hess[k];
                return r;
            }

            DA2Var operator-(const DA2Var& o) const {
                DA2Var r;
                r.val = val - o.val;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] - o.grad[i];
                for (int k = 0; k < NHESS; ++k) r.hess[k] = hess[k] - o.hess[k];
                return r;
            }

            DA2Var operator*(const DA2Var& o) const {
                DA2Var r;
                r.val = val * o.val;
                for (int i = 0; i < N; ++i)
                    r.grad[i] = val * o.grad[i] + grad[i] * o.val;
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j <= i; ++j) {
                        int k = hidx(i, j);
                        r.hess[k] = val * o.hess[k] + o.val * hess[k] +
                                    grad[i] * o.grad[j] + grad[j] * o.grad[i];
                    }
                return r;
            }

            DA2Var operator/(const DA2Var& o) const {
                if (std::abs(o.val) < 1e-30)
                    throw std::runtime_error("DA2Var division by near-zero");
                DA2Var r;
                double inv = 1.0 / o.val;
                double inv2 = inv * inv;
                double inv3 = inv2 * inv;
                r.val = val * inv;
                for (int i = 0; i < N; ++i)
                    r.grad[i] = (grad[i] - r.val * o.grad[i]) * inv;
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j <= i; ++j) {
                        int k = hidx(i, j);
                        r.hess[k] = inv * (hess[k] - r.val * o.hess[k]) -
                                    inv2 * (grad[i] * o.grad[j] + grad[j] * o.grad[i]) +
                                    2.0 * inv3 * r.val * o.grad[i] * o.grad[j];
                    }
                return r;
            }

            DA2Var operator+(double s) const {
                DA2Var r = *this;
                r.val += s;
                return r;
            }

            DA2Var operator-(double s) const {
                DA2Var r = *this;
                r.val -= s;
                return r;
            }

            DA2Var operator*(double s) const {
                DA2Var r;
                r.val = val * s;
                for (int i = 0; i < N; ++i) r.grad[i] = grad[i] * s;
                for (int k = 0; k < NHESS; ++k) r.hess[k] = hess[k] * s;
                return r;
            }

            DA2Var operator/(double s) const { return *this * (1.0 / s); }

            friend DA2Var operator+(double s, const DA2Var& x) { return x + s; }
            friend DA2Var operator*(double s, const DA2Var& x) { return x * s; }

            DA2Var& operator+=(const DA2Var& o) {
                *this = *this + o;
                return *this;
            }

            DA2Var& operator-=(const DA2Var& o) {
                *this = *this - o;
                return *this;
            }

            DA2Var& operator*=(double s) {
                *this = *this * s;
                return *this;
            }
        };

        // Elementary functions for DA2Var
        template <int N>
        DA2Var<N> da2_sqrt(const DA2Var<N>& x) {
            double sq = std::sqrt(x.val);
            double c1 = 0.5 / (sq + 1e-300);
            double c2 = -0.25 / (x.val * sq + 1e-300);
            DA2Var<N> r;
            r.val = sq;
            for (int i = 0; i < N; ++i) r.grad[i] = c1 * x.grad[i];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j) {
                    int k = DA2Var<N>::hidx(i, j);
                    r.hess[k] = c1 * x.hess[k] + c2 * x.grad[i] * x.grad[j];
                }
            return r;
        }

        template <int N>
        DA2Var<N> da2_sin(const DA2Var<N>& x) {
            double s = std::sin(x.val), c = std::cos(x.val);
            DA2Var<N> r;
            r.val = s;
            for (int i = 0; i < N; ++i) r.grad[i] = c * x.grad[i];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j) {
                    int k = DA2Var<N>::hidx(i, j);
                    r.hess[k] = c * x.hess[k] - s * x.grad[i] * x.grad[j];
                }
            return r;
        }

        template <int N>
        DA2Var<N> da2_cos(const DA2Var<N>& x) {
            double s = std::sin(x.val), c = std::cos(x.val);
            DA2Var<N> r;
            r.val = c;
            for (int i = 0; i < N; ++i) r.grad[i] = -s * x.grad[i];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j) {
                    int k = DA2Var<N>::hidx(i, j);
                    r.hess[k] = -s * x.hess[k] - c * x.grad[i] * x.grad[j];
                }
            return r;
        }

        template <int N>
        DA2Var<N> da2_exp(const DA2Var<N>& x) {
            double e = std::exp(x.val);
            DA2Var<N> r;
            r.val = e;
            for (int i = 0; i < N; ++i) r.grad[i] = e * x.grad[i];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j) {
                    int k = DA2Var<N>::hidx(i, j);
                    r.hess[k] = e * (x.hess[k] + x.grad[i] * x.grad[j]);
                }
            return r;
        }

        template <int N>
        DA2Var<N> da2_pow(const DA2Var<N>& x, double p) {
            double b = std::pow(x.val, p);
            double c1 = p * std::pow(x.val, p - 1.0);
            double c2 = p * (p - 1.0) * std::pow(x.val, p - 2.0);
            DA2Var<N> r;
            r.val = b;
            for (int i = 0; i < N; ++i) r.grad[i] = c1 * x.grad[i];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j) {
                    int k = DA2Var<N>::hidx(i, j);
                    r.hess[k] = c1 * x.hess[k] + c2 * x.grad[i] * x.grad[j];
                }
            return r;
        }

        using DA2_6 = DA2Var<6>;

        // Extract STM from order-2 result
        template <int N>
        std::array<std::array<double, N>, N> stm_from_da2(
            const std::array<DA2Var<N>, N>& F) {
            std::array<std::array<double, N>, N> stm{};
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    stm[i][j] = F[i].grad[j];
            return stm;
        }

        // Extract STT (State Transition Tensor) from order-2 result
        template <int N>
        std::vector<std::vector<std::vector<double>>> stt_from_da2(
            const std::array<DA2Var<N>, N>& F) {
            using T3 = std::vector<std::vector<std::vector<double>>>;
            T3 stt(N, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    for (int k = 0; k <= j; ++k) {
                        double v = F[i].hess[DA2Var<N>::hidx(j, k)];
                        stt[i][j][k] = v;
                        stt[i][k][j] = v;
                    }
            return stt;
        }

    } // namespace da

    // ========================================================================
    // SECTION 2: CONFIGURATION & RESULT TYPES
    // ========================================================================

    constexpr double MU_EARTH = 398600.4418;
    constexpr double J2 = 1.08263e-3;
    constexpr double RE = 6378.137;
    constexpr double H_SCALE = 60.0;

    struct PropConfig {
        // Time stepping
        double dt_step_s = 60.0;

        // Force model flags
        bool use_j2 = true;
        bool use_drag = true;
        bool use_srp = false;

        // Drag model
        double bstar = 0.0;
        double f107 = 150.0;
        double ap = 15.0;

        // Integrator selection (RK78 is now default)
        bool use_rk78 = true;
        bool adaptive = false;
        double abs_tol = 1e-9;
        double rel_tol = 1e-9;

        // Solar radiation pressure
        double cr_am = 0.01;

        // Order-2 DA propagation
        bool order2 = false;
    };

    struct PropResult {
        std::array<double, 6> state;
        std::array<std::array<double, 6>, 6> stm;
    };

    struct PropResult2 {
        std::array<double, 6> state;
        std::array<std::array<double, 6>, 6> stm;
        std::vector<std::vector<std::vector<double>>> stt;
    };


    // ===== Forward declarations (MOVE HERE) =====
    std::array<std::array<double, 6>, 6>
    propagate_covariance(
            const std::array<std::array<double, 6>, 6>& P0,
            const std::array<std::array<double, 6>, 6>& stm);

    std::array<std::array<double, 6>, 6>
    propagate_covariance_order2(
            const std::array<std::array<double, 6>, 6>& P0,
            const PropResult2& res2);


    // ========================================================================
    // SECTION 3: FORCE MODELS (DA-compatible)
    // ========================================================================

    template <int N>
    da::DAVar<N> da_sqrt_t(const da::DAVar<N>& x) {
        return da::da_sqrt(x);
    }

    template <int N>
    da::DA2Var<N> da_sqrt_t(const da::DA2Var<N>& x) {
        return da::da2_sqrt(x);
    }

    // Gravity + J2 perturbation
    template <typename DV>
    std::array<DV, 3> gravity_j2(const DV& x, const DV& y, const DV& z) {
        auto r2 = x * x + y * y + z * z;
        auto r = da_sqrt_t(r2);
        auto r3 = r2 * r;
        auto r5 = r3 * r2;
        auto mu_r3 = DV(-MU_EARTH) / r3;

        auto ax = x * mu_r3;
        auto ay = y * mu_r3;
        auto az = z * mu_r3;

        auto fac_j2 = DV(1.5 * J2 * MU_EARTH * RE * RE) / r5;
        auto z2_r2 = z * z / r2;

        ax = ax + x * fac_j2 * (z2_r2 * DV(5.0) - DV(1.0));
        ay = ay + y * fac_j2 * (z2_r2 * DV(5.0) - DV(1.0));
        az = az + z * fac_j2 * (z2_r2 * DV(5.0) - DV(3.0));

        return {ax, ay, az};
    }

    // Atmospheric drag
    template <typename DV>
    std::array<DV, 3> drag_accel(const DV& r_mag, const DV& vx, const DV& vy,
                                  const DV& vz, double bstar) {
        if (bstar < 1e-20) {
            DV zero;
            zero.val = 0.0;
            return {zero, zero, zero};
        }

        auto vmag = da_sqrt_t(vx * vx + vy * vy + vz * vz);
        double rho0 = bstar;
        auto rho = DV(rho0 * std::exp(-(r_mag.val - RE) / H_SCALE));
        auto drag_coeff = DV(-0.5) * rho * vmag;

        return {vx * drag_coeff, vy * drag_coeff, vz * drag_coeff};
    }

    // Solar radiation pressure (pushes in +X, away from Sun)
    template <typename DV>
    std::array<DV, 3> srp_accel(double cr_am_m2kg) {
        constexpr double P_SRP = 4.56e-9;
        double a_srp = P_SRP * cr_am_m2kg;
        DV zero;
        zero.val = 0.0;
        return {DV(a_srp), zero, zero};
    }

    // ========================================================================
    // SECTION 4: RK7(8) DORMAND-PRINCE INTEGRATOR
    // ========================================================================

    namespace rk78_coeff {
        constexpr double c[13] = {
            0.0, 0.526001519587677318e-1, 0.789002279381515978e-1,
            0.118350341907227397e0, 0.281649658092772603e0,
            0.333333333333333333e0, 0.25e0, 0.307692307692307692e0,
            0.651282051282051282e0, 0.6e0, 0.857142857142857142e0, 1.0, 1.0};

        constexpr double a[13][13] = {
            {0},
            {5.26001519587677318e-2},
            {1.97250569845378994e-2, 5.91751709536136983e-2},
            {2.95875854768068491e-2, 0, 8.87627564304205475e-2},
            {2.41365641822914963e-1, 0, -8.84549479328286085e-1,
             9.24834003261792003e-1},
            {3.7037037037037037e-2, 0, 0, 1.70828608729473871e-1,
             1.25467687566822429e-1},
            {3.7109375e-2, 0, 0, 1.70252211019544039e-1,
             6.02165389804559092e-2, -1.7578125e-2},
            {3.70920001185047927e-2, 0, 0, 1.70383925712239993e-1,
             1.07262030446373284e-1, -1.53194377486244882e-2,
             8.27378916792447911e-3},
            {6.24110958716075717e-1, 0, 0, -3.36089262944694129e0,
             -8.68219346841726006e-1, 2.72650648845738596e1,
             2.01540675504778934e1, -4.34898841810699588e1},
            {4.77662536438264366e-1, 0, 0, -2.48811461997166764e0,
             -1.76287080009571588e-1, 1.79989028948022595e1,
             8.88604957030499060e0, -2.49607993600813902e1,
             8.08833156971006187e-1},
            {-9.31463175788752287e-1, 0, 0, 5.64921658898383065e0,
             5.25849293967075292e-1, -3.01890788765146788e1,
             -1.25984660808467874e1, 3.70011551284967882e1,
             7.94842736506603932e-2, 4.16997685086485485e-1},
            {2.27331014751653820e-1, 0, 0, -1.05344954667372501e1,
             -2.00087205822486249e0, 1.56188776810425616e1,
             1.60639537096148620e1, -1.05823994216558174e1,
             6.99980777049686202e-1, 5.05753418697966050e-2,
             3.93207804527917290e-2, 0.0},
            {5.42937341165687296e-2, 0, 0, 0, 0, 4.45031289275240888e0,
             1.89151789931450038e0, -5.8012039600105847e0,
             3.1116436695781989e-1, -1.52160949662516078e-1,
             2.01365400804030348e-1, 4.47106157277725905e-2, 0.0}};

        constexpr double b8[13] = {
            5.42937341165687296e-2, 0, 0, 0, 0, 4.45031289275240888e0,
            1.89151789931450038e0, -5.8012039600105847e0,
            3.1116436695781989e-1, -1.52160949662516078e-1,
            2.01365400804030348e-1, 4.47106157277725905e-2, 0.0};

        constexpr double b7[13] = {
            4.17474911415302462e-2, 0, 0, 0, 0, 4.59772537927328376e0,
            7.48985350394364451e0, -1.80420529928680670e1,
            9.16630253978622205e-1, 0.0, 0.0, -1.57059164374920028e-1,
            1.0e-1};
    }

    // RK78 step implementation
    template <typename DV>
    std::pair<std::array<DV, 6>, double> rk78_step(
        const std::array<DV, 6>& state, double dt, const PropConfig& cfg) {
        using namespace rk78_coeff;
        constexpr int S = 13;

        std::array<std::array<DV, 6>, S> K;

        auto force = [&](const std::array<DV, 6>& s) -> std::array<DV, 6> {
            auto x = s[0], y = s[1], z = s[2];
            auto vx = s[3], vy = s[4], vz = s[5];

            auto [ax, ay, az] = gravity_j2(x, y, z);

            auto r_mag = da_sqrt_t(x * x + y * y + z * z);

            if (cfg.use_drag && cfg.bstar != 0.0) {
                auto [drag_x, drag_y, drag_z] =
                    drag_accel(r_mag, vx, vy, vz, cfg.bstar);
                ax = ax + drag_x;
                ay = ay + drag_y;
                az = az + drag_z;
            }

            if (cfg.use_srp) {
                auto [srp_x, srp_y, srp_z] = srp_accel(cfg.cr_am);
                ax = ax + srp_x;
                ay = ay + srp_y;
                az = az + srp_z;
            }

            return {vx, vy, vz, ax, ay, az};
        };

        K[0] = force(state);

        for (int i = 1; i < S; ++i) {
            std::array<DV, 6> si = state;
            for (int j = 0; j < i; ++j) {
                double aij = a[i][j];
                if (aij == 0.0) continue;
                for (int k = 0; k < 6; ++k)
                    si[k] = si[k] + K[j][k] * (dt * aij);
            }
            K[i] = force(si);
        }

        std::array<DV, 6> out = state;
        for (int i = 0; i < S; ++i) {
            if (b8[i] == 0.0) continue;
            for (int k = 0; k < 6; ++k)
                out[k] = out[k] + K[i][k] * (dt * b8[i]);
        }

        // Error estimation
        double err = 0.0;
        for (int i = 0; i < S; ++i) {
            double db = b8[i] - b7[i];
            if (db == 0.0) continue;
            for (int k = 0; k < 6; ++k) {
                double e = dt * db * K[i][k].val;
                err += e * e;
            }
        }
        err = std::sqrt(err / 6.0);

        return {out, err};
    }

    // ========================================================================
    // SECTION 5: ADAPTIVE STEP-SIZE CONTROL
    // ========================================================================

    template <typename DV>
    std::array<DV, 6> propagate_adaptive_rk78(
        const std::array<DV, 6>& state0, double dt_total,
        const PropConfig& cfg) {
        constexpr double ORDER = 8.0;
        constexpr double ALPHA = 0.7 / ORDER;
        constexpr double BETA = 0.4 / ORDER;
        constexpr double FAC_MIN = 0.1;
        constexpr double FAC_MAX = 5.0;
        constexpr double FAC_SAF = 0.9;

        double tol = std::max(cfg.abs_tol, cfg.rel_tol);
        auto state = state0;
        double t = 0.0;
        double h = cfg.dt_step_s;
        if (dt_total < 0) h = -std::abs(h);

        double err_prev = 1.0;

        while (std::abs(t) < std::abs(dt_total)) {
            if (std::abs(t + h) > std::abs(dt_total))
                h = dt_total - t;

            auto [next, err] = rk78_step<DV>(state, h, cfg);

            if (err < tol || std::abs(h) < 0.01) {
                state = next;
                t += h;
                double fac = FAC_SAF * std::pow(err / tol + 1e-10, -ALPHA) *
                             std::pow(err_prev + 1e-10, BETA);
                fac = std::min(FAC_MAX, std::max(FAC_MIN, fac));
                h *= fac;
                err_prev = std::max(err, 1e-10);
            } else {
                h *= 0.5;
            }
        }
        return state;
    }

    // ========================================================================
// SECTION 6: PUBLIC API
// ========================================================================

// Forward declarations
    std::array<std::array<double, 6>, 6> 
propagate_covariance(
        const std::array<std::array<double, 6>, 6>& P0,
        const std::array<std::array<double, 6>, 6>& stm);

    std::array<std::array<double, 6>, 6> 
propagate_covariance_order2(
        const std::array<std::array<double, 6>, 6>& P0,
        const PropResult2& res2);

    std::array<double, 6> propagate_mean_order2(
        const std::array<std::array<double, 6>, 6>& P0,
        const PropResult2& res2);

    // Primary API: RK78 propagation (order-1)
    inline PropResult da_propagate_rk78(
        const std::array<double, 6>& x0_km, double dt_total_s,
        const PropConfig& cfg = PropConfig{}) {
        using DV = da::DA6;

        std::array<DV, 6> state;
        for (int i = 0; i < 6; ++i)
            state[i] = DV::identity(i, x0_km[i]);

        if (cfg.adaptive) {
            state = propagate_adaptive_rk78<DV>(state, dt_total_s, cfg);
        }
        else {
            int n = static_cast<int>(
                std::ceil(std::abs(dt_total_s) / cfg.dt_step_s));
            if (n < 1) n = 1;
            double dt = dt_total_s / n;
            for (int step = 0; step < n; ++step) {
                auto [next, err] = rk78_step<DV>(state, dt, cfg);
                state = next;
            }
        }

        PropResult res;
        for (int i = 0; i < 6; ++i) {
            res.state[i] = state[i].val;
            for (int j = 0; j < 6; ++j)
                res.stm[i][j] = state[i].grad[j];
        }
        return res;
    }

    // Legacy alias: da_propagate = da_propagate_rk78 (backward compatibility)
    inline PropResult da_propagate(
        const std::array<double, 6>& x0_km, double dt_total_s,
        const PropConfig& cfg = PropConfig{}) {
        return da_propagate_rk78(x0_km, dt_total_s, cfg);
    }

    // Order-2 RK78 propagation
    inline PropResult2 da_propagate_order2(
        const std::array<double, 6>& x0_km, double dt_total_s,
        const PropConfig& cfg = PropConfig{}) {
        using DV = da::DA2_6;

        std::array<DV, 6> state;
        for (int i = 0; i < 6; ++i)
            state[i] = DV::identity(i, x0_km[i]);

        if (cfg.adaptive) {
            state = propagate_adaptive_rk78<DV>(state, dt_total_s, cfg);
        }
        else {
            int n = static_cast<int>(
                std::ceil(std::abs(dt_total_s) / cfg.dt_step_s));
            if (n < 1) n = 1;
            double dt = dt_total_s / n;
            for (int step = 0; step < n; ++step) {
                auto [next, err] = rk78_step<DV>(state, dt, cfg);
                state = next;
            }
        }

        PropResult2 res;
        for (int i = 0; i < 6; ++i) {
            res.state[i] = state[i].val;
            for (int j = 0; j < 6; ++j)
                res.stm[i][j] = state[i].grad[j];
        }
        res.stt = da::stt_from_da2(state);
        return res;
    }
    // ===== Forward declarations needed by PropResult2Full =====
    std::array<std::array<double, 6>, 6>
        propagate_covariance(
            const std::array<std::array<double, 6>, 6>& P0,
            const std::array<std::array<double, 6>, 6>& stm);

    std::array<std::array<double, 6>, 6>
        propagate_covariance_order2(
            const std::array<std::array<double, 6>, 6>& P0,
            const PropResult2& res2);
    ``



    // Full bundle: state + STM + STT + mean correction
    inline PropResult2Full da_propagate_order2_full(
        const std::array<double, 6>& x0_km, double dt_total_s,
        const std::array<std::array<double, 6>, 6>& P0,
        const PropConfig& cfg = PropConfig{}) {
        auto res2 = da_propagate_order2(x0_km, dt_total_s, cfg);

        PropResult2Full full;
        full.state = res2.state;
        full.stm = res2.stm;
        full.stt = res2.stt;
        full.mean = propagate_mean_order2(P0, res2);
        return full;
    }

    // ========================================================================
    // SECTION 7: COVARIANCE PROPAGATION
    // ========================================================================

    inline std::array<std::array<double, 6>, 6> propagate_covariance(
        const std::array<std::array<double, 6>, 6>& P0,
        const std::array<std::array<double, 6>, 6>& stm) {
        std::array<std::array<double, 6>, 6> tmp{}, Pf{};

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < 6; ++k)
                    tmp[i][j] += stm[i][k] * P0[k][j];

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < 6; ++k)
                    Pf[i][j] += tmp[i][k] * stm[j][k];

        return Pf;
    }

    inline std::array<double, 6> propagate_mean_order2(
        const std::array<std::array<double, 6>, 6>& P0,
        const PropResult2& res2) {
        std::array<double, 6> mean = res2.state;
        const auto& stt = res2.stt;

        for (int i = 0; i < 6; ++i) {
            double corr = 0.0;
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < 6; ++k)
                    corr += stt[i][j][k] * P0[j][k];
            mean[i] += 0.5 * corr;
        }
        return mean;
    }

    inline std::array<std::array<double, 6>, 6>
    propagate_covariance_order2(
        const std::array<std::array<double, 6>, 6>& P0,
        const PropResult2& res2) {
        auto Pf = propagate_covariance(P0, res2.stm);
        const auto& stt = res2.stt;

        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                double lam = 0.0;
                for (int k = 0; k < 6; ++k)
                    for (int l = 0; l < 6; ++l)
                        for (int m = 0; m < 6; ++m)
                            for (int n = 0; n < 6; ++n)
                                lam += stt[i][k][l] * stt[j][m][n] * P0[k][m] *
                                       P0[l][n];
                lam *= 0.5;
                Pf[i][j] += lam;
                if (i != j) Pf[j][i] += lam;
            }
        }

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                Pf[i][j] = 0.5 * (Pf[i][j] + Pf[j][i]);

        return Pf;
    }

    // ========================================================================
    // SECTION 8: EKF MEASUREMENT UPDATE
    // ========================================================================

    template <int M>
    struct EKFResult {
        std::array<double, 6> xp;
        std::array<std::array<double, 6>, 6> Pp;
        std::array<double, M> innovation;
        std::array<std::array<double, M>, M> S;
        double nis;
    };

    template <int M>
    EKFResult<M> ekf_update(
        const std::array<double, M>& z, const double H[M][6],
        const double R[M][M], const std::array<double, 6>& xm,
        const std::array<std::array<double, 6>, 6>& Pm) {
        EKFResult<M> out;

        // Innovation
        for (int i = 0; i < M; ++i) {
            double Hx = 0.0;
            for (int j = 0; j < 6; ++j) Hx += H[i][j] * xm[j];
            out.innovation[i] = z[i] - Hx;
        }

        // PH^T
        double PHt[6][M] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < M; ++j)
                for (int k = 0; k < 6; ++k)
                    PHt[i][j] += Pm[i][k] * H[j][k];

        // Innovation covariance S = H P H^T + R
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j) {
                double s = R[i][j];
                for (int k = 0; k < 6; ++k) s += H[i][k] * PHt[k][j];
                out.S[i][j] = s;
            }

        // Compute S^{-1} via LU with partial pivoting
        double S_inv[M][M] = {};
        {
            double Lu[M][M];
            int piv[M];
            for (int i = 0; i < M; ++i) {
                piv[i] = i;
                for (int j = 0; j < M; ++j) Lu[i][j] = out.S[i][j];
            }

            // Gaussian elimination
            for (int col = 0; col < M; ++col) {
                int pivot_row = col;
                double max_val = std::abs(Lu[col][col]);
                for (int row = col + 1; row < M; ++row)
                    if (std::abs(Lu[row][col]) > max_val) {
                        max_val = std::abs(Lu[row][col]);
                        pivot_row = row;
                    }
                if (pivot_row != col) {
                    std::swap(piv[col], piv[pivot_row]);
                    for (int k = 0; k < M; ++k)
                        std::swap(Lu[col][k], Lu[pivot_row][k]);
                }
                if (std::abs(Lu[col][col]) < 1e-30)
                    throw std::runtime_error("EKF: S singular");
                double inv_diag = 1.0 / Lu[col][col];
                for (int row = col + 1; row < M; ++row) {
                    double fac = Lu[row][col] * inv_diag;
                    for (int k = col; k < M; ++k)
                        Lu[row][k] -= fac * Lu[col][k];
                }
            }

            // Back-substitution with pivoting
            for (int c = 0; c < M; ++c) {
                double b[M] = {};
                b[c] = 1.0;
                double x[M];
                for (int i = 0; i < M; ++i) x[i] = b[piv[i]];

                for (int row = 0; row < M; ++row)
                    for (int k = 0; k < row; ++k)
                        x[row] -= Lu[row][k] * x[k];

                for (int row = M - 1; row >= 0; --row) {
                    for (int k = row + 1; k < M; ++k)
                        x[row] -= Lu[row][k] * x[k];
                    x[row] /= Lu[row][row];
                }
                for (int row = 0; row < M; ++row) S_inv[row][c] = x[row];
            }
        }

        // Kalman gain
        double K[6][M] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < M; ++j)
                for (int k = 0; k < M; ++k)
                    K[i][j] += PHt[i][k] * S_inv[k][j];

        // Posterior mean
        out.xp = xm;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < M; ++j)
                out.xp[i] += K[i][j] * out.innovation[j];

        // Posterior covariance (Joseph form)
        double A[6][6] = {};
        for (int i = 0; i < 6; ++i) A[i][i] = 1.0;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < M; ++k)
                    A[i][j] -= K[i][k] * H[k][j];

        double APm[6][6] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < 6; ++k)
                    APm[i][j] += A[i][k] * Pm[k][j];

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) {
                out.Pp[i][j] = 0.0;
                for (int k = 0; k < 6; ++k)
                    out.Pp[i][j] += APm[i][k] * A[j][k];
            }

        double KR[6][M] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < M; ++j)
                for (int k = 0; k < M; ++k)
                    KR[i][j] += K[i][k] * R[k][j];

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                for (int k = 0; k < M; ++k)
                    out.Pp[i][j] += KR[i][k] * K[j][k];

        // Symmetrize
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                out.Pp[i][j] = 0.5 * (out.Pp[i][j] + out.Pp[j][i]);

        // NIS
        out.nis = 0.0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                out.nis +=
                    out.innovation[i] * S_inv[i][j] * out.innovation[j];

        return out;
    }

} // namespace casas
#endif // CASAS_DA_HPP
``