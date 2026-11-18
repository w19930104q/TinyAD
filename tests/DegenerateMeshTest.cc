/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 *
 * Regression tests for degenerate meshes (collapsed / repeated-vertex
 * elements), repeated parameter blocks, deferred lambda instantiation,
 * and sparse index consistency of assembled Jacobians/Hessians.
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/VectorFunction.hh>
#include <Eigen/SparseCholesky>
#include <random>
#include <set>

namespace
{

// Central finite differences of a scalar function
Eigen::VectorXd fd_gradient(
        const std::function<double(const Eigen::VectorXd&)>& f,
        const Eigen::VectorXd& x,
        double h = 1e-5)
{
    Eigen::VectorXd g(x.size());
    Eigen::VectorXd xp = x, xm = x;
    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        xp[i] = x[i] + h; xm[i] = x[i] - h;
        g[i] = (f(xp) - f(xm)) / (2.0 * h);
        xp[i] = x[i]; xm[i] = x[i];
    }
    return g;
}

Eigen::MatrixXd fd_hessian(
        const std::function<double(const Eigen::VectorXd&)>& f,
        const Eigen::VectorXd& x,
        double h = 1e-4)
{
    const Eigen::Index n = x.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd xpp = x, xpm = x, xmp = x, xmm = x;
    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            xpp[i] += h; xpp[j] += h;
            xpm[i] += h; xpm[j] -= h;
            xmp[i] -= h; xmp[j] += h;
            xmm[i] -= h; xmm[j] -= h;
            H(i, j) = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h);
            xpp = x; xpm = x; xmp = x; xmm = x;
        }
    }
    return H;
}

Eigen::MatrixXd fd_jacobian(
        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& r,
        const Eigen::VectorXd& x,
        double h = 1e-5)
{
    Eigen::MatrixXd J(r(x).size(), x.size());
    Eigen::VectorXd xp = x, xm = x;
    for (Eigen::Index j = 0; j < x.size(); ++j)
    {
        xp[j] = x[j] + h; xm[j] = x[j] - h;
        J.col(j) = (r(xp) - r(xm)) / (2.0 * h);
        xp[j] = x[j]; xm[j] = x[j];
    }
    return J;
}

double max_rel_err(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
    const double denom = std::max(1.0, B.cwiseAbs().maxCoeff());
    return (A - B).cwiseAbs().maxCoeff() / denom;
}

bool sparse_indices_in_range(const Eigen::SparseMatrix<double>& M)
{
    for (int k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
            if (it.row() < 0 || it.row() >= M.rows() || it.col() < 0 || it.col() >= M.cols())
                return false;
    return true;
}

} // namespace

// Degenerate triangles with repeated vertex indices.
// Edge-length energy stays finite on degenerate elements and yields
// legitimate zero Hessian blocks for the repeated vertex.
TEST(DegenerateMeshTest, DegenerateTriangleHessian)
{
    // 4 vertices, one good triangle and two degenerate ones (repeated vertex)
    Eigen::MatrixXi F(3, 3);
    F << 0, 1, 2,   // good
         0, 2, 2,   // degenerate: vertex 2 twice
         1, 1, 3;   // degenerate: vertex 1 twice
    const int n_v = 4;

    auto func = TinyAD::scalar_function<2>(TinyAD::range(n_v));
    func.add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int f = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(F(f, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f, 2));
        // Finite edge-length energy (no inverses / determinants)
        return (a - b).squaredNorm() + (b - c).squaredNorm() + (c - a).squaredNorm();
    });

    std::mt19937 rng(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd x(2 * n_v);
    for (auto& v : x) v = nd(rng);

    auto f_passive = [&] (const Eigen::VectorXd& xx) { return func.eval(xx); };
    auto [fv, g, H] = func.eval_with_derivatives(x);
    const Eigen::MatrixXd Hd = Eigen::MatrixXd(H);

    EXPECT_EQ(H.rows(), 2 * n_v);
    EXPECT_EQ(H.cols(), 2 * n_v);
    EXPECT_LT((g - fd_gradient(f_passive, x)).norm(), 1e-5);
    EXPECT_LT(max_rel_err(Hd, fd_hessian(f_passive, x)), 1e-4);
    EXPECT_TRUE(sparse_indices_in_range(H));
    EXPECT_LT((Hd - Hd.transpose()).norm(), 1e-12);
}

// Repeated parameter block access (same handle requested twice must be
// de-duplicated) combined with conditional (partial) variable access.
TEST(DegenerateMeshTest, RepeatedAndPartialAccess)
{
    const int n_v = 5;
    auto func = TinyAD::scalar_function<2>(TinyAD::range(n_v));
    // Valence-3 element but the lambda sometimes touches only 2 handles,
    // and one handle is requested twice.
    func.add_elements<3>(TinyAD::range(2), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int e = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(0);
        Eigen::Vector<T, 2> b = element.variables(1);
        Eigen::Vector<T, 2> a2 = element.variables(0); // repeated block
        if (e == 0)
        {
            Eigen::Vector<T, 2> c = element.variables(2);
            return (a - b).squaredNorm() + (b - c).squaredNorm() + a2.squaredNorm();
        }
        else
        {
            // Only 2 of 3 handles accessed -> local block smaller than n_element
            return (a - b).squaredNorm() + a2.squaredNorm();
        }
    });

    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd x(2 * n_v);
    for (auto& v : x) v = nd(rng);

    auto f_passive = [&] (const Eigen::VectorXd& xx) { return func.eval(xx); };
    auto [fv, g, H] = func.eval_with_derivatives(x);

    EXPECT_LT((g - fd_gradient(f_passive, x)).norm(), 1e-5);
    EXPECT_LT(max_rel_err(Eigen::MatrixXd(H), fd_hessian(f_passive, x)), 1e-4);
    EXPECT_TRUE(sparse_indices_in_range(H));
}

// Vector function on a degenerate mesh: Jacobian vs FD, residual
// consistency, and per-residual Hessian index ranges.
TEST(DegenerateMeshTest, VectorFunctionDegenerate)
{
    Eigen::MatrixXi F(2, 3);
    F << 0, 1, 1,   // degenerate
         1, 2, 3;
    const int n_v = 4;

    auto vf = TinyAD::vector_function<2>(TinyAD::range(n_v));
    vf.add_elements<3, 3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int f = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(F(f, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f, 2));
        Eigen::Vector<T, 3> r;
        r << (a - b).squaredNorm(), (b - c).squaredNorm(), (c - a).dot(a);
        return r;
    });

    Eigen::VectorXd x(2 * n_v);
    x << 0.3, -0.7, 1.2, 0.4, -1.1, 0.9, 0.2, -0.3;

    auto r_passive = [&] (const Eigen::VectorXd& xx) { return vf.eval(xx); };
    auto [r, J] = vf.eval_with_jacobian(x);

    EXPECT_EQ(r.size(), 6);
    EXPECT_LT((r - r_passive(x)).norm(), 1e-12);
    EXPECT_EQ(J.rows(), 6);
    EXPECT_EQ(J.cols(), 2 * n_v);
    EXPECT_LT(max_rel_err(Eigen::MatrixXd(J), fd_jacobian(r_passive, x)), 1e-4);
    EXPECT_TRUE(sparse_indices_in_range(J));

    // Second-order path
    auto [r2, J2, Hs] = vf.eval_with_derivatives(x);
    EXPECT_EQ((Eigen::Index)Hs.size(), r2.size());
    EXPECT_LT((r2 - r).norm(), 1e-12);
    EXPECT_LT((Eigen::MatrixXd(J2) - Eigen::MatrixXd(J)).norm(), 1e-12);
    for (const auto& Hm : Hs)
    {
        EXPECT_EQ(Hm.rows(), 2 * n_v);
        EXPECT_EQ(Hm.cols(), 2 * n_v);
        EXPECT_TRUE(sparse_indices_in_range(Hm));
    }
}

// Multiple objective terms with overlapping handles and deferred lambdas:
// assembled gradient/Hessian consistency and quadratic Hessian stability.
TEST(DegenerateMeshTest, MultipleTermsAssembledResidual)
{
    const int n_v = 6;
    auto func = TinyAD::scalar_function<1>(TinyAD::range(n_v));
    func.add_elements<2>(TinyAD::range(3), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int e = (int)element.handle;
        T a = element.variable(e);
        T b = element.variable(e + 1);
        return sqr(a - 2.0 * b);
    });
    func.add_elements<3>(TinyAD::range(2), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int e = (int)element.handle;
        T a = element.variable(e);
        T b = element.variable(e + 2);
        T c = element.variable(e + 3);
        return 0.5 * sqr(a + b - c);
    });

    Eigen::VectorXd x(n_v);
    x << 0.5, -1.0, 2.0, 0.25, -3.0, 1.5;

    auto f_passive = [&] (const Eigen::VectorXd& xx) { return func.eval(xx); };
    auto [fv, g, H] = func.eval_with_derivatives(x);

    EXPECT_LT((g - fd_gradient(f_passive, x)).norm(), 1e-6);
    EXPECT_LT(max_rel_err(Eigen::MatrixXd(H), fd_hessian(f_passive, x)), 1e-4);

    // Quadratic: Hessian must be x-independent
    const Eigen::SparseMatrix<double> H0 = func.eval_hessian_of_quadratic();
    EXPECT_LT((Eigen::MatrixXd(H0) - Eigen::MatrixXd(H)).norm(), 1e-12);
}

// Fixed-solution invariance: at a minimizer with pinned DOFs, the free
// residual and the Newton step must vanish.
TEST(DegenerateMeshTest, FixedSolutionInvariance)
{
    // Quadratic Dirichlet energy on a small chain, vertices 0 and 4 pinned.
    const int n_v = 5;
    const std::set<Eigen::Index> pinned = {0, 4};
    auto func = TinyAD::scalar_function<1>(TinyAD::range(n_v));
    func.add_elements<2>(TinyAD::range(n_v - 1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int e = (int)element.handle;
        T a = element.variable(e);
        T b = element.variable(e + 1);
        return sqr(a - b);
    });

    Eigen::VectorXd x_star(n_v);
    x_star << 3.0, 3.0, 3.0, 3.0, 3.0; // minimizer with pinning (constant)
    auto [f, g, H] = func.eval_with_derivatives(x_star);

    // Free-DOF residual must vanish at the fixed solution
    double free_res = 0.0;
    for (Eigen::Index i = 0; i < n_v; ++i)
        if (!pinned.count(i)) free_res += g[i] * g[i];
    EXPECT_LT(std::sqrt(free_res), 1e-12);

    // Solve H d = -g with pinned rows/cols identity-ized; d must be zero
    Eigen::VectorXd rhs = -g;
    std::vector<Eigen::Triplet<double>> trips;
    for (int k = 0; k < H.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it)
        {
            if (pinned.count(it.row()) || pinned.count(it.col())) continue;
            trips.emplace_back(it.row(), it.col(), it.value());
        }
    for (Eigen::Index p : pinned)
    {
        trips.emplace_back(p, p, 1.0);
        rhs[p] = 0.0;
    }
    Eigen::SparseMatrix<double> Hff(n_v, n_v);
    Hff.setFromTriplets(trips.begin(), trips.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(Hff);
    ASSERT_EQ(solver.info(), Eigen::Success);
    const Eigen::VectorXd direction = solver.solve(rhs);

    double free_step = 0.0;
    for (Eigen::Index i = 0; i < n_v; ++i)
        if (!pinned.count(i)) free_step += direction[i] * direction[i];
    EXPECT_LT(std::sqrt(free_step), 1e-10);
}

// Collapsed elements produce sqrt(0) / hypot(0, 0) inside AD.
// Values and derivatives must stay finite (subgradient convention).
TEST(DegenerateMeshTest, CollapsedEdgeSqrtZero)
{
    // Two vertices pinned to the same position: edge length sqrt(0) inside AD.
    auto func = TinyAD::scalar_function<1>(TinyAD::range(2));
    func.add_elements<2>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T a = element.variable(0);
        T b = element.variable(1);
        T d = b - a;
        return sqrt(d * d);
    });

    Eigen::VectorXd x(2);
    x << 1.0, 1.0; // collapsed
    auto [f, g, H] = func.eval_with_derivatives(x);
    EXPECT_TRUE(std::isfinite(f));
    EXPECT_TRUE(g.allFinite());
    EXPECT_TRUE(Eigen::MatrixXd(H).allFinite());
    EXPECT_DOUBLE_EQ(f, 0.0);

    // Same via hypot
    auto func2 = TinyAD::scalar_function<2>(TinyAD::range(2));
    func2.add_elements<2>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 2> a = element.variables(0);
        Eigen::Vector<T, 2> b = element.variables(1);
        return hypot(b[0] - a[0], b[1] - a[1]);
    });
    auto [f2, g2, H2] = func2.eval_with_derivatives(Eigen::VectorXd::Zero(4));
    EXPECT_TRUE(std::isfinite(f2));
    EXPECT_TRUE(g2.allFinite());
    EXPECT_TRUE(Eigen::MatrixXd(H2).allFinite());

    // Multiplication by a structurally zero factor must not produce NaN either
    auto func3 = TinyAD::scalar_function<1>(TinyAD::range(1));
    func3.add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        T a = element.variable(0);
        T s = sqrt(a * a);
        return (a - a) * s;
    });
    auto [f3, g3, H3] = func3.eval_with_derivatives(Eigen::VectorXd::Zero(1));
    EXPECT_TRUE(std::isfinite(f3));
    EXPECT_TRUE(g3.allFinite());
    EXPECT_TRUE(Eigen::MatrixXd(H3).allFinite());

    // Away from zero, sqrt derivatives must be unaffected by the guard
    auto func4 = TinyAD::scalar_function<1>(TinyAD::range(1));
    func4.add_elements<1>(TinyAD::range(1), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        return sqrt(element.variable(0));
    });
    Eigen::VectorXd x4(1);
    x4 << 2.0;
    auto [f4, g4, H4] = func4.eval_with_derivatives(x4);
    EXPECT_DOUBLE_EQ(f4, std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(g4[0], 0.5 / std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(Eigen::MatrixXd(H4)(0, 0), -0.25 / std::pow(2.0, 1.5));
}

// Move semantics: the deferred lambda and the settings reference must stay
// valid after move-construction / move-assignment, even if the moved-from
// object is destroyed.
TEST(DegenerateMeshTest, MoveKeepsDeferredLambdaAlive)
{
    Eigen::VectorXd x(4);
    x << 1.0, 0.5, -1.0, 2.0;

    // Reference values from a function that never moves
    auto make_func = [] ()
    {
        auto f = TinyAD::scalar_function<1>(TinyAD::range(4));
        f.add_elements<2>(TinyAD::range(3), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
        {
            using T = TINYAD_SCALAR_TYPE(element);
            T a = element.variable((int)element.handle);
            T b = element.variable((int)element.handle + 1);
            return sqr(a - b);
        });
        return f;
    };
    auto ref_func = make_func();
    const auto ref = ref_func.eval_with_derivatives(x);

    // Move-construct, then destroy the source object before evaluating
    auto moved = [&] ()
    {
        auto f1 = make_func();
        auto f2 = std::move(f1);
        return f2;
    }();
    const auto mv = moved.eval_with_derivatives(x);
    EXPECT_DOUBLE_EQ(std::get<0>(mv), std::get<0>(ref));
    EXPECT_LT((std::get<1>(mv) - std::get<1>(ref)).norm(), 1e-12);
    EXPECT_LT((Eigen::MatrixXd(std::get<2>(mv)) - Eigen::MatrixXd(std::get<2>(ref))).norm(), 1e-12);

    // Move-assign, then destroy the source object before evaluating
    auto move_assigned = [&] ()
    {
        auto f1 = make_func();
        auto f3 = TinyAD::scalar_function<1>(TinyAD::range(1));
        f3 = std::move(f1);
        return f3;
    }();
    const auto ma = move_assigned.eval_with_derivatives(x);
    EXPECT_DOUBLE_EQ(std::get<0>(ma), std::get<0>(ref));
    EXPECT_LT((std::get<1>(ma) - std::get<1>(ref)).norm(), 1e-12);
    EXPECT_LT((Eigen::MatrixXd(std::get<2>(ma)) - Eigen::MatrixXd(std::get<2>(ref))).norm(), 1e-12);

    // Settings changed on the moved-to object must take effect (the objective
    // terms must not keep referencing the moved-from object's settings).
    moved.settings.n_threads = 1;
    const auto after_setting = moved.eval_with_derivatives(x);
    EXPECT_DOUBLE_EQ(std::get<0>(after_setting), std::get<0>(ref));

    // Same for vector functions
    auto make_vfunc = [] ()
    {
        auto vf = TinyAD::vector_function<1>(TinyAD::range(4));
        vf.add_elements<2, 1>(TinyAD::range(3), [] (auto& element) -> TINYAD_VECTOR_TYPE(element)
        {
            using T = TINYAD_SCALAR_TYPE(element);
            T a = element.variable((int)element.handle);
            T b = element.variable((int)element.handle + 1);
            return Eigen::Vector<T, 1>(a - b);
        });
        return vf;
    };
    auto moved_vf = [&] ()
    {
        auto vf1 = make_vfunc();
        return std::move(vf1);
    }();
    EXPECT_LT((moved_vf.eval(x) - make_vfunc().eval(x)).norm(), 1e-12);
}

// Valence-0 and constant elements: zero Hessian blocks, no crash.
TEST(DegenerateMeshTest, ValenceZeroElements)
{
    auto func = TinyAD::scalar_function<1>(TinyAD::range(3));
    func.add_elements<0>(TinyAD::range(2), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Accesses no variables at all
        return TINYAD_SCALAR_TYPE(element)(1.5);
    });
    Eigen::VectorXd x(3);
    x << 0.1, -0.2, 0.3;

    auto [f, g, H] = func.eval_with_derivatives(x);
    EXPECT_DOUBLE_EQ(f, 3.0);
    EXPECT_LT(g.norm(), 1e-12);
    EXPECT_EQ(H.nonZeros(), 0);
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 3);

    auto [fp, gp, Hp] = func.eval_with_hessian_proj(x);
    EXPECT_DOUBLE_EQ(fp, 3.0);
    EXPECT_EQ(Hp.nonZeros(), 0);
}

// Large randomized mesh with heavy degeneracy (duplicate vertices per face).
// Gradient/Hessian vs FD, symmetry, and positive-definiteness of the
// projected Hessian.
TEST(DegenerateMeshTest, RandomDegenerateMesh)
{
    const int n_v = 25, n_f = 200;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> vid(0, n_v - 1);
    std::normal_distribution<double> coord(0.0, 1.0);

    Eigen::MatrixXi F(n_f, 3);
    for (int f = 0; f < n_f; ++f)
    {
        // Bias towards degeneracy: repeated indices on purpose
        if (f % 3 == 0) { int a = vid(rng); F(f, 0) = a; F(f, 1) = a; F(f, 2) = vid(rng); }
        else if (f % 3 == 1) { int a = vid(rng), b = vid(rng); F(f, 0) = a; F(f, 1) = b; F(f, 2) = a; }
        else { F(f, 0) = vid(rng); F(f, 1) = vid(rng); F(f, 2) = vid(rng); }
    }

    auto func = TinyAD::scalar_function<2>(TinyAD::range(n_v));
    func.add_elements<3>(TinyAD::range(n_f), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int f = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(F(f, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f, 2));
        // Edge-length energy: finite even for collapsed triangles
        return (a - b).squaredNorm() + (b - c).squaredNorm() + (c - a).squaredNorm();
    });

    Eigen::VectorXd x(2 * n_v);
    for (auto& v : x) v = coord(rng);

    auto f_passive = [&] (const Eigen::VectorXd& xx) { return func.eval(xx); };
    auto [f, g, H] = func.eval_with_derivatives(x);
    const Eigen::MatrixXd Hd = Eigen::MatrixXd(H);

    EXPECT_LT((g - fd_gradient(f_passive, x)).norm(), 1e-5);
    EXPECT_LT(max_rel_err(Hd, fd_hessian(f_passive, x)), 1e-4);
    EXPECT_EQ(H.rows(), 2 * n_v);
    EXPECT_EQ(H.cols(), 2 * n_v);
    EXPECT_LT((Hd - Hd.transpose()).norm(), 1e-10);
    EXPECT_TRUE(sparse_indices_in_range(H));

    // Projected Hessian must be positive semi-definite and keep f and g
    auto [fp, gp, Hp] = func.eval_with_hessian_proj(x);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig{Eigen::MatrixXd(Hp)};
    EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-10);
    EXPECT_DOUBLE_EQ(fp, f);
    EXPECT_LT((gp - g).norm(), 1e-10);
}

// Repeated deferred instantiation with changing thread settings must not
// change results.
TEST(DegenerateMeshTest, DeferredLambdaThreadSettings)
{
    const int n_v = 8;
    auto func = TinyAD::scalar_function<1>(TinyAD::range(n_v));
    func.add_elements<3>(TinyAD::range(5), [] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int e = (int)element.handle;
        T a = element.variable(e);
        T b = element.variable(e + 1);
        T c = element.variable(e + 2);
        return sqr(a - 2.0 * b) + 0.5 * sqr(b + c);
    });
    Eigen::VectorXd x(n_v);
    x << 0.3, -1.2, 0.8, 2.1, -0.4, 1.7, -2.3, 0.9;

    const auto ref = func.eval_with_derivatives(x);
    for (int n_threads : {1, 2, 4})
    {
        func.settings.n_threads = n_threads;
        const auto cur = func.eval_with_derivatives(x);
        EXPECT_DOUBLE_EQ(std::get<0>(cur), std::get<0>(ref));
        EXPECT_LT((std::get<1>(cur) - std::get<1>(ref)).norm(), 1e-12);
        EXPECT_LT((Eigen::MatrixXd(std::get<2>(cur)) - Eigen::MatrixXd(std::get<2>(ref))).norm(), 1e-12);
    }
}

// Vector function on a random degenerate mesh: Jacobian vs FD, second-order
// path consistency, and sum-of-squares gradient vs FD.
TEST(DegenerateMeshTest, VectorFunctionRandomDegenerate)
{
    const int n_v = 10, n_f = 60;
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> vid(0, n_v - 1);
    std::normal_distribution<double> coord(0.0, 1.0);

    Eigen::MatrixXi F(n_f, 3);
    for (int f = 0; f < n_f; ++f)
    {
        if (f % 2 == 0) { int a = vid(rng); F(f, 0) = a; F(f, 1) = a; F(f, 2) = vid(rng); }
        else { F(f, 0) = vid(rng); F(f, 1) = vid(rng); F(f, 2) = vid(rng); }
    }

    auto vf = TinyAD::vector_function<2>(TinyAD::range(n_v));
    vf.add_elements<3, 2>(TinyAD::range(n_f), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int f = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(F(f, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f, 2));
        Eigen::Vector<T, 2> r;
        r << (a - b).squaredNorm() + (b - c).squaredNorm(),
             (c - a).dot(b);
        return r;
    });

    Eigen::VectorXd x(2 * n_v);
    for (auto& v : x) v = coord(rng);

    auto [r, J] = vf.eval_with_jacobian(x);
    auto r_passive = [&] (const Eigen::VectorXd& xx) { return vf.eval(xx); };

    EXPECT_EQ(r.size(), 2 * n_f);
    EXPECT_EQ(J.rows(), 2 * n_f);
    EXPECT_EQ(J.cols(), 2 * n_v);
    EXPECT_LT(max_rel_err(Eigen::MatrixXd(J), fd_jacobian(r_passive, x)), 1e-4);
    EXPECT_TRUE(sparse_indices_in_range(J));

    auto [r2, J2, Hs] = vf.eval_with_derivatives(x);
    EXPECT_LT((r2 - r).norm(), 1e-12);
    EXPECT_LT((Eigen::MatrixXd(J2) - Eigen::MatrixXd(J)).norm(), 1e-12);
    EXPECT_EQ((Eigen::Index)Hs.size(), r.size());
    for (const auto& Hm : Hs)
        EXPECT_TRUE(sparse_indices_in_range(Hm));

    // Sum-of-squares gradient via J must match FD
    auto sose = [&] (const Eigen::VectorXd& xx) { return vf.eval_sum_of_squares(xx); };
    auto [fv, g_sos, r_sos, J_sos] = vf.eval_sum_of_squares_with_derivatives(x);
    EXPECT_NEAR(fv, sose(x), 1e-10);
    EXPECT_LT((g_sos - fd_gradient(sose, x)).norm(), 1e-5);
}

// Dynamic element valences on a degenerate mesh: faces with repeated
// vertices access fewer distinct handles and must be grouped correctly.
TEST(DegenerateMeshTest, DynamicElementsDegenerate)
{
    // Mix of good and degenerate faces (repeated vertex indices)
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,   // good, 3 distinct handles
         0, 2, 2,   // degenerate, 2 distinct handles
         1, 1, 1,   // fully collapsed, 1 distinct handle
         2, 3, 0;   // good
    const int n_v = 4;

    auto eval_element = [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        const int f = (int)element.handle;
        Eigen::Vector<T, 2> a = element.variables(F(f, 0));
        Eigen::Vector<T, 2> b = element.variables(F(f, 1));
        Eigen::Vector<T, 2> c = element.variables(F(f, 2));
        return (a - b).squaredNorm() + (b - c).squaredNorm() + (c - a).squaredNorm();
    };

    // Static valence-3 version as reference
    auto func_static = TinyAD::scalar_function<2>(TinyAD::range(n_v));
    func_static.add_elements<3>(TinyAD::range(F.rows()), eval_element);

    // Dynamic version: elements are grouped by number of accessed handles
    auto func_dynamic = TinyAD::scalar_function<2>(TinyAD::range(n_v));
    func_dynamic.add_elements_dynamic<1, 2, 3>(TinyAD::range(F.rows()), eval_element);

    std::mt19937 rng(5);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd x(2 * n_v);
    for (auto& v : x) v = nd(rng);

    const auto ref = func_static.eval_with_derivatives(x);
    const auto cur = func_dynamic.eval_with_derivatives(x);

    // Dynamic grouping must reproduce the static assembly exactly
    EXPECT_DOUBLE_EQ(std::get<0>(cur), std::get<0>(ref));
    EXPECT_LT((std::get<1>(cur) - std::get<1>(ref)).norm(), 1e-12);
    EXPECT_LT((Eigen::MatrixXd(std::get<2>(cur)) - Eigen::MatrixXd(std::get<2>(ref))).norm(), 1e-12);

    // And both must match finite differences
    auto f_passive = [&] (const Eigen::VectorXd& xx) { return func_dynamic.eval(xx); };
    EXPECT_LT((std::get<1>(cur) - fd_gradient(f_passive, x)).norm(), 1e-5);
    EXPECT_LT(max_rel_err(Eigen::MatrixXd(std::get<2>(cur)), fd_hessian(f_passive, x)), 1e-4);
}
