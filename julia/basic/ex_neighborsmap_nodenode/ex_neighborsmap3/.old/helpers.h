#ifndef HELPERS_H
#define HELPERS_H
// #include "kernel.h"



#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <typeinfo>
#include <utility>
#include <any>
#include <vector>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <functional>
#include <iostream>
#include <tuple>
#include <boost/math/interpolators/cubic_b_spline.hpp>
#include <boost/math/special_functions/binomial.hpp>

// #include <libmesh/tensor_tools.h>
#include <libmesh/libmesh.h>
#include <libmesh/point_locator_tree.h>
#include "libmesh/string_to_enum.h"

#include <libmesh/point_locator_tree.h>
// #include "./exact_solution.h"
#include "libmesh/equation_systems.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include <libmesh/fem_system.h>

#include <libmesh/elem.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exact_solution.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/getpot.h>

#include <libmesh/dirichlet_boundaries.h>

#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_refinement.h>
#include <libmesh/replicated_mesh.h>
#include <libmesh/serial_mesh.h>

#include <libmesh/node.h>
#include <libmesh/system_norm.h>

#include <libmesh/quadrature_composite.h>

#include <libmesh/point_locator_tree.h>
#include "libmesh/string_to_enum.h"

// #include "./operators.h"
// #include "./exact_solution.h"
// #include ".libs/solution_function.h"

#include <libmesh/discontinuity_measure.h>
#include <libmesh/error_vector.h>
#include <libmesh/exact_error_estimator.h>
#include <libmesh/hp_coarsentest.h>
#include <libmesh/hp_selector.h>
#include <libmesh/hp_singular.h>
#include <libmesh/kelly_error_estimator.h>
#include <libmesh/patch_recovery_error_estimator.h>
#include <libmesh/sibling_coupling.h>
#include <libmesh/uniform_refinement_estimator.h>
#include "libmesh/function_base.h"
#include <libmesh/parsed_fem_function.h>
#include "libmesh/wrapped_function.h"
#include "libmesh/zero_function.h"
#include <libmesh/mesh_function.h>
#include "libmesh/fparser.hh"
#include <libmesh/parsed_function_parameter.h>
using namespace libMesh;

typedef std::map<const Elem *, std::vector<const Elem *>> ElemPatchMap;
typedef std::map<const Elem *, std::vector<const Node *>> ElemNodeMap;
typedef std::map<Node, std::vector<const Node *>> NodePatchMap;
typedef std::map<Node, std::vector<const Elem *>> NodeElemMap;



// template <typename... Args>
// std::vector<std::string> AccumulateStringVector(Args... args) {
// 	std::vector<std::string> result;
// 	auto initList = {args...};
// 	using T = typename decltype(initList)::value_type;
// 	std::vector<T> expanded{initList};
// 	result.resize(expanded.size());
// 	std::transform(expanded.begin(), expanded.end(), result.begin(), [](T value) { return std::to_string(value); });
// 	return result;
// }

template <typename T>
class Range
{
  public:
    class iterator
    {
      public:
        explicit iterator(T val, T stop, T step)
            : m_val(val), m_stop(stop), m_step(step) {}
        iterator &operator++()
        {
            m_val += m_step;
            if ((m_step > 0 && m_val >= m_stop) || (m_step < 0 && m_val <= m_stop))
            {
                m_val = m_stop;
            }
            return *this;
        }
        iterator operator++(int)
        {
            iterator retval = *this;
            ++(*this);
            return retval;
        }
        bool operator==(iterator other) const { return m_val == other.m_val; }
        bool operator!=(iterator other) const { return !(*this == other); }
        T operator*() const { return m_val; }

      private:
        T m_val, m_stop, m_step;
    };

    explicit Range(T stop) : m_start(0), m_stop(stop), m_step(1) {}

    explicit Range(T start, T stop, T step = 1)
        : m_start(start), m_stop(stop), m_step(step) {}

    iterator begin() const { return iterator(m_start, m_stop, m_step); }
    iterator end() const { return iterator(m_stop, m_stop, m_step); }

  private:
    T m_start, m_stop, m_step;
};

template <typename T>
Range<T> range(T stop) { return Range<T>(stop); }

template <typename T>
Range<T> range(T start, T stop, T step = 1)
{
    return Range<T>(start, stop, step);
}

double signed_distance_fcn(const Point &p);

void write_libmesh_info(EquationSystems &equation_systems);

Number exact_solution(const Point &p, const Parameters &parameters,
                      const std::string &, const std::string &);

Number exact_laplacian(const Point &p, const Parameters &parameters,
                       const std::string &, const std::string &);

Number exact_divergence(const Point &p, const Parameters &parameters,
                        const std::string &, const std::string &);

VectorValue<Number> exact_gradient(const Point &p, const Parameters &parameters,
                                   const std::string &, const std::string &);

Number kernel(const Point &p, const Parameters &parameters,
              const std::string &, const std::string &);
Number kernel(const Point &p, const Real &horizon);
Number kernel(const Point &p);

void write_error(std::vector<Number> &simvals,
                 std::vector<Number> &errvals,
                 std::string &savename,
                 std::string results_prefix);



std::unique_ptr<MeshFunction>
define_meshfunction(EquationSystems &equation_systems);

std::unique_ptr<NumericVector<Number>>
integrate_scalar_exact(EquationSystems &equation_systems,
                       FunctionBase<Number> &exact_solution);

void print_quadrule(EquationSystems &equation_systems);

template <typename... Args>
std::vector<std::string> toStringVector(Args... args);


template <typename T, typename... Args>
void push_back_vec(std::vector<T> &v, Args &&... args);


template <typename... Args>
void FoldWrite(
    std::ofstream &outfile,
    std::string &savename,
    Args &&... args);


template <typename... Args>
void FoldWrite(
    std::ofstream &outfile,
    Args &&... args);


template <typename... Args>
void FoldPrint(Args &&... args);

template <typename T, typename... Args>
void FoldPushBack(std::vector<T> &v, Args &&... args);

Number
scalar_boundary_indicator(const Real &x);


Number
boundary_indicator(const Point &p);

Number
exact_solution(const Point &p, const Parameters &parameters,
               const std::string &, const std::string &);

// VectorValue<Number>
Gradient
exact_gradient(const Point &p, const Parameters &parameters, const std::string &, const std::string &);


Number
laplacian(const Point &p, const Parameters &parameters, const std::string &,
          const std::string &);


Number kernel_bspline3(Real x, const Parameters &parameters);

Number kernel_bspline3(Real x);
Number kernel_CubicConvolution3(Real x, const Parameters &parameters);
Number kernel_CubicConvolution3(Real x, const double &a);
Number kernel_CubicConvolution4(Real x);

Number kernel_piecewise_linear(Real x, const Parameters &parameters);














Number kernel_piecewise_linear(Real x);

#endif //HELPERS_H