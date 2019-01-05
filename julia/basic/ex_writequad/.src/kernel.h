#ifndef KERNEL_H
#define KERNEL_H
#include "helpers.h"

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

#include <boost/math/special_functions/binomial.hpp>

#include <libmesh/tensor_tools.h>
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
#include <boost/math/interpolators/cubic_b_spline.hpp>

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

Number kernel_bspline3(Real x, const Parameters &parameters);
Number kernel_bspline3(Real x, const Parameters &parameters)
{
    Number val = 0;
    if (abs(x) < 0.5)
        val += 1. / 6. - pow(x, 2) + pow(abs(x), 3);
    else if (abs(x) < 1.)
        val += -(1. / 3.) * (-1 + pow(abs(x), 3));
    else
        0;

    return val;
}

Number kernel_bspline3(Real x);
Number kernel_bspline3(Real x)
{
    Number val = 0;
    if (abs(x) < 0.5)
        val += 1. / 6. - pow(x, 2) + pow(abs(x), 3);
    else if (abs(x) < 1.)
        val += -(1. / 3.) * (-1 + pow(abs(x), 3));
    else
        0;

    return val;
}


Number kernel_CubicConvolution3(Real x, const Parameters &parameters);
Number kernel_CubicConvolution3(Real x, const Parameters &parameters)
{
    Number val = 0;
    double a = -0.5;
    // a += parameters.get([0];

    if (abs(x) <= 0.5)
        val += 1 - 4 * (3. + a) * std::pow(abs(x), 2) + 8 * (2. + a) * std::pow(abs(x), 3);
    else if (abs(x) <= 1.)
        val += 4 * a * std::pow((-1 + abs(x)), 2) * (-1 + 2 * abs(x));
    else
        0;
    return val;
}

Number kernel_CubicConvolution3(Real x, const double &a);
Number kernel_CubicConvolution3(Real x, const double &a = -0.5)
{
    Number val = 0;
    // double a = -0.5;

    if (abs(x) <= 0.5)
        val += 1 - 4 * (3. + a) * std::pow(abs(x), 2) + 8 * (2. + a) * std::pow(abs(x), 3);
    else if (abs(x) <= 1.)
        val += 4 * a * std::pow((-1 + abs(x)), 2) * (-1 + 2 * abs(x));
    else
        0;
    return val;
}

Number kernel_CubicConvolution4(Real x);
Number kernel_CubicConvolution4(Real x) {
  Real x3 = 0.5 + TOLERANCE;
  Real x2 = 1./3. + TOLERANCE;
  Real x1 = 1./6. + TOLERANCE;

  Real ax = abs(x);
  Eigen::Vector4d axvec;
  axvec << std::pow(ax, 3), ax * ax, ax, 1.;
  if (ax <= x3) {
    Eigen::Vector4d coeff = {4. / 3., -7. / 3., 0., 1.};
    return coeff.dot(axvec) * 1;
  }

  if (ax <= x2) {
    Eigen::Vector4d coeff = {-7. / 12., 3., -59. / 12., 5. / 2.};
    return coeff.dot(axvec) * 1;
  }
  if (ax <= x1) {
    Eigen::Vector4d coeff = {1. / 12., -2. / 3., 7. / 4., -3. / 2.};
    return coeff.dot(axvec) * 1;
  } else {
    return 0;
  }
}

Number kernel_piecewise_linear(Real x, const Parameters &parameters);
Number kernel_piecewise_linear(Real x, const Parameters &parameters)
{
    Number val = 0;

    if (abs(x) > 1)
        1;
    else
        val += 1 - abs(x);

    return val;
}



Number kernel_piecewise_linear(Real x);
Number kernel_piecewise_linear(Real x)
{
    Number val = 0;

    if (abs(x) > 1)
        1;
    else
        val += 1 - abs(x);

    return val;
}

Number kernel(const Point &p, const Parameters &parameters,
              const std::string &, const std::string &)
{
    auto horizon = parameters.get<Real>("horizon");
    auto ndimensions = parameters.get<unsigned int>("ndimensions");

    Real kernelval = 0;

    // Real x = p(0) / horizon, y = p(1) / horizon, z = p(2) / horizon;
    Real x = p(0), y = p(1), z = p(2);

    // kernelval += kernel_piecewise_linear(x, parameters) * kernel_piecewise_linear(y, parameters);
    kernelval += kernel_CubicConvolution3(x, parameters) * kernel_CubicConvolution3(y, parameters);

    // return kernelval;
    return kernelval / (horizon * horizon);
    // return kernelval * (horizon * horizon);
}

Number kernel(const Point &p, const Real &horizon)
{
    Real kernelval = 0;

    // Real x = p(0) / horizon, y = p(1) / horizon, z = p(2) / horizon;
    Real x = p(0), y = p(1), z = p(2);

    // kernelval += kernel_piecewise_linear(x, horizon) * kernel_piecewise_linear(y, horizon);
    kernelval += kernel_CubicConvolution3(x, -0.5) * kernel_CubicConvolution3(y, -0.5);

    // return kernelval;
    return kernelval / (horizon * horizon);
    // return kernelval * (horizon * horizon);
}

// Number kernel(const Point &p);
Number kernel(const Point &p)
{
    Real x = p(0), y = p(1), z = p(2);
    // return kernel_bspline3(x) * kernel_bspline3(y);
    // return kernel_piecewise_linear(x) * kernel_piecewise_linear(y);
    // return kernel_CubicConvolution3(x, -0.5) * kernel_CubicConvolution3(y, -0.5);
    return kernel_CubicConvolution4(x) * kernel_CubicConvolution4(y);
    // return kernelval;
}


#endif //KERNEL_H