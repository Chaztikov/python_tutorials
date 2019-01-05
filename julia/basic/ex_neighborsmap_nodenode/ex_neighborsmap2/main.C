#define DISCRETE_MOMENT_CONDITIONS

#include <algorithm>
#include <any>
#include <boost/math/interpolators/cubic_b_spline.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <stdint.h>
#include <string>
#include <sys/time.h>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

// #include <libmesh/tensor_tools.h>
#include "libmesh/string_to_enum.h"
#include <libmesh/libmesh.h>
#include <libmesh/point_locator_tree.h>

#include <libmesh/point_locator_tree.h>
// #include "./exact_solution.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/equation_systems.h"
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

#include "libmesh/string_to_enum.h"
#include <libmesh/point_locator_tree.h>

// #include "./operators.h"
// #include "./exact_solution.h"
// #include ".libs/solution_function.h"

#include "libmesh/fparser.hh"
#include "libmesh/function_base.h"
#include "libmesh/wrapped_function.h"
#include "libmesh/zero_function.h"
#include <libmesh/discontinuity_measure.h>
#include <libmesh/error_vector.h>
#include <libmesh/exact_error_estimator.h>
#include <libmesh/hp_coarsentest.h>
#include <libmesh/hp_selector.h>
#include <libmesh/hp_singular.h>
#include <libmesh/kelly_error_estimator.h>
#include <libmesh/mesh_function.h>
#include <libmesh/parsed_fem_function.h>
#include <libmesh/parsed_function_parameter.h>
#include <libmesh/patch_recovery_error_estimator.h>
#include <libmesh/sibling_coupling.h>
#include <libmesh/uniform_refinement_estimator.h>
using namespace libMesh;

// #include "kernel.h"
// #include "helpers.h"
#include "/home/chaztikov/src/libmesh/libmesh-1.3.1/clang8/include/libmesh/string_to_enum.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/vectormap.h"
#include <libmesh/analytic_function.h>
#include <sstream>
// #include <random>
#include <functional>
#include <iostream>

// #include <execution>
#include </usr/include/leatherman/execution/execution.hpp>

// #include <experimental/execution>
using namespace libMesh;
using namespace std::placeholders; // for _1, _2, _3...

typedef class libMesh::NumericVector<Number> MyInput;

typedef NumericVector<Number> MyOutput;

typedef std::unique_ptr<MyOutput> MyUniqueOutput;

typedef std::unordered_map<const Node *, std::vector<const Node *>>
    type_MapNodeToNodeNeighbor;

template <typename T> class Range {
public:
  class iterator {
  public:
    explicit iterator(T val, T stop, T step)
        : m_val(val), m_stop(stop), m_step(step) {}
    iterator &operator++() {
      m_val += m_step;
      if ((m_step > 0 && m_val >= m_stop) || (m_step < 0 && m_val <= m_stop)) {
        m_val = m_stop;
      }
      return *this;
    }
    iterator operator++(int) {
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

template <typename T> Range<T> range(T stop) { return Range<T>(stop); }

template <typename T> Range<T> range(T start, T stop, T step = 1) {
  return Range<T>(start, stop, step);
}

namespace {
ElemType elem_type;
unsigned int ndimensions;

// const QuadratureType quad_type = QTRAP;
QuadratureType quad_type;
Order quad_order;

Order p_order;
FEFamily fe_family;

double xminmax[] = {-0.5, 0.5};
double xmin = xminmax[0], xmax = xminmax[1];

// double horizon = 0.1;
double horizon_boundary_width;

// n_elem * horizon > 1
double nelem_div_horizon;
int nelem;
// double nelem_double;
double h_elem;
double horizon;

std::unordered_map<const Node *, std::vector<const Elem *>>
    MapNodeToElemNeighbor;

std::vector<std::vector<const Elem *>> nodes_to_elem_vecvec;

std::unordered_map<const Node *, std::vector<const Node *>>
    MapNodeToNodeNeighbor;

unsigned int min_node_neighbor_count = 3;

// static_cast<int>(nelem_div_horizon / horizon);
std::vector<std::unique_ptr<MeshBase>> meshvec;
// std::vector< std::shared_ptr<MeshBase>> meshvec;

} // namespace

// std::tuple<std::vector<std::vector<const Elem *>>, std::vector<const Node *>>
auto compute_nodal_neighbors(const MeshBase &mesh) {
  // find_nodal_neighbors() needs a data structure which is prepared by another
  // function
  std::vector<std::vector<const Elem *>> nodes_to_elem_map;
  MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);

  // Loop over the nodes and call find_nodal_neighbors()

  std::unordered_map<const Node *, std::vector<const Node *>>
      MapNodeToNodeNeighbor;

  for (const auto &node : mesh.node_ptr_range()) {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_map,
                                    neighbor_node);

    MapNodeToNodeNeighbor.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_map, MapNodeToNodeNeighbor);
}

auto map_node_node_neighbor(const MeshBase &mesh) {
  MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_vecvec);

  for (const auto &node : mesh.node_ptr_range()) {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_vecvec,
                                    neighbor_node);

    MapNodeToNodeNeighbor.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_vecvec, MapNodeToNodeNeighbor);
}

void map_node_elem_neighbor(const MeshBase &mesh) {
  std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();

  for (const auto &node : mesh.node_ptr_range()) {

    point_locator->enable_out_of_mesh_mode();

    const Elem *elem = (*point_locator)(*node);

    // MapNodeToElemNeighbor[node].emplace_back(elem);
    for (const auto node_neighbor : MapNodeToNodeNeighbor[node]) {
      // for(const auto node_neighbor : node_neighborhood)
      {
        const Elem *elem_neighbor = (*point_locator)(*node_neighbor);
        MapNodeToElemNeighbor[node].emplace_back(elem_neighbor);
      }
    }
  }
}

Number boundary_indicator(const Real &x, Real tol = 0) {
  return abs(x) + tol > xmax ? 0 : 1;
}

Number boundary_indicator(const Point &p, Real tol = 0) {
  return boundary_indicator(p(0), tol) * boundary_indicator(p(1), tol);
}

Number kernel3(const Real &x);
Number kernel3(const Real &x) {
  Real val = 0;

  Real absx = abs(x / horizon);

  if (absx > 1.)
    return 0;
  else if (absx <= 0.5)
    val += -8. * (1 - absx) * pow(absx, 2) + 4. / 3.;
  else
    val += 8. * pow((1 - absx), 3) / 3.;
  return val * (27. - 120. * pow(absx, 2)) / 17. * boundary_indicator(x) /
         horizon;
}

Number kernel1(const Real &x);
Number kernel1(const Real &x) {
  Real val = 0;

  Real absx = abs(x / horizon);

  if (absx > 1.)
    return 0;
  else
    return (1 - absx) * boundary_indicator(x) / horizon;
}

Number kernel(const Real &x);
Number kernel(const Real &x) {
  // return kernel1(x);
  return kernel3(x);
}

Number kernel(const Point &p) { return kernel(p(0)) * kernel(p(1)); }

Number cosine_solution(Real x);
Number cosine_solution(Real x) {
  return 0.5 * (1. + cos(M_PIl * 2 * x)) * boundary_indicator(x);
}
Number hat_solution(Real x) {
  return 0.5 * (1. - abs(x)) * boundary_indicator(x);
}

Number exact_solution1d(Real x);
Number exact_solution1d(Real x) {
  return  cosine_solution(x);
  // return hat_solution(x);
}

Number exact_solution(const Point &p, const Parameters &parameters,
                      const std::string &, const std::string &);
Number exact_solution(const Point &p, const Parameters &parameters,
                      const std::string &, const std::string &) {
  return exact_solution1d(p(0)) * exact_solution1d(p(1));
}

Number integrate_patch(const Node &node, const Parameters &parameters,
                       const std::string &system, const std::string &var);

Number integrate_patch(const Node &node, const Parameters &parameters,
                       const std::string &system, const std::string &var) {

  Real elemval = 0;

  std::unique_ptr<FEBase> fe(
      FEBase::build(ndimensions, FEType(p_order, fe_family)));
  std::unique_ptr<QBase> qrule(
      QBase::build(quad_type, ndimensions, quad_order));
  fe->attach_quadrature_rule(qrule.get());

  const std::vector<Real> &JxW = fe->get_JxW();
  const std::vector<Point> &qpoint = fe->get_xyz();
  // const std::vector<std::vector<Real>> &phi = fe->get_phi();
  // std::cout << "\n qrule->quad_order() " << qrule->get_order();

  const Point p(node);

  auto patch = MapNodeToElemNeighbor[&node];

  for (const auto &neighbor : patch) {

    // dof_map.dof_indices(elem, dof_indices_neighbor);

    fe->reinit(neighbor);

    for (auto qp : range(qrule->n_points())) {
      const Point &xi((p - qpoint[qp]));
      Real kernelval = kernel(xi);
      elemval += JxW[qp] * kernelval *
                 exact_solution(qpoint[qp], parameters, system, var);
    }
  }

#ifdef DISCRETE_MOMENT_CONDITIONS
  Real kernel_moment_0 = 0;
  Gradient kernel_moment_1(0, 0);
  // Real kernel_moment_1x = 0;
  // Real kernel_moment_1y = 0;
  Real kernel_moment_2 = 0;
  Real kernel_moment_3 = 0;
  // if (fmax(abs(p(0)), fmax(abs(p(1)), abs(p(2)))) < xmax -
  // horizon_boundary_width)
  if (boundary_indicator(p, horizon_boundary_width)) {

    for (const auto &neighbor : patch) {
      // dof_map.dof_indices(elem, dof_indices_neighbor);
      fe->reinit(neighbor);
      for (auto qp : range(qrule->n_points())) {
        // const Point &xi((p - qpoint[qp]));
        // const Point &xi( qpoint[qp] );
        Real kernel_val_qp = kernel(qpoint[qp] - p);
        kernel_moment_0 += kernel_val_qp * JxW[qp];
        for (auto i : range(ndimensions))
          kernel_moment_1(i) += kernel_val_qp * JxW[qp] * p.slice(i);
        // kernel_moment_1(i) += kernel_val_qp * JxW[qp] * qpoint[qp].slice(i);
      }
    }

    for (auto i : range(ndimensions))
      kernel_moment_1(i) -= p.slice(i);

    std::cout << " \n ";
    std::cout << "\n kernel_moment_0 " << kernel_moment_0;
    std::cout << "\n | kernel_moment_1 - p | " << kernel_moment_1;
    std::cout << " \n";
  }

#endif

  return elemval;
}

// Number exact_solution(const Point &p, const Parameters &parameters,
//                       const std::string &, const std::string &);

Real pw_error_linf(EquationSystems &equation_systems,
                   FunctionBase<Number> &exact_solution);

int main(int argc, char **argv) {

  LibMeshInit init(argc, argv);
  GetPot input_file("input.in");
  input_file.parse_command_line(argc, argv);

  nelem_div_horizon = input_file("nelem_div_horizon ", 1.0);

  nelem = input_file("nelem", 32);

  elem_type = static_cast<ElemType>(input_file("elem_type", 5));

  ndimensions = input_file("ndimensions", 2);

  quad_type = static_cast<QuadratureType>(input_file("quad_type", 0));

  quad_order = static_cast<Order>((input_file("quad_order", 2)));

  p_order = static_cast<Order>((input_file("p_order", 1)));

  fe_family = LAGRANGE;
  
  // static_cast<FEFamily>(input_file("fe_family",LAGRANGE));

  // nelem = 32;
  h_elem = (xmax - xmin) * (double)(1. / (static_cast<int>(nelem)));
  horizon = nelem_div_horizon * h_elem;
  horizon_boundary_width = input_file("horizon_boundary_width ", 1.0) * horizon;

  SerialMesh mesh(init.comm(), ndimensions);

  MeshTools::Generation::build_square(mesh, nelem, nelem, xmin, xmax, xmin,
                                      xmax, elem_type);

  mesh.prepare_for_use();

  map_node_node_neighbor(mesh);
  map_node_elem_neighbor(mesh);

  // // Iterate and print keys and values of unordered_map
  // for (auto node : mesh.node_ptr_range()) {
  //   std::cout << "\n Node " << node;

  //   std::cout << "\n nodal neighbors   ";
  //   for (auto val : MapNodeToNodeNeighbor[node]) {
  //     std::cout << val << " ";
  //   }

  //   std::cout << "\n elem neighbors   ";
  //   for (auto val : MapNodeToElemNeighbor[node]) {
  //     std::cout << val << " ";
  //   }

  //   std::cout << "\n";
  // }

  for (auto node : mesh.node_ptr_range()) {
    std::cout << "\n Node " << node;
    std::cout << " elem neighbor count : " << MapNodeToElemNeighbor[node].size()
              << std::endl;
    std::cout << " node neighbor count : " << MapNodeToNodeNeighbor[node].size()
              << std::endl;
  }

  EquationSystems equation_systems(mesh);

  {
    std::string system_name("system");
    std::string system_var(system_name + "var_0");
    equation_systems.add_system<ExplicitSystem>(system_name);
    equation_systems.get_system(system_name)
        .add_variable(system_var, FEType(p_order, fe_family));
  }

  {
    std::string system_name("output");
    std::string system_var(system_name + "var_0");
    equation_systems.add_system<ExplicitSystem>(system_name);
    equation_systems.get_system(system_name)
        .add_variable(system_var, FEType(p_order, fe_family));
  }

  {
    std::string system_name("error");
    std::string system_var(system_name + "var_0");
    equation_systems.add_system<ExplicitSystem>(system_name);
    equation_systems.get_system(system_name)
        .add_variable(system_var, FEType(p_order, fe_family));
  }

  auto &input_system = equation_systems.get_system("system");
  auto &output_system = equation_systems.get_system("output");

  equation_systems.init();

  WrappedFunction<Number> exact_function(input_system, *exact_solution,
                                         &equation_systems.parameters, 0);

  input_system.project_solution(*exact_solution, libmesh_nullptr,
                                equation_systems.parameters);

  equation_systems.update();
  Real errlinf = pw_error_linf(equation_systems, exact_function);
  libMesh::out.flush();
  std::cout << "\n \n errlinf " << errlinf << std::endl;
  libMesh::out.flush();
  // output_system.solution->add(-1, *temp_integrate_scalar_strong);

  libMesh::out.flush();
  for (auto node : mesh.node_ptr_range()) {
    // std::cout << "\n dof_number(0, 0, 0) " << node->dof_number(0, 0, 0);
    //  std::cout << "\n ";
    unsigned int node_neighbor_count = MapNodeToNodeNeighbor[node].size();
    if (MapNodeToNodeNeighbor[node].size() > min_node_neighbor_count) {
      // std::cout << " \n count " << node_neighbor_count;
      Real val = integrate_patch(*node, equation_systems.parameters, "", "");
      auto i = node->dof_number(0, 0, 0);
      output_system.solution->add(i, val);
    }
  }
  libMesh::out.flush();

  equation_systems.update();

  ExodusII_IO(mesh).write_discontinuous_exodusII("equation_system_out.e",
                                                 equation_systems);

  return 0;

  // output_approx = approx;

  // output_system.solution->add(output_approx);
  // output_system.project_solution(integrate_patch, libmesh_nullptr,
  // equation_systems.parameters);
}

Real pw_error_linf(EquationSystems &equation_systems,
                   FunctionBase<Number> &exact_solution) {
  Real error_linf = 0;
  const MeshBase &mesh = equation_systems.get_mesh();
  auto &system = equation_systems.get_system("error");



  for (const auto p : mesh.node_ptr_range()) {
    unsigned int node_neighbor_count = MapNodeToNodeNeighbor[p].size();
    if (MapNodeToNodeNeighbor[p].size() > min_node_neighbor_count) {
      if (boundary_indicator(*p, horizon_boundary_width)) {
      std::cout << " \n count " << node_neighbor_count;

        Real approx = 0;
        Real exact = 0;
        approx += integrate_patch(*p, equation_systems.parameters, "", "");
        exact += exact_solution(*p); //, equation_systems.parameters, "" , "");

        Real val = abs(exact - approx);

      auto i = p->dof_number(0, 0, 0);
      system.solution->add(i, val);


        error_linf = std::max(error_linf, val);
        
      }
    }
  }
  system.solution->close();
  return error_linf;
}
