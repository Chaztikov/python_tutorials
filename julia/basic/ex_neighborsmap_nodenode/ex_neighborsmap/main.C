#define DISCRETE_MOMENT_CONDITIONS

// #include "kernel.h"
#include "helpers.h"
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

namespace
{
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

std::unordered_map<const Node *, std::vector<const Elem *>> MapNodeToElemNeighbor;
std::vector<std::vector<const Elem *>> nodes_to_elem_vecvec;
std::unordered_map<const Node *, std::vector<const Node *>> MapNodeToNodeNeighbor;

// static_cast<int>(nelem_div_horizon / horizon);
std::vector<std::unique_ptr<MeshBase>> meshvec;
// std::vector< std::shared_ptr<MeshBase>> meshvec;

} // namespace

// std::tuple<std::vector<std::vector<const Elem *>>, std::vector<const Node *>>
auto compute_nodal_neighbors(const MeshBase &mesh)
{
  // find_nodal_neighbors() needs a data structure which is prepared by another
  // function
  std::vector<std::vector<const Elem *>> nodes_to_elem_map;
  MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);

  // Loop over the nodes and call find_nodal_neighbors()

  std::unordered_map<const Node *, std::vector<const Node *>> MapNodeToNodeNeighbor;

  for (const auto &node : mesh.node_ptr_range())
  {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_map,
                                    neighbor_node);

    MapNodeToNodeNeighbor.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_map, MapNodeToNodeNeighbor);
}

auto map_node_node_neighbor(const MeshBase &mesh)
{
  MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_vecvec);

  for (const auto &node : mesh.node_ptr_range())
  {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_vecvec,
                                    neighbor_node);

    MapNodeToNodeNeighbor.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_vecvec, MapNodeToNodeNeighbor);
}

void map_node_elem_neighbor(const MeshBase &mesh)
{
  std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();

  for (const auto &node : mesh.node_ptr_range())
  {

    point_locator->enable_out_of_mesh_mode();

    const Elem *elem = (*point_locator)(*node);

    for (const auto node_neighbor : MapNodeToNodeNeighbor[node])
    {
      // for(const auto node_neighbor : node_neighborhood)
      {
        const Elem *elem_neighbor = (*point_locator)(*node_neighbor);
        MapNodeToElemNeighbor[node].emplace_back(elem_neighbor);
      }
    }
  }
}

Number kernel(const Real &x);
// Number kernel(const Real &x)
// {
//   if (x < 1.)
//     return 1 - abs(x);
//   else
//     return 0;
// }

Number kernel(const Real &x)
{
    Real val=0;
    Real absx = abs(x);
    if (absx >1)
        return 0;
    else if(absx <= 0.5)
        val+=-8.*(1-absx) * pow(x,2) + 4./ 3.;
    else if ( 0.5 <= absx <= 1.)
        val+=8. * pow((1-absx) , 3 ) / 3.;
    return val * (27.-120. * pow(x,2))/17;
}


Number kernel(const Point &p);
Number kernel(const Point &p)
{
  return kernel(p(0)) * kernel(p(1));
}

Number integrate_patch(const Node &node, const Parameters &parameters,
                       const std::string &system, const std::string &var);

Number integrate_patch(const Node &node, const Parameters &parameters,
                       const std::string &system, const std::string &var)
{

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

  for (const auto &neighbor : patch)
  {

    // dof_map.dof_indices(elem, dof_indices_neighbor);

    fe->reinit(neighbor);

    for (auto qp : range(qrule->n_points()))
    {
      const Point &xi((p - qpoint[qp]));
      Real kernelval = kernel(xi / horizon) / (horizon * horizon);
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
  if (fmax(abs(p(0)), fmax(abs(p(1)), abs(p(2)))) < xmax - horizon_boundary_width)
  {

    for (const auto &neighbor : patch)
    {

      // dof_map.dof_indices(elem, dof_indices_neighbor);

      fe->reinit(neighbor);
      for (auto qp : range(qrule->n_points()))
      {
        const Point &xi((p - qpoint[qp]));
        Real kernel_val_qp = kernel(xi / horizon) / (horizon * horizon);
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

Number exact_solution(const Point &p, const Parameters &parameters,
                      const std::string &, const std::string &);

Number integrate_patch(const Point &p, const Parameters &parameters,
                       const std::string &system, const std::string &var, const MeshBase &mesh);

Number integrate_patch(const Point &p, const Parameters &parameters,
                       const std::string &system, const std::string &var);

Real pw_error_linf(EquationSystems &equation_systems,
                   FunctionBase<Number> &exact_solution);

int main(int argc, char **argv)
{

  LibMeshInit init(argc, argv);
  GetPot input_file("input.in");
  input_file.parse_command_line(argc, argv);

  nelem_div_horizon = input_file("nelem_div_horizon ", 1.0);
  nelem = input_file("nelem", 32);
  elem_type = QUAD9;
  // static_cast<ElemType>(input_file("elem_type",QUAD4));
  ndimensions = input_file("ndimensions", 2);
  quad_type = QTRAP;
  // static_cast<QuadratureType>(input_file("quad_type",QGAUSS));
  quad_order = FIFTH;
  // static_cast<Order>(std::to_string(input_file("quad_order",FIFTH)));
  p_order = FIRST;
  // static_cast<Order>(std::to_string(input_file("p_order",FIRST)));
  fe_family = LAGRANGE;
  // static_cast<FEFamily>(input_file("fe_family",LAGRANGE));

  // nelem = 32;
  h_elem = (xmax - xmin) * (double)(1. / static_cast<int>(nelem));
  horizon = nelem_div_horizon * h_elem;
  horizon_boundary_width = input_file("horizon_boundary_width ", 1.0) * horizon;

  SerialMesh mesh(init.comm(), ndimensions);

  MeshTools::Generation::build_square(mesh, nelem, nelem, xmin, xmax, xmin,
                                      xmax, elem_type);

  mesh.prepare_for_use();

  map_node_node_neighbor(mesh);
  map_node_elem_neighbor(mesh);

  // Iterate and print keys and values of unordered_map
  for (auto node : mesh.node_ptr_range())
  {
    std::cout << "\n Node " << node;

    std::cout << "\n nodal neighbors   ";
    for (auto val : MapNodeToNodeNeighbor[node])
    {
      std::cout << val << " ";
    }

    std::cout << "\n elem neighbors   ";
    for (auto val : MapNodeToElemNeighbor[node])
    {
      std::cout << val << " ";
    }

    std::cout << "\n";
  }
}
