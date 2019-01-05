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
typedef class libMesh::NumericVector<Number> MyInput;

typedef NumericVector<Number> MyOutput;

typedef std::unique_ptr<MyOutput> MyUniqueOutput;

typedef std::unordered_map<const Node *, std::vector<const Node *>>
    type_neighbor_node_map;

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
double nelem_double;
double h_elem;
double horizon;

std::unordered_map<const Node*,const Elem*> MapNodeToElemNeighbor;
std::vector<std::vector<const Elem *>> nodes_to_elem_map;
std::unordered_map<const Node *, std::vector<const Node *>> neighbor_node_map;

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

  std::unordered_map<const Node *, std::vector<const Node *>> neighbor_node_map;

  for (const auto &node : mesh.node_ptr_range())
  {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_map,
                                    neighbor_node);

    neighbor_node_map.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_map, neighbor_node_map);
}

auto map_node_elem_neighbor(const MeshBase &mesh)
{
  // find_nodal_neighbors() needs a data structure which is prepared by another
  // function
  MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);

  // Loop over the nodes and call find_nodal_neighbors()


  for (const auto &node : mesh.node_ptr_range())
  {
    std::vector<const Node *> neighbor_node;

    MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_map,
                                    neighbor_node);

    neighbor_node_map.emplace(node, neighbor_node);
  }

  return std::make_tuple(nodes_to_elem_map, neighbor_node_map);
}

using namespace std::placeholders; // for _1, _2, _3...

// Number Point::Max(const Point &p);
// Number Point::Max(const Point &p)
// {
//   return (fmax(abs(p(0)), fmax(abs(p(1)), abs(p(2)))));
// }

Number kernel(const Point &p)
{
  Real x = p(0), y = p(1), z = p(2);
  // return kernel_bspline3(x) * kernel_bspline3(y);

  // std::string enum_elem_type = Utility::enum_to_string(elem_type);
  // enum_elem_type == "QUAD4";
  // compare(QUAD4 );
  // std::cout << (enum_elem_type == "QUAD4");
  //  << std::endl;
  // if( (enum_elem_type == "QUAD4") ):

  // return kernel_piecewise_linear(x) * kernel_piecewise_linear(y);
  // return kernel_CubicConvolution3(x, -0.5) * kernel_CubicConvolution3(y, -0.5);
  return kernel_CubicConvolution4(x) * kernel_CubicConvolution4(y);
  // return kernelval;
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

  map_node_elem_neighbor(mesh);

  std::string system_name("system");
  std::string system_var(system_name + "var_0");

  /*MESH BOUNDARY*/
  std::set<boundary_id_type> bcids;
  bcids.insert(0);
  bcids.insert(1);
  bcids.insert(2);
  bcids.insert(3);

  //     /* SET EQUATION_SYSTEMS */
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

  auto &input_system = equation_systems.get_system("system");
  auto &output_system = equation_systems.get_system("output");

  equation_systems.init();

  WrappedFunction<Number> exact_function(input_system, *exact_solution,
                                         &equation_systems.parameters, 0);

  input_system.project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);

  // output_system.

  Real errlinf = pw_error_linf(equation_systems, exact_function);
  std::cout << " errlinf " << errlinf << std::endl;

  // output_system.solution->add(-1, *temp_integrate_scalar_strong);

  equation_systems.update();

  ExodusII_IO(mesh).write_discontinuous_exodusII("equation_system_out.e",
                                                 equation_systems);

  // WrappedFunction<Number> approx_function(input_system, *integrate_patch,
  //                                         &equation_systems.parameters, 0);

  // auto integrate_patch_bound = std::bind(integrate_patch, _1, _2, _3, _4, mesh);

  // output_system.project_solution( &(*integrate_patch)(const Point &p, const Parameters &parameters,
  //                      const std::string &system, const std::string &var, std::bind(mesh)),
  //                      libmesh_nullptr,
  //                       equation_systems.parameters);

  return 0;
}

/* 
void func ( void (*f)(int) )
void func ( void (*f)(int) ) {
  for ( int ctr = 0 ; ctr < 5 ; ctr++ ) {
    (*f)(ctr);
  }
} */

// Real pw_error_linf(EquationSystems &equation_systems
//                        Number (*exact_solution),
//                        Number (*integrate_patch))

// template <class T>
// T GetMax (T a, T b) {
//   T result;
//   result = (a>b)? a : b;
//   return (result);
// }

Real pw_error_linf(EquationSystems &equation_systems,
                   FunctionBase<Number> &exact_solution)
{
  Real error_linf = 0;
  const MeshBase &mesh = equation_systems.get_mesh();

  for (const auto p : mesh.node_ptr_range())
  {
    // if (fmax(abs(p(0)), fmax(abs(p(1)), abs(p(2)))) < xmax - 2 * horizon)
    if ((p->norm()) < xmax - 2 * horizon)
    {
      Real approx = 0;
      Real exact = 0;
      approx += integrate_patch(*p, equation_systems.parameters, "", "", mesh);
      exact += exact_solution(*p); //, equation_systems.parameters, "" , "");

      error_linf = std::max(error_linf, abs(exact - approx));
    }
  }
  return error_linf;
}

// Number integrate_patch(const Point &p, const Parameters &parameters,
//                        const std::string &system, const std::string &var)
// {
//   return integrate_patch(p,parameters,system,var,mesh)
// }

Number integrate_patch(const Point &p, const Parameters &parameters,
                       const std::string &system, const std::string &var,
                       const MeshBase &mesh)
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

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();
  point_locator->enable_out_of_mesh_mode();
  const Elem *elem = (*point_locator)(p);

  // MeshTools::find_nodal_neighbors(mesh, Node(p), nodes_to_elem_map,
  // neighbor_nodes); std::vector<dof_id_type>
  // neighbor_node_ids(neighbor_nodes.size());

  libMesh::Patch patch(mesh.processor_id());
  patch.build_around_element(elem,
                             std::pow(1 + 2 * nelem_div_horizon, ndimensions),
                             &Patch::add_point_neighbors);

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

Number kernel(const Point &p);

Number cosine_solution(Real x)
{
  Real val = 0;
  val += 0.5 * (1. + cos(M_PIl * 2 * x));
  return val * boundary_indicator(x);
}

template <typename... Args>
std::vector<std::string> AccumulateStringVector(Args... args)
{

  std::vector<std::string> result;

  auto initList = {args...};

  using T = typename decltype(initList)::value_type;

  std::vector<T> expanded{initList};

  result.resize(expanded.size());

  std::transform(expanded.begin(), expanded.end(), result.begin(),
                 [](T value) { return std::to_string(value); });
  return result;
}

// void write_libmesh_info(EquationSystems &equation_systems)
// {

//   const MeshBase &mesh = equation_systems.get_mesh();
//   const unsigned int ndimensions = mesh.mesh_dimension();
//   ExplicitSystem &system =
//       equation_systems.get_system<ExplicitSystem>("system");
//   const DofMap &dof_map = system.get_dof_map();
//   // auto n_dofs_total = dof_map.n_dofs();
//   auto nelem = mesh.n_active_elem();
//   // AnalyticFunction<Number> exact_solution_object(exact_solution);

//   // FEType fe_type = system.variable_type(0);

//   // auto& exact = system.get_vector("exact");
//   // auto& approx = system.get_vector("approx");
//   // auto& error = system.get_vector("error");

//   std::unique_ptr<NumericVector<Number>> centroid_values =
//       system.solution->zero_clone();

//   std::unique_ptr<FEBase> fe(
//       FEBase::build(ndimensions, FEType(p_order, fe_family)));
//   std::unique_ptr<QBase> qrule(QBase::build(QGAUSS, ndimensions, quad_order));
//   fe->attach_quadrature_rule(qrule.get());

//   const std::vector<Real> &JxW = fe->get_JxW();
//   const std::vector<Point> &qpoint = fe->get_xyz();
//   // const std::vector<std::vector<Real>> &phi = fe->get_phi();
//   std::cout << "\n qrule->quad_order() " << qrule->get_order();

//   // std::unique_ptr<FEBase> fe_face(
//   //     FEBase::build(ndimensions, FEType(p_order, fe_family)));
//   // std::unique_ptr<QBase> qface(
//   //     QBase::build(QTRAP, ndimensions - 1, quad_order));
//   // fe_face->attach_quadrature_rule(qface.get());

//   // const std::vector<Real> &JxW_face = fe_face->get_JxW();
//   // const std::vector<Point> &qpoint_face = fe_face->get_xyz();
//   // const std::vector<std::vector<Real>> &phi_face = fe_face->get_phi();
//   // std::cout << "\n qrule->quad_order() " << qface->get_order();

//   std::unique_ptr<NumericVector<Number>> scalarfield_elem =
//       NumericVector<Number>::build(mesh.comm());
//   scalarfield_elem->init(nelem, false, SERIAL);

//   std::unique_ptr<NumericVector<Number>> exact_scalarfield_elem =
//       scalarfield_elem->zero_clone();

//   std::unique_ptr<NumericVector<Number>> scalarfield_node =
//       system.solution->zero_clone();

//   // std::vector<Number> scalarfield_elem(mesh.n_elem());
//   // std::vector<Number> exact_scalarfield_elem(mesh.n_elem());

//   // auto elemval = scalarfield_elem.begin();
//   // auto elemval_exact = exact_scalarfield_elem.begin();

//   std::stringstream sstm;
//   // sstm << "_" << nelem << horizon;
//   std::string savename = sstm.str();

//   std::ofstream outfile_elem;
//   outfile_elem.open(savename + "elem_values.e", std::ios_base::app);

//   std::ofstream outfile_qtrapside;
//   outfile_qtrapside.open(savename + "qtrap_side.e", std::ios_base::app);

//   std::ofstream outfile_qgausselem;
//   outfile_qgausselem.open(savename + "qgauss_elem.e", std::ios_base::app);

//   DenseMatrix<Number> Ke;
//   DenseVector<Number> Fe;
//   // DenseVector<Number> Fexact;
//   std::vector<dof_id_type> dof_indices;
//   std::vector<dof_id_type> one_dof_index(1);

//   // for (auto elem : mesh.element_ptr_range())
//   for (auto elem : mesh.element_ptr_range())
//   {
//     // Real elemval_exact = 0;
//     // Real elemval = 0;

//     dof_map.dof_indices(elem, dof_indices);
//     const unsigned int n_dofs = dof_indices.size();

//     Fe.zero();
//     Fe.resize(n_dofs);

//     libMesh::Patch patch(mesh.processor_id());
//     patch.build_around_element(elem,
//                                std::pow(1 + 2 * nelem_div_horizon, ndimensions),
//                                &Patch::add_point_neighbors);

//     const Point &point_elem = elem->centroid();

//     std::ofstream outfile;
//     outfile.open(savename + "patch.e", std::ios_base::app);

//     for (const auto &neighbor : patch)
//     {

//       fe->reinit(neighbor);

//       // qpoints
//       for (auto qp : range(qrule->n_points()))
//         for (auto i : range(3))
//           outfile << qpoint[qp](i) << " ";

//       // JxW
//       for (auto val : JxW)
//         outfile << val << " ";

//       // xi
//       for (auto xprime : qpoint)
//         outfile << (point_elem - xprime) << " ";

//       // kernel
//       for (auto xprime : qpoint)
//         outfile << kernel(point_elem - xprime) << " ";
//     }

//     outfile << " \n ";
//   }
// }

MyUniqueOutput integrate_exact(EquationSystems &equation_systems)
{
  const MeshBase &mesh = equation_systems.get_mesh();
  auto nelem = mesh.n_active_elem();
  const unsigned int ndimensions = mesh.mesh_dimension();

  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");

  const DofMap &dof_map = system.get_dof_map();
  // auto n_dofs_total = dof_map.n_dofs();

  double horizon = equation_systems.parameters.get<double>("horizon");

  // FEType fe_type = system.variable_type(0);
  std::unique_ptr<NumericVector<Number>> centroid_values =
      system.solution->zero_clone();

  std::unique_ptr<NumericVector<Number>> scalarfield_elem =
      NumericVector<Number>::build(mesh.comm());
  scalarfield_elem->init(nelem, false, SERIAL);

  std::unique_ptr<NumericVector<Number>> exact_scalarfield_elem =
      scalarfield_elem->zero_clone();

  std::unique_ptr<NumericVector<Number>> scalarfield_node =
      system.solution->zero_clone();

  std::stringstream sstm;
  // sstm << "_" << nelem << horizon;
  std::string savename = sstm.str();

  std::ofstream outfile_elem;
  outfile_elem.open(savename + "elem_values.e", std::ios_base::app);

  // auto& exact = system.get_vector("exact");
  // auto& approx = system.get_vector("approx");
  // auto& error = system.get_vector("error");

  std::unique_ptr<FEBase> fe(
      FEBase::build(ndimensions, FEType(p_order, fe_family)));
  std::unique_ptr<QBase> qrule(
      QBase::build(quad_type, ndimensions, quad_order));
  fe->attach_quadrature_rule(qrule.get());

  const std::vector<Real> &JxW = fe->get_JxW();
  const std::vector<Point> &qpoint = fe->get_xyz();
  // const std::vector<std::vector<Real>> &phi = fe->get_phi();
  std::cout << "\n qrule->quad_order() " << qrule->get_order();

  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions, FEType(p_order,
  // fe_family))); std::unique_ptr<QBase> qface(QBase::build(QTRAP, ndimensions
  // - 1, quad_order)); fe_face->attach_quadrature_rule(qface.get());

  // const std::vector<Real> &JxW_face = fe_face->get_JxW();
  // const std::vector<Point> &qpoint_face = fe_face->get_xyz();
  // const std::vector<std::vector<Real>> &phi_face = fe_face->get_phi();
  // std::cout << "\n qrule->quad_order() " << qface->get_order();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;
  // DenseVector<Number> Fexact;
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_neighbor;
  std::vector<dof_id_type> one_dof_index(1);

  // for (auto elem : mesh.element_ptr_range())
  for (auto elem : mesh.element_ptr_range())
  {
    // Real elemval_exact = 0;
    Real elemval = 0;

    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();

    Fe.zero();
    Fe.resize(n_dofs);

    libMesh::Patch patch(mesh.processor_id());
    patch.build_around_element(elem,
                               std::pow(1 + 2 * nelem_div_horizon, ndimensions),
                               &Patch::add_point_neighbors);

    const Point &point_elem = elem->centroid();
    Real JxWkernelsum = 0;
    for (const auto &neighbor : patch)
    {

      // dof_map.dof_indices(elem, dof_indices_neighbor);

      fe->reinit(neighbor);

      for (auto qp : range(qrule->n_points()))
      {
        const Point &xi((point_elem - qpoint[qp]));

        Real kernelval = (kernel(xi * horizon)) / (horizon * horizon);
        // kernelval += 1;
        JxWkernelsum += JxW[qp] * kernelval;

        elemval += JxW[qp] * kernelval *
                   exact_solution(qpoint[qp], equation_systems.parameters,
                                  std::string(), std::string());
      }
    }
  }

  return scalarfield_node;
}

libMesh::Patch build_neighborhood(const MeshBase &mesh, const Elem &base_elem,
                                  const unsigned int target_patch_size)
{

  libMesh::Patch patch_elems(mesh.processor_id());
  patch_elems.build_around_element(&base_elem, target_patch_size,
                                   &Patch::add_point_neighbors);
  return patch_elems;
}

// using namespace clt;

// C++ template to print vector container elements
/* template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) 
{
  for (auto i != v.size()) 
    os << v[i];
  
  return os;
}
 */

// typedef MyOutput = std::unique_ptr<NumericVector<Number>>;

// template <typename T1 , typename T2>
// std::tuple<T2,T2> = integrate_input(EquationSystems &equation_systems, T1
// &input);
/*
template <typename T>
std::tuple<T,T> = integrate_input(EquationSystems &equation_systems, T &input);

MyInput
integrate_input(EquationSystems &equation_systems, MyInput &input)
{


return std::make_tuple<input,input>
}
 */

// MyUniqueOutput
// integrate_exact(EquationSystems &equation_systems);

// const MyOutput&
// MyUniqueOutput

// void locate(const Point &p, const ElemType elem_type)
// {
//     std::unique_ptr<PointLocatorBase> locator = mesh.sub_point_locator();
//     locator->enable_out_of_mesh_mode();
//     const Elem *elem = locator->operator()(p);
//     const Node *node = locator->locate_node(p);
// }

/*
std::transform example

vector<int> numbers1 = {1, 5, 42, 7, 8};
vector<int> numbers2 = {10, 7, 4, 2, 2};
vector<int> results;
std::transform(numbers1.begin(), numbers1.end(),
               numbers2.begin(),
               std::back_inserter(results),
               [](int i, int j) {return i+j;});

 */

// demonstrates argument reordering and pass-by-reference
// (_1 and _2 are from std::placeholders, and represent future
// arguments that will be passed to f1)

// Number test_bind(const Point &p, const Parameters &parameters) {
//   p.print();
// return 0;
// }
