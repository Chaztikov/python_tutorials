// #include "kernel.h"
#include "helpers.h"
#include <sstream>
#include <libmesh/analytic_function.h>
#include "libmesh/vectormap.h"
#include "libmesh/mesh_tools.h"

using namespace libMesh;

typedef class libMesh::NumericVector<Number>
    MyInput;

typedef NumericVector<Number>
    MyOutput;

typedef std::unique_ptr<MyOutput>
    MyUniqueOutput;

typedef std::unordered_map<const Node *, std::vector<const Node *>>
type_neighbor_node_map;

namespace
{
const ElemType elem_type = QUAD9;
const unsigned int ndimensions = 2;

const QuadratureType quad_type = QTRAP;

const Order quad_order = FIFTH;

const Order p_order = FIRST;
const FEFamily fe_family = LAGRANGE;

double xminmax[] = {-0.5, 0.5};
double xmin = xminmax[0], xmax = xminmax[1];

const double horizon = 0.1;

// n_elem * horizon > 1
const double nelem_div_horizon = 1.0;

int nelem1 = static_cast<int>(nelem_div_horizon / horizon);

// MeshBase &mesh;

// libMesh::Communicator comm(MPI_COMM_WORD):
// Mesh mesh(comm);
// Mesh mesh();

// (MPI_COMM_WORLD);
// std::unique_ptr<Mesh> all_boundary_mesh;
// std::unique_ptr<Mesh> left_boundary_mesh;
// std::unique_ptr<Mesh> internal_boundary_mesh;

std::vector<std::unique_ptr<MeshBase>> meshvec;
// std::vector< std::shared_ptr<MeshBase>> meshvec;

} // namespace

// std::tuple<std::vector<std::vector<const Elem *>>, std::vector<const Node *>>
auto compute_nodal_neighbors(const MeshBase &mesh)
{
    // find_nodal_neighbors() needs a data structure which is prepared by another function
    std::vector<std::vector<const Elem *>> nodes_to_elem_map;
    MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);

    // Loop over the nodes and call find_nodal_neighbors()

    std::unordered_map<const Node *, std::vector<const Node *>> neighbor_node_map;

    for (const auto &node : mesh.node_ptr_range())
    {
        std::vector<const Node *> neighbor_node;

        MeshTools::find_nodal_neighbors(mesh, *node, nodes_to_elem_map, neighbor_node);

        neighbor_node_map.emplace(node, neighbor_node);
    }

    return std::tie(nodes_to_elem_map, neighbor_node_map);
}




    // for (const auto &node : mesh.node_ptr_range())
    // {
    //     std::cout << "node " << node 
    //     <<"valence " << node->valence()<<std::endl;
    // }
    

    // libMesh::out << " ";
    // libMesh::out.flush();

    // std::cout << nodes_to_elem_map.size() << std::endl;
    // for (auto v1 : nodes_to_elem_map)
    // {
    //     std::cout << "\n node size "
    //               << " " << v1.size();
    //     ;
    //     for (auto v2 : v1)
    //         std::cout << " elem id " << v2->id() << std::endl;
    // }
// std::find

    // return neighbor_node_map;
    // return std::make_tuple<nodes_to_elem_map , neighbor_nodes>;

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

Number integrate_patch(const Point &p, const Parameters &parameters,
                       const std::string &system, const std::string &var);

int main(int argc, char **argv)
{

    // LibMeshInit init(argc, argv);
    // GetPot input_file("input.in");
    // input_file.parse_command_line(argc, argv);

    LibMeshInit init(argc, argv);

    SerialMesh mesh(init.comm(), ndimensions);
    // for (const auto &node : mesh.node_ptr_range())
    // {

    // meshvec.push_back(mesh);
    // meshvec.resize(0);
    // // const std::unique_ptr<Parameters> parameters;// = Parameters(init.comm());
    // // parameters->set<Mesh>("mesh") = mesh.clone();

    // const std::unique_ptr<Elem> test_elem = Elem::build(elem_type);

    // std::vector<double> nelems;
    // std::vector<double> linf_errors;
    // std::vector<double> l1_errors;
    // std::vector<double> l2_errors;

    // std::vector<double> nelemsvec = {10, 20, 40, 80};
    // for (unsigned int nelem1 : nelemsvec)
    // {

    MeshTools::Generation::build_square(mesh, nelem1, nelem1, xmin, xmax, xmin, xmax, elem_type);

    mesh.prepare_for_use();

    // std::vector<std::vector<const Elem *>> nodes_to_elem_map;
    // std::vector<const Node *> neighbor_nodes;
    // auto [nodes_to_elem_map , neighbor_nodes] = compute_nodal_neighbors(mesh);

    // std::vector<std::vector<const Elem *>>> nodes_to_elem_map;

    // std::vector<std::vector<const Elem *>> nodes_to_elem_map;
    // std::unordered_map<const Node *, std::vector<const Node *>> neighbor_node_map;
    auto [nodes_to_elem_map, neighbor_node_map] = compute_nodal_neighbors(mesh);
    // auto neighbor_node_map = compute_nodal_neighbors(mesh);

    // std::vector<std::vector<const Elem *>> nodes_to_elem_map;
    // MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);

    // std::vector<std::vector<const Elem *>> nodes_to_elem_map;

    // MeshTools::build_nodes_to_elem_map(mesh, nodes_to_elem_map);
    
    Parameters params;
    // params.set<std::vector<std::vector<const Elem *>>>("nodes_to_elem_map") = nodes_to_elem_map;
    params.set<std::unordered_map<const Node *, std::vector<const Node *>>>("neighbor_node_map") = neighbor_node_map;



    // /*MESH BOUNDARY*/
    // std::set<boundary_id_type> bcids;
    // bcids.insert(0);
    // bcids.insert(1);
    // bcids.insert(2);
    // bcids.insert(3);

    // //     /* SET EQUATION_SYSTEMS */
    // EquationSystems equation_systems(mesh);

    // {
    //     std::string system_name("system");
    //     std::string system_var(system_name + "var_0");
    //     equation_systems.add_system<ExplicitSystem>(system_name);
    //     equation_systems.get_system(system_name).add_variable(system_var, FEType(p_order, fe_family));
    // }

    // {
    //     std::string system_name("output");
    //     std::string system_var(system_name + "var_0");
    //     equation_systems.add_system<ExplicitSystem>(system_name);
    //     equation_systems.get_system(system_name).add_variable(system_var, FEType(p_order, fe_family));
    // }

    // std::string system_name("output");
    // std::string system_var(system_name + "var_0");

    // auto &input_system = equation_systems.get_system("system");
    // auto &output_system = equation_systems.get_system("output");

    // equation_systems.init();

    // input_system.project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);
    // output_system.project_solution(*integrate_patch, libmesh_nullptr, params);


    // equation_systems.update();
    // ExodusII_IO(mesh).write_discontinuous_exodusII("equation_system_out.e",
    //                                                equation_systems);

    

    //     std::ofstream outfile;
    //     outfile.open("exact_output_node.e", std::ios_base::app);
    //     equation_systems.get_system(system_name).solution->print(outfile);

    //     // equation_systems.get_system("output_exact").project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);

    //     ZeroFunction<Number> zero_func;
    //     std::vector<unsigned int> variables_sys;
    //     variables_sys.push_back(0);
    //     input_system.get_dof_map().add_dirichlet_boundary(DirichletBoundary(bcids, variables_sys, zero_func));

    //     // // lambda fcn to fill empty vector with centroids, then evaluate another fcn
    //     // constexpr centroid = []( const auto elem ){return elem->centroid();};
    //     // std::vector<Point> centroids( mesh.n_elem() );
    //     // std::for_each(std::begin(centroids), std::end(centroids), centroid_coord );

    //     // std::unique_ptr<NumericVector<Number>>
    //     // const MyOutput &
    //     // MyUniqueOutput
    //     //     scalarfield_node = integrate_exact(equation_systems);

    //     // output_system.solution = scalarfield_node->clone();

    //     write_libmesh_info(equation_systems);

    //     ExactSolution exact_sol(equation_systems);
    //     exact_sol.attach_exact_value(*exact_solution);
    //     exact_sol.attach_exact_deriv(*exact_gradient);

    // equation_systems.update();
    // ExodusII_IO(mesh).write_discontinuous_exodusII("equation_system_out.e",
    //                                                equation_systems);

    //     // The patch recovery estimator should give a
    //     // good estimate of the solution interpolawtion error.
    //     ErrorVector error;
    //     PatchRecoveryErrorEstimator error_estimator;
    //     error_estimator.estimate_error(equation_systems.get_system("system"),
    //                                    error);
    //     // Output error estimate magnitude
    //     libMesh::out << "Error estimate\nl2 norm = "
    //                  << error.l2_norm()
    //                  << "\nmaximum = "
    //                  << error.maximum()
    //                  << std::endl;

    //     //  l1_errors.push_back( error.l1_norm() );
    //     l2_errors.push_back(error.l2_norm());
    //     linf_errors.push_back(error.maximum());

    //     nelems.push_back(sqrt(mesh.n_elem()));
    // }

    // for (int i = 0; i < linf_errors.size() - 1; i++)
    // {
    //     // Real slope = log(linf_errors[i+1]) - log(linf_errors[i]) / ( log(nelems[i+1]) - log(nelems[i]) ) ;
    //     Real slope = log(linf_errors[i + 1]) - log(linf_errors[i]) / (log(nelems[i + 1]) - log(nelems[i]));
    //     std::cout << " iter " << i << " slope " << slope;
    //     // std::
    // }

    return 0;
}

Number integrate_patch(const Point &p, const Parameters &parameters,
                       const std::string &system, const std::string &var)
{

    Real elemval = 0;

    //  nelem = mesh.n_active_elem();
    //  ndimensions = mesh.mesh_dimension();

    // parameters.get

    std::unique_ptr<FEBase> fe(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qrule(QBase::build(quad_type, ndimensions, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &qpoint = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();
    // std::cout << "\n qrule->quad_order() " << qrule->get_order();

    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    // DenseVector<Number> Fexact;
    std::vector<dof_id_type> dof_indices;
    std::vector<dof_id_type> dof_indices_neighbor;
    std::vector<dof_id_type> one_dof_index(1);

    auto nodes_to_elem_map = parameters.get<std::vector<std::vector<const Elem *>>>("nodes_to_elem_map");

    // const Mesh &mesh = meshvec[0];
    // auto mesh = meshvec[0];
    // // parameters.get<Mesh>("mesh").sub_point_locator();
    // std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();
    // point_locator->enable_out_of_mesh_mode();
    // const Elem *elem = (*point_locator)(p);

    // MeshTools::find_nodal_neighbors(mesh, Node(p), nodes_to_elem_map, neighbor_nodes);
    // std::vector<dof_id_type> neighbor_node_ids(neighbor_nodes.size());

    // libMesh::Patch patch(mesh.processor_id());
    // patch.build_around_element(elem,
    //                            std::pow(1 + 2 * nelem_div_horizon, ndimensions),
    //                            &Patch::add_point_neighbors);

    // const Point &point_elem = elem->centroid();

    // for (const auto &neighbor : patch)
    // {

    //     // dof_map.dof_indices(elem, dof_indices_neighbor);

    //     fe->reinit(neighbor);

    //     for (auto qp : range(qrule->n_points()))
    //     {
    //         const Point &xi((point_elem - qpoint[qp]));

    //         Real kernelval = (kernel(xi * horizon)) / (horizon * horizon);

    //         elemval += JxW[qp] * kernelval * exact_solution(qpoint[qp], parameters, std::string(), std::string());
    //     }
    // }

    return elemval;
}

// using namespace clt;

// C++ template to print vector container elements
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
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

    std::transform(expanded.begin(), expanded.end(), result.begin(), [](T value) { return std::to_string(value); });
    return result;
}





// typedef MyOutput = std::unique_ptr<NumericVector<Number>>;

// template <typename T1 , typename T2>
// std::tuple<T2,T2> = integrate_input(EquationSystems &equation_systems, T1 &input);
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

void write_libmesh_info(EquationSystems &equation_systems)
{

    const MeshBase &mesh = equation_systems.get_mesh();
    const unsigned int ndimensions = mesh.mesh_dimension();
    ExplicitSystem &system =
        equation_systems.get_system<ExplicitSystem>("system");
    const DofMap &dof_map = system.get_dof_map();
    auto n_dofs_total = dof_map.n_dofs();
    auto nelem = mesh.n_active_elem();
    // AnalyticFunction<Number> exact_solution_object(exact_solution);

    FEType fe_type = system.variable_type(0);

    // auto& exact = system.get_vector("exact");
    // auto& approx = system.get_vector("approx");
    // auto& error = system.get_vector("error");

    std::unique_ptr<NumericVector<Number>> centroid_values =
        system.solution->zero_clone();

    std::unique_ptr<FEBase> fe(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qrule(QBase::build(QGAUSS, ndimensions, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &qpoint = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();
    std::cout << "\n qrule->quad_order() " << qrule->get_order();

    std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qface(QBase::build(QTRAP, ndimensions - 1, quad_order));
    fe_face->attach_quadrature_rule(qface.get());

    const std::vector<Real> &JxW_face = fe_face->get_JxW();
    const std::vector<Point> &qpoint_face = fe_face->get_xyz();
    const std::vector<std::vector<Real>> &phi_face = fe_face->get_phi();
    std::cout << "\n qrule->quad_order() " << qface->get_order();

    std::unique_ptr<NumericVector<Number>> scalarfield_elem =
        NumericVector<Number>::build(mesh.comm());
    scalarfield_elem->init(nelem, false, SERIAL);

    std::unique_ptr<NumericVector<Number>> exact_scalarfield_elem = scalarfield_elem->zero_clone();

    std::unique_ptr<NumericVector<Number>> scalarfield_node = system.solution->zero_clone();

    // std::vector<Number> scalarfield_elem(mesh.n_elem());
    // std::vector<Number> exact_scalarfield_elem(mesh.n_elem());

    // auto elemval = scalarfield_elem.begin();
    // auto elemval_exact = exact_scalarfield_elem.begin();

    std::stringstream sstm;
    // sstm << "_" << nelem << horizon;
    std::string savename = sstm.str();

    std::ofstream outfile_elem;
    outfile_elem.open(savename + "elem_values.e", std::ios_base::app);

    std::ofstream outfile_qtrapside;
    outfile_qtrapside.open(savename + "qtrap_side.e", std::ios_base::app);

    std::ofstream outfile_qgausselem;
    outfile_qgausselem.open(savename + "qgauss_elem.e", std::ios_base::app);

    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    // DenseVector<Number> Fexact;
    std::vector<dof_id_type> dof_indices;
    std::vector<dof_id_type> one_dof_index(1);

    // for (auto elem : mesh.element_ptr_range())
    for (auto elem : mesh.element_ptr_range())
    {
        Real elemval_exact = 0;
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

        std::ofstream outfile;
        outfile.open(savename + "patch.e", std::ios_base::app);

        for (const auto &neighbor : patch)
        {

            fe->reinit(neighbor);

            //qpoints
            for (auto qp : range(qrule->n_points()))
                for (auto i : range(3))
                    outfile << qpoint[qp](i) << " ";

            //JxW
            for (auto val : JxW)
                outfile << val << " ";

            //xi
            for (auto xprime : qpoint)
                outfile << (point_elem - xprime) << " ";

            //kernel
            for (auto xprime : qpoint)
                outfile << kernel(point_elem - xprime) << " ";
        }

        outfile << " \n ";
    }
}

MyUniqueOutput
integrate_exact(EquationSystems &equation_systems)
{
    const MeshBase &mesh = equation_systems.get_mesh();
    auto nelem = mesh.n_active_elem();
    const unsigned int ndimensions = mesh.mesh_dimension();

    ExplicitSystem &system = equation_systems.get_system<ExplicitSystem>("system");

    const DofMap &dof_map = system.get_dof_map();
    auto n_dofs_total = dof_map.n_dofs();

    double horizon = equation_systems.parameters.get<double>("horizon");

    FEType fe_type = system.variable_type(0);
    std::unique_ptr<NumericVector<Number>> centroid_values =
        system.solution->zero_clone();

    std::unique_ptr<NumericVector<Number>> scalarfield_elem =
        NumericVector<Number>::build(mesh.comm());
    scalarfield_elem->init(nelem, false, SERIAL);

    std::unique_ptr<NumericVector<Number>> exact_scalarfield_elem = scalarfield_elem->zero_clone();

    std::unique_ptr<NumericVector<Number>> scalarfield_node = system.solution->zero_clone();

    std::stringstream sstm;
    // sstm << "_" << nelem << horizon;
    std::string savename = sstm.str();

    std::ofstream outfile_elem;
    outfile_elem.open(savename + "elem_values.e", std::ios_base::app);

    // auto& exact = system.get_vector("exact");
    // auto& approx = system.get_vector("approx");
    // auto& error = system.get_vector("error");

    std::unique_ptr<FEBase> fe(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qrule(QBase::build(quad_type, ndimensions, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &qpoint = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();
    std::cout << "\n qrule->quad_order() " << qrule->get_order();

    // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    // std::unique_ptr<QBase> qface(QBase::build(QTRAP, ndimensions - 1, quad_order));
    // fe_face->attach_quadrature_rule(qface.get());

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
        Real elemval_exact = 0;
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

                elemval += JxW[qp] * kernelval * exact_solution(qpoint[qp], equation_systems.parameters, std::string(), std::string());
            }
        }

        fe->reinit(elem);
        for (auto qp : range(qrule->n_points()))
            for (auto i : range(n_dofs))
                Fe(i) += JxW[qp] * phi[i][qp] * elemval;

        // elemval /= JxWkernelsum;
        // Fe.scale( 1.0 / JxWkernelsum);

        elemval_exact += exact_solution(point_elem, equation_systems.parameters, std::string(), std::string());

        for (auto i : range(3))
            outfile_elem << point_elem(i) << " ";
        outfile_elem << elemval << " " << elemval_exact;
        outfile_elem << " \n ";

        dof_map.constrain_element_vector(Fe, dof_indices);

        scalarfield_node->add_vector(Fe, dof_indices);
    }

    {
        std::ofstream outfile;
        outfile.open("scalarfield_node.e", std::ios_base::app);
        scalarfield_node->print(outfile);
    }

    return scalarfield_node;
}

libMesh::Patch
build_neighborhood(const MeshBase &mesh,
                   const Elem &base_elem,
                   const Point &X,
                   const unsigned int target_patch_size,
                   const Real eps
                   //  std::unique_ptr<PointLocatorTree> point_locator
)
{
    // PointLocatorBase*& point_locator
    // std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();
    // PointLocatorTree &point_locator = mesh.sub_point_locator(); //sub_point_locator();
    /*   if (!point_locator->initialized())
  {
    libMesh::out << " initialize point locator...";
    libMesh::out.flush();
    point_locator->init();
  }

  if (point_locator == nullptr)
    try
    {
      mesh.sub_point_locator();
    }
    catch (const std::exception &)
    {
      point_locator = new PointLocatorTree(mesh);
    }

  if (!mesh.is_serial()) 
    point_locator->enable_out_of_mesh_mode();
    */
    libMesh::Patch patch_elems(mesh.processor_id());
    // unsigned int target_patch_size = 9;
    int nlocalelem = mesh.n_local_elem();
    patch_elems.build_around_element(&base_elem, target_patch_size,
                                     &Patch::add_point_neighbors);
    int patchsize = patch_elems.size();

    // if(elem==nullptr)
    //   *base_elem = (*point_locator)(X);
    // else if(X==std::nullptr_t)
    //   &X = base_elem->centroid();

    // std::set<const Elem *> patch_elems;
    // patch_elems->insert(base_elem);
    // std::set<const Elem *> visited_elems = patch_elems;

    /*   
  std::set<const Elem *> frontier_elems = patch_elems;
  while (!frontier_elems.empty())
  {
    std::set<const Elem *> new_frontier;
    for (auto e : frontier_elems)
    {
      for (auto neighbor : e->neighbor_ptr_range())
      {
        if (neighbor != nullptr && !patch_elems.count(neighbor))
        {
          for (int k : neighbor->node_index_range())
          {

            if (signed_distance_fcn(X - neighbor->point(k), eps) > 0.0)
            {
              patch_elems.insert(neighbor);
              new_frontier.insert(neighbor);
              break;
            }
          }
          // visited_elems.insert(neighbor);
        }
      }
    }
    frontier_elems = new_frontier;
  } */

    return patch_elems;
}