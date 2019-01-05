// #include "kernel.h"
#include "helpers.h"
#include <sstream>
#include <libmesh/analytic_function.h>
#include "libmesh/vectormap.h"
// libMesh::AnalyticFunction< Output >::AnalyticFunction	(	OutputFunction 	fptr	)
// libMesh::vectormap

using namespace libMesh;

namespace clt
{

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

// const double helems_div_horizon = 1.0; // h_elem / horizon < 1

// const int nelem1 = static_cast<int>((xmax - xmin) * nelem_div_horizon / horizon);

const int nelem1 = static_cast<int>(nelem_div_horizon / horizon);

// std::unique_ptr<NumericVector<Number>> test;
// NumericVector<Number>::build(mesh.comm());
// scalarfield_elem->init(nelem, false, SERIAL);

} // namespace clt

using namespace clt;

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

// Number exact_solution(Real x);
// Number exact_solution(const Point &p);
Number kernel(const Point &p);

Number cosine_solution(Real x)
{
    Real val = 0;
    val += 0.5 * (1. + cos(M_PIl * 2 * x));
    return val * boundary_indicator(x);
}

// Number exact_solution(const Point &p)
// {
//     Real val = 0;
//     Real x = 0, y = 0;
//     x += p(0);
//     y += p(1);
//     val += cosine_solution(x) * cosine_solution(y);
//     return val * boundary_indicator(p);
// }

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

void write_libmesh_info(EquationSystems &equation_systems);
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

// //random lambda
// bool allBetweenAandB = std::all_of(numbers.begin(), numbers.end(),
//            [a,b](int x) { return a <= x && x <= b; });

// //indecision...
// struct
// {

// }
// 1
// auto [output1, output2] = f(const Input& input);

#include <iostream>
#include <tuple>
#include <functional>

// std::tuple< std::unique_ptr<NumericVector<Number>>,
// std::unique_ptr<NumericVector<Number>>,
// std::unique_ptr<NumericVector<Number>>

// alias

typedef class libMesh::NumericVector<Number>
    MyInput;

typedef NumericVector<Number>
    MyOutput;

typedef std::unique_ptr<MyOutput>
    MyUniqueOutput;

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
MyUniqueOutput
integrate_exact(EquationSystems &equation_systems)
{
    const MeshBase &mesh = equation_systems.get_mesh();
    auto nelem = mesh.n_active_elem();
    const unsigned int ndimensions = mesh.mesh_dimension();

    ExplicitSystem &system = equation_systems.get_system<ExplicitSystem>("system");

    const DofMap &dof_map = system.get_dof_map();
    auto n_dofs_total = dof_map.n_dofs();

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

                Real kernelval = 0;
                kernelval += kernel(xi) / (horizon * horizon );
                JxWkernelsum += JxW[qp] * kernelval;

                elemval += JxW[qp] * kernelval * exact_solution(qpoint[qp], equation_systems.parameters, std::string(), std::string());

                for (auto i : range(n_dofs))
                    Fe(i) += JxW[qp] * phi[i][qp] * exact_solution(qpoint[qp], equation_systems.parameters, std::string(), std::string()) * kernelval;
            }
        }

        // elemval /= JxWkernelsum;
        // Fe.scale( 1.0 / JxWkernelsum);

        elemval_exact += exact_solution(point_elem, equation_systems.parameters, std::string(), std::string());

        for (auto i : range(3))
            outfile_elem << point_elem(i) << " ";
        outfile_elem << elemval << " " << elemval_exact;
        outfile_elem << " \n ";

        dof_map.constrain_element_vector(Fe, dof_indices);

        scalarfield_node->add_vector(Fe, dof_indices);

        // exact_scalarfield_elem->add(Fexact , one_dof_index);
    }

    {
        std::ofstream outfile;
        outfile.open("scalarfield_node.e", std::ios_base::app);
        scalarfield_node->print(outfile);
    }

    return scalarfield_node;
}


int main(int argc, char **argv)
{

    LibMeshInit init(argc, argv);

    SerialMesh mesh(init.comm(), ndimensions);
    const ElemType elem_type = QUAD9;



std::vector<double> nelems;
std::vector<double> linf_errors;
std::vector<double> l1_errors;
std::vector<double> l2_errors;

std::vector<double> nelemsvec = {10, 20, 40, 80};
    for(unsigned int nelem1 : nelemsvec)
    {

    

    MeshTools::Generation::build_square(mesh, nelem1, nelem1, xmin, xmax, xmin, xmax, elem_type);

    mesh.prepare_for_use();

    /*MESH BOUNDARY*/
    std::set<boundary_id_type> bcids;
    bcids.insert(0);
    bcids.insert(1);
    bcids.insert(2);
    bcids.insert(3);

    /* SET EQUATION_SYSTEMS */
    EquationSystems equation_systems(mesh);

    {
        std::string system_name("system");
        std::string system_var(system_name + "var_0");
        equation_systems.add_system<ExplicitSystem>(system_name);
        equation_systems.get_system(system_name).add_variable(system_var, FEType(p_order, fe_family));
    }

    {
        std::string system_name("output");
        std::string system_var(system_name + "var_0");
        equation_systems.add_system<ExplicitSystem>(system_name);
        equation_systems.get_system(system_name).add_variable(system_var, FEType(p_order, fe_family));
    }

    std::string system_name("output");
    std::string system_var(system_name + "var_0");

    auto &input_system = equation_systems.get_system("system");
    auto &output_system = equation_systems.get_system("output");

    equation_systems.init();

    input_system.project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);
    // input_system.project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);

    std::ofstream outfile;
    outfile.open("exact_output_node.e", std::ios_base::app);
    equation_systems.get_system(system_name).solution->print(outfile);

    // equation_systems.get_system("output_exact").project_solution(*exact_solution, libmesh_nullptr, equation_systems.parameters);

    ZeroFunction<Number> zero_func;
    std::vector<unsigned int> variables_sys;
    variables_sys.push_back(0);
    input_system.get_dof_map().add_dirichlet_boundary(DirichletBoundary(bcids, variables_sys, zero_func));

    // // lambda fcn to fill empty vector with centroids, then evaluate another fcn
    // constexpr centroid = []( const auto elem ){return elem->centroid();};
    // std::vector<Point> centroids( mesh.n_elem() );
    // std::for_each(std::begin(centroids), std::end(centroids), centroid_coord );

    // std::unique_ptr<NumericVector<Number>>
    // const MyOutput &
    MyUniqueOutput
        scalarfield_node = integrate_exact(equation_systems);

    output_system.solution = scalarfield_node->clone();

    write_libmesh_info(equation_systems);

    ExactSolution exact_sol(equation_systems);
    exact_sol.attach_exact_value(*exact_solution);
    exact_sol.attach_exact_deriv(*exact_gradient);

    equation_systems.update();
    ExodusII_IO(mesh).write_discontinuous_exodusII("equation_system_out.e",
                                                   equation_systems);

    // The patch recovery estimator should give a
    // good estimate of the solution interpolation error.
    ErrorVector error;
    PatchRecoveryErrorEstimator error_estimator;
    error_estimator.estimate_error(equation_systems.get_system("system"),
                                   error);
    // Output error estimate magnitude
    libMesh::out << "Error estimate\nl2 norm = "
                 << error.l2_norm()
                 << "\nmaximum = "
                 << error.maximum()
                 << std::endl;

                //  l1_errors.push_back( error.l1_norm() );
                 l2_errors.push_back( error.l2_norm() );
                 linf_errors.push_back( error.maximum() );

                 nelems.push_back( sqrt( mesh.n_elem()));
    }


    for(int i =0; i<linf_errors.size() - 1; i++)
    {
        // Real slope = log(linf_errors[i+1]) - log(linf_errors[i]) / ( log(nelems[i+1]) - log(nelems[i]) ) ; 
        Real slope = log(linf_errors[i+1]) - log(linf_errors[i]) / ( log(nelems[i+1]) - log(nelems[i]) ) ; 
        std::cout << " iter " << i << " slope " << slope;
        // std:: 
    }

    return 0;
}
/* 
OutputFunction

// Boundary conditions for the 3D test case
class OutputFunction : public FunctionBase<Number>
{
public:
  OutputFunction (unsigned int u_var, ValueFunctionPointer)
    : _u_var(u_var)
  { this->_initialized = true; }
//   { libmesh_not_implemented(); }

  virtual Number operator() (const Point & p, const Real = 0)
  {
      return exact_solution(p);
  }


  virtual void operator() (const Point & p,
                           const Real,
                           DenseVector<Number> & output)
  {
    libmesh_not_implemented();
  }

  virtual std::unique_ptr<FunctionBase<Number>> clone() const
  { return libmesh_make_unique<OutputFunction>(_u_var); }

private:
  const unsigned int _u_var;

};
 */
