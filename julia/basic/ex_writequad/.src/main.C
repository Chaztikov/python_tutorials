#include "kernel.h"
#include "helpers.h"

using namespace libMesh;

namespace clt {
 
    const unsigned int ndimensions = 2;

    const QuadratureType quad_type = QTRAP;

    const Order quad_order = SECOND;

    const Order p_order = SECOND;
    const FEFamily fe_family = LAGRANGE;


    double xminmax[] = {-0.5, 0.5};
    double xmin = xminmax[0], xmax = xminmax[1]; 

    const double horizon = 0.1;
    const int nelem1 = static_cast<int>(  (xmax-xmin) / horizon);


}

using namespace clt;

Number exact_solution(Real x);
Number exact_solution(const Point &p);

Number exact_solution(Real x)
{
    Real val = 0;
    val += 0.5 * (1. + cos(M_PIl * 2*x) );
    return val * boundary_indicator(x);
}

Number exact_solution(const Point &p)
{
    Real val = 0;
    Real x = 0, y = 0;
    x += p(0);
    y += p(1);
    val += exact_solution(x) * exact_solution(y);
    return val * boundary_indicator(p);
}



int main(int argc, char **argv)
{

    LibMeshInit init(argc, argv);
    
    SerialMesh mesh(init.comm(), ndimensions);
    const ElemType elem_type= QUAD9;
    MeshTools::Generation::build_square(mesh, nelem1, nelem1, xmin, xmax, xmin, xmax, elem_type);

}

std::unique_ptr<NumericVector<Number>>
integrate_exact(EquationSystems &equation_systems)
{

    const MeshBase &mesh = equation_systems.get_mesh();
    const unsigned int ndimensions = mesh.mesh_dimension();
    ExplicitSystem &system =
        equation_systems.get_system<ExplicitSystem>("system");
    const DofMap &dof_map = system.get_dof_map();
    auto n_dofs_total = dof_map.n_dofs();
    auto nelem = mesh.n_active_elem();


    FEType fe_type = system.variable_type(0);


    std::unique_ptr<NumericVector<Number>> field_values_node = system.solution->zero_clone();

  std::unique_ptr<NumericVector<Number>> field_values_elem =
    NumericVector<Number>::build(mesh.comm());
  field_values_elem->init( nelem, false, SERIAL);
//   X.localize(*X_localized);



    std::unique_ptr<NumericVector<Number>> centroid_values =
        system.solution->zero_clone();


    std::unique_ptr<FEBase> fe(FEBase::build(ndimensions, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qrule(QBase::build(QGAUSS , ndimensions, quad_order));
    fe->attach_quadrature_rule(qrule.get());

    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &qpoint = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();
    std::cout << "\n qrule->quad_order() " << qrule->get_order();


    std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, FEType(p_order, fe_family)));
    std::unique_ptr<QBase> qface(QBase::build(QTRAP , ndimensions-1, quad_order));
    fe_face->attach_quadrature_rule(qface.get());

    const std::vector<Real> &JxW_face = fe_face->get_JxW();
    const std::vector<Point> &qpoint_face = fe_face->get_xyz();
    const std::vector<std::vector<Real>> &phi_face = fe_face->get_phi();
    std::cout << "\n qrule->quad_order() " << qface->get_order();



    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    std::vector<dof_id_type> dof_indices;


    for (auto elem : mesh.element_ptr_range())
    {
        Real elemval = 0;

        dof_map.dof_indices(elem, dof_indices);
        const unsigned int n_dofs = dof_indices.size();
        Fe.zero();
        Fe.resize(n_dofs);

        fe->reinit(elem);

        Real JxWkernelsum = 0;

        const Point &point_elem = elem->centroid();

        for (auto qp : range(qrule->n_points()))
        {
            const Point &xi((point_elem - qpoint[qp]) );

            Real kernelval = 0;
            kernelval += kernel(xi);
            JxWkernelsum += JxW[qp] * kernelval;

            elemval += JxW[qp] * kernelval * exact_solution(qpoint[qp]);

            for (auto i : range(n_dofs))
                Fe(i) += JxW[qp] * phi[i][qp] * exact_solution(qpoint[qp]) * kernelval;
        }

        elemval /= JxWkernelsum;
        Fe.scale(JxWkernelsum);

        dof_map.constrain_element_vector(Fe, dof_indices);

        field_values_node->add_vector(Fe, dof_indices);

    }

    std::ofstream outfile2;
    outfile2.open("field_values_node.txt", std::ios_base::app);
    field_values_node->print(outfile2);

    return field_values_node;
}
