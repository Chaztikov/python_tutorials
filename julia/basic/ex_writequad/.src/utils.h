
#include "operators.h"
#include <iostream>
#include <memory>

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

double signed_distance_fcn(const Point &p, const double eps) {
  double D_oo = 0.0;
  for (int d = 0; d < p.size(); ++d)
    D_oo = std::max(D_oo, std::abs(p(d)));
  // for (auto const& coord : D.begin)
  //     D_oo = std::max(D_oo, std::abs(coord));
  // for(auto const& value: a)

  return eps - D_oo;
}

double rho(const Point &D, const double eps) {
  double rho = 1.0;
  for (int d = 0; d < D.size(); ++d) {
    if (std::abs(D(d)) < eps) {
      rho *= 0.5 * (cos(M_PI * D(d) / eps) + 1.0) / eps;
    } else {
      rho = 0.0;
    }
  }
  return rho;
}

std::set<const Elem *>
build_neighborhood(const MeshBase &mesh, const Point &X,
                   const double eps) //, PointLocatorBase*& point_locator)
{
  // PointLocatorBase*& point_locator
  std::unique_ptr<PointLocatorBase> point_locator = mesh.sub_point_locator();

  // if (point_locator == nullptr) point_locator = new PointLocatorTree(mesh);
  if (!point_locator->initialized()) {
    libMesh::out << " initialize point locator...";
    libMesh::out.flush();
    point_locator->init();
  }
  const Elem *base_elem = (*point_locator)(X);


  std::set<const Elem *> patch_elems;
  patch_elems.insert(base_elem);
  std::set<const Elem *> visited_elems = patch_elems;
  std::set<const Elem *> frontier_elems = patch_elems;
  while (!frontier_elems.empty()) {
    std::set<const Elem *> new_frontier;
    for (auto e : frontier_elems) {
      for (auto neighbor : e->neighbor_ptr_range()) {
        if (neighbor != nullptr && !visited_elems.count(neighbor)) {
          for (int k : neighbor->node_index_range()) {
            const Point &Y = neighbor->point(k);
            const Point D = X - Y;
            if (signed_distance_fcn(D, eps) > 0.0) {
              patch_elems.insert(neighbor);
              new_frontier.insert(neighbor);
              break;
            }
          }
          visited_elems.insert(neighbor);
        }
      }
    }
    frontier_elems = new_frontier;
  }
  return patch_elems;
}

Number kernel_bspline3(Real x, const Parameters &parameters) {
  Number val = 0;
  if (abs(x) < 0.5)
    val += 1. / 6. - pow(x, 2) + pow(abs(x), 3);
  else if (abs(x) < 1.)
    val += -(1. / 3.) * (-1 + pow(abs(x), 3));
  else
    0;

  return val;
}

Number skernel(const Point &p, const Parameters &parameters,
               const std::string &, const std::string &) {
  const std::string kernel_string =
      parameters.get<std::string>("kernel_string");
  auto horizon = parameters.get<Real>("horizon");
  auto kernelparam = parameters.get<Real>("kernelparam");
  auto ndimensions = parameters.get<unsigned int>("ndimensions");
  // Real x = p(0), y = p(1), z = p(2);
  Real x = p(0) , y = p(1) , z = p(2) ;
  // Real x = p(0) / horizon, y = p(1) / horizon, z = p(2) / horizon;
  Real kernelval = 0;
  // if (p.norm() > horizon)

  if (abs(x) > 1 || abs(y) > 1)
    1;
  else {
    kernelval +=
        kernel_bspline3(x, parameters) * kernel_bspline3(y, parameters);

    // ParserWithConsts fparser;
    // int res = fparser.Parse(kernel_string, "x,y,z");
    // double vars[3] = {x,y,z};
    // kernelval += fparser.Eval(vars);
 // + (Number)kernelparam);
    // std::cout << "\n coord " << x << " , " << y << " kernel value  " <<
    // kernelval;
    kernelval /=
        std::pow(horizon, (Number)ndimensions);
  }
  libMesh::out.flush();
  return kernelval;
}

// Number spline_kernel(const Point &p, const Parameters &parameters,
//                const std::string &, const std::string &) {
//   const std::string kernel_string =
//       parameters.get<std::string>("kernel_string");
//   auto horizon = parameters.get<Real>("horizon");
//   auto kernelparam = parameters.get<Real>("kernelparam");
//   auto ndimensions = parameters.get<unsigned int>("ndimensions");
//   Real x = p(0), y = p(1), z = p(2);
//   Real kernelval = 0;
//   // if (p.norm() > horizon)
//   if (abs(x) > horizon || abs(y) > horizon)
//     return 0;
//   else {
//     ParserWithConsts fparser;
//     int res = fparser.Parse(kernel_string, "x,y,z");
//     double vars[3] = {p(0) / horizon, p(1) / horizon, p(2) / horizon};
//     // double vars[3] = {p(0), p(1), p(2)};
//     kernelval += fparser.Eval(vars);
//     kernelval /= std::pow(horizon, (Number)ndimensions);// +
//     (Number)kernelparam);
//     // kernelval /= std::pow(horizon, (Number)ndimensions +
//     (Number)kernelparam); return kernelval;
//   }
// }

Real findiff_laplace5(FunctionBase<Number> &exact_solution, const Point &p,
                      Real eps, const unsigned int ndimensions) {
  const Real x = p(0);
  const Real y = p(1);
  const Real z = p(2);
  // const Real eps = 1.e-3;

  const Real uxx = exact_solution(Point(x - eps, y, z)) +
                   exact_solution(Point(x + eps, y, z)) +
                   -2. * exact_solution(Point(x, y, z)) / eps / eps;

  const Real uyy = exact_solution(Point(x, y - eps, z)) +
                   exact_solution(Point(x, y + eps, z)) +
                   -2. * exact_solution(Point(x, y, z)) / eps / eps;

  const Real uzz = exact_solution(Point(x, y, z - eps)) +
                   exact_solution(Point(x, y, z + eps)) +
                   -2. * exact_solution(Point(x, y, z)) / eps / eps;
  return -(uxx + uyy + ((ndimensions == 2) ? 0. : uzz));
}

/*
double
signed_distance_fcn(const Point &D, const double eps)
{
  double D_oo = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    D_oo = std::max(D_oo, std::abs(D(d)));
  }
  return eps - D_oo;
}

double
rho(const Point &D, const double eps)
{
  double rho = 1.0;
  for (int d = 0; d < dim; ++d)
  {
    if (std::abs(D(d)) < eps)
    {
      rho *= 0.5 * (cos(M_PI * D(d) / eps) + 1.0) / eps;
    }
    else
    {
      rho = 0.0;
    }
  }
  return rho;
}

std::set<const Elem *>
build_neighborhood(const MeshBase &mesh, const Point &X, const double eps,
PointLocatorBase *&point_locator)
{
  if (point_locator == nullptr)
    point_locator = new PointLocatorTree(mesh);
  if (!point_locator->initialized())
    point_locator->init();
  const Elem *base_elem = (*point_locator)(X);

  std::set<const Elem *> patch_elems;
  patch_elems.insert(base_elem);
  std::set<const Elem *> visited_elems = patch_elems;
  std::set<const Elem *> frontier_elems = patch_elems;
  while (!frontier_elems.empty())
  {
    std::set<const Elem *> new_frontier;
    for (auto e : frontier_elems)
    {
      for (auto neighbor : e->neighbor_ptr_range())
      {
        if (neighbor != nullptr && !visited_elems.count(neighbor))
        {
          for (int k : neighbor->node_index_range())
          {
            const Point &Y = neighbor->point(k);
            const Point D = X - Y;
            if (signed_distance_fcn(D, eps) > 0.0)
            {
              patch_elems.insert(neighbor);
              new_frontier.insert(neighbor);
              break;
            }
          }
          visited_elems.insert(neighbor);
        }
      }
    }
    frontier_elems = new_frontier;
  }
  return patch_elems;
}
 */

Real compute_error_scalar_weak(EquationSystems &equation_systems,
                               NumericVector<Number> &scalar_field_in,
                               FunctionBase<Number> &exact_function) {
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  // auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  // p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;
  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  // const std::vector<std::vector<RealGradient>> &dphi_elem =
  // fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  std::vector<dof_id_type> dof_indices;
  DenseVector<Number> Fe;
  Real L2error = 0;

  auto scalar_field_out = scalar_field_in.zero_clone();

  // for (const auto & elem : as_range(mesh.active_subdomain_elements_begin(1),
  // mesh.active_subdomain_elements_begin(1))) for (const auto & elem :
  // as_range(mesh.active_subdomain_elements_begin(0),
  // mesh.active_subdomain_elements_begin(0)))
  for (const auto &elem : mesh.element_ptr_range()) {
    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);
    if (elem->subdomain_id() == 0) {
      Real kernelsum = 0;

      for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {
        Real u = 0;
        for (unsigned int l = 0; l < n_dofs; l++)
          u += (phi_elem[l][qp] * exact_function(qpoint_elem[qp]) -
                scalar_field_in(l));

        // u += phi_elem[l][qp] * exact_function(qpoint_elem[qp] ,
        // equation_systems.parameters, std::string(), std::string());

        for (unsigned int i = 0; i < n_dofs; i++) {
          Real val = JxW_elem[qp] * phi_elem[i][qp] * (u * u);
          Fe(i) += val;
        }
        // l1err_system.solution->add_vector( Fe.l1_norm())

      } // qp
    } else {
      //...
    }
    scalar_field_out->add_vector(Fe, dof_indices);
  }
  // scalar_field_out->sqrt();
  // scalar_field_out->close();

  Real l2error = scalar_field_out->sum(); // comm().sum(scalar_field_out);
  L2error = sqrt(L2error);
  l2error = sqrt(l2error);

  libMesh::out << "\n \n L2error " << L2error << " \n l2error " << l2error;
  ////libMesh::out.flush();
  return L2error;
}

std::unique_ptr<NumericVector<Number>>
compute_error_scalar_weak(EquationSystems &equation_systems,
                          NumericVector<Number> &scalar_field_in) {

  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  // auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  // p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  // const std::vector<std::vector<RealGradient>> &dphi_elem =
  // fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  std::vector<dof_id_type> dof_indices;
  DenseVector<Number> Fe;
  Real L2error = 0;

  std::unique_ptr<NumericVector<Number>> scalar_field_out =
      scalar_field_in.zero_clone();
  // auto scalar_field_out = scalar_field_in.zero_clone();

  // for (const auto & elem : as_range(mesh.active_subdomain_elements_begin(1),
  // mesh.active_subdomain_elements_begin(1))) for (const auto & elem :
  // as_range(mesh.active_subdomain_elements_begin(0),
  // mesh.active_subdomain_elements_begin(0)))
  for (const auto &elem : mesh.element_ptr_range()) {
    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);
    if (elem->subdomain_id() == 0) {
      Real kernelsum = 0;

      for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {
        Real u = 0;
        for (unsigned int l = 0; l < n_dofs; l++)
          u += (phi_elem[l][qp] * scalar_field_in(l));

        // u += phi_elem[l][qp] * exact_function(qpoint_elem[qp] ,
        // equation_systems.parameters, std::string(), std::string());

        for (unsigned int i = 0; i < n_dofs; i++) {
          Real val = JxW_elem[qp] * phi_elem[i][qp] * (u * u);
          Fe(i) += val;
          L2error += val;
        }
        // l1err_system.solution->add_vector( Fe.l1_norm())

      } // qp
    } else {
      //...
    }
    scalar_field_out->add_vector(Fe, dof_indices);
  }
  // scalar_field_out->sqrt();
  scalar_field_out->close();

  Real l2error = scalar_field_out->sum(); // comm().sum(scalar_field_out);
  L2error = sqrt(L2error);
  l2error = sqrt(l2error);

  // libMesh::out << "\n \n L2error " << L2error << " \n l2error " << l2error;
  ////libMesh::out.flush();
  // return libmesh_make_unique<NumericVector<Number> >(scalar_field_out);
  return (scalar_field_out);
}

Real compute_error_scalar_weak(EquationSystems &equation_systems,
                               NumericVector<Number> &scalar_field_in,
                               NumericVector<Number> &scalar_field_out,
                               FunctionBase<Number> &exact_function) {
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  // auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  // p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  std::vector<dof_id_type> dof_indices;
  DenseVector<Number> Fe;
  Real L2error = 0, Linferror = 0, L1error = 0;
  // auto scalar_field_out = scalar_field_in.zero_clone();

  // for (const auto & elem : as_range(mesh.active_subdomain_elements_begin(1),
  // mesh.active_subdomain_elements_begin(1))) for (const auto & elem :
  // as_range(mesh.active_subdomain_elements_begin(0),
  // mesh.active_subdomain_elements_begin(0)))
  int count = 0;
  for (const auto &elem : mesh.element_ptr_range()) {
    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);
    if (elem->subdomain_id() == 0) {
      Real kernelsum = 0;

      for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {
        Real u = 0;
        for (unsigned int l = 0; l < n_dofs; l++)
          u += (phi_elem[l][qp] * exact_function(qpoint_elem[qp]) -
                scalar_field_in(l));

        for (unsigned int i = 0; i < n_dofs; i++) {
          Real val1 = JxW_elem[qp] * phi_elem[i][qp] * (1 * u);
          Real val2 = JxW_elem[qp] * phi_elem[i][qp] * (u * u);
          Real val3 = JxW_elem[qp] * phi_elem[i][qp] * abs(u);
          Fe(i) += val1;
          L1error += val3;
          L2error += val2;
          Linferror = std::max(Linferror, val3);
        }
      } // qp
    } else {
      //...
    }
    scalar_field_out.add_vector(Fe, dof_indices);
    count++;
  }

  Real l2error = scalar_field_out.sum(); // comm().sum(scalar_field_out);
  l2error = sqrt(l2error);

  L1error = abs(L2error / (double)count);
  L2error = sqrt(L2error) / (double)count;
  Linferror = abs(L2error / (double)count);

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();
  //   std::vector<double> errvec = {L1error, L2error, Linferror};
  return L2error;
}

Real compute_error_scalar_weak(EquationSystems &equation_systems,
                               NumericVector<Number> &scalar_field_in,
                               NumericVector<Number> &scalar_field_out,
                               FunctionBase<Number> &exact_function,
                               const std::string &results_prefix) {
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  // auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  // p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  // const std::vector<std::vector<RealGradient>> &dphi_elem =
  // fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  std::vector<dof_id_type> dof_indices;
  DenseVector<Number> Fe;
  Real L1error = 0;
  Real L2error = 0;
  Real Linferror = 0;
  Real vol = 0;
  Real vol_total = 0;
  int count = 0;

  // auto scalar_field_out = scalar_field_in.zero_clone();
  auto temp_scalar_field_out = scalar_field_in.zero_clone();
  system.project_vector(*temp_scalar_field_out, &exact_function,
                        libmesh_nullptr);
  auto temp_scalar_field_out2 = scalar_field_in.zero_clone();
  temp_scalar_field_out2->add(-1, scalar_field_in);

  // for (const auto & elem : as_range(mesh.active_subdomain_elements_begin(1),
  // mesh.active_subdomain_elements_begin(1))) for (const auto & elem :
  // as_range(mesh.active_subdomain_elements_begin(0),
  // mesh.active_subdomain_elements_begin(0)))
  for (const auto &elem : mesh.active_element_ptr_range()) {
    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);
    Real kernelsum = 0;

    count++;
    vol = 0;
    vol += elem->volume();
    vol_total += elem->volume();

    for (auto qp : range(n_dofs)) {

      Real u = 0;

      for (unsigned int l = 0; l < n_dofs; l++)
        u += JxW_elem[qp] * phi_elem[l][qp] *
             (temp_scalar_field_out2->el(dof_indices[l]));

      for (unsigned int l = 0; l < n_dofs; l++) {
        Real val1 = JxW_elem[qp] * (1 * u);
        Real val2 = JxW_elem[qp] * (u * u);
        Real val3 = JxW_elem[qp] * abs(u);
        L1error += val3;
        L2error += val2;
        Linferror = std::max(Linferror, val3);
        Fe(l) += JxW_elem[qp] * phi_elem[l][qp] * val1;
      }
    }

    scalar_field_out.add_vector(Fe, dof_indices);
  }
  //   Real l2error =
  //       scalar_field_out.dot(scalar_field_out); //
  //       comm().sum(scalar_field_out);
  //   l2error = sqrt(l2error);

  //   Real libmesh_l2error =
  //       scalar_field_out.l2_norm(); // comm().sum(scalar_field_out);

  //   // / (double)count
  //   L1error = abs(L1error) / vol_total;
  //   L2error = sqrt(L2error / vol_total);
  //   Linferror = abs(Linferror) / vol_total;

  //   L1error = scalar_field_out.l1_norm();
  //   L2error = scalar_field_out.l2_norm();
  //   Linferror = scalar_field_out.linfty_norm();

  //   L1error = scalar_field_out.l1_norm();
  //   L2error = scalar_field_out.l2_norm();
  //   Linferror = scalar_field_out.linfty_norm();

  //   L1error /= temp_scalar_field_out->l1_norm();
  //   L2error /= temp_scalar_field_out->l2_norm();
  //   Linferror /= temp_scalar_field_out->linfty_norm();

  // {
  //   libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
  //                << "\n l2error " << l2error << "\n libmesh_l2error "
  //                << libmesh_l2error << "\n Linferror " << Linferror;
  //   libMesh::out.flush();

  //   std::ofstream outfile;
  //   outfile.open("rhs_error_summary", std::ios_base::app);
  //   outfile << "\n " << results_prefix << " " << L1error << " " << L2error <<
  //   " "
  //           << Linferror << std::endl;
  // }
  L1error = abs(L1error);
  L2error = sqrt(L2error);
  Linferror = abs(Linferror);

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary_elemwise", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }

  L1error = scalar_field_out.l1_norm();
  L2error = scalar_field_out.l2_norm();
  Linferror = scalar_field_out.linfty_norm();

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }

  L1error /= temp_scalar_field_out->l1_norm();
  L2error /= temp_scalar_field_out->l2_norm();
  Linferror /= temp_scalar_field_out->linfty_norm();

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary_normalized", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }
  // std::vector<double> errvec = {L1error, L2error, Linferror};
  return L2error;
}

Real compute_error_scalar_strong(EquationSystems &equation_systems,
                                 NumericVector<Number> &scalar_field_in,
                                 NumericVector<Number> &scalar_field_out,
                                 FunctionBase<Number> &exact_function,
                                 const std::string &results_prefix) {
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  // auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  // p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  // const std::vector<std::vector<RealGradient>> &dphi_elem =
  // fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  std::vector<dof_id_type> dof_indices;
  DenseVector<Number> Fe;
  Real L1error = 0;
  Real L2error = 0;
  Real Linferror = 0;
  Real vol = 0;
  Real vol_total = 0;
  int count = 0;

  // auto scalar_field_out = scalar_field_in.zero_clone();
  auto temp_scalar_field_out = scalar_field_in.zero_clone();
  system.project_vector(*temp_scalar_field_out, &exact_function,
                        libmesh_nullptr);
  auto temp_scalar_field_out2 = scalar_field_in.zero_clone();
  temp_scalar_field_out2->add(-1, scalar_field_in);

  //   scalar_field_out.zero();
  //   system.project_vector(scalar_field_out, &exact_function,
  //   libmesh_nullptr); scalar_field_out.add( -1 , scalar_field_in);

  // for (const auto & elem : as_range(mesh.active_subdomain_elements_begin(1),
  // mesh.active_subdomain_elements_begin(1))) for (const auto & elem :
  // as_range(mesh.active_subdomain_elements_begin(0),
  // mesh.active_subdomain_elements_begin(0)))
  for (const auto &elem : mesh.active_element_ptr_range()) {
    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);
    Real kernelsum = 0;

    count++;
    vol = 0;
    vol += elem->volume();
    vol_total += elem->volume();

    // for (unsigned int qp : elem->node_index_range()) {
    Real u = 0;

    for (unsigned int l = 0; l < n_dofs; l++) {
      Real u = 0;
      u += (temp_scalar_field_out2->el(l));

      Real val1 = (1 * u);
      Real val2 = (u * u);
      Real val3 = abs(u);
      L1error += vol * val3;
      L2error += vol * val2;
      Linferror = std::max(Linferror, val3);
      Fe(l) += vol * val3; // abs(u);;
    }
    scalar_field_out.add_vector(Fe, dof_indices);
  }
  scalar_field_out.scale(1 / vol_total);
  //   Real l2error = scalar_field_out.dot(scalar_field_out); //
  //   comm().sum(scalar_field_out); l2error = sqrt(l2error);

  //   Real libmesh_l2error = scalar_field_out.l2_norm(); //
  //   comm().sum(scalar_field_out);

  L1error = abs(L1error) / vol_total;
  L2error = sqrt(L2error / vol_total);
  Linferror = abs(Linferror);

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary_elemwise", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }

  L1error = scalar_field_out.l1_norm();
  L2error = scalar_field_out.l2_norm();
  Linferror = scalar_field_out.linfty_norm();

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }

  L1error /= temp_scalar_field_out->l1_norm();
  L2error /= temp_scalar_field_out->l2_norm();
  Linferror /= temp_scalar_field_out->linfty_norm();

  libMesh::out << "\n \n L1error " << L1error << "\n L2error " << L2error
               << "\n Linferror " << Linferror;
  libMesh::out.flush();

  {
    std::ofstream outfile;
    outfile.open("rhs_error_summary_normalized", std::ios_base::app);
    outfile << "\n " << results_prefix << " " << L1error << " " << L2error
            << " " << Linferror << std::endl;
  }
  // std::vector<double> errvec = {L1error, L2error, Linferror};
  return L2error;
}

// std::unique_ptr<NumericVector<Number>>
//   std::unique_ptr<NumericVector<Number>> scalar_field_out =
//   scalar_field_in.zero_clone();
//     return scalar_field_out;

void set_results_filename(GetPot &input_file, std::string &results_prefix) {
  /* SET SAVE FILE SUFFIX */
  std::string delimiter("_");

  //   Utility::enum_to_string
  results_prefix += delimiter;
  auto _quad_enum = input_file("_quad_type", 0);
  results_prefix += std::to_string(_quad_enum);
  results_prefix += delimiter;
  results_prefix += std::to_string(input_file("p_order", 1));
  results_prefix += delimiter;
  //   std::string fe_family_string(input_file("fe_family", LAGRANGE));
  //   results_prefix += fe_family_string;
  //   results_prefix += delimiter;
  auto element_type_string = input_file("element_type", "QUAD8");
  results_prefix += element_type_string;
  results_prefix += delimiter;
  auto _quad_order_int = input_file("_quad_order", 1);
  results_prefix += std::to_string(_quad_order_int);
  results_prefix += delimiter;
  auto set_horizon = input_file("set_horizon", 1);
  results_prefix += std::to_string(set_horizon);
  results_prefix += delimiter;
  auto nelem = input_file("nelem", 1);
  results_prefix += std::to_string(nelem);
  // return results_prefix;
}

RealTensor outer_product(Point xi);
RealTensor outer_product(Point xi) {
  // Point xi = xq_nl - xq;
  Real knorm = xi.norm_sq();
  return (knorm < TOLERANCE)
             ? TensorValue<Real>(0., 0., 0., 0., 0., 0., 0., 0., 0.)
             : TensorValue<Real>(knorm * xi(0) * xi(0), knorm * xi(0) * xi(1),
                                 0., knorm * xi(0) * xi(1),
                                 knorm * xi(1) * xi(1), 0., 0.);
}

void integrate_scalar_field(
    EquationSystems &equation_systems, NumericVector<Number> &scalar_field_in,
    NumericVector<Number> &scalar_field_out,
    std::map<const Elem *, std::vector<const Elem *>> &map_elem_patch) {
  /* SETUP PARAMETERS */
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();

  // Get a reference to the ExplicitSystem we are solving
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  // Get some parameters that we need during assembly
  const Real penalty = equation_systems.parameters.get<Real>("penalty");
  const int kernel_type = equation_systems.parameters.get<int>("kernel_type");
  const Real kernelparam = equation_systems.parameters.get<Real>("kernelparam");
  const Real horizon = equation_systems.parameters.get<Real>("horizon");
  const QuadratureType _quad_type =
      equation_systems.parameters.get<QuadratureType>("_quad_type");
  const Order _quad_order =
      equation_systems.parameters.get<Order>("_quad_order");
  const Order extra_quadrature =
      equation_systems.parameters.set<Order>("extra_quadrature");
  const Order extra_quadrature_neighbor =
      equation_systems.parameters.set<Order>("_quad_order_neighbor");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  /* SETUP QUADRATURE*/
  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  const std::vector<std::vector<RealGradient>> &dphi_elem = fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  const std::vector<std::vector<RealTensor>> &d2phi_elem = fe_elem->get_d2phi();

  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices;
  for (auto elem : mesh.element_ptr_range()) {

    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);

    Real kernelsum = 0;
    auto &patch = map_elem_patch[elem];
    Number patchsize = patch.size();
    Real u = 0;
    for (unsigned int l = 0; l < n_dofs; l++)
      u += scalar_field_in(dof_indices[l]);

    DenseVector<Number> Fn(n_dofs);
    Real inner_integral_value = integrate_function(
        equation_systems, elem, 0, u, qpoint_elem, patch, scalar_field_in, Fn);
    for (unsigned int i = 0; i < n_dofs; i++)
      Fe(i) += (inner_integral_value);

    scalar_field_out.add_vector(Fe, dof_indices);

  } // elem

  scalar_field_out.close();
} // fcn

void composite_integrate_scalar_field(
    EquationSystems &equation_systems, NumericVector<Number> &scalar_field_in,
    NumericVector<Number> &scalar_field_out,
    std::map<const Elem *, std::vector<const Elem *>> &map_elem_patch) {
  /* SETUP PARAMETERS */
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();

  // Get a reference to the ExplicitSystem we are solving
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  // Get some parameters that we need during assembly
  const bool use_strong_form =
      equation_systems.parameters.get<bool>("use_strong_form");
  const Real penalty = equation_systems.parameters.get<Real>("penalty");
  const int kernel_type = equation_systems.parameters.get<int>("kernel_type");
  const Real kernelparam = equation_systems.parameters.get<Real>("kernelparam");
  const Real horizon = equation_systems.parameters.get<Real>("horizon");
  const QuadratureType _quad_type =
      equation_systems.parameters.get<QuadratureType>("_quad_type");
  const Order _quad_order =
      equation_systems.parameters.get<Order>("_quad_order");
  const Order extra_quadrature =
      equation_systems.parameters.set<Order>("extra_quadrature");
  const Order extra_quadrature_neighbor =
      equation_systems.parameters.set<Order>("_quad_order_neighbor");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  const std::vector<std::vector<RealGradient>> &dphi_elem = fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  const std::vector<std::vector<RealTensor>> &d2phi_elem = fe_elem->get_d2phi();
  // const std::vector<std::vector<RealGradient>> &dphi_elem =
  // fe_elem->get_dphi();

  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices;
  /* SETUP ELEMENT LOOP*/

  for (auto elem : mesh.element_ptr_range()) {

    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);

    Real kernelsum = 0;
    auto &patch = map_elem_patch[elem];
    Number patchsize = patch.size();
    if (use_strong_form) {
      Real u = 0;
      for (unsigned int l = 0; l < n_dofs; l++)
        u += scalar_field_in(dof_indices[l]);

      DenseVector<Number> Fn(n_dofs);
      Real inner_integral_value =
          integrate_function(equation_systems, elem, 0, u, qpoint_elem, patch,
                             scalar_field_in, Fn);
      for (unsigned int i = 0; i < n_dofs; i++)
        Fe(i) += (Fn(i) - 0);
      // Fe(i) += (inner_integral_value - u);
    } else {
      for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {
        Real u = 0;
        for (unsigned int l = 0; l < n_dofs; l++)
          u += phi_elem[l][qp] * scalar_field_in(dof_indices[l]);

        if (elem->subdomain_id() == 0 && horizon > 0 &&
            boundary_indicator(qpoint_elem[qp])) {

          DenseVector<Number> Fn(n_dofs);
          Real inner_integral_value =
              integrate_function(equation_systems, elem, qp, u, qpoint_elem,
                                 patch, scalar_field_in, Fn);
          for (unsigned int i = 0; i < n_dofs; i++)
            Fe(i) += JxW_elem[qp] * phi_elem[i][qp] * (Fn(i) - 0);
          // Fe(i) += JxW_elem[qp] * phi_elem[i][qp] * (inner_integral_value -
          // u); Fe(i) += JxW_elem[qp] * phi_elem[i][qp] * (Fn(i) );; Fe(i) +=
          // JxW_elem[qp] * Fn(i); Fe(i) += 1 * JxW_elem[qp] * phi_elem[i][qp] *
          // Fn(i);
          //   Fe(i) += 1 * JxW_elem[qp] * phi_elem[i][qp] *
          //   (inner_integral_value); Fe(i) += 1 * JxW_elem[qp] *
          //   phi_elem[i][qp] * (inner_integral_value - u); Fe(i) += 1 *
          //   JxW_elem[qp] * phi_elem[i][qp] * (Fn(i) - 0);
          //  inner_integral_value; // Fn(i); // * inner_integral_value
        }
        //   else {
        //     Gradient grad_u_old(0., 0., 0.);
        //     for (unsigned int l = 0; l < n_dofs; l++) {
        //       // for (std::size_t d = 0; d < ndimensions; d++)
        //       grad_u_old.add_scaled(dphi_elem[l][qp],
        //                             scalar_field_in(dof_indices[l]));
        //     }

        //     for (unsigned int i = 0; i != n_dofs; i++) {
        //       Fe(i) += JxW_elem[qp] * grad_u_old * dphi_elem[i][qp];
        //     }
        //   }

      } // qp
    }

    // dof_map.constrain_element_vector(Fe, dof_indices);
    /* OUT */

    scalar_field_out.add_vector(Fe, dof_indices);

  } // elem

  scalar_field_out.close();
  // scalar_field_out->print_matlab(std::string("approx_nonlocal_rhs.mat"));

} // fcn

void compare_integrate_scalar_field(
    EquationSystems &equation_systems, NumericVector<Number> &scalar_field_in,
    NumericVector<Number> &scalar_field_out,
    std::map<const Elem *, std::vector<const Elem *>> &map_elem_patch) {
  /* SETUP PARAMETERS */
  // {
  // EquationSystems & es = this->get_equation_systems();
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();

  // Get a reference to the ExplicitSystem we are solving
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  // Get some parameters that we need during assembly
  const Real penalty = equation_systems.parameters.get<Real>("penalty");
  std::string refinement_type =
      equation_systems.parameters.get<std::string>("refinement");
  const unsigned int target_patch_size =
      equation_systems.parameters.get<unsigned int>("target_patch_size");
  const unsigned int target_patch_type =
      equation_systems.parameters.get<unsigned int>("target_patch_type");
  const int kernel_type = equation_systems.parameters.get<int>("kernel_type");
  const Real kernelparam = equation_systems.parameters.get<Real>("kernelparam");
  const Real horizon = equation_systems.parameters.get<Real>("horizon");
  const QuadratureType _quad_type =
      equation_systems.parameters.get<QuadratureType>("_quad_type");
  const Order _quad_order =
      equation_systems.parameters.get<Order>("_quad_order");
  const Order extra_quadrature =
      equation_systems.parameters.set<Order>("extra_quadrature");
  const Order extra_quadrature_neighbor =
      equation_systems.parameters.set<Order>("extra_quadrature_neighbor");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  const bool scale_by_xiarg =
      equation_systems.parameters.get<bool>("scale_by_xiarg");
  const bool scale_by_xinorm =
      equation_systems.parameters.get<bool>("scale_by_xinorm");
  const bool scale_by_neighborvolume =
      equation_systems.parameters.get<bool>("scale_by_neighborvolume");
  const bool scale_by_patchsize =
      equation_systems.parameters.get<bool>("scale_by_patchsize");
  const bool scale_by_kernelsum =
      equation_systems.parameters.get<bool>("scale_by_kernelsum");
  const bool use_jxw_neighbor =
      equation_systems.parameters.get<bool>("use_jxw_neighbor");
  const bool use_strong_form =
      equation_systems.parameters.get<bool>("use_strong_form");
  const bool composite_quad =
      equation_systems.parameters.get<bool>("composite_quad");
  const bool print_xi_kernel =
      equation_systems.parameters.get<bool>("print_xi_kernel");
  const bool apply_exact_rhs_boundary =
      equation_systems.parameters.get<bool>("apply_exact_rhs_boundary");

  // }
  /* SETUP QUADRATURE*/
  // std::unique_ptr<QBase> qrule_ptr(QBase::build(_quad_type, ndimensions,
  // _quad_order));

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  const std::vector<std::vector<RealGradient>> &dphi_elem = fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();

  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices;
  /* SETUP ELEMENT LOOP*/

  for (auto elem : mesh.element_ptr_range()) {

    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);

    Real kernelsum = 0;
    auto &patch = map_elem_patch[elem];
    Number patchsize = patch.size();
    for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {

      Real u = 0;
      for (unsigned int l = 0; l < n_dofs; l++)
        u += phi_elem[l][qp] * scalar_field_in(dof_indices[l]);

      Gradient grad_u_old(0., 0., 0.);
      for (unsigned int l = 0; l < n_dofs; l++)
        grad_u_old.add_scaled(dphi_elem[l][qp],
                              scalar_field_in(dof_indices[l]));

      for (unsigned int i = 0; i < n_dofs; i++)
        Fe(i) += 1 * JxW_elem[qp] * dphi_elem[i][qp] * grad_u_old;

      if (horizon > 0) {
        DenseVector<Number> Fn(n_dofs);
        Real inner_integral_value =
            integrate_function(equation_systems, elem, qp, u, qpoint_elem,
                               patch, scalar_field_in, Fn);
        for (unsigned int i = 0; i < n_dofs; i++)
          Fe(i) += 1 * JxW_elem[qp] * phi_elem[i][qp] * (Fn(i) - 0);
      }

    } // qp
    dof_map.constrain_element_vector(Fe, dof_indices);
    /* OUT */

    scalar_field_out.add_vector(Fe, dof_indices);

  } // elem

  scalar_field_out.close();
  // scalar_field_out->print_matlab(std::string("approx_nonlocal_rhs.mat"));

} // fcn

Real signed_distance(Point p, Real eps) {
  Real D_oo = 0.0;
  for (int d = 0; d < p.size(); ++d) {
    D_oo = std::max(D_oo, std::abs(p(d)));
  }
  return D_oo - eps;
}

// quadratic_form(const RealVectorValue & xi,  const RealVectorValue & u)
Real integrate_function(EquationSystems &equation_systems, const Elem *elem,
                        int qp, Real u, const std::vector<Point> &qpoint_elem,
                        std::vector<const Elem *> patch,
                        NumericVector<Number> &scalar_field_in,
                        DenseVector<Number> &int_val) {
  /* SETUP PARAMETERS*/
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");

  const Real horizon = equation_systems.parameters.get<Real>("horizon");
  const QuadratureType _quad_type =
      equation_systems.parameters.get<QuadratureType>("_quad_type");
  auto _quad_enum = equation_systems.parameters.set<unsigned int>("_quad_enum");
  const Order _quad_order =
      equation_systems.parameters.get<Order>("_quad_order");
  const Order extra_quadrature =
      equation_systems.parameters.set<Order>("extra_quadrature");
  const Order extra_quadrature_neighbor =
      equation_systems.parameters.set<Order>("extra_quadrature_neighbor");
  auto p_order = equation_systems.parameters.get<Order>("p_order");
  auto fe_family = equation_systems.parameters.get<FEFamily>("fe_family");
  auto use_strong_form =
      equation_systems.parameters.get<bool>("use_strong_form");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);
  // }

  /* SETUP QUADRATURE*/

  std::vector<Real> vertex_distance;
  QComposite<QTrap> qrule(ndimensions, extra_quadrature_neighbor);
  std::unique_ptr<FEBase> fe(
      FEBase::build(mesh.mesh_dimension(), FEType(p_order, fe_family)));
  const std::vector<Point> &qpoint_neighbor = fe->get_xyz();
  const std::vector<std::vector<Real>> &phi_neighbor = fe->get_phi();
  const std::vector<Real> &JxW_neighbor = fe->get_JxW();

  /* SETUP PATCH LOOP*/
  Real patch_volume = 0;
  const Point pmin(0, 0, 0);
  const Point pmax(1, 1, 1);
  libMesh::BoundingBox physical_boundingbox(pmin, pmax);

  // DenseVector<Number> Fn;
  Number patchsize = patch.size();
  Real value = 0;
  for (const auto &neighbor : patch) {

    if (physical_boundingbox.contains_point(neighbor->centroid())) {
      libmesh_assert_not_equal_to(neighbor->id(), elem->id());

      Real neighbor_volume = neighbor->volume();
      neighbor_volume = sqrt(neighbor_volume);
      patch_volume += neighbor_volume;

      vertex_distance.clear();

      std::vector<dof_id_type> neighbor_dof_indices;
      dof_map.dof_indices(neighbor, neighbor_dof_indices);
      const unsigned int n_neighbor_dofs = neighbor_dof_indices.size();

      for (unsigned int v = 0; v < elem->n_vertices(); v++) {
        Real dist = 0;
        if (use_strong_form) {
          dist += signed_distance(neighbor->point(v), horizon);
          //   dist += signed_distance(neighbor->point(v) - elem->centroid(),
          //                          horizon, ndimensions);
        } else {
          dist += signed_distance(neighbor->point(v) - qpoint_elem[qp], horizon);
        }
        vertex_distance.push_back(dist);
      }

      /* BEGIN MUTE */
      std::streambuf *cout_sbuf = std::cout.rdbuf(); // save original sbuf
      std::ofstream fout("/dev/null");
      std::cout.rdbuf(fout.rdbuf()); // redirect 'cout' to a 'fout'

      qrule.init(*neighbor, vertex_distance, 0);

      // /* END MUTE */
      std::cout.rdbuf(cout_sbuf); // restore the original stream buffer

      fe->reinit(neighbor, &(qrule.get_points()), &(qrule.get_weights()));

      for (std::size_t qp2 = 0; qp2 < qpoint_neighbor.size(); qp2++) {
        Point xi;
        if (use_strong_form)
          xi.assign(qpoint_neighbor[qp2] - elem->centroid());

        else
          xi.assign(qpoint_neighbor[qp2] - qpoint_elem[qp]);

        Real u_neighbor = 0;
        for (unsigned int l = 0; l < n_neighbor_dofs; l++) {
          u_neighbor +=
              phi_neighbor[l][qp2] * scalar_field_in(neighbor_dof_indices[l]);
        }

        Real kernelval = 0;
        kernelval += skernel(xi, equation_systems.parameters, std::string(),
                             std::string());
        // if (kernel_function.get())
        //     kernelval += (*kernel_function)(xi)
        // boost::variant<Number, VectorValue<Number>, TensorValue<Number>>
        // kernelval = kernel(xi , equation_systems.parameters, std::string(),
        // std::string()); kernelval += boost::get<Real> ( kernelval
        //     ;

        // if (neighbor->subdomain_id() != 0)
        //     u_neighbor = 0;

        for (unsigned int l = 0; l < n_neighbor_dofs; l++) {
          Real temp = JxW_neighbor[qp2] * phi_neighbor[l][qp2] * kernelval *
                      (u_neighbor);
          int_val(l) += temp;
          value += temp;
        }

      } // qp2
    }
  }             // patch
  return value; // int_val
}

void Apply_Laplacian(EquationSystems &equation_systems,
                     NumericVector<Number> &scalar_field_in,
                     std::unique_ptr<NumericVector<Number>> &scalar_field_out) {
  /* SETUP ASSEMBLE */
  // {
  // EquationSystems & es = this->get_equation_systems();
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();

  // Get a reference to the ExplicitSystem we are solving
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  // Get some parameters that we need during assembly
  const Real penalty = equation_systems.parameters.get<Real>("penalty");
  std::string refinement_type =
      equation_systems.parameters.get<std::string>("refinement");
  const unsigned int target_patch_size =
      equation_systems.parameters.get<unsigned int>("target_patch_size");
  const unsigned int target_patch_type =
      equation_systems.parameters.get<unsigned int>("target_patch_type");
  const int kernel_type = equation_systems.parameters.get<int>("kernel_type");
  const Real kernelparam = equation_systems.parameters.get<Real>("kernelparam");
  const Real horizon = equation_systems.parameters.get<Real>("horizon");
  const QuadratureType _quad_type =
      equation_systems.parameters.get<QuadratureType>("_quad_type");
  const Order _quad_order =
      equation_systems.parameters.get<Order>("_quad_order");
  const Order extra_quadrature =
      equation_systems.parameters.get<Order>("extra_quadrature");
  const Order extra_quadrature_neighbor =
      equation_systems.parameters.get<Order>("extra_quadrature_neighbor");

  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  std::unique_ptr<FEBase> fe(FEBase::build(ndimensions, fe_type));
  fe->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi = fe->get_phi();
  // const std::vector<std::vector<RealGradient>> &dphi = fe->get_dphi();
  const std::vector<Real> &JxW = fe->get_JxW();
  const std::vector<Point> &qpoint = fe->get_xyz();
  const std::vector<std::vector<RealTensor>> &d2phi = fe->get_d2phi();
  // }

  DenseVector<Number> Fe;
  std::vector<dof_id_type> dof_indices;
  for (const auto &elem : mesh.active_element_ptr_range()) {
    fe->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);

    /* LOCAL */
    // shape_laplacian.zero();
    // shape_laplacian.resize(n_dofs);
    for (unsigned int qp = 0; qp < qrule->n_points(); qp++) {
      Real d2u = 0;
      for (int l = 0; l < n_dofs; ++l)
        for (int d = 0; d < ndimensions; ++d)
          d2u += d2phi[l][qp](d, d) * scalar_field_in(dof_indices[l]);

      // apply to field
      for (unsigned int i = 0; i < n_dofs; i++)
        Fe(i) += JxW[qp] * (phi[i][qp] * d2u);
    }
    dof_map.constrain_element_vector(Fe, dof_indices);
    scalar_field_out->add_vector(Fe, dof_indices);
  } // elem
  // scalar_field_out->print();
  scalar_field_out->close();
} // fcn

void init_patch_growth(
    MeshBase &mesh,
    std::map<const Elem *, std::vector<const Elem *>> &map_elem_patch,
    Real horizon) {
  Real horizon2 = horizon * horizon;
  for (const auto &elem : mesh.active_local_element_ptr_range()) {

    libMesh::Patch patch(mesh.processor_id());
    unsigned int target_patch_size = 9;
    int nlocalelem = mesh.n_local_elem();
    // Real dist = 0;

    patch.build_around_element(elem, target_patch_size,
                               &Patch::add_point_neighbors);
    int patchsize = patch.size();

    /* active set of patch elements */
    std::set<const Elem *> newpatch(patch.begin(), patch.end());
    bool grow_patch = 1;
    while (grow_patch) {
      grow_patch = 0;
      for (const auto &neighbor : newpatch) {
        // std::max(dist, (neighbor->centroid() - elem->centroid()).norm_sq());
        Real dist = 0;
        for (const auto neighbor_node : neighbor->node_ref_range())
        {

          for (const auto node : neighbor->node_ref_range()) {
            dist = signed_distance(neighbor_node - node, horizon);
            if (dist <= horizon) {
              for (const auto &neighbor2 : neighbor->neighbor_ptr_range()) {
                if (neighbor2 != nullptr && patch.count(neighbor2) == 0) {
                  newpatch.insert(neighbor2);
                  patch.insert(neighbor2);
                  grow_patch = 1;
                }
              }
            }
          }
        }
      newpatch.erase(neighbor);
      }

    }
    std::vector<const Elem *> elemvec(patch.begin(), patch.end());
    map_elem_patch.insert(
        std::pair<Elem *, std::vector<const Elem *>>(elem, elemvec));
  }
}


std::unique_ptr<NumericVector<Number>> integrate_scalar_strong(
    EquationSystems &equation_systems,
    const NumericVector<Number> &scalar_field_in,
    std::map<const Elem *, std::vector<const Elem *>> &map_elem_patch) {
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  std::unique_ptr<NumericVector<Number>> scalar_field_out =
      scalar_field_in.zero_clone();

  // Get some parameters that we need during assembly
  //   auto horizon = equation_systems.parameters.set<Order>("horizon");// =
  //   p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  std::unique_ptr<FEBase> fe_elem(FEBase::build(ndimensions, fe_type));
  std::unique_ptr<QBase> qrule(
      QBase::build(_quad_type, ndimensions, _quad_order));
  fe_elem->attach_quadrature_rule(qrule.get());
  // std::unique_ptr<FEBase> fe_face(FEBase::build(ndimensions-1, fe_type));
  const std::vector<std::vector<Real>> &phi_elem = fe_elem->get_phi();
  const std::vector<std::vector<RealGradient>> &dphi_elem = fe_elem->get_dphi();
  const std::vector<Real> &JxW_elem = fe_elem->get_JxW();
  const std::vector<Point> &qpoint_elem = fe_elem->get_xyz();
  const std::vector<std::vector<RealTensor>> &d2phi_elem = fe_elem->get_d2phi();
  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices;

  for (auto elem : mesh.element_ptr_range()) {

    fe_elem->reinit(elem);
    dof_map.dof_indices(elem, dof_indices);
    const unsigned int n_dofs = dof_indices.size();
    Fe.zero();
    Fe.resize(n_dofs);

    Real kernelsum = 0;
    auto &patch = map_elem_patch[elem];
    Number patchsize = patch.size();

    // Real u=0;
    // for (unsigned int l = 0; l < n_dofs; l++)
    //     u +=  scalar_field_in(dof_indices[l]);

    DenseVector<Number> Fn(n_dofs);
    Real inner_integral_value = patch_integrate_scalar_strong(
        equation_systems, elem, patch, scalar_field_in, Fn);
    // for (unsigned int i = 0; i < n_dofs; i++)
    for (auto i : range(n_dofs)) {
      Real u = 0;
      u += scalar_field_in(dof_indices[i]);
      Fe(i) += (Fn(i) - 0);
    }
    // Fe(i) += (inner_integral_value - u);

    // dof_map.constrain_element_vector(Fe, dof_indices);
    scalar_field_out->add_vector(Fe, dof_indices);

  } // elem

  scalar_field_out->close();
  // scalar_field_out->print_matlab(std::string("approx_nonlocal_rhs.mat"));
  // return std::make_unique< NumericVector<Number> >(&scalar_field_out);
  return scalar_field_out;
  // return std::make_unique(&scalar_field_out);
  // return std::make_unique< libMesh::NumericVector<Number>
  // >(scalar_field_out);
}

void find_mesh_quadpoints_for_elem(EquationSystems &equation_systems,
                                   const Elem *elem,
                                   const std::set<const Point *> qpointset,
                                   std::set<const Elem *> &elempatch,
                                   std::set<const Point *> &pointpatch) {
  const MeshBase &mesh = equation_systems.get_mesh();

  std::unique_ptr<PointLocatorBase> locator = mesh.sub_point_locator();

  //  std::set<const Node *> nodepatch;
  // elempatch.reserve(27);

  for (const auto &epoint : elem->node_ref_range()) {
    for (const auto &qpoint : qpointset) {
      Point eqpoint(epoint + *qpoint);

      const Elem *top_elem = (*locator)(eqpoint);
      if (top_elem == libmesh_nullptr)
        1;
      else {
        elempatch.insert(top_elem);
        // pointpatch.insert(eqpoint);
      }
    }
  }
}

// quadratic_form(const RealVectorValue & xi,  const RealVectorValue & u)
Real patch_integrate_scalar_strong(EquationSystems &equation_systems,
                                   const Elem *elem,
                                   std::vector<const Elem *> patch,
                                   const NumericVector<Number> &scalar_field_in,
                                   DenseVector<Number> &int_val) {
  /* SETUP PARAMETERS*/
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  auto horizon = equation_systems.parameters.set<Real>("horizon"); // = p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  // }

  /* SETUP QUADRATURE*/
  // QComposite<QTrap> qrule(ndimensions, _quad_order_neighbor);
  QComposite<QGauss> qrule(ndimensions, _quad_order_neighbor);

  std::vector<Real> vertex_distance;
  std::unique_ptr<FEBase> fe(
      FEBase::build(mesh.mesh_dimension(), FEType(p_order, fe_family)));
  const std::vector<Point> &qpoint_neighbor = fe->get_xyz();
  const std::vector<std::vector<Real>> &phi_neighbor = fe->get_phi();
  const std::vector<Real> &JxW_neighbor = fe->get_JxW();

  /* SETUP PATCH LOOP*/
  Real patch_volume = 0;

  // DenseVector<Number> Fn;
  std::vector<dof_id_type> dof_indices;
  dof_map.dof_indices(elem, dof_indices);
  const unsigned int n_dofs = dof_indices.size();
  libMesh::BoundingBox physical_boundingbox(Point(0 + horizon, 0 + horizon, 0),
                                            Point(1 - horizon, 1 - horizon, 0));

  Real value = 0;
  // for (unsigned int qp = 0; qp < elem->n_vertices(); qp++)
  // for (auto qp : range(elem->n_vertices()))
  for (auto qp : range(n_dofs)) {
    const Point &point_elem = elem->point(qp);
    // libMesh::out << " \n " << " elem " << elem->id();
    // point_elem.print();
    if (physical_boundingbox.contains_point(elem->centroid()))
      for (const auto &neighbor : patch) {
        if (physical_boundingbox.contains_point(neighbor->centroid()) &&
            neighbor->subdomain_id() == 0) {
          // libmesh_assert_not_equal_to(neighbor->id(), elem->id());

          Real neighbor_volume = neighbor->volume();
          neighbor_volume = sqrt(neighbor_volume);
          patch_volume += neighbor_volume;

          vertex_distance.clear();

          std::vector<dof_id_type> neighbor_dof_indices;
          dof_map.dof_indices(neighbor, neighbor_dof_indices);
          const unsigned int n_neighbor_dofs = neighbor_dof_indices.size();
          // for (auto v : range(neighbor->n_vertices()))
          for (auto v : range(n_neighbor_dofs)) {
            Real dist = 0;
            const Point &point_neighbor = neighbor->point(v);
            // libMesh::out << " \n " << " elem " << elem->id();
            //     point_neighbor.print();
            // point_elem.print();

            dist += signed_distance(point_neighbor - point_elem, horizon);
            vertex_distance.push_back(dist);
          }

          std::streambuf *cout_sbuf = std::cout.rdbuf();
          std::ofstream fout("/dev/null");
          std::cout.rdbuf(fout.rdbuf());
          libMesh::out << " patchsize " << patch.size();
          qrule.init(*neighbor, vertex_distance, 0);
          std::cout.rdbuf(cout_sbuf);
          libMesh::out.flush();

          fe->reinit(neighbor, &(qrule.get_points()), &(qrule.get_weights()));

          for (std::size_t qp2 = 0; qp2 < qpoint_neighbor.size(); qp2++) {
            Point xi;
            xi.assign(qpoint_neighbor[qp2] - point_elem);

            Real kernelval = 0;
            kernelval += skernel(xi, equation_systems.parameters, std::string(),
                                 std::string());
            /*
            libMesh::out<< " \n " << " XI ";
            xi.print();
            libMesh::out<< " \n " << " XINORM " <<  xi.norm();
            libMesh::out<< " \n " << " kernelval " <<  kernelval;
            //  << std::endl;
            libMesh::out.flush();
            */
            Real u = 0;
            u += scalar_field_in(dof_indices[qp]);

            Real u_neighbor = 0;
            for (unsigned int l = 0; l < n_neighbor_dofs; l++)
              u_neighbor += phi_neighbor[l][qp2] *
                            scalar_field_in(neighbor_dof_indices[l]);

            for (unsigned int l = 0; l < n_neighbor_dofs; l++) {
              Real temp = 0;
              temp += JxW_neighbor[qp2] * kernelval * (u_neighbor);
              // Real temp = JxW_neighbor[qp2] * phi_neighbor[l][qp2] *
              // kernelval * (u_neighbor);
              int_val(l) += temp;
              value += temp;
            }

          } // qp2
        }   // bbox
        else {
          // libMesh::out << " INTERACTION DOMAIN ELEMENT ";
          // libMesh::out.flush();
        }
      } // patch
  }
  return value; // int_val
}

// quadratic_form(const RealVectorValue & xi,  const RealVectorValue & u)
Real patch_composite_integrate_scalar_strong(
    EquationSystems &equation_systems, const Elem *elem,
    std::vector<const Elem *> patch,
    const NumericVector<Number> &scalar_field_in,
    DenseVector<Number> &int_val) {
  /* SETUP PARAMETERS*/
  const MeshBase &mesh = equation_systems.get_mesh();
  const unsigned int ndimensions = mesh.mesh_dimension();
  ExplicitSystem &system =
      equation_systems.get_system<ExplicitSystem>("system");
  const DofMap &dof_map = system.get_dof_map();
  FEType fe_type = system.variable_type(0);

  auto horizon = equation_systems.parameters.set<Real>("horizon"); // = p_order;
  auto p_order =
      equation_systems.parameters.set<Order>("p_order"); // = p_order;
  auto fe_family =
      equation_systems.parameters.set<FEFamily>("fe_family"); //  = fe_family;
  auto _quad_type = equation_systems.parameters.set<QuadratureType>(
      "_quad_type"); //  = _quad_type;
  auto _quad_order =
      equation_systems.parameters.set<Order>("_quad_order"); //  = _quad_order;
  auto _quad_order_neighbor = equation_systems.parameters.set<Order>(
      "_quad_order_neighbor"); //  = _quad_order;

  // }

  /* SETUP QUADRATURE*/
  // QComposite<QTrap> qrule(ndimensions, _quad_order_neighbor);
  QComposite<QGauss> qrule(ndimensions, _quad_order_neighbor);

  std::vector<Real> vertex_distance;
  std::unique_ptr<FEBase> fe(
      FEBase::build(mesh.mesh_dimension(), FEType(p_order, fe_family)));
  const std::vector<Point> &qpoint_neighbor = fe->get_xyz();
  const std::vector<std::vector<Real>> &phi_neighbor = fe->get_phi();
  const std::vector<Real> &JxW_neighbor = fe->get_JxW();

  /* SETUP PATCH LOOP*/
  Real patch_volume = 0;

  // DenseVector<Number> Fn;
  std::vector<dof_id_type> dof_indices;
  dof_map.dof_indices(elem, dof_indices);
  const unsigned int n_dofs = dof_indices.size();
  libMesh::BoundingBox physical_boundingbox(Point(0 + horizon, 0 + horizon, 0),
                                            Point(1 - horizon, 1 - horizon, 0));

  Real value = 0;
  // for (unsigned int qp = 0; qp < elem->n_vertices(); qp++)
  // for (auto qp : range(elem->n_vertices())){
  for (auto qp : range(n_dofs)) {
    const Point &point_elem = elem->point(qp);
    // libMesh::out << " \n " << " elem " << elem->id();
    // point_elem.print();
    if (physical_boundingbox.contains_point(elem->centroid()))
      for (const auto &neighbor : patch) {
        if (physical_boundingbox.contains_point(neighbor->centroid()) &&
            neighbor->subdomain_id() == 0) {
          // libmesh_assert_not_equal_to(neighbor->id(), elem->id());

          Real neighbor_volume = neighbor->volume();
          neighbor_volume = sqrt(neighbor_volume);
          patch_volume += neighbor_volume;

          vertex_distance.clear();

          std::vector<dof_id_type> neighbor_dof_indices;
          dof_map.dof_indices(neighbor, neighbor_dof_indices);
          const unsigned int n_neighbor_dofs = neighbor_dof_indices.size();
          // for (auto v : range(neighbor->n_vertices()))
          for (auto v : range(n_neighbor_dofs)) {
            Real dist = 0;
            const Point &point_neighbor = neighbor->point(v);
            // libMesh::out << " \n " << " elem " << elem->id();
            //     point_neighbor.print();
            // point_elem.print();

            dist += signed_distance(point_neighbor - point_elem, horizon);
            vertex_distance.push_back(dist);
          }

          std::streambuf *cout_sbuf = std::cout.rdbuf();
          std::ofstream fout("/dev/null");
          std::cout.rdbuf(fout.rdbuf());
          libMesh::out << " patchsize " << patch.size();
          qrule.init(*neighbor, vertex_distance, 0);
          std::cout.rdbuf(cout_sbuf);
          libMesh::out.flush();

          fe->reinit(neighbor, &(qrule.get_points()), &(qrule.get_weights()));

          for (std::size_t qp2 = 0; qp2 < qpoint_neighbor.size(); qp2++) {
            Point xi;
            xi.assign(qpoint_neighbor[qp2] - point_elem);

            Real kernelval = 0;
            kernelval += skernel(xi, equation_systems.parameters, std::string(),
                                 std::string());
            /*
            libMesh::out<< " \n " << " XI ";
            xi.print();
            libMesh::out<< " \n " << " XINORM " <<  xi.norm();
            libMesh::out<< " \n " << " kernelval " <<  kernelval;
            //  << std::endl;
            libMesh::out.flush();
            */
            Real u = 0;
            u += scalar_field_in(dof_indices[qp]);

            Real u_neighbor = 0;
            for (unsigned int l = 0; l < n_neighbor_dofs; l++)
              u_neighbor += phi_neighbor[l][qp2] *
                            scalar_field_in(neighbor_dof_indices[l]);

            for (unsigned int l = 0; l < n_neighbor_dofs; l++) {
              Real temp = JxW_neighbor[qp2] * phi_neighbor[l][qp2] * kernelval *
                          (u_neighbor);
              int_val(l) += temp;
              value += temp;
            }

          } // qp2
        }   // bbox
        else {
          // libMesh::out << " INTERACTION DOMAIN ELEMENT ";
          // libMesh::out.flush();
        }
      } // patch
  }
  return value; // int_val
}
