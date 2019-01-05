#include "helpers.h"
// #include "kernel.h"

using namespace libMesh;

// typedef std::map<const Elem *, std::vector<const Elem *>> ElemPatchMap;
// typedef std::map<const Elem *, std::vector<const Node *>> ElemNodeMap;
// typedef std::map<Node, std::vector<const Node *>> NodePatchMap;
// typedef std::map<Node, std::vector<const Elem *>> NodeElemMap;

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
/* 
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
 */


double signed_distance_fcn(const Point &p);
double signed_distance_fcn(const Point &p)
{
  double D_oo = 0.0;
  for (int d = 0; d < p.size(); ++d)
    D_oo = std::max(D_oo, std::abs(p(d)));
  return 1 - D_oo;
}

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
define_meshfunction(EquationSystems &equation_systems)
{
    auto &system = equation_systems.get_system("system");
    std::vector<unsigned int> variables;
    system.get_all_variable_numbers(variables);
    std::unique_ptr<NumericVector<Number>> mesh_function_vector =
        NumericVector<Number>::build(equation_systems.comm());
    mesh_function_vector->init(system.n_dofs(), false, SERIAL);
    system.solution->localize(*mesh_function_vector);

    MeshFunction mesh_function(equation_systems, *mesh_function_vector,
                               system.get_dof_map(), variables);
    return std::make_unique<MeshFunction>(mesh_function);
    // mesh_function.init();
}

std::unique_ptr<NumericVector<Number>>
integrate_scalar_exact(EquationSystems &equation_systems,
                       FunctionBase<Number> &exact_solution);

void print_quadrule(EquationSystems &equation_systems);

template <typename... Args>
std::vector<std::string> toStringVector(Args... args)
{
    std::vector<std::string> result;
    auto initList = {args...};
    using T = typename decltype(initList)::value_type;
    std::vector<T> expanded{initList};
    result.resize(expanded.size());
    std::transform(expanded.begin(), expanded.end(), result.begin(), [](T value) { return std::to_string(value); });
    return result;
}

template <typename T, typename... Args>
void push_back_vec(std::vector<T> &v, Args &&... args)
{
    (v.push_back(args), ...);
}

template <typename... Args>
void FoldWrite(
    std::ofstream &outfile,
    std::string &savename,
    Args &&... args)
{
    outfile.open(savename, std::ios_base::app);
    (outfile << ... << std::forward<Args>(args)) << "\n";
}

template <typename... Args>
void FoldWrite(
    std::ofstream &outfile,
    Args &&... args)
{
    (outfile << ... << std::forward<Args>(args)) << "\n";
}

template <typename... Args>
void FoldPrint(Args &&... args)
{
    (std::cout << ... << std::forward<Args>(args)) << "\n";
}

template <typename T, typename... Args>
void FoldPushBack(std::vector<T> &v, Args &&... args)
{
    (v.push_back(args), ...);
}

Number
scalar_boundary_indicator(const Real &x)
{
    return abs(x) > 0.5 ? 0 : 1;
    // return ( abs(x) > 0.5 || x < ) ? 0 : 1;
}

Number
boundary_indicator(const Point &p)
{
    return scalar_boundary_indicator(p(0)) * scalar_boundary_indicator(p(1));
}

Number
exact_solution(const Point &p, const Parameters &parameters,
               const std::string &, const std::string &)
{
    Real val = 0;
    Real x = 0, y = 0;
    x += p(0);
    y += p(1);
    val += 0.25 * (1. + cos(M_PIl * 2*x) ) * (1. +  cos( M_PIl * 2*y) );
    return val * boundary_indicator(p);
}

// VectorValue<Number>
Gradient
exact_gradient(const Point &p, const Parameters &parameters, const std::string &, const std::string &)
{
    const auto horizon = parameters.get<Real>("horizon");
    const auto ndimensions = parameters.get<unsigned int>("ndimensions");
    Real temp = 0;
    temp += std::pow(horizon, ndimensions);
    Real x = 0; x+= p(0);
    Real y = 0; y+= p(1);
    
    Real cy = cos(M_PIl * y);
    Real cx = cos(M_PIl * x);
    Real s2y = sin(M_PIl * 2*y);
    Real s2x = sin(M_PIl * 2*x);
    


    // Real cx = cos(M_PIl * p(0)), cy = cos(M_PIl * p(1)), cz = cos(M_PIl * p(2));
    // Real sx = sin(M_PIl * p(0)), sy = sin(M_PIl * p(1)), sz = sin(M_PIl * p(2));

    // VectorValue<Number>
    // Gradient gradvec(cx * sy * sz / temp, sx * cy * sz / temp, sx * sy * cz / temp);

    // Gradient gradvec(sx * cy / temp, cx * sy / temp, 0.);

    return Gradient(-M_PIl * cy*cy * s2x , -M_PIl * cx * cx * s2y );
    // VectorValue<Number> kernelvec( cos(M_PIl*x) * sin(M_PIl*y) * sin(M_PIl*z), cos(M_PIl*y) * sin(M_PIl*x) * sin(M_PIl*z), cos(M_PIl*z) * sin(M_PIl*x) * sin(M_PIl*y) );
}

Number
laplacian(const Point &p, const Parameters &parameters, const std::string &,
          const std::string &)
{
    Real val = 0;

    const std::string func_string = parameters.get<std::string>("laplacian_string");
    val = -2 * M_PIl * M_PIl * sin(M_PIl * p(0)) * sin(M_PIl * p(1));

    return val * boundary_indicator(p);
}


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
Number kernel_piecewise_linear(Real x, const Parameters &parameters)
{
    Number val = 0;

    if (abs(x) > 1)
        1;
    else
        val += 1 - abs(x);

    return val;
}


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
