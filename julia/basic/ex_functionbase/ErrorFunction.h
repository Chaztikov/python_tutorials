// libMesh Includes
#include "libmesh/auto_ptr.h" // libmesh_make_unique
#include "libmesh/function_base.h"
#include "libmesh/meshfree_interpolation.h"
#include "libmesh/threads.h"

// C++ includes
#include <cstddef>

namespace libMesh {

// Forward Declarations
template <typename T> class DenseVector;

// ------------------------------------------------------------
// ErrorFunction class definition
class ErrorFunction : public FunctionBase<Number> {
private:
  const Number &_fapprox;
  const Number &_fexact;
  mutable std::vector<Point> _pts;
  mutable std::vector<Number> _vals;
  // Threads::spin_mutex &_mutex;

public:
  /**
   * Constructor.  Requires a MeshlessInterpolation object.
   */
  ErrorFunction(  const Number &_fapprox,
                  const Number &_fexact)
      : _funcapprox(_fapprox), _funcexact(_fexact) {}

  /**
   * The actual initialization process.
   */
  void init();

  /**
   * Clears the function.
   */
  void clear();

  /**
   * Returns a new deep copy of the function.
   */
  virtual std::unique_ptr<FunctionBase<Number>> clone() const;

  /**
   * @returns the value at point p and time
   * time, which defaults to zero.
   */
  Number operator()(const Point &p, const Real time = 0.);

  /**
   * Like before, but returns the values in a
   * writable reference.
   */
  void operator()(const Point &p, const Real time, DenseVector<Number> &output);
};