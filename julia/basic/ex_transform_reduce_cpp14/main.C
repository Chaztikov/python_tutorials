#include <iostream>
#include <vector>
 
namespace std17
{
  template<class InputIt, class T, class BinaryOp, class UnaryOp>
  T transform_reduce(InputIt  first   ,
                     InputIt  last    ,
                     T        init    ,
                     BinaryOp binop   ,
                     UnaryOp  unary_op)
  {
    T generalizedSum = init;
    for (auto iter = first; iter != last; iter++)
    {
      generalizedSum = binop(unary_op(*iter), generalizedSum);
    }
    return generalizedSum;
  }
}
 
int main()
{
  std::vector<int> myInputVector{1, 2, 3, 4, 5};
  int              result;
 
  result = std17::transform_reduce(myInputVector.begin(),
                                   myInputVector.end()  ,
                                   0                    ,
                                  [](auto a, auto b) {return a + b;},
                                  [](auto a        ) {return a * a;});
  std::cout << result << std::endl;
  return 0;
}
// Output: 55
