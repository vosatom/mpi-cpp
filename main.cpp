#include <iostream>
#include <iomanip>
#include <math.h>
#include <tuple>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#define N 20
#define i_type double

typedef Matrix<i_type, N, N> mat_n;
typedef Matrix<i_type, N, 1> vec_n;

mat_n mat() {
  return mat_n::Zero();
}

vec_n vec() {
  return vec_n::Zero();
}

bool converges(mat_n & matrix) {
  auto eigenvals = matrix.eigenvalues();

  for (int x = 0; x < N; x += 1) {
    if (abs(eigenvals[x]) >= 1) {
      return false;
    }
  }

  return true;
}

class CMethod {
public:
  int K = 1200;
  i_type eps = 1e-6;

  virtual const string name() = 0;
  virtual tuple<mat_n, vec_n, bool> _init(mat_n & A, vec_n & b) = 0;

  // x_0 = .zero
  // for k in 1...K:
  //                P                      R
  //          /-----------\            /------\.
  //    x_k = Q^{-1}*(Q - A)*x_{k-1} + Q^{-1}*b
  //    if ||A*x_k - b|| < eps {
  //      return x_k
  //    }
  // return nil
  pair<vec_n, int> solve(mat_n & A, vec_n & b) {
    double b_norm = b.norm();

    vec_n x = vec();

    // GEt P and R based on selected method
    auto PR = _init(A, b);
    mat_n & P = get<0>(PR);
    vec_n & R = get<1>(PR);
    bool convergence = get<2>(PR);

    if (!convergence) {
      cout << "Diverges" << endl;
      return pair<vec_n, int>(x, 0);
    }

    int k = K;
    long double rk_norm = 0;
    long double curr_eps = 0;
    long double prev_eps = 0;
    // Do only K iterations
    while (k) {
      rk_norm = (A*x - b).norm();
      curr_eps = rk_norm / b_norm;

      // Stop, if the precision is enough
      if (curr_eps < eps) {
        return pair<vec_n, int>(x, (K - k));
      }

      x = P*x + R;
      k -= 1;
      prev_eps = curr_eps;
    }

    return pair<vec_n, int>(x, -(K - k));
  }
};

/**
 * Set 0 to a_yx except diagonal
 */
mat_n keepDiagonal(mat_n matrix) {
  for (int y = 0; y < N; y += 1) {
    for (int x = 0; x < N; x += 1) {
      if (x != y) {
        matrix(y, x) = 0;
      }
    }
  }
  return matrix;
}

/**
 * Set 0 to a_yx below diagonal
 */
mat_n keepUpper(mat_n matrix) {
  for (int y = 0; y < N; y += 1) {
    for (int x = 0; x <= y; x += 1) {
      matrix(y, x) = 0;
    }
  }
  return matrix;
}

/**
 * Set 0 to a_yx above diagonal
 */
mat_n keepLower(mat_n matrix) {
  for (int y = 0; y < N; y += 1) {
    for (int x = y; x < N; x += 1) {
      matrix(y, x) = 0;
    }
  }
  return matrix;
}

/**
 * Set value to a_xx
 */
mat_n & fillDiagonal(mat_n & matrix, double value) {
  for (int x = 0; x < N; x += 1) {
    matrix(x, x) = value;
  }
  return matrix;
}

/**
 * Set value to siblings of a_xx
 */
mat_n & fillDiagonalSiblings(mat_n & matrix, double value) {
  for (int y = 0; y < N; y += 1) {
    for (int x = 0; x < N; x += 1) {
      if (abs(y - x) == 1) {
        matrix(y, x) = value;
      }
    }
  }
  return matrix;
}

/**
 * Set value as defined in HW
 */
vec_n & fillPyramid(vec_n & vector, double value) {
  // for (int x = 1; x <= N / 2; x += 1) {
  //   vector(x - 1) = value - x;
  //   vector(N - x) = value - x;
  // }
  for (int x = 0; x < N; x += 1) {
    if (x == 0 || x + 1 == N) {
      vector(x) = value - 1;
    } else {
      vector(x) = value - 2;
    }
  }
  return vector;
}

class CJacobiMethod: public CMethod {
public:

  const string name() {
    return "Jacobi";
  }
  
  // Q = D
  virtual tuple<mat_n, vec_n, bool> _init(mat_n & A, vec_n & b) {
    mat_n D = keepDiagonal(A);
    mat_n Dinv = D.inverse();
    mat_n mini = D - A;

    mat_n W = mat_n::Identity() - Dinv * A;

    return tuple<mat_n, vec_n, bool>(
      Dinv * mini,
      Dinv * b,
      converges(W)
    );
  }
};

class CGSMethod: public CMethod {
public:
  const string name() {
    return "GS";
  }

  // Q = L + D
  virtual tuple<mat_n, vec_n, bool> _init(mat_n & A, vec_n & b) {
    mat_n L = keepLower(A);
    mat_n D = keepDiagonal(A);
    mat_n U = keepUpper(A);

    mat_n DL = (D + L).inverse();

    mat_n W = mat_n::Identity() - DL * A;

    return tuple<mat_n, vec_n, bool>(
      DL * (-U),
      DL * b,
      converges(W)
    );
  }
};

class CSolver {
public:
  void solve(double gamma, CMethod * method) {

    cout << "method: " << method->name() << endl;
    cout << "gamma: " << gamma << endl;
    mat_n A = mat();
    A = fillDiagonal(A, gamma);
    A = fillDiagonalSiblings(A, -1);

    vec_n b = vec();
    b = fillPyramid(b, gamma);

    auto result = method->solve(A, b);

    if (get<1>(result) > 0) {
      cout << "Result (done in " << (get<1>(result)) << " iterations):" << endl;
      cout << get<0>(result).transpose() << endl;
    } else {
      cout << "No result (done in " << (-get<1>(result)) << " iterations)" << endl;
    }

    cout << endl;
  }
};

int main() {
  CSolver solver;
  CMethod * methods[] = {
    new CJacobiMethod(),
    new CGSMethod(),
  };
  int gammas[] = {
    3,
    2,
    1,
  };

  for (auto & method : methods) {
    for (auto & gamma : gammas) {
      solver.solve(gamma, method);
    }
  }

  return 0;
}
