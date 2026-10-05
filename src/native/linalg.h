// Small fixed-size dense linear algebra for the control laws, row-major.
//
// Every function evaluates in the order of the numpy expression it replaces in
// aiofranka/controller.py, so the native laws match FrankaController to rounding.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace aiofranka::la {

template <std::size_t R, std::size_t C>
using Mat = std::array<double, R * C>;

template <std::size_t N>
using Vec = std::array<double, N>;

template <std::size_t R, std::size_t K, std::size_t C>
Mat<R, C> matmul(const Mat<R, K>& a, const Mat<K, C>& b) {
  Mat<R, C> out{};
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * C + j];
      }
      out[i * C + j] = sum;
    }
  }
  return out;
}

template <std::size_t R, std::size_t C>
Vec<R> matvec(const Mat<R, C>& a, const Vec<C>& x) {
  Vec<R> out{};
  for (std::size_t i = 0; i < R; ++i) {
    double sum = 0.0;
    for (std::size_t k = 0; k < C; ++k) {
      sum += a[i * C + k] * x[k];
    }
    out[i] = sum;
  }
  return out;
}

template <std::size_t R, std::size_t C>
Mat<C, R> transpose(const Mat<R, C>& a) {
  Mat<C, R> out{};
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
      out[j * R + i] = a[i * C + j];
    }
  }
  return out;
}

template <std::size_t N>
Mat<N, N> identity() {
  Mat<N, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i * N + i] = 1.0;
  }
  return out;
}

// LU decomposition with partial pivoting, choosing the first largest pivot like LAPACK's
// getrf, which np.linalg.det and np.linalg.inv use.
template <std::size_t N>
struct LU {
  Mat<N, N> lu;
  std::array<std::size_t, N> pivot;
  double sign;
  bool singular;
};

template <std::size_t N>
LU<N> lu_decompose(const Mat<N, N>& a) {
  LU<N> f{a, {}, 1.0, false};
  auto& m = f.lu;
  for (std::size_t k = 0; k < N; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < N; ++i) {
      if (std::abs(m[i * N + k]) > std::abs(m[p * N + k])) {
        p = i;
      }
    }
    f.pivot[k] = p;
    if (m[p * N + k] == 0.0) {
      f.singular = true;
      continue;
    }
    if (p != k) {
      for (std::size_t j = 0; j < N; ++j) {
        std::swap(m[k * N + j], m[p * N + j]);
      }
      f.sign = -f.sign;
    }
    for (std::size_t i = k + 1; i < N; ++i) {
      m[i * N + k] /= m[k * N + k];
      for (std::size_t j = k + 1; j < N; ++j) {
        m[i * N + j] -= m[i * N + k] * m[k * N + j];
      }
    }
  }
  return f;
}

template <std::size_t N>
double det(const Mat<N, N>& a) {
  const LU<N> f = lu_decompose<N>(a);
  double d = f.sign;
  for (std::size_t i = 0; i < N; ++i) {
    d *= f.lu[i * N + i];
  }
  return d;
}

template <std::size_t N>
Mat<N, N> inverse(const Mat<N, N>& a) {
  const LU<N> f = lu_decompose<N>(a);
  if (f.singular) {
    throw std::runtime_error("Singular matrix");
  }
  Mat<N, N> out{};
  for (std::size_t c = 0; c < N; ++c) {
    Vec<N> x{};
    x[c] = 1.0;
    for (std::size_t k = 0; k < N; ++k) {
      std::swap(x[k], x[f.pivot[k]]);
    }
    for (std::size_t i = 0; i < N; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        x[i] -= f.lu[i * N + j] * x[j];
      }
    }
    for (std::size_t i = N; i-- > 0;) {
      for (std::size_t j = i + 1; j < N; ++j) {
        x[i] -= f.lu[i * N + j] * x[j];
      }
      x[i] /= f.lu[i * N + i];
    }
    for (std::size_t r = 0; r < N; ++r) {
      out[r * N + c] = x[r];
    }
  }
  return out;
}

// Singular value decomposition a = u diag(s) v^T of a square matrix by one-sided Jacobi
// rotations, accurate to rounding for the small matrices here.
template <std::size_t N>
void svd(const Mat<N, N>& a, Mat<N, N>& u, Vec<N>& s, Mat<N, N>& v) {
  u = a;
  v = identity<N>();
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (int sweep = 0; sweep < 100; ++sweep) {
    bool rotated = false;
    for (std::size_t p = 0; p + 1 < N; ++p) {
      for (std::size_t q = p + 1; q < N; ++q) {
        double alpha = 0.0, beta = 0.0, gamma = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
          alpha += u[i * N + p] * u[i * N + p];
          beta += u[i * N + q] * u[i * N + q];
          gamma += u[i * N + p] * u[i * N + q];
        }
        if (gamma == 0.0 || std::abs(gamma) <= eps * std::sqrt(alpha * beta)) {
          continue;
        }
        rotated = true;
        const double zeta = (beta - alpha) / (2.0 * gamma);
        const double t = std::copysign(1.0, zeta) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double sn = c * t;
        for (std::size_t i = 0; i < N; ++i) {
          const double up = u[i * N + p], uq = u[i * N + q];
          u[i * N + p] = c * up - sn * uq;
          u[i * N + q] = sn * up + c * uq;
          const double vp = v[i * N + p], vq = v[i * N + q];
          v[i * N + p] = c * vp - sn * vq;
          v[i * N + q] = sn * vp + c * vq;
        }
      }
    }
    if (!rotated) {
      break;
    }
  }
  for (std::size_t j = 0; j < N; ++j) {
    double norm = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      norm += u[i * N + j] * u[i * N + j];
    }
    norm = std::sqrt(norm);
    s[j] = norm;
    if (norm > 0.0) {
      for (std::size_t i = 0; i < N; ++i) {
        u[i * N + j] /= norm;
      }
    }
  }
}

// np.linalg.pinv with its default cutoff, 1e-15 times the largest singular value:
// v @ (diag(1/s) @ u^T).
template <std::size_t N>
Mat<N, N> pinv(const Mat<N, N>& a) {
  Mat<N, N> u{}, v{};
  Vec<N> s{};
  svd<N>(a, u, s, v);
  const double cutoff = 1e-15 * *std::max_element(s.begin(), s.end());
  Mat<N, N> scaled_ut{};  // diag(1/s) @ u^T
  for (std::size_t k = 0; k < N; ++k) {
    const double inv = s[k] > cutoff ? 1.0 / s[k] : 0.0;
    for (std::size_t j = 0; j < N; ++j) {
      scaled_ut[k * N + j] = inv * u[j * N + k];
    }
  }
  return matmul<N, N, N>(v, scaled_ut);
}

// FrankaController's inverse: inv if |det| > 1e-2, else pinv.
template <std::size_t N>
Mat<N, N> robust_inverse(const Mat<N, N>& a) {
  return std::abs(det<N>(a)) > 1e-2 ? inverse<N>(a) : pinv<N>(a);
}

// np.clip, which keeps NaN.
inline double clip(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

}  // namespace aiofranka::la
