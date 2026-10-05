// The rotation arithmetic of FrankaController's OSC, ported from scipy.spatial.transform:
// R.from_matrix(goal) * R.from_matrix(ee).inv(), then as_rotvec().
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "linalg.h"

namespace aiofranka::rot {

// Quaternion in scipy's scalar-last order (x, y, z, w).
using Quat = std::array<double, 4>;

inline Quat from_orthogonal_matrix(const la::Mat<3, 3>& m) {
  const double trace = m[0] + m[4] + m[8];
  const std::array<double, 4> decision{m[0], m[4], m[8], trace};
  int choice = 0;
  for (int i = 1; i < 4; ++i) {
    if (decision[i] > decision[choice]) {
      choice = i;
    }
  }
  Quat q{};
  switch (choice) {
    case 0:
      q = {1 - trace + 2 * m[0], m[3] + m[1], m[6] + m[2], m[7] - m[5]};
      break;
    case 1:
      q = {m[3] + m[1], 1 - trace + 2 * m[4], m[7] + m[5], m[2] - m[6]};
      break;
    case 2:
      q = {m[6] + m[2], m[7] + m[5], 1 - trace + 2 * m[8], m[3] - m[1]};
      break;
    default:
      q = {m[7] - m[5], m[2] - m[6], m[3] - m[1], 1 + trace};
      break;
  }
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm == 0.0) {
    throw std::runtime_error("Found zero norm quaternions in `quat`.");
  }
  for (double& c : q) {
    c /= norm;
  }
  return q;
}

// Rotation.from_matrix: refuses left-handed or null frames and replaces a matrix that is not
// orthogonal by the nearest rotation, u @ vt from its SVD.
inline Quat from_matrix(la::Mat<3, 3> m) {
  if (!(la::det<3>(m) > 0.0)) {
    throw std::runtime_error(
        "Non-positive determinant (left-handed or null coordinate frame) in rotation matrix 0.");
  }
  const la::Mat<3, 3> gram = la::matmul<3, 3, 3>(m, la::transpose<3, 3>(m));
  bool orthogonal = true;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double expected = i == j ? 1.0 : 0.0;
      // isclose(gram, eye, atol=1e-12) with its default rtol of 1e-5
      if (!(std::abs(gram[i * 3 + j] - expected) <= 1e-12 + 1e-5 * std::abs(expected))) {
        orthogonal = false;
      }
    }
  }
  if (!orthogonal) {
    la::Mat<3, 3> u{}, v{};
    la::Vec<3> s{};
    la::svd<3>(m, u, s, v);
    m = la::matmul<3, 3, 3>(u, la::transpose<3, 3>(v));
  }
  return from_orthogonal_matrix(m);
}

inline Quat inv(const Quat& q) {
  return {-q[0], -q[1], -q[2], q[3]};
}

// p * q, the rotation q followed by p.
inline Quat compose(const Quat& p, const Quat& q) {
  const double cx = p[1] * q[2] - p[2] * q[1];
  const double cy = p[2] * q[0] - p[0] * q[2];
  const double cz = p[0] * q[1] - p[1] * q[0];
  return {p[3] * q[0] + q[3] * p[0] + cx, p[3] * q[1] + q[3] * p[1] + cy,
          p[3] * q[2] + q[3] * p[2] + cz, p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2]};
}

inline la::Vec<3> as_rotvec(Quat q) {
  const bool negate =
      q[3] < 0 ||
      (q[3] == 0 && (q[0] < 0 || (q[0] == 0 && (q[1] < 0 || (q[1] == 0 && q[2] < 0)))));
  if (negate) {
    for (double& c : q) {
      c = -c;
    }
  }
  const double ax_norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
  const double angle = 2 * std::atan2(ax_norm, q[3]);
  const double angle2 = angle * angle;
  const double scale = angle <= 1e-3 ? 2 + angle2 / 12 + 7 * (angle2 * angle2) / 2880
                                     : angle / std::sin(angle / 2.0);
  return {scale * q[0], scale * q[1], scale * q[2]};
}

}  // namespace aiofranka::rot
