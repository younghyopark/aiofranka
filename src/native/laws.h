// The control laws of FrankaController (aiofranka/controller.py), and the structs they read.
//
// Python reads the layouts of Params and LawState from the module (PARAMS_LAYOUT,
// STATE_LAYOUT), so custom laws compiled with Numba see the same fields.
#pragma once

#include <cstdint>

#include "linalg.h"
#include "rotation.h"

namespace aiofranka {

constexpr int kJoints = 7;
constexpr int kCustomParams = 1024;  // doubles for the parameters of custom laws
constexpr int kMemory = 1024;        // doubles of memory a custom law keeps between cycles
constexpr double kDt = 1e-3;         // FrankaController's step for rate limits and the integral

enum Mode : int64_t { kImpedance = 0, kPid = 1, kOsc = 2, kTorque = 3, kCustom = 4 };

// Controller attributes, written from Python and copied by the loop at each cycle.
struct Params {
  int64_t mode;
  int64_t clip;
  int64_t custom_law;
  int64_t reserved;
  double torque_diff_limit[7];
  double kp[7];
  double kd[7];
  double ki[7];
  double ee_kp[6];
  double ee_kd[6];
  double null_kp[7];
  double null_kd[7];
  double q_desired[7];
  double ee_desired[16];
  double torque[7];
  double initial_qpos[7];
  double control_transform[16];
  double torque_limit[7];
  double custom[kCustomParams];
};

// What a control law reads at each cycle: FrankaController.state, plus time.
struct LawState {
  int64_t cycle;
  double time;
  double dt;
  double qpos[7];
  double qvel[7];
  double ee[16];   // pose of the flange (attachment_site), 4x4
  double jac[42];  // its Jacobian, 6x7, linear rows first
  double mm[49];   // joint-space mass matrix
  double last_torque[7];
};

// Custom law compiled by Numba: fills tau, returns 0, or an error code that stops the loop.
using LawFn = int32_t (*)(const LawState*, const Params*, double* memory, double* tau);

// The rate limit and clip of FrankaController (clip=True).
inline void limit_rate(const Params& p, const double last[7], double tau[7]) {
  for (int i = 0; i < kJoints; ++i) {
    double diff = (tau[i] - last[i]) / kDt;
    diff = la::clip(diff, -p.torque_diff_limit[i], p.torque_diff_limit[i]);
    tau[i] = last[i] + diff * kDt;
  }
}

inline void clip_torque(const Params& p, double tau[7]) {
  for (int i = 0; i < kJoints; ++i) {
    tau[i] = la::clip(tau[i], -p.torque_limit[i], p.torque_limit[i]);
  }
}

// FrankaController._impedance_step
inline void impedance(const LawState& s, const Params& p, double tau[7]) {
  for (int i = 0; i < kJoints; ++i) {
    const double position_error = p.q_desired[i] - s.qpos[i];
    tau[i] = position_error * p.kp[i] - s.qvel[i] * p.kd[i];
  }
  if (p.clip) {
    limit_rate(p, s.last_torque, tau);
    clip_torque(p, tau);
  }
}

// FrankaController._pid_step, which rate-limits but does not clip.
inline void pid(const LawState& s, const Params& p, double integral[7], double tau[7]) {
  for (int i = 0; i < kJoints; ++i) {
    const double position_error = p.q_desired[i] - s.qpos[i];
    integral[i] += position_error * kDt;
    integral[i] = la::clip(integral[i], -10.0, 10.0);
    tau[i] = position_error * p.kp[i] + integral[i] * p.ki[i] - s.qvel[i] * p.kd[i];
  }
  if (p.clip) {
    limit_rate(p, s.last_torque, tau);
  }
}

// FrankaController._osc_step
inline void osc(const LawState& s, const Params& p, double tau[7]) {
  using la::Mat;
  using la::Vec;
  Mat<4, 4> flange{}, transform{};
  std::copy(s.ee, s.ee + 16, flange.begin());
  std::copy(p.control_transform, p.control_transform + 16, transform.begin());
  const Mat<4, 4> ee = la::matmul<4, 4, 4>(flange, transform);

  // Jacobian of the control frame, whose origin moves with v + w x r.
  const Vec<3> offset{ee[3] - flange[3], ee[7] - flange[7], ee[11] - flange[11]};
  Mat<6, 7> jac{};
  std::copy(s.jac, s.jac + 42, jac.begin());
  for (int c = 0; c < kJoints; ++c) {
    const double wx = jac[3 * 7 + c], wy = jac[4 * 7 + c], wz = jac[5 * 7 + c];
    jac[0 * 7 + c] += wy * offset[2] - wz * offset[1];
    jac[1 * 7 + c] += wz * offset[0] - wx * offset[2];
    jac[2 * 7 + c] += wx * offset[1] - wy * offset[0];
  }

  const double* goal = p.ee_desired;
  const Vec<3> position_error{goal[3] - ee[3], goal[7] - ee[7], goal[11] - ee[11]};
  const Mat<3, 3> goal_rotation{goal[0], goal[1], goal[2], goal[4], goal[5],
                                goal[6], goal[8], goal[9], goal[10]};
  const Mat<3, 3> ee_rotation{ee[0], ee[1], ee[2], ee[4], ee[5], ee[6], ee[8], ee[9], ee[10]};
  const rot::Quat error_quat =
      rot::compose(rot::from_matrix(goal_rotation), rot::inv(rot::from_matrix(ee_rotation)));
  const Vec<3> rotation_error = rot::as_rotvec(error_quat);

  Vec<6> twist{position_error[0], position_error[1], position_error[2],
               rotation_error[0], rotation_error[1], rotation_error[2]};
  Vec<7> dq{};
  std::copy(s.qvel, s.qvel + 7, dq.begin());
  const Vec<6> ee_vel = la::matvec<6, 7>(jac, dq);

  Mat<7, 7> mm{};
  std::copy(s.mm, s.mm + 49, mm.begin());
  const Mat<7, 7> minv = la::robust_inverse<7>(mm);
  const Mat<7, 6> jac_t = la::transpose<6, 7>(jac);
  const Mat<6, 6> mx_inv = la::matmul<6, 7, 6>(la::matmul<6, 7, 7>(jac, minv), jac_t);
  const Mat<6, 6> mx = la::robust_inverse<6>(mx_inv);

  // operational space feedback torque
  Vec<6> wrench_acc{};
  for (int i = 0; i < 6; ++i) {
    wrench_acc[i] = p.ee_kp[i] * twist[i] - p.ee_kd[i] * ee_vel[i];
  }
  const Vec<7> feedback = la::matvec<7, 6>(la::matmul<7, 6, 6>(jac_t, mx), wrench_acc);

  // null space torque
  Vec<7> ddq{};
  for (int i = 0; i < kJoints; ++i) {
    ddq[i] = p.null_kp[i] * (p.initial_qpos[i] - s.qpos[i]) - p.null_kd[i] * s.qvel[i];
  }
  const Mat<7, 6> jbar = la::matmul<7, 6, 6>(la::matmul<7, 7, 6>(minv, jac_t), mx);
  const Mat<7, 7> projected = la::matmul<7, 6, 7>(jac_t, la::transpose<7, 6>(jbar));
  Mat<7, 7> projector = la::identity<7>();
  for (int i = 0; i < 49; ++i) {
    projector[i] -= projected[i];
  }
  const Vec<7> null = la::matvec<7, 7>(projector, ddq);

  for (int i = 0; i < kJoints; ++i) {
    tau[i] = feedback[i] + null[i];
  }
  if (p.clip) {
    limit_rate(p, s.last_torque, tau);
    clip_torque(p, tau);
  }
}

}  // namespace aiofranka
