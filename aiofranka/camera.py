"""
Calibrate a fixed camera against the robot: where it is in the robot's base frame.

The arm holds an AprilCube on its flange and is moved by hand in Programming mode,
holding the guiding button on the end effector, to 15 to 25 poses spread over the
image, near and far, with the wrist turned about at least two axes. Whenever the arm
rests at a new pose with the cube in view, the cube's tag corners are detected in a
fresh frame and recorded with the flange pose, and the terminal beeps. FCI does not run
in Programming mode, so the joint positions come from Desk, as its web UI gets them.
The fit finds the camera in the base frame (T_base_camera) and the cube on the flange
(T_ee_cube) that best reproject the corners in every view, with the stream's factory
intrinsics held fixed. Every fifth view is held out of a first fit to measure the error
on views it has not seen; the result is then fitted on all views.

The default cube is aprilcube's robot calibration cube (calibration_cube.json). Print
https://github.com/younghyopark/aprilcube/blob/main/models/calibration_cube/cube.3mf
and mount it on the flange with its connector. Frames come from aiocamera, which must be
running (aiocamera start). Both come with: pip install "aiofranka[camera]".

    aiofranka camera calibrate      # capture and fit into camera_calibration/<date>/
    aiofranka camera fit SESSION    # fit a recorded session again

Distances are in meters, and T_A_B maps B coordinates into A. Camera axes are right,
down and forward (OpenCV). The ee frame is state['ee'], the attachment_site (flange).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import select
import ssl
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aiofranka.payload import MODEL_PATH, _kinematics

CUBE = Path(__file__).resolve().parent / "calibration_cube.json"
STILL_SPEED = 0.01  # rad/s: a capture needs every joint slower than this ...
STILL_TIME = 0.7  # s: ... for this long
NEW_VIEW_DISTANCE = 0.05  # m: a pose is new this far from every captured one ...
NEW_VIEW_ANGLE = 10.0  # deg: ... or turned this much
MIN_TAGS = 2  # cube tags in a captured view
MAX_VIEW_RMS = 2.0  # px: PnP reprojection RMS of a captured view
MIN_VIEWS = 12  # distinct flange poses, at least 2 mm or 2 deg apart
MIN_ROTATION_SPREAD = (15.0, 5.0)  # deg, about the two most excited axes
HELD_OUT_EVERY = 5  # views 4, 9, 14, ... are held out of the first fit
TRANSFORM_CONVENTION = "T_A_B maps B to A; meters; optical camera right/down/forward"
EE_FRAME = "aiofranka state['ee']: MuJoCo attachment_site"

BOLD, DIM, GREEN, YELLOW, RED, CYAN, RST = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m")
PICTURE_WIDTH = 60  # characters across the camera image
REGIONS = (("top left", "top", "top right"), ("left", "center", "right"),
           ("bottom left", "bottom", "bottom right"))


class Cube:
    """An AprilCube: its tags' corners and outward normals in the cube frame [m], and
    aprilcube's detector for its dictionary."""

    def __init__(self, path=CUBE):
        from aprilcube.detect import build_tag_corner_map, create_detector, load_cube_config
        from aprilcube.generate import FACE_DEFS

        self.path = Path(path)
        self.config = json.loads(self.path.read_text())
        config, faces = load_cube_config(str(self.path))
        self.corners = {int(tag): np.asarray(corners, dtype=float) / 1000.0
                        for tag, corners in build_tag_corner_map(config).items()}
        self.normals = {int(tag): np.eye(3)[axis] * sign
                        for name, axis, sign, *_ in FACE_DEFS for tag in faces.get(name, ())}
        self.normals.update({int(tag): np.asarray(normal, dtype=float)
                             for tag, normal in (config.marker_normals or {}).items()})
        self.detector = create_detector(config.dict_id)  # the accurate preset: subpixel corners

    def detect(self, image, K, D):
        """
        The cube's pose measured from this image alone: its tags' corners and the PnP pose
        with every seen tag facing the camera. Raises ValueError saying why there is none.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        quads, ids, _ = self.detector.detectMarkers(gray)
        seen = {}
        for quad, tag in zip(quads, ids.ravel() if ids is not None else ()):
            if int(tag) in self.corners:
                if int(tag) in seen:
                    raise ValueError(f"tag {tag} is seen twice; move other targets out of view")
                seen[int(tag)] = quad.reshape(4, 2).astype(float)
        if len(seen) < MIN_TAGS:
            raise ValueError(f"{len(seen)} cube tags in view, need {MIN_TAGS}")
        tags = sorted(seen)
        points = np.concatenate([self.corners[tag] for tag in tags])
        pixels = np.concatenate([seen[tag] for tag in tags])
        centers = np.array([self.corners[tag].mean(axis=0) for tag in tags])
        normals = np.array([self.normals[tag] for tag in tags])

        # SQPnP; IPPE too when only one face shows, whose two mirror poses it returns.
        planar = np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-9) == 2
        candidates = []
        for flag in [cv2.SOLVEPNP_SQPNP] + ([cv2.SOLVEPNP_IPPE] if planar else []):
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(points, pixels, K, D, flags=flag)
            for rvec, tvec in zip(rvecs, tvecs) if ok else ():
                rvec, tvec = cv2.solvePnPRefineLM(points, pixels, K, D, rvec.copy(), tvec.copy())
                rotation = cv2.Rodrigues(rvec)[0]
                facing = np.einsum("ij,ij->i", normals @ rotation.T, centers @ rotation.T + tvec.ravel()) < 0
                if not facing.all():
                    continue
                projected = cv2.projectPoints(points, rvec, tvec, K, D)[0].reshape(-1, 2)
                rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
                candidates.append((rms, rotation, tvec.ravel()))
        if not candidates:
            raise ValueError("no cube pose has every seen tag facing the camera")
        candidates.sort(key=lambda candidate: candidate[0])
        rms, rotation, translation = candidates[0]
        for other_rms, other_rotation, _ in candidates[1:]:
            if other_rms - rms < 0.1 and Rotation.from_matrix(rotation.T @ other_rotation).magnitude() > np.radians(5):
                raise ValueError("one face's pose is ambiguous; turn the cube to show another face")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        return {"T_camera_cube": pose.tolist(), "object_points_m": points.tolist(),
                "image_points_px": pixels.tolist(), "tag_ids": tags, "reprojection_rms_px": rms}


def calibrate(robot_ip, stream=None, cube=CUBE, out="camera_calibration",
              username="admin", password="admin", protocol="https"):
    """
    Capture views of the cube while the arm is moved by hand, and fit. Returns the
    session folder: views.json and images/, and calibration.json once fitted. The robot
    must be in Programming mode with its joints unlocked (aiofranka mode program); the
    joint positions come from Desk, logged in with username and password.

    Args:
        robot_ip (str): Robot IP
        stream (str | None): aiocamera stream; None for the only RealSense color stream
        cube (str | Path): AprilCube config.json
        out (str | Path): Folder for the session folder
        username (str): Desk username
        password (str): Desk password
        protocol (str): Desk's protocol, "https" or "http"
    """
    from aiocamera import CameraClient
    from aiocamera.transport import IPC_ENDPOINT

    camera = CameraClient()
    stream, camera_info = _pick_stream(camera, stream)
    intrinsics = camera.get_intrinsics(camera_info["id"])
    K = np.array([[intrinsics["fx"], 0.0, intrinsics["cx"]],
                  [0.0, intrinsics["fy"], intrinsics["cy"]],
                  [0.0, 0.0, 1.0]])
    D = np.array(intrinsics.get("dist_coeffs") or [0.0] * 5, dtype=float)
    cube = Cube(cube)

    session = Path(out) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (session / "images").mkdir(parents=True)
    dataset = {
        "schema_version": 1,
        "setup": "fixed_camera_robot_held_cube",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transform_convention": TRANSFORM_CONVENTION,
        "camera": {
            "backend": "aiocamera",
            "endpoint": IPC_ENDPOINT,
            "serial": camera_info.get("serial", ""),
            "name": camera_info.get("name", ""),
            "stream": stream,
            "camera_id": camera_info["id"],
            "width": int(intrinsics["width"]),
            "height": int(intrinsics["height"]),
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.tolist(),
            "intrinsics_source": "aiocamera: factory intrinsics of the running stream",
        },
        "robot_ip": robot_ip,
        "ee_frame": EE_FRAME,
        "robot_model_sha256": _sha256(MODEL_PATH),
        "cube_config": cube.config,
        "cube_config_sha256": _sha256(cube.path),
        "views": [],
    }
    header = (f"{BOLD}aiofranka{RST} {DIM}|{RST} camera calibrate {DIM}({robot_ip}){RST}   "
              f"{stream} {intrinsics['width']}x{intrinsics['height']}, cube {cube.path.name}")

    arm = _DeskArm(robot_ip, username, password, protocol)
    try:
        _capture(arm, camera, stream, cube, K, D, dataset, session, header)
    finally:
        arm.close()
    return session


def _capture(arm, camera, stream, cube, K, D, dataset, session, header):
    """The capture loop: a live view in the terminal. It captures by itself when the arm
    (_DeskArm) rests at a new pose with the cube in view, and takes single-key commands."""
    views = dataset["views"]
    speeds = deque()  # (time, fastest joint speed)
    message = (f"Hold the guiding button on the end effector to move the arm, and let go: each new "
               f"pose is captured after {STILL_TIME:g} s at rest.")
    motion = _motion([])
    armed = True  # after a capture or an undo, wait for the arm to move before the next one
    drawn = 0.0
    with _Terminal() as terminal:
        while True:
            frame = camera.get_frame(stream)
            try:
                view, problem = cube.detect(frame, K, D), None
            except ValueError as error:
                view, problem = None, str(error)
            state = arm.state
            now = time.monotonic()
            speeds.append((now, float(np.abs(state["qvel"]).max())))
            while now - speeds[0][0] > STILL_TIME + 0.2:
                speeds.popleft()
            speed = max(s for t, s in speeds if now - t <= STILL_TIME)
            still = now - speeds[0][0] >= STILL_TIME and speed < STILL_SPEED
            armed = armed or speeds[-1][1] >= STILL_SPEED
            new = _is_new(state["ee"], views)
            capture = (armed and still and new and view is not None
                       and view["reprojection_rms_px"] <= MAX_VIEW_RMS)

            for key in terminal.keys():
                if key == " ":  # capture now, also near another view
                    capture = still
                    if not still:
                        message = f"{YELLOW}Hold the arm still for {STILL_TIME:g} s first.{RST}"
                elif key == "u" and views:
                    (session / views.pop()["image"]).unlink(missing_ok=True)
                    _write_json(session / "views.json", dataset)
                    motion = _motion(_poses(views))
                    armed, capture = False, False
                    message = f"Removed view {len(views) + 1}."
                elif key in "\r\n":
                    try:
                        _save_calibration(dataset, fit(dataset), session)
                        return
                    except ValueError as error:
                        message = f"{YELLOW}Not fitted: {error}{RST}"
                elif key == "q":
                    return

            if capture:
                armed = False
                try:
                    message = _record(arm, camera, stream, cube, K, D, views, session, speed)
                    terminal.bell()
                    _write_json(session / "views.json", dataset)
                    motion = _motion(_poses(views))
                except ValueError as error:
                    message = f"{YELLOW}Not captured: {error}; move on.{RST}"

            if now - drawn >= 0.1:
                terminal.draw(_screen(header, frame.shape, view, problem, views, state["ee"],
                                      still, speed, new, motion, message))
                drawn = now


def _record(arm, camera, stream, cube, K, D, views, session, speed):
    """Detect the cube in a new frame and record it with the flange pose; raises ValueError."""
    frame = camera.get_frame(stream)  # a frame taken now, with the arm still
    view = cube.detect(frame, K, D)
    if view["reprojection_rms_px"] > MAX_VIEW_RMS:
        raise ValueError(f"the cube fits its tags to {view['reprojection_rms_px']:.1f} px, "
                         f"more than {MAX_VIEW_RMS:g} px")
    state = arm.state
    if np.abs(state["qvel"]).max() > STILL_SPEED:
        raise ValueError("the arm moved")
    image = f"images/{len(views):04d}.png"
    cv2.imwrite(str(session / image), frame)
    views.append({
        "id": len(views),
        "image": image,
        "robot_timestamp_s": time.time(),
        "qpos": state["qpos"].tolist(),
        "T_base_ee": state["ee"].tolist(),
        "max_joint_speed_rad_s": speed,
        **view,
    })
    return (f"{GREEN}Captured view {len(views)}{RST}: {len(view['tag_ids'])} tags, "
            f"{view['reprojection_rms_px']:.2f} px.")


class _DeskArm:
    """
    The arm's joint positions as Desk's web UI gets them, from its websocket, about ten
    times a second. They come in Programming mode too, where FCI does not run. state is a
    controller's: qpos, qvel from the last two, and ee, the flange pose.
    """

    def __init__(self, robot_ip, username, password, protocol="https"):
        from websockets.sync.client import connect

        from aiofranka.server import _DeskClientV2

        cookie = _DeskClientV2(robot_ip, username, password, protocol=protocol)._login_cookie()
        context = None
        if protocol == "https":
            context = ssl.create_default_context()
            context.check_hostname = False  # Desk's certificate is self-signed
            context.verify_mode = ssl.CERT_NONE
        url = f"{'wss' if protocol == 'https' else 'ws'}://{robot_ip}/desk/api/robot/configuration"
        for attempt in range(3):
            try:
                self._socket = connect(url, ssl=context, open_timeout=3,
                                       additional_headers={"Cookie": f"authorization={cookie}"})
                break
            except TimeoutError:  # Desk leaves a handshake unanswered now and then
                if attempt == 2:
                    raise
        self._samples = deque(maxlen=2)  # (time [s], joint positions [rad])
        self._lock = threading.Lock()
        threading.Thread(target=self._receive, daemon=True).start()
        deadline = time.monotonic() + 3.0
        while len(self._samples) < 2:
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError("Desk sends no joint positions")
            time.sleep(0.05)

    def _receive(self):
        try:
            for message in self._socket:
                qpos = np.array(json.loads(message)["jointAngles"], dtype=float)
                with self._lock:
                    self._samples.append((time.monotonic(), qpos))
        except Exception:
            pass  # closed or lost; state notices that the joint positions stop

    @property
    def state(self):
        with self._lock:
            (t0, q0), (t1, q1) = self._samples
        if time.monotonic() - t1 > 1.0:
            raise RuntimeError("Desk stopped sending the joint positions")
        return {"qpos": q1, "qvel": (q1 - q0) / max(t1 - t0, 1e-3), "ee": _flange(q1)}

    def close(self):
        self._socket.close()


def _flange(qpos):
    """T_base_ee at joint positions qpos: the attachment_site of aiofranka's model, as
    state['ee'] is."""
    model, data, site = _kinematics()
    data.qpos[:7] = qpos
    mujoco.mj_kinematics(model, data)
    return _transform(data.site_xmat[site].reshape(3, 3), data.site_xpos[site])


def fit_session(session):
    """Fit a session folder's views.json again; writes and returns its calibration.json."""
    session = Path(session)
    dataset = json.loads((session / "views.json").read_text())
    return _save_calibration(dataset, fit(dataset), session)


def _save_calibration(dataset, result, session):
    """Write calibration.json: the fit, with the session's camera, cube and provenance."""
    views = Path(session) / "views.json"
    calibration = {
        **result,
        "schema_version": 1,
        "setup": dataset["setup"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transform_convention": TRANSFORM_CONVENTION,
        "ee_frame": dataset["ee_frame"],
        "camera": dataset["camera"],
        "robot_model_sha256": dataset["robot_model_sha256"],
        "cube_config": dataset["cube_config"],
        "dataset": str(views.resolve()),
        "dataset_sha256": _sha256(views),
    }
    _write_json(Path(session) / "calibration.json", calibration)
    return calibration


def fit(dataset):
    """
    Fit X = T_base_camera and Y = T_ee_cube to a session's views, such that
    T_base_ee @ Y = X @ T_camera_cube in every view, by the reprojection error of the
    cube's tag corners, with the camera's intrinsics fixed.

    Returns calibration.json's transforms, intrinsics and metrics. Raises ValueError if
    the views are too few or too alike to determine both transforms.
    """
    K = np.array(dataset["camera"]["camera_matrix"], dtype=float)
    D = np.array(dataset["camera"]["dist_coeffs"], dtype=float)
    views = [{
        "index": index,
        "ee": np.array(view["T_base_ee"], dtype=float),
        "cube": np.array(view["T_camera_cube"], dtype=float),
        "points": np.array(view["object_points_m"], dtype=float),
        "pixels": np.array(view["image_points_px"], dtype=float),
    } for index, view in enumerate(dataset["views"])]
    held_out = [view for view in views if view["index"] % HELD_OUT_EVERY == HELD_OUT_EVERY - 1]
    training = [view for view in views if view["index"] % HELD_OUT_EVERY != HELD_OUT_EVERY - 1]
    motion = _check_motion([view["ee"] for view in views], MIN_VIEWS)
    _check_motion([view["ee"] for view in training], MIN_VIEWS - len(held_out))

    first = _solve(training, K, D, _initial_guess(training))
    final = _solve(views, K, D, first.x)
    singular = np.linalg.svd(final.jac, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-8:
        raise ValueError("the fit is poorly constrained; capture more varied cube orientations")
    X, Y = _unpack(final.x)
    return {
        "T_base_camera": X.tolist(),
        "T_camera_base": np.linalg.inv(X).tolist(),
        "T_ee_cube": Y.tolist(),
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.tolist(),
        "metrics": {
            "motion": motion,
            "training": _errors(first.x, training, K, D),
            "held_out": _errors(first.x, held_out, K, D),
            "all_views": _errors(final.x, views, K, D),
            "jacobian_singular_values": singular.tolist(),
            "jacobian_condition_number": float(singular[0] / singular[-1]),
            "optimizer_evaluations": int(final.nfev),
            "intrinsics_refined": False,
        },
    }


def _motion(poses):
    """Distinct flange poses (2 mm or 2 deg apart) and the rotation spread [deg] about the
    three principal axes of the poses' pairwise rotations."""
    distinct = []
    for pose in poses:
        if all(np.linalg.norm(pose[:3, 3] - other[:3, 3]) > 0.002
               or Rotation.from_matrix(pose[:3, :3] @ other[:3, :3].T).magnitude() > np.radians(2)
               for other in distinct):
            distinct.append(pose)
    rotations = [Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
                 for i, a in enumerate(distinct) for b in distinct[i + 1:]]
    spread = (np.degrees(np.linalg.svd(rotations, compute_uv=False) / math.sqrt(len(rotations)))
              if len(rotations) >= 3 else np.zeros(3))
    return {"distinct_poses": len(distinct), "rotation_excitation_deg": spread.tolist()}


def _check_motion(poses, minimum):
    motion = _motion(poses)
    if motion["distinct_poses"] < minimum:
        raise ValueError(f"{motion['distinct_poses']} distinct poses, need {minimum} "
                         "(at least 2 mm or 2 deg apart)")
    spread = motion["rotation_excitation_deg"]
    if spread[0] < MIN_ROTATION_SPREAD[0] or spread[1] < MIN_ROTATION_SPREAD[1]:
        raise ValueError(f"rotation spread {spread[0]:.0f}/{spread[1]:.0f} deg, need "
                         f"{MIN_ROTATION_SPREAD[0]:.0f}/{MIN_ROTATION_SPREAD[1]:.0f}: "
                         "turn the wrist about at least two axes, 20 to 40 deg each")
    return motion


def _initial_guess(views):
    """X from OpenCV's hand-eye solve, Y from averaging what each view then implies."""
    # calibrateHandEye solves A_i X B_i = constant. For a fixed camera, A_i = T_ee_base
    # gives X = T_base_camera, with the constant T_ee_cube.
    ee_base = [np.linalg.inv(view["ee"]) for view in views]
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in ee_base], [pose[:3, 3] for pose in ee_base],
        [view["cube"][:3, :3] for view in views], [view["cube"][:3, 3] for view in views],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    X = _transform(rotation, np.ravel(translation))
    mounts = [pose @ X @ view["cube"] for pose, view in zip(ee_base, views)]
    Y = _transform(Rotation.from_matrix([pose[:3, :3] for pose in mounts]).mean().as_matrix(),
                   np.mean([pose[:3, 3] for pose in mounts], axis=0))
    return _pack(X, Y)


def _residuals(parameters, views, K, D):
    X, Y = _unpack(parameters)
    camera_base = np.linalg.inv(X)
    residuals = []
    for view in views:
        camera_cube = camera_base @ view["ee"] @ Y
        projected = cv2.projectPoints(view["points"], cv2.Rodrigues(camera_cube[:3, :3])[0],
                                      camera_cube[:3, 3], K, D)[0].reshape(-1, 2)
        residuals.append((projected - view["pixels"]).ravel())
    return np.concatenate(residuals)


def _solve(views, K, D, initial):
    result = least_squares(_residuals, initial, args=(views, K, D), loss="soft_l1", f_scale=2.0,
                           x_scale="jac", max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-8)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"the fit did not converge: {result.message}")
    X, Y = _unpack(result.x)
    for view in views:
        camera_cube = np.linalg.inv(X) @ view["ee"] @ Y
        depths = view["points"] @ camera_cube[2, :3] + camera_cube[2, 3]
        if np.any(depths <= 0):
            raise ValueError(f"the fit puts view {view['index']}'s cube behind the camera")
    return result


def _errors(parameters, views, K, D):
    """Reprojection errors [px] of views' corners: summary and per view."""
    per_view, everything = [], []
    for view in views:
        distances = np.linalg.norm(_residuals(parameters, [view], K, D).reshape(-1, 2), axis=1)
        everything.extend(distances)
        per_view.append({"view_index": view["index"], "point_count": len(distances),
                         "rms_px": float(np.sqrt(np.mean(distances ** 2)))})
    everything = np.asarray(everything)
    return {"view_count": len(views), "point_count": len(everything),
            "rms_px": float(np.sqrt(np.mean(everything ** 2))), "median_px": float(np.median(everything)),
            "p95_px": float(np.percentile(everything, 95)), "max_px": float(everything.max()),
            "per_view": per_view}


def _transform(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return pose


def _pack(X, Y):
    return np.concatenate([Rotation.from_matrix(X[:3, :3]).as_rotvec(), X[:3, 3],
                           Rotation.from_matrix(Y[:3, :3]).as_rotvec(), Y[:3, 3]])


def _unpack(parameters):
    return (_transform(Rotation.from_rotvec(parameters[0:3]).as_matrix(), parameters[3:6]),
            _transform(Rotation.from_rotvec(parameters[6:9]).as_matrix(), parameters[9:12]))


def _poses(views):
    return [np.array(view["T_base_ee"]) for view in views]


def _is_new(ee, views):
    """Whether the flange is NEW_VIEW_DISTANCE or NEW_VIEW_ANGLE from every captured view."""
    return all(np.linalg.norm(ee[:3, 3] - pose[:3, 3]) >= NEW_VIEW_DISTANCE
               or Rotation.from_matrix(pose[:3, :3].T @ ee[:3, :3]).magnitude() >= np.radians(NEW_VIEW_ANGLE)
               for pose in _poses(views))


def _pick_stream(camera, stream):
    """The aiocamera stream to calibrate, and its camera's id, name and serial."""
    try:
        status = camera.status()
    except Exception:
        raise RuntimeError("aiocamera is not running; start it with: aiocamera start") from None
    streams = status.get("active_streams", [])
    if stream is None:
        color = [name for name in streams if name.startswith("rs_") and name.endswith("_color")]
        if len(color) != 1:
            raise ValueError(f"pick the stream with --stream; aiocamera serves {streams}")
        stream = color[0]
    if stream not in streams:
        raise ValueError(f"aiocamera has no stream {stream!r}; it serves {streams}")
    return stream, next(info for info in status["cameras"] if stream.startswith(info["id"] + "_"))


def _screen(header, shape, view, problem, views, ee, still, speed, new, motion, message):
    """The terminal's lines: the camera image with the cube and the captured views, then status."""
    height, width = shape[:2]
    centers = [np.mean(captured["image_points_px"], axis=0) for captured in views]
    corners = None if view is None else np.array(view["image_points_px"])
    if view is None:
        cube = f"{RED}not found{RST}: {problem}"
    else:
        cube = (f"{GREEN}{len(view['tag_ids'])} tags{RST}, "
                f"{np.linalg.norm(np.array(view['T_camera_cube'])[:3, 3]):.2f} m away, "
                f"{view['reprojection_rms_px']:.2f} px")
    if not still:
        arm = f"{YELLOW}moving{RST} {DIM}({speed:.2f} rad/s){RST}"
    elif not new:
        arm = (f"still, {YELLOW}near a captured view{RST}: move {100 * NEW_VIEW_DISTANCE:.0f} cm "
               f"or turn {NEW_VIEW_ANGLE:.0f} deg for a new one")
    else:
        arm = f"{GREEN}still{RST}"
    lines = [header, "", *_picture(width, height, corners, centers),
             f"  cube   {cube}", f"  arm    {arm}"]
    if views:
        depths = [np.array(captured["T_camera_cube"])[2, 3] for captured in views]
        spread = motion["rotation_excitation_deg"]
        poses = _poses(views)
        nearest = min(np.linalg.norm(pose[:3, 3] - ee[:3, 3]) for pose in poses)
        turned = min(Rotation.from_matrix(pose[:3, :3].T @ ee[:3, :3]).magnitude() for pose in poses)
        seen = {(min(2, int(3 * v / height)), min(2, int(3 * u / width))) for u, v in centers}
        empty = [REGIONS[row][col] for row in range(3) for col in range(3) if (row, col) not in seen]
        lines += [
            f"  views  {len(views)} of 15 to 25, {min(depths):.2f} to {max(depths):.2f} m deep, "
            f"rotation spread {spread[0]:.0f}/{spread[1]:.0f} deg {DIM}(need "
            f"{MIN_ROTATION_SPREAD[0]:.0f}/{MIN_ROTATION_SPREAD[1]:.0f}){RST}",
            f"  next   {'no view yet at ' + ', '.join(empty) if empty else 'every image region has a view'}",
            f"  near   the nearest view is {100 * nearest:.0f} cm and {math.degrees(turned):.0f} deg away",
        ]
    lines += ["", f"  {message}",
              f"  {DIM}new still poses are captured   space capture now   u undo   enter fit   q quit{RST}"]
    return lines


def _picture(width, height, corners, centers):
    """The camera image as characters: captured views as dots, the cube in view as a box."""
    cols = PICTURE_WIDTH
    rows = max(6, round(cols * height / width / 2))  # characters are about twice as tall as wide
    grid = [[" "] * cols for _ in range(rows)]

    def cell(u, v):
        return min(rows - 1, max(0, int(v * rows / height))), min(cols - 1, max(0, int(u * cols / width)))

    for u, v in centers:
        row, col = cell(u, v)
        grid[row][col] = f"{GREEN}●{RST}"
    if corners is not None:
        (row0, col0), (row1, col1) = cell(*corners.min(axis=0)), cell(*corners.max(axis=0))
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                if grid[row][col] == " ":
                    grid[row][col] = f"{CYAN}░{RST}"
        row, col = cell(*corners.mean(axis=0))
        grid[row][col] = f"{BOLD}{CYAN}◆{RST}"
    return ["  ┌" + "─" * cols + "┐", *("  │" + "".join(row) + "│" for row in grid), "  └" + "─" * cols + "┘"]


class _Terminal:
    """Full-screen terminal for the capture loop: keys without Enter, redrawn in place."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("camera calibration needs an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, cursor hidden
        sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def keys(self):
        """The keys pressed since the last call."""
        if not select.select([self.fd], [], [], 0)[0]:
            return ""
        return os.read(self.fd, 64).decode(errors="ignore")

    def bell(self):
        sys.stdout.write("\a")
        sys.stdout.flush()

    def draw(self, lines):
        sys.stdout.write("\033[H" + "".join(f"{line}\033[K\n" for line in lines) + "\033[J")
        sys.stdout.flush()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
