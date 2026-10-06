"""
Render flange_frame.png, the frame that FrankaController.set_tcp() offsets are in.

    python docs/source/images/render_flange_frame.py

Draws the flange frame (attachment_site) of the FR3 at the home pose, x red, y green
and z blue, and in the close-up an example tool center point 10 cm out along z.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
MODEL = HERE.parents[2] / "aiofranka" / "model" / "fr3.xml"
HOME = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
SIZE = 720
COLORS = [(214, 39, 40), (44, 160, 44), (31, 119, 180)]
FONT = "/System/Library/Fonts/Helvetica.ttc"


def model_with_axes(frames):
    """The FR3 with capsules along the axes of frames (offset, length, radius, alpha) on the flange."""
    spec = mujoco.MjSpec.from_file(str(MODEL))
    site = next(s for s in spec.sites if s.name == "attachment_site")
    link = spec.body("fr3_link7")
    for offset, length, radius, alpha in frames:
        origin = site.pos + np.asarray(offset)
        for axis, color in enumerate(COLORS):
            link.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=np.concatenate([origin, origin + length * np.eye(3)[axis]]),
                size=[radius, 0.0, 0.0], rgba=[*np.array(color) / 255, alpha],
                contype=0, conaffinity=0, group=0, mass=0.0,
            )
    model = spec.compile()
    model.vis.global_.offwidth = model.vis.global_.offheight = SIZE
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.6
    return model


def project(scene, point):
    """Pixel coordinates of a point, seen from between the two eyes of the scene camera."""
    left, right = scene.camera[0], scene.camera[1]
    position = (np.array(left.pos) + np.array(right.pos)) / 2
    forward, up = np.array(left.forward), np.array(left.up)
    side = np.cross(forward, up)
    v = point - position
    depth = v @ forward
    x, y = v @ side * left.frustum_near / depth, v @ up * left.frustum_near / depth
    half = 0.5 * (left.frustum_top - left.frustum_bottom)
    return (x + half) / (2 * half) * SIZE, (left.frustum_top - y) / (2 * half) * SIZE


def render(frames, azimuth, elevation, distance, lookat_offset, labels):
    """Render the home pose with frames on the flange and labels at points in the flange frame."""
    model = model_with_axes(frames)
    data = mujoco.MjData(model)
    data.qpos[:7] = HOME
    mujoco.mj_forward(model, data)
    site = model.site("attachment_site").id
    origin, rotation = data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()

    camera = mujoco.MjvCamera()
    camera.lookat[:] = origin + np.asarray(lookat_offset)
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
    with mujoco.Renderer(model, SIZE, SIZE) as renderer:
        renderer.update_scene(data, camera)
        image = renderer.render().copy()
        scene = renderer.scene
        renderer.enable_depth_rendering()
        depth = renderer.render()

    # White background where nothing was drawn.
    image[depth >= depth.max() - 1e-6] = 255
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    for point, text, color, size in labels:
        try:
            font = ImageFont.truetype(FONT, size)
        except OSError:
            font = ImageFont.load_default(size)
        draw.text(project(scene, origin + rotation @ np.asarray(point)), text, fill=color,
                  font=font, anchor="mm")
    return picture


def main():
    axes = [(np.eye(3)[i], "xyz"[i], COLORS[i]) for i in range(3)]
    overview = render(
        frames=[([0, 0, 0], 0.15, 0.007, 1.0)],
        azimuth=160, elevation=-20, distance=1.7, lookat_offset=[-0.25, 0.0, -0.25],
        labels=[(0.19 * axis, text, color, 34) for axis, text, color in axes],
    )
    close = render(
        frames=[([0, 0, 0], 0.06, 0.0035, 1.0), ([0, 0, 0.1], 0.04, 0.0025, 0.55)],
        azimuth=135, elevation=30, distance=0.42, lookat_offset=[0.0, 0.0, -0.04],
        labels=[(0.075 * axis, text, color, 34) for axis, text, color in axes]
        + [([-0.055, 0.0, 0.1], "TCP 10 cm\nalong z", (90, 90, 90), 24)],
    )
    figure = Image.new("RGB", (2 * SIZE, SIZE), "white")
    figure.paste(overview, (0, 0))
    figure.paste(close, (SIZE, 0))
    figure.save(HERE / "flange_frame.png")


if __name__ == "__main__":
    main()
