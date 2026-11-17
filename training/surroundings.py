"""Surroundings for the walk scene: a head camera, people walking past the robot, and objects around it.

- Head camera: 640 x 480 RGB on the robot's head, looking forward, 5 images a second (published on ROS by the
  walk scene, for vision_node).
- People: NVIDIA's human characters walk across in front of the robot at random distances, one after another.
- Objects: a table with a mug, a bowl and a banana beside the robot, a power drill and a bottle on the floor.
- Showing something: publish a name on /scene/show ("apple", "banana", "mug", "bottle", "drill", "scissors",
  or "none") and it appears on a small table right in front of the robot, as if a person showed it.

NVIDIA's assets are loaded from NVIDIA's asset server; the apple is built here (there is no apple model there).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

ASSETS = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac"
PEOPLE = [f"{ASSETS}/People/Characters/{name}/{name}.usd" for name in ("F_Business_02", "M_Medical_01", "F_Medical_01")]
WALK_ANIMATION = f"{ASSETS}/People/Animations/stand_walk_loop_in_place.skelanim.usd"
YCB = f"{ASSETS}/Props/YCB/Axis_Aligned"
SHOWABLE = {  # name: USD (None = built here), rotation about X that stands it the right way up (degrees)
    "apple": ("apple", 0.0),  # built here (apple_usd), loaded like the others
    "banana": (f"{YCB}/011_banana.usd", 90.0),
    "mug": (f"{YCB}/025_mug.usd", 90.0),
    "bottle": (f"{YCB}/006_mustard_bottle.usd", -90.0),
    "drill": (f"{YCB}/035_power_drill.usd", 90.0),
    "scissors": (f"{YCB}/037_scissors.usd", 0.0),
}
CAMERA_PERIOD_S = 0.3  # 3.3 images a second: above what the vision node uses (3 a second by default, 1.5 in the
# stall demo); each image costs the sim a ~30 ms render, so more would only be thrown away
WALK_SPEED = 1.0  # m/s, people
PASS_HALF_LENGTH = 4.0  # people walk from 4 m on one side of the robot to 4 m on the other
PASS_DISTANCE = (1.5, 3.5)  # how far in front of the robot they pass (m)
SHOW_DISTANCE_M = 0.5  # a shown object stands on a small table this far in front of the robot
SHOW_TABLE_HEIGHT_M = 0.55  # well below the head camera (about 0.95 m), so it is seen from above
PAUSE_S = (2.0, 6.0)  # between one person passing and the next


def configure_head_camera(env_cfg) -> None:
    """Add the head camera to the scene (needs the simulation app started with cameras enabled)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_pitch_link/head_camera",
        update_period=CAMERA_PERIOD_S,
        width=640,
        height=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.05, 30.0)),
        # tilted 15 degrees down, like a person looking at a table in front of them; (x, y, z, w)
        offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.08), rot=(0.0, math.sin(math.radians(7.5)), 0.0,
                                                               math.cos(math.radians(7.5))), convention="world"),
    )


def apple_texture() -> str:
    """Path to an apple-skin texture (red with yellow-green streaks and a lighter top), made on first use."""
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "apple_skin.png")
    if os.path.exists(path):
        return path
    import numpy as np
    from PIL import Image

    height, width = 256, 512
    v, u = np.mgrid[0:height, 0:width] / np.array([height, width])[:, None, None]
    rng = np.random.default_rng(0)
    # mostly deep red, with fine, faint streaks running from the stem down, as on a real apple
    streaks = np.sin(2 * np.pi * (u * 48 + 0.4 * rng.normal(0, 1, (1, width)).cumsum(1) / 30))
    base = np.stack([0.58 + 0.06 * streaks, 0.07 + 0.03 * streaks, 0.06 + 0.02 * streaks], -1)
    # a yellow-green blush on one side, a lighter ring around the stem, a darker bottom
    blush = np.exp(-(((u - 0.3) / 0.12) ** 2) - (((v - 0.45) / 0.25) ** 2))[..., None]
    base = base * (1 - 0.75 * blush) + np.array([0.78, 0.6, 0.18]) * 0.75 * blush
    top = (np.clip(1 - v / 0.12, 0, 1) ** 2)[..., None]
    base = base * (1 - top) + np.array([0.6, 0.5, 0.2]) * top
    base = base * (1 - 0.3 * np.clip((v - 0.8) / 0.2, 0, 1))[..., None]
    speckles = (rng.random((height, width)) > 0.997)[..., None] * np.array([0.35, 0.3, 0.12])
    image = np.clip(base + speckles, 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)
    return path


def apple_usd() -> str:
    """Path to the apple model (training/assets/apple.usd), built on first use (Surroundings._build_apple)."""
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "apple.usd")
    if not os.path.exists(path):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        Surroundings._build_apple(stage, "/Apple")
        # sit it on the ground: the bottom of the apple at z = 0
        from pxr import Gf

        UsdGeom.Xformable(stage.GetPrimAtPath("/Apple")).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.033))
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Apple"))
        stage.GetRootLayer().Save()
    return path


@dataclass
class _Walker:
    prim_path: str
    start: tuple[float, float]
    end: tuple[float, float]
    progress_m: float = 0.0
    pause_s: float = 0.0

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)


class Surroundings:
    """People and objects around the robot. Call step(dt, robot_xy, robot_yaw) every control step."""

    def __init__(self, stage, origin: tuple[float, float], people: int = 2, seed: int = 0, room: bool = True) -> None:
        """``room``: the small room (a table with products, things on the floor); off inside the GITEX hall."""
        self._stage = stage
        self._origin = origin
        self._random = random.Random(seed)
        self._walkers: list[_Walker] = []
        self._shown: str | None = None
        if room:
            self._build_room()
        for index in range(people):
            path = f"/World/People/person_{index}"
            self._add_person(path, PEOPLE[index % len(PEOPLE)])
            walker = _Walker(path, (0, 0), (0, 0), pause_s=self._random.uniform(0.5, 4.0) + 3.0 * index)
            self._new_pass(walker)
            self._walkers.append(walker)

    # --- building the scene ---

    def _add_reference(self, path: str, url: str, position, yaw_deg: float = 0.0, scale: float = 1.0):
        from pxr import Gf, UsdGeom

        # the model goes in a child prim, so its own transform stays as it is; this prim places it
        xform = UsdGeom.Xform.Define(self._stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateZOp().Set(yaw_deg)
        xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))
        model = self._stage.DefinePrim(f"{path}/model")
        model.GetReferences().AddReference(url)
        return model

    def _box(self, path: str, size, position, colour) -> None:
        from pxr import Gf, UsdGeom

        cube = UsdGeom.Cube.Define(self._stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddScaleOp().Set(Gf.Vec3d(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

    def _build_room(self) -> None:
        ox, oy = self._origin
        # a table to the robot's front left with a mug, a bowl and a banana on it
        self._box("/World/Room/table_top", (0.8, 0.6, 0.04), (ox + 1.4, oy + 1.0, 0.74), (0.55, 0.35, 0.2))
        for i, (dx, dy) in enumerate(((-0.35, -0.25), (0.35, -0.25), (-0.35, 0.25), (0.35, 0.25))):
            self._box(f"/World/Room/table_leg_{i}", (0.04, 0.04, 0.72), (ox + 1.4 + dx, oy + 1.0 + dy, 0.36), (0.4, 0.25, 0.15))
        self._add_reference("/World/Room/mug", SHOWABLE["mug"][0], (ox + 1.3, oy + 0.9, 0.80))
        self._add_reference("/World/Room/bowl", f"{YCB}/024_bowl.usd", (ox + 1.55, oy + 1.1, 0.76))
        self._add_reference("/World/Room/banana", SHOWABLE["banana"][0], (ox + 1.25, oy + 1.15, 0.76))
        # things on the floor to the front right
        self._add_reference("/World/Room/drill", SHOWABLE["drill"][0], (ox + 1.6, oy - 1.0, 0.0))
        self._add_reference("/World/Room/bottle", SHOWABLE["bottle"][0], (ox + 1.4, oy - 0.7, 0.0))

    def _add_person(self, path: str, url: str) -> None:
        from pxr import Usd, UsdSkel

        self._add_reference(path, url, (0.0, 0.0, -50.0))  # out of sight until their first pass
        anim_path = f"{path}_walk"
        anim = self._stage.DefinePrim(anim_path)
        anim.GetReferences().AddReference(WALK_ANIMATION)
        for prim in Usd.PrimRange(self._stage.GetPrimAtPath(path)):
            if prim.IsA(UsdSkel.Root) or prim.IsA(UsdSkel.Skeleton):
                UsdSkel.BindingAPI.Apply(prim).CreateAnimationSourceRel().SetTargets([anim.GetPath()])

    @staticmethod
    def _build_apple(stage, path: str) -> None:
        """An apple about 8 cm across: apple-shaped (dimpled top and bottom), with an apple-skin texture (red with
        yellow-green streaks), matte, with a stem and a leaf."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        rings, segments, radius = 24, 32, 0.04
        points, uvs, counts, indices = [], [], [], []
        for i in range(rings + 1):
            theta = math.pi * i / rings  # 0 = top
            # apple profile: wider at the shoulders, dimples at the top and bottom
            r = radius * (1.0 + 0.08 * math.sin(theta) - 0.18 * math.exp(-((theta / 0.35) ** 2))
                          - 0.12 * math.exp(-(((math.pi - theta) / 0.45) ** 2)))
            for j in range(segments + 1):  # one extra column: the texture seam
                phi = 2 * math.pi * j / segments
                points.append(Gf.Vec3f(r * math.sin(theta) * math.cos(phi), r * math.sin(theta) * math.sin(phi),
                                       0.9 * r * math.cos(theta)))
                uvs.append(Gf.Vec2f(j / segments, 1.0 - i / rings))
        columns = segments + 1
        for i in range(rings):
            for j in range(segments):
                a, b = i * columns + j, i * columns + j + 1
                counts.append(4)
                indices.extend([a, b, b + columns, a + columns])
        body = UsdGeom.Mesh.Define(stage, f"{path}/body")
        body.CreatePointsAttr(points)
        body.CreateFaceVertexCountsAttr(counts)
        body.CreateFaceVertexIndicesAttr(indices)
        body.CreateSubdivisionSchemeAttr("none")
        body.CreateDoubleSidedAttr(True)
        body.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        UsdGeom.PrimvarsAPI(body).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                UsdGeom.Tokens.vertex).Set(uvs)
        stem = UsdGeom.Cylinder.Define(stage, f"{path}/stem")
        stem.CreateRadiusAttr(0.003)
        stem.CreateHeightAttr(0.025)
        stem.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.035))
        stem.AddRotateYOp().Set(12.0)
        leaf = UsdGeom.Sphere.Define(stage, f"{path}/leaf")
        leaf.CreateRadiusAttr(0.02)
        leaf.AddTranslateOp().Set(Gf.Vec3d(0.012, 0, 0.044))
        leaf.AddRotateYOp().Set(-25.0)
        leaf.AddScaleOp().Set(Gf.Vec3f(1.0, 0.45, 0.12))
        for part, colour, roughness in ((body, None, 0.3), (stem, (0.28, 0.17, 0.08), 0.8), (leaf, (0.16, 0.5, 0.12), 0.6)):
            material = UsdShade.Material.Define(stage, f"{part.GetPath()}_material")
            shader = UsdShade.Shader.Define(stage, f"{part.GetPath()}_material/shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            if colour is None:  # the apple skin texture
                reader = UsdShade.Shader.Define(stage, f"{part.GetPath()}_material/st")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                texture = UsdShade.Shader.Define(stage, f"{part.GetPath()}_material/skin")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(apple_texture())
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb")
            else:
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)

    # --- moving things ---

    def _new_pass(self, walker: _Walker) -> None:
        distance = self._random.uniform(*PASS_DISTANCE)
        side = self._random.choice((-1.0, 1.0))
        ox, oy = self._origin
        walker.start = (ox + distance, oy + side * PASS_HALF_LENGTH)
        walker.end = (ox + distance + self._random.uniform(-0.5, 0.5), oy - side * PASS_HALF_LENGTH)
        walker.progress_m = 0.0

    def _place(self, path: str, x: float, y: float, z: float, yaw_deg: float) -> None:
        from pxr import Gf, UsdGeom

        ops = UsdGeom.Xformable(self._stage.GetPrimAtPath(path)).GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(x, y, z))
        ops[1].Set(yaw_deg)

    def step(self, dt: float) -> None:
        for walker in self._walkers:
            if walker.pause_s > 0:
                walker.pause_s -= dt
                continue
            walker.progress_m += WALK_SPEED * dt
            if walker.progress_m >= walker.length:
                self._place(walker.prim_path, 0.0, 0.0, -50.0, 0.0)  # out of sight
                walker.pause_s = self._random.uniform(*PAUSE_S) * len(self._walkers)
                self._new_pass(walker)
                continue
            share = walker.progress_m / walker.length
            x = walker.start[0] + share * (walker.end[0] - walker.start[0])
            y = walker.start[1] + share * (walker.end[1] - walker.start[1])
            heading = math.degrees(math.atan2(walker.end[1] - walker.start[1], walker.end[0] - walker.start[0]))
            self._place(walker.prim_path, x, y, 0.0, heading - 90.0)  # the characters face +Y

    def show(self, name: str, robot_xy: tuple[float, float], robot_yaw: float) -> str:
        """Put ``name`` on a small table right in front of the robot, as if someone showed it ("none" takes it
        away). Returns what is shown now."""
        from pxr import Gf, UsdGeom

        for path in ("/World/Shown", ):
            if self._stage.GetPrimAtPath(path):
                self._stage.RemovePrim(path)
        name = name.strip().lower()
        if name not in SHOWABLE:
            self._shown = None
            return "nothing"
        distance, height = SHOW_DISTANCE_M, SHOW_TABLE_HEIGHT_M
        x = robot_xy[0] + distance * math.cos(robot_yaw)
        y = robot_xy[1] + distance * math.sin(robot_yaw)
        self._box("/World/Shown/stand", (0.25, 0.25, 0.03), (x, y, height - 0.015), (0.85, 0.85, 0.82))
        self._box("/World/Shown/stand_leg", (0.05, 0.05, height - 0.03), (x, y, (height - 0.03) / 2), (0.6, 0.6, 0.6))
        url, upright = SHOWABLE[name]
        url = apple_usd() if url == "apple" else url
        model = self._add_reference("/World/Shown/object", url, (x, y, height), math.degrees(robot_yaw) + 90)
        UsdGeom.Xformable(model).AddRotateXOp().Set(upright)  # the YCB models are not all stored upright
        # the models are centred on their middle: lift so the bottom rests on the table
        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bottom = cache.ComputeWorldBound(self._stage.GetPrimAtPath("/World/Shown/object")).ComputeAlignedRange().GetMin()[2]
        self._place("/World/Shown/object", x, y, height + (height - bottom) + 0.002, math.degrees(robot_yaw) + 90)
        self._shown = name
        return name
