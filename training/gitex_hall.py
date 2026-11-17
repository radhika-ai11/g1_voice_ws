"""A GITEX exhibition hall around the robot: booths, screens, products, signs and visitors walking the aisles.

Everything visible is built as triangle meshes (mesh_kit) and shown twice from the same triangles:
- in the USD stage, for the robot's head camera (Isaac Sim's renderer, which YOLO looks at), and
- in the Newton viewer (log_mesh), which draws only the physics model unless it is given meshes.

The robot stands in the Aideology booth at the origin, facing +x onto the main aisle. Visitors are NVIDIA's biped
character walking (skinned_people), each in their own clothes, on looping routes along the aisles (visitors.py);
some stop in front of the robot. /scene/show still puts a product on the small stand in front of the robot.

A test visitor (stall_visit.py) can be sent to the stall carrying one of HELD_COLOURS' objects: request_visit(name)
walks them up from the aisle to stand in front of the robot holding it out, end_visit() sends them away.
"""

from __future__ import annotations

import math
import os
import random
import urllib.request

import numpy as np

from mesh_kit import MeshPart, box, cylinder, merge_by_colour, sphere, triangulate
from skinned_people import load_walk_cycle, outfit_texture, region_uvs
from stall_visit import StallVisit, VisitPose, hand_track, hold_out_pose
from visitors import Visitor, Waypoint

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gitex")
YCB = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Props/YCB/Axis_Aligned"

# colours
CARPET = (0.16, 0.18, 0.24)
AIDEOLOGY = (0.36, 0.16, 0.62)
WHITE_WALL = (0.9, 0.9, 0.88)
SCREEN_FRAME = (0.05, 0.05, 0.06)
DESK = (0.95, 0.95, 0.95)
WOOD = (0.62, 0.45, 0.28)
POT = (0.85, 0.85, 0.82)
LEAVES = (0.2, 0.5, 0.18)
GITEX_GREEN = (0.0, 0.75, 0.45)
BRANDS = [(0.0, 0.55, 0.6), (0.9, 0.45, 0.05), (0.15, 0.6, 0.25), (0.8, 0.12, 0.2), (0.1, 0.3, 0.75),
          (0.95, 0.75, 0.1), (0.55, 0.2, 0.55), (0.2, 0.2, 0.25)]
# what a visitor to the robot's stall can bring (surroundings.SHOWABLE), and its plain colour in the Newton viewer
HELD_COLOURS = {"apple": (0.72, 0.06, 0.06), "banana": (0.95, 0.85, 0.2), "mug": (0.8, 0.12, 0.1),
                "bottle": (0.95, 0.78, 0.08), "scissors": (0.15, 0.3, 0.8)}
HIDDEN_Z = -50.0  # where things not in the hall wait (under the floor)
FAR_VISITOR_M = 8.0  # visitors farther than this from the robot are small in its head camera
FAR_POSE_EVERY = 4  # ... so their walking pose is refreshed there every 4th camera frame (their place every frame)
SCREEN_CONTENT = [(0.15, 0.55, 0.95), (0.95, 0.35, 0.55), (0.25, 0.85, 0.65), (0.98, 0.7, 0.2)]
OUTFITS = [  # shoes, trousers, shirt, skin
    ((0.1, 0.1, 0.1), (0.15, 0.2, 0.35), (0.9, 0.9, 0.92), (0.87, 0.68, 0.55)),
    ((0.3, 0.2, 0.1), (0.25, 0.25, 0.27), (0.2, 0.35, 0.65), (0.6, 0.42, 0.3)),
    ((0.1, 0.1, 0.1), (0.75, 0.68, 0.55), (0.75, 0.15, 0.15), (0.93, 0.78, 0.66)),
    ((0.9, 0.9, 0.9), (0.1, 0.1, 0.12), (0.12, 0.12, 0.14), (0.45, 0.3, 0.2)),
    ((0.2, 0.12, 0.08), (0.2, 0.3, 0.5), (0.95, 0.85, 0.3), (0.8, 0.6, 0.45)),
    ((0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (0.3, 0.6, 0.35), (0.7, 0.5, 0.38)),
    ((0.15, 0.15, 0.15), (0.95, 0.95, 0.95), (0.85, 0.85, 0.85), (0.9, 0.72, 0.6)),  # a kandura
    ((0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.85, 0.66, 0.52)),  # an abaya
    ((0.25, 0.15, 0.1), (0.35, 0.25, 0.2), (0.55, 0.35, 0.65), (0.55, 0.38, 0.27)),
    ((0.1, 0.1, 0.1), (0.2, 0.22, 0.3), (0.95, 0.5, 0.15), (0.92, 0.75, 0.62)),
]

# 5 x 7 block letters for the signs
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    " ": ["00000"] * 7,
}


def sign(name: str, text: str, start, along_yaw: float, cell: float, colour, depth: float = 0.06) -> list[MeshPart]:
    """Block letters standing upright, reading along ``along_yaw`` from ``start`` (the bottom-left corner)."""
    sx, sy, sz = start
    ux, uy = math.cos(along_yaw), math.sin(along_yaw)
    parts = []
    for n, letter in enumerate(text):
        rows = FONT[letter]
        for r, row in enumerate(rows):
            for c, on in enumerate(row):
                if on != "1":
                    continue
                along = (n * 6 + c + 0.5) * cell
                parts.append(box(f"{name}_{n}_{r}_{c}", (sx + ux * along, sy + uy * along, sz + (6 - r + 0.5) * cell),
                                 (cell, depth, cell), colour, yaw=along_yaw, roughness=0.4))
    return parts


def booth(name: str, x: float, y: float, facing: float, brand, width: float = 6.0, depth: float = 4.0,
          screen=(0.15, 0.55, 0.95), label: str | None = None) -> list[MeshPart]:
    """An exhibitor booth: carpet, back wall with a brand header and a screen, side walls, a counter, plants.

    (x, y) is the middle of the booth's front edge; ``facing`` the direction its open front faces.
    """
    fx, fy = math.cos(facing), math.sin(facing)  # out of the booth, onto the aisle
    sx, sy = -fy, fx  # along the front edge

    def at(forward: float, side: float, z: float):
        return (x + fx * forward + sx * side, y + fy * forward + sy * side, z)

    yaw = facing + math.pi / 2  # boxes sized (along the front, across, up) turn with the booth
    parts = [
        # the booth carpet's top is 2.5 cm up, clear of the hall carpet (or the two flicker in the viewers)
        box(f"{name}_carpet", at(-depth / 2, 0, 0.0175), (width, depth, 0.015), tuple(0.35 * c + 0.2 for c in brand), yaw),
        box(f"{name}_back", at(-depth, 0, 1.5), (width, 0.15, 3.0), WHITE_WALL, yaw),
        box(f"{name}_header", at(-depth + 0.1, 0, 3.25), (width, 0.2, 0.5), brand, yaw, roughness=0.4),
        box(f"{name}_stripe", at(-depth + 0.12, 0, 2.6), (width, 0.05, 0.12), brand, yaw, roughness=0.4),
        box(f"{name}_side_l", at(-depth / 2 - 0.5, -width / 2, 1.25), (0.1, depth - 1.0, 2.5), WHITE_WALL, yaw),
        box(f"{name}_side_r", at(-depth / 2 - 0.5, width / 2, 1.25), (0.1, depth - 1.0, 2.5), WHITE_WALL, yaw),
        box(f"{name}_screen", at(-depth + 0.12, 0, 1.6), (2.4, 0.06, 1.4), SCREEN_FRAME, yaw, roughness=0.2),
        box(f"{name}_screen_content", at(-depth + 0.16, 0, 1.6), (2.2, 0.02, 1.2), screen, yaw, roughness=0.2),
        box(f"{name}_counter", at(-1.0, width / 2 - 1.1, 0.5), (1.6, 0.6, 1.0), DESK, yaw, roughness=0.3),
        box(f"{name}_counter_band", at(-0.69, width / 2 - 1.1, 0.75), (1.6, 0.02, 0.2), brand, yaw),
        box(f"{name}_table", at(-1.8, -width / 2 + 1.2, 0.74), (1.2, 0.8, 0.04), WOOD, yaw),
        box(f"{name}_table_leg", at(-1.8, -width / 2 + 1.2, 0.36), (0.08, 0.08, 0.72), SCREEN_FRAME, yaw),
        box(f"{name}_rollup", at(-0.4, -width / 2 + 0.3, 1.0), (0.8, 0.04, 2.0), brand, yaw, roughness=0.4),
        box(f"{name}_rollup_base", at(-0.4, -width / 2 + 0.3, 0.04), (0.85, 0.25, 0.08), SCREEN_FRAME, yaw),
    ]
    for side in (-1, 1):
        px, py, _ = at(-depth + 0.5, side * (width / 2 - 0.5), 0)
        parts += [cylinder(f"{name}_pot_{side}", (px, py, 0), 0.25, 0.45, POT),
                  sphere(f"{name}_plant_{side}", (px, py, 0.95), 0.45, LEAVES, scale=(1, 1, 1.3), roughness=0.9)]
    if label:
        lx, ly, _ = at(-depth + 0.22, -width / 2 + 0.3, 0)
        parts += sign(f"{name}_label", label, (lx, ly, 3.04), facing - math.pi / 2 + math.pi, 0.055, WHITE_WALL)
    return parts


def hall_parts(ox: float, oy: float) -> list[MeshPart]:
    """Every static piece of the hall, around the robot at (ox, oy)."""
    # the carpet sits clearly above the physics ground plane (12 mm), or the two flicker in the viewers
    parts = [box("carpet", (ox + 9, oy, 0.006), (36, 30, 0.012), CARPET, roughness=0.95)]
    for x0 in range(-8, 28, 4):  # aisle markings
        parts.append(box(f"aisle_line_{x0}", (ox + x0 + 2, oy + 6.5, 0.0135), (3.0, 0.08, 0.003), (0.35, 0.37, 0.42)))
        parts.append(box(f"aisle_line_s{x0}", (ox + x0 + 2, oy - 6.5, 0.0135), (3.0, 0.08, 0.003), (0.35, 0.37, 0.42)))
    # the hall's outer walls (open on top, so the Newton viewer can look in)
    parts += [box("wall_back", (ox - 9, oy, 3.0), (0.3, 30, 6.0), (0.12, 0.13, 0.18)),
              box("wall_left", (ox + 9, oy + 15, 3.0), (36, 0.3, 6.0), (0.12, 0.13, 0.18)),
              box("wall_right", (ox + 9, oy - 15, 0.5), (36, 0.3, 1.0), (0.12, 0.13, 0.18))]  # low: the viewer looks over it
    # the entrance arch with the GITEX sign
    parts += [box("arch_left", (ox + 26, oy - 7.5, 3.5), (0.8, 1.0, 7.0), GITEX_GREEN, roughness=0.4),
              box("arch_right", (ox + 26, oy + 7.5, 3.5), (0.8, 1.0, 7.0), GITEX_GREEN, roughness=0.4),
              box("arch_top", (ox + 26, oy, 7.4), (0.8, 16.0, 1.6), (0.05, 0.08, 0.1), roughness=0.4)]
    parts += sign("gitex_sign", "GITEX 2026", (ox + 25.5, oy + 6.8, 6.75), -math.pi / 2, 0.17, GITEX_GREEN, depth=0.1)
    parts += sign("gitex_sign_out", "GITEX 2026", (ox + 26.5, oy - 6.8, 6.75), math.pi / 2, 0.17, GITEX_GREEN, depth=0.1)
    # the robot's own booth: Aideology, open towards the main aisle (+x)
    parts += booth("aideology", ox + 2.2, oy, 0.0, AIDEOLOGY, width=8.0, depth=5.0, screen=(0.45, 0.3, 0.95))
    parts += sign("aideology_sign", "AIDEOLOGY", (ox - 2.66, oy + 3.0, 2.97), -math.pi / 2, 0.075, (1, 1, 1))
    # exhibitors across the main aisle, facing the robot
    for n, by in enumerate((-9.0, -3.0, 3.0, 9.0)):
        parts += booth(f"east_{n}", ox + 9.0, oy + by, math.pi, BRANDS[n], screen=SCREEN_CONTENT[n % 4])
    # exhibitors along the side aisles, facing them
    for n, bx in enumerate((-4.5, 1.5, 13.5, 19.5)):
        parts += booth(f"north_{n}", ox + bx, oy + 10.0, -math.pi / 2, BRANDS[(n + 4) % 8], width=5.5, depth=4.0,
                       screen=SCREEN_CONTENT[(n + 1) % 4])
        parts += booth(f"south_{n}", ox + bx, oy - 10.0, math.pi / 2, BRANDS[(n + 2) % 8], width=5.5, depth=4.0,
                       screen=SCREEN_CONTENT[(n + 2) % 4])
    # a lounge area by the entrance: benches and a couple of tall tables
    for n, (lx, ly) in enumerate(((19.5, -3.0), (19.5, 3.0))):
        parts += [box(f"bench_{n}", (ox + lx, oy + ly, 0.23), (2.4, 0.6, 0.46), (0.3, 0.32, 0.38)),
                  cylinder(f"tall_table_{n}", (ox + lx + 2.0, oy + ly, 0), 0.05, 1.05, SCREEN_FRAME, segments=12),
                  cylinder(f"tall_table_top_{n}", (ox + lx + 2.0, oy + ly, 1.05), 0.4, 0.03, DESK, segments=24)]
    return parts


# products on the booth tables (YCB models, textured), as (url, booth table position offset)
PRODUCTS = ["025_mug.usd", "011_banana.usd", "006_mustard_bottle.usd", "024_bowl.usd", "035_power_drill.usd",
            "037_scissors.usd", "003_cracker_box.usd", "010_potted_meat_can.usd"]


def visitor_routes(ox: float, oy: float) -> list[list[Waypoint]]:
    """Looping routes along the aisles; some stop in front of the robot to look."""
    w = lambda x, y, pause=0.0: Waypoint(ox + x, oy + y, pause)  # noqa: E731
    return [
        [w(4.5, -13), w(4.5, 13)],  # up and down the main aisle, past the robot
        [w(5.8, 13), w(5.8, -13)],
        [w(4.0, -12), w(4.0, -1), w(1.6, 0.0, 5.0), w(4.0, 1), w(4.0, 12)],  # stops in front of the robot
        [w(-6, 6.5), w(24, 6.5)],  # the north side aisle
        [w(24, -6.5), w(-6, -6.5)],  # the south side aisle
        [w(-5, 6.5), w(5, 6.5), w(5, -6.5), w(16, -6.5), w(16, 6.5)],  # a loop round the booths
        [w(16, 13), w(16, -13)],  # the east aisle behind the exhibitors
        [w(5.2, 12), w(5.2, 1.0), w(1.7, -0.4, 6.0), w(5.2, -1.0), w(5.2, -12)],  # another visit to the robot
        [w(22, -10), w(22, 10), w(12, 6.5), w(12, -6.5)],
        [w(-6, -6.5), w(3.5, -6.5), w(3.5, 6.5), w(-6, 6.5)],
    ]


class GitexHall:
    """Builds the hall in the stage and in the Newton viewer, and moves the visitors every step."""

    def __init__(self, stage, origin: tuple[float, float], newton=None, visitors: int = 10, seed: int = 3) -> None:
        self._stage, self._origin, self._newton = stage, origin, newton
        self._random = random.Random(seed)
        os.makedirs(ASSET_DIR, exist_ok=True)
        ox, oy = origin
        self._static = merge_by_colour(hall_parts(ox, oy), prefix="gitex_hall")
        self._write_parts_usd(self._static, os.path.join(ASSET_DIR, "hall.usdc"), "/Hall")
        self._reference("/World/GITEX/hall", os.path.join(ASSET_DIR, "hall.usdc"), (0, 0, 0), 0)
        self._products = self._place_products(ox, oy)
        self._cycle = load_walk_cycle()
        routes = visitor_routes(ox, oy)[:visitors]
        self._visitors = []
        for n, route in enumerate(routes):
            start = self._random.uniform(0, 30)
            visitor = Visitor(route, self._cycle.walk_speed, self._cycle.cycle_m, len(self._cycle.frames),
                              self._cycle.standing_frame, start_s=start)
            texture = os.path.join(ASSET_DIR, f"outfit_{n}.png")
            from PIL import Image

            Image.fromarray(outfit_texture(*OUTFITS[n % len(OUTFITS)])).save(texture)
            path = self._write_visitor_usd(n, texture)
            self._reference(f"/World/GITEX/visitors/v{n}", path, (0, 0, 0), 0)
            self._visitors.append((visitor, f"/World/GITEX/visitors/v{n}", texture))
        self._uvs = region_uvs(self._cycle.regions)
        count = len(self._visitors)
        self._add_stall_visitor(count, ox, oy)
        # for the Newton viewer: one plain-coloured mesh per clothing colour, shared by all visitors (its texture path
        # draws them white), filled each frame from all the visitors' points (the stall visitor last)
        points_each = len(self._cycle.frames[0])
        self._colour_groups = visitor_colour_groups(self._cycle.indices, self._cycle.regions,
                                                    [OUTFITS[n % len(OUTFITS)] for n in range(count + 1)], points_each)
        self._all_points = np.zeros(((count + 1) * points_each, 3), dtype=np.float32)
        # their normals too: computed once per body shape (walk-cycle frame), then only turned with the visitor,
        # so the viewer does not recompute every visitor's normals at every redraw
        self._all_normals = np.zeros_like(self._all_points)
        self._all_normals[:, 2] = 1.0
        self._shape_normals: dict[object, np.ndarray] = {}
        self._last_frames: dict[int, int] = {}
        self._time = 0.0
        self._static_logged = False
        if newton is not None and getattr(newton, "_viewer", None) is not None:
            newton._viewer.show_ui = False  # the whole hall in view; press H in the viewer for its controls
        self._textured: list[str] = []  # textured product meshes whose texturing is turned on once they exist
        # the head camera's view of far visitors: their walking pose is refreshed in the stage only every
        # FAR_POSE_EVERY camera frames (they are a few pixels tall there); their place is updated every frame
        self.robot_xy: tuple[float, float] | None = None  # set by the scene before each step
        self._stage_updates = 0
        self._viewer_indices: dict[str, object] = {}  # each mesh's triangles on the GPU, sent once (they never change)
        self._held_hidden: set[str] = set()  # held objects already hidden in the viewer (no need to send them again)
        # with the viewer in its own process, the visitors' meshes are built there: here only each visitor's place,
        # heading and body shape are recorded (rows of x, y, yaw, shape, visible), and the held object's pose
        self._remote = bool(getattr(newton, "builds_people", False))
        if self._remote:
            shapes = list(self._cycle.frames) + [self._hold_points]  # body shapes: the walk cycle, then holding out
            self._hold_shape = len(shapes) - 1
            newton.setup_people(
                np.stack([np.asarray(points, dtype=np.float32) for points in shapes]),
                np.stack([vertex_normals(points, self._cycle.indices) for points in shapes]),
                self._colour_groups, count + 1,
                [(name, points, indices, HELD_COLOURS[name]) for name, (_p, _o, points, indices) in self._held.items()],
            )
            self._people_rows = np.zeros((count + 1, 5), dtype=np.float32)
            self._held_row = np.full(5, -1.0, dtype=np.float32)  # held object number, centre x, y, z, yaw
            self._held_names = list(self._held)

    # --- the stage ---

    def _reference(self, path: str, url: str, position, yaw_deg: float, rotate_x: float = 0.0):
        from pxr import Gf, UsdGeom

        xform = UsdGeom.Xform.Define(self._stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateZOp().Set(yaw_deg)
        model = self._stage.DefinePrim(f"{path}/model")
        model.GetReferences().AddReference(url)
        if rotate_x:
            UsdGeom.Xformable(model).AddRotateXOp().Set(rotate_x)
        return model

    @staticmethod
    def _material(stage, path: str, colour, roughness: float, texture: str | None = None):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        if texture:
            reader = UsdShade.Shader.Define(stage, f"{path}/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            image = UsdShade.Shader.Define(stage, f"{path}/image")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(image.ConnectableAPI(), "rgb")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    @classmethod
    def _write_mesh(cls, stage, path: str, points, indices, colour, roughness, uvs=None, texture=None):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(indices, dtype=np.int32)))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*colour)])
        if uvs is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
                Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uvs, dtype=np.float32)))
        material = cls._material(stage, f"{path}_material", colour, roughness, texture)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return mesh

    def _write_parts_usd(self, parts: list[MeshPart], path: str, root: str) -> None:
        from pxr import Usd, UsdGeom

        if os.path.exists(path):
            os.remove(path)
        stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, root)
        for part in parts:
            self._write_mesh(stage, f"{root}/{part.name}", part.points, part.indices, part.colour, part.roughness)
        stage.SetDefaultPrim(stage.GetPrimAtPath(root))
        stage.GetRootLayer().Save()

    def _write_visitor_usd(self, n: int, texture: str) -> str:
        from pxr import Usd, UsdGeom

        path = os.path.join(ASSET_DIR, f"visitor_{n}.usdc")
        if os.path.exists(path):
            os.remove(path)
        stage = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/Visitor")
        cycle = self._cycle if hasattr(self, "_cycle") else load_walk_cycle()
        self._write_mesh(stage, "/Visitor/body", cycle.frames[cycle.standing_frame], cycle.indices, (0.8, 0.8, 0.8),
                         0.8, uvs=region_uvs(cycle.regions), texture=texture)
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Visitor"))
        stage.GetRootLayer().Save()
        return path

    def _place_products(self, ox: float, oy: float) -> list[str]:
        """YCB products on the exhibitors' tables (textured models, loaded from NVIDIA's asset server)."""
        paths = []
        tables = []
        for n, by in enumerate((-9.0, -3.0, 3.0, 9.0)):  # east booths: table at forward -1.8, side -(6/2 - 1.2)
            tables.append((ox + 9.0 + 1.8, oy + by - 1.8))
        for n, bx in enumerate((-4.5, 1.5, 13.5, 19.5)):
            tables.append((ox + bx - 1.55, oy + 10.0 + 1.8))
            tables.append((ox + bx + 1.55, oy - 10.0 - 1.8))
        for n, (tx, ty) in enumerate(tables):
            for k in range(2):
                name = PRODUCTS[(2 * n + k) % len(PRODUCTS)]
                path = f"/World/GITEX/products/p{n}_{k}"
                self._reference(path, f"{YCB}/{name}", (tx - 0.25 + 0.5 * k, ty, 0.76 + 0.08), self._random.uniform(0, 360),
                                rotate_x=90.0)
                paths.append(path)
        return paths

    def _add_stall_visitor(self, n: int, ox: float, oy: float) -> None:
        """The test visitor (hidden until sent) and the objects they can carry (hidden until carried)."""
        from pxr import UsdGeom

        from surroundings import SHOWABLE, apple_usd

        texture = os.path.join(ASSET_DIR, f"outfit_{n}.png")
        from PIL import Image

        Image.fromarray(outfit_texture(*OUTFITS[n % len(OUTFITS)])).save(texture)
        self._stall_path = "/World/GITEX/stall_visitor"
        self._reference(self._stall_path, self._write_visitor_usd(n, texture), (0, 0, HIDDEN_Z), 0)
        UsdGeom.Imageable(self._stage.GetPrimAtPath(self._stall_path)).MakeInvisible()
        standing = self._cycle.frames[self._cycle.standing_frame]
        self._hold_points, hand_ids, self._hold_hand = hold_out_pose(standing)
        self._hand_track = hand_track(self._cycle.frames, hand_ids)
        # they stand a little to the robot's left, so the object in their left hand is in the middle of its view
        self._visit = StallVisit(entry=(ox + 4.6, oy - 5.0), stop=(ox + 1.5, oy + 0.27), exit=(ox + 4.6, oy + 5.0),
                                 robot=(ox, oy), speed=self._cycle.walk_speed, cycle_m=self._cycle.cycle_m,
                                 frames=len(self._cycle.frames), standing_frame=self._cycle.standing_frame,
                                 max_stay_s=600.0)  # a whole conversation; the tests send "leave" when done
        self._visit_pose = VisitPose("idle")
        self._held: dict[str, tuple] = {}  # name: (prim path, centre offset from its origin, Newton mesh)
        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for name in HELD_COLOURS:
            url, upright = SHOWABLE[name]
            path = f"/World/GITEX/held/{name}"
            self._reference(path, apple_usd() if url == "apple" else url, (0, 0, HIDDEN_Z), 0, rotate_x=upright)
            centre = np.array(cache.ComputeWorldBound(self._stage.GetPrimAtPath(path)).ComputeAlignedRange().GetMidpoint())
            offset = centre - np.array([0.0, 0.0, HIDDEN_Z])
            pieces = stage_meshes(self._stage, path)
            points = np.concatenate([piece[0] for piece in pieces]) - centre  # around its middle
            indices, start = [], 0
            for piece in pieces:
                indices.append(piece[1] + start)
                start += len(piece[0])
            self._held[name] = (path, offset, points.astype(np.float32), np.concatenate(indices).astype(np.int32))
            UsdGeom.Imageable(self._stage.GetPrimAtPath(path)).MakeInvisible()
        self._shown_held: str | None = None

    def request_visit(self, held: str) -> bool:
        """Send the test visitor to the stall carrying ``held`` (a HELD_COLOURS name); False if one is there."""
        return held in HELD_COLOURS and self._visit.request(held, self._time)

    def end_visit(self) -> None:
        self._visit.leave(self._time)

    def people_xy(self) -> list[tuple[float, float] | None]:
        """Where each visitor stands on the floor (world x, y), the stall visitor last (None while not sent)."""
        people: list[tuple[float, float] | None] = [visitor.pose(self._time)[:2] for visitor, _, _ in self._visitors]
        pose = self._visit_pose
        people.append(None if pose.state == "idle" else (pose.x, pose.y))
        return people

    def visit_state(self) -> str:
        """ "idle", or the visitor's state and what they carry, e.g. "at_stall apple"."""
        pose = self._visit_pose
        return pose.state if pose.state == "idle" else f"{pose.state} {pose.held}"

    def _stall_visitor_points(self, pose: VisitPose) -> tuple[np.ndarray, object, np.ndarray]:
        """(body points in the character's frame, a key for the body shape, the hand in the character's frame)."""
        if pose.frame is None:
            return self._hold_points, "hold", self._hold_hand
        return self._cycle.frames[pose.frame], pose.frame, self._hand_track[pose.frame]

    def _move_held(self, pose: VisitPose, hand_world: np.ndarray, stage: bool, viewer: bool) -> None:
        """The carried object in the visitor's hand (its middle a few centimetres above the hand's middle)."""
        from pxr import Gf, UsdGeom

        name = pose.held if pose.state != "idle" else None
        if stage and name != self._shown_held:
            for other, (path, *_rest) in self._held.items():
                prim = UsdGeom.Imageable(self._stage.GetPrimAtPath(path))
                prim.MakeVisible() if other == name else prim.MakeInvisible()
            self._shown_held = name
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        if viewer and self._remote:
            if name is None:
                self._held_row[0] = -1.0
            else:
                centre = hand_world + np.array([0.0, 0.0, 0.04])
                self._held_row[:] = (self._held_names.index(name), centre[0], centre[1], centre[2], pose.yaw)
            viewer = False  # the meshes are built in the viewer process
        for other, (path, offset, points, indices) in self._held.items():
            centre = hand_world + np.array([0.0, 0.0, 0.04]) if other == name else np.array([0.0, 0.0, HIDDEN_Z])
            if stage and other == name:
                turned = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]])
                ops = UsdGeom.Xformable(self._stage.GetPrimAtPath(path)).GetOrderedXformOps()
                ops[0].Set(Gf.Vec3d(*(centre - turned).tolist()))
                ops[1].Set(math.degrees(pose.yaw))
            if viewer and self._newton is not None:
                if other != name and other in self._held_hidden:
                    continue  # still hidden where it was: nothing to send
                self._held_hidden.discard(other)
                if other != name:
                    self._held_hidden.add(other)
                world = np.empty_like(points)
                world[:, 0] = centre[0] + c * points[:, 0] - s * points[:, 1]
                world[:, 1] = centre[1] + s * points[:, 0] + c * points[:, 1]
                world[:, 2] = centre[2] + points[:, 2]
                self._log(f"/gitex/held_{other}", world, indices, HELD_COLOURS[other], 0.5, dynamic=True)

    # --- the Newton viewer ---

    def _log(self, name: str, points: np.ndarray, indices: np.ndarray, colour, roughness=0.7, uvs=None,
             texture=None, dynamic: bool = False, normals: np.ndarray | None = None) -> None:
        if getattr(self._newton, "accepts_numpy", False):  # the viewer in its own process (viewer_process.py)
            self._newton.log_mesh(name, points, indices, colour, roughness, uvs=uvs, texture=texture, dynamic=dynamic,
                                  normals=normals)
            return
        import warp as wp

        # the viewer's own device (the sim may run on the CPU): mixing devices crashes Warp
        device = getattr(getattr(self._newton, "_viewer", None), "device", None) or (
            "cuda:0" if wp.is_cuda_available() else "cpu")
        # moving meshes (visitors, held objects) keep their triangles: send those to the GPU once, not every frame
        index_array = self._viewer_indices.get(name) if dynamic else None
        if index_array is None or len(index_array) != len(indices):
            index_array = wp.array(np.ascontiguousarray(indices, dtype=np.int32), dtype=wp.int32, device=device)
            if dynamic:
                self._viewer_indices[name] = index_array
        self._newton.log_mesh(
            name,
            wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=device),
            index_array,
            normals=None if normals is None else wp.array(np.ascontiguousarray(normals, dtype=np.float32),
                                                          dtype=wp.vec3, device=device),
            uvs=None if uvs is None else wp.array(np.ascontiguousarray(uvs, dtype=np.float32), dtype=wp.vec2, device=device),
            texture=texture, color=tuple(colour), roughness=roughness, dynamic=dynamic, backface_culling=False,
        )

    def _log_static(self) -> None:
        for part in self._static:
            self._log(f"/gitex/{part.name}", part.points, part.indices, part.colour, part.roughness)
        for path in self._products:
            for number, piece in enumerate(stage_meshes(self._stage, path)):
                self._log(f"/gitex{path}_{number}", *piece)
                self._textured.append(f"/gitex{path}_{number}")

    def _log_shown(self) -> None:
        """The product on the stand in front of the robot (/scene/show), whatever is there now."""
        pieces = stage_meshes(self._stage, "/World/Shown") if self._stage.GetPrimAtPath("/World/Shown") else []
        for number in range(8):
            if number < len(pieces):
                self._log(f"/gitex/shown_{number}", *pieces[number])
                if pieces[number][4] is not None and f"/gitex/shown_{number}" not in self._textured:
                    self._textured.append(f"/gitex/shown_{number}")
            else:
                self._log(f"/gitex/shown_{number}", np.zeros((3, 3), np.float32), np.array([0, 1, 2], np.int32),
                          (0, 0, 0))  # nothing there

    # --- every step ---

    def step(self, dt: float, shown_changed: bool = False, stage: bool = True, viewer: bool = True) -> None:
        """Move the visitors by ``dt``. ``stage``: update them in the USD stage (only needed before the camera
        renders); ``viewer``: update them in the Newton viewer (only needed before it redraws)."""
        from pxr import Gf, UsdGeom, Vt

        self._time += dt
        if self._newton is not None and not self._static_logged:
            self._log_static()
            self._log_shown()
            self._static_logged = True
        elif self._newton is not None and shown_changed:
            self._log_shown()
        if self._newton is not None and self._textured:
            for name in list(self._textured):
                self._enable_texture(name)
        if not (stage or (viewer and self._newton is not None)):
            return
        if stage:
            self._stage_updates += 1
        for n, (visitor, path, texture) in enumerate(self._visitors):
            x, y, yaw, frame = visitor.pose(self._time)
            points = self._cycle.frames[frame]
            if stage:
                ops = UsdGeom.Xformable(self._stage.GetPrimAtPath(path)).GetOrderedXformOps()
                ops[0].Set(Gf.Vec3d(x, y, 0.0))
                ops[1].Set(math.degrees(yaw))
                far = self.robot_xy is not None and math.hypot(x - self.robot_xy[0], y - self.robot_xy[1]) > FAR_VISITOR_M
                if self._last_frames.get(n) != frame and (not far or self._stage_updates % FAR_POSE_EVERY == 0):
                    body = UsdGeom.Mesh(self._stage.GetPrimAtPath(f"{path}/model/body"))
                    body.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                    self._last_frames[n] = frame
            if viewer and self._remote:
                self._people_rows[n] = (x, y, yaw, frame, 1.0)
            elif viewer and self._newton is not None:
                c, s = math.cos(yaw), math.sin(yaw)
                world = self._all_points[n * len(points):(n + 1) * len(points)]
                world[:, 0] = x + c * points[:, 0] - s * points[:, 1]
                world[:, 1] = y + s * points[:, 0] + c * points[:, 1]
                world[:, 2] = points[:, 2]
                self._turn_normals(n, frame, points, c, s)
        self._step_stall_visitor(stage, viewer)
        if viewer and self._remote:
            self._newton.publish_people(self._people_rows, self._held_row)
        elif viewer and self._newton is not None and self._visitors:
            for number, (colour, vertex_ids, local) in enumerate(self._colour_groups):
                self._log(f"/gitex/visitors_{number}", self._all_points[vertex_ids], local, colour, 0.8, dynamic=True,
                          normals=self._all_normals[vertex_ids])

    def _turn_normals(self, n: int, key, points: np.ndarray, c: float, s: float) -> None:
        """Visitor ``n``'s normals in the world: its body shape's normals (computed once), turned by its heading."""
        local = self._shape_normals.get(key)
        if local is None:
            local = self._shape_normals[key] = vertex_normals(points, self._cycle.indices)
        out = self._all_normals[n * len(points):(n + 1) * len(points)]
        out[:, 0] = c * local[:, 0] - s * local[:, 1]
        out[:, 1] = s * local[:, 0] + c * local[:, 1]
        out[:, 2] = local[:, 2]

    def _step_stall_visitor(self, stage: bool, viewer: bool) -> None:
        from pxr import Gf, UsdGeom, Vt

        pose = self._visit_pose = self._visit.update(self._time)
        n, path = len(self._visitors), self._stall_path
        points_each = len(self._cycle.frames[0])
        world = self._all_points[n * points_each:(n + 1) * points_each]
        if pose.state == "idle":
            if stage and self._last_frames.get(n) != "gone":
                UsdGeom.Imageable(self._stage.GetPrimAtPath(path)).MakeInvisible()
                self._last_frames[n] = "gone"
            world[:, 2] = HIDDEN_Z
            if self._remote:
                self._people_rows[n, 4] = 0.0  # not there
            self._move_held(pose, np.zeros(3), stage, viewer)
            return
        points, key, hand = self._stall_visitor_points(pose)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        if stage:
            prim = self._stage.GetPrimAtPath(path)
            if self._last_frames.get(n) == "gone":
                UsdGeom.Imageable(prim).MakeVisible()
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(pose.x, pose.y, 0.0))
            ops[1].Set(math.degrees(pose.yaw))
            if self._last_frames.get(n) != key:
                UsdGeom.Mesh(self._stage.GetPrimAtPath(f"{path}/model/body")).GetPointsAttr().Set(
                    Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, dtype=np.float32)))
                self._last_frames[n] = key
        if viewer and self._remote:
            self._people_rows[n] = (pose.x, pose.y, pose.yaw, self._hold_shape if key == "hold" else key, 1.0)
        elif viewer:
            world[:, 0] = pose.x + c * points[:, 0] - s * points[:, 1]
            world[:, 1] = pose.y + s * points[:, 0] + c * points[:, 1]
            world[:, 2] = points[:, 2]
            self._turn_normals(n, key, points, c, s)
        hand_world = np.array([pose.x + c * hand[0] - s * hand[1], pose.y + s * hand[0] + c * hand[1], hand[2]])
        self._move_held(pose, hand_world, stage, viewer)

    def _enable_texture(self, name: str) -> None:
        """Newton's log_mesh loads a mesh's texture but never turns on texturing (the material's 4th value), so
        a textured mesh draws plain white; turn it on for the viewer's copy of the mesh, once it exists."""
        viewer = getattr(self._newton, "_viewer", None)
        objects = getattr(viewer, "objects", None)
        if objects is None:
            return
        mesh = objects.get(viewer._qualify(name)) if hasattr(viewer, "_qualify") else objects.get(name)
        if mesh is not None and mesh.material[3] != 1.0:
            roughness, metallic, checker, _ = mesh.material
            mesh.material = (roughness, metallic, checker, 1.0)


def vertex_normals(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Unit vertex normals (N, 3) of a triangle mesh: the area-weighted sum of the faces around each vertex."""
    triangles = np.asarray(indices).reshape(-1, 3)
    corners = np.asarray(points, dtype=np.float64)[triangles]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])  # length = 2 x area
    normals = np.zeros((len(points), 3))
    for k in range(3):
        np.add.at(normals, triangles[:, k], faces)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(length > 1e-12, normals / np.maximum(length, 1e-12), np.array([0.0, 0.0, 1.0]))
    return normals.astype(np.float32)


def visitor_colour_groups(indices: np.ndarray, regions: np.ndarray, outfits: list, points_each: int) -> list[tuple]:
    """Split all visitors' triangles by clothing colour: ``[(colour, vertex ids into all visitors' points, triangle
    indices into those vertices)]``. Visitor ``n`` wears ``outfits[n]`` (a colour per body region) and owns points
    ``n * points_each`` to ``(n + 1) * points_each``; a triangle takes the region of its first point."""
    triangles = indices.reshape(-1, 3)
    triangle_region = regions[triangles[:, 0]]
    by_colour: dict[tuple, list[np.ndarray]] = {}
    for n, outfit in enumerate(outfits):
        for region, colour in enumerate(outfit):
            chosen = triangles[triangle_region == region]
            if len(chosen):
                by_colour.setdefault(tuple(colour), []).append(chosen + n * points_each)
    groups = []
    for colour, parts in by_colour.items():
        tris = np.concatenate(parts)
        vertex_ids, local = np.unique(tris.ravel(), return_inverse=True)
        groups.append((colour, vertex_ids.astype(np.int64), local.astype(np.int32)))
    return groups

def stage_meshes(stage, root: str) -> list[tuple]:
    """Every mesh under ``root`` in world coordinates, for the Newton viewer: (points, indices, colour, roughness,
    uvs, texture), one vertex per triangle corner so face-varying texture coordinates work."""
    from pxr import Usd, UsdGeom, UsdShade

    cache = UsdGeom.XformCache()
    pieces = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        if len(points) == 0:
            continue
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        matrix = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        world = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        corners = triangulate(counts, np.arange(len(face_indices)))  # corner numbers, triangulated
        corner_points = world[face_indices[corners]]
        uvs, texture, colour = None, None, (0.7, 0.7, 0.7)
        primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if primvar and primvar.HasValue():
            values = np.array(primvar.ComputeFlattened(), dtype=np.float32)
            interpolation = primvar.GetInterpolation()
            if interpolation == UsdGeom.Tokens.faceVarying and len(values) == len(face_indices):
                uvs = values[corners]
            elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(values) == len(points):
                uvs = values[face_indices[corners]]
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            for shader_prim in Usd.PrimRange(material.GetPrim()):
                shader = UsdShade.Shader(shader_prim)
                if not shader:
                    continue
                for name in ("file", "diffuse_texture"):
                    texture_input = shader.GetInput(name)
                    if texture_input and texture_input.Get() is not None:
                        texture = _local_texture(texture_input.Get())
                colour_input = shader.GetInput("diffuseColor") or shader.GetInput("diffuse_color_constant")
                if colour_input and colour_input.Get() is not None:
                    colour = tuple(float(v) for v in colour_input.Get())
        if uvs is None:
            texture = None
        else:
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]  # USD's texture origin is bottom-left, the viewer's top-left
        pieces.append((corner_points.astype(np.float32), np.arange(len(corners), dtype=np.int32), colour, 0.6, uvs,
                       texture))
    return pieces


def _local_texture(asset) -> str | None:
    """A texture file the viewer can open: local files as they are, remote ones downloaded once."""
    path = asset.resolvedPath or asset.path
    if not path:
        return None
    if path.startswith("http"):
        local = os.path.join(ASSET_DIR, "textures", path.split("/Axis_Aligned/")[-1].replace("/", "_"))
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            urllib.request.urlretrieve(path, local)
        return local
    return path if os.path.exists(path) else None
