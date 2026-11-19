# SPDX-License-Identifier: Apache-2.0
"""What IsaacLabSuturebot reads from this package: names, fields, parameters and defaults.

That repository builds its Isaac Lab task, its reference recordings and its benchmarks on
the scene pool, the grasp controller and the thread helpers named in the tables below.
Every check is an import or an ``inspect`` of a signature; no scene is built. A change
that fails here has to be made in that repository as well.
"""

import dataclasses
import importlib
import inspect
import types
import unittest

import warp as wp

PACKAGE = "newton_solver_suturebot"
PARALLEL = f"{PACKAGE}.examples.grasp.parallel"
PROCEDURE = f"{PACKAGE}.examples.grasp._grasp_procedure"
LAYOUT = f"{PACKAGE}.examples.grasp._dvrk_layout"
DRIVE = f"{PACKAGE}.examples.needle._needle_insertion_drive"
INSERTION_GEOMETRY = f"{PACKAGE}.examples.needle._needle_insertion_geometry"
THREAD_PINS = f"{PACKAGE}.examples.needle._thread_pins"
THREAD_VIEW = f"{PACKAGE}.examples.needle._thread_view"

REQUIRED = inspect.Parameter.empty
POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
KEYWORD = inspect.Parameter.KEYWORD_ONLY
VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

# Every name it imports, with the module that holds it here.
NAMES = (
    (PACKAGE, "NeedleGeometry"),
    (PACKAGE, "SolverThread"),
    (PACKAGE, "ThreadModel"),
    (PACKAGE, "ThreadTissueCoupling"),
    (PACKAGE, "load_mesh_regions"),
    (PARALLEL, "SutureSceneConfig"),
    (PARALLEL, "ParallelSutureScenes"),
    (PARALLEL, "_make_scene"),
    (PARALLEL, "_apply_action"),
    (PARALLEL, "_observe"),
    (PARALLEL, "_geometry"),
    (PROCEDURE, "ProceduralGraspController"),
    (PROCEDURE, "gripper_target"),
    (LAYOUT, "source_layout_asset_paths"),
    (DRIVE, "ramp_distance"),
    (INSERTION_GEOMETRY, "_distance_to_triangles"),
    (THREAD_PINS, "bridge"),
    (THREAD_PINS, "lay_along_path"),
    (THREAD_PINS, "serpentine"),
    (THREAD_VIEW, "_sample_centerline"),
)

# The fields of SutureSceneConfig with their defaults, each in value and in type: it writes
# ``asdict`` of its configs, most fields left at their defaults, into its records as JSON.
# A new field needs a default.
SCENE_FIELDS = {
    "device": "cuda:0",
    "asset": None,
    "region": 1,
    "entry_xy": None,
    "exit_xy": None,
    "substeps": 4,
    "frame_dt": 1.0 / 60.0,
    "translation_step": 0.0001,
    "rotation_step": 0.01,
    "reset_position_noise_m": 0.0,
    "seed": 0,
    "control_mode": "free_needle",
    "rigid_backend": "native",
    "needle_weight": True,
    "initial_grasp": None,
    "jaw_step_rad": 0.02,
    "thread_elements": 90,
    "thread_length_m": 0.18,
    "thread_radius_m": 0.00015,
    "thread_modulus_pa": 2.0e6,
    "example_arguments": (),
}

# The parameters after ``self`` of the pool's step, and of the step of the one scene in the
# caller's process, which takes the same calls. A new parameter needs a default.
STEPS = {
    "LocalSutureScene": (
        ("actions", POSITIONAL, REQUIRED),
        ("include_tissue", KEYWORD, True),
        ("include_thread_diagnostics", KEYWORD, True),
        # The next three exist only here. It does not import this class and passes none of them.
        ("deadline", KEYWORD, None),
        ("unit", KEYWORD, "attempt"),
        ("observe_native_trials", KEYWORD, False),
        ("commands", KEYWORD, None),
    ),
    "ParallelSutureScenes": (
        ("actions", POSITIONAL, REQUIRED),
        ("include_tissue", KEYWORD, True),
        ("include_thread_diagnostics", KEYWORD, True),
        ("commands", KEYWORD, None),
    ),
}

# The constructor of the grasp controller after ``self``. Every further parameter is
# keyword-only with a default.
CONTROLLER = (
    ("geometry", POSITIONAL, REQUIRED),
    ("config", POSITIONAL, REQUIRED),
    ("speed", POSITIONAL, 0.003),
    ("settle_seconds", POSITIONAL, 0.5),
    ("lift_height", KEYWORD, 0.0),
)

# How it calls them: the number of positional arguments and the keywords of each call.
CALLS = (
    (PARALLEL, "SutureSceneConfig", 0, ()),
    (PARALLEL, "SutureSceneConfig", 0, ("device", "control_mode", "substeps", "example_arguments")),
    (
        PARALLEL,
        "SutureSceneConfig",
        0,
        ("device", "control_mode", "frame_dt", "substeps", "reset_position_noise_m", "seed"),
    ),
    (PARALLEL, "ParallelSutureScenes", 1, ("num_envs", "log_dir")),
    (PARALLEL, "ParallelSutureScenes.__enter__", 0, ()),
    (PARALLEL, "ParallelSutureScenes.__exit__", 3, ()),
    (PARALLEL, "ParallelSutureScenes.observe", 0, ()),
    (PARALLEL, "ParallelSutureScenes.observe", 0, ("include_tissue",)),
    (PARALLEL, "ParallelSutureScenes.geometry", 0, ()),
    (PARALLEL, "ParallelSutureScenes.step", 1, ()),
    (PARALLEL, "ParallelSutureScenes.step", 1, ("include_tissue",)),
    (PARALLEL, "ParallelSutureScenes.reset", 0, ()),
    (PARALLEL, "ParallelSutureScenes.reset", 1, ()),
    (PARALLEL, "ParallelSutureScenes.close", 0, ()),
    (PARALLEL, "_make_scene", 2, ()),
    (PARALLEL, "_apply_action", 3, ()),
    (PARALLEL, "_observe", 2, ("include_tissue",)),
    (PARALLEL, "_geometry", 2, ()),
    (PROCEDURE, "ProceduralGraspController", 2, ()),
    (PROCEDURE, "ProceduralGraspController", 2, ("settle_seconds",)),
    (PROCEDURE, "ProceduralGraspController", 2, ("speed", "settle_seconds", "lift_height")),
    (PROCEDURE, "ProceduralGraspController.__call__", 1, ()),
    (PROCEDURE, "ProceduralGraspController.report", 0, ()),
    (PROCEDURE, "gripper_target", 2, ()),
    (LAYOUT, "source_layout_asset_paths", 0, ()),
    (DRIVE, "ramp_distance", 2, ()),
    (INSERTION_GEOMETRY, "_distance_to_triangles", 2, ()),
    (THREAD_PINS, "serpentine", 7, ()),
    (THREAD_PINS, "bridge", 5, ()),
    (THREAD_PINS, "lay_along_path", 3, ()),
    (PACKAGE, "NeedleGeometry.load", 0, ()),
    (PACKAGE, "load_mesh_regions", 1, ()),
    (PACKAGE, "load_mesh_regions", 2, ()),
    (PACKAGE, "ThreadModel.create", 1, ("young_modulus", "radius", "density", "device")),
    (
        PACKAGE,
        "SolverThread",
        2,
        (
            "gravity",
            "mass_damping",
            "max_iterations",
            "tolerance",
            "contact_distance",
            "contact_stiffness",
            "contact_smoothing",
            "friction",
            "self_contact",
        ),
    ),
    (PACKAGE, "SolverThread.step", 5, ()),
    (PACKAGE, "ThreadTissueCoupling", 5, ("tissue_stiffness", "contact_margin", "max_tracks")),
    (PACKAGE, "ThreadTissueCoupling.update", 1, ()),
    (PACKAGE, "ThreadTissueCoupling.add_forces", 1, ()),
)

# The defaults it relies on: every parameter that its calls above leave to the package, but
# those of the config, the pool's step and the controller's constructor, which have their own tables.
LEFT_OUT = (
    (PARALLEL, "ParallelSutureScenes", {"timeout": 600.0}),
    (
        PARALLEL,
        "ParallelSutureScenes.observe",
        {"env_ids": None, "include_tissue": True, "include_thread_diagnostics": True},
    ),
    (PARALLEL, "ParallelSutureScenes.geometry", {"env_id": 0}),
    (PARALLEL, "ParallelSutureScenes.reset", {"env_ids": None}),
    (PARALLEL, "_observe", {"include_thread_diagnostics": True, "body_q": None}),
    (DRIVE, "ramp_distance", {"ramp_time": 0.5}),
    (THREAD_PINS, "bridge", {"reach": 1.0 / 3.0}),
    (PACKAGE, "NeedleGeometry.load", {"asset_path": None}),
    (PACKAGE, "load_mesh_regions", {"region_ids": None}),
    (
        PACKAGE,
        "SolverThread",
        {
            "n_envs": 1,
            "fixed_dofs": (),
            "max_eyelets": 0,
            "eyelet_stiffness": 0.0,
            "obstacle_distance": None,
            "multiples": (4.0, 2.0, 1.0, 0.5, 0.25, 1.0 / 16.0, 1.0 / 64.0, 0.0),
        },
    ),
    (PACKAGE, "ThreadTissueCoupling.update", {"body_q": None}),
)

# Attributes it reads from the classes without a call: dataclass fields and properties.
ATTRIBUTES = (
    (PARALLEL, "SutureSceneConfig", ("action_dimension",)),
    (PACKAGE, "NeedleGeometry", ("asset_path", "centerline", "arc_normal", "span", "thickness", "tip")),
    (PACKAGE, "ThreadModel", ("lengths",)),
)

# The inputs of a launch of the kernel that samples the thread's centerline, an array by the
# type of its elements: it allocates the two arrays of points itself.
CENTERLINE_INPUTS = [
    ("q", wp.float64),
    ("lengths", wp.float64),
    ("samples", int),
    ("centers", wp.vec3f),
    ("nodes", wp.vec3f),
]


def _resolve(module, path):
    """The object at a dotted path of a module, and the object that holds it."""
    holder = found = importlib.import_module(module)
    for name in path.split("."):
        holder, found = found, getattr(found, name)
    return holder, found


def _parameters(function):
    """Name, kind and default of every parameter of a method after ``self``."""
    parameters = list(inspect.signature(function).parameters.values())[1:]
    return [(item.name, item.kind, item.default) for item in parameters]


def _signature_of_a_call(module, path):
    """The signature a caller binds to: a method of an instance without its ``self``."""
    holder, found = _resolve(module, path)
    signature = inspect.signature(found)
    name = path.rpartition(".")[2]
    if inspect.isclass(holder) and isinstance(inspect.getattr_static(holder, name), types.FunctionType):
        signature = signature.replace(parameters=list(signature.parameters.values())[1:])
    return signature


class ConsumerSurfaceTests(unittest.TestCase):
    def assert_same(self, found, pinned):
        """Equal in value and in type: 4.0 for 4, or 1.0 for True, is a change."""
        self.assertEqual(found, pinned)
        self.assertIs(type(found), type(pinned), f"{found!r} in place of {pinned!r}")

    def assert_parameters(self, function, pinned, *, others_keyword_only=False):
        """The pinned parameters are as written; every other one has a default and, if asked, is keyword-only."""
        found = _parameters(function)
        positional = [row for row in pinned if row[1] is POSITIONAL]
        self.assertEqual(found[: len(positional)], positional)
        for row in pinned:
            self.assertIn(row, found)
        for name, kind, default in found:
            if (name, kind, default) in pinned:
                continue
            with self.subTest(parameter=name):
                if others_keyword_only:
                    self.assertIs(kind, KEYWORD)
                if kind not in VARIADIC:
                    self.assertIsNot(default, REQUIRED, "a new parameter needs a default")
        return found

    def test_names_are_at_their_modules(self):
        for module, name in NAMES:
            with self.subTest(module=module, name=name):
                self.assertTrue(hasattr(importlib.import_module(module), name))

    def test_scene_config_fields(self):
        config = _resolve(PARALLEL, "SutureSceneConfig")[1]
        fields = {field.name: field for field in dataclasses.fields(config)}
        for name, default in SCENE_FIELDS.items():
            with self.subTest(field=name):
                self.assertIn(name, fields.keys())
                self.assert_same(fields[name].default, default)
        for name in fields.keys() - SCENE_FIELDS.keys():
            with self.subTest(field=name):
                missing = (fields[name].default, fields[name].default_factory)
                self.assertNotEqual(missing, (dataclasses.MISSING, dataclasses.MISSING), "a new field needs a default")

    def test_step_parameters(self):
        for scene, pinned in STEPS.items():
            with self.subTest(scene=scene):
                self.assert_parameters(_resolve(PARALLEL, f"{scene}.step")[1], pinned)

    def test_grasp_controller_constructor(self):
        constructor = _resolve(PROCEDURE, "ProceduralGraspController.__init__")[1]
        found = self.assert_parameters(constructor, CONTROLLER, others_keyword_only=True)
        defaults = {name: default for name, _, default in found}
        # It builds the arc with an optional lift, so the pull-through stays off unless asked for.
        if "pull_through" in defaults:
            self.assertIs(defaults["pull_through"], False)

    def test_calls_bind(self):
        for module, path, positional, keywords in CALLS:
            with self.subTest(call=path, positional=positional, keywords=keywords):
                signature = _signature_of_a_call(module, path)
                try:
                    signature.bind(*[None] * positional, **dict.fromkeys(keywords))
                except TypeError as error:
                    raise self.failureException(f"{path}{signature}: {error}") from None

    def test_defaults_it_leaves_out(self):
        for module, path, pinned in LEFT_OUT:
            for name, default in pinned.items():
                with self.subTest(call=path, parameter=name):
                    parameters = _signature_of_a_call(module, path).parameters
                    self.assertIn(name, parameters)
                    self.assert_same(parameters[name].default, default)

    def test_attributes_are_fields_or_properties(self):
        for module, path, names in ATTRIBUTES:
            owner = _resolve(module, path)[1]
            fields = {field.name for field in dataclasses.fields(owner)}
            for name in names:
                with self.subTest(owner=path, attribute=name):
                    if name not in fields:
                        self.assertIsInstance(inspect.getattr_static(owner, name, None), property)

    def test_centerline_kernel_inputs(self):
        kernel = _resolve(THREAD_VIEW, "_sample_centerline")[1]
        self.assertIsInstance(kernel, wp.Kernel)
        inputs = inspect.signature(kernel.func).parameters.values()
        found = [(item.name, getattr(item.annotation, "dtype", item.annotation)) for item in inputs]
        self.assertEqual(found, CENTERLINE_INPUTS)


if __name__ == "__main__":
    unittest.main()
