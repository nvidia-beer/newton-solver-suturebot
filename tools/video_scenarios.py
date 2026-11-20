"""Script public example interactions for representative FEM recordings."""

import numpy as np


def configure(name, example):
    if name == "fem_surgical_incision":
        example.recording_events = []
        example.recording_drag = None
    if name == "fem_phantom_regions":
        example.recording_events = []
        example.recording_drag = None
    elif name in ("fem_dvrk", "fem_versius"):
        example.recording_seen_frames = set()
    if name in (
        "fem_phantom_regions",
        "fem_dvrk",
        "fem_versius",
        "fem_needle_insertion",
        "fem_surgical_incision",
        "fem_needle_suture",
        "needle_thread",
    ):
        if name in ("fem_dvrk", "fem_versius") or getattr(example.screen_pick, "window", None) is not None:
            # Use the ordinary example panel, with unrelated viewer sections
            # collapsed so its clickable pad map remains visible in the clip.
            def collapse_viewer_sections(ui):
                storage = ui.get_state_storage()
                for title in ("Model Information", "Visualization", "Rendering Options"):
                    storage.set_int(ui.get_id(title), 0)

            example.viewer.register_ui_callback(collapse_viewer_sections, position="panel")


def _screen_point(example, point):
    """Project a physical attachment target to the same window used for mouse input."""
    camera = example.viewer.camera
    view = camera.get_view_matrix().reshape((4, 4), order="F")
    projection = camera.get_projection_matrix().reshape((4, 4), order="F")
    clip = projection @ view @ np.append(point, 1.0)
    if clip[3] <= 0 or np.any(np.abs(clip[:3]) > clip[3]):
        raise RuntimeError("The recorded phantom grab is outside the camera view")
    return (clip[:2] / clip[3] + 1.0) * np.asarray(example.screen_pick.window.get_size()) / 2.0


def _phantom_grab_point(example, region_id):
    selection = example.selection
    faces = selection.exterior_surface_triangles
    fixed = np.zeros(len(selection.vertices), dtype=bool)
    fixed[example.fixed_vertex_indices] = True
    candidates = (selection.exterior_surface_region_ids == region_id) & ~np.any(fixed[faces], axis=1)
    corners = selection.vertices[faces]
    centers = corners.mean(axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    candidates &= normals @ np.asarray(example.view_direction) < 0.0
    if not np.any(candidates):
        raise RuntimeError(f"Region {region_id} has no visible, unconstrained exterior face to grab")
    candidates &= centers[:, 2] >= np.quantile(centers[candidates, 2], 0.65)
    available = np.flatnonzero(candidates)
    middle = np.median(centers[available], axis=0)
    return centers[available[np.argmin(np.linalg.norm(centers[available] - middle, axis=1))]]


def _phantom_begin_drag(example, region_id, frame):
    point = _phantom_grab_point(example, region_id)
    origin = np.asarray(example.view_position)
    direction = point - origin
    direction /= np.linalg.norm(direction)
    tool = example.screen_pick
    if tool.window is not None:
        pixel = _screen_point(example, point)
        tool.on_mouse_press(*pixel, 4, 0)
        if not tool.active:
            raise RuntimeError("The recorded right mouse press did not grab the active FEM surface")
        hit = tool.target.copy()
    else:
        # The profiler has no window. Replay the same ray and physical target
        # through the public callbacks used by ScreenPickTool.
        hit = example.begin_grab(origin, direction)
        pixel = None
        if hit is None:
            raise RuntimeError("The recorded ray did not grab the active FEM surface")
    example.recording_drag = {"point": np.asarray(hit), "pixel": pixel, "frame": frame}
    example.recording_events.append({"frame": frame, "action": "begin_right_drag", "region": region_id})


def _phantom_pointer(example, region_id):
    tool = example.screen_pick
    gui = getattr(example.viewer, "gui", None)
    io = getattr(getattr(gui, "ui", None), "io", None)
    if tool.window is not None and io is not None:
        pixel = _screen_point(example, _phantom_grab_point(example, region_id))
        size = np.asarray(tool.window.get_size())
        framebuffer = np.asarray(tool.window.get_framebuffer_size())
        pixel = pixel * framebuffer / size
        io.add_mouse_pos_event(float(pixel[0]), float(framebuffer[1] - pixel[1]))


def _phantom_move_drag(example, fraction):
    drag = example.recording_drag
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    point = drag["point"] + 0.003 * smooth * np.asarray(example.view_up)
    tool = example.screen_pick
    if tool.window is not None:
        pixel = _screen_point(example, point)
        delta = pixel - drag["pixel"]
        tool.on_mouse_drag(*pixel, *delta, 4, 0)
        drag["pixel"] = pixel
    else:
        example.move_grab(point)


def _phantom_finish_drag(example, frame):
    tool = example.screen_pick
    if tool.window is not None:
        tool.on_mouse_release(*example.recording_drag["pixel"], 4, 0)
    else:
        example.finish_grab()
    displacement = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
    maximum = float(displacement.max())
    if maximum < 1.0e-5:
        raise RuntimeError("The recorded spring grab did not produce measurable FEM deformation")
    example.test_final()
    example.recording_events.append({"frame": frame, "action": "release_to_hold", "maximum_displacement_m": maximum})
    example.recording_drag = None


def _phantom_frame(example, frame):
    if frame == 0:
        example.test_final()
        example.select_regions((4,))
        example.recording_events.append(
            {
                "frame": frame,
                "action": "select_regions",
                "regions": list(example.selected_region_ids),
                "tetrahedra": len(example.selection.indices),
            }
        )
    # Each selection uses the same map input as the user. Mouse movement,
    # press and release span separate frames so ImGui processes the full click.
    # The null profiler replays the identical selection changes without a UI.
    gui = getattr(example.viewer, "gui", None)
    io = getattr(getattr(gui, "ui", None), "io", None)
    clicks = ((170, 6, (6,)), (330, None, ()), (360, 4, (4,)))
    for clicked_frame, region_id, expected in clicks:
        if frame == clicked_frame - 2 and io is not None:
            pixel = (
                example.region_none_button_position
                if region_id is None
                else example.region_map.screen_position(region_id)
            )
            io.add_mouse_pos_event(*pixel)
        elif frame == clicked_frame:
            example.test_final()
            if io is not None:
                io.add_mouse_button_event(0, True)
            else:
                example.select_regions(expected)
        elif frame == clicked_frame + 1 and io is not None:
            io.add_mouse_button_event(0, False)
        elif frame == clicked_frame + 3:
            if tuple(example.selected_region_ids) != expected:
                raise RuntimeError(
                    f"Pad map click on region {region_id} selected {example.selected_region_ids}; expected {expected}"
                )
            example.recording_events.append(
                {
                    "frame": clicked_frame,
                    "action": (
                        "select_regions"
                        if io is None
                        else "click_none_button"
                        if region_id is None
                        else "click_region_map"
                    ),
                    "clicked_region": region_id,
                    "regions": list(example.selected_region_ids),
                    "tetrahedra": len(example.selection.indices),
                }
            )
    for start, region in ((20, 4), (190, 6), (390, 4)):
        if frame == start - 2:
            _phantom_pointer(example, region)
        elif frame == start:
            _phantom_begin_drag(example, region, frame)
        elif start < frame <= start + 70:
            _phantom_move_drag(example, (frame - start) / 70)
        elif frame == start + 71:
            _phantom_finish_drag(example, frame)
    if frame in (130, 300, 500):
        example.clear_holds()
        example.recording_events.append({"frame": frame, "action": "clear_holds"})


# The clip of fem_surgical_incision, after the picture the user gave of it: the body from straight above,
# one cut along the middle of the belly that does not go through, held open by two grips. Scene frame [m];
# the body lies with its arm span along x, its head at +y and its belly up. The lengths are read off that
# picture as fractions of the hand span (135 mm), with the belly top 19 % larger in it than the hands.
INCISION = {
    # Length of the cut along the midline, centred on the middle of the body.
    "wound_length": 0.0285,
    # Where a grip takes the skin: this far from the wound, at its mid-length.
    "grip_offset": 0.0059,
    # How far each grip is dragged away from the wound: the first one (-x) half as far as the second (+x).
    # The floor holds the body against the grips by its friction, and the two pulls open the wound to
    # about 8.5 mm (measured; on a floor without friction the body follows the first grip and the wound
    # opens by about as much as the second pull, 7.6 mm).
    "pull": (0.004, 0.008),
    # Speed of a drag [m/s]: below the speed limit of a grip's target (2.96 mm/s), so that the marker
    # and the grip stay together.
    "drag_speed": 0.0025,
    # The cut comes before frame 40, where the scripted user of --test looks whether somebody else is
    # at work and then stands down. Then the first press, and the frames between a release and a press.
    "cut_frame": 38,
    "first_press": 70,
    "pause": 20,
}


def _incision_surface(example):
    """Current positions [m] of the corners of the boundary triangles, three rows per triangle."""
    example.contact.update_surface(example.state_0.particle_q)
    return example.contact.points.numpy()[: 3 * example._surface_count]


def _incision_plan(example):
    """Frames of the two drags: (press, last frame of the drag, release) of each."""
    plan, press = [], INCISION["first_press"]
    for pull in INCISION["pull"]:
        frames = int(np.ceil(pull / (INCISION["drag_speed"] * example.frame_dt)))
        plan.append((press, press + frames, press + frames + 1))
        press += frames + 1 + INCISION["pause"]
    return plan


def _incision_limited(example):
    """Steps that the volume guard has shortened so far."""
    return int(example.volume_guard.limited_steps.numpy()[0])


def _incision_cut(example, frame):
    """One planar cut along the midline of the belly, from above the skin to a depth that leaves the back whole.

    The Line tool of the window cuts through the whole depth of the view. A blade that ends inside the
    body goes through ``Example.cut``, the call that the tool's stroke ends in: here to the depth of the
    scene's own wound (``_baymax_body.WOUND_DEPTH``, 13.5 mm of the 42 mm that the body is thick there).
    """
    from newton_solver_suturebot.examples.incision import _baymax_body as body  # noqa: PLC0415

    depth = body.SCALE * body.WOUND_DEPTH
    lower, upper = example.reference_positions.min(axis=0), example.reference_positions.max(axis=0)
    middle, span = 0.5 * (lower + upper), upper - lower
    half = 0.5 * INCISION["wound_length"]
    # The skin at the middle of the cut, where the body lies now (it settles on the floor in the first
    # frames): the highest point of the surface within a tenth of the cut's length of it.
    surface = _incision_surface(example)
    near = np.linalg.norm(surface[:, :2] - middle[:2], axis=1) < 0.1 * INCISION["wound_length"]
    skin = float(surface[near, 2].max())
    low, high = skin - depth, skin + 0.1 * float(span.min())
    x = middle[0]
    corners = np.array(
        [
            (x, middle[1] - half, low),
            (x, middle[1] + half, low),
            (x, middle[1] + half, high),
            (x, middle[1] - half, high),
        ]
    )
    mass_before = float(example.model.cut_physical_mass.numpy().sum())
    if not example.cut(corners[[[0, 1, 2], [0, 2, 3]]]):
        raise RuntimeError(f"Recorded incision failed: {example.cut_error}")
    components = len(np.unique(example.cutter.material_components.numpy()[: example.fragment_count]))
    mass_after = float(example.model.cut_physical_mass.numpy().sum())
    if components != 1 or not np.isclose(mass_after, mass_before, rtol=1e-5):
        raise RuntimeError("The incision must leave the body in one piece and conserve material mass")
    # The rim of the wound: corners of the new surface that lie in the blade's plane within a millimetre
    # of the skin, in bins of a millimetre along the cut. Both lips start in the plane; the opening of
    # the wound in a bin is the spread of its corners across the cut.
    surface = _incision_surface(example)
    along = surface[:, 1] - (middle[1] - half)
    in_plane = (np.abs(surface[:, 0] - x) < 1.0e-5) & (along >= 0.0) & (along <= 2.0 * half) & (surface[:, 2] > low)
    bins = np.floor(along / 0.001).astype(int)
    rim = []
    for index in range(int(np.ceil(2.0 * half / 0.001))):
        rows = np.flatnonzero(in_plane & (bins == index))
        if len(rows):
            rim.append(rows[surface[rows, 2] > surface[rows, 2].max() - 0.001])
    if not rim:
        raise RuntimeError("The incision left no rim at the skin")
    example.recording_wound = {"rim": rim, "limited_before": _incision_limited(example)}
    example.recording_events.append(
        {
            "frame": frame,
            "action": "planar_cut_of_limited_depth",
            "length_m": INCISION["wound_length"],
            "depth_below_skin_m": depth,
            "sunk_since_start_m": float(upper[2]) - skin,
            "material_components": components,
            "mass_before_kg": mass_before,
            "mass_after_kg": mass_after,
        }
    )


def _incision_wound(example):
    """Where the wound is now [m]: the middle of its rim, and its largest opening at the skin."""
    surface = _incision_surface(example)
    rim = example.recording_wound["rim"]
    opening = max(float(np.ptp(surface[rows, 0])) for rows in rim)
    return surface[np.concatenate(rim)].mean(axis=0), opening


def _incision_press(example, side, frame):
    """Right button down on the skin beside the wound: ``side`` -1 toward -x, +1 toward +x."""
    lower, upper = example.reference_positions.min(axis=0), example.reference_positions.max(axis=0)
    # The body may have moved with the grip that came before: the press is placed from the wound.
    wound, _ = _incision_wound(example)
    # A ray straight down, from half the body's thickness above its top as it was loaded.
    origin = np.array((wound[0] + side * INCISION["grip_offset"], wound[1], upper[2] + 0.5 * (upper[2] - lower[2])))
    down = np.array((0.0, 0.0, -1.0))
    tool = example.screen_pick
    if tool.window is not None:
        point = example._pick_cut_surface(origin, down)
        if point is None:
            raise RuntimeError("The recorded grip found no skin beside the wound")
        pixel = _screen_point(example, np.asarray(point, dtype=np.float64))
        tool.on_mouse_press(*pixel, 4, 0)
        if not tool.active:
            raise RuntimeError("The recorded right mouse press did not grip the body")
        hit = tool.target.copy()
    else:
        # Without a window (the profiler) the same grip goes through the callbacks of the screen tool.
        hit, pixel = example._begin_pick(origin, down), None
        if hit is None:
            raise RuntimeError("The recorded grip found no skin beside the wound")
    example.recording_drag = {"point": np.asarray(hit, dtype=np.float64), "pixel": pixel, "side": side}
    example.recording_events.append(
        {
            "frame": frame,
            "action": "begin_right_drag",
            "side": "+x" if side > 0 else "-x",
            "point_m": [float(value) for value in hit],
        }
    )


def _incision_drag(example, distance):
    """Move the pressed grip to ``distance`` [m] from where it took the skin, away from the wound."""
    drag = example.recording_drag
    target = drag["point"] + np.array((drag["side"] * distance, 0.0, 0.0))
    tool = example.screen_pick
    if tool.window is not None:
        pixel = _screen_point(example, target)
        tool.on_mouse_drag(*pixel, *(pixel - drag["pixel"]), 4, 0)
        drag["pixel"] = pixel
    else:
        example._move_pick(target)


def _incision_release(example, frame, pull):
    drag, tool = example.recording_drag, example.screen_pick
    if tool.window is not None:
        tool.on_mouse_release(*drag["pixel"], 4, 0)
    else:
        example._finish_pick()
    example.recording_events.append(
        {
            "frame": frame,
            "action": "release_to_hold",
            "side": "+x" if drag["side"] > 0 else "-x",
            "pull_m": pull,
            "wound_opening_m": _incision_wound(example)[1],
        }
    )
    example.recording_drag = None


def _incision_frame(example, frame):
    """Cut the belly along the midline, not through; then two right-drags, one on each side, pull the wound open."""
    if frame == INCISION["cut_frame"]:
        _incision_cut(example, frame)
    for side, pull, (press, last, release) in zip((-1.0, 1.0), INCISION["pull"], _incision_plan(example), strict=True):
        if frame == press - 2 and example.screen_pick.window is not None:
            # The pointer goes where the button will come down, as before a click.
            pixel = _screen_point(example, _incision_wound(example)[0] + (side * INCISION["grip_offset"], 0.0, 0.0))
            example.screen_pick._pointer(*pixel)
        elif frame == press:
            _incision_press(example, side, frame)
        elif press < frame <= last:
            _incision_drag(example, min(pull, (frame - press) * INCISION["drag_speed"] * example.frame_dt))
        elif frame == release:
            _incision_release(example, frame, pull)
    if frame % 60 == 0:
        example.test_final()


# The clip of fem_needle_insertion: the first and the last captured frame in which the pad is drawn
# opaque. They are the frames in which the clip shows the mesh edges, the "edges" interval of its
# recipe in docs/videos.json; tests/test_record_examples.py keeps the two equal.
INSERTION_OPAQUE_FRAMES = (120, 251)


def before_frame(name, example, frame):
    if name == "fem_surgical_incision":
        _incision_frame(example, frame)
        return
    if name == "fem_needle_insertion":
        if frame == 5:
            example.select_point_xy("entry", 0.030, -0.040)
        elif frame == 25:
            example.select_point_xy("exit", 0.030, -0.048)
        elif frame == 45:
            example.preview()
            example.frame_insertion()
        elif frame in (75, 180):
            example.start()
        elif frame == 155:
            example.running = False
            example.outcome = "Paused"
        # The scripted user switches the panel's "See through pad" off for the frames in which the
        # recipe shows the mesh edges and on again after them. Through a translucent pad the edge
        # lines of its underside and of the slit walls lie over those of the top surface; on the
        # opaque pad only the lines of the visible surface are drawn.
        if frame == INSERTION_OPAQUE_FRAMES[0]:
            example.args.transparent_pad = False
        elif frame == INSERTION_OPAQUE_FRAMES[1] + 1:
            example.args.transparent_pad = True
        return
    if name in ("fem_dvrk", "fem_versius"):
        # The replay needs no scripted action, and its view is the camera of the recipe (docs/videos.json).
        example.recording_seen_frames.add(example.frame_index)
        if frame % 60 == 0:
            example.test_final()
        return
    if name == "fem_phantom_regions":
        _phantom_frame(example, frame)
        return


def frame_label(name, example):
    if name == "fem_surgical_incision":
        return f"Baymax body ({len(example.reference_elements):,} source tets) | Cuts: {example.cut_revision} | Grips: {example.holds.count} | Simulation-time playback"
    if name == "fem_needle_insertion":
        return f"{example.outcome} | Simulation-time playback"
    if name in ("fem_dvrk", "fem_versius"):
        return f"{example.recording.manifest['sample_id']} | frame {example.frame_index} | {example.recording.time_s[example.frame_index]:.2f} s | {example.recording.camera_count} camera overlays"
    if name == "fem_phantom_regions":
        regions = ", ".join(str(value) for value in example.selected_region_ids) or "none"
        action = "Right-drag: physical spring grab" if example.recording_drag else "Click pad map | fixed inactive pad"
        return f"Active: {regions} | {len(example.selection.indices):,} tets | {action}"
    if name == "fem_needle_suture":
        return f"{example.outcome.split(';')[0]} | Simulation-time playback"
    if name == "needle_thread":
        return "Simulation-time playback | illustrative thread forces"
    if name == "fem_tool_tissue":
        # The phase and the motion of the scripted path that the drawn frame ended, as the example's panel names them.
        path = example.path
        segment = path.segment_of(min(max(example.frame - 1, 0), path.frames - 1))
        phases = list(dict.fromkeys(item.phase for item in path.segments))
        return (
            f"Phase {phases.index(segment.phase) + 1}/{len(phases)}: {segment.phase.replace('_', ' ')} "
            f"({segment.label}) | --contact {example.args.contact} | Simulation-time playback"
        )
    return ""


def recording_metadata(name, example):
    if name == "fem_surgical_incision":
        example.test_final()
        wound = getattr(example, "recording_wound", None)
        holds = example.holds
        grips = [holds.grip_position(example.state_0.particle_q, grip) for grip in range(holds.count)]
        limited = None if wound is None else _incision_limited(example) - wound["limited_before"]
        return {
            "original_vertices": len(example.reference_positions),
            "original_tetrahedra": len(example.reference_elements),
            "all_material_active": True,
            "fixed_nodes": 0,
            "cuts": example.cut_revision,
            "grips": holds.count,
            # At the end of the clip: the wound's largest opening at the skin and the distance of the two grips.
            "wound_opening_m": None if wound is None else _incision_wound(example)[1],
            "grip_distance_m": float(np.linalg.norm(grips[1] - grips[0])) if len(grips) == 2 else None,
            # Steps that the volume guard shortened after the cut, and the cut that was refused, if any.
            "volume_guard_limited_steps": limited,
            "cut_error": example.cut_error,
            "events": example.recording_events,
            "interaction": "a planar cut of limited depth through Example.cut; two right-drags through the mouse "
            "handlers of the screen pick tool (through its callbacks where there is no window)",
        }
    if name == "fem_needle_insertion":
        return example.report() | {
            "entry_xy_m": [0.030, -0.040],
            "exit_xy_m": [0.030, -0.048],
            "interaction": "public material-selection, preview, start and pause controls",
            "needle_motion": "finite PD wrench in MuJoCo, with equal-and-opposite FEM interaction",
        }
    if name in ("fem_dvrk", "fem_versius"):
        example.test_final()
        example.recording_seen_frames.add(example.frame_index)
        if example.overlay is None:
            raise RuntimeError("The recording must include its configured model/video overlays")
        if not all(count > 0 for count in example.overlay.coverage.values()):
            raise RuntimeError("The recording must render geometry in every camera view")
        return {
            "sample_id": example.recording.manifest["sample_id"],
            "source_episode": example.recording.manifest["source_episode"],
            "source_frame_count": example.recording.frame_count,
            "observed_source_frames": len(example.recording_seen_frames),
            "supported_source_frames": [example.recording.first_valid_frame, example.recording.last_valid_frame],
            "alignment": example.recording.manifest["alignment"],
            "final_overlay_coverage_pixels": example.overlay.coverage,
            "motion": example.recording.manifest.get("motion", {}).get(
                "description", "measured Cartesian/joint kinematic replay"
            ),
            "camera_order": list(example.recording.camera_order),
            "placeholder_views": [
                key for key, camera in example.recording.cameras.items() if camera.get("placeholder")
            ],
            "region_id": example.recording.manifest.get("region_id"),
            "point_labels": example.recording.manifest.get("point_labels"),
        }
    if name == "fem_phantom_regions":
        return {"drag_distance_m": 0.003, "events": example.recording_events}
    return None
