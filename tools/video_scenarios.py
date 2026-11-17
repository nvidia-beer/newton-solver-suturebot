"""Script public example interactions for representative FEM recordings."""

import json

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
        example.recording_view_changes = []
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


def _incision_frame(example, frame):
    """Slice off the low-x end of the body (a hand of the Baymax body), lift it through a physical grip, then release."""
    lower, upper = example.reference_positions.min(axis=0), example.reference_positions.max(axis=0)
    span = upper - lower
    # The blade reaches beyond the body by a tenth of its smallest size; a ray starts half its thickness above it.
    margin, above = 0.1 * float(span.min()), upper[2] + 0.5 * span[2]
    x = lower[0] + 0.14 * span[0]
    if frame == 30:
        corners = np.array(
            [
                (x, lower[1] - margin, lower[2] - margin),
                (x, upper[1] + margin, lower[2] - margin),
                (x, upper[1] + margin, upper[2] + margin),
                (x, lower[1] - margin, upper[2] + margin),
            ]
        )
        mass_before = float(example.model.cut_physical_mass.numpy().sum())
        if not example.cut(corners[[[0, 1, 2], [0, 2, 3]]]):
            raise RuntimeError(f"Recorded incision failed: {example.cut_error}")
        components = len(np.unique(example.cutter.material_components.numpy()[: example.fragment_count]))
        mass_after = float(example.model.cut_physical_mass.numpy().sum())
        if components < 2 or not np.isclose(mass_after, mass_before, rtol=1e-5):
            raise RuntimeError("The incision must separate pieces and conserve material mass")
        example.recording_events.append(
            {
                "frame": frame,
                "action": "finite_line_incision",
                "material_components": components,
                "mass_before_kg": mass_before,
                "mass_after_kg": mass_after,
            }
        )
    if frame == 60:
        # Straight above the middle of the piece that was cut off, wherever it lies by now.
        example.contact.update_surface(example.state_0.particle_q)
        surface = example.contact.points.numpy()[: 3 * example._surface_count]
        piece = surface[surface[:, 0] < x]
        if not len(piece):
            raise RuntimeError("The cut piece must remain available for a physical surface grip")
        origin = np.array((*piece[:, :2].mean(axis=0), above))
        point = example._begin_pick(origin, np.array((0.0, 0.0, -1.0)))
        if point is None:
            raise RuntimeError("The cut piece must remain available for a physical surface grip")
        example.recording_drag = np.asarray(point).copy()
        example.recording_events.append({"frame": frame, "action": "begin_surface_grip"})
    if 60 < frame <= 210:
        fraction = (frame - 60) / 150
        fraction = fraction * fraction * (3 - 2 * fraction)
        example._move_pick(example.recording_drag + (0, 0, 0.03 * fraction))
    if frame == 211:
        example._finish_pick()
        example.recording_events.append({"frame": frame, "action": "release_to_hold", "target_lift_m": 0.03})
    if frame == 270:
        example.clear_holds()
        example.recording_events.append({"frame": frame, "action": "release_cut_strip"})
    if frame % 60 == 0:
        example.test_final()


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
        return
    if name in ("fem_dvrk", "fem_versius"):
        example.recording_seen_frames.add(example.frame_index)
        if frame == 120:
            example.frame_phantom()
            example.recording_view_changes.append({"capture_frame": frame, "view": "phantom"})
        elif frame == 180 and "point_labels" in example.recording.manifest:
            labels = example.recording.manifest["point_labels"]
            report = json.loads(example.recording.asset(labels["report"]).read_text())
            center = np.mean([point["position_world_m"] for point in report["points"]], axis=0)
            # Keep the physical-size markers legible after video compression.
            example.viewer.set_camera(pos=center + np.array([-0.06, 0.06, 0.08]))
            example.viewer.camera.look_at(center)
            example.recording_view_changes.append({"capture_frame": frame, "view": "surface point labels"})
        elif frame == 850:
            example.frame_phantom()
            example.recording_view_changes.append({"capture_frame": frame, "view": "phantom"})
        elif frame == 1000:
            example.reset_camera()
            example.recording_view_changes.append({"capture_frame": frame, "view": "complete rig"})
        if frame % 60 == 0:
            example.test_final()
        return
    if name == "fem_phantom_regions":
        _phantom_frame(example, frame)
        return


def frame_label(name, example):
    if name == "fem_surgical_incision":
        return f"Baymax body ({len(example.reference_elements):,} source tets) | Cuts: {example.cut_revision} | Simulation-time playback"
    if name == "fem_needle_insertion":
        return f"{example.outcome} | Simulation-time playback | coupled MuJoCo/FEM, illustrative interaction"
    if name in ("fem_dvrk", "fem_versius"):
        return f"{example.recording.manifest['sample_id']} | frame {example.frame_index} | {example.recording.time_s[example.frame_index]:.2f} s | {example.recording.camera_count} camera overlays"
    if name == "fem_phantom_regions":
        regions = ", ".join(str(value) for value in example.selected_region_ids) or "none"
        action = "Right-drag: physical spring grab" if example.recording_drag else "Click pad map | fixed inactive pad"
        return f"Active: {regions} | {len(example.selection.indices):,} tets | {action}"
    if name == "fem_needle_suture":
        return f"{example.outcome} | Simulation-time playback | illustrative thread and tissue forces"
    if name == "needle_thread":
        return "Simulation-time playback | illustrative thread forces"
    return ""


def recording_metadata(name, example):
    if name == "fem_surgical_incision":
        example.test_final()
        return {
            "original_vertices": len(example.reference_positions),
            "original_tetrahedra": len(example.reference_elements),
            "all_material_active": True,
            "fixed_nodes": 0,
            "cuts": example.cut_revision,
            "events": example.recording_events,
            "interaction": "finite incision and physical surface grip through example callbacks",
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
            "view_changes": example.recording_view_changes,
        }
    if name == "fem_phantom_regions":
        return {"drag_distance_m": 0.003, "events": example.recording_events}
    return None
