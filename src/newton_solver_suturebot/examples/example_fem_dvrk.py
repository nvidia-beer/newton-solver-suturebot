# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay da Vinci Research Kit (dVRK) recordings with four source-video camera overlays.

Loads a prepared recording.json, USD geometry and timestamped poses. All camera
fitting, frame extraction and asset preparation live in the MSc workbench.
Dataset insertion/exit labels appear as spheres on the phantom surface. Source-defined
approach clips retain their original frame numbers. The prepared scene contains
the tools, cameras, fixture and phantom; motion is replayed without dynamics.

Run: python -m newton_solver_suturebot.examples fem_dvrk
"""

import newton.examples

from newton_solver_suturebot.examples._recording_example import RecordingExample
from newton_solver_suturebot.examples._suturebot_recording import DEFAULT_RECORDING, bundled_recordings


class Example(RecordingExample):
    """Replay measured dVRK tool poses with the prepared four-camera recordings."""

    default_recording = DEFAULT_RECORDING
    recording_catalog = staticmethod(bundled_recordings)
    title = "da Vinci Research Kit (dVRK) replay"
    motion_note = "Measured motion replay. Camera and phantom alignment are fitted estimates."


if __name__ == "__main__":
    from newton_solver_suturebot.examples._suturebot_cli import configure_sample

    parser = Example.create_parser()
    configure_sample(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
