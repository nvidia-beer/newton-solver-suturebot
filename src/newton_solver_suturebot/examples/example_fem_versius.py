# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preview CMR Surgical Versius instruments over the full pad with placeholder stereo video.

Load prepared USD geometry, generated articulation and a configurable camera
list. The two bundled video views are unrelated dVRK placeholders. Asset
conversion, scene authoring and image preparation belong to the separate
MSc/suturebot_sim preparation folder; the viewer only reads bundled outputs.

Run: python -m newton_solver_suturebot.examples fem_versius
"""

from pathlib import Path

import newton.examples

from newton_solver_suturebot.examples._recording_example import RecordingExample
from newton_solver_suturebot.examples._suturebot_cli import configure_sample

RECORDINGS = Path(__file__).parent / "assets/versius/recordings"
DEFAULT_RECORDING = RECORDINGS / "versius_stereo_preview/recording.json"


def bundled_recordings(*, include_aliases=True):
    """Discover portable Versius scenes; this initial catalog has no aliases."""
    return {path.parent.name: path for path in sorted(RECORDINGS.glob("*/recording.json"))}


class Example(RecordingExample):
    """Show a generated Versius assembly preview with labeled dVRK placeholder images."""

    default_recording = DEFAULT_RECORDING
    recording_catalog = staticmethod(bundled_recordings)
    title = "CMR Surgical Versius preview"
    motion_note = (
        "Generated articulation preview over a static pad. "
        "dVRK placeholder videos are unrelated to Versius motion; stereo placement is illustrative."
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    configure_sample(
        parser,
        catalog=bundled_recordings,
        default_recording=DEFAULT_RECORDING,
        title="CMR Surgical Versius prepared scenes",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
