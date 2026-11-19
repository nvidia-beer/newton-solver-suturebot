# SPDX-License-Identifier: Apache-2.0
"""Where the thread goes through the tissue, as the panel of fem_needle_grasp says it."""

import unittest
from types import SimpleNamespace

from newton_solver_suturebot.examples.grasp import example_fem_needle_grasp as grasp_example


def report(made=(), threaded=0, inside=0.0, refused=0, reserve=(36, 4)):
    """A scene's thread report, as far as the panel reads it: ``made`` is the number of eyelets of every passage."""
    return {
        "eyelets": sum(made),
        "eyelet_passages": list(made),
        "eyelets_threaded": threaded,
        "eyelets_refused": refused,
        "eyelet_reserve": list(reserve),
        "thread_inside_tissue_m": inside,
    }


class _Panel:
    """The side panel of a viewer: keeps the lines it is given."""

    def __init__(self):
        self.lines = []

    def text_wrapped(self, line):
        self.lines.append(line)


class ThreadLineTests(unittest.TestCase):
    def test_the_line_names_the_threaded_eyelets_of_every_passage_and_the_thread_inside(self):
        line = grasp_example.thread_line
        # A scene whose thread stays on the surface has no eyelet slot, and no line; neither has a missing report.
        self.assertIsNone(line(None))
        self.assertIsNone(line({}))
        self.assertIsNone(line(report(reserve=(0, 1))))
        self.assertEqual(line(report()), "Thread: no eyelets yet, the needle has made no track")
        self.assertEqual(
            line(report((4, 4))),
            "Thread: 0 of 8 eyelets threaded (passage 1: 0 of 4, passage 2: 0 of 4), 0.0 mm inside the tissue",
        )
        self.assertEqual(
            line(report((4, 4), threaded=5, inside=0.00612)),
            "Thread: 5 of 8 eyelets threaded (passage 1: 4 of 4, passage 2: 1 of 4), 6.1 mm inside the tissue",
        )
        self.assertEqual(
            line(report((7,), threaded=7, inside=0.0105, refused=2)),
            "Thread: 7 of 7 eyelets threaded (passage 1: 7 of 7), 10.5 mm inside the tissue; "
            "2 eyelets found no room in the reserve",
        )

    def test_the_threaded_eyelets_are_counted_to_the_passages_in_the_order_the_eyelets_were_made(self):
        order = grasp_example.threaded_in_order
        self.assertEqual(order([], 0), [])
        self.assertEqual([order([4, 4], count) for count in (0, 3, 4, 5, 8)], [[0, 0], [3, 0], [4, 0], [4, 1], [4, 4]])
        self.assertEqual(order([3, 5, 2], 4), [3, 1, 0])
        # Never more than a passage has, and never more than were made.
        self.assertEqual(order([2, 2], 9), [2, 2])

    def test_the_panel_has_one_line_per_scene(self):
        example = grasp_example.Example.__new__(grasp_example.Example)
        example.args = SimpleNamespace(num_envs=1)
        example.observations = [{"env_id": 0, "thread_diagnostics": report((4, 4), threaded=2, inside=0.002)}]
        panel = _Panel()
        example.gui(panel)
        self.assertEqual(
            panel.lines,
            ["Thread: 2 of 8 eyelets threaded (passage 1: 2 of 4, passage 2: 0 of 4), 2.0 mm inside the tissue"],
        )
        # Several scenes: every line names its scene; a scene without eyelet slots has none.
        example.args = SimpleNamespace(num_envs=3)
        example.observations = [
            {"env_id": 0, "thread_diagnostics": report()},
            {"env_id": 1, "thread_diagnostics": report(reserve=(0, 1))},
            {"env_id": 2, "thread_diagnostics": report((3,), threaded=3, inside=0.004)},
        ]
        panel = _Panel()
        example.gui(panel)
        self.assertEqual(
            panel.lines,
            [
                "Scene 0: Thread: no eyelets yet, the needle has made no track",
                "Scene 2: Thread: 3 of 3 eyelets threaded (passage 1: 3 of 3), 4.0 mm inside the tissue",
            ],
        )


if __name__ == "__main__":
    unittest.main()
