# SPDX-License-Identifier: Apache-2.0
"""CPU-only spawned-worker probe used by test_parallel_thread_limits."""

import json
import multiprocessing as mp
import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


def snapshot():
    return {
        "environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
        },
        "pools": {info["filepath"]: info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"},
    }


def probe_worker(channel, failure):
    # No scene, native solver or Warp device is initialized in this test.
    from newton_solver_suturebot.examples.grasp import parallel

    result = {"before": snapshot(), "events": {}, "messages": []}

    def record(name):
        result["events"][name] = snapshot()

    def close_scene():
        record("scene_close")
        if failure == "cleanup":
            raise RuntimeError("injected cleanup failure")

    def make_scene(config, env_id):
        record("make_scene")
        if failure == "setup":
            raise RuntimeError("injected setup failure")
        # Scene imports may load another BLAS library. Changing this existing
        # pool emulates a library whose initial setting needs the second scan.
        threadpool_limits(limits=7, user_api="blas")
        return SimpleNamespace(
            close=close_scene,
            viewer=SimpleNamespace(close=lambda: record("viewer_close")),
        )

    def observe(scene, env_id):
        record("observe")
        if failure == "observe":
            raise RuntimeError("injected observation failure")
        return {"pid": os.getpid()}

    class Connection:
        def send(self, value):
            result["messages"].append(value)

        def recv(self):
            return "close", None

        def close(self):
            record("connection_close")
            if failure == "connection":
                raise RuntimeError("injected connection cleanup failure")

    try:
        with mock.patch.object(parallel, "_make_scene", make_scene), mock.patch.object(parallel, "_observe", observe):
            parallel._worker(Connection(), parallel.SutureSceneConfig(device="cpu"), 0, None)
    except RuntimeError as error:
        result["raised"] = str(error)
    result["after"] = snapshot()
    channel.send(result)
    channel.close()


def main():
    # Deliberately initialize NumPy before spawning, as the actual viewer does.
    np.linalg.solve(np.eye(2), np.ones(2))
    before = snapshot()
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=probe_worker, args=(child, sys.argv[1]))
    process.start()
    child.close()
    if not parent.poll(60):
        process.terminate()
        process.join()
        raise RuntimeError("Worker probe timed out")
    result = parent.recv()
    parent.close()
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("Worker probe did not exit")
    if process.exitcode != 0:
        raise RuntimeError(f"Worker probe exited {process.exitcode}")
    print(json.dumps({"parent_before": before, "parent_after": snapshot(), "worker": result}))


if __name__ == "__main__":
    main()
