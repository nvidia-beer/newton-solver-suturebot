# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recognize explicit material-reserve limits without masking invalid cuts."""

import ast


def capacity_resource(error, fragment_capacity, *, fixed=False):
    """Return the exhausted reserve, or None for any other diagnostic.

    ``fixed``: the storage of the cutter is a fixed reserve (``--cut_reserve``). An edit that
    needed more than it is named "cut_reserve"; one that ran out of fragments is recognised by
    its geometry error alone (the merge that follows clears the fragment count).
    """
    marker = "linear tetrahedron edit rejected: "
    message = str(error)
    if marker not in message:
        return None
    try:
        diagnostics = ast.literal_eval(message.split(marker, 1)[1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(diagnostics, dict):
        return None
    if diagnostics.get("refused_growth"):
        return "cut_reserve"
    if any(diagnostics.get(name) != 0 for name in ("integration_error", "surface_error")):
        return None
    geometry, connectivity = diagnostics.get("geometry_error"), diagnostics.get("connectivity_error")
    if geometry == 0 and connectivity == 2:
        return "node_copies"
    count = diagnostics.get("fragment_count")
    if geometry == 4 and connectivity == 3 and isinstance(count, int) and count > fragment_capacity:
        return "fragment_capacity"
    if fixed and geometry == 4:
        return "fragment_capacity"
    return None


def rejection_message(error, fragment_capacity, *, fixed=False):
    """Keep the precise diagnostic after an actionable reserve-limit message.

    ``fixed``: the storage of the cutter is a fixed reserve (``--cut_reserve``).
    """
    if "boundary surface reserve exhausted" in str(error):
        return f"Cut refused: {error}; restart with a larger --cut_reserve, or without it to let the storage grow."
    resource = capacity_resource(error, fragment_capacity, fixed=fixed)
    if resource is None:
        return f"Cut rejected: {error}"
    if resource == "cut_reserve":
        return (
            "Cut refused: the fixed cut reserve is exhausted; restart with a larger --cut_reserve or "
            f"--cut_reserve_polygons, or without them to let the storage grow. {error}"
        )
    label = "Material node" if resource == "node_copies" else "Material fragment"
    option = "cut_reserve" if fixed and resource == "fragment_capacity" else resource
    return f"Cut rejected: {label} reserve exhausted; restart with a larger --{option}. {error}"
