# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suture thread: a chain of cable elements and its attachment to a rigid needle."""

from .attachment import ThreadAttachment
from .capsules import ThreadCapsuleContact
from .model import ThreadModel
from .solver import SolverThread
from .tissue import ThreadTissueCoupling
from .track import NeedleTrackEyelets

__all__ = [
    "NeedleTrackEyelets",
    "SolverThread",
    "ThreadAttachment",
    "ThreadCapsuleContact",
    "ThreadModel",
    "ThreadTissueCoupling",
]
