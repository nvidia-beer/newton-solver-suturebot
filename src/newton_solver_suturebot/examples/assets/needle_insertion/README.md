# Needle insertion asset

`needle.usda` is the authored `/SutureBot/Needle` subtree extracted from
[`../dvrk/suturebot_rig.usda`](../dvrk/suturebot_rig.usda). It preserves
the original CAD display mesh, all 32 collision capsules, body-local coordinates,
mass and diagonal inertia. Only the source scene placement is removed. The two
bound original materials are copied under `/Needle/Looks`; their bindings and
shader connections resolve within this standalone asset.

The collision centerline is a circular arc of radius 9.449 mm and span 2.3748 rad,
with capsule radius 0.28 mm. Its first endpoint, `ContactArc_00`'s beginning, is
the insertion model's tip. The rounded collision tip approximates the visible
CAD tip; it does not resolve its bevel. Runtime shaft quadrature subdivides the
original straight capsule centerline segments without smoothing or changing
the authored geometry.

The inherited 0.3 g mass and diagonal inertia `(1e-8, 1e-8, 2e-8)` kg m² are
illustrative simulation parameters, as are the collision proxies. They are not
measurements or a calibrated needle specification. `provenance.json` records
the source hash and extraction changes.

The original needle mesh's upstream license is unspecified. See the retained
[SutureBot third-party notices](../dvrk/THIRD_PARTY.md) and repository
[`THIRD_PARTY_NOTICES`](../../../../../THIRD_PARTY_NOTICES). The package license
does not establish a license for this third-party geometry.
