# Baymax tetrahedral body

`tet_baymax.usda` is the body that `fem_surgical_incision` cuts: one `TetMesh`
prim with 8,965 points, 38,730 tetrahedra and 9,882 surface triangles. The
solid is filled (no cavity), the surface is closed and every tetrahedron has
positive volume as stored.

The file is a byte-for-byte copy of
`src/newton_solver_fem/examples/assets/tet_baymax.usda` of the local
`newton-solver-fem` repository at revision
`6968f2742bb91f429601bd2b1853ede57d002220` (1,484,097 bytes, SHA-256
`c989ec2eb70d88ff2739c209720354e528440d54a4e353f25c0be3cd469876c4`). Nothing
in it is edited here. Scale and placement are applied when the example loads
it: [`_baymax_body.py`](../../incision/_baymax_body.py) holds both, with their
derivations, and the node sets, the wound line and the stitch frame that the
later steps of the example use.

## Licence and origin

- The file itself carries no licence line and no SPDX line.
- The FEM repository attributes its bundled example meshes, this one among
  them, to The Newton Developers under Apache-2.0 (its `THIRD_PARTY_NOTICES`
  and the README of its assets folder).
- No upstream location is recorded for the mesh, in that repository or in
  the file.
- The shape is a film character.

`provenance.json` records the source path, revision and hash.

## Frame and size

The asset's own unit is its span from hand to hand: 1.000 along x. The
figure's height runs along y (head at +y, feet at -y, 0.632) and its depth
along z (belly at +z, back at -z, 0.327).

The header says `upAxis = "Z"`, while the figure stands along y. The package's
loader follows the header and applies no rotation, so the body is read as it
is stored: lying on its back with the belly up. The example keeps that. (The
FEM repository's own example turns the figure upright by hand.)

In the scene the span is 135 mm: the body measures 135.0 x 85.4 x 44.1 mm and
weighs 92 g at 1,000 kg/m3.
