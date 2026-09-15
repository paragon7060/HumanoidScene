# S200062-derived independent two-finger claw

Source: this repository's `kuavo_s200062/urdf/biped_s200062.urdf` and meshes,
derived from LejuRobotics/kuavo-ros-opensource revision
`5d60846b092b425a7a3c06479bdfdbc2b100e890`.
Source subtree: src/kuavo_assets/models/biped_s200062.

Each side extracts only the twofinger_base descendant branch: housing, two
four-bar jaws and the physical D405 camera/bracket/reference links (14 links,
13 tree joints). Left/right retain their original names and CAD frames;
their origins are the respective twofinger_base, not the robot root or EEF.
Meshes and visual colors are copied without recoloring or shape changes.
config.json records source-URDF and mesh SHA-256 hashes and donor mount poses.

Masses, COMs and diagonal inertias are copied from the project's simulation
estimates, NOT manufacturer calibration. Mass is 0.740 kg per extracted hand
including its D405 branch, excluding the three robot-side EEF helper frames.
The closed-loop anchors and bar_4 correction reuse robots/twofinger_linkage.py.
Only bar_1 is driven; bar_3/bar_4 are passive. The URDF alone is a tree and
does not encode the loop-closing hinges; finalized USDs include two physical
external hinges per hand. Housing and separate finger meshes have convex
contact geometry, matching the project's simplified runtime contact model.

This is a reusable simulation asset extracted from S200062, not an independent
manufacturer-issued gripper package or proof that S63 uses identical hardware.
Original upstream asset rights remain applicable; see THIRD_PARTY_ASSETS.md.
