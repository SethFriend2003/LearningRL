"""Where the RPO's "hand" is: it has none, so the tracked point is the wrist end of the forearm.

Pure data -- no Isaac Lab import -- so the env cfg can read it before Kit starts (F9).

The RPO's arm ends at ``<side>_elbow_yaw_link`` (the forearm). Its link frame sits at the
elbow-yaw joint, the forearm's ROOT, and the forearm shell extends 0.138 m along that frame's +x
(``rpo.urdf`` joint origins all have ``rpy="0 0 0"``, so the URDF link frame, the Isaac Lab body
frame and the MJCF body frame coincide). Tracking the body origin would leave the forearm's
direction free; tracking a point near its tip makes the policy place the wrist, which is what a
two-forearm squeeze needs.
"""

WRIST_OFFSET: tuple[float, float, float] = (0.135, 0.0, 0.0)
"""Wrist point in the ``<side>_elbow_yaw_link`` frame (m): 3 mm short of the forearm shell's tip."""
