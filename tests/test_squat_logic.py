import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from squat_logic import (  # noqa: E402
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
    SquatCounter, State,
)

FPS = 30
DT = 1 / FPS
SHIN = THIGH = 0.45


def leg(knee_deg: float, x: float, ankle_lift: float = 0.0, ankle_dx: float = 0.0):
    """Yon tomondan ko'rinish: boldir vertikal, son tizza burchagiga qarab buriladi (y pastga)."""
    ankle = np.array([x + ankle_dx, 0.9 - ankle_lift])
    knee = ankle + np.array([0.0, -SHIN])
    th = math.radians(knee_deg)
    hip = knee + THIGH * np.array([math.sin(th), math.cos(th)])
    return ankle, knee, hip


def pose(knee_l=178.0, knee_r=None, lift_l=0.0, lift_r=0.0, dx=0.0, vis=0.99):
    knee_r = knee_l if knee_r is None else knee_r
    world = np.zeros((33, 3))
    img = np.zeros((33, 3))
    img[:, 2] = vis
    parts = {}
    for side, k, lift, x in (("l", knee_l, lift_l, -0.1), ("r", knee_r, lift_r, 0.1)):
        parts[side] = leg(k, x, lift, dx)
    # Tos — ikkala son uchining o'rtasi (tana bitta), shu sababli ikkala son ham shu nuqtaga ulanadi.
    hip_y = (parts["l"][2][1] + parts["r"][2][1]) / 2
    for side, (ai, ki, hi, si) in (
        ("l", (L_ANKLE, L_KNEE, L_HIP, L_SHOULDER)),
        ("r", (R_ANKLE, R_KNEE, R_HIP, R_SHOULDER)),
    ):
        ankle, knee, hip = parts[side]
        for idx, p in ((ai, ankle), (ki, knee), (hi, hip)):
            world[idx, :2] = p
            img[idx, :2] = p * 0.8 + 0.05
        img[si, :2] = np.array([hip[0], hip_y - 0.5]) * 0.8 + 0.05
        world[si, :2] = [hip[0], hip_y - 0.5]
    # Rasmdagi tos balandligi — ikkala tomonning o'rtachasi
    for idx in (L_HIP, R_HIP):
        img[idx, 1] = hip_y * 0.8 + 0.05
    return img, world


class Sim:
    def __init__(self):
        self.c = SquatCounter()
        self.t = 0.0
        self.events = []

    def feed(self, frames: int, **kw):
        img, world = pose(**kw) if kw.get("vis", 1) is not None else (None, None)
        last = None
        for _ in range(frames):
            last = self.c.update(self.t, img, world)
            if last.event:
                self.events.append((last.event, last.message))
            self.t += DT
        return last

    def ramp(self, a: float, b: float, frames: int, **kw):
        for i in range(frames + 1):
            k = a + (b - a) * i / frames
            self.feed(1, knee_l=k, knee_r=kw.get("knee_r_fn", lambda v: v)(k),
                      lift_l=kw.get("lift_l", 0.0), dx=kw.get("dx", 0.0))

    def calibrate(self):
        self.feed(40)
        assert self.c.state == State.READY, self.c.message

    def squat(self, bottom=80.0, down=15, hold=5, up=15, **kw):
        self.ramp(178, bottom, down, **kw)
        self.feed(hold, knee_l=bottom, knee_r=kw.get("knee_r_fn", lambda v: v)(bottom),
                  lift_l=kw.get("lift_l", 0.0), dx=kw.get("dx", 0.0))
        self.ramp(bottom, 178, up, **kw)
        self.feed(5)


def test_counts_real_squats():
    s = Sim()
    s.calibrate()
    for _ in range(3):
        s.squat()
    assert s.c.count == 3
    assert s.c.rejected == 0


def test_no_count_without_calibration():
    s = Sim()
    s.feed(3)  # juda qisqa — hali kalibrovka yo'q
    s.squat()
    assert s.c.count == 0


def test_shallow_squat_rejected():
    s = Sim()
    s.calibrate()
    s.squat(bottom=130)
    assert s.c.count == 0
    assert s.c.rejected == 1


def test_tiny_bob_ignored():
    s = Sim()
    s.calibrate()
    s.squat(bottom=165)
    assert s.c.count == 0
    assert s.c.rejected == 0


def test_lifting_one_leg_not_counted():
    """Bitta oyoqni ko'tarish: bitta tizza bukiladi, ikkinchisi to'g'ri, to'piq ko'tariladi."""
    s = Sim()
    s.calibrate()
    for _ in range(3):
        s.ramp(178, 70, 15, knee_r_fn=lambda v: 178, lift_l=0.2)
        s.feed(5, knee_l=70, knee_r=178, lift_l=0.2)
        s.ramp(70, 178, 15, knee_r_fn=lambda v: 178)
        s.feed(5)
    assert s.c.count == 0


def test_asymmetric_knees_rejected():
    s = Sim()
    s.calibrate()
    s.squat(knee_r_fn=lambda v: min(178, v + 50))
    assert s.c.count == 0
    assert any("tizza" in m for _, m in s.events)


def test_heel_or_foot_lift_during_squat_rejected():
    s = Sim()
    s.calibrate()
    s.squat(lift_l=0.15)
    assert s.c.count == 0
    assert any("ko'tarildi" in m for _, m in s.events)


def test_too_fast_rejected():
    s = Sim()
    s.calibrate()
    s.squat(down=3, hold=2, up=3)
    assert s.c.count == 0


def test_body_leaving_frame_resets():
    s = Sim()
    s.calibrate()
    s.feed(30, vis=0.1)
    assert s.c.state == State.NO_PERSON
    s.squat()
    assert s.c.count == 0


def test_moving_feet_while_standing_recalibrates():
    s = Sim()
    s.calibrate()
    s.feed(5, dx=0.3)
    assert s.c.state == State.CALIBRATING
