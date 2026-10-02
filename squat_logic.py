"""Squat sanash mantig'i — kameraga bog'liq emas, alohida test qilinadi.

Bir takror faqat quyidagilarning HAMMASI bajarilsa sanaladi:
  1. Tana to'liq ko'rinadi (yelka, son, tizza, to'piq — ikkala tomon).
  2. Avval tik turib kalibrovka qilinadi (tos va to'piq holati eslab qolinadi).
  3. IKKALA tizza ham belgilangan burchakdan pastga bukiladi (3D burchak).
  4. Tos haqiqatan pastga tushadi (oyoq uzunligiga nisbatan).
  5. Ikkala tizza deyarli bir xil bukiladi (bir oyoqni ko'tarish ishlamaydi).
  6. Ikkala oyoq yerda qoladi va joyidan siljimaydi.
  7. Pastki nuqtada bir necha kadr ushlanadi va yana to'liq tik turiladi.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

# MediaPipe Pose landmark indekslari
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
LIFT = "oyoq yerdan ko'tarildi"
REQUIRED = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)


class State(Enum):
    NO_PERSON = "Odam ko'rinmayapti"
    CALIBRATING = "Sozlanmoqda"
    READY = "Tayyor"
    SQUATTING = "O'tirish"


@dataclass
class Config:
    min_visibility: float = 0.5
    stand_knee_angle: float = 155.0   # ikkala tizza shundan katta bo'lsa — tik turibdi
    bottom_knee_angle: float = 100.0  # ikkala tizza shundan kichik bo'lishi shart
    min_hip_drop: float = 0.25        # tos kamida oyoq uzunligining shuncha qismiga tushishi kerak
    start_hip_drop: float = 0.08      # harakat boshlandi deb hisoblash chegarasi
    max_knee_asym: float = 30.0       # tizzalar orasidagi ruxsat etilgan farq (gradus)
    asym_frames: int = 4              # farq shuncha kadr ketma-ket bo'lsa — rad etiladi
    max_ankle_lift: float = 0.10      # to'piq ko'tarilishi (oyoq uzunligiga nisbatan)
    max_ankle_move: float = 0.15      # to'piqning umumiy siljishi
    min_rep_time: float = 0.6         # s — bundan tezroq takror sanalmaydi
    bottom_hold_frames: int = 2
    calib_time: float = 1.0           # s — kalibrovka uchun tik turish vaqti
    calib_max_jitter: float = 0.03
    min_leg_len: float = 0.15         # kadr balandligiga nisbatan; kichik bo'lsa — juda uzoqda
    lost_grace: float = 0.5           # s — tana qisqa vaqt yo'qolsa kechiriladi


@dataclass
class Measure:
    knee_l: float
    knee_r: float
    hip_x: float
    hip_y: float
    ankles: np.ndarray  # (2, 2) — [chap, o'ng] x, y (rasm koordinatalari)


@dataclass
class Baseline:
    hip_x: float
    hip_y: float
    ankles: np.ndarray
    leg_len: float


@dataclass
class Rep:
    start: float
    reached_bottom: bool = False
    bottom_frames: int = 0
    asym_count: int = 0
    max_depth: float = 0.0
    fail_reason: str | None = None


@dataclass
class Update:
    state: State
    count: int
    rejected: int
    message: str
    message_ok: bool | None      # True — yaxshi, False — xato, None — neytral
    depth: float                 # 0..1 — pastki nuqtaga qanchalik yaqin
    knee_l: float | None
    knee_r: float | None
    event: str | None            # "rep" | "reject" | None
    fail: bool                   # joriy takror buzilganmi


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """b nuqtadagi burchak (gradus), a-b-c bo'yicha."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


class SquatCounter:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.rejected = 0
        self.state = State.NO_PERSON
        self.base: Baseline | None = None
        self.rep: Rep | None = None
        self._calib: deque[tuple[float, Measure]] = deque()
        self._last_seen: float | None = None
        self._last: Measure | None = None
        self.message = "Kameraga to'liq ko'rining (boshdan oyoqqacha)"
        self.message_ok: bool | None = None

    # ------------------------------------------------------------------ public
    def update(self, t: float, img: np.ndarray | None, world: np.ndarray | None) -> Update:
        """img: (33, 3) — x, y, visibility (0..1); world: (33, 3) — metrdagi x, y, z."""
        cfg = self.cfg
        visible = (
            img is not None
            and world is not None
            and all(img[i, 2] >= cfg.min_visibility for i in REQUIRED)
        )
        if not visible:
            return self._handle_lost(t)

        self._last_seen = t
        m = self._measure(img, world)
        self._last = m

        if self.state in (State.NO_PERSON, State.CALIBRATING):
            return self._calibrate(t, m)
        if self.state == State.READY:
            return self._ready(t, m)
        return self._squatting(t, m)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _measure(img: np.ndarray, world: np.ndarray) -> Measure:
        w = world[:, :3]
        return Measure(
            knee_l=joint_angle(w[L_HIP], w[L_KNEE], w[L_ANKLE]),
            knee_r=joint_angle(w[R_HIP], w[R_KNEE], w[R_ANKLE]),
            hip_x=float((img[L_HIP, 0] + img[R_HIP, 0]) / 2),
            hip_y=float((img[L_HIP, 1] + img[R_HIP, 1]) / 2),
            ankles=np.array([img[L_ANKLE, :2], img[R_ANKLE, :2]], dtype=float),
        )

    def _out(self, depth: float = 0.0, event: str | None = None) -> Update:
        m = self._last
        return Update(
            state=self.state,
            count=self.count,
            rejected=self.rejected,
            message=self.message,
            message_ok=self.message_ok,
            depth=depth,
            knee_l=m.knee_l if m else None,
            knee_r=m.knee_r if m else None,
            event=event,
            fail=bool(self.rep and self.rep.fail_reason),
        )

    def _say(self, text: str, ok: bool | None = None) -> None:
        self.message, self.message_ok = text, ok

    def _is_standing(self, m: Measure) -> bool:
        return min(m.knee_l, m.knee_r) >= self.cfg.stand_knee_angle

    def _hip_drop(self, m: Measure) -> float:
        return (m.hip_y - self.base.hip_y) / self.base.leg_len

    def _start_calibration(self, text: str) -> None:
        self.state = State.CALIBRATING
        self.base = None
        self.rep = None
        self._calib.clear()
        self._say(text)

    # ------------------------------------------------------------------ states
    def _handle_lost(self, t: float) -> Update:
        self._last = None
        lost_for = math.inf if self._last_seen is None else t - self._last_seen
        if lost_for < self.cfg.lost_grace:
            return self._out()
        event = None
        if self.state == State.SQUATTING and self.rep and self.rep.max_depth > 0.3:
            self.rejected += 1
            event = "reject"
        self.state = State.NO_PERSON
        self.base = None
        self.rep = None
        self._calib.clear()
        self._say("Butun tanangiz (oyoqlar bilan) kadrda ko'rinsin", False)
        return self._out(event=event)

    def _calibrate(self, t: float, m: Measure) -> Update:
        cfg = self.cfg
        self.state = State.CALIBRATING
        leg_len = float(m.ankles[:, 1].mean() - m.hip_y)
        if leg_len < cfg.min_leg_len:
            self._calib.clear()
            self._say("Kameraga yaqinroq keling")
            return self._out()
        if not self._is_standing(m):
            self._calib.clear()
            self._say("To'g'ri tik turing")
            return self._out()

        self._calib.append((t, m))
        while self._calib and self._calib[0][0] < t - cfg.calib_time:
            self._calib.popleft()
        span = t - self._calib[0][0]
        hips = np.array([s.hip_y for _, s in self._calib])
        if hips.max() - hips.min() > cfg.calib_max_jitter * leg_len:
            self._calib.clear()
            self._say("Qimirlamay tik turing...")
            return self._out()
        if span < cfg.calib_time * 0.9:
            self._say("Qimirlamay tik turing...")
            return self._out()

        samples = [s for _, s in self._calib]
        ankles = np.mean([s.ankles for s in samples], axis=0)
        hip_y = float(np.mean([s.hip_y for s in samples]))
        self.base = Baseline(
            hip_x=float(np.mean([s.hip_x for s in samples])),
            hip_y=hip_y,
            ankles=ankles,
            leg_len=float(ankles[:, 1].mean() - hip_y),
        )
        self._calib.clear()
        self.state = State.READY
        self._say("Tayyor! O'tirib-turishni boshlang", True)
        return self._out()

    def _ankle_problem(self, m: Measure) -> str | None:
        cfg, b = self.cfg, self.base
        for side in (0, 1):
            dx, dy = m.ankles[side] - b.ankles[side]
            if -dy / b.leg_len > cfg.max_ankle_lift:  # rasmda y pastga qarab o'sadi
                return LIFT
            if math.hypot(dx, dy) / b.leg_len > cfg.max_ankle_move:
                return "joyingizdan siljidingiz"
        return None

    def _ready(self, t: float, m: Measure) -> Update:
        cfg, b = self.cfg, self.base
        drop = self._hip_drop(m)
        problem = self._ankle_problem(m)
        started = (
            min(m.knee_l, m.knee_r) < cfg.stand_knee_angle - 10
            or drop > cfg.start_hip_drop
            or problem == LIFT  # oyoq ko'tarilishi ham harakat — rad etilishi uchun takrorni boshlaymiz
        )
        if started:
            self.state = State.SQUATTING
            self.rep = Rep(start=t)
            return self._squatting(t, m)

        if problem:
            # Tik turgan holda joyini o'zgartirgan — qaytadan sozlaymiz.
            self._start_calibration("Joyingiz o'zgardi — qayta sozlanmoqda")
            return self._out()

        # Tik turganda bazani sekin yangilab boramiz (kichik siljishlarga moslashish).
        a = 0.05
        b.hip_y += a * (m.hip_y - b.hip_y)
        b.hip_x += a * (m.hip_x - b.hip_x)
        b.ankles += a * (m.ankles - b.ankles)
        b.leg_len = float(b.ankles[:, 1].mean() - b.hip_y)
        return self._out()

    def _squatting(self, t: float, m: Measure) -> Update:
        cfg, rep = self.cfg, self.rep
        drop = self._hip_drop(m)
        deeper_knee = max(m.knee_l, m.knee_r)  # IKKALA tizza ham bukilishi kerak

        knee_prog = (cfg.stand_knee_angle - deeper_knee) / (cfg.stand_knee_angle - cfg.bottom_knee_angle)
        drop_prog = drop / cfg.min_hip_drop
        depth = float(np.clip(min(knee_prog, drop_prog), 0.0, 1.0))
        rep.max_depth = max(rep.max_depth, depth)

        if rep.fail_reason is None:
            problem = self._ankle_problem(m)
            if problem is None:
                if abs(m.knee_l - m.knee_r) > cfg.max_knee_asym:
                    rep.asym_count += 1
                else:
                    rep.asym_count = 0
                if rep.asym_count >= cfg.asym_frames:
                    problem = "ikkala tizza bir xil bukilmadi"
            if problem:
                rep.fail_reason = problem

        if rep.fail_reason is None:
            if deeper_knee <= cfg.bottom_knee_angle and drop >= cfg.min_hip_drop:
                rep.bottom_frames += 1
                if rep.bottom_frames >= cfg.bottom_hold_frames:
                    rep.reached_bottom = True
            else:
                rep.bottom_frames = 0

        # Takror tugadimi? — ikkala tizza to'g'rilangan va tos joyiga qaytgan.
        if self._is_standing(m) and drop < cfg.start_hip_drop:
            return self._finish(t)

        if rep.fail_reason:
            self._say(f"Hisoblanmaydi: {rep.fail_reason}", False)
        elif rep.reached_bottom:
            self._say("Yaxshi! Endi to'liq turing", True)
        else:
            self._say("Pastroq tushing...")
        return self._out(depth)

    def _finish(self, t: float) -> Update:
        cfg, rep = self.cfg, self.rep
        self.state = State.READY
        self.rep = None
        duration = t - rep.start

        reason = rep.fail_reason
        if reason is None and not rep.reached_bottom:
            if rep.max_depth < 0.3:
                # Kichik qimirlash — takror deb hisoblamaymiz ham, rad ham etmaymiz.
                self._say("Tayyor", None)
                return self._out()
            reason = "yetarlicha pastga tushmadingiz"
        if reason is None and duration < cfg.min_rep_time:
            reason = "juda tez"

        if reason:
            self.rejected += 1
            self._say(f"Hisoblanmadi: {reason}", False)
            return self._out(event="reject")

        self.count += 1
        self._say(f"Zo'r! {self.count}", True)
        return self._out(event="rep")
