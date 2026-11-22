"""Training progress: one live status line on a terminal, plain lines elsewhere.

On a terminal it redraws a single line with a coloured spinner, steps done,
percentage, elapsed time, time remaining and speed. When output is a file or a
pipe (a GUI reading it, a log) it prints one parseable line every `every` steps
instead, so both ways of launching training show the same numbers.
"""

from __future__ import annotations

import os
import sys
import time

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
COLOURS = (196, 202, 208, 214, 220, 190, 118, 48, 51, 45, 39, 63, 99, 135, 171, 207)


def clock(seconds: float) -> str:
    s = int(max(seconds, 0))
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    def __init__(self, total: int, start: int = 0, every: int = 100, stream=None):
        self.total, self.start, self.every = total, start, every
        self.out = stream or sys.stdout
        self.tty = self.out.isatty()
        self.frames = FRAMES
        if self.tty:
            if os.name == "nt":
                os.system("")  # switches the Windows console to ANSI escape handling
            try:
                FRAMES.encode(getattr(self.out, "encoding", None) or "ascii")
            except (UnicodeError, LookupError):
                self.frames = "|/-\\"
        self.t0 = self.drawn = time.time()
        self.paused = 0.0  # time spent outside training (saving, writing samples)
        self.frame = 0
        # CAUSALVC_PROGRESS_SECONDS: when not on a terminal, print by time instead of every `every` steps (for a GUI)
        self.seconds = float(os.environ.get("CAUSALVC_PROGRESS_SECONDS", 0) or 0)

    def pause(self):
        self._pause = time.time()

    def resume(self):
        self.paused += time.time() - self._pause

    def line(self, step: int, extra: str = "") -> str:
        elapsed = time.time() - self.t0
        done = step - self.start
        rate = done / max(elapsed - self.paused, 1e-6)
        left = (self.total - step) / rate if rate > 0 else 0.0
        pct = 100 * step / self.total
        return f"step {step}/{self.total} ({pct:.0f}%)  elapsed {clock(elapsed)}  remaining {clock(left)}  {rate:.2f} it/s" + (f"  {extra}" if extra else "")

    def update(self, step: int, extra: str = "") -> None:
        """Call once per step; `extra` (losses) is shown as given."""
        now = time.time()
        if self.tty:
            if now - self.drawn < 0.1 and step < self.total:
                return
            self.drawn, self.frame = now, self.frame + 1
            spin = f"\x1b[38;5;{COLOURS[self.frame % len(COLOURS)]}m{self.frames[self.frame % len(self.frames)]}\x1b[0m"
            self.out.write(f"\r\x1b[2K{spin} {self.line(step, extra)}")
            if step >= self.total:
                self.out.write("\n")
            self.out.flush()
        elif (now - self.drawn >= self.seconds if self.seconds else step % self.every == 0) or step >= self.total:
            self.drawn = now
            print(self.line(step, extra), file=self.out, flush=True)

    def note(self, text: str) -> None:
        """A message that should stay on screen (checkpoint saved, and so on)."""
        if self.tty:
            self.out.write(f"\r\x1b[2K{text}\n")
            self.out.flush()
        else:
            print(text, file=self.out, flush=True)
