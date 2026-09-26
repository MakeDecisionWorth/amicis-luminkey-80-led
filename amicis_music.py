"""AMICIS 80HE JIS: music-reactive LEDs driven by the PC's audio output.

Captures an output device through WASAPI loopback (PyAudioWPatch), so whatever
is playing on the PC drives the keyboard. Defaults to the default output device
(or Voicemeeter Out B1 when the default output is a Voicemeeter input).
--audio can also name a WASAPI input device, e.g. a Voicemeeter output bus
("Voicemeeter Out B1"), which is where the mix lives when Voicemeeter owns the
hardware output and its loopback stays silent.

Modes:
    spectrum  columns are frequency bands (bass left, treble right); bars grow
              up from the bottom row, with a slowly falling peak marker
    pulse     bass beats launch colour ripples from the centre of the board

Both use a solid colour (--colour, default #8000FF); --rainbow switches to
rainbow colours (spectrum: hue across the board, pulse: new hue per beat).

Usage:
    python amicis_music.py --list-audio          # capture devices
    python amicis_music.py                       # spectrum until Ctrl+C
    python amicis_music.py --mode pulse
    python amicis_music.py --rainbow             # rainbow instead of solid colour
    python amicis_music.py --colour "#00C8FF"    # another solid colour
    python amicis_music.py --audio "Out B1"      # pick a capture device by name
"""

import argparse
import sys
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

import amicis_led

STALE_AFTER = 0.15      # loopback delivers nothing while silent; treat old data as silence


def capture_devices(pa):
    """WASAPI loopback devices first, then ordinary WASAPI input devices."""
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
    loopbacks = list(pa.get_loopback_device_info_generator())
    inputs = [d for d in map(pa.get_device_info_by_index, range(pa.get_device_count()))
              if d["hostApi"] == wasapi and d["maxInputChannels"] > 0
              and not d.get("isLoopbackDevice")]
    return loopbacks + inputs


class AudioCapture:
    """Keeps the most recent mono samples of a loopback or input stream."""

    def __init__(self, name=None, keep_seconds=0.5):
        self.pa = pyaudio.PyAudio()
        dev = self._pick(name)
        self.name = dev["name"]
        self.rate = int(dev["defaultSampleRate"])
        self.channels = dev["maxInputChannels"]
        self.buf = np.zeros(int(self.rate * keep_seconds), np.float32)
        self.last_data = 0.0
        self.lock = threading.Lock()
        self.stream = self.pa.open(format=pyaudio.paFloat32, channels=self.channels,
                                   rate=self.rate, input=True, input_device_index=dev["index"],
                                   frames_per_buffer=512, stream_callback=self._callback)

    def _pick(self, name):
        devices = capture_devices(self.pa)
        if name:
            match = ([d for d in devices if d["name"] == name]
                     or [d for d in devices if name.lower() in d["name"].lower()])
        else:
            wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            out = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            # Voicemeeter's VAIO loopback is silent; its main mix is the Out B1 bus.
            match = ([d for d in devices if not d.get("isLoopbackDevice")
                      and d["name"].startswith("Voicemeeter Out B1")]
                     if out["name"].startswith("Voicemeeter") else [])
            match += [d for d in devices
                      if d.get("isLoopbackDevice") and d["name"].startswith(out["name"])]
        if not match:
            self.pa.terminate()
            raise SystemExit(f"No capture device for {name or 'the default output'!r}; see --list-audio")
        return match[0]

    def _callback(self, data, frames, time_info, status):
        x = np.frombuffer(data, np.float32).reshape(-1, self.channels).mean(axis=1)
        n = min(len(x), len(self.buf))
        with self.lock:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = x[-n:]
            self.last_data = time.monotonic()
        return None, pyaudio.paContinue

    def latest(self, n):
        with self.lock:
            if time.monotonic() - self.last_data > STALE_AFTER:
                return np.zeros(n, np.float32)
            return self.buf[-n:].copy()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


def band_centres(bands, fmin=40.0, fmax=16000.0):
    edges = np.geomspace(fmin, fmax, bands + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


class Spectrum:
    """Log-spaced band levels in dB from the latest audio window."""

    def __init__(self, rate, bands, fmin=40.0, fmax=16000.0, window_s=0.05):
        self.n = 1 << int(np.log2(rate * window_s))
        self.win = np.hanning(self.n).astype(np.float32)
        self.norm = self.win.sum() / 2          # full-scale sine -> 0 dB
        freqs = np.fft.rfftfreq(self.n, 1 / rate)
        edges, self.centres = band_centres(bands, fmin, fmax)
        self.bins = []
        for lo, hi, c in zip(edges[:-1], edges[1:], self.centres):
            idx = np.nonzero((freqs >= lo) & (freqs < hi))[0]
            self.bins.append(idx if len(idx) else np.array([np.argmin(np.abs(freqs - c))]))
        # Music falls off ~3 dB/octave; compensate so treble bars are not always short.
        self.tilt = 3.0 * np.log2(self.centres / 1000.0)

    def band_db(self, x, tilt=True):
        mag = np.abs(np.fft.rfft(x * self.win)) / self.norm
        db = np.array([10 * np.log10(np.mean(mag[i] ** 2) + 1e-12) for i in self.bins])
        return db + self.tilt if tilt else db


class AutoLevel:
    """Maps dB to 0..1 against a slowly released running peak (automatic gain).

    per_band=True gives every band its own running peak, so quiet bands still move.
    Those peaks never drop more than band_floor_db below the loudest band, so bands
    that are close to silent are not amplified into noise.
    """

    def __init__(self, range_db, floor_db=-50.0, release_db_s=6.0, per_band=False, band_floor_db=30.0):
        self.range = range_db
        self.floor = floor_db
        self.release = release_db_s
        self.per_band = per_band
        self.band_floor = band_floor_db
        self.global_ref = floor_db
        self.band_ref = None

    def __call__(self, db, dt, offset_db=0.0):
        """offset_db (scalar or per band) shifts the result after the gain, like an EQ."""
        self.global_ref = max(self.global_ref - self.release * dt, float(np.max(db)), self.floor)
        ref = self.global_ref
        if self.per_band:
            if self.band_ref is None:
                self.band_ref = np.full(len(db), self.floor)
            self.band_ref = np.maximum.reduce([self.band_ref - self.release * dt, db,
                                               np.full(len(db), self.global_ref - self.band_floor)])
            ref = self.band_ref
        return np.clip((db + offset_db - (ref - self.range)) / self.range, 0.0, 1.0)


def hsv(h, s=1.0, v=1.0):
    """Vectorised HSV -> RGB; returns an (n, 3) array."""
    h = np.asarray(h) % 1.0
    k = (np.array([5.0, 3.0, 1.0]) + h[..., None] * 6) % 6
    return v - v * s * np.clip(np.minimum(k, 4 - k), 0, 1)


def parse_colour(text):
    """'#8000FF' / '8000FF' -> RGB array in 0..1."""
    value = text.lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {text!r}")
    try:
        return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {text!r}") from None


def parse_eq(text):
    """'0,0,3,...' -> per-band dB offsets (missing bands are 0)."""
    try:
        return [float(v) for v in text.split(",")] if text.strip() else []
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated dB values, got {text!r}") from None


def eq_offsets(args, centres=None):
    """The --eq values (one per --bands band). With centres, the EQ is read as a
    curve over frequency and sampled there, so other band layouts can use it."""
    eq = np.zeros(args.bands)
    values = list(args.eq)[:args.bands]
    eq[:len(values)] = values
    if centres is None:
        return eq
    _, base = band_centres(args.bands)
    return np.interp(np.log(centres), np.log(base), eq)


def band_layout(xs, ys, layout, bands):
    """Assign keys to bands and give each key its fill threshold along the bar.

    columns: `bands` equal-width columns, bars grow up from the bottom row.
    rows:    one band per keyboard row (bass at the bottom), bars grow left to right,
             each row scaled to its own length.
    Returns (band index per key, threshold per key, number of bands).
    """
    if layout == "rows":
        values, counts = np.unique(ys, return_counts=True)
        rows = values[counts > 1]                   # the tall Enter key is not a row
        # Nearest row; the small bias sends a tie (Enter) to the lower row.
        row = np.argmin(np.abs(ys[:, None] - rows[None, :]) - 1e-3 * rows[None, :], axis=1)
        band = len(rows) - 1 - row
        pos = np.zeros(len(xs))
        for r in range(len(rows)):
            m = row == r
            span = xs[m].max() - xs[m].min()
            pos[m] = (xs[m] - xs[m].min()) / span if span else 0.0
        return band, 0.04 + pos * 0.92, len(rows)
    edges = np.linspace(xs.min() - 0.5, xs.max() + 0.5, bands + 1)
    band = np.clip(np.digitize(xs, edges) - 1, 0, bands - 1)
    height = (ys.max() - ys) / (ys.max() - ys.min())    # 0 = bottom row, 1 = top row
    return band, 0.04 + height * 0.92, bands


def parse_point(text):
    """'9,3' -> (9.0, 3.0) in layout units (1 unit = one key)."""
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y, got {text!r}") from None
    return x, y


def key_geometry(leds):
    xs = np.array([x for _, x, _, _ in leds])
    ys = np.array([y for _, _, y, _ in leds])
    return xs, ys


def spectrum_renderer(cap, leds, args):
    """Returns render(t, dt) -> colours; reads args on every frame so they can change live."""
    xs, ys = key_geometry(leds)
    # Each key belongs to one band, so a bar moves as a unit instead of blending between bands.
    column, threshold, n_bands = band_layout(xs, ys, args.layout, args.bands)
    an = Spectrum(cap.rate, n_bands)
    level = AutoLevel(args.range, per_band=True)
    hue_column = column / max(n_bands - 1, 1) * 0.8
    bars = np.zeros(n_bands)
    peaks = np.zeros(n_bands)
    onset = {"avg": None, "last": np.full(n_bands, -1.0)}

    def render(t, dt):
        level.range = args.range
        level.per_band = args.reference == "per-band"
        level.band_floor = args.band_floor
        db, eq = an.band_db(cap.latest(an.n)), eq_offsets(args, an.centres)
        lv = level(db, dt, eq)        # always run, so the auto peaks stay current when switched to
        if args.reference == "fixed":
            lv = np.clip((db + eq - (args.top_db - args.range)) / args.range, 0.0, 1.0)
        if args.drive == "onset":
            # Only a band's attacks push its bar: the band has to jump onset_db above its
            # own recent average. Sustained sound does not hold the bar up; it falls back.
            now_db = db + eq
            if onset["avg"] is None:
                onset["avg"] = now_db.copy()
            hit = (now_db - onset["avg"] > args.onset_db) & (t - onset["last"] > args.onset_gap)
            onset["last"][hit] = t
            onset["avg"] += (now_db - onset["avg"]) * (1 - np.exp(-dt / 0.35))
            lv = np.where(hit, lv, 0.0)
        bars[:] = np.maximum(lv, bars - args.fall * dt)
        peaks[:] = np.maximum(bars, peaks - args.peak_fall * dt)
        bar = bars[column]
        peak = peaks[column]
        colour = hsv(hue_column + t * args.hue_drift) if args.rainbow else np.tile(args.colour, (len(xs), 1))
        fill = np.clip((bar - threshold) / 0.12 + 0.5, 0, 1)
        marker = np.clip(1 - np.abs(peak - threshold) / 0.1, 0, 1) * (peak > 0.08) * (args.peak_level > 0)
        out = colour * fill[:, None]
        # Rainbow peaks are tinted towards white; solid peaks are a dimmer version of the
        # colour so they read as a marker rather than a gap in the bar.
        peak_colour = 0.55 + 0.45 * colour if args.rainbow else colour * args.peak_level
        return np.maximum(out, marker[:, None] * peak_colour)

    return render


def pulse_centre(leds, args):
    """Effect origin: --centre if given; otherwise the Space bar for the campfire,
    and the mean of all key centres for the pulse ripples."""
    if args.centre:
        return args.centre
    if args.mode == "fire":
        for _, x, y, label in leds:
            if label == "Space":
                return x, y
    xs, ys = key_geometry(leds)
    return float(xs.mean()), float(ys.mean())


class BeatDetector:
    """Bass beats: bass energy jumping --threshold times above its recent average.

    Call once per frame; returns (beat, glow) where glow is the 0..1 bass level.
    """

    def __init__(self, rate, args):
        self.args = args
        self.an = Spectrum(rate, 24)
        self.bass = self.an.centres < 150
        self.level = AutoLevel(args.range)
        self.avg = 0.0
        self.last_beat = -1.0

    def __call__(self, cap, t, dt):
        self.level.range = self.args.range
        db = self.an.band_db(cap.latest(self.an.n), tilt=False)
        energy = float(np.mean(10 ** (db[self.bass] / 10)))
        glow = float(np.max(self.level(db[self.bass], dt)))
        self.avg += (energy - self.avg) * (1 - np.exp(-dt / 0.4))
        beat = (energy > self.avg * self.args.threshold and glow > 0.3
                and t - self.last_beat > self.args.min_gap)
        if beat:
            self.last_beat = t
        return beat, glow


def pulse_renderer(cap, leds, args):
    beats = BeatDetector(cap.rate, args)
    xs, ys = key_geometry(leds)
    state = {"hue": 0.0}
    rings = []                                         # (start time, hue)

    def colour_of(hue):
        return hsv(np.full(len(xs), hue)) if args.rainbow else np.tile(args.colour, (len(xs), 1))

    def render(t, dt):
        cx, cy = pulse_centre(leds, args)
        dist = np.hypot(xs - cx, ys - cy)
        beat, glow = beats(cap, t, dt)
        if beat:
            rings.append((t, state["hue"]))
            state["hue"] = (state["hue"] + args.hue_step) % 1.0
        rings[:] = [r for r in rings if t - r[0] < 1.5]

        flash = np.exp(-(t - beats.last_beat) / 0.12) * 0.3
        out = colour_of(state["hue"]) * (0.06 + 0.2 * glow + flash)
        for start, hue in rings:
            age = t - start
            wave = np.exp(-((dist - age * args.ring_speed) / args.ring_width) ** 2) * np.exp(-age / 0.6)
            out = np.maximum(out, colour_of(hue) * wave[:, None])
        return out

    return render


# heat -> colour stops: black, ember red, red, orange, yellow, bright yellow core.
# Almost no blue: a blue-tinted "white-hot" core makes the LED's separate blue die
# show as light blue / white fringes through the gaps around the keycaps.
FIRE_PALETTE = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.20, 0.30, 0.00, 0.00],
    [0.45, 0.90, 0.07, 0.00],
    [0.70, 1.00, 0.36, 0.00],
    [0.88, 1.00, 0.72, 0.00],
    [1.00, 1.00, 0.92, 0.08],
])
FIRE_CELL = 0.5          # simulation cell size in keys
FIRE_STEPS_PER_S = 40
SPARK_COLOUR = np.array([1.0, 0.62, 0.0])      # warm orange ember, no blue


def fire_colour(heat):
    h = np.clip(heat, 0, 1)
    return np.stack([np.interp(h, FIRE_PALETTE[:, 0], FIRE_PALETTE[:, i]) for i in (1, 2, 3)], axis=-1)


def fire_renderer(cap, leds, args):
    """Campfire: heat rises and cools on a fine grid, fed from the bottom middle of the
    board; each bass beat makes the flames flare up and throws sparks."""
    beats = BeatDetector(cap.rate, args)
    rng = np.random.default_rng()
    xs, ys = key_geometry(leds)
    cell = FIRE_CELL
    w = int(np.ceil((xs.max() + 1.5) / cell))
    h = int(np.ceil((ys.max() + 0.5) / cell)) + 1        # +1: hidden fuel row below the keys
    heat = np.zeros((h, w))
    gx = (np.arange(w) + 0.5) * cell

    # Each key shows the mean heat of the cells it covers.
    cover = np.zeros((len(leds), h, w))
    for i, (_, x, y, label) in enumerate(leds):
        kw, kh = amicis_led.key_size(label)
        x0, x1 = int((x - kw / 2) / cell), int(np.ceil((x + kw / 2) / cell))
        y0, y1 = int((y - kh / 2) / cell), int(np.ceil((y + kh / 2) / cell))
        cover[i, y0:y1, x0:x1] = 1
    cover = cover.reshape(len(leds), -1)
    cover /= cover.sum(axis=1, keepdims=True)

    sparks = []                                          # [x, y, vx, vy, age, life]
    state = {"sim": 0.0}

    def spawn_sparks(count, cx):
        for _ in range(count):
            sparks.append([cx + rng.normal(0, args.fire_width * 0.5), ys.max() + 0.3,
                           rng.normal(0, 1.2), -rng.uniform(6, 11), 0.0, rng.uniform(0.5, 1.0)])

    def render(t, dt):
        cx = pulse_centre(leds, args)[0]
        beat, glow = beats(cap, t, dt)
        if beat:
            spawn_sparks(args.sparks, cx)
        elif rng.random() < 0.8 * dt:                     # the odd spark even without music
            spawn_sparks(1, cx)
        flare = np.exp(-(t - beats.last_beat) / 0.35)
        fuel = (0.8 + 0.3 * glow) * (1 + 0.8 * flare)
        cooling = 0.22 / args.fire_height * (1 - 0.4 * flare)
        profile = np.exp(-0.5 * ((gx - cx) / args.fire_width) ** 2)

        state["sim"] += dt
        while state["sim"] > 0:                          # fixed-rate simulation at any frame rate
            state["sim"] -= 1 / FIRE_STEPS_PER_S
            heat[-1] = profile * fuel * rng.uniform(0.55, 1.0, w)
            below = heat[1:]
            spread = 0.25 * np.roll(below, 1, axis=1) + 0.5 * below + 0.25 * np.roll(below, -1, axis=1)
            heat[:-1] = np.maximum(spread * (1 - cooling * rng.uniform(0.6, 1.4, below.shape)), 0)

        out = fire_colour(cover @ heat.ravel())
        for s in sparks:
            s[0] += s[2] * dt
            s[1] += s[3] * dt
            s[3] += 4 * dt                               # sparks slow down as they rise
            s[4] += dt
        sparks[:] = [s for s in sparks if s[4] < s[5] and s[1] > -0.5]
        for x, y, _, _, age, life in sparks:
            spark = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / 0.3) * (1 - age / life)
            out = np.maximum(out, SPARK_COLOUR * spark[:, None])
        return out

    return render


RENDERERS = {"spectrum": spectrum_renderer, "pulse": pulse_renderer, "fire": fire_renderer}


def list_audio():
    pa = pyaudio.PyAudio()
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
        for d in capture_devices(pa):
            loopback = bool(d.get("isLoopbackDevice"))
            mark = "*" if loopback and d["name"].startswith(default) else " "
            kind = "loopback" if loopback else "input   "
            print(f"{mark} {d['index']:3d}  {kind}  {int(d['defaultSampleRate'])} Hz  {d['name']}")
    finally:
        pa.terminate()


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    amicis_led.add_common_args(ap, fps=60, seconds=0, gamma=2.2, green=0.55)
    ap.add_argument("--mode", choices=tuple(RENDERERS), default="spectrum")
    ap.add_argument("--audio", help="capture device name substring (default: default output's loopback)")
    ap.add_argument("--list-audio", action="store_true", help="list capture devices and exit")
    ap.add_argument("--colour", "--color", type=parse_colour, default="#8000FF",
                    help="solid LED colour, #RRGGBB (default #8000FF)")
    ap.add_argument("--rainbow", action="store_true", help="use rainbow colours instead of --colour")
    ap.add_argument("--range", type=float, default=18, help="dynamic range shown, dB (smaller = livelier)")
    ap.add_argument("--reference", choices=("fixed", "per-band", "global"), default="fixed",
                    help="spectrum: bar top is --top-db (no auto gain), each band's own peak, "
                         "or the loudest band's peak")
    ap.add_argument("--top-db", type=float, default=-27,
                    help="spectrum fixed: level (dB, treble-compensated) that fills a bar")
    ap.add_argument("--band-floor", type=float, default=30,
                    help="spectrum per-band: a band's reference stays within this many dB of the loudest")
    # Default EQ lifts the top four bands: measured on real music they sit 5-17 dB
    # below the rest even after the 3 dB/octave tilt, leaving the right columns dark.
    ap.add_argument("--eq", type=parse_eq, default=[0.0] * 12 + [5.0, 11.0, 12.0, 12.0],
                    help="spectrum: per-band dB offsets after the gain, low to high, e.g. 3,3,0,0,-2")
    ap.add_argument("--layout", choices=("columns", "rows"), default="rows",
                    help="spectrum: vertical bars per column, or one left-to-right bar per row (6 bands)")
    ap.add_argument("--bands", type=int, default=16, help="spectrum columns: frequency bands (also the EQ grid)")
    ap.add_argument("--drive", choices=("onset", "level"), default="onset",
                    help="spectrum: bars pushed only by each band's attacks, or following the level")
    ap.add_argument("--onset-db", type=float, default=4.5,
                    help="spectrum onset: rise above the band's recent average that counts as an attack")
    ap.add_argument("--onset-gap", type=float, default=0.1, help="spectrum onset: minimum time between pushes (s)")
    ap.add_argument("--fall", type=float, default=0.8, help="spectrum: bar fall speed (bar lengths/s)")
    ap.add_argument("--peak-fall", type=float, default=0.5, help="spectrum: peak fall speed (height/s)")
    ap.add_argument("--peak-level", type=float, default=0.35,
                    help="spectrum: peak marker brightness in solid mode (0 hides markers)")
    ap.add_argument("--hue-drift", type=float, default=0.03, help="spectrum --rainbow: colour drift (cycles/s)")
    ap.add_argument("--threshold", type=float, default=1.6, help="pulse/fire: beat energy vs recent average")
    ap.add_argument("--min-gap", type=float, default=0.18, help="pulse/fire: minimum time between beats (s)")
    ap.add_argument("--fire-width", type=float, default=3.5, help="fire: spread of the fire bed (keys)")
    ap.add_argument("--fire-height", type=float, default=1.0, help="fire: flame height scale")
    ap.add_argument("--sparks", type=int, default=5, help="fire: sparks thrown per beat")
    ap.add_argument("--hue-step", type=float, default=0.13, help="pulse --rainbow: hue change per beat")
    ap.add_argument("--ring-speed", type=float, default=18.0, help="pulse: ripple speed (keys/s)")
    ap.add_argument("--ring-width", type=float, default=1.6, help="pulse: ripple width (keys)")
    ap.add_argument("--centre", "--center", type=parse_point,
                    help="pulse: ripple origin X,Y in key units (default: middle of the keys)")
    return ap


def main():
    args = build_parser().parse_args()

    if args.list_audio:
        list_audio()
        return 0

    def effect(kb, leds):
        cap = AudioCapture(args.audio)
        print(f"Audio: {cap.name} ({cap.rate} Hz, {cap.channels} ch), mode: {args.mode}")
        try:
            render = RENDERERS[args.mode](cap, leds, args)
            amicis_led.run_frames(kb, leds, render, args.fps, args.seconds, args.gamma, args.led_green)
        finally:
            cap.close()

    return amicis_led.run(args, effect)


if __name__ == "__main__":
    sys.exit(main())
