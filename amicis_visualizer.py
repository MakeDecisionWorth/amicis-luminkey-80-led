"""AMICIS 80HE JIS: on-screen visualizer for the music LED effects.

Draws the keyboard with every key showing its LED colour live, with controls
for mode, colour, rainbow, sensitivity and audio source. Frames can be mirrored
to the real keyboard; without one it runs as a preview only.

Usage:
    python amicis_visualizer.py                   # accepts amicis_music.py options
    python amicis_visualizer.py --mode pulse --rainbow
    python amicis_visualizer.py --no-keyboard     # preview only

In pulse and fire modes a crosshair marks the ripple origin / campfire position: left-click the keyboard to
move it, right-click to reset it to the default (fire: Space, pulse: middle of the keys). The
status bar shows the value to pass to amicis_music.py --centre.
"""

import ctypes
import sys
import threading
import time
import tkinter as tk
from tkinter import colorchooser, ttk

import numpy as np
import pyaudiowpatch as pyaudio

import amicis_led
import amicis_music

BG = "#15151c"
UNLIT = np.array([0x26, 0x26, 0x30]) / 255.0     # keycap colour with the LED off
LABEL = "#8a8a9c"
UNIT = 46                                          # px per key unit at 96 dpi
GAP = 4
MARGIN = 18
CENTRED_MODES = ("pulse", "fire")     # modes using --centre (fire uses its x as the campfire position)


class KeyboardSender(threading.Thread):
    """Sends the newest frame to the keyboard off the UI thread, dropping stale ones."""

    FADE = object()

    def __init__(self, kb, fade_seconds):
        super().__init__(daemon=True)
        self.kb = kb
        self.fade_seconds = fade_seconds
        self.pending = None
        self.running = True
        self.error = None
        self.cv = threading.Condition()

    def submit(self, frame):
        with self.cv:
            self.pending = frame
            self.cv.notify()

    def fade(self):
        self.submit(self.FADE)

    def run(self):
        while True:
            with self.cv:
                while self.running and self.pending is None:
                    self.cv.wait()
                if not self.running:
                    return
                item, self.pending = self.pending, None
            try:
                if item is self.FADE:
                    self.kb.fade_out(self.fade_seconds)
                else:
                    self.kb.send_frame(item)
            except Exception as e:  # unplugged etc.; keep the preview running
                self.error = e
                return

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify()
        self.join(timeout=2)
        try:
            if self.error is None:
                self.kb.fade_out(self.fade_seconds)
        finally:
            self.kb.close()


class EqWindow:
    """Sub-window for the spectrum gain reference: mode, band floor and a per-band EQ."""

    EQ_LIMIT = 12.0

    def __init__(self, app):
        self.app = app
        a = app.args
        self.win = tk.Toplevel(app.root)
        self.win.title("Range / EQ")
        self.win.configure(bg=BG)
        self.win.transient(app.root)
        f = ttk.Frame(self.win, style="Dark.TFrame", padding=12)
        f.pack(fill="both", expand=True)

        ttk.Label(f, text="Reference", style="Dark.TLabel").grid(row=0, column=0, sticky="w")
        self.reference = tk.StringVar(value=a.reference)
        refs = ttk.Frame(f, style="Dark.TFrame")
        refs.grid(row=0, column=1, columnspan=a.bands, sticky="w")
        for value, text in (("fixed", "Fixed (no auto gain)"),
                            ("per-band", "Auto, per band"),
                            ("global", "Auto, loudest band")):
            ttk.Radiobutton(refs, text=text, value=value, variable=self.reference, style="Dark.TRadiobutton",
                            command=self.changed).pack(side="left", padx=(0, 12))

        ttk.Label(f, text="Top level (dB)", style="Dark.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.top_label = ttk.Label(f, width=5, style="Dark.TLabel")
        self.top_label.grid(row=1, column=a.bands, sticky="w", pady=(8, 0))
        self.top = ttk.Scale(f, from_=-60, to=0, value=a.top_db, command=self.top_changed)
        self.top.grid(row=1, column=1, columnspan=a.bands - 1, sticky="we", pady=(8, 0))

        ttk.Label(f, text="Band floor (dB)", style="Dark.TLabel").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.floor_label = ttk.Label(f, width=5, style="Dark.TLabel")
        self.floor_label.grid(row=7, column=a.bands, sticky="w", pady=(8, 0))
        self.floor = ttk.Scale(f, from_=10, to=60, value=a.band_floor, command=self.floor_changed)
        self.floor.grid(row=7, column=1, columnspan=a.bands - 1, sticky="we", pady=(8, 0))

        ttk.Label(f, text="EQ (dB)", style="Dark.TLabel").grid(row=2, column=0, sticky="nw", pady=(12, 0))
        _, centres = amicis_music.band_centres(a.bands)
        eq = amicis_music.eq_offsets(a)
        self.eq_scales, self.eq_labels = [], []
        for i, hz in enumerate(centres):
            value = ttk.Label(f, width=5, anchor="center", style="Dark.TLabel")
            value.grid(row=2, column=1 + i, pady=(12, 0))
            s = ttk.Scale(f, orient="vertical", from_=self.EQ_LIMIT, to=-self.EQ_LIMIT, value=eq[i],
                          length=160, command=lambda v, i=i: self.eq_changed(i, v))
            s.grid(row=3, column=1 + i, padx=3)
            s.bind("<Double-Button-1>", lambda e, s=s: s.set(0))  # double-click: back to 0 dB
            hz_text = f"{hz / 1000:.1f}k" if hz >= 1000 else f"{hz:.0f}"
            ttk.Label(f, text=hz_text, style="Dark.TLabel").grid(row=4, column=1 + i)
            self.eq_scales.append(s)
            self.eq_labels.append(value)
        ttk.Label(f, text="Double-click a slider to reset it to 0 dB", style="Dark.TLabel",
                  foreground=LABEL).grid(row=5, column=1, columnspan=a.bands, sticky="w", pady=(4, 0))

        ttk.Button(f, text="Restore defaults", command=self.restore_defaults).grid(
            row=8, column=0, sticky="w", pady=(12, 0))
        self.cli = tk.StringVar()
        ttk.Entry(f, textvariable=self.cli, state="readonly", width=80).grid(
            row=8, column=1, columnspan=a.bands, sticky="we", pady=(12, 0), padx=(8, 0))
        self.refresh()

    def top_changed(self, v):
        self.app.args.top_db = round(float(v))
        self.refresh()

    def floor_changed(self, v):
        self.app.args.band_floor = round(float(v))
        self.refresh()

    def eq_changed(self, i, v):
        eq = [float(v) for v in amicis_music.eq_offsets(self.app.args)]
        eq[i] = round(float(v) * 2) / 2      # 0.5 dB steps
        self.app.args.eq = eq
        self.refresh()

    def changed(self):
        self.app.args.reference = self.reference.get()
        self.refresh()

    def restore_defaults(self):
        d = amicis_music.build_parser().parse_args([])
        a = self.app.args
        a.reference, a.top_db, a.band_floor, a.eq = d.reference, d.top_db, d.band_floor, list(d.eq)
        self.reference.set(d.reference)
        self.top.set(d.top_db)
        self.floor.set(d.band_floor)
        for s, v in zip(self.eq_scales, amicis_music.eq_offsets(d)):
            s.set(v)
        self.app.set_slider("range", d.range)
        self.refresh()

    def refresh(self):
        a = self.app.args
        self.top_label.configure(text=f"{a.top_db:g}")
        self.top.state(["!disabled"] if a.reference == "fixed" else ["disabled"])
        self.floor_label.configure(text=f"{a.band_floor:g}")
        self.floor.state(["!disabled"] if a.reference == "per-band" else ["disabled"])
        eq = amicis_music.eq_offsets(a)
        for label, v in zip(self.eq_labels, eq):
            label.configure(text=f"{v:+g}" if v else "0")
        flags = [f"--range {a.range:g}", f"--reference {a.reference}"]
        flags.append(f"--top-db {a.top_db:g}" if a.reference == "fixed" else "")
        flags.append(f"--band-floor {a.band_floor:g}" if a.reference == "per-band" else "")
        flags = [f for f in flags if f]
        if eq.any():
            flags.append("--eq " + ",".join(f"{v:g}" for v in eq))
        self.cli.set(" ".join(flags))


def rounded_rect(canvas, x0, y0, x1, y1, r, **kw):
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


def to_hex(rgb):
    r, g, b = (np.clip(rgb, 0, 1) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


class App:
    def __init__(self, root, args, kb, leds):
        self.root = root
        self.args = args
        self.leds = leds
        self.sender = KeyboardSender(kb, args.fade) if kb else None
        if self.sender:
            self.sender.start()
        self.cap = None
        self.render = None
        self.closing = False

        scale = root.winfo_fpixels("1i") / 96
        self.unit = UNIT * scale
        root.title("AMICIS 80HE Visualizer")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_canvas(scale)
        self._build_controls()
        self.open_audio(args.audio)

        self.t0 = self.prev = time.monotonic()
        self.frames, self.fps_mark = 0, self.t0
        self.tick()

    # --- UI -----------------------------------------------------------------

    def _build_canvas(self, scale):
        u, gap, margin = self.unit, GAP * scale, MARGIN * scale
        self.margin = margin
        right = max(x + amicis_led.key_size(label)[0] / 2 for _, x, _, label in self.leds)
        bottom = max(y + amicis_led.key_size(label)[1] / 2 for _, _, y, label in self.leds)
        self.canvas = tk.Canvas(self.root, width=right * u + 2 * margin,
                                height=bottom * u + 2 * margin, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=8, pady=(8, 0))
        self.key_items, self.label_items = [], []
        font = ("Segoe UI", max(7, int(8 * scale)))
        for _, x, y, label in self.leds:
            w, h = amicis_led.key_size(label)
            x0, y0 = margin + (x - w / 2) * u + gap / 2, margin + (y - h / 2) * u + gap / 2
            x1, y1 = margin + (x + w / 2) * u - gap / 2, margin + (y + h / 2) * u - gap / 2
            self.key_items.append(rounded_rect(self.canvas, x0, y0, x1, y1, 6 * scale,
                                               fill=to_hex(UNLIT), outline=""))
            self.label_items.append(self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label,
                                                            fill=LABEL, font=font))
        self.last_fill = [None] * len(self.leds)

        # Pulse origin crosshair, drawn above the keys.
        r = 0.32 * u
        self.centre_items = [
            self.canvas.create_oval(-r, -r, r, r, outline="#ffffff", width=2 * scale),
            self.canvas.create_line(-1.6 * r, 0, 1.6 * r, 0, fill="#ffffff", width=scale),
            self.canvas.create_line(0, -1.6 * r, 0, 1.6 * r, fill="#ffffff", width=scale),
        ]
        self.centre_drawn = None
        self.canvas.bind("<Button-1>", self.set_centre)
        self.canvas.bind("<Button-3>", self.reset_centre)

    def _build_controls(self):
        style = ttk.Style()
        style.configure("Dark.TFrame", background=BG)
        style.configure("Dark.TLabel", background=BG, foreground="#d8d8e4")
        style.configure("Dark.TCheckbutton", background=BG, foreground="#d8d8e4")
        style.configure("Dark.TRadiobutton", background=BG, foreground="#d8d8e4")
        bar = ttk.Frame(self.root, style="Dark.TFrame", padding=(12, 8))
        bar.pack(fill="x")
        a = self.args

        ttk.Label(bar, text="Mode", style="Dark.TLabel").grid(row=0, column=0, sticky="w")
        self.mode = tk.StringVar(value=a.mode)
        modes = ttk.Frame(bar, style="Dark.TFrame")
        modes.grid(row=0, column=1, columnspan=2, sticky="w")
        for m in amicis_music.RENDERERS:
            ttk.Radiobutton(modes, text=m.capitalize(), value=m, variable=self.mode, style="Dark.TRadiobutton",
                            command=self.rebuild_renderer).pack(side="left", padx=(0, 8))
        self.layout = tk.StringVar(value=a.layout)
        layouts = ttk.Frame(bar, style="Dark.TFrame")
        layouts.grid(row=1, column=7, sticky="w", pady=(6, 0))
        ttk.Label(layouts, text="Spectrum", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        for value, text in (("columns", "Columns"), ("rows", "Rows")):
            ttk.Radiobutton(layouts, text=text, value=value, variable=self.layout, style="Dark.TRadiobutton",
                            command=self.rebuild_renderer).pack(side="left", padx=(0, 6))
        self.drive = tk.StringVar(value=a.drive)
        drives = ttk.Frame(bar, style="Dark.TFrame")
        drives.grid(row=3, column=7, sticky="w", pady=(6, 0))
        ttk.Label(drives, text="Drive", style="Dark.TLabel").pack(side="left", padx=(0, 4))
        for value, text in (("onset", "Onset"), ("level", "Level")):
            ttk.Radiobutton(drives, text=text, value=value, variable=self.drive, style="Dark.TRadiobutton",
                            command=lambda: setattr(a, "drive", self.drive.get())).pack(side="left", padx=(0, 6))

        ttk.Label(bar, text="Colour", style="Dark.TLabel").grid(row=0, column=3, sticky="e", padx=(16, 4))
        self.swatch = tk.Button(bar, width=3, relief="flat", bg=to_hex(a.colour),
                                activebackground=to_hex(a.colour), command=self.pick_colour)
        self.swatch.grid(row=0, column=4, sticky="w")
        self.rainbow = tk.BooleanVar(value=a.rainbow)
        ttk.Checkbutton(bar, text="Rainbow", variable=self.rainbow, style="Dark.TCheckbutton",
                        command=lambda: setattr(a, "rainbow", self.rainbow.get())
                        ).grid(row=0, column=5, sticky="w", padx=8)

        self.send = tk.BooleanVar(value=self.sender is not None)
        ttk.Checkbutton(bar, text="Send to keyboard", variable=self.send, style="Dark.TCheckbutton",
                        command=self.toggle_send,
                        state="normal" if self.sender else "disabled").grid(row=0, column=6, sticky="w", padx=8)
        self.labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Labels", variable=self.labels, style="Dark.TCheckbutton",
                        command=self.toggle_labels).grid(row=0, column=7, sticky="w")

        self.sliders = {}
        sliders = [("Range (dB)", "range", 10, 50), ("Fall (spectrum)", "fall", 0.2, 6.0),
                   ("Onset threshold dB (spectrum)", "onset_db", 1.0, 12.0),
                   ("Beat threshold (pulse/fire)", "threshold", 1.1, 3.0),
                   ("Flame height (fire)", "fire_height", 0.4, 2.5),
                   ("LED gamma (keyboard only)", "gamma", 1.0, 3.5),
                   ("LED green (keyboard only)", "led_green", 0.2, 1.0)]
        for i, (text, attr, lo, hi) in enumerate(sliders):
            ttk.Label(bar, text=text, style="Dark.TLabel").grid(row=1 + i, column=0, columnspan=2,
                                                                sticky="w", pady=(6, 0))
            fmt = ".2f" if attr == "led_green" else ".1f"
            value = ttk.Label(bar, text=f"{getattr(a, attr):{fmt}}", width=5, style="Dark.TLabel")
            value.grid(row=1 + i, column=6, sticky="w", pady=(6, 0))

            def changed(v, attr=attr, value=value, fmt=fmt):
                setattr(a, attr, float(v))
                value.configure(text=f"{float(v):{fmt}}")

            s = ttk.Scale(bar, from_=lo, to=hi, value=getattr(a, attr), command=changed)
            s.grid(row=1 + i, column=2, columnspan=4, sticky="we", pady=(6, 0))
            self.sliders[attr] = (s, value)

        self.eq_window = None
        ttk.Button(bar, text="Range / EQ...", command=self.open_eq).grid(row=2, column=7, sticky="w", pady=(6, 0))

        audio_row = 1 + len(sliders)
        ttk.Label(bar, text="Audio", style="Dark.TLabel").grid(row=audio_row, column=0, sticky="w", pady=(8, 0))
        pa = pyaudio.PyAudio()
        try:
            names = [d["name"] for d in amicis_music.capture_devices(pa)]
        finally:
            pa.terminate()
        self.audio = tk.StringVar()
        box = ttk.Combobox(bar, textvariable=self.audio, values=names, state="readonly", width=60)
        box.grid(row=audio_row, column=1, columnspan=6, sticky="we", pady=(8, 0))
        box.bind("<<ComboboxSelected>>", lambda e: self.open_audio(self.audio.get()))

        self.status = ttk.Label(bar, text="", style="Dark.TLabel")
        self.status.grid(row=audio_row + 1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        bar.columnconfigure(2, weight=1)

    # --- actions ------------------------------------------------------------

    def open_audio(self, name):
        if self.cap:
            self.cap.close()
            self.cap = None
        try:
            self.cap = amicis_music.AudioCapture(name or None)
        except SystemExit as e:  # AudioCapture reports a missing device this way
            self.render = None
            self.status_text = str(e)
            return
        self.audio.set(self.cap.name)
        self.rebuild_renderer()

    def rebuild_renderer(self):
        self.args.mode = self.mode.get()
        self.args.layout = self.layout.get()
        if self.cap:
            self.render = amicis_music.RENDERERS[self.args.mode](self.cap, self.leds, self.args)

    def pick_colour(self):
        _, hex_colour = colorchooser.askcolor(color=to_hex(self.args.colour), parent=self.root)
        if hex_colour:
            self.args.colour = amicis_music.parse_colour(hex_colour)
            self.swatch.configure(bg=hex_colour, activebackground=hex_colour)

    def set_slider(self, attr, value):
        scale, label = self.sliders[attr]
        scale.set(value)  # fires the slider's command, which updates args and the label

    def open_eq(self):
        if self.eq_window and self.eq_window.win.winfo_exists():
            self.eq_window.win.lift()
        else:
            self.eq_window = EqWindow(self)

    def toggle_send(self):
        if not self.send.get() and self.sender:
            self.sender.fade()

    def set_centre(self, event):
        u = self.unit
        self.args.centre = (round((event.x - self.margin) / u, 2), round((event.y - self.margin) / u, 2))

    def reset_centre(self, _event=None):
        self.args.centre = None

    def update_centre_marker(self):
        pulse = self.args.mode in CENTRED_MODES
        centre = amicis_music.pulse_centre(self.leds, self.args) if pulse else None
        if centre == self.centre_drawn:
            return
        if self.centre_drawn:  # undo the previous move
            dx, dy = self.centre_drawn
            for item in self.centre_items:
                self.canvas.move(item, -(self.margin + dx * self.unit), -(self.margin + dy * self.unit))
        for item in self.centre_items:
            self.canvas.itemconfigure(item, state="normal" if pulse else "hidden")
            if centre:
                self.canvas.move(item, self.margin + centre[0] * self.unit, self.margin + centre[1] * self.unit)
            self.canvas.tag_raise(item)
        self.centre_drawn = centre

    def toggle_labels(self):
        state = "normal" if self.labels.get() else "hidden"
        for item in self.label_items:
            self.canvas.itemconfigure(item, state=state)

    # --- frame loop ---------------------------------------------------------

    def tick(self):
        if self.closing:
            return
        start = time.monotonic()
        if self.render:
            colours = np.asarray(self.render(start - self.t0, start - self.prev), dtype=float)
            for i, c in enumerate(colours):
                fill = to_hex(UNLIT + c * (1 - UNLIT))
                if fill != self.last_fill[i]:
                    self.canvas.itemconfigure(self.key_items[i], fill=fill)
                    self.last_fill[i] = fill
            if self.sender and self.send.get() and self.sender.error is None:
                self.sender.submit(amicis_led.frame_from_colors(self.leds, colours, self.args.gamma,
                                                                self.args.led_green))
        self.prev = start
        self.update_centre_marker()

        self.frames += 1
        if start - self.fps_mark >= 1:
            fps = self.frames / (start - self.fps_mark)
            self.frames, self.fps_mark = 0, start
            if not self.sender:
                kb = "preview only (no keyboard)"
            elif self.sender.error:
                kb = f"keyboard error: {self.sender.error}"
            else:
                kb = "keyboard: sending" if self.send.get() else "keyboard: paused"
            audio = self.cap.name if self.cap else getattr(self, "status_text", "no audio")
            parts = [f"{fps:4.0f} fps", kb, audio]
            if self.args.mode in CENTRED_MODES:
                cx, cy = amicis_music.pulse_centre(self.leds, self.args)
                origin = "custom" if self.args.centre else "default"
                parts.insert(1, f"centre {cx:.2f},{cy:.2f} ({origin})")
            self.status.configure(text="   |   ".join(parts))

        delay = 1 / self.args.fps - (time.monotonic() - start) if self.args.fps else 0
        self.root.after(max(1, int(delay * 1000)), self.tick)

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.cap:
            self.cap.close()
        if self.sender:
            self.sender.stop()
        self.root.destroy()


def main():
    ap = amicis_music.build_parser()
    ap.description = __doc__
    ap.add_argument("--no-keyboard", action="store_true", help="preview only, do not open the keyboard")
    args = ap.parse_args()
    if args.list_audio:
        amicis_music.list_audio()
        return 0

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp on high-DPI screens
    except (AttributeError, OSError):
        pass

    kb, leds = None, amicis_led.preview_led_map()
    ifaces = [] if args.no_keyboard else amicis_led.find_interfaces()
    if ifaces:
        d = ifaces[0]
        print(f"Using {amicis_led.PIDS[d['product_id']]} (PID {d['product_id']:04X})")
        kb = amicis_led.Keyboard(d["path"])
        leds = amicis_led.build_led_map(kb.default_keymap())
    elif not args.no_keyboard:
        print("No keyboard found; preview only.")

    root = tk.Tk()
    app = App(root, args, kb, leds)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
