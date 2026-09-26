"""AMICIS 80HE JIS: shared protocol and layout for driving the per-key LEDs.

Uses the firmware's undocumented direct-LED command (0x55 0xDD), which writes
into a RAM buffer only (no flash erase/program, unlike the configurator's 0x0B).
While frames keep arriving the firmware overrides the current effect; if they
stop, it falls back to the normal effect after a short timeout (about 3 s).
No command ends direct mode early, so on exit we fade to black and let the
timeout run out on a dark frame instead of a frozen one.

LED buffer slot == keymap slot. The slot -> key mapping is built at runtime by
reading the default keymap (0x55 0x07) and matching HID usages against the
AMICIS_K80_JIS layout coordinates, the same way the web configurator does.
"""

import time

import hid

VID = 0x19F5
PIDS = {
    0xFC68: "80-JIS (Yuki Sakuna)",
    0xFC6A: "80-JIS (Aoi Sakura)",
    0xFB34: "80-JIS (Kagura Nana)",
    0xFB61: "80-JIS (Emilia)",
    0xFB62: "80-JIS (Ram&Rem)",
}
# Vendor channel: the configurator accepts either collection and prefers FF00/01.
VENDOR_COLLECTIONS = [(0xFF00, 0x01), (0x0001, 0x00)]

CMD_GROUP = 0x55
CMD_READ_DEFAULT_KEYMAP = 0x07
CMD_DIRECT_LED = 0xDD

MAX_CHUNK = 56          # payload bytes per packet
LED_SLOTS = 100         # firmware renders slots 0..99 in direct mode
DIRECT_BUF_LIMIT = 0x180

FN_CODE = 255
MODIFIER_BITS = {0x01: 224, 0x02: 225, 0x04: 226, 0x08: 227,
                 0x10: 228, 0x20: 229, 0x40: 230, 0x80: 231}

# HID usage -> (x centre, y centre, label), from the configurator's AMICIS_K80_JIS layout.
LAYOUT = {
    41: (0.5, 0.5, "Esc"), 58: (2.5, 0.5, "F1"), 59: (3.5, 0.5, "F2"),
    60: (4.5, 0.5, "F3"), 61: (5.5, 0.5, "F4"), 62: (7, 0.5, "F5"),
    63: (8, 0.5, "F6"), 64: (9, 0.5, "F7"), 65: (10, 0.5, "F8"),
    66: (11.5, 0.5, "F9"), 67: (12.5, 0.5, "F10"), 68: (13.5, 0.5, "F11"),
    69: (14.5, 0.5, "F12"), 70: (15.8, 0.5, "PtSc"), 71: (16.8, 0.5, "ScLk"),
    72: (17.8, 0.5, "Pause"),
    53: (0.5, 1.7, "E/J"), 30: (1.5, 1.7, "1"), 31: (2.5, 1.7, "2"),
    32: (3.5, 1.7, "3"), 33: (4.5, 1.7, "4"), 34: (5.5, 1.7, "5"),
    35: (6.5, 1.7, "6"), 36: (7.5, 1.7, "7"), 37: (8.5, 1.7, "8"),
    38: (9.5, 1.7, "9"), 39: (10.5, 1.7, "0"), 45: (11.5, 1.7, "-"),
    46: (12.5, 1.7, "^"), 137: (13.5, 1.7, "Yen"), 42: (14.5, 1.7, "Bksp"),
    73: (15.8, 1.7, "Ins"), 74: (16.8, 1.7, "Home"), 75: (17.8, 1.7, "PgUp"),
    43: (0.75, 2.7, "Tab"), 20: (2, 2.7, "Q"), 26: (3, 2.7, "W"),
    8: (4, 2.7, "E"), 21: (5, 2.7, "R"), 23: (6, 2.7, "T"),
    28: (7, 2.7, "Y"), 24: (8, 2.7, "U"), 12: (9, 2.7, "I"),
    18: (10, 2.7, "O"), 19: (11, 2.7, "P"), 47: (12, 2.7, "@"),
    48: (13, 2.7, "["), 40: (14.25, 3.2, "Enter"), 76: (15.8, 2.7, "Del"),
    77: (16.8, 2.7, "End"), 78: (17.8, 2.7, "PgDn"),
    57: (0.875, 3.7, "Caps"), 4: (2.25, 3.7, "A"), 22: (3.25, 3.7, "S"),
    7: (4.25, 3.7, "D"), 9: (5.25, 3.7, "F"), 10: (6.25, 3.7, "G"),
    11: (7.25, 3.7, "H"), 13: (8.25, 3.7, "J"), 14: (9.25, 3.7, "K"),
    15: (10.25, 3.7, "L"), 51: (11.25, 3.7, ";"), 52: (12.25, 3.7, ":"),
    50: (13.25, 3.7, "]"),
    225: (1.125, 4.7, "LShift"), 29: (2.75, 4.7, "Z"), 27: (3.75, 4.7, "X"),
    6: (4.75, 4.7, "C"), 25: (5.75, 4.7, "V"), 5: (6.75, 4.7, "B"),
    17: (7.75, 4.7, "N"), 16: (8.75, 4.7, "M"), 54: (9.75, 4.7, ","),
    55: (10.75, 4.7, "."), 56: (11.75, 4.7, "/"), 135: (12.75, 4.7, "Ro"),
    229: (14.125, 4.7, "RShift"), 82: (16.8, 4.7, "Up"),
    224: (0.625, 5.7, "LCtrl"), 227: (1.875, 5.7, "Win"), 226: (3.125, 5.7, "LAlt"),
    139: (4.25, 5.7, "Muhenkan"), 44: (7, 5.7, "Space"), 138: (9.75, 5.7, "Henkan"),
    136: (10.875, 5.7, "Kana"), 230: (12.125, 5.7, "RAlt"), FN_CODE: (13.25, 5.7, "Fn"),
    228: (14.375, 5.7, "RCtrl"), 80: (15.8, 5.7, "Left"), 81: (16.8, 5.7, "Down"),
    79: (17.8, 5.7, "Right"),
}

# Key size in layout units (w, h) where it is not 1x1; only needed for drawing.
KEY_SIZE = {
    43: (1.5, 1), 40: (1.5, 2), 57: (1.75, 1), 225: (2.25, 1), 229: (1.75, 1),
    224: (1.25, 1), 227: (1.25, 1), 226: (1.25, 1), 44: (4.5, 1), 136: (1.25, 1),
    230: (1.25, 1), 228: (1.25, 1),
}
_USAGE_BY_LABEL = {label: usage for usage, (_, _, label) in LAYOUT.items()}


def key_size(label):
    return KEY_SIZE.get(_USAGE_BY_LABEL.get(label), (1, 1))


def preview_led_map():
    """LED map for drawing without a keyboard: every layout key, with made-up slots."""
    return [(i, x, y, label) for i, (x, y, label) in enumerate(LAYOUT.values())]


def build_packet(cmd, addr=0, data=b"", length=None):
    """[55 cmd 00 chk len addrLo addrHi 00 data...] padded to 64 bytes."""
    length = len(data) if length is None else length
    body = bytes([length, addr & 0xFF, (addr >> 8) & 0xFF, 0]) + bytes(data)
    checksum = sum(body) & 0xFF
    return (bytes([CMD_GROUP, cmd, 0x00, checksum]) + body).ljust(64, b"\0")


def is_async_report(r):
    # Unsolicited reports (calibration/profile/reset/mac mode, wireless/battery).
    return r[0] in (0xA0, 0xA1, 0xA2, 0xA3) or (r[0] == 0xFA and r[1] == 0xFB)


class Keyboard:
    def __init__(self, path):
        self.dev = hid.device()
        self.dev.open_path(path)
        self.last_frame = b""

    def close(self):
        self.dev.close()

    def transact(self, packet, timeout_ms=500):
        # Leading 0x00 is the report ID; the device uses unnumbered reports.
        self.dev.write(b"\x00" + packet)
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise TimeoutError(f"no response to cmd 0x{packet[1]:02X}")
            r = self.dev.read(64, remaining)
            # Replies echo the header ([1] cmd, [4] len, [5..6] address). Input reports reach
            # every open handle, so another program talking to the keyboard shows up here too.
            if r and not is_async_report(r) and r[1] == packet[1] and r[5:7] == list(packet[5:7]):
                return bytes(r)

    def read_block(self, cmd, start, size):
        out = bytearray()
        for off in range(start, start + size, MAX_CHUNK):
            n = min(MAX_CHUNK, start + size - off)
            out += self.transact(build_packet(cmd, off, length=n))[8:8 + n]
        return bytes(out)

    def default_keymap(self, slots=LED_SLOTS):
        raw = self.read_block(CMD_READ_DEFAULT_KEYMAP, 0, slots * 3)  # profile 0, layer 0
        return [tuple(raw[i * 3:i * 3 + 3]) for i in range(slots)]

    def send_frame(self, rgb):
        """rgb: bytes of length 3*n (n <= 100), slot order."""
        assert len(rgb) <= DIRECT_BUF_LIMIT
        for off in range(0, len(rgb), MAX_CHUNK):
            self.transact(build_packet(CMD_DIRECT_LED, off, rgb[off:off + MAX_CHUNK]))
        self.last_frame = rgb

    def fade_out(self, seconds, fps=30):
        """Fade the last frame to black and leave a black frame in the buffer."""
        if not self.last_frame:
            return
        start = self.last_frame
        steps = int(seconds * fps)
        try:
            for i in range(steps - 1, 0, -1):
                self.send_frame(bytes(v * i // steps for v in start))
                time.sleep(1 / fps)
        finally:
            self.send_frame(bytes(len(start)))


def keymap_entry_to_usage(entry):
    t, c1, c2 = entry
    if t == 0x10:
        if c1 == 0 and c2:
            return c2
        if c2 == 0 and c1 in MODIFIER_BITS:
            return MODIFIER_BITS[c1]
    if t == 0xF0 and c1 == 0xFF:
        return FN_CODE
    return None


def build_led_map(keymap):
    """Return [(slot, x, y, label)] for every slot that maps to a layout key."""
    leds, used = [], set()
    for slot, entry in enumerate(keymap):
        usage = keymap_entry_to_usage(entry)
        if usage in LAYOUT and usage not in used:
            used.add(usage)
            x, y, label = LAYOUT[usage]
            leds.append((slot, x, y, label))
    return leds


def find_interfaces():
    found = [d for d in hid.enumerate(VID, 0)
             if d["product_id"] in PIDS
             and (d["usage_page"], d["usage"]) in VENDOR_COLLECTIONS]
    return sorted(found, key=lambda d: VENDOR_COLLECTIONS.index((d["usage_page"], d["usage"])))


def frame_from_colors(leds, colors, gamma=1.0, green=1.0):
    """colors[i] is (r, g, b) in 0..1 for leds[i]; returns the slot-ordered frame.

    The firmware drives the LEDs' PWM linearly (no gamma), so dim values look far
    brighter than on a monitor. gamma is applied to each LED's intensity (its max
    channel) only, keeping the channel ratios: contrast follows the monitor while
    a colour like #8000FF stays the same hue as the keyboard's own effects.

    green scales the green channel: the LEDs' green is much stronger than a
    monitor's, which turns oranges and yellows greenish. Colours without green
    (e.g. #8000FF) are unaffected.
    """
    buf = bytearray((max(s for s, *_ in leds) + 1) * 3)
    for (slot, *_), rgb in zip(leds, colors):
        rgb = [min(max(v, 0.0), 1.0) for v in rgb]
        peak = max(rgb)
        scale = peak ** (gamma - 1) if peak > 0 else 0.0
        balance = (1.0, green, 1.0)
        buf[slot * 3:slot * 3 + 3] = bytes(round(v * scale * b * 255) for v, b in zip(rgb, balance))
    return bytes(buf)


def run_frames(kb, leds, render, fps, seconds, gamma=1.0, green=1.0):
    """Call render(t, dt) -> colors each frame and send it; prints the achieved fps."""
    frame_period = 1 / fps if fps else 0
    t0 = prev = time.monotonic()
    frames, last_report = 0, t0
    while seconds <= 0 or time.monotonic() - t0 < seconds:
        start = time.monotonic()
        kb.send_frame(frame_from_colors(leds, render(start - t0, start - prev), gamma, green))
        prev = start
        frames += 1
        now = time.monotonic()
        if now - last_report >= 1:
            print(f"\r{frames / (now - last_report):5.1f} fps", end="", flush=True)
            frames, last_report = 0, now
        sleep = frame_period - (now - start)
        if sleep > 0:
            time.sleep(sleep)
    print()


def add_common_args(ap, fps=30, seconds=20, gamma=1.0, green=1.0):
    ap.add_argument("--list", action="store_true", help="list matching HID interfaces and exit")
    ap.add_argument("--map", action="store_true", help="print the slot -> key mapping and exit")
    ap.add_argument("--fps", type=float, default=fps, help="target frame rate (0 = as fast as possible)")
    ap.add_argument("--seconds", type=float, default=seconds, help="effect duration (0 = until Ctrl+C)")
    ap.add_argument("--fade", type=float, default=0.5, help="fade-to-black time on exit (s)")
    ap.add_argument("--gamma", type=float, default=gamma,
                    help=f"LED gamma; ~2.2 matches a monitor's contrast (default {gamma})")
    ap.add_argument("--led-green", type=float, default=green,
                    help=f"LED green channel scale; <1 stops oranges/yellows looking green (default {green})")


def run(args, effect):
    """Handle --list/--map, open the keyboard, run effect(kb, leds), then fade out."""
    ifaces = find_interfaces()
    if args.list or not ifaces:
        if not ifaces:
            print("No AMICIS 80HE vendor interface found (VID 19F5, usage FF00/01 or 0001/00).")
        for d in ifaces:
            print(f"{d['product_id']:04X} {PIDS[d['product_id']]}  {d['product_string']}  {d['path']!r}")
        return 0 if ifaces else 1

    d = ifaces[0]
    print(f"Using {PIDS[d['product_id']]} (PID {d['product_id']:04X})")
    kb = Keyboard(d["path"])
    try:
        leds = build_led_map(kb.default_keymap())
        print(f"Mapped {len(leds)} LEDs")
        if args.map:
            for slot, x, y, label in sorted(leds, key=lambda l: (round(l[2]), l[1])):
                print(f"slot {slot:3d}  x={x:6.3f} y={y:4.1f}  {label}")
            return 0
        effect(kb, leds)
    except KeyboardInterrupt:
        print()
    finally:
        try:
            kb.fade_out(args.fade)
        except KeyboardInterrupt:
            kb.fade_out(0)  # second Ctrl+C: skip the fade, still go dark
        finally:
            kb.close()
    print("Stopped; LEDs stay dark until the keyboard resumes its own effect (~3 s).")
    return 0
