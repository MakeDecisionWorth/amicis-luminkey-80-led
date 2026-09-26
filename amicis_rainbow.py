"""AMICIS 80HE JIS: left-to-right rainbow and LED mapping check.

Usage:
    python amicis_rainbow.py --list        # show matching HID interfaces
    python amicis_rainbow.py --map         # print slot -> key mapping
    python amicis_rainbow.py --walk        # light keys one by one (mapping check)
    python amicis_rainbow.py               # left-to-right rainbow
"""

import argparse
import colorsys
import sys
import time

import amicis_led


def rainbow(kb, leds, fps, seconds, spread, speed, gamma):
    xs = [x for _, x, _, _ in leds]
    x0, span = min(xs), (max(xs) - min(xs)) or 1

    def render(t, dt):
        # Subtracting time moves the hue bands toward +x (left to right).
        return [colorsys.hsv_to_rgb(((x - x0) / span * spread - t * speed) % 1.0, 1.0, 1.0)
                for _, x, _, _ in leds]

    amicis_led.run_frames(kb, leds, render, fps, seconds, gamma)


def walk(kb, leds, delay):
    n_slots = max(s for s, *_ in leds) + 1
    order = sorted(leds, key=lambda l: (round(l[2]), l[1]))
    for slot, _, _, label in order:
        buf = bytearray(n_slots * 3)
        buf[slot * 3:slot * 3 + 3] = b"\xff\xff\xff"
        print(f"slot {slot:3d}  {label}")
        end = time.monotonic() + delay
        while time.monotonic() < end:  # keep refreshing so direct mode stays active
            kb.send_frame(bytes(buf))
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    amicis_led.add_common_args(ap)
    ap.add_argument("--walk", action="store_true", help="light keys one by one to verify the mapping")
    ap.add_argument("--walk-delay", type=float, default=0.4)
    ap.add_argument("--spread", type=float, default=1.0, help="rainbow cycles across the board width")
    ap.add_argument("--speed", type=float, default=0.5, help="hue cycles per second")
    args = ap.parse_args()

    def effect(kb, leds):
        if args.walk:
            walk(kb, leds, args.walk_delay)
        else:
            rainbow(kb, leds, args.fps, args.seconds, args.spread, args.speed, args.gamma)

    return amicis_led.run(args, effect)


if __name__ == "__main__":
    sys.exit(main())
