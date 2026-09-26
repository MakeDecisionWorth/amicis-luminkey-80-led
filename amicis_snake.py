"""AMICIS 80HE JIS: self-playing snake on the per-key LEDs.

The keys form a graph (left/right within a row, up/down to the nearest key in
the next row), so the snake follows the real staggered layout. It takes the
shortest path to the food only if it can still reach its own tail afterwards;
otherwise it picks the move that leaves it the most room. When it is stuck it
blinks and a new game starts.

Usage:
    python amicis_snake.py                          # until Ctrl+C
    python amicis_snake.py --speed 12 --colour "#00C8FF"
"""

import argparse
import math
import random
import sys
from collections import deque

import amicis_led

GROW = 2             # segments added per food
DEATH_BLINKS = 3
STALL_FACTOR = 3     # after this many board-sizes of moves without food, go for it anyway


def hex_colour(text):
    value = text.lstrip("#")
    try:
        if len(value) != 6:
            raise ValueError
        return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {text!r}") from None


def key_graph(leds):
    """Adjacency over led indices: row neighbours plus the nearest key above/below."""
    ys = [y for _, _, y, _ in leds]
    rows = sorted(y for y in set(ys) if ys.count(y) > 1)      # the tall Enter is not a row
    members = {r: [i for i, y in enumerate(ys) if abs(y - r) <= 0.5] for r in rows}  # Enter joins two
    adj = {i: set() for i in range(len(leds))}

    def link(a, b):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    for r in rows:
        row = sorted(members[r], key=lambda i: leds[i][1])
        for a, b in zip(row, row[1:]):
            link(a, b)
    for upper, lower in zip(rows, rows[1:]):
        for src, dst in ((upper, lower), (lower, upper)):
            for a in members[src]:
                b = min(members[dst], key=lambda i: abs(leds[i][1] - leds[a][1]))
                if abs(leds[b][1] - leds[a][1]) <= 1.0:
                    link(a, b)
    return {i: sorted(n) for i, n in adj.items()}


class Snake:
    def __init__(self, adj, rng):
        self.adj = adj
        self.rng = rng
        self.reset()

    def reset(self):
        self.body = [self.rng.randrange(len(self.adj))]
        self.grow = GROW
        self.hungry = 0
        self.place_food()

    def place_food(self):
        free = [i for i in self.adj if i not in self.body]
        self.food = self.rng.choice(free) if free else None

    def path(self, start, goal, blocked):
        """Shortest path start -> goal (excluding start), avoiding blocked keys."""
        prev = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == goal:
                out = []
                while node != start:
                    out.append(node)
                    node = prev[node]
                return out[::-1]
            for n in self.adj[node]:
                if n not in prev and n not in blocked:
                    prev[n] = node
                    queue.append(n)
        return None

    def room(self, start, blocked):
        seen, stack = {start}, [start]
        while stack:
            for n in self.adj[stack.pop()]:
                if n not in seen and n not in blocked:
                    seen.add(n)
                    stack.append(n)
        return len(seen)

    def moved(self, body, grow, steps):
        """Body after following steps (grow = segments still to add)."""
        body = list(body)
        for s in steps:
            body.insert(0, s)
            if grow:
                grow -= 1
            else:
                body.pop()
        return body

    def can_reach_tail(self, body):
        return len(body) == 1 or self.path(body[0], body[-1], set(body[1:-1])) is not None

    def choose(self):
        head = self.body[0]
        blocked = set(self.body if self.grow else self.body[:-1])   # the tail moves unless growing
        route = self.path(head, self.food, blocked)
        if route:
            # Circling its tail forever is safe but dull: past the stall limit, risk it.
            if self.hungry > STALL_FACTOR * len(self.adj):
                return route[0]
            after = self.moved(self.body, self.grow + GROW, route)
            if self.can_reach_tail(after) or len(after) >= len(self.adj):
                return route[0]
        # No safe route to the food: take the move that keeps the most room,
        # preferring moves from which the tail is still reachable.
        best, best_score = None, None
        for n in self.adj[head]:
            if n in blocked:
                continue
            after = self.moved(self.body, self.grow, [n])
            score = (self.can_reach_tail(after), self.room(n, set(after[1:-1])))
            if best_score is None or score > best_score:
                best, best_score = n, score
        return best

    def step(self):
        """Advance one move; returns 'moved', 'ate', 'dead' or 'won'."""
        nxt = self.choose()
        if nxt is None:
            return "dead"
        self.body.insert(0, nxt)
        if self.grow:
            self.grow -= 1
        else:
            self.body.pop()
        self.hungry += 1
        if nxt == self.food:
            self.hungry = 0
            self.grow += GROW
            self.place_food()
            return "won" if self.food is None else "ate"
        return "moved"


def snake_renderer(leds, args):
    game = Snake(key_graph(leds), random.Random(args.seed))
    head_colour = tuple(0.5 + 0.5 * c for c in args.colour)
    state = {"next": 0.0, "phase": "play", "since": 0.0, "flash": -1.0}

    def render(t, dt):
        phase = state["phase"]
        if phase == "play" and t >= state["next"]:
            state["next"] = t + 1 / args.speed
            result = game.step()
            if result == "ate":
                state["flash"] = t
            elif result in ("dead", "won"):
                state["phase"], state["since"] = "over", t
        elif phase == "over" and t - state["since"] > DEATH_BLINKS * 0.4 + 0.4:
            game.reset()
            state["phase"], state["next"] = "play", t + 0.3

        out = [(0.0, 0.0, 0.0)] * len(leds)
        if state["phase"] == "over":
            age = t - state["since"]
            if age < DEATH_BLINKS * 0.4 and int(age / 0.2) % 2:
                return out                                    # blink off
            fade = max(0.0, 1 - max(0.0, age - DEATH_BLINKS * 0.4) / 0.4)
        else:
            fade = 1.0
        if game.food is not None and state["phase"] == "play":
            pulse = 0.55 + 0.45 * math.sin(t * 2 * math.pi * 2)
            out[game.food] = tuple(c * pulse for c in args.food)
        boost = max(0.0, 1 - (t - state["flash"]) / 0.15)   # brief flash after eating
        n = len(game.body)
        for i, key in enumerate(game.body):
            if i == 0:
                colour = head_colour
            else:
                level = 0.25 + 0.75 * (1 - i / n)
                colour = tuple(c * level for c in args.colour)
            colour = tuple(c + (1 - c) * 0.4 * boost for c in colour)
            out[key] = tuple(c * fade for c in colour)
        return out

    return render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    amicis_led.add_common_args(ap, fps=30, seconds=0, gamma=2.2, green=0.55)
    ap.add_argument("--speed", type=float, default=8, help="moves per second")
    ap.add_argument("--colour", "--color", type=hex_colour, default="#8000FF", help="snake colour")
    ap.add_argument("--food", type=hex_colour, default="#FF3060", help="food colour")
    ap.add_argument("--seed", type=int, help="random seed (repeatable games)")
    args = ap.parse_args()

    def effect(kb, leds):
        amicis_led.run_frames(kb, leds, snake_renderer(leds, args), args.fps, args.seconds,
                              args.gamma, args.led_green)

    return amicis_led.run(args, effect)


if __name__ == "__main__":
    sys.exit(main())
