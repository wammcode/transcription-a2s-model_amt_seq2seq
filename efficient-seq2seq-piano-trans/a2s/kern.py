"""Structural/rhythmic checks for **kern-only scores, plus independent Verovio QA.

This is deliberately not advertised as a complete Humdrum conformance checker.
It accepts an arbitrary number of spines and tracks dynamic spine operations.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
import re


@dataclass
class Spine:
    root: int
    remaining: Fraction = Fraction(0)
    pending: bool = False


def duration(token):
    match = re.search(r"(0+|[1-9][0-9]*)(?:%([1-9][0-9]*))?(\.*)", token)
    if not match:
        raise ValueError(f"Missing explicit reciprocal duration: {token}")
    n, divisor, dots = match.groups()
    value = Fraction(4 * int(divisor or 1), int(n)) if int(n) else Fraction(4 * 2 ** len(n))
    value *= 2 - Fraction(1, 2 ** len(dots))
    return Fraction(0) if "q" in token or "Q" in token else value


def validate(text, external=False):
    spines, next_root, records, max_spines = [], 0, 0, 0
    saw_header = ended = False
    now = Fraction(0)
    for lineno, line in enumerate(text.splitlines(), 1):
        def fail(message):
            raise ValueError(f"**kern line {lineno}: {message}")
        if not line or line.startswith("!!"):
            continue
        if any(ord(c) < 32 and c != "\t" for c in line):
            fail("control character")
        cells = line.split("\t")
        if any(not c for c in cells):
            fail("empty field (use the null token '.')")
        if not saw_header:
            if any(c != "**kern" for c in cells):
                fail("expected **kern exclusive interpretations")
            spines = [Spine(i) for i in range(len(cells))]
            next_root = len(spines)
            saw_header = True
            max_spines = len(spines)
            continue
        if ended:
            fail("records after all spines terminated")
        if len(cells) != len(spines):
            fail(f"expected {len(spines)} fields, got {len(cells)}")
        if line.startswith("!"):
            if not all(c.startswith("!") for c in cells):
                fail("mixed comment/data record")
            continue
        if any(s.pending for s in spines):
            if any(c != "**kern" if s.pending else c != "*" for c, s in zip(cells, spines)):
                fail("new spines require **kern on the next record")
            for s in spines:
                s.pending = False
            continue
        if line.startswith("*"):
            if not all(c.startswith("*") for c in cells):
                fail("mixed interpretation/data record")
            manip = {"*^", "*v", "*x", "*+", "*-"}
            if not any(c in manip for c in cells):
                if any(c.startswith("**") for c in cells):
                    fail("exclusive interpretation changes are unsupported")
                continue
            if any(c not in manip | {"*"} for c in cells):
                fail("mixed tandem/spine-path interpretations")
            exchanges = [i for i, c in enumerate(cells) if c == "*x"]
            if len(exchanges) not in (0, 2):
                fail("spine exchanges require exactly two *x")
            if exchanges:
                a, b = exchanges
                spines[a], spines[b] = spines[b], spines[a]
            result, i = [], 0
            while i < len(cells):
                c, s = cells[i], spines[i]
                if c == "*v":
                    j = i
                    while j < len(cells) and cells[j] == "*v":
                        j += 1
                    group = spines[i:j]
                    if len(group) < 2 or len({s.root for s in group}) != 1:
                        fail("merge requires adjacent branches of the same staff")
                    if any(s.remaining for s in group):
                        fail("merge before notes have ended")
                    result.append(replace(s))
                    i = j
                    continue
                if c == "*-":
                    if s.remaining:
                        fail("termination before note has ended")
                else:
                    result.append(s)
                    if c == "*^":
                        result.append(replace(s))
                    elif c == "*+":
                        result.append(Spine(next_root, pending=True))
                        next_root += 1
                i += 1
            spines = result
            max_spines = max(max_spines, len(spines))
            ended = not spines
            continue
        if line.startswith("="):
            if not all(re.fullmatch(r"=[=0-9A-Za-z:|!.';\-]*", c) for c in cells):
                fail("invalid or incomplete barline record")
            continue
        if any(c.startswith(("*", "!", "=")) for c in cells):
            fail("mixed record types")
        grace = any(c != "." and ("q" in c or "Q" in c) for c in cells)
        for cell, spine in zip(cells, spines):
            if cell == ".":
                if spine.remaining == 0 and not grace:
                    fail("null token without a sustaining event")
                continue
            if spine.remaining:
                fail("new event overlaps a sustaining event in its voice")
            members = cell.split(" ")
            if any(not m for m in members):
                fail("empty chord member")
            lengths = []
            for member in members:
                pitches = re.findall(r"[a-gA-G]+|r+", member)
                if len(pitches) != 1 or len(set(pitches[0])) != 1:
                    fail(f"missing pitch/rest: {member}")
                # Kern notation plus common RDF signifiers. Unknown alphabetic words
                # are not silently accepted as musical events.
                if re.search(r"[^0-9a-gA-GrqQ#n%\.\-\[\]_(){}<>/\\;,:'\"`~^vLJKklyYzZtTmMsSxXoONiIPV+!?@&|=]", member):
                    fail(f"unsupported note signifier: {member}")
                lengths.append(duration(member))
            if len(set(lengths)) != 1:
                fail("mixed-duration chord: encode independent durations as voices")
            if len(members) > 1 and any("r" in m for m in members):
                fail("rest in a chord")
            spine.remaining = lengths[0]
        if not grace:
            if not any(s.remaining for s in spines):
                fail("data record has no rhythmic event")
            delta = min(s.remaining for s in spines if s.remaining)
            for s in spines:
                s.remaining = max(Fraction(0), s.remaining - delta)
            now += delta
        records += 1
    if not saw_header or not ended or not records:
        raise ValueError("Expected complete, nonempty **kern score with spine terminators")
    report = {"records": records, "max_spines": max_spines, "quarter_length": str(now),
              "structural_rhythm_valid": True}
    if external:
        import verovio
        toolkit = verovio.toolkit()
        toolkit.setOptions({"inputFrom": "humdrum"})
        if not toolkit.loadData(text) or toolkit.getPageCount() < 1:
            raise ValueError("Verovio rejected the score: " + toolkit.getLog())
        svg = toolkit.renderToSVG(1)
        if "<svg" not in svg:
            raise ValueError("Verovio could not render the score")
        report["verovio_rendered"] = True
    return report
