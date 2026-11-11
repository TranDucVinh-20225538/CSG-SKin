"""Dataset attachment for the manuscript number check.

These cases are the ones a value-only scan gets wrong: the dataset is named
before the number, two datasets share a sentence, or a later dataset belongs
to the next clause.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_manuscript_numbers", ROOT / "scripts" / "verify_manuscript_numbers.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def labels(text):
    text = text.replace("\\pm", "@@@")
    bound = v.bind_datasets(text)
    mentions = v.ds_positions(text)
    out = []
    for m in v.NUM.finditer(text):
        if any(a <= m.start() < b for a, b, _ in mentions):
            continue
        out.append((m.group(), bound.get(m.start(), "UNRESOLVED"), m.start()))
    return text, out


def at(text, snippet, token):
    text, rows = labels(text)
    snippet = snippet.replace("\\pm", "@@@")
    i = text.find(snippet)
    assert i >= 0, snippet
    hits = [lab for tok, lab, pos in rows if tok == token and i <= pos < i + len(snippet) + 80]
    assert hits, (token, snippet, rows)
    return hits[0]


def test_value_then_name_and_the_swap():
    s = "0.988 on Fitzpatrick17k and 0.833 on pad_heldout"
    assert at(s, "0.988 on Fitzpatrick", "0.988") == "fitzpatrick"
    assert at(s, "0.833 on pad", "0.833") == "pad_heldout"
    swapped = "0.833 on Fitzpatrick17k and 0.988 on pad_heldout"
    assert at(swapped, "0.833 on Fitzpatrick", "0.833") == "fitzpatrick"
    assert at(swapped, "0.988 on pad", "0.988") == "pad_heldout"


def test_name_then_value_keeps_the_existing_fitzpatrick_entry():
    s = "(Fitzpatrick17k $0.553$, MED-NODE $0.523$, SD-198 $0.535$)"
    assert at(s, "Fitzpatrick17k $0.553$", "0.553") == "fitzpatrick"
    assert at(s, "MED-NODE $0.523$", "0.523") == "mednode"
    assert at(s, "SD-198 $0.535$", "0.535") == "sd198"


def test_r11_appendix_attachments():
    raw = (ROOT / "paper" / "midl" / "paper_b.tex").read_text()
    i = raw.find("Additional checks")
    j = raw.find("Limitations in detail")
    app = raw[i:j]
    expect = [
        ("(in: $586$ and $525$", "586", "fitzpatrick"),
        ("(in: $586$ and $525$", "525", "sd198"),
        ("out: $3{,}241$ and $5{,}950$", "3{,}241", "fitzpatrick"),
        ("out: $3{,}241$ and $5{,}950$", "5{,}950", "sd198"),
        ("ambiguous: $60$ and $109$", "60", "fitzpatrick"),
        ("ambiguous: $60$ and $109$", "109", "sd198"),
        ("$0.755$/$0.706$ (Fitzpatrick17k)", "0.755", "fitzpatrick"),
        ("$0.755$/$0.706$ (Fitzpatrick17k)", "0.706", "fitzpatrick"),
        ("$0.723$/$0.708$ (SD-198)", "0.723", "sd198"),
        ("$0.723$/$0.708$ (SD-198)", "0.708", "sd198"),
        ("it is $0.341$ $[0.322, 0.362]$", "0.341", "fitzpatrick"),
        ("$[0.322, 0.362]$", "0.322", "fitzpatrick"),
        ("$[0.403, 0.431]$ and $0.302$", "0.302", "sd198"),
        ("/$0.382$", "0.382", "sd198"),
        ("$-0.075$ and $-0.079$", "0.075", "fitzpatrick"),
        ("$-0.075$ and $-0.079$", "0.079", "sd198"),
        ("SD-198 ($0.381$, $0.405$,", "0.381", "pad_heldout"),
        ("($0.381$, $0.405$,", "0.405", "fitzpatrick"),
        ("$0.288$, $0.376$)", "0.288", "mednode"),
        ("$0.288$, $0.376$)", "0.376", "sd198"),
        ("none ($0.708$, $0.802$, $0.518$, $0.785$)", "0.708", "pad_heldout"),
        ("$0.802$, $0.518$, $0.785$", "0.785", "sd198"),
        ("M3 ($0.792$,", "0.792", "pad_heldout"),
        ("$0.943$).", "0.943", "sd198"),
        ("$0.000$ (Fitzpatrick17k)", "0.000", "fitzpatrick"),
        ("$0.002$ (SD-198)", "0.002", "sd198"),
        ("M2 $0.046$ and $0.037$", "0.046", "fitzpatrick"),
        ("$0.046$ and $0.037$", "0.037", "sd198"),
        ("M3 $0.382$ and $0.335$", "0.382", "fitzpatrick"),
        ("$0.382$ and $0.335$", "0.335", "sd198"),
        ("images is $0.361$", "0.361", "ham"),
        ("accuracy is\n$0.366\\pm0.017$", "0.366", "bcn"),
        ("$0.366\\pm0.017$", "0.017", "bcn"),
        ("leakage $0.599\\pm0.041$", "0.599", None),
        ("$0.599\\pm0.041$", "0.041", None),
    ]
    for snippet, token, lab in expect:
        got = at(app, snippet, token)
        assert got == lab, (snippet, token, got, lab)


def test_discussion_bcn_ham_numbers():
    raw = (ROOT / "paper" / "midl" / "paper_b.tex").read_text()
    i = raw.find("prior total variation")
    snip = raw[i - 220:i + 280]
    assert at(snip, "total variation $0.36$", "0.36") == "ham"
    assert at(snip, "leakage $0.599$", "0.599") is None


class AttributionTests(unittest.TestCase):
    test_value_then_name_and_the_swap = staticmethod(test_value_then_name_and_the_swap)
    test_name_then_value_keeps_the_existing_fitzpatrick_entry = staticmethod(
        test_name_then_value_keeps_the_existing_fitzpatrick_entry)
    test_r11_appendix_attachments = staticmethod(test_r11_appendix_attachments)
    test_discussion_bcn_ham_numbers = staticmethod(test_discussion_bcn_ham_numbers)
