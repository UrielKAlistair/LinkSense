# Tests

I do not maintain this folder. Every test in it was written by Claude, on its own
initiative: I never asked for a test suite and but it apparently helps claude from breaking everything with some of its overenthusiastic diffs.

## A note from Claude

These are a consistency check, not a correctness check. They come from the same
reading of the problem that produced the code, so they can tell you it still does
what it did last week. They cannot tell you it does what you wanted, because
anywhere I misread the intent I misread it into both files at once.

Three things follow from that, and are worth knowing before trusting a green run.
Reading the code has caught more defects here than running these has. A test can
pass while proving nothing, when its fixture is built from the same assumption as
the implementation it checks. And coverage has followed what was easy to write
rather than what carries risk, so the metrics the headline numbers rest on are
among the least covered things in the repository.

There is no runner. `pytest` is not installed in the virtual environment, and the
files import through the repository root, so each is run on its own:

    PYTHONPATH=. .venv/bin/python3 scripts/tests/test_models.py
