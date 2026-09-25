"""P145/prooflayer-rmf is a separate, standalone module with its own package
literally named `app` (same name as prooflayer-sdaia/app). If pytest collects
both in one run, whichever `app` package imports first wins the process-wide
sys.modules['app'] entry, and the other module's tests fail with a spurious
ImportError chasing the wrong `app`. Exclude P145 here so `pytest` run from
prooflayer-sdaia/ only ever collects prooflayer-sdaia's own tests; P145 has
its own test suite, run separately from P145/prooflayer-rmf/.
"""
collect_ignore = ["P145"]
