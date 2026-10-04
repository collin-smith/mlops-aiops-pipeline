"""Stage 7 monitoring, all run by hand, nothing on a schedule (D-045).

* ``datasets`` / ``inject_drift``: the CSVs Model Monitor's analyzer compares (Layer A)
* ``pipeline_health``: job history against a 2× rule (Layer B)
* ``civic_scorecard``: the companion findings as monthly series (Layer C)
* ``detect``: the detectors B and C share
"""
