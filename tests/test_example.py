"""
Runs examples/example_adoption.py and checks its adoption numbers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import examples.example_adoption as ex  # noqa: E402

TECHNOLOGIES = {"solar_panel", "heat_pump", "electric_vehicle"}
AGENT_COUNT = 200       # set in run_example()
SNAPSHOTS = 8           # 48 steps, data collected every 6 steps


def _curves(monkeypatch):
    monkeypatch.setattr(ex, "plot_results", lambda curves: None)
    results = ex.run_example()
    return [snapshot["adoption_counts"] for snapshot in results["simulation_data"]]


@pytest.fixture
def curves(monkeypatch):
    return _curves(monkeypatch)


def test_snapshots_cover_every_technology(curves):
    assert len(curves) == SNAPSHOTS
    for counts in curves:
        assert set(counts) == TECHNOLOGIES


def test_counts_within_population(curves):
    for counts in curves:
        for count in counts.values():
            assert 0 <= count <= AGENT_COUNT


def test_adoption_never_decreases(curves):
    for tech in TECHNOLOGIES:
        series = [counts[tech] for counts in curves]
        assert series == sorted(series), f"{tech}: {series}"


def test_same_seed_same_numbers(curves, monkeypatch):
    assert _curves(monkeypatch) == curves


# adoption counts with random_seed 42, recorded with Python 3.10
EXPECTED = [
    {"solar_panel": 0, "heat_pump": 0, "electric_vehicle": 0},
    {"solar_panel": 0, "heat_pump": 0, "electric_vehicle": 0},
    {"solar_panel": 27, "heat_pump": 10, "electric_vehicle": 160},
    {"solar_panel": 67, "heat_pump": 29, "electric_vehicle": 192},
    {"solar_panel": 85, "heat_pump": 41, "electric_vehicle": 196},
    {"solar_panel": 89, "heat_pump": 48, "electric_vehicle": 200},
    {"solar_panel": 100, "heat_pump": 55, "electric_vehicle": 200},
    {"solar_panel": 105, "heat_pump": 59, "electric_vehicle": 200},
]


def test_adoption_counts_match_recorded_run(curves):
    assert curves == EXPECTED
