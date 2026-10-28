import pytest
from pathlib import Path
import ghost_cli.utils as utils
from ghost_cli.utils import add_xp, truncate_output, get_rank


def test_truncate_output_boundaries():
    # Below boundary
    text_below = "a" * 50
    assert truncate_output(text_below, max_chars=100) == text_below

    # Exactly boundary
    text_exact = "b" * 100
    assert truncate_output(text_exact, max_chars=100) == text_exact

    # Above boundary
    text_above = "c" * 150
    truncated = truncate_output(text_above, max_chars=100)
    assert len(truncated) > 0
    assert "[truncated" in truncated
    assert truncated.startswith("c" * 50)
    assert truncated.endswith("c" * 50)


def test_add_xp_level_up(tmp_path):
    # Isolated stats file
    stats_file = tmp_path / "test_stats.json"
    utils._stats_cache = {"level": 1, "xp": 0, "edits": 0, "commands": 0}

    # Adding 50 XP at Level 1 (threshold 100): no level up
    stats = add_xp(50, reason="test", stats_file=stats_file)
    assert stats["level"] == 1
    assert stats["xp"] == 50

    # Adding 60 more XP (total 110 at Level 1, threshold 100):
    # Level increases to 2, remaining XP is 110 - 100 = 10
    stats = add_xp(60, reason="test", stats_file=stats_file)
    assert stats["level"] == 2
    assert stats["xp"] == 10

    # Multiple level jump: add 500 XP at Level 2
    # Level 2 threshold is 200: 510 - 200 = 310, Level 3
    # Level 3 threshold is 300: 310 - 300 = 10, Level 4
    stats = add_xp(500, reason="mega boost", stats_file=stats_file)
    assert stats["level"] == 4
    assert stats["xp"] == 10

    # Clean up cache
    utils._stats_cache = None


def test_get_rank():
    assert get_rank(1) == "Rookie"
    assert get_rank(5) == "Operative"
    assert get_rank(15) == "Phantom"
    assert get_rank(30) == "Wraith"
    assert get_rank(50) == "Ghost Prime"
