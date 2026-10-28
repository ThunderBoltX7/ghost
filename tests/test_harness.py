"""Tests for Ghost CLI harness and reliability modules."""

import pytest
from ghost_cli.utils import truncate_output, _count_tokens
from ghost_cli.permissions import classify_command, RISK_LOW, RISK_RISKY, RISK_CRITICAL


def test_truncate_output_basic():
    text = "line1\nline2\nline3"
    res = truncate_output(text, max_chars=100)
    assert res == text


def test_truncate_output_long():
    text = "a" * 10000
    res = truncate_output(text, max_chars=500)
    assert len(res) < 1000
    assert "truncated" in res


def test_token_counting():
    assert _count_tokens("hello world") > 0


def test_command_classification_harness():
    assert classify_command("git status") == RISK_LOW
    assert classify_command("pip install requests") == RISK_RISKY
    assert classify_command("rm -rf /") == RISK_CRITICAL
