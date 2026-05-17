from datetime import datetime, timezone

from futures_lab.cli import _deadline_remaining_seconds, _record_deadline


def test_record_deadline_uses_wall_clock_utc() -> None:
    now = datetime(2026, 5, 17, 1, 30, tzinfo=timezone.utc)

    deadline = _record_deadline(300, current=now)

    assert deadline == datetime(2026, 5, 17, 1, 35, tzinfo=timezone.utc)


def test_deadline_remaining_goes_negative_after_wall_clock_overshoot() -> None:
    deadline = datetime(2026, 5, 17, 1, 35, tzinfo=timezone.utc)
    resumed_after_sleep = datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc)

    assert _deadline_remaining_seconds(deadline, current=resumed_after_sleep) < 0
