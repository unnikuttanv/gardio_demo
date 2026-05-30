"""Synthetic multi-user, multi-day monitoring cohort for the full demo.

Five patients, each with a distinct clinical story that produces an interesting
multi-day trend. For each (user, day) we generate a synthetic ECG with that day's
target heart rate and variability, run the deterministic DSP pipeline to get a
FeatureSet, then ask the LLM for one trend-aware weekly summary per user.

Deterministic given the profiles (seeded per user/day), so the demo is repeatable.

CLI:
    python -m ecg_llm.cohort            # generate + save to data/cohort/cohort.json
    python -m ecg_llm.cohort --days 7   # custom window length
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from ecg_llm.config import Settings, get_settings
from ecg_llm.features import build_features
from ecg_llm.ingest import generate_synthetic_ecg
from ecg_llm.report.trends import generate_weekly_summary
from ecg_llm.schemas import Cohort, DailyRecord, UserWeek

DEFAULT_COHORT_PATH = Path("data/cohort/cohort.json")


@dataclass
class Profile:
    """A patient persona and how their daily HR / HRV evolve over the window."""

    user_id: str
    name: str
    description: str
    # day_index, n_days -> target mean HR (bpm)
    hr_fn: Callable[[int, int], float]
    # day_index, n_days -> beat-to-beat HR std (bpm); larger => higher HRV
    hrv_fn: Callable[[int, int], float]


def _lerp(a: float, b: float, day: int, n_days: int) -> float:
    if n_days <= 1:
        return a
    return a + (b - a) * (day / (n_days - 1))


# Five distinct stories. Trends are linear interpolations across the window.
PROFILES: list[Profile] = [
    Profile(
        "P001", "Ava Reyes", "healthy adult, stable",
        hr_fn=lambda d, n: 70.0 + (1.5 if d % 2 else -1.5),
        hrv_fn=lambda d, n: 5.0,
    ),
    Profile(
        "P002", "Ben Okafor", "developing tachycardia (rising HR, falling HRV)",
        hr_fn=lambda d, n: _lerp(78.0, 112.0, d, n),
        hrv_fn=lambda d, n: _lerp(5.0, 2.0, d, n),
    ),
    Profile(
        "P003", "Carla Nguyen", "endurance athlete, resting bradycardia",
        hr_fn=lambda d, n: 50.0 + (1.0 if d % 2 else -1.0),
        hrv_fn=lambda d, n: 9.0,
    ),
    Profile(
        "P004", "Dmitri Sokolov", "recovering patient (HR normalizing, HRV improving)",
        hr_fn=lambda d, n: _lerp(104.0, 74.0, d, n),
        hrv_fn=lambda d, n: _lerp(2.5, 6.0, d, n),
    ),
    Profile(
        "P005", "Elena Costa", "rising autonomic stress (stable HR, declining HRV)",
        hr_fn=lambda d, n: 80.0 + (1.0 if d % 2 else -1.0),
        hrv_fn=lambda d, n: _lerp(7.0, 2.0, d, n),
    ),
]


def _generate_user(profile: Profile, n_days: int, end_date: date, settings: Settings) -> UserWeek:
    start_date = end_date - timedelta(days=n_days - 1)
    daily: list[DailyRecord] = []

    for day in range(n_days):
        target_hr = profile.hr_fn(day, n_days)
        hr_std = profile.hrv_fn(day, n_days)
        # Stable per-user/day seed for reproducibility.
        seed = (abs(hash(profile.user_id)) % 10_000) * 100 + day
        signal = generate_synthetic_ecg(
            duration_s=30.0,
            mean_hr_bpm=target_hr,
            hr_std_bpm=hr_std,
            seed=seed,
        )
        features = build_features(signal)
        rec_date = (start_date + timedelta(days=day)).isoformat()
        daily.append(DailyRecord(day_index=day, date=rec_date, features=features))

    summary, model, is_mock = generate_weekly_summary(
        profile.name, profile.description, daily, settings
    )
    return UserWeek(
        user_id=profile.user_id,
        name=profile.name,
        profile=profile.description,
        daily=daily,
        weekly_summary=summary,
        summary_model=model,
        summary_is_mock=is_mock,
    )


def generate_cohort(
    n_users: int = 5,
    n_days: int = 7,
    end_date: date | None = None,
    settings: Settings | None = None,
) -> Cohort:
    """Generate the full cohort: per-day features + one weekly LLM summary per user.

    Args:
        n_users: Number of profiles to use (capped at len(PROFILES)).
        n_days: Days in the monitoring window.
        end_date: Last day of the window (defaults to today).
        settings: Optional settings override.

    Returns:
        A populated `Cohort`.
    """
    settings = settings or get_settings()
    end_date = end_date or date.today()
    profiles = PROFILES[: min(n_users, len(PROFILES))]

    users = [_generate_user(p, n_days, end_date, settings) for p in profiles]
    return Cohort(
        generated_at=date.today().isoformat(),
        n_days=n_days,
        users=users,
    )


def save_cohort(cohort: Cohort, path: Path = DEFAULT_COHORT_PATH) -> Path:
    """Write the cohort to JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cohort.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_cohort(path: Path = DEFAULT_COHORT_PATH) -> Cohort:
    """Load a previously saved cohort from JSON."""
    return Cohort.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate the demo monitoring cohort.")
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", default=str(DEFAULT_COHORT_PATH))
    args = parser.parse_args()

    settings = get_settings()
    print(
        f"Generating {min(args.users, len(PROFILES))} users x {args.days} days "
        f"(LLM: {'mock' if settings.use_mock_llm else settings.llm_model})..."
    )
    cohort = generate_cohort(n_users=args.users, n_days=args.days, settings=settings)
    out = save_cohort(cohort, Path(args.out))
    live = sum(0 if u.summary_is_mock else 1 for u in cohort.users)
    print(f"Saved {out}  |  {len(cohort.users)} users, {live} live LLM summaries")


if __name__ == "__main__":
    _main()
