"""Shared configuration for the four revolute flap joints on local boxes."""

from __future__ import annotations

from dataclasses import dataclass
import os


# Code-level defaults. Static friction must be greater than or equal to
# dynamic friction for the PhysX joint-axis friction model.
DEFAULT_FLAP_STATIC_FRICTION = 0.45
DEFAULT_FLAP_DYNAMIC_FRICTION = 0.32
DEFAULT_FLAP_STATIC_FRICTION_RANGE = (0.25, 0.75)
DEFAULT_FLAP_DYNAMIC_FRICTION_RANGE = (0.15, 0.50)


@dataclass(frozen=True)
class FlapFrictionSettings:
    static: float
    dynamic: float
    randomize: bool
    static_range: tuple[float, float]
    dynamic_range: tuple[float, float]


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    return fallback if value is None else float(value)


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got '{value}'.")


def _env_range(name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None:
        return fallback
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 'MIN,MAX', got '{value}'.")
    return (float(parts[0]), float(parts[1]))


def resolve_flap_friction_settings(
    *,
    static: float | None = None,
    dynamic: float | None = None,
    randomize: bool | None = None,
    static_range: tuple[float, float] | None = None,
    dynamic_range: tuple[float, float] | None = None,
    randomize_default: bool = False,
) -> FlapFrictionSettings:
    """Resolve CLI, environment, and code defaults, then validate them."""
    static_value = (
        _env_float("KUAVO_FLAP_STATIC_FRICTION", DEFAULT_FLAP_STATIC_FRICTION)
        if static is None
        else float(static)
    )
    dynamic_value = (
        _env_float("KUAVO_FLAP_DYNAMIC_FRICTION", DEFAULT_FLAP_DYNAMIC_FRICTION)
        if dynamic is None
        else float(dynamic)
    )
    randomize_value = (
        _env_bool("KUAVO_RANDOMIZE_FLAP_FRICTION", randomize_default)
        if randomize is None
        else bool(randomize)
    )
    static_range_value = (
        _env_range("KUAVO_FLAP_STATIC_FRICTION_RANGE", DEFAULT_FLAP_STATIC_FRICTION_RANGE)
        if static_range is None
        else tuple(float(value) for value in static_range)
    )
    dynamic_range_value = (
        _env_range("KUAVO_FLAP_DYNAMIC_FRICTION_RANGE", DEFAULT_FLAP_DYNAMIC_FRICTION_RANGE)
        if dynamic_range is None
        else tuple(float(value) for value in dynamic_range)
    )

    if static_value < 0.0 or dynamic_value < 0.0:
        raise ValueError("Flap joint friction values cannot be negative.")
    if dynamic_value > static_value:
        raise ValueError("Flap dynamic friction cannot exceed static friction.")
    for label, value_range in (
        ("static", static_range_value),
        ("dynamic", dynamic_range_value),
    ):
        if value_range[0] < 0.0 or value_range[1] < 0.0:
            raise ValueError(f"Flap {label} friction range cannot contain negative values.")
        if value_range[0] > value_range[1]:
            raise ValueError(f"Flap {label} friction range MIN must not exceed MAX.")
    if dynamic_range_value[0] > static_range_value[1]:
        raise ValueError(
            "Flap friction ranges never satisfy dynamic <= static; lower the dynamic range."
        )

    return FlapFrictionSettings(
        static=static_value,
        dynamic=dynamic_value,
        randomize=randomize_value,
        static_range=static_range_value,
        dynamic_range=dynamic_range_value,
    )


def export_flap_friction_environment(settings: FlapFrictionSettings) -> None:
    """Pass launcher settings through the delayed manager_env import."""
    os.environ["KUAVO_FLAP_STATIC_FRICTION"] = str(settings.static)
    os.environ["KUAVO_FLAP_DYNAMIC_FRICTION"] = str(settings.dynamic)
    os.environ["KUAVO_RANDOMIZE_FLAP_FRICTION"] = "1" if settings.randomize else "0"
    os.environ["KUAVO_FLAP_STATIC_FRICTION_RANGE"] = ",".join(map(str, settings.static_range))
    os.environ["KUAVO_FLAP_DYNAMIC_FRICTION_RANGE"] = ",".join(map(str, settings.dynamic_range))
