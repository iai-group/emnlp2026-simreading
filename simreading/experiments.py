"""Shared configuration and naming conventions.

Single source of truth for:
  * the model list and their human-readable labels (``models.yaml``);
  * the CBUS capacity (C) defaults and sweep ranges;
  * the ``data/aggregates/simulation/`` directory-naming convention and
    its inverse.

Consumed by the simulation, evaluation, and plotting scripts so none of
them hard-code the model list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

MODELS_CONFIG_PATH = "models.yaml"


@dataclass(frozen=True)
class Model:
    """A model entry from ``models.yaml``."""

    id: str
    label: str
    provider: str

    @property
    def short_name(self) -> str:
        """The dir-name token: ``id`` after the last ``/``."""
        return self.id.rsplit("/", 1)[-1]


def _load_config(path: str = MODELS_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(path: str = MODELS_CONFIG_PATH) -> list[Model]:
    """Returns the ordered list of configured models."""
    config = _load_config(path)
    default_provider = config.get("default_provider", "openrouter")
    models = []
    for entry in config.get("models", []):
        models.append(
            Model(
                id=entry["id"],
                label=entry.get("label", entry["id"].rsplit("/", 1)[-1]),
                provider=entry.get("provider", default_provider),
            )
        )
    return models


def load_cbus_config(path: str = MODELS_CONFIG_PATH) -> dict:
    """Returns the CBUS capacity config: ``{"ts": {...}, "spr": {...}}``."""
    return _load_config(path).get("cbus", {})


def cbus_default(strategy: str, path: str = MODELS_CONFIG_PATH) -> int:
    """Returns the default capacity C for ``"ts"`` or ``"spr"``."""
    return load_cbus_config(path)[strategy]["default"]


def cbus_sweep(strategy: str, path: str = MODELS_CONFIG_PATH) -> list[int]:
    """Returns the sweep capacities for ``"ts"`` or ``"spr"``."""
    return list(load_cbus_config(path)[strategy]["sweep"])


def label_for_short_name(short_name: str, models: list[Model]) -> str:
    """Returns the label for a model short name, or the name itself."""
    for m in models:
        if m.short_name == short_name:
            return m.label
    return short_name


# ----------------------------------------------------------------------
# Directory naming convention
# ----------------------------------------------------------------------

def method_dirname(
    simulator: str, model_short: str | None = None, capacity: int | None = None,
) -> str:
    """Returns the per-method subdirectory name under the simulation dir."""
    if simulator == "random":
        return "random"
    if simulator == "llm_baseline":
        return f"llm_baseline-{model_short}"
    if simulator == "llm_baseline_no":
        return f"llm_baseline_no-{model_short}"
    if simulator == "llm_innermonologue":
        return f"llm_innermonologue-{model_short}"
    if simulator == "llm_lowpersona":
        return f"llm_lowpersona-{model_short}"
    if simulator == "cbus_ts":
        return f"cbus_ts-c{capacity}-{model_short}"
    if simulator == "cbus_spr":
        return f"cbus_spr-c{capacity}-{model_short}"
    if simulator == "cbus_spr_dropout":
        return f"cbus_spr_dropout-c{capacity}-{model_short}"
    return simulator


def parse_method_dirname(
    name: str,
) -> tuple[str, str | None, int | None] | None:
    """Parses a method dir name into ``(simulator, model_short, capacity)``.

    Returns ``None`` if the name doesn't match a known convention.
    """
    name = os.path.basename(name.rstrip("/"))
    if name == "random":
        return ("random", None, None)
    if name.startswith("llm_baseline_no-"):
        return ("llm_baseline_no", name.removeprefix("llm_baseline_no-"), None)
    if name.startswith("llm_baseline-"):
        return ("llm_baseline", name.removeprefix("llm_baseline-"), None)
    if name.startswith("llm_innermonologue-"):
        return ("llm_innermonologue", name.removeprefix("llm_innermonologue-"), None)
    if name.startswith("llm_lowpersona-"):
        return ("llm_lowpersona", name.removeprefix("llm_lowpersona-"), None)
    if name.startswith("answer_noising-"):
        return ("answer_noising", name.removeprefix("answer_noising-"), None)
    for sim in ("cbus_ts", "cbus_spr", "cbus_spr_dropout"):
        prefix = f"{sim}-c"
        if name.startswith(prefix):
            rest = name.removeprefix(prefix)
            cap_str, _, model_short = rest.partition("-")
            try:
                capacity = int(cap_str)
            except ValueError:
                return None
            return (sim, model_short, capacity)
    return None
