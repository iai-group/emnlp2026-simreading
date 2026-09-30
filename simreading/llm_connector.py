"""Unified LLM access via litellm with YAML-based configuration.

Provides ``LLMConnector``, a thin wrapper around ``litellm.completion``
that loads provider credentials and generation defaults from a single
YAML config file (``llm_config.yaml``).
"""

import os
import time
from typing import Optional

import litellm
import yaml


_DEFAULT_CONFIG_PATH = "llm_config.yaml"
_MAX_LLM_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 5
_LLM_TIMEOUT_SECONDS = 120


class LLMConnector:
    """Sends prompts to an LLM provider configured via YAML.

    The connector reads ``llm_config.yaml`` once on construction and
    resolves the provider, model, and generation parameters.  All
    subsequent ``generate`` calls reuse this configuration.

    Attributes:
        provider: The active provider name (e.g. ``"ollama"``).
        model: The resolved model identifier.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ) -> None:
        """Initialise the connector from a YAML config file.

        Args:
            provider: Provider name (``"ollama"``, ``"openrouter"``,
                …).  Falls back to ``default_provider`` in the config.
            model: Model name override.  Falls back to the provider's
                ``default_model`` in the config.
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the requested provider is not defined in the
                config.
        """
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        self.provider = provider or config.get("default_provider", "ollama")

        providers = config.get("providers", {})
        if self.provider not in providers:
            raise ValueError(
                f"Provider '{self.provider}' not found in {config_path}. "
                f"Available: {list(providers.keys())}"
            )

        provider_cfg = providers[self.provider]
        self.model = model or provider_cfg.get("default_model", "")
        self._api_base: Optional[str] = provider_cfg.get("api_base")
        self._api_key: Optional[str] = provider_cfg.get("api_key")
        # Prefix litellm uses to route the call. Defaults to the provider
        # name, but can be overridden (e.g. an OpenAI-compatible endpoint
        # served under a differently named provider sets this to "openai").
        self._litellm_provider: str = provider_cfg.get(
            "litellm_provider", self.provider
        )

        gen = config.get("generation", {})
        self._temperature: float = gen.get("temperature", 1.0)
        self._max_tokens: int = gen.get("max_tokens", 2000)

        # Provider-specific setup
        self._extra_headers: dict = {}
        if self.provider == "openrouter" and self._api_key:
            os.environ["OPENROUTER_API_KEY"] = self._api_key
        elif self.provider == "ollama" and self._api_key:
            self._extra_headers["Authorization"] = (
                f"Bearer {self._api_key}"
            )

    @property
    def _litellm_model(self) -> str:
        """Return the model string with the litellm provider prefix."""
        return f"{self._litellm_provider}/{self.model}"

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the response text.

        Uses ``litellm.completion`` with OpenAI-style chat messages.
        Retries up to ``_MAX_LLM_ATTEMPTS`` times if the response
        content is empty, or if the call itself raises (timeout,
        provider error, etc.). A short backoff is applied between
        retries.

        Args:
            prompt: The prompt string to send.

        Returns:
            The response text from the LLM, stripped of whitespace.
            Returns an empty string (with a warning) if every attempt
            fails or yields empty content. Downstream parsing treats
            an empty string as a missing prediction and falls back to
            type-specific defaults.
        """
        kwargs: dict = {
            "model": self._litellm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": _LLM_TIMEOUT_SECONDS,
        }

        if self._api_base:
            kwargs["api_base"] = self._api_base

        # OpenAI-compatible endpoints take the key via the api_key kwarg
        # (openrouter uses an env var; ollama uses an auth header).
        if self._api_key and self._litellm_provider == "openai":
            kwargs["api_key"] = self._api_key

        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        for attempt in range(1, _MAX_LLM_ATTEMPTS + 1):
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content
            except Exception as e:
                print(
                    f"WARN: LLM call failed (attempt {attempt}/"
                    f"{_MAX_LLM_ATTEMPTS}): {type(e).__name__}: {e}"
                )
                content = None

            if content is not None and content.strip():
                return content.strip()

            if attempt < _MAX_LLM_ATTEMPTS:
                if content is None or not content.strip():
                    print(
                        f"WARN: empty LLM response (attempt {attempt}/"
                        f"{_MAX_LLM_ATTEMPTS}), retrying in "
                        f"{_RETRY_BACKOFF_SECONDS}s..."
                    )
                time.sleep(_RETRY_BACKOFF_SECONDS)

        print(
            f"WARN: LLM call did not produce usable content after "
            f"{_MAX_LLM_ATTEMPTS} attempts; returning empty string."
        )
        return ''
