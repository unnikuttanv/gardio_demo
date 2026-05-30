"""Environment-driven configuration.

Single source of truth for runtime settings. Reads from environment / `.env`.
Provider-aware: the key for the model's provider is resolved automatically, and
if that key is missing, `use_mock_llm` is True so the report layer falls back to
a deterministic mock instead of calling a real model.

Default provider is Groq (groq.com). Switch models by changing `LLM_MODEL` to any
LiteLLM model string (e.g. `xai/grok-4`, `openai/gpt-4o`, `ollama/llama3`) and
supplying the matching key.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

# Maps a LiteLLM provider prefix -> the Settings field holding its API key.
_PROVIDER_KEY_FIELDS = {
    "groq": "groq_api_key",
    "xai": "xai_api_key",
    "openai": "openai_api_key",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 800

    # Provider API keys (only the one matching llm_model's provider is used).
    groq_api_key: str = ""
    xai_api_key: str = ""
    openai_api_key: str = ""

    # Data
    data_dir: str = "data"
    sample_record: str = "synthetic"

    @property
    def provider(self) -> str:
        """LiteLLM provider prefix of the configured model (e.g. 'groq')."""
        return self.llm_model.split("/", 1)[0] if "/" in self.llm_model else ""

    @property
    def api_key(self) -> str:
        """API key for the configured model's provider ('' if none set)."""
        field = _PROVIDER_KEY_FIELDS.get(self.provider)
        return getattr(self, field, "") if field else ""

    @property
    def use_mock_llm(self) -> bool:
        """Use the deterministic mock when no matching credential is available."""
        return not self.api_key.strip()


def get_settings() -> Settings:
    """Return a fresh Settings instance (re-reads env each call; cheap)."""
    return Settings()
