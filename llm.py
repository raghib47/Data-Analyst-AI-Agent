"""Chat LLM wrapper around the Hugging Face Inference API."""
from __future__ import annotations

from huggingface_hub import InferenceClient

from config import config
from utils.logger import get_logger

logger = get_logger(__name__)


class ChatLLM:
    """Thin wrapper providing a simple chat completion interface."""

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            model: Model repo id (defaults to config).
            token: HF token (defaults to config).
        """
        self.model = model or config.chat_model
        self._client = InferenceClient(token=token or config.hf_token)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a chat completion.

        Args:
            system: System prompt.
            user: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens.

        Returns:
            The assistant's text reply, or an error message string.
        """
        try:
            response = self._client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM request failed: %s", exc)
            return f"[LLM error] {type(exc).__name__}: {exc}"
