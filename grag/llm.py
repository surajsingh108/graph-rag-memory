import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class LLM:
    """Lazy-loading wrapper around a HuggingFace text-generation pipeline."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._pipe = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import transformers
        transformers.logging.set_verbosity_error()
        logger.info("Loading LLM %s (this may take a while on first run)", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._pipe = pipeline(
            "text-generation",
            model=self._model_id,
            tokenizer=self._tokenizer,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            return_full_text=False,
        )

    def _format(self, user_message: str) -> str:
        """Apply chat template when available, otherwise return the raw string."""
        if self._tokenizer and getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_message

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        self._load()
        formatted = self._format(prompt)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        result = self._pipe(formatted, **gen_kwargs)
        return result[0]["generated_text"].strip()

    def answer(self, prompt: str) -> str:
        """Generate a free-form answer."""
        return self._generate(prompt, max_new_tokens=512, do_sample=True)

    def summarise(self, text: str) -> str:
        """Condense text to 2-3 sentences."""
        prompt = (
            "Summarise the following in 2-3 concise sentences, preserving key facts:\n\n"
            f"{text}\n\nSummary:"
        )
        return self._generate(prompt, max_new_tokens=128, do_sample=True)

    def extract_triples(self, text: str) -> str:
        """Prompt the LLM to extract (subject, relation, object) triples as JSON."""
        prompt = (
            'Extract up to 10 factual (subject, relation, object) triples from the text below.\n'
            'Return ONLY a valid JSON array, no explanation. Example:\n'
            '[{"subject": "Einstein", "relation": "born_in", "object": "Ulm"}]\n\n'
            f"Text: {text[:1000]}\n\nJSON:"
        )
        return self._generate(prompt, max_new_tokens=256, do_sample=False)
