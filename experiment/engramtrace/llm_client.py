"""
LocalLLMClient — drop-in replacement for LangChainClient that uses local models:
  - LLM    : Qwen2.5-72B-Instruct-AWQ or Qwen2.5-7B-Instruct via vLLM
  - Encoder: BAAI/bge-base-en-v1.5 via SentenceTransformers

Interface is identical to LangChainClient so memory.py and brain.py require
zero changes to the call sites.

Additional method `generate_query_embedding(query)` applies the bge-base
asymmetric instruction prefix — brain.py calls it when available.
"""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_SYSTEM_HTML = """\
You are converting unstructured text into a well-organized HTML document.

INPUT: Raw, unstructured text containing various concepts and facts.
OUTPUT: A clean HTML document with logical structure.

STRUCTURE RULES:
1. Organize content hierarchically using <h1>, <h2>, <h3> for topics and subtopics.
2. ALL content must be strictly nested inside its semantic parent. Wrap each heading and ALL of its associated content (<p>, <ul>, <ol>, <table>, etc.) in a <section> tag. Nest sections hierarchically: h3-sections inside h2-sections inside h1-sections. Nothing should float as an orphan sibling of a heading it belongs to.
3. Each <p> tag must contain exactly one coherent concept or fact. This is critical — downstream systems use individual <p> tags as atomic retrieval units for semantic search. Break large blocks into multiple <p> tags.
4. Lists (<ul>/<ol>) belong inside the <section> of the heading they relate to, after any introductory <p> tag.
5. Preserve all original content and its semantic meaning. Do not omit or summarize.
6. When contradictions exist in the source text, keep the latest version.
7. Do NOT add id attributes. The engine assigns deterministic IDs automatically.
8. Return only clean HTML markup. No markdown, no code fences."""

_SYSTEM_HTML_COMPRESS = """\
You are restructuring an existing HTML knowledge base for long-term storage efficiency.

INPUT: An HTML document that has grown organically over multiple sessions.
OUTPUT: A reorganized HTML document with improved logical grouping.

STRUCTURE RULES:
1. Use <h1>, <h2>, <h3> headers to create a clear topical hierarchy.
2. ALL content must be strictly nested inside its semantic parent. Wrap each heading and ALL of its associated content in a <section> tag. Nest sections hierarchically.
3. Each <p> tag must contain exactly one coherent concept or fact.
4. Merge duplicate or near-duplicate information. When facts conflict, the most recently added version takes priority.
5. Preserve the substantive content. Do not aggressively summarize — reorganize and deduplicate.
6. Do NOT add id attributes. The engine assigns deterministic IDs automatically.
7. Return only clean HTML markup. No markdown, no code fences."""

_SYSTEM_SYNTHESIZE = """\
You are merging new knowledge from a conversation into existing HTML knowledge fragments.

INPUT:
- "Original HTML context": Existing HTML fragments from the knowledge base that are topically relevant to the conversation.
- "Conversation log": A sequence of Q&A pairs containing new information to integrate.

OUTPUT: Updated or new HTML fragments ready for DOM insertion.

RULES:
1. If new knowledge fits into the provided Original HTML fragments, rewrite the ENTIRE fragment incorporating the new information. Preserve the original id attributes on parent tags you did not change textually.
2. If the new knowledge is unrelated to any provided fragment, create a NEW standalone <section> with appropriate headings and content.
3. ALL content must be strictly nested inside its semantic parent.
4. Each <p> tag must contain exactly one coherent concept — these are the atomic retrieval units for semantic search.
5. For any content you rewrite or create new, do NOT include id attributes.
6. Output ONLY raw HTML tags. No full <html>/<body> wrapper. No markdown. No commentary."""


class LocalLLMClient:
    def __init__(
        self,
        llm,
        tokenizer,
        encoder_model: str = "BAAI/bge-base-en-v1.5",
        encoder_device: str = "cpu",
        max_tokens_html: int = 4096,
        max_tokens_answer: int = 256,
        temperature: float = 0.1,
    ):
        """
        Parameters
        ----------
        llm        : loaded vLLM LLM instance
        tokenizer  : loaded HuggingFace tokenizer (for chat-template formatting)
        encoder_model : SentenceTransformer model name
        encoder_device: device for the encoder ("cpu" or "cuda")
        max_tokens_html   : max new tokens for HTML generation calls
        max_tokens_answer : max new tokens for QA answer generation
        temperature       : sampling temperature (0.1 matches Gemini default)
        """
        from vllm import SamplingParams

        self.llm       = llm
        self.tokenizer = tokenizer
        self.temperature = temperature

        self.html_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_html,
        )
        self.answer_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_answer,
            stop=["\n\nQuestion:", "\n\nQ:"],
        )

        logger.info("[LocalLLMClient] Loading encoder: %s on %s", encoder_model, encoder_device)
        self.encoder = SentenceTransformer(encoder_model, device=encoder_device)

        # Cumulative token counters — cover ALL LLM calls (KB structuring + QA answers).
        # Reset with reset_token_counts() before each unit of measurement.
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def reset_token_counts(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def get_token_counts(self) -> dict:
        return {
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    # ── Private helper ────────────────────────────────────────────────────────

    def _chat(self, system_prompt: str, human_prompt: str, params) -> str:
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": human_prompt},
        ]
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output  = self.llm.generate([prompt], params)
        self.total_input_tokens  += len(output[0].prompt_token_ids)
        self.total_output_tokens += len(output[0].outputs[0].token_ids)
        result  = output[0].outputs[0].text.strip()
        return result.replace("```html", "").replace("```", "").strip()

    # ── LangChainClient interface ─────────────────────────────────────────────

    def generate_structured_html(self, raw_text: str, compress: bool = False) -> str:
        """Converts raw text (or existing KB HTML) into structured HTML."""
        if compress:
            system  = _SYSTEM_HTML_COMPRESS
            human   = f"Restructure and deduplicate the following knowledge base HTML:\n\n{raw_text}"
        else:
            system  = _SYSTEM_HTML
            human   = f"Convert the following text into structured HTML:\n\n{raw_text}"
        return self._chat(system, human, self.html_params)

    def synthesize_session(self, log_history, context_html: str) -> str:
        """Merges an active Stage Log into the anchoring KB context."""
        if isinstance(log_history, list):
            history_str = "\n".join(
                f"Q: {item['query']}\nA: {item['response']}" for item in log_history
            )
        else:
            history_str = log_history
        human = f"Original HTML context:\n{context_html}\n\nConversation log:\n{history_str}"
        return self._chat(_SYSTEM_SYNTHESIZE, human, self.html_params)

    def generate_embeddings(self, text_list: list) -> list:
        """
        Batch-generates document-style embeddings (no instruction prefix).
        Returns list of float lists — same shape as Gemini Embedding-001 output
        (but 768-dim instead of 256-dim).
        """
        embeddings = self.encoder.encode(
            text_list,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def generate_query_embedding(self, query: str) -> list:
        """
        Query-side embedding with bge-base asymmetric instruction prefix.
        Called by brain.py's run_inference() when available.
        """
        emb = self.encoder.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb[0].tolist()

    def generate_response(self, query: str, context: str, history: list,
                          session_history: list = None) -> str:
        """Generates a conversational response with long-term + short-term context."""
        history_str = "\n".join(
            f"Q: {item['query']}\nA: {item['response']}" for item in history
        ) if history else ""

        system_content = (
            "You are an intelligent conversational assistant. "
            "Answer the user's questions using your full general knowledge.\n\n"
            "You also have access to supplementary memory sources that may contain "
            "relevant context from past interactions. Use them naturally — like a person "
            "recalling relevant memories — but they are not your only source of knowledge. "
            "If the user asks something outside of what's stored in memory, answer freely "
            "from your own understanding.\n\n"
            f"Long-term memory (retrieved from knowledge base, might be entirely irrelevant):\n"
            f"{context if context else '(No relevant memories retrieved)'}\n\n"
            f"Recent conversation context (current topic buffer):\n"
            f"{history_str if history_str else '(New conversation regarding this topic)'}"
        )

        messages = [{"role": "system", "content": system_content}]
        if session_history:
            for item in session_history:
                messages.append({"role": "user",      "content": item["query"]})
                messages.append({"role": "assistant",  "content": item["response"]})
        messages.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self.llm.generate([prompt], self.answer_params)
        self.total_input_tokens  += len(output[0].prompt_token_ids)
        self.total_output_tokens += len(output[0].outputs[0].token_ids)
        return output[0].outputs[0].text.strip()
