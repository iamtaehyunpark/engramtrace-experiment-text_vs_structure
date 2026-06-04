"""
Cognitive Processing Layer — verbatim port from:
    iamtaehyunpark/engramtrace/backend/src/core/brain.py

Porting changes (minimum necessary):
  1. `from src.core.memory import MemoryManager` → `from engramtrace.memory import MemoryManager`
  2. EngramTrace.__init__() accepts `base_dir` so session/stage files go to the
     per-conversation data directory instead of hardcoded "src/memory/...".
  3. Brain.__init__() forwards `base_dir` to EngramTrace.
  4. brain.run_inference(): query embedding uses `llm.generate_query_embedding(query)`
     when available (asymmetric retrieval with bge-base instruction prefix), falling
     back to `llm.generate_embeddings([query])[0]` for API compatibility.
"""

import json
import logging
import os
import time
from datetime import datetime
from functools import wraps

import numpy as np

from engramtrace.memory import MemoryManager

logger = logging.getLogger(__name__)


def trace_timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("Function '%s' started", func.__name__)
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.time() - start_time
            logger.debug("Function '%s' finished in %.4fs", func.__name__, elapsed)
    return wrapper


# ════════════════════════════════════════════════════════════════════════════
# EngramTrace — session / stage management
# ════════════════════════════════════════════════════════════════════════════

class EngramTrace:
    @trace_timing
    def __init__(self, base_dir="src/memory"):
        self.base_dir     = base_dir
        self.sessions_dir = os.path.join(base_dir, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.sessions = {}

        old_session_log = os.path.join(base_dir, "session_log.json")
        default_session = os.path.join(self.sessions_dir, "default.json")
        if os.path.exists(old_session_log) and not os.path.exists(default_session):
            import shutil
            shutil.move(old_session_log, default_session)

        old_stage_log   = os.path.join(base_dir, "current_stage_log.json")
        default_stage   = os.path.join(self.sessions_dir, "default_stage.json")
        if os.path.exists(old_stage_log) and not os.path.exists(default_stage):
            import shutil
            shutil.move(old_stage_log, default_stage)

        self.set_session("default")

    def set_session(self, session_id):
        self.active_session_id = session_id
        if session_id not in self.sessions:
            self.sessions[session_id] = {"current_trace": set(), "qa_vecs": []}

        self.session_log_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(self.session_log_path):
            with open(self.session_log_path, "w") as f:
                json.dump([], f)

        self.stage_log_path = os.path.join(self.sessions_dir, f"{session_id}_stage.json")
        if not os.path.exists(self.stage_log_path):
            with open(self.stage_log_path, "w") as f:
                json.dump([], f)

    @property
    def current_trace(self):
        return self.sessions[self.active_session_id]["current_trace"]

    @current_trace.setter
    def current_trace(self, value):
        self.sessions[self.active_session_id]["current_trace"] = value

    @property
    def qa_vecs(self):
        return self.sessions[self.active_session_id]["qa_vecs"]

    @qa_vecs.setter
    def qa_vecs(self, value):
        self.sessions[self.active_session_id]["qa_vecs"] = value

    def wipe(self, wipe_stage=True, wipe_session=True, wipe_trace=True):
        if wipe_trace:
            self.current_trace = set()
        if wipe_stage:
            self.qa_vecs = []
            os.makedirs(os.path.dirname(self.stage_log_path), exist_ok=True)
            with open(self.stage_log_path, "w") as f:
                json.dump([], f)
            history_dir = os.path.join(self.base_dir, "stage_history")
            if os.path.exists(history_dir):
                for fname in os.listdir(history_dir):
                    if fname.endswith(".json"):
                        os.remove(os.path.join(history_dir, fname))
        if wipe_session:
            with open(self.session_log_path, "w") as f:
                json.dump([], f)

    def _clear_stage_log(self):
        with open(self.stage_log_path, "w") as f:
            json.dump([], f)

    def _get_stage_log(self):
        try:
            if not os.path.exists(self.stage_log_path):
                return []
            with open(self.stage_log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _get_session_log(self):
        try:
            if not os.path.exists(self.session_log_path):
                return []
            with open(self.session_log_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _get_last_stage_time(self, log=None):
        try:
            if log is None:
                with open(self.stage_log_path, "r") as f:
                    log = json.load(f)
            if log and isinstance(log, list):
                return datetime.fromisoformat(log[-1]["timestamp"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    @trace_timing
    def _get_stage_context(self):
        """Returns structurally minimized subset of KB: parent nodes of current trace."""
        logger.debug("[EngramTrace] Abstracting parent-node vectors for Ecphory...")
        parents = set()
        for pid in self.current_trace:
            tag = self.memory.soup.find(id=pid) if hasattr(self, 'memory') else None
            if tag and tag.parent:
                parents.add(tag.parent)
        return "\n".join([str(p) for p in parents])

    @trace_timing
    def _calculate_stage_drift(self, query_vec):
        """Detects if the user is still on the same topic/stage via cosine similarity."""
        if len(self.qa_vecs) < 1:
            session_log = self._get_session_log()
            if session_log and "last_qa_vec" in session_log[-1]:
                avg_vec = np.array(session_log[-1]["last_qa_vec"])
            else:
                return 0.0
        else:
            avg_vec = np.mean(self.qa_vecs, axis=0)

        norm_product = np.linalg.norm(avg_vec) * np.linalg.norm(query_vec)
        if norm_product == 0:
            return 0
        return np.dot(avg_vec, query_vec) / norm_product

    @trace_timing
    def BufferQAPair(self, query, response, qa_vec=None, no_memorize=False):
        """Appends the latest interaction to the Stage Log and permanent Session Log."""
        qa_block = {
            "query":     query,
            "response":  response,
            "timestamp": datetime.now().isoformat(),
        }

        if not no_memorize:
            stage = self._get_stage_log()
            stage.append(qa_block)
            with open(self.stage_log_path, "w") as f:
                json.dump(stage, f)

        session_qa_block = dict(qa_block)
        if qa_vec is not None:
            session_qa_block["last_qa_vec"] = (qa_vec if isinstance(qa_vec, list)
                                                else getattr(qa_vec, "tolist", lambda: qa_vec)())

        session = self._get_session_log()
        if session and "last_qa_vec" in session[-1]:
            del session[-1]["last_qa_vec"]
        session.append(session_qa_block)
        with open(self.session_log_path, "w") as f:
            json.dump(session, f)

    @trace_timing
    def start_new_stage(self, preserve_trace=False):
        log = self._get_stage_log() if os.path.exists(self.stage_log_path) else []
        if log:
            self._archive_stage(log)
        self._clear_stage_log()
        if not preserve_trace:
            self.current_trace = set()
        self.qa_vecs = []

    @trace_timing
    def _archive_stage(self, log):
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir = os.path.join(self.base_dir, "stage_history")
        archive_path = os.path.join(history_dir, f"stage_{timestamp}.json")
        os.makedirs(history_dir, exist_ok=True)
        with open(archive_path, "w") as f:
            json.dump(log, f)


# ════════════════════════════════════════════════════════════════════════════
# Brain — main cognitive orchestrator
# ════════════════════════════════════════════════════════════════════════════

class Brain:
    @trace_timing
    def __init__(self, memory_manager, llm_client, base_dir="src/memory",
                 stage_threshold=0.83, search_threshold=0.80):
        self.memory          = memory_manager
        self.llm             = llm_client
        self.stage_threshold = stage_threshold
        self.search_threshold = search_threshold
        self.engram_trace    = EngramTrace(base_dir=base_dir)

        self.engram_trace.memory        = memory_manager
        self.engram_trace.current_trace = set()

        if not os.path.exists(self.engram_trace.stage_log_path):
            os.makedirs(os.path.dirname(self.engram_trace.stage_log_path), exist_ok=True)
            self.engram_trace._clear_stage_log()

        if not os.path.exists(self.engram_trace.session_log_path):
            os.makedirs(os.path.dirname(self.engram_trace.session_log_path), exist_ok=True)
            with open(self.engram_trace.session_log_path, "w") as f:
                json.dump([], f)
        else:
            try:
                log      = self.engram_trace._get_stage_log()
                active_qa = [f"Q: {item['query']}\nA: {item['response']}" for item in log[-3:]]
                if active_qa:
                    self.engram_trace.qa_vecs = self.llm.generate_embeddings(active_qa)
            except Exception:
                pass

    @trace_timing
    def consolidate_and_transition(self, preserve_trace=False):
        """Merges the ephemeral Stage Log into the HTML KB via the LLM."""
        logger.debug("[Brain.consolidate_and_transition] Merging Stage Log into KB...")
        log = self.engram_trace._get_stage_log()
        if not log:
            return

        parents = set()
        for p_id in self.engram_trace.current_trace:
            tag = self.memory.soup.find(id=p_id)
            if not tag:
                continue
            parents.add(tag.parent if tag.parent else tag)

        if len(parents) == 0:
            updated_parts = self.llm.synthesize_session(log, self.memory.soup.get_text())
        else:
            context_str   = "\n".join([str(p) for p in parents])
            updated_parts = self.llm.synthesize_session(log, context_str)

        from bs4 import BeautifulSoup
        soup      = BeautifulSoup(updated_parts, "lxml")
        container = soup.find('body') if soup.find('body') else soup

        for tag in container.find_all(recursive=False):
            if tag.name:
                if tag.name == 'main' and tag.get('id') == 'root':
                    for sub_tag in tag.find_all(recursive=False):
                        if sub_tag.name:
                            if sub_tag.get('id'):
                                self.memory.rewrite(f"{sub_tag.name}#{sub_tag.get('id')}", str(sub_tag))
                            else:
                                self.memory.rewrite(None, str(sub_tag))
                    continue
                if tag.get('id'):
                    self.memory.rewrite(f"{tag.name}#{tag.get('id')}", str(tag))
                else:
                    self.memory.rewrite(None, str(tag))

        self.memory._finalize_and_sync(self.llm)
        self.engram_trace.start_new_stage(preserve_trace=preserve_trace)

    @trace_timing
    def _update_query_vector(self, query_vec_raw, stage_log):
        """Modifies the raw query vector based on session history and stage state."""
        q_vec = np.array(query_vec_raw, dtype='float32')
        if len(stage_log) >= 1 and self.engram_trace.qa_vecs:
            last_qa_vec = np.array(self.engram_trace.qa_vecs[-1], dtype='float32')
            q_vec = q_vec * 0.7 + last_qa_vec * 0.3
        return q_vec.tolist()

    @trace_timing
    def run_inference(self, query, stage_threshold=None, search_threshold=None,
                      no_search=False, no_memorize=False):
        """The main cognitive loop: Drift Check → Retrieval → Inference → Buffer."""
        logger.debug("[Brain.run_inference] Cognitive loop: '%s...'", query[:40])

        # Porting note: two embeddings — doc-style for drift/buffering (matches original
        # symmetric Gemini behaviour), query-style for retrieval (bge-base instruction prefix).
        q_vec_doc = self.llm.generate_embeddings([query])[0]
        if hasattr(self.llm, 'generate_query_embedding'):
            q_vec_search = self.llm.generate_query_embedding(query)
        else:
            q_vec_search = q_vec_doc

        stage_log   = self.engram_trace._get_stage_log()
        session_log = self.engram_trace._get_session_log()

        active_stage_threshold  = stage_threshold  if stage_threshold  is not None else self.stage_threshold
        active_search_threshold = search_threshold if search_threshold is not None else self.search_threshold

        if len(stage_log) >= 15:
            logger.debug("Q-A pairs >= 15. Consolidating Stage...")
            self.consolidate_and_transition(preserve_trace=no_search)
        elif len(stage_log) == 0 and not (session_log and "last_qa_vec" in session_log[-1]):
            logger.debug("Stage log empty and no prior session history. Skip drift check.")
        else:
            logger.debug("[EngramTrace] Drift check...")
            # Drift detection uses doc-style embedding, consistent with qa_vecs (also doc-style).
            similarity = self.engram_trace._calculate_stage_drift(q_vec_doc)
            logger.debug("Similarity: %.4f (threshold: %.2f)", similarity, active_stage_threshold)
            if similarity < active_stage_threshold:
                last_stage_time = self.engram_trace._get_last_stage_time(log=stage_log)
                if last_stage_time and (datetime.now() - last_stage_time).total_seconds() > 4000:
                    logger.debug("Day changed — consolidating and compressing KB...")
                    if len(stage_log) > 0:
                        self.consolidate_and_transition(preserve_trace=no_search)
                    self.memory.atomizer(self.llm, compress=True)
                else:
                    logger.debug("Topic drift detected (%.2f). Consolidating...", similarity)
                    self.consolidate_and_transition(preserve_trace=no_search)

        # Apply topic-anchor blending to the search vector (30% last qa_vec).
        # _update_query_vector is symmetric — the blend logic is the same regardless of
        # the embedding style, so passing q_vec_search is correct.
        q_vec_search = self._update_query_vector(q_vec_search, stage_log)

        if not no_search:
            hit_ids = self.memory.semantic_search(q_vec_search, threshold=active_search_threshold)
            self.engram_trace.current_trace.update(hit_ids)
            if len(stage_log) == 0:
                kw_hit_ids = self.memory.keyword_search(query)
                self.engram_trace.current_trace.update(kw_hit_ids)

        working_context = self.engram_trace._get_stage_context()
        stage_history   = self.engram_trace._get_stage_log()
        session_history = session_log[-4:]

        response = self.llm.generate_response(
            query          = query,
            context        = working_context,
            history        = stage_history,
            session_history= session_history,
        )

        # qa_vec is doc-style — used for drift detection of future queries.
        qa_text = f"Q: {query}\nA: {response}"
        qa_vec  = self.llm.generate_embeddings([qa_text])[0]

        self.engram_trace.qa_vecs.append(qa_vec)
        if len(self.engram_trace.qa_vecs) > 3:
            self.engram_trace.qa_vecs.pop(0)

        self.engram_trace.BufferQAPair(query, response, qa_vec=qa_vec, no_memorize=no_memorize)
        return response
