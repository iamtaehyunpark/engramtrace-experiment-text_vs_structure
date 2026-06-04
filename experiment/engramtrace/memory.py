"""
Physical Memory Orchestration — verbatim port from:
    iamtaehyunpark/engramtrace/backend/src/core/memory.py

Porting changes (minimum necessary):
  1. `from src.utils.nlp import ...`  →  `from engramtrace.nlp import ...`
  2. Constructor paths are required arguments (no hardcoded "src/memory/..." defaults).
  3. `trace_timing` prints redirected through Python's logging module.
"""

import hashlib
import json
import logging
import os
import time
from functools import wraps

import numpy as np
from bs4 import BeautifulSoup

from engramtrace.nlp import nltk_stop_words, lemmatizer, word_tokenize

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


class MemoryManager:
    STRUCTURAL_TAGS = {'body', 'main', 'section', 'article', 'div'}

    @trace_timing
    def __init__(self, kb_path, p_embeddings_path, structural_embeddings_path):
        self.kb_path = kb_path
        self.p_embeddings_path = p_embeddings_path
        self.structural_embeddings_path = structural_embeddings_path
        self.soup = self._load_or_create_kb()

    @trace_timing
    def wipe(self):
        """Wipes the physical HTML Graph and Embeddings JSON."""
        for path in (self.kb_path, self.p_embeddings_path, self.structural_embeddings_path):
            if os.path.exists(path):
                os.remove(path)
        self.soup = self._load_or_create_kb()

    # ── INITIALISE KNOWLEDGE BASE ────────────────────────────────────────────

    @trace_timing
    def _load_or_create_kb(self):
        """Loads the KB if it exists, otherwise initialises a root skeleton."""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, "r", encoding="utf-8") as f:
                return BeautifulSoup(f, "lxml")

        default_html = "<html><body><main id='root'></main></body></html>"
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, "w", encoding="utf-8") as f:
            f.write(default_html)
        return BeautifulSoup(default_html, "lxml")

    @trace_timing
    def atomizer(self, llm_client, raw_text=None, compress=False):
        """
        Uses the LLM to semantically structure raw text into HTML.
        If raw_text is None, globally regenerates the KB (compressing if compress=True).
        """
        if raw_text is None:
            root = self.soup.find(id="root") or self.soup.find("body") or self.soup
            kb_html = str(root)
            generated_html = llm_client.generate_structured_html(kb_html, compress=compress)
        else:
            generated_html = llm_client.generate_structured_html(raw_text, compress=compress)

        self.soup = BeautifulSoup(generated_html, "lxml")
        return self._finalize_and_sync(llm_client, hierarchical=True)

    @trace_timing
    def _sectionize(self):
        container = self.soup.find(id="root") or self.soup.find("body")
        if not container:
            return
        self._wrap_heading_level(container, 1)

    def _wrap_heading_level(self, parent, level):
        if level > 6:
            return

        h_tag = f"h{level}"
        if not parent.find_all(h_tag, recursive=False):
            self._wrap_heading_level(parent, level + 1)
            return

        children = list(parent.children)
        for child in children:
            child.extract()

        current_section = None
        for child in children:
            is_heading          = hasattr(child, 'name') and child.name == h_tag
            is_already_section  = hasattr(child, 'name') and child.name == 'section'

            if is_heading:
                current_section = self.soup.new_tag("section")
                parent.append(current_section)
                current_section.append(child)
            elif is_already_section:
                parent.append(child)
                current_section = None
            elif current_section is not None:
                current_section.append(child)
            else:
                parent.append(child)

        for section in parent.find_all("section", recursive=False):
            self._wrap_heading_level(section, level + 1)

    @trace_timing
    def _finalize_and_sync(self, llm_client, hierarchical=False):
        """
        Synchronises structural DOM edits back onto the physical KB.
        1. Assign deterministic IDs to all nodes.
        2. Trigger vector rebuilding.
        """
        finalized_html = self.finalize_atomization(str(self.soup))
        self.soup = BeautifulSoup(finalized_html, "lxml")
        self.save_kb(finalized_html)

        all_active_ids = [p['id'] for p in self.soup.find_all('p')
                          if p.get('id') and p.get_text(strip=True)]

        if hierarchical:
            self.sync_embeddings_hierarchical(llm_client, all_active_ids)
        else:
            self.sync_embeddings(llm_client, all_active_ids)

        return all_active_ids

    @trace_timing
    def rewrite(self, selector, updated_content):
        """
        Safely swaps interior content of an existing node, or splices new blocks globally.
        """
        logger.debug("[MemoryManager.rewrite] Applying DOM mutation hook onto tag '%s'...", selector)

        target = None
        if selector:
            if '#' in selector and len(selector.split('#')) == 2:
                tag_name, tag_id = selector.split('#')
                target = self.soup.find(tag_name, id=tag_id)
            else:
                try:
                    target = self.soup.select_one(selector)
                except Exception:
                    pass

        new_tag_source = BeautifulSoup(updated_content, "html.parser")
        new_nodes = [n for n in new_tag_source.children if n.name]

        if target and new_nodes:
            target.replace_with(new_nodes[0])
            return True

        root_container = self.soup.find(id="root") or self.soup.find("body")
        if root_container and new_nodes:
            for node in new_nodes:
                root_container.append(node)
            return True

        return False

    def _generate_deterministic_id(self, text, prefix='p'):
        hash_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{prefix}-{hash_digest[:12]}"

    @trace_timing
    def finalize_atomization(self, generated_html):
        """Processes the AI's HTML, ensuring every node has a stable, deterministic ID."""
        soup = BeautifulSoup(generated_html, "lxml")
        structural_tags = [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'article', 'section', 'div', 'main', 'span', 'b', 'strong', 'i', 'em', 'u',
        ]
        for tag in soup.find_all(structural_tags, id=False):
            if tag.get_text(strip=True):
                stable_id = self._generate_deterministic_id(tag.get_text(), prefix=tag.name)
                tag['id'] = stable_id
        return soup.prettify()

    @trace_timing
    def save_kb(self, content):
        """Persists the BeautifulSoup object to the HTML file."""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, "w", encoding="utf-8") as f:
            f.write(content)

    # ── SEARCH ───────────────────────────────────────────────────────────────

    @trace_timing
    def get_all_p_contents(self):
        return {p['id']: p.get_text() for p in self.soup.find_all('p')}

    def _get_structural_lineage(self, tag):
        ancestors = []
        for parent in tag.parents:
            if parent.name and parent.name in self.STRUCTURAL_TAGS and parent.get('id'):
                ancestors.append(parent)
        ancestors.reverse()
        return ancestors

    def _build_selector_path(self, tag):
        parents = list(tag.parents)
        path_parts = []
        for p in reversed(parents):
            if p.name and p.name != '[document]':
                if p.get('id'):
                    path_parts.append(f"{p.name}#{p.get('id')}")
                else:
                    path_parts.append(p.name)
        path_parts.append(f"{tag.name}#{tag.get('id')}")
        return " > ".join(path_parts)

    @trace_timing
    def sync_embeddings_hierarchical(self, llm_client, active_ids):
        """
        Full top-down hierarchical embedding rebuild.
        Called only during atomizer (init + day change).
        """
        logger.debug("[MemoryManager.sync_embeddings_hierarchical] Building hierarchical structural vectors...")
        from datetime import datetime

        structural_cache = {}
        texts_to_embed   = {}
        lineage_cache    = {}

        def get_lineage(tag):
            tid = tag.get('id')
            if tid not in lineage_cache:
                lineage_cache[tid] = self._get_structural_lineage(tag)
            return lineage_cache[tid]

        p_tags = []
        for p_id in active_ids:
            tag = self.soup.find(id=p_id)
            if not tag:
                continue
            p_tags.append((p_id, tag))
            lineage = get_lineage(tag)
            for ancestor in lineage:
                aid = ancestor.get('id')
                if aid and aid not in texts_to_embed:
                    texts_to_embed[aid] = ancestor.get_text()
            if p_id not in texts_to_embed:
                texts_to_embed[p_id] = tag.get_text()

        if not texts_to_embed:
            return

        batch_ids   = list(texts_to_embed.keys())
        batch_texts = [texts_to_embed[tid] for tid in batch_ids]
        logger.debug("[Hierarchical] Batch embedding %d nodes", len(batch_texts))
        raw_vectors = llm_client.generate_embeddings(batch_texts)

        raw_embeddings = {tid: np.array(raw_vectors[i], dtype='float32')
                          for i, tid in enumerate(batch_ids)}

        p_embedding_map = {}
        for p_id, tag in p_tags:
            lineage    = get_lineage(tag)
            full_chain = lineage + [tag]

            for node in full_chain:
                nid = node.get('id')
                if nid in structural_cache:
                    continue
                raw = raw_embeddings.get(nid)
                if raw is None:
                    continue
                parent_ancestors = get_lineage(node)
                if parent_ancestors:
                    parent_id = parent_ancestors[-1].get('id')
                    if parent_id and parent_id in structural_cache:
                        structural_cache[nid] = raw * 0.7 + structural_cache[parent_id] * 0.3
                    else:
                        structural_cache[nid] = raw
                else:
                    structural_cache[nid] = raw

            if p_id in structural_cache:
                p_embedding_map[p_id] = {
                    "selector": self._build_selector_path(tag),
                    "vector":   structural_cache[p_id].tolist(),
                    "last_consolidated": datetime.now().isoformat(),
                }

        os.makedirs(os.path.dirname(self.p_embeddings_path), exist_ok=True)
        with open(self.p_embeddings_path, "w") as f:
            json.dump(p_embedding_map, f, indent=4)

        structural_persist = {sid: vec.tolist()
                               for sid, vec in structural_cache.items()
                               if sid not in active_ids}
        os.makedirs(os.path.dirname(self.structural_embeddings_path), exist_ok=True)
        with open(self.structural_embeddings_path, "w") as f:
            json.dump(structural_persist, f, indent=4)

        logger.debug("[Hierarchical] Persisted %d p-vectors, %d structural vectors",
                     len(p_embedding_map), len(structural_persist))

    @trace_timing
    def sync_embeddings(self, llm_client, active_ids):
        """Lightweight stage-update path. Reads cached structural vectors (read-only)."""
        logger.debug("[MemoryManager.sync_embeddings] Stage-update: syncing with structural cache...")
        from datetime import datetime

        embedding_map = {}
        if os.path.exists(self.p_embeddings_path):
            with open(self.p_embeddings_path, "r") as f:
                try:
                    embedding_map = json.load(f)
                except json.JSONDecodeError:
                    pass

        structural_cache = {}
        if os.path.exists(self.structural_embeddings_path):
            with open(self.structural_embeddings_path, "r") as f:
                try:
                    structural_cache = json.load(f)
                except json.JSONDecodeError:
                    pass

        if isinstance(embedding_map, dict):
            for key in [k for k in embedding_map if k not in active_ids]:
                del embedding_map[key]

        new_nodes, new_contents, new_selectors, new_parent_ids = [], [], [], []
        for p_id in active_ids:
            if p_id not in embedding_map:
                tag = self.soup.find(id=p_id)
                if tag:
                    new_nodes.append(p_id)
                    new_contents.append(tag.get_text())
                    new_selectors.append(self._build_selector_path(tag))
                    lineage = self._get_structural_lineage(tag)
                    parent_id = None
                    for ancestor in reversed(lineage):
                        aid = ancestor.get('id')
                        if aid and aid in structural_cache:
                            parent_id = aid
                            break
                    new_parent_ids.append(parent_id)

        if new_nodes:
            vectors = llm_client.generate_embeddings(new_contents)
            for i, p_id in enumerate(new_nodes):
                raw_vec   = np.array(vectors[i], dtype='float32')
                parent_id = new_parent_ids[i]
                if parent_id and parent_id in structural_cache:
                    parent_vec = np.array(structural_cache[parent_id], dtype='float32')
                    final_vec  = (raw_vec * 0.7 + parent_vec * 0.3).tolist()
                else:
                    final_vec  = raw_vec.tolist()
                embedding_map[p_id] = {
                    "selector": new_selectors[i],
                    "vector":   final_vec,
                    "last_consolidated": datetime.now().isoformat(),
                }

        os.makedirs(os.path.dirname(self.p_embeddings_path), exist_ok=True)
        with open(self.p_embeddings_path, "w") as f:
            json.dump(embedding_map, f, indent=4)

    @trace_timing
    def keyword_search(self, query):
        """Traditional keyword search — returns list of hit IDs sorted by score."""
        logger.debug("[MemoryManager.keyword_search] Performing keyword search...")
        if not self.soup:
            return []

        tokens   = word_tokenize(query)
        keywords = {lemmatizer.lemmatize(t.lower()) for t in tokens
                    if t.isalnum() and t.lower() not in nltk_stop_words and len(t) > 2}
        if not keywords:
            return []

        hit_scores = {}
        for tag in self.soup.find_all(['p', 'li']):
            if tag.get('id') and tag.get_text(strip=True):
                text_lower = tag.get_text().lower()
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    hit_scores[tag['id']] = score

        return sorted(hit_scores, key=lambda k: hit_scores[k], reverse=True)

    @trace_timing
    def semantic_search(self, query_vector, threshold=0.80):
        """Compares query vector against all p-embeddings. Returns list of hit IDs."""
        logger.debug("[MemoryManager.semantic_search] Traversing dense node space...")
        if not os.path.exists(self.p_embeddings_path):
            return []

        with open(self.p_embeddings_path, "r") as f:
            embedding_map = json.load(f)

        if not embedding_map:
            return []

        ids     = list(embedding_map.keys())
        vectors = np.array([item['vector'] for item in embedding_map.values()], dtype='float32')
        q_vec   = np.array(query_vector, dtype='float32')

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        dot_product  = np.dot(vectors, q_vec)
        norms        = np.linalg.norm(vectors, axis=1) * q_norm
        norms        = np.where(norms == 0, 1e-10, norms)
        similarities = dot_product / norms

        mask = similarities >= threshold
        if len(np.where(mask)[0]) == 0:
            return [ids[i] for i in np.argsort(similarities)[-3:]]
        return [ids[i] for i in np.where(mask)[0]]
