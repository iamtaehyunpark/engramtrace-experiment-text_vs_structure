"""Shared utilities for all experiment phases."""

import re
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Condition A — Linear text
# ---------------------------------------------------------------------------

def build_linear_text(conversation: dict) -> str:
    lines = []
    for session in conversation['sessions']:
        date = session.get('date', 'Unknown date')
        lines.append(f'--- Session: {date} ---')
        for turn in session['turns']:
            speaker = turn['speaker']
            ts      = turn.get('timestamp', '')
            content = re.sub(r'<[^>]+>', '', turn['content'])
            lines.append(f'[{speaker}, {ts}]: {content}')
        lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Condition B — Chunked RAG
# ---------------------------------------------------------------------------

def build_chunks(conversation: dict) -> list:
    chunks = []
    for s_idx, session in enumerate(conversation['sessions']):
        for t_idx, turn in enumerate(session['turns']):
            chunks.append({
                'text':        (f"[{turn['speaker']}, {session['date']}, "
                                f"Session {s_idx+1}]\n{turn['content']}"),
                'speaker':     turn['speaker'],
                'session_idx': s_idx,
                'turn_idx':    t_idx,
                'date':        session['date'],
                'timestamp':   turn.get('timestamp', ''),
            })
    return chunks


def build_faiss_flat_index(chunks: list, encoder) -> faiss.Index:
    texts = [c['text'] for c in chunks]
    embs  = encoder.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False).astype('float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve_chunks(query: str, chunks: list, index: faiss.Index,
                    encoder, k: int = 5) -> list:
    q_emb    = encoder.encode([query], normalize_embeddings=True).astype('float32')
    _, idxs  = index.search(q_emb, k)
    retrieved = [chunks[i] for i in idxs[0]]
    retrieved.sort(key=lambda x: (x['session_idx'], x['turn_idx']))
    return retrieved


# ---------------------------------------------------------------------------
# Condition C — Structured XML
# ---------------------------------------------------------------------------

def escape_xml(text: str) -> str:
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def build_xml(conversation: dict) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<conversation>']
    for s_idx, session in enumerate(conversation['sessions']):
        date = session.get('date', f'session-{s_idx+1}')
        parts.append(f'  <session id="{s_idx+1}" date="{date}">')
        for turn in session['turns']:
            speaker = escape_xml(turn['speaker'])
            ts      = turn.get('timestamp', '')
            content = re.sub(r'<[^>]+>', '', turn['content'])
            parts.append(f'    <turn speaker="{speaker}" timestamp="{ts}">')
            for sent in re.split(r'(?<=[.!?])\s+', content.strip()):
                if sent.strip():
                    parts.append(f'      <utterance>{escape_xml(sent.strip())}</utterance>')
            parts.append('    </turn>')
        parts.append('  </session>')
    parts.append('</conversation>')
    return '\n'.join(parts)


def validate_xml(original: dict, xml_str: str) -> bool:
    """Well-formedness check + content integrity (every turn's first 30 chars preserved)."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return False
    all_utterances = ' '.join(u.text or '' for u in root.iter('utterance'))
    for session in original['sessions']:
        for turn in session['turns']:
            key = re.sub(r'<[^>]+>', '', turn['content'])[:30]
            if key and key not in all_utterances:
                return False
    return True


# ---------------------------------------------------------------------------
# Condition C2 — Hierarchical XML retrieval
# ---------------------------------------------------------------------------

ALPHA = 0.7


def extract_nodes(xml_str: str) -> list:
    """
    Parse XML into a flat node list. Each node stores its parent_id so we can
    reconstruct the ancestor chain at retrieval time. Nodes are ordered
    top-down (parent always before child), which is required by
    compute_hierarchical_embeddings().
    """
    root  = ET.fromstring(xml_str)
    nodes = []

    def recurse(el, parent_id: int, depth: int, path: list):
        node_id     = len(nodes)
        direct_text = (el.text or '').strip()
        for child in el:
            if child.tail:
                direct_text += ' ' + child.tail.strip()
        attribs    = ' '.join(f'{k}="{v}"' for k, v in el.attrib.items())
        path_entry = f'<{el.tag} {attribs}>'.strip()
        nodes.append({
            'node_id':      node_id,
            'tag':          el.tag,
            'text_content': direct_text,
            'full_path':    path + [path_entry],
            'depth':        depth,
            'parent_id':    parent_id,
            'xml_snippet':  ET.tostring(el, encoding='unicode'),
        })
        for child in el:
            recurse(child, node_id, depth + 1, path + [path_entry])

    recurse(root, -1, 0, [])
    return nodes


def compute_hierarchical_embeddings(nodes: list, encoder,
                                     alpha: float = ALPHA) -> np.ndarray:
    """
    v_n = alpha * v_local_n + (1-alpha) * v_parent_n
    Root nodes use v_local only. Because nodes are stored top-down,
    hier_embs[pid] is always computed before hier_embs[nid].
    """
    texts      = [n['text_content'] if n['text_content'] else n['tag']
                  for n in nodes]
    local_embs = encoder.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)
    hier_embs  = np.zeros_like(local_embs)
    for node in nodes:
        nid = node['node_id']
        pid = node['parent_id']
        hier_embs[nid] = (local_embs[nid] if pid == -1
                          else alpha * local_embs[nid] + (1 - alpha) * hier_embs[pid])
    norms = np.linalg.norm(hier_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (hier_embs / norms).astype('float32')


def build_c2_index(nodes: list, encoder) -> faiss.Index:
    hier_embs = compute_hierarchical_embeddings(nodes, encoder)
    index     = faiss.IndexFlatIP(hier_embs.shape[1])
    index.add(hier_embs)
    return index


def retrieve_nodes_hierarchical(query: str, nodes: list, index: faiss.Index,
                                  encoder, k: int = 5) -> list:
    q_emb    = encoder.encode([query], normalize_embeddings=True).astype('float32')
    _, idxs  = index.search(q_emb, k)
    results  = []
    for idx in idxs[0]:
        node      = nodes[idx]
        ancestors = []
        pid = node['parent_id']
        while pid != -1:
            ancestors.insert(0, nodes[pid])
            pid = nodes[pid]['parent_id']
        results.append({'node': node, 'ancestors': ancestors})
    return results


def format_c2_context(retrieved: list) -> str:
    parts = []
    for i, item in enumerate(retrieved):
        node      = item['node']
        ancestors = item['ancestors']
        path      = (' > '.join(f"<{a['tag']}>" for a in ancestors)
                     if ancestors else '<root>')
        parts.append(
            f"--- Retrieved Node {i+1} [path: {path} > <{node['tag']}>] ---\n"
            f"{node['xml_snippet']}"
        )
    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_f1(pred: str, ref: str) -> float:
    pred_toks = pred.lower().split()
    ref_toks  = ref.lower().split()
    common    = sum((Counter(pred_toks) & Counter(ref_toks)).values())
    if not common:
        return 0.0
    p = common / len(pred_toks)
    r = common / len(ref_toks)
    return 2 * p * r / (p + r)


def compute_bleu1(pred: str, ref: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu(
        [ref.lower().split()], pred.lower().split(),
        weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    )


def compute_all_metrics(pred: str, ref: str, rouge, sbert) -> dict:
    from sentence_transformers import util
    r        = rouge.score(ref, pred)
    pred_emb = sbert.encode(pred, convert_to_tensor=True)
    ref_emb  = sbert.encode(ref,  convert_to_tensor=True)
    from nltk.translate.meteor_score import meteor_score
    return {
        'f1':        compute_f1(pred, ref),
        'bleu1':     compute_bleu1(pred, ref),
        'rougeL':    r['rougeL'].fmeasure,
        'rouge2':    r['rouge2'].fmeasure,
        'meteor':    meteor_score([ref.split()], pred.split()),
        'sbert_sim': float(util.cos_sim(pred_emb, ref_emb)),
    }
