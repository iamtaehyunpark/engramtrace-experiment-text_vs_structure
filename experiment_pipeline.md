
# Experiment Pipeline Specification

## Structured vs. Flat Memory Representation for LLM Agents

### Concept Verification Experiment — EngramTrace Research

> **Format choice:** Condition C uses **XML** (not HTML) for full-context structured representation. Condition C2 adds **hierarchical XML retrieval** — the core EngramTrace mechanism. Together they isolate two separate claims: (1) does structure alone help reasoning? (2) does structured retrieval beat flat RAG? See Section 2.3 and 2.4 for full rationale.

**Author:** Taehyun Park · University of Wisconsin–Madison

**Model:** Qwen2.5 (72B + 7B) · **Benchmark:** LoCoMo (Full Dataset, 1,540 QA pairs)

**Hardware:** NVIDIA H200 · **Year:** 2026

---

## Table of Contents

1. [Experiment Overview](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#1-experiment-overview)
2. [Experimental Conditions](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#2-experimental-conditions)
3. [Dataset Preparation](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#3-dataset-preparation)
4. [Inference Pipeline](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#4-inference-pipeline)
5. [Evaluation Metrics](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#5-evaluation-metrics)
6. [Results Storage and Analysis](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#6-results-storage-and-analysis)
7. [Execution Checklist](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#7-execution-checklist)
8. [How to Use Results in the Paper](https://claude.ai/chat/d804a5da-b5ba-4572-81ee-db6402ecc1fa#8-how-to-use-results-in-the-paper)

---

## 1. Experiment Overview

This document specifies the complete pipeline for the concept verification experiment underlying the EngramTrace paper. The goal is **not** to evaluate EngramTrace as a full system — it is to verify the foundational premise: that hierarchical/structural representation of memory content enables an LLM to reason more effectively and more efficiently than flat linear representation of identical content.

### 1.1 Research Question

> **Core Questions:**
>
> 1. Does structured XML markup (vs. flat linear text) of identical content improve LLM reasoning quality? *(Condition C)*
> 2. Does hierarchical XML retrieval (vs. flat chunk retrieval) achieve better accuracy at lower token cost? *(Condition C2)*
>
> **Hypotheses:**
>
> * **H1 (Structure):** Hierarchical XML structure improves accuracy on multi-hop and temporal questions by preserving contextual relationships, even when all content is present.
> * **H2 (Retrieval):** Retrieving hierarchically-embedded XML nodes with ancestral context outperforms flat chunk retrieval on multi-hop reasoning while consuming fewer tokens — resolving the accuracy-efficiency tradeoff.

### 1.2 What This Experiment Is NOT

This is a controlled **format comparison** experiment, not a full EngramTrace system evaluation. The following EngramTrace components are  **not used** :

* Online consolidation pipeline (no staged memory accumulation)
* Stage drift detection or dynamic stage management
* Homeostatic Day System (no periodic restructuring)
* Ecphory dual-mode retrieval mechanism
* SHA-256 node identification and delta tracking

The experiment isolates exactly  **one variable per comparison** : the representation format of memory content. Everything else — LLM, questions, underlying knowledge — is held constant across all conditions. The exception is Condition C2, which additionally introduces hierarchical embedding computation — the one EngramTrace mechanism required to test the retrieval claim.

### 1.3 Experiment at a Glance

| Dimension            | Specification                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| Benchmark            | LoCoMo (Maharana et al., 2024) — full dataset, 1,540 QA pairs                |
| Question categories  | Single-hop, Multi-hop, Temporal, Open-domain, Adversarial                     |
| Conditions           | 5 (Full Linear, Chunked RAG, Full XML, XML Hierarchical Retrieval, No Memory) |
| Primary models       | Qwen2.5-72B-Instruct, Qwen2.5-7B-Instruct                                     |
| Hardware             | NVIDIA H200                                                                   |
| Inference framework  | vLLM (offline batch inference)                                                |
| Primary metrics      | F1 score, BLEU-1, input token count, inference time                           |
| Secondary metrics    | ROUGE-L, METEOR, SBERT similarity                                             |
| Total inference runs | 1,540 × 5 conditions × 2 models =**15,400 runs**                      |

---

## 2. Experimental Conditions

Five conditions are defined. All conditions receive identical questions from LoCoMo. The **only variables** are how the conversation history is formatted and presented, and whether retrieval is used.

---

### Condition A — Full Linear Context *(Ceiling Baseline)*

**Description:** The entire LoCoMo conversation history relevant to a question is concatenated as plain prose text and provided in the LLM prompt. No structural markup. No retrieval. All information present.

**Purpose:** Establishes the accuracy ceiling — the maximum performance achievable when the LLM has access to all information in an unfiltered form. Also establishes the token cost ceiling.

**Expected token count:** ~16,000–17,000 per query (matching LoCoMo baseline reported in prior work).

**Prompt template:**

```
System: You are a helpful assistant. Answer the question based on
        the conversation history below.

Conversation History:
[Speaker A, 2023-10-01]: <turn content>
[Speaker B, 2023-10-01]: <turn content>
[Speaker A, 2023-10-03]: <turn content>
... (all turns concatenated chronologically as plain text)

Question: <question text>
Answer:
```

**Construction rule:** Concatenate all conversation turns in chronological order. Preserve speaker labels and timestamps. Do not summarize, chunk, or restructure. Strip any HTML if present in source data.

---

### Condition B — Chunked RAG *(Retrieval Baseline)*

**Description:** The conversation history is split into fixed-size chunks. At query time, the top-k most semantically similar chunks are retrieved via cosine similarity and provided to the LLM.

**Purpose:** Represents the dominant existing approach — standard RAG applied to conversational memory. Tests whether chunking and retrieval helps or hurts compared to full context.

**Expected failure mode:** Semantic isolation on multi-hop and temporal questions, where relevant information is split across chunks and retrieved without surrounding context.

**Chunking specification:**

* Chunk unit: one conversation turn (one speaker utterance per chunk)
* Chunk metadata: speaker label, timestamp, session index — prepended to each chunk
* Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (matches A-MEM's setup for comparability)
* Retrieval: cosine similarity, top-k = **5** chunks per query
* No re-ranking. Chunks returned in chronological order after retrieval.

**Prompt template:**

```
System: You are a helpful assistant. Answer the question based on
        the retrieved conversation excerpts below.

Retrieved Excerpts (most relevant to the question):
--- Excerpt 1 [Speaker A, 2023-10-01, Session 2] ---
<chunk content>
--- Excerpt 2 [Speaker B, 2023-10-03, Session 3] ---
<chunk content>
... (top-5 chunks by cosine similarity)

Question: <question text>
Answer:
```

**Construction rule:** Embed all chunks offline before inference. At query time, embed the question, compute cosine similarity against all chunk embeddings, retrieve top-5, assemble prompt. Use FAISS for efficient similarity search.

---

### Condition C — Structured XML *(Test Condition)*

> ⚠️ **Critical design choice — XML, not HTML:** XML is used rather than HTML for two reasons. First, XML lets you name tags after what they *mean* — `<session>`, `<turn>`, `<utterance>` are semantically transparent domain concepts, whereas HTML tags like `<div>` and `<span>` carry layout and presentation-layer associations from web pre-training. Second, XML is stricter and schema-free in a controlled way — every tag means exactly what you define it to mean, with no inherited browser-rendering semantics. This produces the cleanest possible test of the structural hypothesis. If an LLM were used to restructure the content, summarization quality would become a confound — so conversion is rule-based only.

**Description:** The conversation history is converted into a hierarchical XML document using rule-based structuring (no LLM preprocessing). The full structured document is provided in the LLM prompt context. The LLM reads the XML natively.

**Purpose:** Tests whether semantic XML structure — hierarchical containment, explicit speaker attribution, chronological session grouping — enables better reasoning compared to flat text of identical content. This is the  **core claim to validate** .

**Why XML over HTML for this experiment:**

|                  | HTML                                                     | XML (chosen)                                     |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------ |
| Tag semantics    | Layout/presentation priors (`<div>`= visual container) | Domain-defined (`<turn>`= conversational unit) |
| LLM associations | Web rendering context                                    | Pure semantic structure                          |
| Schema           | Inherited browser schema                                 | Fully custom, experiment-controlled              |
| Confound risk    | Presentation associations may influence reasoning        | None — tags are novel and meaningful            |
| Reproducibility  | Browser quirks possible                                  | Strict, deterministic                            |

**XML conversion rules (deterministic, rule-based):**

* XML declaration: `<?xml version="1.0" encoding="UTF-8"?>`
* Root element: `<conversation>`
* Each LoCoMo dialogue session → `<session id="N" date="YYYY-MM-DD">`
* Each speaker turn → `<turn speaker="Name" timestamp="HH:MM">`
* Each sentence within a turn → individual `<utterance>` element
* No attributes beyond speaker, timestamp, date, and id — keep schema minimal

**XML structure template:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<conversation>

  <session id="1" date="2023-10-01">
    <turn speaker="Alice" timestamp="10:32">
      <utterance>I just got back from hiking the Appalachian Trail last weekend.</utterance>
      <utterance>It was exhausting but I loved every moment of it.</utterance>
    </turn>
    <turn speaker="Bob" timestamp="10:35">
      <utterance>That sounds amazing. How long was the section you hiked?</utterance>
    </turn>
  </session>

  <session id="2" date="2023-10-08">
    <turn speaker="Alice" timestamp="14:20">
      <utterance>I started a new job at the design firm downtown this week.</utterance>
    </turn>
  </session>

</conversation>
```

**Prompt template:**

```
System: You are a helpful assistant. Answer the question based on
        the conversation record below. The record is formatted as XML.
        Use the tag structure to understand the speaker, chronological,
        and topical organization of the conversation.

<?xml version="1.0" encoding="UTF-8"?>
<conversation>
... (full structured XML document)
</conversation>

Question: <question text>
Answer:
```

---

### Condition D — No Memory *(Floor Baseline)*

**Description:** The LLM receives only the question, with no conversation history of any kind. Relies entirely on parametric knowledge.

**Purpose:** Establishes the absolute performance floor. Shows how much memory in any format contributes relative to the LLM's pretrained knowledge alone.

**Prompt template:**

```
System: You are a helpful assistant. Answer the question as best you can.

Question: <question text>
Answer:
```

---

### Condition Summary

|                                |  A: Full Linear  | B: Chunked RAG |    C: Full XML    |        C2: XML Retrieval        | D: No Memory |
| ------------------------------ | :--------------: | :------------: | :---------------: | :------------------------------: | :----------: |
| All content present            |        ✅        |    Partial    |        ✅        |             Partial             |      ❌      |
| Structural XML markup          |        ❌        |       ❌       |        ✅        |                ✅                |      —      |
| Retrieval step                 |        ❌        |       ✅       |        ❌        |                ✅                |      —      |
| Hierarchical embeddings        |        ❌        |       ❌       |        ❌        |                ✅                |      —      |
| Ancestral context in retrieval |        ❌        |       ❌       |        ❌        |                ✅                |      —      |
| LLM preprocessing              |        ❌        |       ❌       |        ❌        |                ❌                |      —      |
| Expected token cost            |     Highest     |      Low      |       High       |   **Lowest non-trivial**   |    Lowest    |
| Tests hypothesis               |        —        |       —       |        H1        |                H2                |      —      |
| Role in paper                  | Accuracy ceiling |  RAG baseline  | Structure benefit | **Core EngramTrace claim** |    Floor    |

**Key comparisons the design enables:**

| Comparison | What it proves                                                                            |
| ---------- | ----------------------------------------------------------------------------------------- |
| C vs A     | Does XML structure itself help, even with full context? (H1)                              |
| C2 vs B    | Does structured hierarchical retrieval beat flat RAG? (H2)                                |
| C2 vs A    | Does hierarchical retrieval match full-context accuracy at lower cost? (efficiency claim) |
| C2 vs C    | Does retrieval over structured data help vs. full structured context?                     |
| B vs A     | Does retrieval help vs. full context? (existing literature replication)                   |

---

## 3. Dataset Preparation

### 3.1 LoCoMo Dataset Statistics

| Property                | Value                                                            |
| ----------------------- | ---------------------------------------------------------------- |
| Total QA pairs          | 1,540 (answerable subset)                                        |
| Unique conversations    | 50 dialogues                                                     |
| Sessions per dialogue   | Up to 35                                                         |
| Turns per dialogue      | 300+                                                             |
| Tokens per conversation | ~9,000–16,000                                                   |
| Question categories     | 5: single-hop, multi-hop, temporal, open-domain, adversarial     |
| Format                  | JSON with conversations, sessions, QA pairs, supporting evidence |

### 3.2 Data Loading

```python
from datasets import load_dataset

dataset = load_dataset('snap-research/LoCoMo', split='test')

# Each item contains:
# item['conversation']     -> list of session dicts
# item['qa_pairs']         -> list of {question, answer, category, evidence}
# item['conversation_id']  -> unique dialogue identifier

# Filter to answerable subset only
dataset = dataset.filter(lambda x: x['answer'] != 'unanswerable')
print(f"Total QA pairs: {len(dataset)}")  # should be 1,540
```

> Filter to the **answerable subset** only. This matches the standard evaluation protocol used by A-MEM, MAGMA, and EverMemOS and is the basis for all published comparisons.

### 3.3 Project Directory Structure

Build all representations before running any inference. Build once, save to disk, then run all inference conditions from saved files. This ensures identical content across conditions.

```
experiment/
├── data/
│   ├── raw/                        # raw LoCoMo JSON
│   ├── condition_A/                # linear text files (one .txt per conversation)
│   ├── condition_B/
│   │   ├── chunks/                 # chunk JSON files (one per conversation)
│   │   └── embeddings/             # FAISS flat index per conversation
│   ├── condition_C/                # XML files (one .xml per conversation)
│   └── condition_C2/
│       ├── nodes/                  # parsed node JSON per conversation
│       └── embeddings/             # hierarchical embedding index per conversation
├── questions/
│   └── locomo_qa.jsonl             # all 1,540 QA pairs with category labels
├── results/
│   ├── condition_A/
│   │   ├── 72B.jsonl
│   │   └── 7B.jsonl
│   ├── condition_B/
│   ├── condition_C/
│   ├── condition_C2/
│   └── condition_D/
└── evaluation/
    ├── scores/                     # per-question metric files
    └── tables/                     # aggregated result tables (CSV)
```

### 3.4 Condition A Builder

```python
import re

def build_linear_text(conversation: dict) -> str:
    lines = []
    for session in conversation['sessions']:
        session_date = session.get('date', 'Unknown date')
        lines.append(f'--- Session: {session_date} ---')
        for turn in session['turns']:
            speaker  = turn['speaker']
            ts       = turn.get('timestamp', '')
            content  = re.sub(r'<[^>]+>', '', turn['content'])  # strip any HTML
            lines.append(f'[{speaker}, {ts}]: {content}')
        lines.append('')  # blank line between sessions
    return '\n'.join(lines)

# Usage
for conv in dataset:
    linear = build_linear_text(conv)
    with open(f"data/condition_A/{conv['conversation_id']}.txt", 'w') as f:
        f.write(linear)
```

### 3.5 Condition B Builder

```python
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np

encoder = SentenceTransformer('all-MiniLM-L6-v2')

def build_chunks(conversation: dict) -> list[dict]:
    chunks = []
    for s_idx, session in enumerate(conversation['sessions']):
        for t_idx, turn in enumerate(session['turns']):
            chunks.append({
                'text': (
                    f"[{turn['speaker']}, {session['date']}, "
                    f"Session {s_idx+1}]\n{turn['content']}"
                ),
                'speaker':     turn['speaker'],
                'session_idx': s_idx,
                'turn_idx':    t_idx,
                'date':        session['date'],
                'timestamp':   turn.get('timestamp', ''),
            })
    return chunks

def build_faiss_index(chunks: list[dict]) -> faiss.Index:
    texts      = [c['text'] for c in chunks]
    embeddings = encoder.encode(texts, normalize_embeddings=True).astype('float32')
    index      = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine for normalized vecs
    index.add(embeddings)
    return index

def retrieve_chunks(query: str, chunks: list, index: faiss.Index, k: int = 5) -> list[dict]:
    q_emb = encoder.encode([query], normalize_embeddings=True).astype('float32')
    _, indices = index.search(q_emb, k)
    retrieved  = [chunks[i] for i in indices[0]]
    # Re-sort by chronological order
    retrieved.sort(key=lambda x: (x['session_idx'], x['turn_idx']))
    return retrieved

# Usage
for conv in dataset:
    chunks = build_chunks(conv)
    index  = build_faiss_index(chunks)
    faiss.write_index(index, f"data/condition_B/embeddings/{conv['conversation_id']}.index")
    with open(f"data/condition_B/chunks/{conv['conversation_id']}.json", 'w') as f:
        json.dump(chunks, f)
```

### 3.6 Condition C Builder

```python
import re
import xml.etree.ElementTree as ET

def escape_xml(text: str) -> str:
    """Escape special characters for XML content."""
    return (text
        .replace('&', '&')
        .replace('<', '<')
        .replace('>', '>')
        .replace('"', '"')
        .replace("'", '''))

def build_xml(conversation: dict) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<conversation>']

    for s_idx, session in enumerate(conversation['sessions']):
        date = session.get('date', f'session-{s_idx + 1}')
        parts.append(f'  <session id="{s_idx + 1}" date="{date}">')

        for turn in session['turns']:
            speaker = escape_xml(turn['speaker'])
            ts      = turn.get('timestamp', '')
            content = re.sub(r'<[^>]+>', '', turn['content'])  # strip any existing markup

            parts.append(f'    <turn speaker="{speaker}" timestamp="{ts}">')

            # Split into sentences -> individual <utterance> elements
            sentences = re.split(r'(?<=[.!?])\s+', content.strip())
            for sent in sentences:
                if sent.strip():
                    parts.append(f'      <utterance>{escape_xml(sent.strip())}</utterance>')

            parts.append('    </turn>')
        parts.append('  </session>')

    parts.append('</conversation>')
    return '\n'.join(parts)


def validate_xml(original: dict, xml_str: str) -> bool:
    """
    Two-stage validation:
    1. Well-formedness: parse with ElementTree — raises on malformed XML
    2. Content integrity: verify first 30 chars of every turn appear in output
    """
    # Stage 1: well-formedness
    try:
        ET.fromstring(xml_str)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return False

    # Stage 2: content integrity
    # Check against plain text extracted from XML
    root = ET.fromstring(xml_str)
    all_utterances = ' '.join(u.text or '' for u in root.iter('utterance'))
    for session in original['sessions']:
        for turn in session['turns']:
            key = re.sub(r'<[^>]+>', '', turn['content'])[:30]
            if key not in all_utterances:
                return False
    return True


# Usage — ALWAYS validate after building
failures = []
for conv in dataset:
    xml_str = build_xml(conv)
    if not validate_xml(conv, xml_str):
        failures.append(conv['conversation_id'])
        print(f"VALIDATION FAILED: {conv['conversation_id']}")
    else:
        with open(f"data/condition_C/{conv['conversation_id']}.xml", 'w') as f:
            f.write(xml_str)

print(f"Built {len(dataset) - len(failures)} XML files. Failures: {len(failures)}")
```

> **Important:** `validate_xml()` runs two checks — XML well-formedness (parse error → malformed output) and content integrity (every turn's first 30 characters survive the conversion). Both must pass for every file. The experiment's validity depends on Condition C containing identical information to Condition A — only the format differs. Note that `xml.etree.ElementTree` is part of Python's standard library — no additional install needed.

### 3.7 Condition C2 Builder — Hierarchical XML Retrieval

Condition C2 requires two things flat RAG does not have: (1) a parsed node list extracted from the XML tree, and (2) hierarchical embeddings that blend each node's local vector with its parent's embedding. At query time, the top-k nodes are retrieved by hierarchical embedding similarity, and each retrieved node is returned **together with its full ancestral path** (all parent nodes from root to the matched node), preserving the context that gives the node its meaning.

```python
import xml.etree.ElementTree as ET
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss, json

encoder = SentenceTransformer('all-MiniLM-L6-v2')
ALPHA   = 0.7   # blending coefficient: local content weight


def extract_nodes(xml_str: str) -> list[dict]:
    """
    Parse XML into a flat list of nodes, each with:
      - node_id:       unique index within this conversation
      - tag:           XML element tag name
      - text_content:  all text directly in this element (not children)
      - full_path:     list of ancestor tag+attrib strings from root to this node
      - depth:         depth in tree (root = 0)
      - parent_id:     node_id of parent (-1 for root)
      - xml_snippet:   the raw XML of this node + its children (for prompt assembly)
    """
    root    = ET.fromstring(xml_str)
    nodes   = []
    id_map  = {}   # ET element -> node_id

    def recurse(el, parent_id, depth, path):
        node_id = len(nodes)
        id_map[el] = node_id

        # Text content = direct text only (not text from child elements)
        direct_text = (el.text or '').strip()
        for child in el:
            # include tail text (text after a child tag but before next sibling)
            if child.tail:
                direct_text += ' ' + child.tail.strip()

        attribs = ' '.join(f'{k}="{v}"' for k, v in el.attrib.items())
        path_entry = f'<{el.tag} {attribs}>'.strip()

        nodes.append({
            'node_id':     node_id,
            'tag':         el.tag,
            'text_content': direct_text,
            'full_path':   path + [path_entry],
            'depth':       depth,
            'parent_id':   parent_id,
            'xml_snippet': ET.tostring(el, encoding='unicode'),
        })

        for child in el:
            recurse(child, node_id, depth + 1, path + [path_entry])

    recurse(root, -1, 0, [])
    return nodes


def compute_hierarchical_embeddings(nodes: list[dict],
                                    alpha: float = ALPHA) -> np.ndarray:
    """
    Compute hierarchical embedding for each node:
      v_n = alpha * v_local_n + (1 - alpha) * v_parent_n
    Root node: v_root = v_local_root (no parent)
    Returns array of shape (num_nodes, embedding_dim).
    """
    texts       = [n['text_content'] if n['text_content'] else n['tag']
                   for n in nodes]
    local_embs  = encoder.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False)

    hier_embs = np.zeros_like(local_embs)
    for node in nodes:
        nid = node['node_id']
        pid = node['parent_id']
        if pid == -1:
            hier_embs[nid] = local_embs[nid]          # root
        else:
            hier_embs[nid] = (alpha * local_embs[nid]
                              + (1 - alpha) * hier_embs[pid])

    # Re-normalize after blending
    norms = np.linalg.norm(hier_embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)            # avoid div by zero
    return (hier_embs / norms).astype('float32')


def build_c2_index(nodes: list[dict]) -> faiss.Index:
    hier_embs = compute_hierarchical_embeddings(nodes)
    index = faiss.IndexFlatIP(hier_embs.shape[1])
    index.add(hier_embs)
    return index


def retrieve_nodes_hierarchical(query: str, nodes: list[dict],
                                 index: faiss.Index, k: int = 5) -> list[dict]:
    """
    Retrieve top-k nodes by hierarchical embedding similarity.
    For each retrieved node, also include its full ancestral path
    so the LLM has the context that gives the node meaning.
    """
    q_emb = encoder.encode([query], normalize_embeddings=True).astype('float32')
    _, idxs = index.search(q_emb, k)

    results = []
    for idx in idxs[0]:
        node = nodes[idx]
        # Collect ancestor chain: walk up parent_id links
        ancestors = []
        pid = node['parent_id']
        while pid != -1:
            ancestors.insert(0, nodes[pid])
            pid = nodes[pid]['parent_id']

        results.append({
            'node':      node,
            'ancestors': ancestors,  # ordered root → direct parent
        })
    return results


def format_c2_context(retrieved: list[dict]) -> str:
    """
    Format retrieved nodes + their ancestors into a readable XML context.
    Each retrieved item shows its ancestral path for context,
    then the matched node's full XML content.
    """
    parts = []
    for i, item in enumerate(retrieved):
        node      = item['node']
        ancestors = item['ancestors']

        # Show ancestor chain as context
        ancestor_path = ' > '.join(
            f"<{a['tag']}>" for a in ancestors
        ) if ancestors else '<root>'

        parts.append(
            f"--- Retrieved Node {i+1} "
            f"[path: {ancestor_path} > <{node['tag']}>] ---\n"
            f"{node['xml_snippet']}"
        )
    return '\n\n'.join(parts)


# Build and save C2 index for all conversations
for conv in dataset:
    xml_str = open(f"data/condition_C/{conv['conversation_id']}.xml").read()
    nodes   = extract_nodes(xml_str)
    index   = build_c2_index(nodes)

    # Save nodes
    with open(f"data/condition_C2/nodes/{conv['conversation_id']}.json", 'w') as f:
        # xml_snippet can be large; save all fields
        json.dump(nodes, f)

    # Save FAISS index
    faiss.write_index(
        index,
        f"data/condition_C2/embeddings/{conv['conversation_id']}.index"
    )

print("C2 indices built for all conversations.")
```

> **Design note:** `extract_nodes()` walks the entire XML tree and creates a flat list of every element. The hierarchical embedding computation in `compute_hierarchical_embeddings()` processes nodes in order (parent always before child, since the tree is walked top-down), so `hier_embs[pid]` is always already computed when needed. This is the core mathematical mechanism from EngramTrace — `alpha=0.7` means each node's embedding is 70% its own content, 30% propagated from its parent.

### 3.8 Questions File

```python
import json

qa_records = []
for conv in dataset:
    for qa in conv['qa_pairs']:
        if qa['answer'] != 'unanswerable':
            qa_records.append({
                'question_id':     qa['id'],
                'conversation_id': conv['conversation_id'],
                'category':        qa['category'],   # single_hop / multi_hop / temporal / open_domain / adversarial
                'question':        qa['question'],
                'answer':          qa['answer'],
                'evidence':        qa.get('evidence', []),
            })

with open('questions/locomo_qa.jsonl', 'w') as f:
    for r in qa_records:
        f.write(json.dumps(r) + '\n')

print(f"Saved {len(qa_records)} QA pairs")
```

---

## 4. Inference Pipeline

### 4.1 Model Configuration

| Parameter          | Qwen2.5-72B                   | Qwen2.5-7B                   |
| ------------------ | ----------------------------- | ---------------------------- |
| HuggingFace ID     | `Qwen/Qwen2.5-72B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| Precision          | bfloat16                      | bfloat16                     |
| Framework          | vLLM                          | vLLM                         |
| Max new tokens     | 256                           | 256                          |
| Temperature        | **0.0**(greedy)         | **0.0**(greedy)        |
| Top-p              | 1.0 (disabled)                | 1.0 (disabled)               |
| Max context length | 32,768 tokens                 | 32,768 tokens                |

> Use `temperature=0.0` (greedy decoding) throughout. This ensures **deterministic outputs** — identical inputs always produce identical outputs — which is critical for reproducibility and for isolating format as the only variable.

### 4.2 vLLM Initialization

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model='Qwen/Qwen2.5-72B-Instruct',
    dtype='bfloat16',
    max_model_len=32768,
    gpu_memory_utilization=0.90,
    enforce_eager=False,       # use CUDA graphs for speed
    tensor_parallel_size=1,    # increase to 2 if OOM on 72B
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    stop=['\n\nQuestion:', '\n\nAnswer:'],
)

# Batch inference — submit all prompts at once for throughput
outputs = llm.generate(prompts, sampling_params)
```

### 4.3 Prompt Assembly

```python
import json, faiss
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-72B-Instruct')

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def load_representations(conv_id: str) -> dict:
    # Load pre-built representations from disk
    with open(f'data/condition_A/{conv_id}.txt') as f:
        linear = f.read()
    with open(f'data/condition_C/{conv_id}.xml') as f:
        xml = f.read()
    with open(f'data/condition_B/chunks/{conv_id}.json') as f:
        chunks = json.load(f)
    with open(f'data/condition_C2/nodes/{conv_id}.json') as f:
        nodes = json.load(f)
    b_index  = faiss.read_index(f'data/condition_B/embeddings/{conv_id}.index')
    c2_index = faiss.read_index(f'data/condition_C2/embeddings/{conv_id}.index')
    return {
        'linear':   linear,
        'xml':      xml,
        'chunks':   chunks,
        'nodes':    nodes,
        'b_index':  b_index,
        'c2_index': c2_index,
    }

def assemble_prompt(condition: str, question: str, conv_id: str,
                    representations: dict) -> dict:
    """
    Returns dict with:
      'prompt'       -> final string for vLLM
      'input_tokens' -> token count measured BEFORE inference
    """
    rep = representations[conv_id]

    if condition == 'A':
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation history below.\n\n"
            f"Conversation History:\n{rep['linear']}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'B':
        retrieved = retrieve_chunks(question, rep['chunks'], rep['b_index'], k=5)
        excerpts  = '\n\n'.join([
            f"--- Excerpt {i+1} [{c['speaker']}, {c['date']}] ---\n{c['text']}"
            for i, c in enumerate(retrieved)
        ])
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation excerpts below.\n\n"
            f"Retrieved Excerpts:\n{excerpts}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'C':
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the conversation record below. The record is formatted as XML. "
            "Use the tag structure to understand the speaker, chronological, "
            "and topical organization of the conversation.\n\n"
            f"{rep['xml']}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'C2':
        retrieved = retrieve_nodes_hierarchical(
            question, rep['nodes'], rep['c2_index'], k=5
        )
        context = format_c2_context(retrieved)
        prompt = (
            "You are a helpful assistant. Answer the question based on "
            "the retrieved conversation nodes below. Each node is shown with "
            "its hierarchical path (ancestors) for context, followed by its "
            "XML content.\n\n"
            f"Retrieved Nodes:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    elif condition == 'D':
        prompt = (
            "You are a helpful assistant. Answer the question as best you can.\n\n"
            f"Question: {question}\nAnswer:"
        )

    return {
        'prompt':       prompt,
        'input_tokens': count_tokens(prompt),
    }
```

### 4.4 Main Inference Loop

```python
import time, json
from datetime import datetime

def run_condition(condition: str, model_name: str, qa_pairs: list,
                  representations: dict, llm, sampling_params) -> list:
    results = []

    # Assemble all prompts first (batch)
    assembled = []
    for qa in qa_pairs:
        item = assemble_prompt(condition, qa['question'],
                               qa['conversation_id'], representations)
        assembled.append({'qa': qa, **item})

    prompts = [a['prompt'] for a in assembled]

    # Batch inference
    t_start = time.perf_counter()
    outputs  = llm.generate(prompts, sampling_params)
    t_total  = time.perf_counter() - t_start
    ms_per   = (t_total / len(outputs)) * 1000

    for item, output in zip(assembled, outputs):
        predicted = output.outputs[0].text.strip()
        results.append({
            'question_id':       item['qa']['question_id'],
            'conversation_id':   item['qa']['conversation_id'],
            'condition':         condition,
            'model':             model_name,
            'category':          item['qa']['category'],
            'question':          item['qa']['question'],
            'reference_answer':  item['qa']['answer'],
            'predicted_answer':  predicted,
            'input_tokens':      item['input_tokens'],
            'output_tokens':     len(output.outputs[0].token_ids),
            'inference_time_ms': ms_per,
            'timestamp':         datetime.utcnow().isoformat(),
        })

    return results

# Save results as JSONL
def save_results(results: list, path: str):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Saved {len(results)} records to {path}")
```

### 4.5 Execution Order

Run conditions in this order to minimize GPU reloading and maximize throughput:

```
1. Build all representations offline (no GPU needed)
   → Condition A (linear text)
   → Condition B (chunks + FAISS flat index)
   → Condition C (XML files)   [C2 reuses these same XML files]
   → Condition C2 (node extraction + hierarchical embeddings + FAISS index)

2. Load Qwen2.5-72B
   → Run Condition D  (shortest prompts)    → save results/condition_D/72B.jsonl
   → Run Condition B  (medium prompts)      → save results/condition_B/72B.jsonl
   → Run Condition C2 (medium prompts)      → save results/condition_C2/72B.jsonl
   → Run Condition C  (long prompts)        → save results/condition_C/72B.jsonl
   → Run Condition A  (longest prompts)     → save results/condition_A/72B.jsonl
   → Unload model

3. Load Qwen2.5-7B
   → Repeat same order
   → Unload model
```

> C2 runs before C in the execution order because C2 uses selective retrieval (short prompts like B), whereas C dumps the full XML (long prompts closer to A). Running short-prompt conditions first warms up the CUDA graph cache.
> C2 reuses the XML files built for Condition C — no need to rebuild them.

---

## 5. Evaluation Metrics

Compute all metrics  **after all inference is complete** , from saved JSONL result files. Do not compute metrics during inference — keep inference as fast as possible.

### 5.1 Primary Accuracy Metrics

#### F1 Score (Token Overlap)

The standard LoCoMo metric, used by A-MEM, MAGMA, and EverMemOS. Measures token-level overlap between predicted and reference answer.

```python
from collections import Counter

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()

    pred_counter = Counter(pred_tokens)
    gt_counter   = Counter(gt_tokens)

    common = sum((pred_counter & gt_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)
```

#### BLEU-1 Score

Unigram precision with brevity penalty. Second standard LoCoMo metric.

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu1(prediction: str, ground_truth: str) -> float:
    reference  = [ground_truth.lower().split()]
    hypothesis = prediction.lower().split()
    smoothing  = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis,
                         weights=(1, 0, 0, 0),
                         smoothing_function=smoothing)
```

### 5.2 Secondary Accuracy Metrics

| Metric           | Library                   | Purpose                                                          |
| ---------------- | ------------------------- | ---------------------------------------------------------------- |
| ROUGE-L          | `rouge-score`           | Longest common subsequence — sentence-level fluency             |
| ROUGE-2          | `rouge-score`           | Bigram overlap — local word order                               |
| METEOR           | `nltk`                  | Semantic similarity with synonym matching                        |
| SBERT Similarity | `sentence-transformers` | Dense semantic similarity — catches paraphrased correct answers |

```python
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util

rouge  = rouge_scorer.RougeScorer(['rougeL', 'rouge2'], use_stemmer=True)
sbert  = SentenceTransformer('all-MiniLM-L6-v2')

def compute_all_metrics(pred: str, ref: str) -> dict:
    r        = rouge.score(ref, pred)
    pred_emb = sbert.encode(pred, convert_to_tensor=True)
    ref_emb  = sbert.encode(ref,  convert_to_tensor=True)
    return {
        'f1':       compute_f1(pred, ref),
        'bleu1':    compute_bleu1(pred, ref),
        'rougeL':   r['rougeL'].fmeasure,
        'rouge2':   r['rouge2'].fmeasure,
        'meteor':   meteor_score([ref.split()], pred.split()),
        'sbert_sim': float(util.cos_sim(pred_emb, ref_emb)),
    }
```

### 5.3 Cost Efficiency Metrics

These establish the efficiency side of the paper's claim. Report all of the following:

| Metric                           | Unit           | How to compute                           |
| -------------------------------- | -------------- | ---------------------------------------- |
| Mean input tokens                | tokens/query   | Average `input_tokens`per condition    |
| Total input tokens               | tokens         | Sum across 1,540 questions               |
| Token reduction vs A             | %              | `1 - (mean_C / mean_A) × 100`         |
| **Accuracy per 1k tokens** | F1 / 1k tokens | `mean_f1 / (mean_input_tokens / 1000)` |
| Mean inference time              | ms/query       | Wall clock via `time.perf_counter()`   |
| Total inference time             | minutes        | Sum across all questions                 |

> **The accuracy-per-1k-tokens metric is the most compelling single efficiency number for the paper.** It normalizes accuracy by cost and puts all conditions on equal footing. A condition achieving F1=0.40 at 2,000 tokens outperforms one achieving F1=0.42 at 17,000 tokens on this metric.

### 5.4 Statistical Significance

With 1,540 questions, you have sufficient statistical power for reliable significance tests.

```python
from scipy import stats
import numpy as np

def compare_conditions(scores_A: list, scores_C: list, label: str) -> dict:
    """Paired t-test comparing two conditions on F1 scores."""
    t_stat, p_value = stats.ttest_rel(scores_C, scores_A)
    cohen_d = (np.mean(scores_C) - np.mean(scores_A)) / np.std(scores_A)
    significant = p_value < 0.05
    print(f"{label}: t={t_stat:.3f}, p={p_value:.4f}, Cohen_d={cohen_d:.3f}, sig={significant}")
    return {'t': t_stat, 'p': p_value, 'd': cohen_d, 'significant': significant}

# Key comparisons to run for each category:
# compare_conditions(f1_A,  f1_C,  "C  vs A  — structure in context vs full linear")
# compare_conditions(f1_B,  f1_C2, "C2 vs B  — hierarchical retrieval vs flat RAG")
# compare_conditions(f1_A,  f1_C2, "C2 vs A  — hierarchical retrieval vs full context")
# compare_conditions(f1_C,  f1_C2, "C2 vs C  — retrieval over XML vs full XML")
# compare_conditions(f1_A,  f1_B,  "B  vs A  — flat RAG vs full context (lit replication)")
# Run separately per category (multi_hop, temporal, etc.)
```

---

## 6. Results Storage and Analysis

### 6.1 Per-Inference Record Schema

Every inference call saves one JSONL record with this exact schema:

```json
{
  "question_id":       "string — unique QA pair ID from LoCoMo",
  "conversation_id":   "string — which conversation this belongs to",
  "condition":         "string — A, B, C, C2, or D",
  "model":             "string — Qwen2.5-72B or Qwen2.5-7B",
  "category":          "string — single_hop / multi_hop / temporal / open_domain / adversarial",
  "question":          "string",
  "reference_answer":  "string",
  "predicted_answer":  "string",
  "input_tokens":      0,
  "output_tokens":     0,
  "inference_time_ms": 0.0,
  "f1":                0.0,
  "bleu1":             0.0,
  "rougeL":            0.0,
  "rouge2":            0.0,
  "meteor":            0.0,
  "sbert_sim":         0.0,
  "timestamp":         "ISO 8601 string"
}
```

### 6.2 Aggregation Script

```python
import pandas as pd, json

# Load all results
dfs = []
for condition in ['A', 'B', 'C', 'C2', 'D']:
    for model in ['72B', '7B']:
        path = f'results/condition_{condition}/{model}.jsonl'
        df = pd.read_json(path, lines=True)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# ── Main accuracy table ───────────────────────────────────────────────────
main_table = (
    df.groupby(['model', 'condition', 'category'])
    .agg(f1=('f1', 'mean'), bleu1=('bleu1', 'mean'))
    .round(4)
    .reset_index()
)

# ── Efficiency table ──────────────────────────────────────────────────────
eff = (
    df.groupby(['model', 'condition'])
    .agg(
        mean_f1=('f1', 'mean'),
        mean_input_tokens=('input_tokens', 'mean'),
        total_input_tokens=('input_tokens', 'sum'),
        mean_time_ms=('inference_time_ms', 'mean'),
    )
    .reset_index()
)
eff['f1_per_1k_tokens'] = eff['mean_f1'] / (eff['mean_input_tokens'] / 1000)

# Compute token reduction vs Condition A
for model in eff['model'].unique():
    baseline = eff.loc[(eff['model'] == model) & (eff['condition'] == 'A'),
                       'mean_input_tokens'].values[0]
    mask = eff['model'] == model
    eff.loc[mask, 'token_reduction_vs_A'] = (
        1 - eff.loc[mask, 'mean_input_tokens'] / baseline
    ) * 100

# Save
main_table.to_csv('evaluation/tables/main_results.csv', index=False)
eff.to_csv('evaluation/tables/efficiency.csv', index=False)

print("=== MAIN RESULTS ===")
print(main_table.to_string())
print("\n=== EFFICIENCY ===")
print(eff.to_string())
```

### 6.3 Expected Results Pattern

Based on the EngramTrace hypothesis and prior literature, results should approximately follow:

| Category            | Expected ranking               | Reasoning                                                                                         |
| ------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| Single-hop          | A ≈ C > C2 ≈ B > D           | Simple lookup — format and retrieval matter less                                                 |
| **Multi-hop** | **C2 > C ≈ A > B >> D** | **Hierarchical retrieval surfaces context-aware nodes; flat RAG loses cross-chunk context** |
| **Temporal**  | **C2 ≈ C > A > B > D**  | **XML `<session date="">`makes ordering explicit; retrieval narrows to right sessions**   |
| Open-domain         | A ≈ C > C2 ≈ B > D           | External knowledge needed; retrieval can help focus but format matters less                       |
| Adversarial         | C ≈ A > C2 ≈ B > D           | Full context needed to confirm absence — retrieval may miss the absence                          |
| Token count         | D << B ≈ C2 << C << A         | C2 retrieves selectively like B but with richer context per token                                 |

> **The two critical results:**
>
> * **C > A on multi-hop** validates H1 — structure alone helps reasoning
> * **C2 > B on multi-hop at C2 tokens ≈ B tokens** validates H2 — hierarchical retrieval beats flat RAG at comparable cost
>
> If both hold, the paper has a clean two-part argument: structure helps (C), and structured retrieval is more efficient than flat retrieval without sacrificing quality (C2).

---

## 7. Execution Checklist

### Phase 1 — Environment Setup

* [ ] Install dependencies: `pip install vllm transformers datasets sentence-transformers faiss-gpu rouge-score nltk scipy pandas`
* [ ] Install NLTK data: `python -m nltk.downloader punkt wordnet`
* [ ] Verify H200 visible: `nvidia-smi`
* [ ] Download `Qwen/Qwen2.5-72B-Instruct` weights (~140GB)
* [ ] Download `Qwen/Qwen2.5-7B-Instruct` weights (~15GB)
* [ ] Download LoCoMo from HuggingFace: `snap-research/LoCoMo`
* [ ] Confirm 1,540 answerable QA pairs after filtering

### Phase 2 — Data Preparation *(no GPU needed)*

* [ ] Build Condition A (linear text) for all 50 conversations → `data/condition_A/`
* [ ] Build Condition C (XML) for all 50 conversations → `data/condition_C/`
* [ ] **Run `validate_xml()` on all 50 files — zero failures required before proceeding**
* [ ] Build Condition B chunks for all 50 conversations → `data/condition_B/chunks/`
* [ ] Build Condition B FAISS flat indices → `data/condition_B/embeddings/`
* [ ] Build Condition C2 nodes (extract from C XML) → `data/condition_C2/nodes/`
* [ ] Build Condition C2 hierarchical embedding indices → `data/condition_C2/embeddings/`
* [ ] Save all 1,540 QA pairs → `questions/locomo_qa.jsonl`
* [ ] **Sanity check:** manually inspect 10 random prompts per condition (all 5)

### Phase 3 — Inference, Qwen2.5-72B

* [ ] Load Qwen2.5-72B with vLLM
* [ ] Run Condition D  → `results/condition_D/72B.jsonl`
* [ ] Run Condition B  → `results/condition_B/72B.jsonl`
* [ ] Run Condition C2 → `results/condition_C2/72B.jsonl`
* [ ] Run Condition C  → `results/condition_C/72B.jsonl`
* [ ] Run Condition A  → `results/condition_A/72B.jsonl`
* [ ] Verify 5 × 1,540 = **7,700 records** saved and non-empty
* [ ] Unload model

### Phase 4 — Inference, Qwen2.5-7B

* [ ] Repeat Phase 3 for Qwen2.5-7B → save to `results/condition_X/7B.jsonl`
* [ ] Verify 5 × 1,540 = **7,700 records** saved and non-empty

### Phase 5 — Evaluation

* [ ] Compute all metrics (F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT) for all 15,400 records
* [ ] Run statistical significance tests: C vs A, C2 vs B, C2 vs A, C2 vs C, B vs A — per category
* [ ] Generate main results table: categories × conditions × models
* [ ] Generate efficiency table: token counts, inference times, accuracy-per-1k-tokens
* [ ] **Identify failure cases:** questions where C2 << B or C << A — examine manually

### Phase 6 — Validation

* [ ] Cross-check Condition A results against published LoCoMo baseline (if available)
* [ ] Confirm Condition D results are plausibly low (near-floor for multi-hop)
* [ ] Check for refusal artifacts — ensure LLM is not refusing to answer questions
* [ ] **Spot-check 20 random predictions per condition manually for quality**

### Estimated Timeline

| Phase                         | Estimated Time          | Notes                                                     |
| ----------------------------- | ----------------------- | --------------------------------------------------------- |
| Environment setup             | 2–3 hours              | Mostly download time for model weights                    |
| Data preparation (A, B, C)    | 1–2 hours              | CPU-only, parallelizable                                  |
| Data preparation (C2 indices) | 1–2 hours              | Hierarchical embedding computation — CPU, parallelizable |
| Inference 72B × 5 conditions | 5–10 hours             | C2 similar speed to B; C and A slowest                    |
| Inference 7B × 5 conditions  | 1–3 hours              | Much faster than 72B                                      |
| Evaluation + analysis         | 2–3 hours              | CPU-only, script-based                                    |
| **Total**               | **~14–23 hours** | **Suitable for one overnight run**                  |

---

## 8. How to Use Results in the Paper

### 8.1 This Experiment's Role in the Paper

This experiment serves as an early validation section (Section 4.1 or a dedicated preliminary section). Its purpose is to validate the foundational premise before presenting the full EngramTrace system. The narrative arc:

1. **Motivation:** we claim structure helps (Introduction)
2. **Validation:** we verify this claim in isolation (this experiment)
3. **System:** we build on this validated insight to design EngramTrace (Methods)
4. **Full evaluation:** we evaluate EngramTrace as a complete system (later sections)

### 8.2 Suggested Section Header

```
Section 4.1: Does Structure Help? A Controlled Format Comparison

— or —

Section 4.1: Validating the Structural Representation Hypothesis
```

### 8.3 Claims to Make Based on Results

Frame claims precisely based on what the numbers actually show:

| If results show...                        | Claim to make                                                                                                                                                                                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C > A on multi-hop                        | "Hierarchical XML structure enables superior multi-hop reasoning compared to full linear context — structural cues help the LLM navigate complex associative queries more effectively than serial search through flat text." (H1 validated) |
| C2 > B on multi-hop at similar token cost | "Hierarchically-embedded XML retrieval outperforms flat chunk retrieval on multi-hop reasoning, confirming that ancestral context in node embeddings resolves the semantic isolation problem of standard RAG." (H2 validated)                |
| C2 ≈ A accuracy, C2 tokens << A tokens   | "Hierarchical XML retrieval matches full-context accuracy at X% lower token cost — resolving the accuracy-efficiency tradeoff that standard RAG cannot."                                                                                    |
| C ≈ A on single-hop                      | "On simple lookup questions, format has minimal effect, confirming the structural benefit is specific to reasoning tasks requiring cross-segment synthesis."                                                                                 |
| C2 > C on multi-hop                       | "Selective hierarchical retrieval further improves over full XML context by reducing noise from irrelevant turns while preserving structural context for retrieved nodes."                                                                   |

### 8.4 Limitations to Acknowledge

* This experiment uses rule-based XML conversion, which may produce less semantically rich structure than EngramTrace's LLM-driven atomization. The full system is expected to perform better than both C and C2.
* Condition C passes the full XML document in context — token counts are high and comparable to Condition A. This is intentional for isolating the structure variable, not a deployment recommendation.
* Condition C2 uses `alpha=0.7` as the hierarchical blending coefficient. This is the EngramTrace default but has not been ablated in this experiment — optimal values may differ by question type.
* Results are on conversational QA. Generalization to other knowledge domains (technical documentation, scientific literature) requires additional experiments.

---

> **Final note:** This experiment is a means to an end, not the paper's main contribution. Run it cleanly, report it honestly, and use it to motivate the EngramTrace system design. If results are weaker than expected in some categories, that is not a failure — it is information that will make your paper's claims more precise and more defensible.
>
