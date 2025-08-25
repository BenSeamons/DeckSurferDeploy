#!/usr/bin/env python3
# lo2anki.py
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import requests
from rapidfuzz import fuzz, process
from typing import Optional, List, Dict, Any

# --- Optional embeddings (sentence-transformers) ---
_EMBEDDINGS_OK = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_OK = True
except Exception:
    _EMBEDDINGS_OK = False

ANKI_CONNECT_URL = "http://127.0.0.1:8765"

# ---------- AnkiConnect helpers ----------
def anki_invoke(action: str, **params) -> Any:
    """
       Core helper for talking to AnkiConnect.

       - Builds a JSON payload with:
           * action: the API command to run (e.g., "findNotes", "notesInfo", "changeDeck")
           * version: protocol version (always 6 for AnkiConnect)
           * params: arguments for that action
       - Sends the payload to AnkiConnect's HTTP server (localhost:8765).
       - Parses the JSON reply.
       - Returns the "result" if successful, or exits if an error occurs.

       This is the universal bridge between Python and Anki.
       """
    payload = {"action": action, "version": 6, "params": params}
    try:
        r = requests.post(ANKI_CONNECT_URL, json=payload, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[AnkiConnect] HTTP error for action '{action}': {e}")
        sys.exit(1)
    data = r.json()
    if data.get("error"):
        print(f"[AnkiConnect] Error for action '{action}': {data['error']}")
        sys.exit(1)
    return data.get("result")

def find_notes(query: str) -> List[int]:
    """
       Wrapper around anki_invoke("findNotes").

       - Input: an Anki search query string (same syntax as Anki's browser, e.g. "deck:AnKing").
       - Sends that query to AnkiConnect.
       - Returns a list of matching note IDs (unique integers).
         Example: [1598471234567, 1598471234568]
       - Note: this only gives you IDs, not the actual card text.
       """
    return anki_invoke("findNotes", query=query) or []

def notes_info(note_ids: List[int]) -> List[Dict[str, Any]]:
    """
        Wrapper around anki_invoke("notesInfo").

        - Input: list of note IDs (from find_notes).
        - Fetches detailed info for each note in batches of 500 (to avoid payload size issues).
        - Output: a list of dicts, one per note, with fields like:
            * noteId
            * modelName (e.g., "Cloze", "Basic")
            * tags (list of strings)
            * fields (dict of field contents, e.g. Front, Back, Extra, Text)
        - This is how we go from raw IDs → actual readable note content
          so we can run fuzzy/semantic matching later.
        """

    # Batch to avoid payload limits

    out = []
    BATCH = 500
    for i in range(0, len(note_ids), BATCH):
        chunk = note_ids[i:i+BATCH]
        res = anki_invoke("notesInfo", notes=chunk) or []
        out.extend(res)
    return out

def set_suspended(note_ids: List[int], suspended: bool) -> None:
    # AnkiConnect only suspends cards, not notes; we’ll map note->cards first
    # Get card IDs per note:
    '''
    This basically takes the notes (which is like all the data about the
    card) and then returns the actual card ID because apparently you can't
    supsend a note? Which make no sense but apparently this is essential
    they do this so you can set the "card"'s value as supended/unsuspended.
    Within the actual note itself
    '''

    card_ids = []
    for nid in note_ids:
        cids = anki_invoke("findCards", query=f"nid:{nid}") or []
        card_ids.extend(cids)
    if not card_ids:
        return
    anki_invoke("setSuspended", cards=card_ids, suspend=suspended)

def change_deck(note_ids: List[int], deck_name: str) -> None:
    # Same: need card IDs
    '''
    Pretty self explanatory-changes them from source deck to our own desired
    deck
    '''
    card_ids = []
    for nid in note_ids:
        cids = anki_invoke("findCards", query=f"nid:{nid}") or []
        card_ids.extend(cids)
    if not card_ids:
        return
    anki_invoke("changeDeck", cards=card_ids, deck=deck_name)

def add_tag(note_ids: List[int], tag: str) -> None:
    #Self explanatory again
    if not note_ids:
        return
    anki_invoke("addTags", notes=note_ids, tags=tag)

# ---------- Data extraction ----------
def build_deck_query(decks: List[str], extra: Optional[str]=None) -> str:
    base = " OR ".join([f'deck:"{d}*"' for d in decks])
    return f"({base}) {extra if extra else ''}".strip()


def extract_note_text(note: Dict[str, Any]) -> str:
    # Concatenate common fields (Front/Back/Text/Extra/etc.)
    fields = note.get("fields", {})
    pieces = []
    for k, v in fields.items():
        # v is like {"value": "...", "order": 0}
        val = v.get("value", "")
        pieces.append(f"{k}: {val}")
    tags = " ".join(note.get("tags", []))
    return ((" ".join(pieces)) + " " + tags).strip()

def index_candidate_pool(decks: List[str], limit: Optional[int] = None, extra_query: Optional[str]=None) -> List[Dict[str, Any]]:
    query = build_deck_query(decks, extra=extra_query)
    note_ids = find_notes(query)
    if limit:
        note_ids = note_ids[:limit]
    print(f"[Index] Found {len(note_ids)} notes in selected decks...")
    notes = notes_info(note_ids)
    pool = []
    for n in notes:
        text = extract_note_text(n)
        # Keep minimal info
        pool.append({
            "noteId": n.get("noteId"),
            "modelName": n.get("modelName"),
            "tags": n.get("tags", []),
            "fields": {k: v.get("value", "") for k, v in n.get("fields", {}).items()},
            "text": text
        })
    return pool

# ---------- Scoring ----------
def safe_norm(s: str) -> str:
    return " ".join(s.lower().split())

def fuzzy_score(lo: str, card_text: str) -> float:
    # Blend partial/token sort/WRatio-ish scores
    a = fuzz.token_set_ratio(lo, card_text) / 100.0
    b = fuzz.partial_ratio(lo, card_text) / 100.0
    c = fuzz.token_sort_ratio(lo, card_text) / 100.0
    return 0.5 * a + 0.3 * b + 0.2 * c
    #consider adjusting these scores. a=how much do the words match each other
    #b=looks for shorter strings in longer ones (may need to increase)
    #c=sorts words alphabetically and compares (may need to decrease)

class EmbeddingIndex:
    def __init__(self):
        if not _EMBEDDINGS_OK:
            self.model = None
            return
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.card_matrix = None  # np.ndarray
        self.cards = []          # list of dicts

    def fit(self, cards: List[Dict[str, Any]]):
        if not self.model:
            return
        texts = [c["text"] for c in cards]
        self.card_matrix = self.model.encode(texts, normalize_embeddings=True)
        self.cards = cards

    def query(self, lo_text: str, top_k: int = 50) -> List[Tuple[int, float]]:
        if not self.model or self.card_matrix is None:
            return []
        q = self.model.encode([lo_text], normalize_embeddings=True)[0]
        sims = self.card_matrix @ q  # cosine since normalized
        # top_k indices
        idxs = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
        ranked = sorted([(int(i), float(sims[i])) for i in idxs], key=lambda x: -x[1])
        return ranked

def combined_top_candidates(
    lo: str,
    cards: List[Dict[str, Any]],
    emb_index: Optional[EmbeddingIndex],
    k_from_emb: int = 50,
    k_final: int = 3,
    alpha: float = 0.6
) -> List[Tuple[Dict[str, Any], float, float, float]]:
    """
    Returns list of tuples: (card, combined_score, fuzzy, emb)
    alpha weights embeddings; (1-alpha) weights fuzzy.
    If embeddings unavailable, falls back to fuzzy only.
    """
    lo_norm = safe_norm(lo)
    # Seed list: either emb top K or all
    seed_idxs: List[int]
    emb_scores_map = {}
    if emb_index and emb_index.model:
        emb_ranked = emb_index.query(lo_norm, top_k=k_from_emb)
        seed_idxs = [i for (i, _) in emb_ranked]
        emb_scores_map = {i: s for (i, s) in emb_ranked}
    else:
        seed_idxs = list(range(len(cards)))  # brute-force fuzzy
    # Score fuzzy on seed set
    scored = []
    for i in seed_idxs:
        c = cards[i]
        fz = fuzzy_score(lo_norm, c["text"])
        em = emb_scores_map.get(i, 0.0)
        if emb_index and emb_index.model:
            combo = alpha * em + (1 - alpha) * fz
        else:
            combo = fz
        scored.append((i, combo, fz, em))
    scored.sort(key=lambda x: -x[1])
    top = scored[:k_final]
    return [(cards[i], combo, fz, em) for (i, combo, fz, em) in top]

# ---------- CLI flow ----------
def print_card_preview(card: Dict[str, Any], idx: int, score: float, fz: float, em: float):
    """Print a short preview of a candidate card."""
    print("=" * 60)
    print(f"[{idx}] noteId={card['noteId']} | combined={score:.3f} | fuzzy={fz:.3f} | emb={em:.3f}")

    # Grab a small snippet from Front/Text/Back if available
    for key in ["Front", "Text", "Back"]:
        if key in card["fields"] and card["fields"][key]:
            snippet = card["fields"][key].strip().replace("\n", " ")
            if len(snippet) > 120:  # shorten to ~120 chars
                snippet = snippet[:120] + " ..."
            print(f"{key}: {snippet}")
            break  # only show first available field

    print("=" * 60)


def _parse_multi_selection(s: str, k: int) -> List[int]:
    """
    Parse inputs like '1,3' or '1-3' or '2,4-5' into 0-based indices.
    Bounds check against k (number of candidates shown).
    """
    picks = set()
    for chunk in s.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                for j in range(min(start, end), max(start, end)+1):
                    if 1 <= j <= k:
                        picks.add(j-1)
        elif chunk.isdigit():
            j = int(chunk)
            if 1 <= j <= k:
                picks.add(j-1)
    return sorted(picks)

def interactive_pick_multi(lo_text: str, topk: List[Tuple[Dict[str, Any], float, float, float]], allow_multi: bool) -> List[int]:
    """
    Show top candidates and return a list of chosen indices (0-based).
    In single mode returns either [] (skip) or [idx].
    In multi mode returns [] (skip) or multiple indices.
    """
    print(f"\nLO: {lo_text}")
    for i, (card, combo, fz, em) in enumerate(topk, start=1):
        print_card_preview(card, idx=i, score=combo, fz=fz, em=em)

    prompt = "Pick [1..{k} / ranges like 1-3 / comma list], (s)kip, (q)uit: ".format(k=len(topk))
    while True:
        choice = input(prompt).strip().lower()
        if choice == "s":
            return []
        if choice == "q":
            print("Exiting...")
            sys.exit(0)
        if allow_multi:
            picks = _parse_multi_selection(choice, k=len(topk))
            if picks:
                return picks
        else:
            if choice in {"1","2","3"} and int(choice) <= len(topk):
                return [int(choice)-1]
        print("Invalid input. Try again.")

def enforce_diversity(chosen: List[Tuple[Dict[str,Any], float, float, float]],
                      mode: str) -> List[Tuple[Dict[str,Any], float, float, float]]:
    """
    Keep first item; then only add candidates that differ per 'mode'.
    mode='model'  => unique modelName
    mode='tags'   => require disjoint tag sets (loose check)
    mode='none'   => no filtering
    """
    if mode == "none" or not chosen:
        return chosen
    kept = []
    seen_models = set()
    seen_tag_sets = []
    for card, combo, fz, em in [c for c in chosen]:
        if mode == "model":
            m = (card.get("modelName") or "")
            if m in seen_models:
                continue
            seen_models.add(m)
            kept.append((card, combo, fz, em))
        elif mode == "tags":
            t = set(card.get("tags", []))
            # simple overlap gate: skip if overlaps heavily with any kept
            if any(len(t & set(k[0].get("tags", []))) >= 2 for k in kept):
                continue
            kept.append((card, combo, fz, em))
    return kept



def main():
    parser = argparse.ArgumentParser(description="Match Lecture Objectives to Anki Cards (top-3 + skip) and move/unsuspend via AnkiConnect.")
    parser.add_argument("--los", required=True, help="CSV file with 'Objective' (or 'LO') column.")
    parser.add_argument("--target-deck", required=True, help="Deck to move selected cards into (e.g., 'USUHS::MS1::Endo::Lecture 07').")
    parser.add_argument("--decks", nargs="+", default=["AnKing Step Deck", "USUHS v2.2"], help="Deck roots to search (default: AnKing, USUHS v2.2).")
    parser.add_argument("--limit-index", type=int, default=None, help="Optional cap on indexed notes for testing.")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable semantic rerank (use fuzzy only).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for embeddings in combined score (0..1).")
    parser.add_argument("--auto-approve-threshold", type=float, default=None, help="If set, auto-accept top match when combined score ≥ threshold.")
    parser.add_argument("--tag", default=None, help="Optional tag to add to selected notes (e.g., 'LO::Endo07').")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify Anki; just print choices.")
    parser.add_argument("--multi", action="store_true", help="Allow selecting multiple cards per LO (comma- or range-based input).")
    parser.add_argument("--max-per-lo", type=int, default=3,help="Maximum number of cards to include per LO in multi mode (default: 3).")
    parser.add_argument("--min-combo", type=float, default=None,help="If set, auto-include any candidate with combined score >= this value (respects --max-per-lo).")
    parser.add_argument("--diversity", choices=["none", "model", "tags"], default="none",help="Diversity constraint among selected cards: none | model | tags.")
    parser.add_argument("--query", default=None,help="Extra AnkiBrowser query to AND with deck filters (e.g., 'is:suspended tag:\"USUHS::*Endo*\"').")
    parser.add_argument("--auto-only", action="store_true", help="Auto-select by thresholds only (never prompt user).")

    args = parser.parse_args()

    # Load LOs
    df = pd.read_csv(args.los)
    col = None
    for candidate in ["Objective", "objective", "LO", "lo", "Objectives"]:
        if candidate in df.columns:
            col = candidate
            break
    if not col:
        print("CSV must contain a column named 'Objective' or 'LO'.")
        sys.exit(1)
    los = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    if not los:
        print("No lecture objectives found.")
        sys.exit(1)

    print(f"[Config] Target deck: {args.target_deck}")
    print(f"[Config] Search decks: {args.decks}")
    print(f"[Config] Embeddings: {'ON' if _EMBEDDINGS_OK and not args.no_embeddings else 'OFF'}")
    if args.auto_approve_threshold:
        print(f"[Config] Auto-approve threshold: {args.auto_approve_threshold:.2f}")
    if args.dry_run:
        print("[Config] DRY RUN (no changes will be made to Anki).")

    # Build candidate pool
    pool = index_candidate_pool(args.decks, limit=args.limit_index, extra_query=args.query)
    if not pool:
        print("No cards found in the specified decks.")
        sys.exit(1)

    # Fit embedding index
    emb_index = None
    if _EMBEDDINGS_OK and not args.no_embeddings:
        emb_index = EmbeddingIndex()
        print("[Embeddings] Building vector index (first run can take ~30–90s depending on pool size)...")
        emb_index.fit(pool)

    accepted_note_ids: List[int] = []
    results_log = []

    for lo in los:
        K_FINAL = 10 if args.multi else 3
        topk = combined_top_candidates(lo, pool, emb_index, k_from_emb=80, k_final=K_FINAL, alpha=args.alpha)
        if not topk:
            print(f"\nLO: {lo}\nNo candidates found.")
            continue

        selected_idxs: List[int] = []

        # Auto-include by threshold (if requested)
        if args.min_combo is not None:
            auto_idxs = [i for i, (_c, combo, _fz, _em) in enumerate(topk) if combo >= args.min_combo]
            if auto_idxs:
                auto_idxs = auto_idxs[:args.max_per_lo] if args.multi else auto_idxs[:1]
                selected_idxs.extend(auto_idxs)

        # Decide whether to prompt the user
        if not args.auto_only:
            remaining_slots = (args.max_per_lo - len(selected_idxs)) if args.multi else (0 if selected_idxs else 1)
            if remaining_slots > 0:
                picks = interactive_pick_multi(lo, topk, allow_multi=args.multi)
                if args.multi:
                    selected_idxs = sorted(set(selected_idxs) | set(picks))[:args.max_per_lo]
                else:
                    selected_idxs = picks  # [] (skip) or [idx]

        # Apply diversity constraint if requested
        if selected_idxs and args.diversity != "none":
            selected_tuples = [topk[i] for i in selected_idxs]
            diversified = enforce_diversity(selected_tuples, mode=args.diversity)
            # map back to indices present in topk
            selected_idxs = [topk.index(t) for t in diversified]

        if not selected_idxs:
            results_log.append({"LO": lo, "decision": "skipped"})
            continue

        # Perform actions for all chosen notes
        chosen_nids = []
        for i in selected_idxs:
            chosen_card, combo, fz, em = topk[i]
            nid = chosen_card["noteId"]
            chosen_nids.append(nid)
            results_log.append({
                "LO": lo,
                "decision": "accepted",
                "noteId": nid,
                "combined": round(combo, 3),
                "fuzzy": round(fz, 3),
                "emb": round(em, 3)
            })

        if not args.dry_run:
            try:
                #set_suspended(chosen_nids, suspended=False)
                change_deck(chosen_nids, args.target_deck)
                if args.tag:
                    add_tag(chosen_nids, args.tag)
                print(f"[Anki] Unsuspended + moved {len(chosen_nids)} notes → '{args.target_deck}'"
                      + (f" + tagged '{args.tag}'" if args.tag else ""))
            except SystemExit:
                raise
            except Exception as e:
                print(f"[Anki] Failed to modify notes {chosen_nids}: {e}")

    # Summary
    print("\n" + "#" * 80)
    print("Summary")
    print(f"Accepted: {sum(1 for r in results_log if r['decision']=='accepted')}")
    print(f"Skipped:  {sum(1 for r in results_log if r['decision']=='skipped')}")
    out = Path("lo2anki_results.csv")
    pd.DataFrame(results_log).to_csv(out, index=False)
    print(f"Saved log → {out.resolve()}")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()