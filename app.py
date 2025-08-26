<script>
let extractedLOs = [];
let sessionId = null;
let matchingResults = null;
let selectedMatches = [];

document.addEventListener('DOMContentLoaded', function() {
    toggleInputMethod();
checkConnection();
});

async function checkConnection() {
try {
const response = await fetch('/api/health');
const data = await response.json();
const statusEl = document.getElementById('connectionStatus');
const textEl = document.getElementById('connectionText');

if (response.ok && data.status === 'healthy') {
statusEl.className = 'connection-status connected';
textEl.textContent = `✅ Backend connected! PDF: ${data.features?.pdf_processing ? 'Yes' : 'No'}, AI: ${data.features?.embeddings_available ? 'Yes' : 'No'}`;
} else {
    statusEl.className = 'connection-status disconnected';
textEl.textContent = '❌ Backend connection failed';
}
} catch (error) {
    const statusEl = document.getElementById('connectionStatus');
const textEl = document.getElementById('connectionText');
statusEl.className = 'connection-status disconnected';
textEl.textContent = '❌ Cannot connect to backend server';
console.error('Connection error:', error);
}
}

function toggleInputMethod() {
const method = document.getElementById('inputMethod').value;
document.getElementById('csvInput').classList.toggle('hidden', method !== 'csv');
document.getElementById('pdfInput').classList.toggle('hidden', method !== 'pdf');
document.getElementById('textInput').classList.toggle('hidden', method !== 'text');
}

async function handleFileSelect(event, type) {
const file = event.target.files[0];
if (!file) return;

const infoElement = document.getElementById(type + 'FileInfo');
infoElement.innerHTML = `🔄 Uploading: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

const formData = new FormData();
formData.append('file', file);

try {
const response = await fetch('/api/upload', {
method: 'POST',
body: formData
});

const data = await response.json();
console.log('Upload response:', data);

if (response.ok) {
    sessionId = data.session_id;
infoElement.innerHTML = `✅ Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) - Uploaded successfully`;
console.log('Session ID stored:', sessionId);
} else {
    throw new Error(data.error || 'Upload failed');
}
} catch (error) {
    console.error('Upload error:', error);
infoElement.innerHTML = `❌ Upload failed: ${error.message}`;
}
}

async function extractObjectives() {
const method = document.getElementById('inputMethod').value;
const extractBtn = document.getElementById('extractBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

extractBtn.disabled = true;
extractBtn.textContent = '🔄 Extracting...';
statusIndicator.classList.remove('hidden');
statusIndicator.className = 'status-indicator';
statusText.textContent = 'Processing learning objectives...';

try {
let requestData = {};

if (method === 'text') {
const textInput = document.getElementById('manualObjectives').value.trim();
if (!textInput) {
throw new Error('Please enter some learning objectives');
}
requestData.text = textInput;
} else if (method === 'csv' || method === 'pdf') {
if (!sessionId) {
throw new Error('Please upload a file first');
}
requestData.session_id = sessionId;
console.log('Sending session ID:', sessionId);
}

console.log('Extract request:', requestData);

const response = await fetch('/api/process/extract-los', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(requestData)
});

const data = await response.json();
console.log('Extract response:', data);

if (response.ok) {
    extractedLOs = data.learning_objectives;
displayLOsPreview();
statusIndicator.className = 'status-indicator status-success';
statusText.textContent = `✅ Successfully extracted ${data.count} learning objectives!`;

// Show step 2
document.getElementById('step2').style.display = 'block';
document.getElementById('step2').scrollIntoView({ behavior: 'smooth' });
} else {
    console.error('Extract error:', data);
throw new Error(data.error || 'Extraction failed');
}
} catch (error) {
    console.error('Extract error:', error);
statusIndicator.className = 'status-indicator status-error';
statusText.textContent = `❌ Error: ${error.message}`;
} finally {
    extractBtn.disabled = false;
extractBtn.textContent = '📖 Extract Learning Objectives';
}
}

function displayLOsPreview() {
const previewEl = document.getElementById('losPreview');
const listEl = document.getElementById('losPreviewList');
const countEl = document.getElementById('losCount');

listEl.innerHTML = '';
extractedLOs.forEach((lo, index) => {
    const li = document.createElement('li');
li.textContent = lo;
li.style.marginBottom = '5px';
listEl.appendChild(li);
});

countEl.textContent = extractedLOs.length;
previewEl.classList.remove('hidden');
}

async function startMatching() {
if (extractedLOs.length === 0) {
alert('Please extract learning objectives first.');
return;
}

const config = gatherConfiguration();
if (!config) return;

const matchBtn = document.getElementById('matchBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

matchBtn.disabled = true;
matchBtn.textContent = '🔄 Matching...';
statusIndicator.classList.remove('hidden');
statusIndicator.className = 'status-indicator';

try {
statusText.textContent = '🔍 Loading candidate cards...';
await new Promise(resolve => setTimeout(resolve, 500));

statusText.textContent = '🤖 Running AI matching algorithm...';

const response = await fetch('/api/process/match-cards', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(config)
});

const data = await response.json();

if (!response.ok) {
    throw new Error(data.error);
}

matchingResults = data;
displayResults();

statusIndicator.className = 'status-indicator status-success';
statusText.textContent = '✅ Card matching complete! Review results below.';

document.getElementById('step3').style.display = 'block';
document.getElementById('step3').scrollIntoView({ behavior: 'smooth' });

} catch (error) {
    statusIndicator.className = 'status-indicator status-error';
statusText.textContent = `❌ Error: ${error.message}`;
} finally {
    matchBtn.disabled = false;
matchBtn.textContent = '🚀 Start Card Matching';
}
}

function gatherConfiguration() {
const targetDeck = document.getElementById('targetDeck').value.trim();
if (!targetDeck) {
alert('Please enter a target deck name.');
return null;
}

const sourceDecks = [];
if (document.getElementById('ankingDeck').checked) {
sourceDecks.push('AnKing Step Deck');
}
if (document.getElementById('usuhs').checked) {
sourceDecks.push('USUHS v2.2');
}

const customDecks = document.getElementById('customDecks').value
                    .split(',')
                    .map(d => d.trim())
.filter(d => d);
sourceDecks.push(...customDecks);

if (sourceDecks.length === 0) {
alert('Please select at least one source deck.');
return null;
}

return {
    learning_objectives: extractedLOs,
    target_deck: targetDeck,
    source_decks: sourceDecks,
    custom_tag: document.getElementById('customTag').value.trim() || null,
    matching_mode: document.getElementById('matchingMode').value,
    auto_threshold: parseFloat(document.getElementById('autoThreshold').value) || null,
    multi_select: document.getElementById('multiSelect').checked,
    max_per_lo: 3,
    alpha: 0.6
};
}

function displayResults() {
displaySummaryStats();
displayMatchCards();

// Initialize selections from auto-selected matches
selectedMatches = [];
matchingResults.results.forEach(result => {
if (result.auto_selected && result.auto_selected.length > 0) {
    result.auto_selected.forEach(match => {
        selectedMatches.push({
            learning_objective: result.learning_objective,
            note_id: match.note_id,
            ...match
        });
});
}
});

updateSelectionUI();
}

function displaySummaryStats() {
const stats = matchingResults.stats;
const results = matchingResults.results;

const totalMatches = results.reduce((sum, r) => sum + r.matches.length, 0);
const autoSelected = results.reduce((sum, r) => sum + (r.auto_selected ? r.auto_selected.length : 0), 0);
const objectivesWithMatches = results.filter(r => r.matches.length > 0).length;

document.getElementById('summaryStats').innerHTML = `
<div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
<div style="font-size: 2rem; font-weight: bold; color: #667eea;">${stats.total_objectives}</div>
<div style="color: #666; margin-top: 5px;">Learning Objectives</div>
</div>
<div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
<div style="font-size: 2rem; font-weight: bold; color: #667eea;">${totalMatches}</div>
<div style="color: #666; margin-top: 5px;">Total Matches Found</div>
</div>
<div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
<div style="font-size: 2rem; font-weight: bold; color: #667eea;">${objectivesWithMatches}</div>
<div style="color: #666; margin-top: 5px;">Objectives with Matches</div>
</div>
<div style="background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; text-align: center;">
<div style="font-size: 2rem; font-weight: bold; color: #667eea;">${selectedMatches.length}</div>
<div style="color: #666; margin-top: 5px;">Currently Selected</div>
</div>
`;
}

function displayMatchCards() {
const container = document.getElementById('resultsContainer');
container.innerHTML = '';

matchingResults.results.forEach((result, resultIndex) => {
    const matchCard = document.createElement('div');
matchCard.style.cssText = 'background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; margin: 15px 0; transition: all 0.3s ease;';

matchCard.innerHTML = `
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                 <div style="font-weight: 600; color: #333; flex: 1; margin-right: 15px;">${result.learning_objective}</div>
                                                                                                                                        </div>
                                                                                                                                          <div id="candidates-${resultIndex}"></div>
                                                                                                                                                                                `;

const candidatesContainer = matchCard.querySelector(`#candidates-${resultIndex}`);

                                                    if (result.matches.length === 0) {
    candidatesContainer.innerHTML = '<div style="margin-top: 10px; font-size: 0.9rem; color: #666;">❌ No matches found for this objective</div>';
} else {
    result.matches.forEach((match, matchIndex) => {
    const candidateCard = createCandidateCard(match, resultIndex, matchIndex, result.learning_objective);
candidatesContainer.appendChild(candidateCard);
});
}

container.appendChild(matchCard);
});
}

function createCandidateCard(match, resultIndex, matchIndex, lo) {
    const card = document.createElement('div');
card.className = 'candidate-card';
card.dataset.resultIndex = resultIndex;
card.dataset.matchIndex = matchIndex;
card.dataset.noteId = match.note_id;
card.dataset.lo = lo;

card.style.cssText = `
background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; margin: 10px 0;
cursor: pointer; transition: all 0.3s ease;
`;

card.innerHTML = `
                 <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <div style="flex: 1;">
                                       <div style="display: flex; gap: 10px;">
                                                  <div style="font-size: 0.85rem; padding: 2px 8px; border-radius: 12px; background: #e9ecef;">
                                                             Combined: ${Math.round(match.combined_score * 100)}%
                                                                        </div>
                                                                          <div style="font-size: 0.85rem; padding: 2px 8px; border-radius: 12px; background: #e9ecef;">
                                                                                     Fuzzy: ${Math.round(match.fuzzy_score * 100)}%
                                                                                             </div>
${match.embedding_score > 0 ? `
                              <div style="font-size: 0.85rem; padding: 2px 8px; border-radius: 12px; background: #e9ecef;">
                                         AI: ${Math.round(match.embedding_score * 100)}%
                                              </div>
                                                ` : ''}
</div>
  </div>
    <input type="checkbox" style="transform: scale(1.3);" onchange="toggleCardSelection(this)">
                                                                   </div>
                                                                     <div style="background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; margin: 10px 0;">
                                                                                <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">
                                                                                           <strong>Model:</strong> ${match.model_name} |
                                                                                                                    <strong>Note ID:</strong> ${match.note_id}
                                                                                                                                               </div>
                                                                                                                                                 <div style="margin-bottom: 8px;">
                                                                                                                                                            <strong>Preview:</strong> ${match.preview_text}
                                                                                                                                                                                       </div>
${match.tags && match.tags.length > 0 ? `
                                        <div style="font-size: 0.85rem; color: #666;">
                                                   <strong>Tags:</strong> ${match.tags.join(', ')}
                                                                           </div>
                                                                             ` : ''}
</div>
  `;

card.addEventListener('click', function(e) {
if (e.target.type !== 'checkbox') {
    const checkbox = card.querySelector('input[type="checkbox"]');
checkbox.checked = !checkbox.checked;
toggleCardSelection(checkbox);
}
});

card.addEventListener('mouseenter', function() {
    card.style.borderColor = '#667eea';
card.style.boxShadow = '0 2px 10px rgba(102, 126, 234, 0.1)';
});

card.addEventListener('mouseleave', function() {
if (!card.classList.contains('selected')) {
    card.style.borderColor = '#e9ecef';
card.style.boxShadow = 'none';
}
});

return card;
}

function toggleCardSelection(checkbox) {
const card = checkbox.closest('.candidate-card');
const noteId = parseInt(card.dataset.noteId);
const lo = card.dataset.lo;

if (checkbox.checked) {
card.classList.add('selected');
card.style.borderColor = '#28a745';
card.style.backgroundColor = '#f8fff9';

// Add to selected matches if not already present
if (!selectedMatches.some(m => m.note_id === noteId && m.learning_objective === lo)) {
const resultIndex = parseInt(card.dataset.resultIndex);
const matchIndex = parseInt(card.dataset.matchIndex);
const match = matchingResults.results[resultIndex].matches[matchIndex];

selectedMatches.push({
learning_objective: lo,
note_id: noteId,
...match
});
}
} else {
    card.classList.remove('selected');
card.style.borderColor = '#e9ecef';
card.style.backgroundColor = 'white';

// Remove from selected matches
selectedMatches = selectedMatches.filter(m =>
!(m.note_id === noteId && m.learning_objective === lo)
);
}

updateSelectionUI();
}

function updateSelectionUI() {
// Update checkboxes to match selectedMatches
document.querySelectorAll('.candidate-card').forEach(card => {
    const noteId = parseInt(card.dataset.noteId);
const lo = card.dataset.lo;
const checkbox = card.querySelector('input[type="checkbox"]');
const isSelected = selectedMatches.some(m =>
                   m.note_id === noteId && m.learning_objective === lo
);

checkbox.checked = isSelected;
if (isSelected) {
    card.classList.add('selected');
card.style.borderColor = '#28a745';
card.style.backgroundColor = '#f8fff9';
} else {
    card.classList.remove('selected');
card.style.borderColor = '#e9ecef';
card.style.backgroundColor = 'white';
}
});

// Update summary stats
const summaryContainer = document.getElementById('summaryStats');
if (summaryContainer) {
const statCards = summaryContainer.querySelectorAll('div');
if (statCards.length >= 4) {
statCards[3].querySelector('div').textContent = selectedMatches.length;
}
}
}

function selectAll() {
selectedMatches = [];

matchingResults.results.forEach(result => {
if (result.matches && result.matches.length > 0) {
                                                 // Select top match for each objective
const topMatch = result.matches[0];
selectedMatches.push({
learning_objective: result.learning_objective,
note_id: topMatch.note_id,
...topMatch
});
}
});

updateSelectionUI();
}

function clearSelections() {
selectedMatches = [];
updateSelectionUI();
}

async function applyChanges() {
if (selectedMatches.length === 0) {
alert('Please select at least one card to apply changes.');
return;
}

const dryRun = document.getElementById('dryRun').checked;
const targetDeck = document.getElementById('targetDeck').value.trim();
const customTag = document.getElementById('customTag').value.trim();

const confirmMsg = dryRun
? `Preview: This would modify ${selectedMatches.length} cards. Continue?`
: `This will modify ${selectedMatches.length} cards in your Anki collection. Continue?`;

if (!confirm(confirmMsg)) return;

const applyBtn = document.getElementById('applyBtn');
const originalText = applyBtn.textContent;
applyBtn.disabled = true;
applyBtn.textContent = '🔄 Applying changes...';

try {
const response = await fetch('/api/apply-changes', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
    selected_matches: selectedMatches,
    target_deck: targetDeck,
    custom_tag: customTag || null,
    dry_run: dryRun
})
});

const data = await response.json();

if (response.ok) {
    alert(data.message);
} else {
    throw new Error(data.error);
}
} catch (error) {
    alert(`Failed to apply changes: ${error.message}`);
} finally {
    applyBtn.disabled = false;
applyBtn.textContent = originalText;
}
}

function exportResults() {
if (!matchingResults) {
alert('No results to export.');
return;
}

const exportData = {
results: matchingResults.results,
selected_matches: selectedMatches,
config: matchingResults.config,
timestamp: new Date().toISOString()
};

const blob = new Blob([JSON.stringify(exportData, null, 2)], {
    type: 'application/json'
});
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `decksurfer_results_${new Date().toISOString().slice(0,10)}.json`;
a.click();
URL.revokeObjectURL(url);

alert(`Results exported successfully!`);
}
</script>#!/usr/bin/env python3
# app.py - Flask backend for DeckSurfer (Final Working Version)
import os
import tempfile
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Heroku-friendly
PORT = int(os.environ.get('PORT', 5000))
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'decksurfer_uploads')
SESSION_FOLDER = os.path.join(tempfile.gettempdir(), 'decksurfer_sessions')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Try to import your existing modules
try:
    # Import fuzzy matching
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
    print("✅ Fuzzy matching available")
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ Fuzzy matching not available")

# Try to import embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
    print("✅ AI embeddings available")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ AI embeddings not available")

# Mock Anki functions for demo/remote use
def mock_anki_invoke(action: str, **params):
    """Mock AnkiConnect for demo purposes"""
    if action == "version":
        raise Exception("AnkiConnect not available - this is a remote server")
    elif action == "findNotes":
        # Return mock note IDs
        query = params.get('query', '')
        if 'AnKing' in query:
            return [1001, 1002, 1003, 1004, 1005]
        elif 'USUHS' in query:
            return [2001, 2002, 2003]
        return [9001, 9002]
    elif action == "notesInfo":
        # Return mock note data
        notes = params.get('notes', [])
        mock_notes = []
        for i, note_id in enumerate(notes):
            mock_notes.append({
                'noteId': note_id,
                'modelName': 'Cloze' if i % 2 == 0 else 'Basic',
                'tags': ['USUHS::Endocrine', 'AnKing::Step1'] if i % 3 == 0 else ['Medical'],
                'fields': {
                    'Front': f'Sample question about diabetes and insulin mechanism {i+1}',
                    'Back': f'Answer explaining pathophysiology and clinical significance {i+1}',
                    'Extra': f'Additional context about endocrine system {i+1}'
                }
            })
        return mock_notes
    return None

# Fuzzy matching functions
def safe_norm(s: str) -> str:
    """Normalize text for comparison"""
    return " ".join(s.lower().split())

def fuzzy_score(lo: str, card_text: str) -> float:
    """Calculate fuzzy similarity score"""
    if not FUZZY_AVAILABLE:
        # Simple fallback scoring
        lo_words = set(lo.lower().split())
        card_words = set(card_text.lower().split())
        intersection = len(lo_words.intersection(card_words))
        union = len(lo_words.union(card_words))
        return intersection / union if union > 0 else 0.0

    # Use rapidfuzz for better scoring
    a = fuzz.token_set_ratio(lo, card_text) / 100.0
    b = fuzz.partial_ratio(lo, card_text) / 100.0
    c = fuzz.token_sort_ratio(lo, card_text) / 100.0
    return 0.5 * a + 0.3 * b + 0.2 * c

def extract_note_text(note: dict) -> str:
    """Extract searchable text from note"""
    fields = note.get("fields", {})
    pieces = []
    for k, v in fields.items():
        if isinstance(v, dict):
            val = v.get("value", "")
        else:
            val = str(v)
        pieces.append(f"{k}: {val}")
    tags = " ".join(note.get("tags", []))
    return ((" ".join(pieces)) + " " + tags).strip()

class EmbeddingIndex:
    """AI similarity search using sentence transformers"""
    def __init__(self):
        if not EMBEDDINGS_AVAILABLE:
            self.model = None
            self.card_matrix = None
            self.cards = []
            return

        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.card_matrix = None
            self.cards = []
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
            self.model = None

    def fit(self, cards: List[dict]):
        if not self.model:
            return
        try:
            texts = [c.get("text", "") for c in cards]
            self.card_matrix = self.model.encode(texts, normalize_embeddings=True)
            self.cards = cards
            print(f"✅ Indexed {len(cards)} cards for similarity search")
        except Exception as e:
            print(f"⚠️ Failed to create embeddings: {e}")
            self.model = None

    def query(self, lo_text: str, top_k: int = 50):
        if not self.model or self.card_matrix is None:
            return []
        try:
            q = self.model.encode([lo_text], normalize_embeddings=True)[0]
            sims = self.card_matrix @ q
            idxs = np.argpartition(-sims, min(top_k, len(sims)-1))[:top_k]
            ranked = sorted([(int(i), float(sims[i])) for i in idxs], key=lambda x: -x[1])
            return ranked
        except Exception as e:
            print(f"⚠️ Embedding query failed: {e}")
            return []

# PDF processing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    if not PDF_AVAILABLE:
        raise ImportError("PyPDF2 not available")

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def clean_and_split_pdf_text(text: str) -> List[str]:
    """Turn raw PDF text into learning objectives"""
    import re

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Common patterns for learning objectives
    patterns = [
        r'(?:Learning\s+Objective|Objective|LO)[s]?[:\-\s]*([^\.]{20,200})',
        r'(?:Students?\s+will|By\s+the\s+end|Upon\s+completion)[^\.]{20,200}',
        r'(?:Understand|Describe|Explain|Identify|Analyze|Compare|Define)[^\.]{15,150}',
        r'(?:\d+\.|\*|\-|\•)\s*([^\.]{15,200})',
    ]

    objectives = []

    # Try pattern matching first
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if 10 < len(cleaned) < 300 and cleaned not in objectives:
                objectives.append(cleaned)

    # If no pattern matches, try splitting on common delimiters
    if len(objectives) < 3:
        candidates = re.split(r'(?:\n|\d+\.|•|\*|\-)\s*', text)
        for candidate in candidates:
            cleaned = candidate.strip()
            # Filter for reasonable length sentences that look like learning objectives
            if (20 < len(cleaned) < 250 and
                    any(word in cleaned.lower() for word in ['understand', 'describe', 'explain', 'identify', 'analyze', 'compare', 'define', 'discuss', 'evaluate', 'demonstrate']) and
                    cleaned not in objectives):
                objectives.append(cleaned)

    # Final cleanup and limit
    final_objectives = []
    for obj in objectives[:20]:  # Limit to 20 objectives
        # Remove common prefixes and clean up
        obj = re.sub(r'^(?:Learning\s+Objective[s]?[:\-\s]*|LO[:\-\s]*|\d+\.\s*)', '', obj, flags=re.IGNORECASE)
        obj = obj.strip()
        if obj and len(obj) > 10:
            final_objectives.append(obj)

    return final_objectives if final_objectives else ['No clear learning objectives found in PDF']

# Try to import PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
    print("✅ PDF processing available")
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF processing not available - install PyPDF2")

# Card matching and processing functions
def index_candidate_pool(decks: List[str], limit: Optional[int] = None, demo_mode: bool = True) -> List[dict]:
    """Build candidate card pool from specified decks"""
    if demo_mode:
        # Generate mock data for demo
        mock_cards = []
        card_templates = [
            {"front": "What hormone is deficient in Type 1 diabetes?", "back": "Insulin", "tags": ["USUHS::Endocrine", "AnKing::Step1"]},
            {"front": "DKA is characterized by what three findings?", "back": "Hyperglycemia, ketosis, acidosis", "tags": ["USUHS::Emergency", "AnKing::Step1"]},
            {"front": "Which cells produce insulin?", "back": "Beta cells of pancreatic islets", "tags": ["USUHS::Endocrine"]},
            {"front": "What is the primary defect in Type 2 diabetes?", "back": "Insulin resistance", "tags": ["USUHS::Endocrine", "AnKing::Step1"]},
            {"front": "What is the normal blood glucose range?", "back": "70-100 mg/dL fasting", "tags": ["USUHS::Lab Values"]},
            {"front": "What is HbA1c and what does it measure?", "back": "Glycated hemoglobin, measures average blood glucose over 2-3 months", "tags": ["USUHS::Lab Values", "AnKing::Step1"]},
            {"front": "What are the classic symptoms of diabetes?", "back": "Polyuria, polydipsia, polyphagia, weight loss", "tags": ["USUHS::Clinical"]},
            {"front": "What is the mechanism of metformin?", "back": "Decreases hepatic glucose production, increases insulin sensitivity", "tags": ["USUHS::Pharmacology", "AnKing::Step1"]},
        ]

        for i, template in enumerate(card_templates * 3):  # Duplicate for more examples
            note_id = 1000 + i
            mock_cards.append({
                'noteId': note_id,
                'modelName': 'Basic' if i % 2 == 0 else 'Cloze',
                'tags': template['tags'],
                'fields': {
                    'Front': template['front'],
                    'Back': template['back'],
                    'Extra': f"Additional clinical context for card {i+1}"
                },
                'text': f"Front: {template['front']} Back: {template['back']} {' '.join(template['tags'])}"
            })

        if limit:
            mock_cards = mock_cards[:limit]

        print(f"[Demo Mode] Generated {len(mock_cards)} mock cards")
        return mock_cards

    else:
        # Real AnkiConnect mode (for local installations)
        try:
            # This would work with real AnkiConnect
            query = " OR ".join([f'deck:"{d}*"' for d in decks])
            note_ids = mock_anki_invoke("findNotes", query=query)
            if limit:
                note_ids = note_ids[:limit]

            notes = mock_anki_invoke("notesInfo", notes=note_ids)
            pool = []
            for n in notes:
                text = extract_note_text(n)
                pool.append({
                    "noteId": n.get("noteId"),
                    "modelName": n.get("modelName"),
                    "tags": n.get("tags", []),
                    "fields": {k: v.get("value", "") if isinstance(v, dict) else str(v)
                               for k, v in n.get("fields", {}).items()},
                    "text": text
                })
            return pool
        except Exception as e:
            print(f"AnkiConnect failed, using demo mode: {e}")
            return index_candidate_pool(decks, limit, demo_mode=True)

def combined_top_candidates(
        lo: str,
        cards: List[dict],
        emb_index: Optional[EmbeddingIndex],
        k_from_emb: int = 50,
        k_final: int = 3,
        alpha: float = 0.6
) -> List[tuple]:
    """
    Get top matching cards using combined fuzzy + AI scoring
    Returns: List of (card, combined_score, fuzzy_score, embedding_score)
    """
    lo_norm = safe_norm(lo)

    # Get embedding candidates if available
    seed_idxs = []
    emb_scores_map = {}

    if emb_index and emb_index.model:
        emb_ranked = emb_index.query(lo_norm, top_k=k_from_emb)
        seed_idxs = [i for (i, _) in emb_ranked]
        emb_scores_map = {i: s for (i, s) in emb_ranked}
    else:
        seed_idxs = list(range(min(len(cards), k_from_emb)))  # Use first N cards

    # Score candidates
    scored = []
    for i in seed_idxs:
        if i >= len(cards):
            continue

        c = cards[i]
        fz = fuzzy_score(lo_norm, c.get("text", ""))
        em = emb_scores_map.get(i, 0.0)

        if emb_index and emb_index.model:
            combo = alpha * em + (1 - alpha) * fz
        else:
            combo = fz

        scored.append((i, combo, fz, em))

    # Sort and return top K
    scored.sort(key=lambda x: -x[1])
    top = scored[:k_final]
    return [(cards[i], combo, fz, em) for (i, combo, fz, em) in top]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_session(session_id: str, data: dict):
    """Save session data to file"""
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_path, 'w') as f:
        json.dump(data, f)

def load_session(session_id: str) -> Optional[dict]:
    """Load session data from file"""
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

# Inline HTML template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeckSurfer - Smart Anki Organization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; color: white; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .main-card { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .connection-status { display: flex; align-items: center; gap: 10px; padding: 10px 15px; border-radius: 8px; margin: 15px 0; }
        .connection-status.connected { background: #d4edda; color: #155724; }
        .connection-status.disconnected { background: #f8d7da; color: #721c24; }
        .step { margin-bottom: 40px; }
        .step-header { display: flex; align-items: center; margin-bottom: 20px; }
        .step-number { background: linear-gradient(135deg, #667eea, #764ba2); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 15px; }
        .step-title { font-size: 1.5rem; color: #333; }
        .form-group { margin-bottom: 25px; }
        label { display: block; font-weight: 600; margin-bottom: 8px; color: #555; }
        input, select, textarea { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 10px; font-size: 16px; transition: all 0.3s ease; }
        input:focus, select:focus, textarea:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
        .file-drop-zone { border: 3px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; background: #f8f9fa; transition: all 0.3s ease; cursor: pointer; }
        .file-drop-zone:hover, .file-drop-zone.dragover { border-color: #667eea; background: #f0f4ff; }
        .file-info { margin-top: 10px; font-size: 0.9rem; color: #666; }
        .btn { padding: 12px 24px; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; text-decoration: none; display: inline-block; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-primary { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3); }
        .hidden { display: none; }
        .los-preview { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 15px 0; max-height: 200px; overflow-y: auto; }
        .status-indicator { padding: 20px; border-radius: 10px; margin: 20px 0; font-weight: 600; }
        .status-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        @media (max-width: 768px) { .container { padding: 10px; } .main-card { padding: 20px; } .header h1 { font-size: 2rem; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏄‍♂️ DeckSurfer</h1>
            <p>Smart Anki deck organization for medical students</p>
        </div>
        <div class="main-card">
            <div id="connectionStatus" class="connection-status disconnected">
                <span>🔍</span>
                <span id="connectionText">Checking backend connection...</span>
            </div>
            <div class="step">
                <div class="step-header">
                    <div class="step-number">1</div>
                    <h2 class="step-title">Upload Learning Objectives</h2>
                </div>
                <div class="form-group">
                    <label>Choose your input method:</label>
                    <select id="inputMethod" onchange="toggleInputMethod()">
                        <option value="text">Manual Text Input</option>
                        <option value="csv">CSV File with Learning Objectives</option>
                        <option value="pdf">PDF Lecture (Auto-extract objectives)</option>
                    </select>
                </div>
                <div id="textInput" class="form-group">
                    <label>Enter Learning Objectives (one per line):</label>
                    <textarea id="manualObjectives" rows="8" placeholder="Enter each learning objective on a new line...

Example:
Describe the pathophysiology of diabetes mellitus
Explain the mechanism of action of insulin
Identify clinical signs of diabetic ketoacidosis"></textarea>
                </div>
                <div id="csvInput" class="form-group hidden">
                    <label>Upload CSV File:</label>
                    <div class="file-drop-zone" onclick="document.getElementById('csvFile').click()">
                        <div>📄 Drop your CSV file here or click to browse</div>
                        <div class="file-info">Should contain a column named 'Objective', 'LO', or 'Objectives'</div>
                    </div>
                    <input type="file" id="csvFile" accept=".csv" style="display: none;" onchange="handleFileSelect(event, 'csv')">
                    <div id="csvFileInfo" class="file-info"></div>
                </div>
                <div id="pdfInput" class="form-group hidden">
                    <label>Upload PDF Lecture:</label>
                    <div class="file-drop-zone" onclick="document.getElementById('pdfFile').click()">
                        <div>📚 Drop your lecture PDF here or click to browse</div>
                        <div class="file-info">We'll automatically extract learning objectives from the content</div>
                    </div>
                    <input type="file" id="pdfFile" accept=".pdf" style="display: none;" onchange="handleFileSelect(event, 'pdf')">
                    <div id="pdfFileInfo" class="file-info"></div>
                </div>
                <div id="losPreview" class="los-preview hidden">
                    <h4>📋 Extracted Learning Objectives:</h4>
                    <ol id="losPreviewList"></ol>
                    <div class="file-info"><span id="losCount">0</span> objectives found.</div>
                </div>
                <button class="btn btn-primary" onclick="extractObjectives()" id="extractBtn">📖 Extract Learning Objectives</button>
            </div>

            <!-- Step 2: Configure Settings -->
            <div class="step" id="step2" style="display: none;">
                <div class="step-header">
                    <div class="step-number">2</div>
                    <h2 class="step-title">Configure Matching Settings</h2>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div class="form-group">
                        <label>Target Deck Name:</label>
                        <input type="text" id="targetDeck" placeholder="e.g., USUHS::MS1::Endo::Lecture 07" value="">
                        <div class="file-info">Where matched cards will be moved</div>
                    </div>
                    <div class="form-group">
                        <label>Source Decks to Search:</label>
                        <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                            <input type="checkbox" id="ankingDeck" checked style="width: auto; transform: scale(1.2);">
                            <label for="ankingDeck" style="margin: 0;">AnKing Step Deck</label>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                            <input type="checkbox" id="usuhs" checked style="width: auto; transform: scale(1.2);">
                            <label for="usuhs" style="margin: 0;">USUHS v2.2</label>
                        </div>
                        <input type="text" id="customDecks" placeholder="Add custom deck names (comma-separated)">
                    </div>
                    <div class="form-group">
                        <label>Optional Tag to Add:</label>
                        <input type="text" id="customTag" placeholder="e.g., LO::Endo07">
                        <div class="file-info">Tags help organize your matched cards</div>
                    </div>
                    <div class="form-group">
                        <label>Matching Mode:</label>
                        <select id="matchingMode">
                            <option value="smart">Smart (AI + Fuzzy matching)</option>
                            <option value="fuzzy">Fuzzy text matching only</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Auto-approval threshold (0.0-1.0):</label>
                        <input type="number" id="autoThreshold" min="0" max="1" step="0.1" placeholder="0.8">
                        <div class="file-info">Automatically select matches above this confidence score</div>
                    </div>
                    <div class="form-group">
                        <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                            <input type="checkbox" id="multiSelect" style="width: auto; transform: scale(1.2);">
                            <label for="multiSelect" style="margin: 0;">Allow multiple cards per objective</label>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">
                            <input type="checkbox" id="dryRun" checked style="width: auto; transform: scale(1.2);">
                            <label for="dryRun" style="margin: 0;">Preview mode (don't modify Anki yet)</label>
                        </div>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="startMatching()" id="matchBtn">🚀 Start Card Matching</button>
            </div>

            <!-- Step 3: Results -->
            <div class="step" id="step3" style="display: none;">
                <div class="step-header">
                    <div class="step-number">3</div>
                    <h2 class="step-title">Review Matches & Apply Changes</h2>
                </div>
                <div id="summaryStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;"></div>
                <div id="resultsContainer"></div>
                <div style="margin-top: 30px; text-align: center; display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                    <button class="btn btn-primary" onclick="applyChanges()" id="applyBtn">✅ Apply Selected Changes</button>
                    <button class="btn" onclick="selectAll()" id="selectAllBtn" style="background: #28a745; color: white;">📋 Select All Matches</button>
                    <button class="btn" onclick="clearSelections()" id="clearBtn" style="background: #6c757d; color: white;">🗑️ Clear Selections</button>
                    <button class="btn" onclick="exportResults()" id="exportBtn" style="background: #17a2b8; color: white;">📥 Export Results</button>
                </div>
            </div>

            <div id="statusIndicator" class="status-indicator hidden">
                <div id="statusText">Processing...</div>
            </div>
        </div>
        <div class="main-card">
            <h3>📋 About DeckSurfer</h3>
            <p style="margin: 20px 0; line-height: 1.6;">
                This is a <strong>demo version</strong> of DeckSurfer deployed on Heroku. 
                It can extract and process learning objectives from text or CSV files.
                The full Anki integration requires a local installation with AnkiConnect.
            </p>
        </div>
    </div>
    <script>
        let extractedLOs = [];
        let sessionId = null;

        document.addEventListener('DOMContentLoaded', function() {
            toggleInputMethod();
            checkConnection();
        });

        async function checkConnection() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                const statusEl = document.getElementById('connectionStatus');
                const textEl = document.getElementById('connectionText');

                if (response.ok && data.status === 'healthy') {
                    statusEl.className = 'connection-status connected';
                    textEl.textContent = `✅ Backend connected successfully! (${data.environment || 'demo'} mode)`;
                } else {
                    statusEl.className = 'connection-status disconnected';
                    textEl.textContent = '❌ Backend connection failed';
                }
            } catch (error) {
                const statusEl = document.getElementById('connectionStatus');
                const textEl = document.getElementById('connectionText');
                statusEl.className = 'connection-status disconnected';
                textEl.textContent = '❌ Cannot connect to backend server';
                console.error('Connection error:', error);
            }
        }

        function toggleInputMethod() {
            const method = document.getElementById('inputMethod').value;
            document.getElementById('csvInput').classList.toggle('hidden', method !== 'csv');
            document.getElementById('pdfInput').classList.toggle('hidden', method !== 'pdf');
            document.getElementById('textInput').classList.toggle('hidden', method !== 'text');
        }

        async function handleFileSelect(event, type) {
            const file = event.target.files[0];
            if (!file) return;

            const infoElement = document.getElementById(type + 'FileInfo');
            infoElement.innerHTML = `🔄 Uploading: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                console.log('Upload response:', data);

                if (response.ok) {
                    sessionId = data.session_id;
                    infoElement.innerHTML = `✅ Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB) - Uploaded successfully`;
                    console.log('Session ID stored:', sessionId);
                } else {
                    throw new Error(data.error || 'Upload failed');
                }
            } catch (error) {
                console.error('Upload error:', error);
                infoElement.innerHTML = `❌ Upload failed: ${error.message}`;
            }
        }

        async function extractObjectives() {
            const method = document.getElementById('inputMethod').value;
            const extractBtn = document.getElementById('extractBtn');
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');

            extractBtn.disabled = true;
            extractBtn.textContent = '🔄 Extracting...';
            statusIndicator.classList.remove('hidden');
            statusIndicator.className = 'status-indicator';
            statusText.textContent = 'Processing learning objectives...';

            try {
                let requestData = {};

                if (method === 'text') {
                    const textInput = document.getElementById('manualObjectives').value.trim();
                    if (!textInput) {
                        throw new Error('Please enter some learning objectives');
                    }
                    requestData.text = textInput;
                } else if (method === 'csv' || method === 'pdf') {
                    if (!sessionId) {
                        throw new Error('Please upload a file first');
                    }
                    requestData.session_id = sessionId;
                    console.log('Sending session ID:', sessionId);
                }

                console.log('Extract request:', requestData);

                const response = await fetch('/api/process/extract-los', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });

                const data = await response.json();
                console.log('Extract response:', data);

                if (response.ok) {
                    extractedLOs = data.learning_objectives;
                    displayLOsPreview();
                    statusIndicator.className = 'status-indicator status-success';
                    statusText.textContent = `✅ Successfully extracted ${data.count} learning objectives!`;
                } else {
                    console.error('Extract error:', data);
                    throw new Error(data.error || 'Extraction failed');
                }
            } catch (error) {
                console.error('Extract error:', error);
                statusIndicator.className = 'status-indicator status-error';
                statusText.textContent = `❌ Error: ${error.message}`;
            } finally {
                extractBtn.disabled = false;
                extractBtn.textContent = '📖 Extract Learning Objectives';
            }
        }

        function displayLOsPreview() {
            const previewEl = document.getElementById('losPreview');
            const listEl = document.getElementById('losPreviewList');
            const countEl = document.getElementById('losCount');

            listEl.innerHTML = '';
            extractedLOs.forEach((lo, index) => {
                const li = document.createElement('li');
                li.textContent = lo;
                li.style.marginBottom = '5px';
                listEl.appendChild(li);
            });

            countEl.textContent = extractedLOs.length;
            previewEl.classList.remove('hidden');
        }
    </script>
</body>
</html>'''

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'environment': 'demo',
            'message': 'DeckSurfer demo backend is running!',
            'features': {
                'text_processing': True,
                'csv_processing': True,
                'pdf_processing': False,
                'anki_integration': False
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads (CSV only in demo)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use CSV or PDF files only.'}), 400

        # Generate unique session ID
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")

        # Save file
        file.save(file_path)

        # Store session data in file
        session_data = {
            'file_path': file_path,
            'filename': filename,
            'file_type': filename.rsplit('.', 1)[1].lower(),
            'upload_time': pd.Timestamp.now().isoformat()
        }
        save_session(session_id, session_data)

        logger.info(f"File uploaded: {filename}, Session ID: {session_id}")

        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process/extract-los', methods=['POST'])
def extract_learning_objectives():
    """Extract learning objectives from uploaded file or text"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        los = []

        if 'session_id' in data:
            # Process uploaded file
            session_id = data['session_id']
            logger.info(f"Looking for session: {session_id}")

            session = load_session(session_id)
            if not session:
                return jsonify({
                    'error': f'Session not found: {session_id}. Please upload the file again.'
                }), 400

            file_path = session['file_path']
            file_type = session['file_type']

            logger.info(f"Processing file: {file_path}, type: {file_type}")

            # Check if file exists
            if not os.path.exists(file_path):
                return jsonify({'error': f'File not found. Please upload again.'}), 400

            if file_type == 'csv':
                try:
                    df = pd.read_csv(file_path)
                    logger.info(f"CSV columns: {list(df.columns)}")

                    col = None
                    for candidate in ["Objective", "objective", "LO", "lo", "Objectives", "objectives"]:
                        if candidate in df.columns:
                            col = candidate
                            break

                    if not col:
                        return jsonify({
                            'error': 'CSV must contain a column named "Objective", "LO", or "Objectives"',
                            'available_columns': list(df.columns)
                        }), 400

                    los = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
                    logger.info(f"Extracted {len(los)} objectives from CSV")

                except Exception as csv_error:
                    logger.error(f"CSV processing error: {csv_error}")
                    return jsonify({'error': f'CSV processing failed: {str(csv_error)}'}), 500

            elif file_type == 'pdf':
                if not PDF_AVAILABLE:
                    return jsonify({'error': 'PDF processing not available in this deployment'}), 500

                try:
                    text = extract_text_from_pdf(file_path)
                    los = clean_and_split_pdf_text(text)
                    logger.info(f"Extracted {len(los)} objectives from PDF")

                except Exception as pdf_error:
                    logger.error(f"PDF processing error: {pdf_error}")
                    return jsonify({'error': f'PDF processing failed: {str(pdf_error)}'}), 500

            else:
                return jsonify({'error': f'Unsupported file type: {file_type}'}), 400

        elif 'text' in data:
            # Process manual text input
            text_input = data['text'].strip()
            los = [line.strip() for line in text_input.split('\n') if line.strip()]
            logger.info(f"Extracted {len(los)} objectives from text input")

        else:
            return jsonify({'error': 'No valid input method provided'}), 400

        if not los:
            return jsonify({'error': 'No learning objectives could be extracted'}), 400

        # Update session with LOs if applicable
        if 'session_id' in data:
            session = load_session(data['session_id'])
            if session:
                session['learning_objectives'] = los
                save_session(data['session_id'], session)

@app.route('/api/process/match-cards', methods=['POST'])
def match_cards_to_objectives():
    """Main processing endpoint - match cards to learning objectives"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No configuration provided'}), 400

        # Extract configuration
        config = {
            'learning_objectives': data.get('learning_objectives', []),
            'target_deck': data.get('target_deck', ''),
            'source_decks': data.get('source_decks', ['AnKing Step Deck', 'USUHS v2.2']),
            'custom_tag': data.get('custom_tag'),
            'matching_mode': data.get('matching_mode', 'smart'),
            'auto_threshold': data.get('auto_threshold'),
            'multi_select': data.get('multi_select', False),
            'max_per_lo': data.get('max_per_lo', 3),
            'alpha': data.get('alpha', 0.6)
        }

        if not config['learning_objectives']:
            return jsonify({'error': 'No learning objectives provided'}), 400

        if not config['target_deck']:
            return jsonify({'error': 'Target deck is required'}), 400

        # Build candidate pool (demo mode for Heroku deployment)
        logger.info(f"Building candidate pool from decks: {config['source_decks']}")
        pool = index_candidate_pool(
            decks=config['source_decks'],
            demo_mode=True  # Always use demo mode for Heroku
        )

        if not pool:
            return jsonify({'error': 'No cards found in specified decks'}), 400

        logger.info(f"Found {len(pool)} candidate cards")

        # Initialize embedding index if requested and available
        emb_index = None
        if config['matching_mode'] == 'smart' and EMBEDDINGS_AVAILABLE:
            logger.info("Building AI embedding index...")
            emb_index = EmbeddingIndex()
            emb_index.fit(pool)

        # Process each learning objective
        results = []

        for i, lo in enumerate(config['learning_objectives']):
            logger.info(f"Processing LO {i+1}/{len(config['learning_objectives'])}: {lo[:50]}...")

            # Get top candidates
            k_final = 10 if config['multi_select'] else 3
            candidates = combined_top_candidates(
                lo=lo,
                cards=pool,
                emb_index=emb_index,
                k_from_emb=80,
                k_final=k_final,
                alpha=config['alpha']
            )

            if not candidates:
                results.append({
                    'learning_objective': lo,
                    'matches': [],
                    'auto_selected': []
                })
                continue

            # Process candidates
            matches = []
            auto_selected = []

            for card, combo_score, fuzzy_score, emb_score in candidates:
                # Extract preview text
                preview_text = extract_preview_text(card)

                match_data = {
                    'note_id': card['noteId'],
                    'model_name': card.get('modelName', ''),
                    'tags': card.get('tags', []),
                    'fields': card.get('fields', {}),
                    'combined_score': round(combo_score, 3),
                    'fuzzy_score': round(fuzzy_score, 3),
                    'embedding_score': round(emb_score, 3),
                    'preview_text': preview_text
                }

                matches.append(match_data)

                # Auto-select if above threshold
                if (config['auto_threshold'] and
                        combo_score >= config['auto_threshold'] and
                        len(auto_selected) < config['max_per_lo']):
                    auto_selected.append(match_data)

            results.append({
                'learning_objective': lo,
                'matches': matches,
                'auto_selected': auto_selected
            })

        return jsonify({
            'results': results,
            'config': config,
            'stats': {
                'total_objectives': len(config['learning_objectives']),
                'total_candidates': len(pool),
                'embeddings_used': emb_index is not None and emb_index.model is not None,
                'demo_mode': True
            },
            'message': 'Card matching completed successfully!'
        })

    except Exception as e:
        logger.error(f"Card matching failed: {e}")
        return jsonify({'error': f'Card matching failed: {str(e)}'}), 500

def extract_preview_text(card: dict) -> str:
    """Extract a readable preview from card fields"""
    fields = card.get('fields', {})

    # Priority order for fields to show
    field_priority = ['Front', 'Text', 'Back', 'Question', 'Answer']

    for field_name in field_priority:
        if field_name in fields and fields[field_name]:
            text = str(fields[field_name]).strip()
            # Clean HTML tags if present
            import re
            text = re.sub(r'<[^>]+>', '', text)
            return text[:200]  # Truncate for preview

    # Fallback: use any available field
    for field_name, content in fields.items():
        if content and str(content).strip():
            import re
            text = re.sub(r'<[^>]+>', '', str(content).strip())
            return text[:200]

    return "No preview available"

@app.route('/api/apply-changes', methods=['POST'])
def apply_changes_to_anki():
    """Apply the selected matches to Anki (demo mode)"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        selected_matches = data.get('selected_matches', [])
        target_deck = data.get('target_deck')
        custom_tag = data.get('custom_tag')
        dry_run = data.get('dry_run', True)

        if not selected_matches:
            return jsonify({'error': 'No matches selected'}), 400

        if not target_deck:
            return jsonify({'error': 'Target deck is required'}), 400

        if dry_run:
            return jsonify({
                'message': f'✅ Dry run completed! Would modify {len(selected_matches)} cards',
                'details': {
                    'cards_to_modify': len(selected_matches),
                    'target_deck': target_deck,
                    'tag_to_add': custom_tag,
                    'actions': ['Unsuspend cards', 'Move to target deck', 'Add custom tag']
                },
                'demo_note': 'This is a demo deployment. For real Anki integration, run locally with AnkiConnect.'
            })

        # In demo mode, we can't actually modify Anki
        return jsonify({
            'message': '⚠️ Demo mode: Cannot modify Anki from remote server',
            'suggestion': 'To apply changes to Anki, please run DeckSurfer locally with AnkiConnect installed.',
            'would_modify': len(selected_matches)
        })

    except Exception as e:
        logger.error(f"Failed to apply changes: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 50MB)'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# ==================== STARTUP ====================

if __name__ == '__main__':
    print("🏄‍♂️ Starting DeckSurfer Demo Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Session folder: {SESSION_FOLDER}")
    print(f"Port: {PORT}")

    app.run(host='0.0.0.0', port=PORT, debug=False)