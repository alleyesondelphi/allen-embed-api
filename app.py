"""
Allen Case Search API — Embedding microservice for Railway.

Receives a search query, embeds it with the local model,
queries Supabase pgvector, and returns results.
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import requests as http_requests

app = Flask(__name__)
CORS(app)

# Load model at startup (cached after first download)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

# Supabase config from environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "allen-embed-api",
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
    })


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = data["query"]
    top_k = data.get("top_k", 5)
    threshold = data.get("threshold", 0.3)

    if not SUPABASE_URL or not SUPABASE_KEY:
        return jsonify({"error": "Supabase not configured"}), 500

    # Embed the query
    embedding = model.encode([query])[0].tolist()

    # Call Supabase RPC function
    rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_chunks"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query_embedding": embedding,
        "match_threshold": threshold,
        "match_count": top_k,
    }

    resp = http_requests.post(rpc_url, headers=headers, json=payload, timeout=10)

    if resp.status_code != 200:
        return jsonify({"error": f"Supabase error: {resp.status_code}", "detail": resp.text[:300]}), 502

    results = resp.json()

    # Format results
    formatted = []
    for r in results:
        formatted.append({
            "content": r.get("content", ""),
            "filename": r.get("filename", ""),
            "drive_id": r.get("drive_id", ""),
            "entry_date": r.get("entry_date", ""),
            "entry_header": r.get("entry_header", ""),
            "month_section": r.get("month_section", ""),
            "chunk_index": r.get("chunk_index", 0),
            "total_chunks": r.get("total_chunks", 1),
            "similarity": round(r.get("similarity", 0), 4),
            "drive_link": f"https://drive.google.com/file/d/{r.get('drive_id', '')}/view"
                         if r.get("drive_id") else "",
        })

    return jsonify({
        "query": query,
        "result_count": len(formatted),
        "results": formatted,
    })


@app.route("/embed", methods=["POST"])
def embed():
    """Standalone embedding endpoint — just returns the vector."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    embedding = model.encode([text])[0].tolist()

    return jsonify({
        "text": text[:100],
        "embedding": embedding,
        "dimension": len(embedding),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
