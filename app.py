import networkx as nx
import sqlite3
from datetime import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import pdfplumber
import fitz
from PIL import Image
import io
import json
import re
from google import genai

def log_query(question, top_score, num_results, retrieved_pages, answer):
    conn = sqlite3.connect('query_logs.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS query_logs 
                   (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    timestamp TEXT, question TEXT, 
                    top_score REAL, num_results INTEGER, 
                    retrieved_pages TEXT, answer TEXT)''')
    conn.execute(
        "INSERT INTO query_logs (timestamp, question, top_score, num_results, retrieved_pages, answer) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, top_score, num_results, str(retrieved_pages), answer)
    )
    conn.commit()
    conn.close()

# ── PAGE CONFIG ──
st.set_page_config(page_title="Danfoss Manual Assistant", page_icon="🔧")
st.title("🔧 Danfoss EZ Clip Assembly Manual Assistant")
st.write("Ask a question about the EZ Clip to 5400 Series assembly manual.")

# ── GEMINI API KEY ──
GEMINI_API_KEY = st.text_input("Enter your Gemini API key", type="password")

if not GEMINI_API_KEY:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# ── LOAD AND PROCESS PIPELINE (cached so it only runs once) ──
@st.cache_resource
def load_pipeline():
    PDF_PATH = "EZ_Clip_to_5400_Series.pdf"

    # Parse text
    pages_data = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page_num, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            raw_tables = page.extract_tables()
            fixed_tables = []
            for table in raw_tables:
                if not table or len(table) < 2:
                    fixed_tables.append({'raw_fallback': raw_text})
                    continue
                headers = table[0]
                if any('\n' in str(h) for h in headers if h):
                    flat = '\n'.join(['\t'.join([str(c) for c in row if c]) for row in table])
                    fixed_tables.append({'raw_fallback': flat})
                else:
                    rows = [dict(zip(headers, row)) for row in table[1:]]
                    fixed_tables.append({'rows': rows})
            pages_data.append({'page': page_num + 1, 'text': raw_text, 'tables': fixed_tables})

    # Semantic chunking
    def chunk_text_semantic(text, max_words=400, overlap=50):
        if not text:
            return []
        splits = re.split(r'\n{2,}|(?=EZ Step \d+)|(?=Step \d+:)', text)
        splits = [s.strip() for s in splits if s.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        for split in splits:
            words = split.split()
            if current_word_count + len(words) > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + words
                current_word_count = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_word_count += len(words)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    text_chunks = []
    chunk_id = 0
    for page in pages_data:
        for chunk in chunk_text_semantic(page['text']):
            text_chunks.append({'id': f"text_chunk_{chunk_id}", 'content': chunk, 'type': 'text', 'page': page['page'], 'source': 'page_text'})
            chunk_id += 1
        for t in page['tables']:
            if 'raw_fallback' in t and t['raw_fallback']:
                text_chunks.append({'id': f"text_chunk_{chunk_id}", 'content': t['raw_fallback'], 'type': 'table', 'page': page['page'], 'source': 'table'})
                chunk_id += 1

    # Load captions from backup
    with open('captions_backup.json', 'r') as f:
        saved_captions = json.load(f)

    # Embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    for chunk in text_chunks:
        chunk['embedding'] = embedder.encode(chunk['content']).tolist()

    for caption_item in saved_captions:
        caption_item['embedding'] = embedder.encode(caption_item['caption']).tolist()

    # Chroma setup
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("danfoss_ezclip")
    except:
        pass
    collection = chroma_client.create_collection(name="danfoss_ezclip", metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c['id'] for c in text_chunks],
        embeddings=[c['embedding'] for c in text_chunks],
        documents=[c['content'] for c in text_chunks],
        metadatas=[{'type': c['type'], 'page': c['page'], 'source': c['source']} for c in text_chunks]
    )

    collection.add(
        ids=[c['filename'] for c in saved_captions],
        embeddings=[c['embedding'] for c in saved_captions],
        documents=[c['caption'] for c in saved_captions],
        metadatas=[{'type': 'image', 'page': 0, 'source': 'image_caption', 'filename': c['filename']} for c in saved_captions]
    )

    return collection, embedder, reranker

with st.spinner("Loading pipeline (this takes a minute on first run)..."):
    collection, embedder, reranker = load_pipeline()


def load_kg(path="mmkg.graphml"):
    """Load the knowledge graph from GraphML file."""
    G = nx.read_graphml(path)
    return G

G = load_kg()
print(f"KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ── KG RETREIVAL FUNCTION ──
def retrieve_from_kg(question, G, top_k=5):
    """Search KG nodes by matching question keywords against entity attributes."""
    question_lower = question.lower()
    keywords = [w for w in question_lower.split() if len(w) > 3]
    
    scored_nodes = []
    
    for node_id, attrs in G.nodes(data=True):
        # Build a searchable string from node attributes
        searchable = ' '.join([
            str(attrs.get('entity_name', '')),
            str(attrs.get('entity_type', '')),
            str(attrs.get('text', '')),
            str(attrs.get('description', '')),
            str(attrs.get('part_number', '')),
            str(attrs.get('alias', '')),
            str(attrs.get('function', '')),
        ]).lower()
        
        # Score by keyword matches
        score = sum(1 for kw in keywords if kw in searchable)
        
        if score > 0:
            # Also get connected nodes for context
            neighbors = list(G.neighbors(node_id))
            neighbor_info = []
            for n in neighbors[:3]:
                n_attrs = G.nodes[n]
                rel = G.edges[node_id, n].get('relation', '')
                neighbor_info.append(f"{rel} → {n_attrs.get('entity_name', n)}")
            
            context = f"[KG | {attrs.get('entity_type', 'Unknown')}] {attrs.get('entity_name', '')}. "
            if attrs.get('text'):
                context += attrs.get('text') + ". "
            if attrs.get('part_number'):
                context += f"Part number: {attrs.get('part_number')}. "
            if neighbor_info:
                context += "Related: " + ", ".join(neighbor_info)
            
            scored_nodes.append((score, context))
    
    # Sort by score and return top_k
    scored_nodes.sort(reverse=True)
    return [context for _, context in scored_nodes[:top_k]]



# ── RAG FUNCTION ──
def expand_query(question):
    prompt = f"""Generate 3 different ways to ask the following question about a technical assembly manual.
Return only the 3 rephrased questions, one per line, no numbering, no extra text.

Original question: {question}"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    rephrased = response.text.strip().split('\n')
    return [question] + [q.strip() for q in rephrased if q.strip()]

def answer_question(question, n_results=5):
    try:
        queries = expand_query(question)
    except Exception:
        queries = [question]

    all_doc_ids = set()
    all_results = []
    for q in queries:
        query_embedding = embedder.encode(q).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=['documents', 'metadatas', 'distances'])
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            doc_key = doc[:100]
            if doc_key not in all_doc_ids:
                all_doc_ids.add(doc_key)
                all_results.append((doc, meta, dist))

    if len(all_results) > 1:
        pairs = [[question, doc] for doc, meta, dist in all_results]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, all_results), reverse=True)
        top_results = [r for _, r in ranked[:5]]
    else:
        top_results = all_results[:5]

    # Vector context
    context_pieces = []
    for doc, meta, dist in top_results:
        if 'Technical diagram' in doc:
            continue
        source = f"[Page {meta['page']} | {meta['type']}]"
        context_pieces.append(f"{source}\n{doc}")

    # KG context
    kg_results = retrieve_from_kg(question, G, top_k=5)
    for kg_context in kg_results:
        context_pieces.append(kg_context)

    context = "\n\n".join(context_pieces)

    prompt = f"""You are a helpful assistant for the Danfoss EZ Clip to 5400 Series assembly manual.
You MUST only use the context provided below to answer the question.
Do NOT use any outside knowledge.
For each piece of information in your answer, cite the source in the format [Page X | type].
If the context does not contain enough information to answer the question, respond EXACTLY with:
"This manual does not contain information to answer that question."

Context:
{context}

Question: {question}
Answer:"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    # Log the query
    top_score = 1 - top_results[0][2] if top_results else 0
    retrieved_pages = [meta['page'] for _, meta, _ in top_results]
    log_query(question, top_score, len(top_results), retrieved_pages, response.text)

    return response.text

# ── UI ──
question = st.text_input("Your question:")
if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        try:
            answer = answer_question(question)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")

# ── QUERY LOG VIEWER ──
if st.checkbox("Show query performance log"):
    try:
        conn = sqlite3.connect('query_logs.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS query_logs 
                       (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        timestamp TEXT, question TEXT, 
                        top_score REAL, num_results INTEGER, 
                        retrieved_pages TEXT, answer TEXT)''')
        conn.commit()
        import pandas as pd
        df = pd.read_sql_query("SELECT timestamp, question, top_score, num_results, retrieved_pages FROM query_logs ORDER BY timestamp DESC", conn)
        conn.close()
        if len(df) == 0:
            st.info("No queries logged yet.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Log error: {e}")

