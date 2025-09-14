"""
Streamlit AI Excel Analysis Assistant (LLM Option B - OpenAI)

Features:
- Upload multiple Excel workbooks
- Ingest and normalize sheets into textual chunks
- Create embeddings locally (sentence-transformers) and index with FAISS
- Retrieve relevant chunks for a user question
- Use OpenAI (chat completion) for summarization, recommendations, checklists
- Generate simple visuals (matplotlib/plotly)
- Export checklist as Excel/CSV and charts as PNG

Usage:
1) Install dependencies: see README or run:
   pip install streamlit pandas openpyxl sentence-transformers faiss-cpu openai python-dotenv matplotlib plotly
   (If faiss-cpu fails on your platform, install 'annoy' and the code will fall back.)
2) Create a .env file in the same folder with: OPENAI_API_KEY=sk-...
3) Run: streamlit run streamlit_ai_excel_assistant.py

Notes:
- This is a single-file prototype. For production, split code into modules and add security / logging.
- OpenAI API usage can incur cost. Use sparingly or choose a smaller model.

"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import os
import pickle
import time
import matplotlib.pyplot as plt
import plotly.express as px

# Embedding / indexing
from sentence_transformers import SentenceTransformer

# Try FAISS, fall back to Annoy if necessary
USE_ANNOY = False
try:
    import faiss
except Exception:
    try:
        from annoy import AnnoyIndex
        USE_ANNOY = True
    except Exception:
        st.error("Neither faiss nor annoy is installed. Please install faiss-cpu or annoy.")

# OpenAI
import openai
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found. Please set it in a .env file or environment variable.")
else:
    openai.api_key = OPENAI_API_KEY

# Global (in-memory) store for prototyping (not persistent across Streamlit reruns unless cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBED_MODEL = load_embedding_model()
EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()

# Index store - small in-memory management with caching
class VectorIndex:
    def __init__(self, dim):
        self.dim = dim
        self.texts = []
        self.meta = []
        self.index = None
        self.next_id = 0
        if USE_ANNOY:
            self.annoy_index = AnnoyIndex(dim, 'angular')
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings, texts, meta):
        # embeddings: np.array (n, dim)
        n = embeddings.shape[0]
        if USE_ANNOY:
            for i in range(n):
                self.annoy_index.add_item(self.next_id, embeddings[i].astype(np.float32))
                self.texts.append(texts[i])
                self.meta.append(meta[i])
                self.next_id += 1
        else:
            self.index.add(embeddings.astype(np.float32))
            self.texts.extend(texts)
            self.meta.extend(meta)

    def build(self):
        if USE_ANNOY:
            # build with 10 trees (trade-off speed/accuracy)
            self.annoy_index.build(10)

    def search(self, q_emb, k=6):
        if USE_ANNOY:
            ids, distances = self.annoy_index.get_nns_by_vector(q_emb[0].astype(np.float32), k, include_distances=True)
            retrieved_texts = [self.texts[i] for i in ids]
            retrieved_meta = [self.meta[i] for i in ids]
            return list(zip(retrieved_texts, retrieved_meta, distances))
        else:
            D, I = self.index.search(q_emb.astype(np.float32), k)
            results = []
            for idx in I[0]:
                if idx < len(self.texts):
                    results.append((self.texts[idx], self.meta[idx], None))
            return results

# Utility: read and normalize Excel files

def read_excel_file_bytes(uploaded_file):
    # uploaded_file is a Streamlit UploadedFile object
    xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
    frames = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
            df = df.fillna('')
            df['_source_file'] = uploaded_file.name
            df['_sheet'] = sheet
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not read sheet {sheet} in {uploaded_file.name}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()


def row_to_chunk_text(row, keep_columns=None):
    parts = []
    for c in row.index:
        if c.startswith('_'):
            continue
        if keep_columns and c not in keep_columns:
            continue
        value = row[c]
        if pd.isna(value) or value == '':
            continue
        parts.append(f"{c}: {value}")
    return " | ".join(parts)


def normalize_dataframe(df, chunk_columns=None):
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    # Create chunk text per row
    df['chunk_text'] = df.apply(lambda r: row_to_chunk_text(r, keep_columns=chunk_columns), axis=1)
    # Add a row id column to reference back
    df['_row_id'] = df.index
    return df

# Embedding + index helpers

@st.cache_resource
def create_empty_index():
    return VectorIndex(EMBED_DIM)

INDEX = create_empty_index()


def embed_texts(texts, batch_size=32):
    return EMBED_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)

# OpenAI helper

def call_openai_chat_system(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.1, max_tokens=800):
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured."
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# Prompt templates
SYSTEM_PROMPT = (
    "You are an experienced QA and Project Management analyst. Given extracted context from audit spreadsheets, "
    "provide concise structured output. Make safe assumptions only when necessary and note if you have insufficient data."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Context chunks:\n{context}\n\n"
    "Task:\n1) Provide 3-6 bullets: what is being done well across the audits (process-level, not individual).\n"
    "2) Provide 3-8 recurring issues / trends.\n"
    "3) Provide 5 general process-improvement recommendations (high-level).\n"
    "4) Produce a PM checklist for the stage named \"{stage_name}\". Output checklist as a markdown table with columns: Item, Description, Evidence required, Owner, Frequency.\n\n"
    "Produce the answer in markdown with clear headings."
)

CHECKLIST_PROMPT_TEMPLATE = (
    "Context chunks:\n{context}\n\nTask:\nGenerate a checklist (table) for project managers for stage '{stage_name}'. "
    "Group by categories (Documentation, Process, Testing, Signoffs). Columns: Item, Description, Evidence required, Owner, Frequency."
)

# Visualization helpers

def plot_issue_counts(df, issue_col_candidates=None):
    # attempt to find a column that looks like issue type / category
    if df.empty:
        return None
    candidates = issue_col_candidates or ['IssueType', 'Issue', 'Finding', 'Category', 'Issue Category']
    found_col = None
    for c in candidates:
        if c in df.columns:
            found_col = c
            break
    if not found_col:
        # try heuristics: a column with a small number of unique values
        for c in df.columns:
            if c.startswith('_'):
                continue
            if df[c].nunique() < 30 and df[c].dtype == object:
                found_col = c
                break
    if not found_col:
        return None
    counts = df.groupby([found_col, '_sheet']).size().reset_index(name='count')
    fig = px.bar(counts, x=found_col, y='count', color='_sheet', barmode='group', title=f'Counts by {found_col} per sheet')
    return fig

# App UI
st.set_page_config(page_title="Excel AI Auditor Assistant", layout="wide")
st.title("Excel AI Auditor Assistant (OpenAI-backed)")

with st.sidebar:
    st.header("Instructions")
    st.write("Upload one or more Excel files (.xlsx). The app will index rows as text chunks. Ask natural language questions about the audits and request summaries, recommendations, or checklists.")
    st.write("Warning: OpenAI API calls may incur costs. Set OPENAI_API_KEY in .env or environment.")
    st.markdown("---")
    model_choice = st.selectbox("OpenAI model (chat)", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

# Main: upload
uploaded_files = st.file_uploader("Upload Excel files", accept_multiple_files=True, type=["xlsx", "xls"], help="Multiple files allowed")
if uploaded_files:
    all_dfs = []
    for up in uploaded_files:
        df = read_excel_file_bytes(up)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        big_df = pd.concat(all_dfs, ignore_index=True)
        big_df = normalize_dataframe(big_df)
        st.success(f"Loaded {len(uploaded_files)} file(s) -> {len(big_df)} rows")
        if st.checkbox("Show sample rows"):
            st.dataframe(big_df.head(50))

        # Indexing controls
        if st.button("Create / Rebuild Embeddings & Index"):
            with st.spinner("Creating embeddings and indexing... this may take a moment"):
                texts = big_df['chunk_text'].fillna('').tolist()
                metas = big_df[[' _source_file' if ' _source_file' in big_df.columns else '_source_file', '_sheet', '_row_id']].to_dict('records') if '_row_id' in big_df.columns else [dict() for _ in texts]
                # ensure non-empty texts
                texts = [t if t.strip() != '' else 'empty' for t in texts]
                embeddings = embed_texts(texts)
                # reset index
                INDEX.texts = []
                INDEX.meta = []
                if USE_ANNOY:
                    INDEX.annoy_index = AnnoyIndex(EMBED_DIM, 'angular')
                    INDEX.next_id = 0
                else:
                    INDEX.index = faiss.IndexFlatL2(EMBED_DIM)
                INDEX.add(embeddings, texts, metas)
                INDEX.build()
                st.success("Index built ✅")

        st.markdown("---")
        st.header("Ask a question")
        user_q = st.text_input("Enter a natural language question about the uploaded audits", key='user_q')
        k = st.number_input("Number of retrieved chunks (k)", min_value=1, max_value=20, value=6)

        if st.button("Run Query") and user_q.strip() != '':
            with st.spinner("Retrieving and querying LLM..."):
                q_emb = embed_texts([user_q])
                retrieved = INDEX.search(q_emb, k=k)
                retrieved_texts = [r[0] for r in retrieved]
                retrieved_meta = [r[1] for r in retrieved]
                context = "\n\n---\n\n".join(retrieved_texts)
                prompt = f"Context chunks:\n{context}\n\nQuestion:\n{user_q}\n\nProvide a concise but thorough answer, with statistics if possible, and suggest recommendations where relevant. If you need more data say 'insufficient data'."
                resp = call_openai_chat_system(SYSTEM_PROMPT, prompt, model=model_choice, temperature=temp, max_tokens=800)
                st.markdown("**LLM Answer**")
                st.markdown(resp)

                # show retrieved chunk preview
                if st.expander("Show retrieved chunks"):
                    for i, (t, m, _) in enumerate(retrieved):
                        st.markdown(f"**Chunk {i+1}** — source: {m.get('_source_file','?')} sheet: {m.get('_sheet','?')} row: {m.get('_row_id','?')}")
                        st.write(t[:800])

        st.markdown("---")
        st.header("Generate Audit Summary + PM Checklist")
        stage_name = st.text_input("Stage name for checklist (e.g., 'Design', 'Build', 'Handover')", value="General")
        if st.button("Generate Summary & Checklist"):
            with st.spinner("Retrieving context and generating summary..."):
                # use top 30 chunks for context
                # create a balanced context by sampling unique sheets
                texts_all = big_df['chunk_text'].tolist()
                sample_texts = texts_all[:200]  # crude cap; adjust for performance
                context = "\n\n---\n\n".join(sample_texts)
                prompt = SUMMARY_PROMPT_TEMPLATE.format(context=context, stage_name=stage_name)
                resp = call_openai_chat_system(SYSTEM_PROMPT, prompt, model=model_choice, temperature=temp, max_tokens=1000)
                st.markdown("**Summary & Recommendations**")
                st.markdown(resp)

                # Generate dedicated checklist table via second call
                checklist_prompt = CHECKLIST_PROMPT_TEMPLATE.format(context=context, stage_name=stage_name)
                checklist_resp = call_openai_chat_system(SYSTEM_PROMPT, checklist_prompt, model=model_choice, temperature=0.0, max_tokens=800)
                st.markdown("**Checklist (raw)**")
                st.markdown(checklist_resp)

                # Try parse simple markdown table into DataFrame for export (very basic parser)
                def parse_markdown_table(md_text):
                    lines = [l.strip() for l in md_text.splitlines() if l.strip()]
                    table_lines = [l for l in lines if '|' in l]
                    if len(table_lines) < 2:
                        return None
                    # find header line (first with |)
                    header = table_lines[0]
                    headers = [h.strip() for h in header.split('|') if h.strip()]
                    # find subsequent rows
                    rows = []
                    for row_line in table_lines[1:]:
                        if set(row_line) == set('-|:'):
                            continue
                        cells = [c.strip() for c in row_line.split('|') if c.strip()]
                        if len(cells) == len(headers):
                            rows.append(cells)
                    if not rows:
                        return None
                    return pd.DataFrame(rows, columns=headers)

                checklist_df = parse_markdown_table(checklist_resp)
                if checklist_df is not None:
                    st.write("Checklist table parsed. You can download it.")
                    st.dataframe(checklist_df)
                    # export
                    towrite = BytesIO()
                    checklist_df.to_excel(towrite, index=False, engine='openpyxl')
                    towrite.seek(0)
                    st.download_button("Download checklist as Excel", towrite, file_name=f"checklist_{stage_name}.xlsx")
                else:
                    st.info("Could not parse checklist into table. You can copy the markdown above manually.")

        st.markdown("---")
        st.header("Visualizations")
        if st.button("Generate Suggested Visual"):
            with st.spinner("Generating visualization suggestion and chart..."):
                fig = plot_issue_counts(big_df)
                if fig is None:
                    st.info("Could not detect a suitable categorical column for plotting. Try checking your column names or provide a sample column.")
                else:
                    st.plotly_chart(fig, use_container_width=True)
                    # allow download
                    buf = BytesIO()
                    fig.write_image(buf, format='png')
                    buf.seek(0)
                    st.download_button("Download chart PNG", buf, file_name="chart.png")

        st.markdown("---")
        st.write("End of session. Re-upload files or rebuild the index to start new analysis.")

else:
    st.info("Upload Excel files to get started. You can drag and drop multiple files.")

# Footer
st.markdown("---")
st.markdown("Built for quick prototyping. For production: implement auth, persistent vector DB (Pinecone/Chroma), batching, throttling, and sensitive data handling.")

# EOF
