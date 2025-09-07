# backend/pdf_handler.py
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import arabic_reshaper
from bidi.algorithm import get_display
import logging

logging.basicConfig(level=logging.INFO)

class PDFHandler:
    def __init__(self, upload_dir="data/uploads", vector_dir="data/vector_store"):
        self.upload_dir = upload_dir
        self.vector_dir = vector_dir

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index_path = os.path.join(vector_dir, "faiss.index")
        self.chunks_path = os.path.join(vector_dir, "chunks.pkl")
        self.metadata_path = os.path.join(vector_dir, "metadata.pkl")

        self.chunks = []
        self.metadata = []
        self.index = None

        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            logging.info("Loading existing FAISS index...")
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "rb") as f:
                        self.metadata = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load index: {e}")
                self.index = None
                self.chunks = []
                self.metadata = []
        else:
            self.index = None

    def save_pdf(self, file):
        """
        Save uploaded PDF and process it.
        """
        file_path = os.path.join(self.upload_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        self._process_pdf(file_path)

    def _process_pdf(self, pdf_path: str):
        """
        Extract text, chunk it, embed, and update FAISS index.
        """
        text = self.extract_text(pdf_path)
        if self._contains_arabic(text):
            text = get_display(arabic_reshaper.reshape(text))

        new_chunks = self.chunk_text(text, chunk_size=500, overlap=50)
        new_metadata = [os.path.basename(pdf_path)] * len(new_chunks)

        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.chunks.extend(new_chunks)
        self.metadata.extend(new_metadata)

        # Persist to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logging.info(f"PDF '{os.path.basename(pdf_path)}' processed and embeddings saved!")

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF using PyMuPDF.
        """
        try:
            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logging.warning(f"Failed to read PDF '{pdf_path}': {e}")
            return ""

    def chunk_text(self, text: str, chunk_size=500, overlap=50):
        """
        Split text into overlapping chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _contains_arabic(self, text: str) -> bool:
        """
        Check if text contains Arabic characters.
        """
        return any("\u0600" <= c <= "\u06FF" for c in text)

    def get_context(self, query: str, top_k=3) -> str:
        """
        Retrieve top-k relevant chunks from PDFs.
        """
        if not self.index or len(self.chunks) == 0:
            return ""

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)

        context = ""
        for idx in indices[0]:
            if idx < len(self.chunks):
                pdf_name = self.metadata[idx] if idx < len(self.metadata) else "Unknown PDF"
                context += f"[From {pdf_name}]\n{self.chunks[idx]}\n\n"
        return context.strip()
