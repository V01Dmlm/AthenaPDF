import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class PDFHandler:
    def __init__(self, upload_dir="data/uploads", vector_dir="data/vector_store"):
        self.upload_dir = upload_dir
        self.vector_dir = vector_dir

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        # Embedding model
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Persistent storage
        self.index_path = os.path.join(vector_dir, "faiss.index")
        self.chunks_path = os.path.join(vector_dir, "chunks.pkl")
        self.metadata_path = os.path.join(vector_dir, "metadata.pkl")

        self.chunks = []      # raw text chunks
        self.metadata = []    # info about which PDF each chunk came from
        self.index = None

        # Load existing embeddings if available
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
        else:
            # Initialize empty FAISS index
            self.index = None
            self.chunks = []
            self.metadata = []

    def save_pdf(self, file):
        file_path = os.path.join(self.upload_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        self._process_pdf(file_path)

    def _process_pdf(self, pdf_path: str):
        text = self.extract_text(pdf_path)
        new_chunks = self.chunk_text(text, chunk_size=500)
        new_metadata = [os.path.basename(pdf_path)] * len(new_chunks)

        # Compute embeddings
        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)

        # Add to existing FAISS index
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

        # Update local storage
        self.chunks.extend(new_chunks)
        self.metadata.extend(new_metadata)

        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"PDF '{os.path.basename(pdf_path)}' processed and embeddings saved!")

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text

    def chunk_text(self, text: str, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def get_context(self, query: str, top_k=3) -> str:
        """
        Retrieve top-k relevant chunks from all PDFs.
        """
        if not self.index or len(self.chunks) == 0:
            return ""

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)

        context = ""
        for i in indices[0]:
            if i < len(self.chunks):
                pdf_name = self.metadata[i] if i < len(self.metadata) else "Unknown PDF"
                context += f"[From {pdf_name}]\n{self.chunks[i]}\n\n"
        return context.strip()
