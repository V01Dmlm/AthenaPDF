# backend/pdf_handler.py (Fully Synchronous + CPU/GPU auto + Lazy Images)
import os
import fitz  # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import arabic_reshaper
from bidi.algorithm import get_display
import logging
import torch

logging.basicConfig(level=logging.INFO)

class PDFHandler:
    def __init__(self, upload_dir="data/uploads", vector_dir="data/vector_store", max_workers=4):
        self.upload_dir = upload_dir
        self.vector_dir = vector_dir
        self.max_workers = max_workers

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        # Auto-detect device for embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device for embeddings: {device}")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=device
        )

        self.index_path = os.path.join(vector_dir, "faiss.index")
        self.chunks_path = os.path.join(vector_dir, "chunks.pkl")
        self.metadata_path = os.path.join(vector_dir, "metadata.pkl")

        self.chunks = []
        self.metadata = []
        self.index = None
        self.pdf_images = {}

        self._load_index()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

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

    # ----------------- Sync PDF Save -----------------
    def save_pdf(self, file):
        """Save PDF, process text and images synchronously"""
        file_path = os.path.join(self.upload_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Process text and images concurrently
        text_future = self.executor.submit(self._process_pdf_text, file_path)
        images_future = self.executor.submit(self.extract_images, file.name)

        images = images_future.result()
        text_future.result()  # wait for text processing
        return images

    # ----------------- Text Processing -----------------
    def _process_pdf_text(self, pdf_path: str):
        text = self.extract_text(pdf_path)
        if not text.strip():
            logging.warning(f"No text found in PDF '{pdf_path}'")
            return

        chunks = self.chunk_text(text, chunk_size=300, overlap=50)
        metadata = [os.path.basename(pdf_path)] * len(chunks)

        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

        # Persist index and chunks
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logging.info(f"PDF '{os.path.basename(pdf_path)}' processed and embeddings saved!")

    def extract_text(self, pdf_path: str) -> str:
        try:
            text = ""
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logging.warning(f"Failed to read PDF '{pdf_path}': {e}")
            return ""

    def chunk_text(self, text: str, chunk_size=300, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # ----------------- Image Extraction -----------------
    def extract_images(self, pdf_name: str):
        """Threaded image extraction"""
        images = []
        pdf_path = os.path.join(self.upload_dir, pdf_name)
        try:
            doc = fitz.open(pdf_path)
            futures = []
            for page_idx, page in enumerate(doc):
                futures.append(self.executor.submit(self._extract_images_from_page, pdf_name, page_idx, page))
            for future in as_completed(futures):
                images.extend(future.result())
        except Exception as e:
            logging.warning(f"Failed to extract images from '{pdf_name}': {e}")
        self.pdf_images[pdf_name] = images
        return images

    def _extract_images_from_page(self, pdf_name, page_idx, page):
        page_images = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_path = os.path.join(self.upload_dir, f"{pdf_name}_{page_idx}_{img_idx}.{ext}")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            page_images.append(img_path)
        return page_images

    # ----------------- Context Retrieval -----------------
    def get_context(self, query: str, top_k=3, pdf_files: list = None):
        """Get top-k context chunks for a query"""
        if not self.index or len(self.chunks) == 0:
            return ""

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k*5)

        context = ""
        added = 0
        for idx in indices[0]:
            if idx >= len(self.chunks):
                continue
            pdf_name = self.metadata[idx]
            if pdf_files and pdf_name not in pdf_files:
                continue
            chunk_text = self.chunks[idx]
            if self._contains_arabic(chunk_text):
                chunk_text = get_display(arabic_reshaper.reshape(chunk_text))
            context += f"[From {pdf_name}]\n{chunk_text}\n\n"
            added += 1
            if added >= top_k:
                break
        return context.strip()

    def _contains_arabic(self, text: str) -> bool:
        return any("\u0600" <= c <= "\u06FF" for c in text)

    # ----------------- Clear Index -----------------
    def clear_index(self):
        self.index = None
        self.chunks = []
        self.metadata = []
        self.pdf_images = {}
        for path in [self.index_path, self.chunks_path, self.metadata_path]:
            if os.path.exists(path):
                os.remove(path)
        logging.info("Vector store cleared.")
