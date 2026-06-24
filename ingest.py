"""
Standalone ingestion script — run once (or per new batch of documents) to populate Pinecone.

Usage:
    python ingest.py

Reads all PDFs from ./data/, indexes them into Pinecone with contextual chunking,
and records each document in the SQLite document store so it appears in the app UI.
Re-running on an already-indexed document will create duplicate vectors — delete first if re-indexing.
"""

import os
import glob
from pinecone_utils import index_document_to_pinecone
from db_utils import insert_document_record

DATA_DIR = "./data"


def ingest_all():
    """Ingest all PDFs in DATA_DIR into Pinecone and register them in the document store."""
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/")
        return

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        print(f"Ingesting {filename}...")
        file_id = insert_document_record(filename)
        success = index_document_to_pinecone(file_path, file_id, filename)
        status = "done" if success else "FAILED"
        print(f"  [{status}] {filename} (file_id={file_id})")


if __name__ == "__main__":
    ingest_all()
