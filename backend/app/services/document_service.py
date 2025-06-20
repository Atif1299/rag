import shutil
from pathlib import Path
from fastapi import UploadFile
import os
DOCUMENTS_DIR = Path("storage/documents")

def count_user_documents(user_id: str) -> int:
    user_dir = DOCUMENTS_DIR / user_id
    return len([f for f in user_dir.iterdir() if f.is_file()]) if user_dir.exists() else 0

def save_document(user_id: str, file: UploadFile) -> bool:
    user_dir = DOCUMENTS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    file_path = user_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return True
