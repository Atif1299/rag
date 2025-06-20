from fastapi import APIRouter, Form, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from typing import List
from ..services.document_service import count_user_documents, save_document
from ..rag.embed_db import embed_documents_from_folder,delete_vectors_by_filename
import json
import os
from datetime import datetime
from pathlib import Path

router = APIRouter()


@router.delete("/documents/{user_id}/{filename}")
async def delete_document(user_id: str, filename: str):
    """
    Deletes a specific document and its metadata.
    """
    folder_path = f"storage/documents/{user_id}"
    file_path = os.path.join(folder_path, filename)
    metadata_file_path = os.path.join(folder_path, "documents_metadata.json")

    # Security check to prevent directory traversal attacks
    if not os.path.abspath(file_path).startswith(os.path.abspath(folder_path)):
        raise HTTPException(status_code=403, detail="Forbidden action.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        # 1. Delete the physical file
        os.remove(file_path)

        # 2. Update the metadata JSON file
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, 'r+') as f:
                metadata_list = json.load(f)

                # Filter out the deleted document
                updated_metadata = [doc for doc in metadata_list if doc.get("documentname") != filename]

                # Write the updated list back to the file
                f.seek(0)
                f.truncate()
                json.dump(updated_metadata, f, indent=2)

        # 3. Delete vectors from Milvus using the actual filename parameter
        result = delete_vectors_by_filename(
            filename=filename,  # Use the actual filename parameter, not hardcoded value
            collection_name="my_collection",
            milvus_host="localhost",
            milvus_port=19530
        )

        print(f"Deletion result: {result}")

        return {"detail": f"Successfully deleted {filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get("/documents/download/{user_id}/{filename}")
async def download_document(user_id: str, filename: str):
    """
    Serves a specific document for download.
    """
    folder_path = f"storage/documents/{user_id}"
    file_path = os.path.join(folder_path, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(folder_path)):
        raise HTTPException(status_code=403, detail="Forbidden action.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)

@router.get("/documents/preview/{user_id}/{filename}", response_class=PlainTextResponse)
async def preview_document(user_id: str, filename: str):
    """
    Returns the content of a text-based document for preview.
    """
    folder_path = f"storage/documents/{user_id}"
    file_path = os.path.join(folder_path, filename)

    if not os.path.abspath(file_path).startswith(os.path.abspath(folder_path)):
        raise HTTPException(status_code=403, detail="Forbidden action.")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    # For now, only allow preview for simple text files
    if not filename.lower().endswith(('.txt', '.csv', '.json', '.html')):
        raise HTTPException(status_code=400, detail="Preview is only available for text-based files.")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return PlainTextResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {str(e)}")

@router.get("/documents/count/{user_id}")
async def get_document_count(user_id: str):
    count = count_user_documents(user_id)
    return {"count": count, "limit": 10}

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks, # CORRECT: Moved to the first position
    user_id: str = Form(...),
    documents: List[UploadFile] = File(...)
):
    current_count = count_user_documents(user_id)
    remaining_slots = 10 - current_count

    if remaining_slots <= 0:
        return JSONResponse(status_code=400, content={
            "detail": "Unable to upload docs, memory is full",
            "uploaded_count": 0,
            "current_document_count": current_count
        })

    uploaded_count, saved_filenames = 0, []
    folder_path = f"storage/documents/{user_id}"
    metadata_file_path = os.path.join(folder_path, "documents_metadata.json")

    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    for file in documents[:remaining_slots]:
        try:
            save_document(user_id, file)
            uploaded_count += 1
            saved_filenames.append(file.filename)

            saved_file_path = os.path.join(folder_path, file.filename)
            file_size_bytes = os.path.getsize(saved_file_path)

            if file_size_bytes < 1024 * 1024:
                file_size_kb = round(file_size_bytes / 1024, 2)
                size_display = f"{file_size_kb}kb"
            else:
                file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
                size_display = f"{file_size_mb}mb"

            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            metadata_entry = {
                "documentname": file.filename,
                "type": file_extension,
                "size": size_display,
                "date": datetime.now().strftime("%B %d, %Y")
            }
            metadata_list.append(metadata_entry)

        except Exception as e:
            print(f"Error uploading {file.filename}: {str(e)}")
            pass

    if uploaded_count > 0:
        os.makedirs(folder_path, exist_ok=True)
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

        # Move the slow embedding task to the background
        background_tasks.add_task(
            embed_documents_from_folder,
            folder_path=folder_path,
            file_names=saved_filenames,
            collection_name="my_collection",
            model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
            milvus_host="127.0.0.1",
            milvus_port=19530
        )

    message = f"{uploaded_count} documents will be processed in the background."
    if uploaded_count < len(documents):
        message += " (some couldn't be uploaded)"

    return {
        "detail": message,
        "uploaded_count": uploaded_count,
        "current_document_count": current_count + uploaded_count
    }

@router.get("/documents/{user_id}")
async def get_documents_metadata(user_id: str):
    """
    Get metadata for all documents uploaded by a specific user
    """
    folder_path = f"storage/documents/{user_id}"
    metadata_file_path = os.path.join(folder_path, "documents_metadata.json")

    if not os.path.exists(metadata_file_path):
        return {
            "user_id": user_id,
            "documents": [],
            "total_documents": 0,
            "message": "No documents found for this user"
        }

    try:
        with open(metadata_file_path, 'r') as f:
            metadata_list = json.load(f)

        return {
            "user_id": user_id,
            "documents": metadata_list,
            "total_documents": len(metadata_list),
            "message": f"Found {len(metadata_list)} documents"
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Error reading documents metadata file"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving documents: {str(e)}"
        )
