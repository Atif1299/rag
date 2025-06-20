"""
Embedding and Vector Store Module - Core functionality for embedding documents
and retrieving them via similarity search using Milvus and HuggingFace API.
"""
import os
import uuid
import time
import re
from loguru import logger
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from pymilvus import Collection, connections, utility

# Import necessary functions from loader
from .loader import get_document_chunks, ContextualDocumentSplitter

# Removed RTL handling code

# HuggingFace API token - should be stored in environment variables
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# New embedding model using HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class EmbeddingModel:
    """Class for handling embeddings generation using local models"""
    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0"):
        """
        Initialize the embedding model with local download
        Args:
            model_name: Name of the model to download from HuggingFace
        """
        self.model_name = model_name

        try:
            # Import the required libraries for local model usage
            from sentence_transformers import SentenceTransformer

            # Try to load the model (will download if not present)
            logger.info(f"Loading/downloading model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)

            # Get the embedding dimension from the model
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully with dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise ValueError(f"Could not load embedding model {model_name}. Error: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using the local embedding model
        Args:
            texts: List of texts to embed
        Returns:
            List of embeddings, one per text
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts using model: {self.model_name}")

        try:
            # Process in batches to optimize memory usage
            batch_size = 16
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(texts) + batch_size - 1)//batch_size}")

                # Encode the batch - returns numpy array
                batch_embeddings = self.embedding_model.encode(batch)

                # Convert numpy arrays to lists
                batch_embeddings = [emb.tolist() for emb in batch_embeddings]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text.
        Args:
            query: Text to embed
        Returns:
            Embedding for the text
        """
        try:
            # Encode the query - returns numpy array
            embedding = self.embedding_model.encode(query)

            # Convert numpy array to list
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def get_sentence_embedding_dimension(self) -> int:
        """
        Return the dimension of the embeddings.
        """
        return self.dimension

# Wrapper class to make EmbeddingModel compatible with LangChain's Embeddings interface
class EmbeddingModelAdapter(Embeddings):
    """
    Adapter to make EmbeddingModel compatible with LangChain's Embeddings interface.
    """

    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0"):
        """
        Initialize the adapter with an EmbeddingModel.
        """
        self.embedding_model = EmbeddingModel(model_name=model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        """
        return self.embedding_model.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        """
        return self.embedding_model.embed_query(text)

    def get_sentence_embedding_dimension(self) -> int:
        """
        Return the dimension of the embeddings.
        """
        return self.embedding_model.dimension

class EmbedStore:
    """
    EmbedStore provides a simple interface for embedding documents and storing
    them in a vector database for similarity search using Milvus.
    """

    def __init__(self, collection_name: str = "my_collection",
                 model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                 milvus_host: str = "127.0.0.1",
                 milvus_port: int = 19530,
                 connection_alias: str = "default"):
        """
        Initialize the EmbedStore with a local embedding model and Milvus vector store.

        Args:
            collection_name: Name of the Milvus collection to use
            model_name: Name of the model to download and use for embeddings
            milvus_host: Host address for Milvus server
            milvus_port: Port for Milvus server
            connection_alias: Alias for the Milvus connection
        """
        # Initialize the embedding model using our adapter
        self.model_name = model_name
        self.connection_alias = connection_alias

        try:
            self.embedding_model = EmbeddingModelAdapter(model_name=model_name)
            logger.info(f"Successfully initialized embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

        # Connect to Milvus
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # Setup connection
        self._setup_milvus_connection()

        # Initialize Milvus vector store
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": milvus_port},
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}}
        )

        self.collection_name = collection_name
        logger.info(f"EmbedStore initialized with collection: {collection_name} and {model_name} model")

    def _setup_milvus_connection(self):
        """
        Helper method to establish or reuse a Milvus connection.
        """
        try:
            # Check if the connection already exists
            if connections.has_connection(alias=self.connection_alias):
                # Get existing connection details
                conn_info = connections.get_connection_addr(self.connection_alias)
                current_host = conn_info.get("host")
                current_port = conn_info.get("port")

                # If configuration is different, disconnect and reconnect
                if current_host != self.milvus_host or current_port != self.milvus_port:
                    logger.info(f"Connection configuration mismatch. Reconnecting to Milvus...")
                    connections.disconnect(alias=self.connection_alias)
                    connections.connect(
                        alias=self.connection_alias,
                        host=self.milvus_host,
                        port=self.milvus_port
                    )
                    logger.info(f"Reconnected to Milvus at {self.milvus_host}:{self.milvus_port}")
                else:
                    logger.info(f"Reusing existing Milvus connection at {self.milvus_host}:{self.milvus_port}")
            else:
                # Create a new connection
                connections.connect(
                    alias=self.connection_alias,
                    host=self.milvus_host,
                    port=self.milvus_port
                )
                logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus at {self.milvus_host}:{self.milvus_port}: {str(e)}")
            raise

    def load_and_store_documents(self, folder_path: str, file_names: List[str] = None) -> List[str]:
        """
        Load documents from a folder using the get_document_chunks function,
        then store them in the vector database.

        Args:
            folder_path: Path to the folder containing documents to process
            file_names: Optional list of specific file names to process within the folder

        Returns:
            List of IDs of the stored documents
        """
        try:
            # Get document chunks using the imported function
            if file_names:
                document_chunks = get_document_chunks(input_source=folder_path, filenames=file_names)
                logger.info(f"Processing specific files: {file_names}")
            else:
                document_chunks = get_document_chunks(input_source=folder_path)
                logger.info(f"Processing all files in folder")

            if not document_chunks:
                logger.warning(f"No document chunks retrieved from: {folder_path}")
                return []

            logger.info(f"Retrieved {len(document_chunks)} document chunks from: {folder_path}")

            # Store the chunks in the vector database
            return self.store_documents(document_chunks)

        except Exception as e:
            logger.error(f"Error loading and storing documents: {str(e)}")
            raise

    def store_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Store pre-processed chunks directly in vector database after embedding them.

        Args:
            documents: List of pre-processed Document chunks to store
            ids: Optional list of IDs for the documents

        Returns:
            List of IDs of the stored documents
        """
        try:
            # Validate input
            if not documents:
                logger.warning("No documents provided to store_documents")
                return []

            logger.info(f"Processing {len(documents)} pre-processed chunks for storage...")
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]

            # Process each document before storage
            for doc in documents:
                # Ensure metadata exists
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}

                # Define all required fields with their default values
                required_fields = {
                    'previous_chunk_summary': "",
                    'next_chunk_summary': "",
                    'document_position': "",
                    'document_headers': ""  # Add document_headers to required fields
                }

                # Set default values for missing required fields
                for field, default_value in required_fields.items():
                    if field not in doc.metadata:
                        doc.metadata[field] = default_value

                # Convert document_headers to string if it's a complex type
                if 'document_headers' in doc.metadata:
                    if not isinstance(doc.metadata['document_headers'], (str, int, float, bool)):
                        doc.metadata['document_headers'] = str(doc.metadata['document_headers'])

            # Add documents to Milvus vector store
            doc_ids = self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info(f"Successfully stored {len(documents)} chunks with IDs: {doc_ids[:5]}...")
            return doc_ids

        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise


    def search_similar(self, query: str, top_k: int = 4,
                      filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Find documents similar to the provided query.

        Args:
            query: The query text to search for
            top_k: Number of top documents to retrieve
            filter: Optional filter to apply to the search

        Returns:
            List of Document objects similar to the query
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided to search_similar")
                return []

            logger.info(f"Searching for documents similar to: {query[:50]}...")

            # Process the query if process_text is available
            processed_query = query
            try:
                # Create a context-aware document splitter
                splitter = ContextualDocumentSplitter()

                # Process the query text
                query_chunks = splitter.process_text(query, {"source": "query"})
                if query_chunks:
                    processed_query = query_chunks[0].page_content
            except:
                # If process_text is not available, use basic preprocessing directly
                processed_query = re.sub(r'\s+', ' ', query).strip()

            logger.info(f"Processed query: {processed_query[:50]}...")

            # Perform similarity search using the vector store
            similar_docs = self.vector_store.similarity_search(
                query=processed_query,
                k=top_k,
                filter=filter
            )

            logger.info(f"Retrieved {len(similar_docs)} similar documents")
            return similar_docs

        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def search_with_scores(self, query: str, top_k: int = 4,
                         filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Find documents similar to the query and return with similarity scores.

        Args:
            query: The query text to search for
            top_k: Number of top documents to retrieve
            filter: Optional filter to apply to the search

        Returns:
            List of tuples containing (Document, similarity_score)
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided to search_with_scores")
                return []

            logger.info(f"Searching for documents with scores for: {query[:50]}...")

            # Process the query if process_text is available
            processed_query = query
            try:
                splitter = ContextualDocumentSplitter()

                # Process the query text
                query_chunks = splitter.process_text(query, {"source": "query"})
                if query_chunks:
                    processed_query = query_chunks[0].page_content
            except:
                # If process_text is not available, use basic preprocessing directly
                processed_query = re.sub(r'\s+', ' ', query).strip()

            logger.info(f"Processed query: {processed_query[:50]}...")

            # Perform similarity search with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=processed_query,
                k=top_k,
                filter=filter
            )

            logger.info(f"Retrieved {len(docs_with_scores)} documents with similarity scores")
            return docs_with_scores

        except Exception as e:
            logger.error(f"Error during similarity search with scores: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Use the existing connection instead of creating a new one
            self._setup_milvus_connection()

            # Check if collection exists
            exists = utility.has_collection(self.collection_name)

            # Initialize stats dictionary with basic info
            stats = {
                "exists": exists,
                "name": self.collection_name,
                "count": 0,
                "dimension": self.embedding_model.get_sentence_embedding_dimension()
            }

            if exists:
                # Get collection info
                collection = Collection(self.collection_name)
                collection.load()  # Load collection to get accurate statistics
                stats["count"] = collection.num_entities

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "error": str(e),
                "name": self.collection_name,
                "exists": False,
                "count": 0,
                "dimension": self.embedding_model.get_sentence_embedding_dimension()
            }

    def delete_collection(self) -> bool:
        """
        Delete the current collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use the existing connection instead of creating a new one
            self._setup_milvus_connection()

            # Check if collection exists first
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
            else:
                logger.warning(f"Collection {self.collection_name} does not exist, nothing to delete")
                return False
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False

    def delete_vectors_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        Delete all vectors that have the specified filename in their metadata.

        Args:
            filename: Name of the file whose vectors should be deleted

        Returns:
            Dictionary with deletion results including count of deleted vectors
        """
        try:
            logger.info(f"Attempting to delete vectors for filename: {filename}")

            # Setup Milvus connection
            self._setup_milvus_connection()

            # Check if collection exists
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection {self.collection_name} does not exist")
                return {
                    "success": False,
                    "message": f"Collection {self.collection_name} does not exist",
                    "deleted_count": 0
                }

            # Get collection
            collection = Collection(self.collection_name)
            collection.load()

            # Get collection stats before deletion
            initial_count = collection.num_entities
            logger.info(f"Collection has {initial_count} entities before deletion")

            # Create filter expression for the filename
            # This assumes the filename is stored in metadata with key 'source' or 'filename'
            # You may need to adjust this based on how filenames are stored in your metadata
            filter_expressions = [
                f'source == "{filename}"',
                f'filename == "{filename}"',
                f'source like "%{filename}%"',  # For cases where source might include path
                f'filename like "%{filename}%"'
            ]

            deleted_count = 0
            successful_filters = []

            # Try different filter expressions to handle various metadata formats
            for filter_expr in filter_expressions:
                try:
                    logger.info(f"Trying filter expression: {filter_expr}")

                    # Delete entities matching the filter
                    delete_result = collection.delete(expr=filter_expr)

                    if hasattr(delete_result, 'delete_count') and delete_result.delete_count > 0:
                        deleted_count += delete_result.delete_count
                        successful_filters.append(filter_expr)
                        logger.info(f"Deleted {delete_result.delete_count} entities with filter: {filter_expr}")

                except Exception as filter_error:
                    logger.debug(f"Filter expression '{filter_expr}' failed: {str(filter_error)}")
                    continue

            # If no standard filters worked, try to search and delete manually
            if deleted_count == 0:
                logger.info("Standard filters didn't work, attempting manual search and delete")
                try:
                    # Search for documents with the filename to get their IDs
                    # This is a fallback method
                    similar_docs = self.search_similar(
                        query=filename,  # Using filename as query
                        top_k=1000,  # Large number to get all potential matches
                    )

                    # Filter results that actually match the filename
                    matching_ids = []
                    for doc in similar_docs:
                        if doc.metadata:
                            # Check various metadata fields that might contain the filename
                            metadata_values = [
                                doc.metadata.get('source', ''),
                                doc.metadata.get('filename', ''),
                                doc.metadata.get('file_name', ''),
                                str(doc.metadata)  # Check the entire metadata string
                            ]

                            if any(filename in str(value) for value in metadata_values):
                                # Try to get the document ID if available
                                if hasattr(doc, 'id') or 'id' in doc.metadata:
                                    doc_id = getattr(doc, 'id', doc.metadata.get('id'))
                                    if doc_id:
                                        matching_ids.append(doc_id)

                    # Delete by IDs if we found any
                    if matching_ids:
                        id_filter = f"id in {matching_ids}"
                        delete_result = collection.delete(expr=id_filter)
                        if hasattr(delete_result, 'delete_count'):
                            deleted_count = delete_result.delete_count
                            logger.info(f"Manually deleted {deleted_count} entities by ID")

                except Exception as manual_error:
                    logger.warning(f"Manual deletion method also failed: {str(manual_error)}")

            # Flush the collection to ensure deletions are persisted
            collection.flush()

            # Get final count
            final_count = collection.num_entities
            actual_deleted = initial_count - final_count

            if actual_deleted > 0:
                logger.info(f"Successfully deleted {actual_deleted} vectors for filename: {filename}")
                return {
                    "success": True,
                    "message": f"Successfully deleted {actual_deleted} vectors for filename: {filename}",
                    "deleted_count": actual_deleted,
                    "initial_count": initial_count,
                    "final_count": final_count,
                    "successful_filters": successful_filters
                }
            else:
                logger.warning(f"No vectors found to delete for filename: {filename}")
                return {
                    "success": False,
                    "message": f"No vectors found for filename: {filename}",
                    "deleted_count": 0,
                    "initial_count": initial_count,
                    "final_count": final_count
                }

        except Exception as e:
            logger.error(f"Error deleting vectors by filename: {str(e)}")
            return {
                "success": False,
                "message": f"Error deleting vectors: {str(e)}",
                "deleted_count": 0
            }

# ========== FUNCTIONS FOR EXTERNAL USE ==========

def delete_vectors_by_filename(filename: str, collection_name: str = "my_collection",
                               model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                               milvus_host: str = "127.0.0.1", milvus_port: int = 19530,
                               connection_alias: str = "default") -> Dict[str, Any]:
    """
    Delete all vectors that have the specified filename in their metadata from a Milvus collection.

    This function can be imported and used in other files to delete vectors by filename.

    Args:
        filename: Name of the file whose vectors should be deleted
        collection_name: Name of the Milvus collection to use
        model_name: Name of the model used for embeddings (needed for EmbedStore initialization)
        milvus_host: Host address for Milvus server
        milvus_port: Port for Milvus server
        connection_alias: Alias for the Milvus connection (should be unique if multiple instances are created)

    Returns:
        Dictionary with deletion results including count of deleted vectors
    """
    try:
        logger.info(f"Deleting vectors for filename: {filename}")
        logger.info(f"Using collection: {collection_name}")
        logger.info(f"Connecting to Milvus at {milvus_host}:{milvus_port}")

        # Create the EmbedStore with the specified parameters
        embed_store = EmbedStore(
            collection_name=collection_name,
            model_name=model_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            connection_alias=connection_alias
        )

        # Call the delete method on the EmbedStore instance
        result = embed_store.delete_vectors_by_filename(filename)

        return result

    except Exception as e:
        logger.error(f"Error in delete_vectors_by_filename: {str(e)}")
        return {
            "success": False,
            "message": f"Error deleting vectors: {str(e)}",
            "deleted_count": 0
        }



def embed_documents_from_folder(folder_path: str, file_names: List[str], collection_name: str = "my_collection",
                               model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                               milvus_host: str = "127.0.0.1", milvus_port: int = 19530,
                               connection_alias: str = "default") -> List[str]:
    """
    Retrieves document chunks from specific files in a folder using loader's get_document_chunks method,
    embeds them, and stores them in a Milvus vector database.

    This function can be imported and used in other files to process selected documents.

    Args:
        folder_path: Path to the folder containing documents to process
        file_names: List of file names within the folder to process
        collection_name: Name of the Milvus collection to use
        model_name: Name of the model to use for embeddings
        milvus_host: Host address for Milvus server
        milvus_port: Port for Milvus server
        connection_alias: Alias for the Milvus connection (should be unique if multiple instances are created)

    Returns:
        List of IDs of the stored document chunks
    """
    try:
        logger.info(f"Embedding documents from folder: {folder_path}")
        logger.info(f"Selected files: {file_names}")
        logger.info(f"Using collection: {collection_name} with model: {model_name}")
        logger.info(f"Connecting to Milvus at {milvus_host}:{milvus_port}")

        # Create the EmbedStore with the specified parameters
        embed_store = EmbedStore(
            collection_name=collection_name,
            model_name=model_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            connection_alias=connection_alias
        )

        # Get document chunks using loader's get_document_chunks method with the folder path and file names
        print("In embed file")
        document_chunks = get_document_chunks(input_source=folder_path, filenames=file_names)
        print("Still In embed file.")
        if not document_chunks:
            logger.warning(f"No document chunks retrieved from files in: {folder_path}")
            return []

        logger.info(f"Retrieved {len(document_chunks)} document chunks from specified files")

        # Store the document chunks
        document_ids = embed_store.store_documents(document_chunks)

        # Get and display collection statistics
        stats = embed_store.get_collection_stats()
        logger.info(f"Collection statistics: {stats}")

        return document_ids  # Fixed: Make sure to return document_ids

    except Exception as e:
        logger.error(f"Error in embed_documents_from_folder: {str(e)}")
        raise

def find_similar_chunks(query: str, collection_name: str = "my_collection",
                       model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
                       milvus_host: str = "127.0.0.1", milvus_port: int = 19530,
                       top_k: int = 4,
                       with_scores: bool = False,
                       filter: Optional[Dict[str, Any]] = None,
                       connection_alias: str = "default") -> List[Any]:
    """
    Takes a query as input, finds similar chunks in the Milvus vector database,
    and returns those chunks.

    This function can be imported and used in other files to search for
    similar content based on a query.

    Args:
        query: The query text to search for
        collection_name: Name of the Milvus collection to use
        model_name: Name of the model to use for embeddings
        milvus_host: Host address for Milvus server
        milvus_port: Port for Milvus server
        top_k: Number of top documents to retrieve
        with_scores: Whether to include similarity scores in the results
        filter: Optional filter to apply to the search
        connection_alias: Alias for the Milvus connection (should be unique if multiple instances are created)

    Returns:
        If with_scores is True:
            List of tuples containing (Document, similarity_score)
        If with_scores is False:
            List of Document objects similar to the query
    """
    try:
        logger.info(f"Searching for chunks similar to query: {query[:50]}...")
        logger.info(f"Connecting to Milvus at {milvus_host}:{milvus_port}")

        # Create the EmbedStore with the specified parameters
        embed_store = EmbedStore(
            collection_name=collection_name,
            model_name=model_name,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            connection_alias=connection_alias
        )

        # Check if the collection exists and has documents
        stats = embed_store.get_collection_stats()
        if not stats["exists"] or stats["count"] == 0:
            logger.warning(f"Collection '{collection_name}' does not exist or is empty")
            return []

        logger.info(f"Found collection with {stats['count']} documents")

        # Search for similar chunks
        if with_scores:
            similar_chunks = embed_store.search_with_scores(
                query=query,
                top_k=top_k,
                filter=filter
            )
        else:
            similar_chunks = embed_store.search_similar(
                query=query,
                top_k=top_k,
                filter=filter
            )

        logger.info(f"Retrieved {len(similar_chunks)} similar chunks")
        return similar_chunks

    except Exception as e:
        logger.error(f"Error in find_similar_chunks: {str(e)}")
        return []

def main():
    """
    Main function to demonstrate the usage of the two main functions:
    embed_documents_from_folder and find_similar_chunks.
    """
    # Define the path to documents
    documents_folder = r"..\..\storage\documents\user_123456"
    file_names = ["Artificial Intelligence.docx", "Atif.pdf"]  # Example file names
    collection_name = "healthcare_docs"
    milvus_host = "localhost"
    milvus_port = 19530

    # Step 1: Embed documents from the folder
    print(f"\n--- Embedding Specific Documents from Folder ---")
    document_ids = embed_documents_from_folder(
        folder_path=documents_folder,
        file_names=file_names,
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port
    )
    print(f"Embedded {len(document_ids)} document chunks")

    # Step 2: Find similar chunks for a query
    query = "How is AI used in the healthcare sector?"
    print(f"\n--- Finding Similar Chunks for Query ---")
    print(f"Query: {query}")

    # Without scores
    similar_chunks = find_similar_chunks(
        query=query,
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        top_k=3
    )

    # Print results
    print(f"\n--- Retrieved {len(similar_chunks)} Similar Chunks ---")
    for i, doc in enumerate(similar_chunks):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)

        # Print metadata if available
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")

    # With scores
    chunks_with_scores = find_similar_chunks(
        query=query,
        collection_name=collection_name,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        top_k=3,
        with_scores=True
    )

    # Print results with scores
    print(f"\n--- Retrieved {len(chunks_with_scores)} Chunks with Similarity Scores ---")
    for i, (doc, score) in enumerate(chunks_with_scores):
        print(f"\nResult {i+1} (Similarity Score: {score:.4f}):")
        print(f"Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else doc.page_content)

if __name__ == "__main__":
    main()
