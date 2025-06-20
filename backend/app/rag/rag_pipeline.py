"""
RAG Pipeline Implementation - Enhanced Version with Hybrid Retrieval

This module implements an enhanced RAG (Retrieval-Augmented Generation) pipeline
with hybrid retrieval (dense + sparse), conservative query enhancement, and advanced filtering.
"""
from typing import List, Optional, Dict, Any, Union, Tuple
import os
import asyncio
import logging
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.documents import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .embed_db import find_similar_chunks
from .prompts import (
    format_document_context,
    qa_answer_prompt,
    summarize_documents_prompt,
    detect_language
)

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY", hf_api_key)

class BM25Retriever:
    """Simple BM25 implementation for sparse retrieval."""

    def __init__(self, documents: List[Document], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

        if documents:
            self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization supporting Arabic and English."""
        # Arabic word pattern + English words
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        english_words = re.findall(r'[A-Za-z]+', text)
        return [word.lower() for word in arabic_words + english_words if len(word) > 1]

    def _build_index(self):
        """Build BM25 index."""
        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc.page_content) for doc in self.documents]

        # Calculate document frequencies and lengths
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Build vocabulary and document frequencies
        vocab = set()
        for doc in tokenized_docs:
            vocab.update(doc)

        # Calculate IDF for each term
        N = len(self.documents)
        for term in vocab:
            df = sum(1 for doc in tokenized_docs if term in doc)
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # Store term frequencies for each document
        self.doc_freqs = []
        for doc in tokenized_docs:
            freq_dict = Counter(doc)
            self.doc_freqs.append(freq_dict)

    def get_scores(self, query: str) -> List[float]:
        """Calculate BM25 scores for query against all documents."""
        if not self.documents:
            return []

        query_terms = self._tokenize(query)
        scores = []

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]

            for term in query_terms:
                if term in doc_freq:
                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0)

                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * (numerator / denominator)

            scores.append(score)

        return scores

class QueryEnhancer:
    """Conservative query enhancement that stays within context."""

    def __init__(self):
        # Very conservative synonym mapping - only common, safe expansions
        self.arabic_synonyms = {
            'من': ['مين'],  # who
            'ماذا': ['ما', 'إيش'],  # what
            'أين': ['وين'],  # where
            'كيف': ['إزاي'],  # how
            'متى': ['إمتى'],  # when
        }

        self.english_synonyms = {
            'who': ['person'],
            'what': ['which'],
            'where': ['location'],
            'how': ['method', 'way'],
            'when': ['time'],
        }

    def enhance_query_conservative(self, query: str, lang: str) -> List[str]:
        """
        Conservative query enhancement - only adds synonyms for question words.
        Does NOT add context that might change meaning.
        """
        enhanced_queries = [query]  # Always include original

        query_lower = query.lower().strip()

        # Only enhance question words, never add assumptions about entities
        if lang == "ar":
            for word, synonyms in self.arabic_synonyms.items():
                if word in query_lower:
                    for synonym in synonyms:
                        enhanced_query = query_lower.replace(word, synonym)
                        if enhanced_query != query_lower:
                            enhanced_queries.append(enhanced_query)
        else:
            for word, synonyms in self.english_synonyms.items():
                if word in query_lower:
                    for synonym in synonyms:
                        enhanced_query = query_lower.replace(word, synonym)
                        if enhanced_query != query_lower:
                            enhanced_queries.append(enhanced_query)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for q in enhanced_queries:
            if q not in seen:
                seen.add(q)
                result.append(q)

        return result[:3]  # Maximum 3 variations to avoid noise

    def classify_query_type(self, query: str) -> str:
        """Classify query type for tailored retrieval."""
        query_lower = query.lower()

        # Question words that indicate specific types
        if any(word in query_lower for word in ['من', 'who', 'مين']):
            return 'person_query'
        elif any(word in query_lower for word in ['ما', 'ماذا', 'what', 'إيش']):
            return 'definition_query'
        elif any(word in query_lower for word in ['أين', 'where', 'وين']):
            return 'location_query'
        elif any(word in query_lower for word in ['كيف', 'how', 'إزاي']):
            return 'process_query'
        elif any(word in query_lower for word in ['متى', 'when', 'إمتى']):
            return 'time_query'
        elif any(word in query_lower for word in ['لماذا', 'why', 'ليش']):
            return 'explanation_query'
        else:
            return 'general_query'

class DocumentFilter:
    """Advanced document filtering with clustering and diversity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)

    def cluster_documents(self, docs: List[Document], max_clusters: int = 3) -> Dict[int, List[Document]]:
        """Cluster documents by semantic similarity."""
        if len(docs) <= max_clusters:
            return {i: [doc] for i, doc in enumerate(docs)}

        try:
            # Create TF-IDF vectors
            texts = [doc.page_content for doc in docs]
            tfidf_matrix = self.vectorizer.fit_transform(texts)

            # Perform clustering
            n_clusters = min(max_clusters, len(docs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Group documents by cluster
            clusters = defaultdict(list)
            for doc, label in zip(docs, cluster_labels):
                clusters[label].append(doc)

            return dict(clusters)

        except Exception as e:
            logger.warning(f"Clustering failed: {e}. Using simple grouping.")
            # Fallback: simple grouping
            return {i: [docs[i]] for i in range(min(max_clusters, len(docs)))}

    def select_diverse_documents(self, clusters: Dict[int, List[Document]],
                               target_count: int) -> List[Document]:
        """Select diverse documents from clusters."""
        selected = []

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

        # Round-robin selection from clusters
        cluster_indices = [0] * len(sorted_clusters)

        while len(selected) < target_count and any(
            cluster_indices[i] < len(cluster_docs)
            for i, (_, cluster_docs) in enumerate(sorted_clusters)
        ):
            for i, (cluster_id, cluster_docs) in enumerate(sorted_clusters):
                if len(selected) >= target_count:
                    break

                if cluster_indices[i] < len(cluster_docs):
                    selected.append(cluster_docs[cluster_indices[i]])
                    cluster_indices[i] += 1

        return selected

    def calculate_temporal_relevance(self, doc: Document) -> float:
        """Calculate temporal relevance score (higher for recent docs)."""
        try:
            # Try to extract date from metadata
            metadata = doc.metadata
            doc_date = None

            # Common date fields
            for date_field in ['date', 'created_at', 'timestamp', 'modified_date']:
                if date_field in metadata:
                    doc_date = metadata[date_field]
                    break

            if not doc_date:
                return 0.5  # Neutral score for undated documents

            # Parse date (handle different formats)
            if isinstance(doc_date, str):
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                    try:
                        doc_date = datetime.strptime(doc_date, fmt)
                        break
                    except ValueError:
                        continue

                if isinstance(doc_date, str):
                    return 0.5  # Could not parse date

            # Calculate recency score (last 2 years = 1.0, older = decay)
            now = datetime.now()
            days_old = (now - doc_date).days

            if days_old < 0:
                return 0.5  # Future dates
            elif days_old <= 365 * 2:  # Within 2 years
                return 1.0 - (days_old / (365 * 2)) * 0.3  # Decay from 1.0 to 0.7
            else:
                return max(0.3, 0.7 - (days_old - 365 * 2) / (365 * 5) * 0.4)  # Further decay

        except Exception:
            return 0.5  # Default neutral score

    def calculate_source_credibility(self, doc: Document) -> float:
        """Calculate source credibility score based on metadata."""
        try:
            metadata = doc.metadata
            score = 0.5  # Base score

            # Check for credibility indicators
            source = metadata.get('source', '').lower()
            author = metadata.get('author', '').lower()

            # Boost for official sources (customize based on your domain)
            official_indicators = ['gov', 'edu', 'official', 'ministry', 'university']
            if any(indicator in source for indicator in official_indicators):
                score += 0.3

            # Boost for verified authors
            if author and len(author) > 0:
                score += 0.1

            # Check document length (very short docs might be less credible)
            content_length = len(doc.page_content)
            if content_length < 100:
                score -= 0.2
            elif content_length > 500:
                score += 0.1

            return min(1.0, max(0.0, score))

        except Exception:
            return 0.5

class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with hybrid retrieval and advanced filtering."""

    def __init__(
        self,
        collection_name: str = "my_collection",
        model_name: str = "tgi",
        base_url: str = "https://m0vtsu71q6nl17e8.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        api_key: str = None,
        max_tokens: int = 1024,
        dense_top_k: int = 15,  # Dense retrieval results
        sparse_weight: float = 0.3,  # Weight for BM25 scores (0.0-1.0)
        final_k: int = 7,  # Final number of documents
        embeddings_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        enable_query_enhancement: bool = True
    ):
        """Initialize the Enhanced RAG Pipeline."""
        self.collection_name = collection_name
        self.dense_top_k = dense_top_k
        self.sparse_weight = sparse_weight
        self.final_k = final_k
        self.max_tokens = max_tokens
        self.embeddings_model = embeddings_model
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.enable_query_enhancement = enable_query_enhancement

        # Initialize components
        self.query_enhancer = QueryEnhancer()
        self.document_filter = DocumentFilter()
        self.bm25_retriever = None  # Will be initialized when needed

        try:
            logger.info(f"Initializing OpenAI client with base_url={base_url}")
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key or openai_api_key
            )
            self.model_name = model_name
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def _hybrid_score_combination(self, dense_scores: List[float],
                                 sparse_scores: List[float]) -> List[float]:
        """Combine dense and sparse scores using weighted average."""
        if not dense_scores or not sparse_scores:
            return dense_scores or sparse_scores

        # Normalize scores to 0-1 range
        def normalize(scores):
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_dense = normalize(dense_scores)
        norm_sparse = normalize(sparse_scores)

        # Weighted combination
        combined = []
        for d, s in zip(norm_dense, norm_sparse):
            combined_score = (1 - self.sparse_weight) * d + self.sparse_weight * s
            combined.append(combined_score)

        return combined

    async def hybrid_retrieve_documents(self, query: str,
                                      query_variants: List[str] = None) -> List[Tuple[Document, float]]:
        """Perform hybrid retrieval combining dense and sparse methods."""
        try:
            all_docs_with_scores = []
            queries_to_process = [query] + (query_variants or [])

            for q in queries_to_process:
                # Step 1: Dense retrieval (embedding-based)
                logger.info(f"Performing dense retrieval for: {q[:50]}")
                dense_results = find_similar_chunks(
                    query=q,
                    collection_name=self.collection_name,
                    model_name=self.embeddings_model,
                    milvus_host=self.milvus_host,
                    milvus_port=self.milvus_port,
                    top_k=self.dense_top_k,
                    with_scores=True,
                    filter=None
                )

                if not dense_results:
                    continue

                docs, dense_scores = zip(*dense_results)
                docs = list(docs)
                dense_scores = list(dense_scores)

                # Step 2: Sparse retrieval (BM25)
                logger.info("Performing sparse retrieval (BM25)")
                if self.bm25_retriever is None:
                    self.bm25_retriever = BM25Retriever(docs)

                sparse_scores = self.bm25_retriever.get_scores(q)

                # Step 3: Combine scores
                if len(sparse_scores) == len(dense_scores):
                    combined_scores = self._hybrid_score_combination(dense_scores, sparse_scores)
                else:
                    logger.warning("Score length mismatch, using dense scores only")
                    combined_scores = dense_scores

                # Add to results
                for doc, score in zip(docs, combined_scores):
                    all_docs_with_scores.append((doc, score, q))

            # Remove duplicates (same document content)
            seen_content = set()
            unique_results = []
            for doc, score, source_query in all_docs_with_scores:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append((doc, score))

            # Sort by combined score
            unique_results.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"Hybrid retrieval completed: {len(unique_results)} unique documents")
            return unique_results

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    def _advanced_document_filtering(self, docs_with_scores: List[Tuple[Document, float]],
                                   query: str, query_type: str) -> List[Document]:
        """Apply advanced filtering with clustering, diversity, and metadata scoring."""
        if not docs_with_scores:
            return []

        docs = [doc for doc, score in docs_with_scores]
        base_scores = [score for doc, score in docs_with_scores]

        logger.info(f"Applying advanced filtering to {len(docs)} documents")

        # Step 1: Calculate additional scores
        temporal_scores = [self.document_filter.calculate_temporal_relevance(doc) for doc in docs]
        credibility_scores = [self.document_filter.calculate_source_credibility(doc) for doc in docs]

        # Step 2: Combine all scores
        final_scores = []
        for i, doc in enumerate(docs):
            # Weighted combination of different factors
            combined_score = (
                0.6 * base_scores[i] +          # Hybrid retrieval score
                0.2 * temporal_scores[i] +       # Temporal relevance
                0.2 * credibility_scores[i]      # Source credibility
            )
            final_scores.append((doc, combined_score))

        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply clustering for diversity
        top_docs = [doc for doc, score in final_scores[:self.final_k * 2]]  # Get more for clustering

        try:
            clusters = self.document_filter.cluster_documents(top_docs, max_clusters=3)
            diverse_docs = self.document_filter.select_diverse_documents(clusters, self.final_k)
            logger.info(f"Selected {len(diverse_docs)} diverse documents from {len(clusters)} clusters")
            return diverse_docs
        except Exception as e:
            logger.warning(f"Clustering failed: {e}. Using top-scored documents.")
            return [doc for doc, score in final_scores[:self.final_k]]

    async def retrieve_documents(self, query: str, with_scores: bool = False) -> Union[List[Document], List[tuple]]:
        """Enhanced document retrieval with hybrid approach."""
        try:
            logger.info(f"Starting enhanced retrieval for: {query[:50]}")

            # Step 1: Query enhancement (conservative)
            query_variants = []
            if self.enable_query_enhancement:
                query_lang = detect_language(query)
                query_type = self.query_enhancer.classify_query_type(query)
                query_variants = self.query_enhancer.enhance_query_conservative(query, query_lang)
                logger.info(f"Query type: {query_type}, Variants: {len(query_variants)}")

            # Step 2: Hybrid retrieval
            docs_with_scores = await self.hybrid_retrieve_documents(query, query_variants[1:])  # Skip original

            if not docs_with_scores:
                logger.warning("No documents retrieved")
                return [] if not with_scores else []

            # Step 3: Advanced filtering
            query_lang = detect_language(query)
            query_type = self.query_enhancer.classify_query_type(query)
            filtered_docs = self._advanced_document_filtering(docs_with_scores, query, query_type)

            logger.info(f"Final retrieval result: {len(filtered_docs)} documents")

            # Return format
            if with_scores:
                # Find original scores for filtered docs
                result = []
                for filtered_doc in filtered_docs:
                    for doc, score in docs_with_scores:
                        if doc.page_content == filtered_doc.page_content:
                            result.append((filtered_doc, score))
                            break
                return result
            else:
                return filtered_docs

        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {str(e)}")
            return []

    async def _call_llm(self, prompt: str, stream: bool = False) -> str:
        """Call the LLM using the OpenAI API."""
        try:
            logger.info(f"Calling LLM with prompt length: {len(prompt)}")

            if stream:
                response_text = ""
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    stream=True
                )

                for message in chat_completion:
                    chunk = message.choices[0].delta.content
                    if chunk:
                        response_text += chunk

                return response_text
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    stream=False
                )

                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    async def generate_answer(self, query: str, docs: List[Document]) -> str:
        """Generate an answer based on the query and filtered documents."""
        try:
            if not docs:
                logger.info("No documents found, returning default message")
                return "No relevant content found."

            # Detect query language
            query_lang = detect_language(query)

            # Format document context
            context = format_document_context(docs, query_lang)

            # Determine query type for prompt selection
            query_type = self.query_enhancer.classify_query_type(query)
            is_general_question = query_type == 'general_query' or len(query.split()) <= 3

            logger.info(f"Query type: {query_type}, General: {is_general_question}")

            # Select appropriate prompt
            if is_general_question:
                prompt_text = summarize_documents_prompt(context, query, query_lang)
            else:
                prompt_text = qa_answer_prompt(context, query, query_lang)

            logger.info("Prompt prepared, invoking LLM")

            # Generate answer
            answer = await self._call_llm(prompt_text)

            logger.info(f"LLM response generated: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"An error occurred while generating an answer: {str(e)}"

    async def process_query(self, query: str, stream_output: bool = False) -> Dict[str, Any]:
        """Process a user query through the enhanced RAG pipeline."""
        result = {
            "original_query": query,
            "query_enhancement_enabled": self.enable_query_enhancement,
            "documents_retrieved": 0,
            "documents_filtered": 0,
            "response": "",
        }

        try:
            logger.info(f"Processing query with enhanced pipeline: {query}")

            # Retrieve and filter relevant documents
            logger.info("Retrieving and filtering documents...")
            docs = await self.retrieve_documents(query)
            result["documents_retrieved"] = self.dense_top_k
            result["documents_filtered"] = len(docs)

            # Add document content to result
            if docs:
                result["documents"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in docs
                ]

            # Generate answer
            logger.info("Generating answer...")

            if stream_output:
                # Setup streaming context
                if not docs:
                    result["response"] = "No relevant content found."
                    return result

                query_lang = detect_language(query)
                context = format_document_context(docs, query_lang)
                query_type = self.query_enhancer.classify_query_type(query)
                is_general_question = query_type == 'general_query' or len(query.split()) <= 3

                if is_general_question:
                    prompt_text = summarize_documents_prompt(context, query, query_lang)
                else:
                    prompt_text = qa_answer_prompt(context, query, query_lang)

                result["streaming_context"] = {
                    "prompt": prompt_text,
                    "model": self.model_name,
                    "max_tokens": self.max_tokens
                }
            else:
                answer = await self.generate_answer(query, docs)
                result["response"] = answer

            logger.info("Enhanced query processing complete")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            result["error"] = str(e)
            return result


# Create singleton instance
default_enhanced_pipeline = None
try:
    logger.info("Creating default enhanced RAG pipeline instance")
    default_enhanced_pipeline = EnhancedRAGPipeline()
    logger.info("Default enhanced pipeline created successfully")
except Exception as e:
    logger.error(f"Failed to create default enhanced pipeline: {str(e)}")


async def process_query(
    query: str,
    stream_output: bool = False,
    pipeline: Optional[EnhancedRAGPipeline] = None
) -> Dict[str, Any]:
    """Process a query using the enhanced RAG pipeline."""
    try:
        logger.info(f"Enhanced process_query called with: {query}")

        if pipeline is None:
            if default_enhanced_pipeline is None:
                logger.warning("Default enhanced pipeline not available, creating new instance")
                pipeline = EnhancedRAGPipeline()
            else:
                pipeline = default_enhanced_pipeline

        return await pipeline.process_query(query, stream_output)
    except Exception as e:
        logger.error(f"Error in enhanced process_query: {str(e)}", exc_info=True)
        return {
            "original_query": query,
            "documents_retrieved": 0,
            "documents_filtered": 0,
            "response": "",
            "error": f"Error in enhanced process_query: {str(e)}"
        }


async def stream_rag_response(query: str):
    """Stream a RAG response using the enhanced pipeline."""
    result = await process_query(query, stream_output=True)

    if "error" in result:
        yield f"Error: {result['error']}"
        return

    if "streaming_context" not in result:
        yield result.get("response", "No response generated")
        return

    context = result["streaming_context"]

    try:
        chat_completion = default_enhanced_pipeline.client.chat.completions.create(
            model=context["model"],
            messages=[{"role": "user", "content": context["prompt"]}],
            max_tokens=context["max_tokens"],
            stream=True
        )

        for message in chat_completion:
            chunk = message.choices[0].delta.content
            if chunk:
                yield chunk
    except Exception as e:
        yield f"Error during streaming: {str(e)}"
