"""
Main Code
Advanced Contextual Document Chunking for Arabic & English Documents

This module provides advanced contextual chunking strategies that preserve semantic coherence
and improve RAG performance for Arabic and English documents.
"""
import os
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
from datetime import datetime
from collections import Counter

# Langchain imports
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter
)

try:
    import pyarabic.araby as araby
    ARABIC_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("pyarabic library not available. Basic Arabic processing will be used.")
    ARABIC_LIBS_AVAILABLE = False

# Language detection imports
try:
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect library not available. Language detection will be limited.")
    LANG_DETECT_AVAILABLE = False

import unicodedata

from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredHTMLLoader
)

class ArabicTextProcessor:
    """
    Utility class for Arabic text detection and processing.
    """

    @staticmethod
    def contains_arabic(text: str) -> bool:
        """
        Check if the input text contains significant Arabic content.

        Returns True if:
        - Arabic characters are found, and
        - There are more than 10 Arabic characters, or
        - Arabic characters make up more than 20% of the text
        """
        if not text:
            return False

        # Arabic Unicode ranges
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        has_arabic = bool(arabic_pattern.search(text))

        if not has_arabic:
            return False

        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return arabic_chars > 10 or (arabic_chars / len(text) > 0.2)

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English."""
    if not text or len(text.strip()) < 10:
        return "unknown"

    if ArabicTextProcessor.contains_arabic(text):
        return 'ar'

    if LANG_DETECT_AVAILABLE:
        try:
            detected = detect(text)
            return detected if detected == 'en' else 'unknown'
        except LangDetectException:
            pass

    if re.search(r'[a-zA-Z]', text):
        return 'en'

    return "unknown"

# Common abbreviations that don't end sentences (mainly for English)
COMMON_ABBREVIATIONS = {
    'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'etc.', 'e.g.',
    'i.e.', 'vs.', 'fig.', 'st.', 'ave.', 'inc.', 'ltd.', 'co.',
    'ph.d.', 'm.d.', 'a.m.', 'p.m.', 'u.s.', 'u.k.'
}
ARABIC_SENTENCE_ENDINGS = r'([.؟!:\u061F])'

@lru_cache(maxsize=1000)
def segment_into_sentences(text: str, lang: str) -> List[str]:
    if not text:
        return []

    if lang == 'en':
        # Fast path: split on newline or period
        simple_splits = re.split(r'[\n\.!?]+', text)
        return [s.strip() for s in simple_splits if s.strip()]

    elif lang == 'ar':
        segments = re.split(ARABIC_SENTENCE_ENDINGS, text)
        sentences = [segments[i] + segments[i+1] for i in range(0, len(segments) - 1, 2)]
        if len(segments) % 2 != 0:
            sentences.append(segments[-1])
        return [s.strip() for s in sentences if s.strip()]

    else:
        return [s.strip() for s in re.split(r'[\n\.!?]+', text) if s.strip()]


def estimate_token_count(text: str, lang: str) -> int:
    """
    Estimate token count for a given text.

    Args:
        text: Text to estimate tokens for
        lang: Language code ('ar' or 'en')

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    if lang == 'ar':
        # For Arabic: average 4.2 characters per token based on research
        return int(len(text) / 4.2)
    else:
        # For English: average 4.7 characters per token
        return int(len(text) / 4.7)

def get_paragraph_importance(paragraph: str, lang: str) -> float:
    """
    Estimate paragraph importance for better chunking decisions.

    Args:
        paragraph: Text paragraph
        lang: Language code ('ar' or 'en')

    Returns:
        Importance score (0-1)
    """
    if not paragraph or len(paragraph.strip()) < 5:
        return 0.0

    # Length-based importance (longer paragraphs often contain more info)
    length_score = min(1.0, len(paragraph) / 100)

    # Keyword-based importance
    keywords = {}
    if lang == 'en':
        keywords = {
            'important': 0.7, 'significant': 0.7, 'critical': 0.8,
            'key': 0.7, 'main': 0.6, 'essential': 0.7, 'crucial': 0.8,
            'conclusion': 0.8, 'therefore': 0.6, 'summary': 0.8,
            'findings': 0.7, 'results': 0.7
        }
    elif lang == 'ar':
        keywords = {
            'مهم': 0.7, 'أساسي': 0.7, 'ضروري': 0.8, 'رئيسي': 0.7,
            'خلاصة': 0.8, 'نتيجة': 0.6, 'ملخص': 0.8, 'نتائج': 0.7
        }

    keyword_score = 0.0
    for keyword, score in keywords.items():
        if keyword in paragraph.lower():
            keyword_score = max(keyword_score, score)

    # Count special markers like numbers and bullet points
    has_numbers = bool(re.search(r'\d', paragraph))
    has_bullets = bool(re.search(r'•|\*|[-–—]\s', paragraph))
    structure_score = 0.3 if has_numbers else 0.0
    structure_score += 0.3 if has_bullets else 0.0

    # Combine scores with weights
    final_score = (0.5 * length_score + 0.3 * keyword_score + 0.2 * min(structure_score, 1.0))
    return min(final_score, 1.0)  # Cap at 1.0

def calculate_adaptive_chunk_size(text: str, lang: str) -> Tuple[int, int]:
    """
    Calculate adaptive chunk size and overlap based on text properties.

    Args:
        text: Text content
        lang: Language code ('ar' or 'en')

    Returns:
        (chunk_size, chunk_overlap) tuple with token counts
    """
    # Base sizes
    if lang == 'ar':
        base_chunk_size = 600
        base_overlap = 150
    else:  # English
        base_chunk_size = 800
        base_overlap = 200

    # Adjust based on text complexity
    avg_sentence_length = len(text) / max(1, len(segment_into_sentences(text, lang)))

    # Adjust size based on average sentence length
    if avg_sentence_length > 30:  # Long sentences
        chunk_size = int(base_chunk_size * 0.8)  # Smaller chunks for complex text
        overlap = int(base_overlap * 1.2)  # More overlap for context
    elif avg_sentence_length < 15:  # Short sentences
        chunk_size = int(base_chunk_size * 1.2)  # Larger chunks for simpler text
        overlap = int(base_overlap * 0.8)  # Less overlap needed
    else:
        chunk_size = base_chunk_size
        overlap = base_overlap

    return chunk_size, overlap


class SemanticTextSplitter(TextSplitter):
    def __init__(
        self,
        language: str = "en",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        **kwargs
    ):
        self.language = language
        # Only pass the parameters that TextSplitter expects
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
        self.min_chunk_size = 50  # If needed, set a default
        self.separator = " "  # If needed, set a default
    def _is_heading(self, text: str) -> bool:
        """
        Determine if a line of text is likely a heading.

        Args:
            text: Line of text to check

        Returns:
            True if the line appears to be a heading
        """
        if not text:
            return False

        # Check for heading patterns
        heading_patterns = [
            # Markdown heading patterns
            r'^#+\s+.+',  # Markdown heading with #
            # HTML-like heading patterns
            r'^<h[1-6]>.*</h[1-6]>$',
            # Numbered heading patterns
            r'^[0-9]+\.[0-9.]*\s+.+',  # Numbered heading (1.1, 1.2.3, etc.)
            # Capitalized short line might be a heading
            r'^[A-Z][^.!?]{3,40}$',  # Capitalized text, not too long, no endings
            # Check for Arabic heading patterns if the text contains Arabic
            r'^[\u0600-\u06FF].*:$',  # Arabic text ending with colon
        ]

        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True

        # Length-based heuristic - very short standalone lines might be headings
        if len(text) < 60 and text.count(' ') < 7:
            return True

        return False

    def generate_chunk_titles(self, chunks: List[str]) -> List[str]:
        """
        Generate meaningful titles for chunks to aid navigation and context.

        Args:
            chunks: List of text chunks

        Returns:
            List of titles corresponding to the chunks
        """
        titles = []
        for i, chunk in enumerate(chunks):
            # Default title
            title = f"Chunk {i+1}"

            # Try to extract a meaningful title
            if chunk:
                # Check for headings first
                lines = chunk.split('\n')
                first_line = lines[0].strip()

                # If first line looks like a heading, use it
                if self._is_heading(first_line):
                    # Truncate if too long
                    if len(first_line) > 50:
                        title = first_line[:50] + "..."
                    else:
                        title = first_line
                else:
                    # Otherwise use first sentence or part of it
                    lang = detect_language(chunk)
                    sentences = segment_into_sentences(chunk, lang)

                    if sentences:
                        first_sentence = sentences[0].strip()
                        if first_sentence:
                            if len(first_sentence) > 40:
                                title = first_sentence[:40] + "..."
                            else:
                                title = first_sentence

            titles.append(title)
        return titles

    def extract_semantic_headers(self, text: str) -> List[str]:
        """
        Extract semantic headers/titles from text.
        Useful for building table of contents.

        Args:
            text: Document text

        Returns:
            List of headers found in the text
        """
        headers = []

        # Split text into lines
        lines = text.split('\n')

        # Check each line for heading patterns
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_heading(line):
                headers.append(line)

        return headers

    def split_text(self, text: str) -> List[str]:
        """
        Split text semantically, preserving natural boundaries.
        Optimized version with reduced computational complexity.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Detect language if not specified
        lang = self.language
        if lang not in ["ar", "en"]:
            lang = detect_language(text)
            if lang == "unknown":
                logger.warning("Unknown language. Using English chunking rules.")
                lang = "en"

        # Calculate adaptive chunk parameters based on text content
        target_size, target_overlap = calculate_adaptive_chunk_size(text, lang)

        # Split text into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Extract semantic headers from the text for TOC (single pass)
        headers = self.extract_semantic_headers(text)

        # Process paragraphs and create segments in single pass
        segments = []
        importance_scores = []

        for paragraph in paragraphs:
            # Calculate importance once per paragraph
            paragraph_importance = get_paragraph_importance(paragraph, lang)

            if len(paragraph) <= target_size:
                segments.append(paragraph)
                importance_scores.append(paragraph_importance)
            else:
                # Split long paragraphs into sentences (single call per paragraph)
                sentences = segment_into_sentences(paragraph, lang)
                segments.extend(sentences)
                # Assign same importance to all sentences from this paragraph
                importance_scores.extend([paragraph_importance] * len(sentences))

        # Create chunks with semantic awareness and overlapping in single pass
        chunks = []
        overlapped_chunks = []
        current_chunk = []
        current_size = 0

        # Pre-calculate sentence splits for overlap creation (cache to avoid re-computation)
        segment_sentences_cache = {}

        for i, segment in enumerate(segments):
            segment_size = len(segment)
            importance = importance_scores[i]

            # Keep important segments together if possible
            if current_size + segment_size <= target_size * 1.2:
                current_chunk.append(segment)
                current_size += segment_size
            elif importance > 0.7 and current_size < target_size * 0.5:
                current_chunk.append(segment)
                current_size += segment_size
            elif importance > 0.8 and segment_size > target_size:
                # Very important but large segment, make it its own chunk
                if current_chunk:
                    chunk_text = self.separator.join(current_chunk)
                    chunks.append(chunk_text)
                    overlapped_chunks.append(chunk_text)

                    # Create overlap with next segment immediately
                    if len(chunk_text) >= self.min_chunk_size:
                        self._create_overlap_with_next(chunk_text, segment, overlapped_chunks, lang)

                # Add the large segment as its own chunk
                chunks.append(segment)
                overlapped_chunks.append(segment)
                current_chunk = []
                current_size = 0
            else:
                # Start a new chunk, but first handle overlap
                if current_chunk:
                    chunk_text = self.separator.join(current_chunk)
                    chunks.append(chunk_text)
                    overlapped_chunks.append(chunk_text)

                    # Create overlap with next segment immediately
                    if len(chunk_text) >= self.min_chunk_size:
                        self._create_overlap_with_next(chunk_text, segment, overlapped_chunks, lang)

                current_chunk = [segment]
                current_size = segment_size

        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = self.separator.join(current_chunk)
            chunks.append(chunk_text)
            overlapped_chunks.append(chunk_text)

        # Generate meaningful titles for chunks (single batch operation)
        chunk_titles = self.generate_chunk_titles(overlapped_chunks)

        # Store titles for later use
        self.chunk_titles = chunk_titles

        # Filter by minimum chunk size and return
        return [chunk for chunk in overlapped_chunks if len(chunk) >= self.min_chunk_size]

    def _create_overlap_with_next(self, current_chunk: str, next_segment: str,
                                overlapped_chunks: List[str], lang: str) -> None:
        """
        Helper method to create overlap between current chunk and next segment.
        Optimized to avoid repeated sentence splitting.
        """
        try:
            # Get end of current chunk (limit sentence splitting)
            current_sentences = segment_into_sentences(current_chunk, lang)
            if len(current_sentences) > 3:
                bridge_end = current_sentences[-3:]
            else:
                bridge_end = current_sentences

            # For next segment, if it's short, use whole segment, otherwise split
            if len(next_segment) <= 200:  # Short segment threshold
                bridge_start = [next_segment]
            else:
                next_sentences = segment_into_sentences(next_segment, lang)
                bridge_start = next_sentences[:min(3, len(next_sentences))]

            # Create bridge if meaningful
            bridge_text = f"{self.separator.join(bridge_end)}{self.separator}{self.separator.join(bridge_start)}"
            if len(bridge_text) > self.min_chunk_size:
                overlapped_chunks.append(bridge_text)

        except Exception as e:
            # Fallback: skip overlap creation if there's an error
            logger.warning(f"Failed to create overlap: {e}")
            pass

class ContextualDocumentSplitter:
    """
    Advanced document splitter with context preservation for bilingual documents.
    """
    def __init__(
        self,
        language: str = None,
        chunk_size_ar: int = 600,
        chunk_overlap_ar: int = 150,
        chunk_size_en: int = 800,
        chunk_overlap_en: int = 200
    ):
        """
        Initialize contextual document splitter.

        Args:
            language: Optional language override ('ar' or 'en')
            chunk_size_ar: Target chunk size for Arabic text
            chunk_overlap_ar: Target chunk overlap for Arabic text
            chunk_size_en: Target chunk size for English text
            chunk_overlap_en: Target chunk overlap for English text
        """
        self.language = language
        self.chunk_size = {"ar": chunk_size_ar, "en": chunk_size_en}
        self.chunk_overlap = {"ar": chunk_overlap_ar, "en": chunk_overlap_en}

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents with context preservation.

        Args:
            documents: List of Document objects

        Returns:
            List of contextually split Document objects
        """
        if not documents:
            return []

        result_docs = []

        for doc in documents:
            # Get document language and content
            text = doc.page_content
            metadata = doc.metadata.copy()

            # Detect language if not specified
            lang = self.language or metadata.get("language")
            if not lang or lang not in ["ar", "en"]:
                lang = detect_language(text)
                if lang == "unknown":
                    logger.warning(f"Document {metadata.get('source', 'unknown')} has unknown language. Skipping.")
                    continue

            # Update metadata with language
            metadata["language"] = lang

            # Extract headers for table of contents
            splitter = SemanticTextSplitter(
                language=lang,
                chunk_size=self.chunk_size.get(lang, 800),
                chunk_overlap=self.chunk_overlap.get(lang, 200)
            )

            # Extract headers for table of contents
            headers = splitter.extract_semantic_headers(text)

            # Split document
            chunks = splitter.split_text(text)

            # Get generated titles
            chunk_titles = getattr(splitter, 'chunk_titles', [])
            if not chunk_titles or len(chunk_titles) != len(chunks):
                # Generate titles if not already available
                chunk_titles = splitter.generate_chunk_titles(chunks)

            # Create new documents with rich metadata
            for i, chunk_text in enumerate(chunks):
                chunk_lang = detect_language(chunk_text)

                # Skip if language changed and is no longer Arabic or English
                if chunk_lang == "unknown":
                    continue

                # Estimate token count
                token_count = estimate_token_count(chunk_text, chunk_lang)

                # Create metadata with semantic info and token metrics
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_language": chunk_lang,
                    "chunk_count": len(chunks),
                    "estimated_tokens": token_count,
                    "chunk_type": "overlap" if "overlap" in metadata.get("chunk_type", "") else "primary",
                    "chunk_title": chunk_titles[i] if i < len(chunk_titles) else f"Chunk {i+1}"
                })

                # Include table of contents in metadata
                if headers:
                    chunk_metadata["document_headers"] = headers

                # Create document with rich metadata
                result_docs.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))

        return result_docs

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a text string with semantic chunking.

        Args:
            text: Text to process
            metadata: Optional metadata for the document

        Returns:
            List of chunked Document objects
        """
        if not text:
            return []

        # Set default metadata
        if metadata is None:
            metadata = {
                "source": "text_input",
                "created_at": str(datetime.now())
            }

        # Create initial document
        doc = Document(page_content=text, metadata=metadata)

        # Process using split_documents
        return self.split_documents([doc])

class ArabicEnglishDocumentProcessor:
    """
    A class to handle loading and processing of documents with specialized
    handling for Arabic and English languages only.

    Supported file types: .txt, .pdf, .docx, .csv, .html, .json, and more.
    """
    def __init__(self, file_path: str = None):
        """
        Initialize document processor with optional file path.

        Args:
            file_path: Path to the document file (optional)
        """
        self.file_path = file_path
        if file_path:
            self.file_extension = os.path.splitext(file_path)[1].lower()
            self.file_name = os.path.basename(file_path)
            self.file_info = {
                'path': file_path,
                'name': self.file_name,
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                                if os.path.exists(file_path) else datetime.now().isoformat(),
                'mime_type': get_mime_type(file_path)
            }
        else:
            self.file_extension = None
            self.file_name = None
            self.file_info = None

    def _get_loader(self):
        """
        Initialize and return the appropriate loader based on the file type.

        Returns:
            loader: The document loader instance.

        Raises:
            ValueError: If the file type is unsupported.
        """
        if self.file_extension == '.txt':
            return TextLoader(self.file_path, encoding='utf-8', autodetect_encoding=True)
        elif self.file_extension == '.pdf':
            return PyMuPDFLoader(self.file_path)
        elif self.file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(self.file_path)
        elif self.file_extension == '.csv':
            return CSVLoader(self.file_path)
        elif self.file_extension in ['.html', '.htm']:
            return UnstructuredHTMLLoader(self.file_path)
        else:
            supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.csv', '.html', '.htm']
            raise ValueError(f"Unsupported file type: {self.file_extension}. Supported types: {', '.join(supported_extensions)}")

    def load_document(self) -> List[Document]:
        """
        Load document and return raw langchain Documents.

        Returns:
            List of Document objects
        """
        if not self.file_path:
            raise ValueError("No file path provided. Set file_path before loading.")

        try:
            loader = self._get_loader()
            documents = loader.load()
            # logger.info(f"Loaded {self.file_extension} file: {self.file_path} ({len(documents)} documents)")
            return documents
        except Exception as e:
            logger.exception(f"Error loading document {self.file_path}: {str(e)}")
            raise

    def process_text(self, text: str) -> List[Document]:
        """
        Process raw text with specialized language handling.
        Only processes Arabic and English text, ignores other languages.

        Args:
            text: Raw text to process

        Returns:
            List of processed Document chunks (empty if not Arabic/English)
        """
        # Detect language
        lang = detect_language(text)

        # If not Arabic or English, return empty list
        if lang == "unknown":
            logger.warning("Text is neither Arabic nor English. Skipping processing.")
            return []

        # logger.info(f"Processing text in detected language: {lang}")

        # Process text based on language
        if lang == 'ar':
            processed_text = clean_arabic_text(text)
        else:  # English
            processed_text = clean_english_text(text)

        # Split into chunks using language-aware splitter
        chunks = split_text_by_language(processed_text , lang)
        # logger.info(f"Text split into {len(chunks)} chunks")

        # If no chunks were created (all content was filtered), return empty list
        if not chunks:
            logger.info("No valid Arabic or English chunks found")
            return []

        # If no file info, create minimal metadata
        if not self.file_info:
            file_info = {
                'path': 'text_input',
                'name': 'text_input',
                'size': len(text),
                'last_modified': datetime.now().isoformat(),
                'mime_type': 'text/plain'
            }
        else:
            file_info = self.file_info

        # Enrich each chunk with metadata
        processed_chunks = []
        for j, chunk in enumerate(chunks):
            enriched_chunk = enrich_document_metadata(chunk, file_info, j)
            processed_chunks.append(enriched_chunk)

        return processed_chunks

    def load_and_process(self) -> List[Document]:
        """
        Load, process, and chunk document with specialized language handling.
        Only processes Arabic and English content, ignores other languages.

        Returns:
            List of processed and chunked Document objects
        """
        if not self.file_path:
            raise ValueError("No file path provided. Set file_path before loading.")

        # Load raw documents
        raw_documents = self.load_document()

        processed_chunks = []
        for i, doc in enumerate(raw_documents):
            # Get content from the document
            text_content = doc.page_content

            # Detect language first
            lang = detect_language(text_content)

            # Skip non-Arabic/English content
            if lang == "unknown":
                logger.info(f"Skipping document {i+1}/{len(raw_documents)}: Content is neither Arabic nor English")
                continue

            # logger.info(f"Processing document {i+1}/{len(raw_documents)} in language: {lang}")

            # Process the text content
            chunks = self.process_text(text_content)

            # Add to our collection
            processed_chunks.extend(chunks)

        # logger.info(f"Final document processing complete: {len(processed_chunks)} total chunks")
        return processed_chunks

    def set_file_path(self, file_path: str) -> None:
        """
        Set or update the file path.

        Args:
            file_path: New file path
        """
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()
        self.file_name = os.path.basename(file_path)
        self.file_info = {
            'path': file_path,
            'name': self.file_name,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            if os.path.exists(file_path) else datetime.now().isoformat(),
            'mime_type': get_mime_type(file_path)
        }


        # Load document using the existing processor
        processor = ArabicEnglishDocumentProcessor(file_path)
        raw_docs = processor.load_document()

        # Handle each document with contextual splitting
        all_chunks = []

        for raw_doc in raw_docs:
            text = raw_doc.page_content
            lang = detect_language(text)

            if lang == "unknown":
                logger.warning(f"Document {file_path} has unknown language. Skipping.")
                continue

            # Create semantic splitter with language-appropriate settings
            contextual_splitter = ContextualDocumentSplitter(language=lang)

            # Create a document with appropriate metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "language": lang,
                    "filename": processor.file_name
                }
            )

            # Split with contextual awareness
            chunks = contextual_splitter.split_documents([doc])

            # Enrich with context from adjacent chunks
            enriched_chunks = enrich_chunks_with_context(chunks)

            all_chunks.extend(enriched_chunks)

        return all_chunks


def enrich_chunks_with_context(chunks: List[Document]) -> List[Document]:
    """
    Enrich chunks with additional context from neighboring chunks.

    Args:
        chunks: List of Document objects

    Returns:
        List of enriched Document objects
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    enriched_chunks = []

    # Group chunks by source
    source_groups = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(chunk)

    # Process each source group
    for source, group in source_groups.items():
        # Sort by chunk index
        sorted_chunks = sorted(group, key=lambda x: x.metadata.get("chunk_index", 0))

        for i, chunk in enumerate(sorted_chunks):
            # Prepare context enrichment
            context = {
                "previous_chunk_summary": "",
                "next_chunk_summary": "",
                "document_position": "middle"
            }

            # Get language
            lang = chunk.metadata.get("language", "en")

            # Position in document
            if i == 0:
                context["document_position"] = "beginning"
            elif i == len(sorted_chunks) - 1:
                context["document_position"] = "end"

            # Get summaries from adjacent chunks
            if i > 0:
                prev_chunk = sorted_chunks[i-1]
                context["previous_chunk_summary"] = summarize_text(prev_chunk.page_content, lang)

            if i < len(sorted_chunks) - 1:
                next_chunk = sorted_chunks[i+1]
                context["next_chunk_summary"] = summarize_text(next_chunk.page_content, lang)

            # Update metadata with context
            enriched_metadata = chunk.metadata.copy()
            enriched_metadata.update(context)

            # Create new document with enriched metadata
            enriched_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=enriched_metadata
            ))

    return enriched_chunks

def summarize_text(text: str, lang: str, max_length: int = 50) -> str:
    """
    Create a simple extractive summary of text for context.

    Args:
        text: Text to summarize
        lang: Language code ('ar' or 'en')
        max_length: Maximum summary length in characters

    Returns:
        Brief summary
    """
    if not text or len(text) <= max_length:
        return text[:max_length]

    # Simple keyword-based extractive summary
    sentences = segment_into_sentences(text, lang)

    if not sentences:
        return text[:max_length]

    # Get first sentence if it's short enough
    if len(sentences[0]) <= max_length:
        return sentences[0]

    # Extract keywords for a basic summary
    if lang == 'en':
        words = re.findall(r'\b\w+\b', text.lower())
    else:  # Arabic
        words = re.findall(r'[\u0600-\u06FF]+', text)

    # Remove common words
    stopwords_en = {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'with', 'by', 'as', 'on', 'at'}
    stopwords_ar = {'من', 'في', 'على', 'و', 'ال', 'إلى', 'أن', 'عن', 'مع'}

    stopwords = stopwords_en if lang == 'en' else stopwords_ar
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]

    # Count word frequency
    word_counts = Counter(filtered_words)
    top_words = [word for word, _ in word_counts.most_common(5)]

    # Create a simple summary with top words
    if lang == 'en':
        summary = f"Contains: {', '.join(top_words)}"
    else:  # Arabic
        summary = f"يحتوي على: {', '.join(top_words)}"

    return summary[:max_length]

def process_document_with_context(file_path: str) -> List[Document]:
    """
    Process a document with advanced contextual chunking.

    Args:
        file_path: Path to the document

    Returns:
        List of processed Document chunks with context
    """
    # Load document using the existing processor
    processor = ArabicEnglishDocumentProcessor(file_path)
    raw_docs = processor.load_document()

    # Handle each document with contextual splitting
    all_chunks = []

    for raw_doc in raw_docs:
        text = raw_doc.page_content
        lang = detect_language(text)

        if lang == "unknown":
            logger.warning(f"Document {file_path} has unknown language. Skipping.")
            continue

        # Create semantic splitter with language-appropriate settings
        contextual_splitter = ContextualDocumentSplitter(language=lang)

        # Create a document with appropriate metadata
        doc = Document(
            page_content=text,
            metadata={
                "source": file_path,
                "language": lang,
                "filename": processor.file_info['name']
            }
        )

        # Split with contextual awareness
        chunks = contextual_splitter.split_documents([doc])

        # Enrich with context from adjacent chunks
        enriched_chunks = enrich_chunks_with_context(chunks)

        all_chunks.extend(enriched_chunks)

    return all_chunks

MIME_TYPE_MAP = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

def get_mime_type(filename: str) -> str:
    """
    Get MIME type based on file extension.

    Args:
        filename: Name of the file

    Returns:
        MIME type string or 'application/octet-stream' if unknown
    """
    ext = os.path.splitext(filename.lower())[1]
    return MIME_TYPE_MAP.get(ext, "application/octet-stream")

def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text by removing diacritics, tatweel, and ligatures.

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized Arabic text
    """
    if not text:
        return ""

    if ARABIC_LIBS_AVAILABLE:
        # Use pyarabic for advanced normalization
        text = araby.normalize_hamza(text)         # Normalize hamza
        text = araby.strip_tashkeel(text)          # Remove diacritics (tashkeel)
        text = araby.strip_tatweel(text)           # Remove tatweel (kashida)
        text = araby.normalize_ligature(text)      # Normalize lam-alef ligatures
    else:
        # Fallback: manual normalization
        text = re.sub(r'[\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)  # Diacritics
        text = text.replace('\u0640', '')  # Remove tatweel

    return text

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)


def clean_arabic_text(text: str) -> str:
    """
    Clean Arabic text by normalizing and removing emojis, HTML, non-Arabic chars, etc.

    Args:
        text: Arabic text to clean

    Returns:
        Cleaned Arabic text
    """
    if not text:
        return ""

    # Normalize text
    text = normalize_arabic_text(text)

    # Normalize Unicode form
    text = unicodedata.normalize("NFKC", text)

    # Remove emojis

    text = EMOJI_PATTERN.sub(' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove non-Arabic characters (keep Arabic range and space)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_english_text(text: str) -> str:
    """
    Clean English text by normalizing, removing emojis, special characters, etc.

    Args:
        text: English text to clean

    Returns:
        Cleaned English text
    """
    if not text:
        return ""

    # Normalize Unicode form and convert to lowercase
    text = unicodedata.normalize("NFKC", text).lower()

    # Remove emojis
    text = EMOJI_PATTERN.sub(' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove punctuation/special characters (keep alphanumeric and space)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def prepare_text_for_embedding(text: str) -> str:
    """
    Prepare text for embedding by applying appropriate language processing.
    Only processes Arabic and English text.

    Args:
        text: Raw text

    Returns:
        Processed text ready for embedding or empty string if not Arabic/English
    """
    # Detect language
    lang = detect_language(text)

    if lang == 'ar':
        # Apply Arabic processing
        return clean_arabic_text(text)
    elif lang == 'en':
        # Apply English processing
        return clean_english_text(text)
    else:
        # For unknown languages, return empty string to signal it should be skipped
        logger.warning("Detected non-Arabic/English text, returning empty string")
        return ""


def create_language_aware_splitter(lang: str) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter optimized for Arabic or English.

    Args:
        lang: Language code ('ar' or 'en')

    Returns:
        Configured RecursiveCharacterTextSplitter
    """
    # Default separators
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    if lang == 'ar':
        # Arabic-specific configuration
        arabic_separators = [
            ".", "،", "؛", ":", "؟", "!", "\n\n", "\n", " ", ""
        ]
        separators = arabic_separators + separators

        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=600,  # Smaller size for Arabic for better handling
            chunk_overlap=150,
            length_function=len
        )
    else:  # English or default
        return RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

def split_text_by_language(text: str , lang: str = None) -> List[Document]:
    """
    Split text into chunks, optimizing for Arabic or English.
    Skip text in other languages.

    Args:
        text: Text to split

    Returns:
        List of Document objects
    """
    # Detect the language
    if lang is None:
      lang = detect_language(text)

    # If it's neither Arabic nor English, return empty list
    if lang == "unknown":
        logger.warning("Text is neither Arabic nor English. Skipping processing.")
        return []

    # logger.info(f"Splitting text in detected language: {lang}")

    # Create a language-optimized splitter
    splitter = create_language_aware_splitter(lang)

    # Split the text
    chunks = splitter.split_text(text)

    # Filter out low-quality chunks
    quality_chunks = []
    for chunk in chunks:
        # Skip chunks that are mostly dots or form fields
        dots_pattern = r'\.{5,}'
        dots_matches = re.findall(dots_pattern, chunk)
        total_dots_length = sum(len(match) for match in dots_matches)

        # If more than 50% is dots or form fields, skip
        if total_dots_length / max(len(chunk), 1) > 0.5:
            logger.debug("Skipping low-quality chunk (mostly form fields)")
            continue

        # Skip very small chunks
        if len(chunk.strip()) < 20:
            continue
        quality_chunks.append(chunk)

    # Convert to Documents with language metadata
    documents = []
    for chunk in quality_chunks:
        # Detect language of each chunk
        chunk_lang = detect_language(chunk)

        # Process the chunk based on language
        processed_chunk = prepare_text_for_embedding(chunk)

        # Create Document with language metadata
        doc = Document(
            page_content=processed_chunk,
            metadata={
                "language": lang,
                "chunk_language": chunk_lang,
                "original_length": len(chunk),
                "processed_length": len(processed_chunk)
            }
        )
        documents.append(doc)

    return documents

def enrich_document_metadata(doc: Document, file_info: Dict[str, Any], chunk_index: int) -> Document:
    """
    Enrich a document with detailed metadata.

    Args:
        doc: Document to enrich
        file_info: File information
        chunk_index: Index of this chunk

    Returns:
        Document with enriched metadata
    """
    file_path = file_info.get('path', '')

    # Create a unique ID based on file path and chunk index
    content_hash = hash(doc.page_content) % 10**10  # Simple hash
    doc_id = f"{file_path}_chunk_{chunk_index}_{content_hash}"

    # Merge existing metadata with new metadata
    metadata = {
        **doc.metadata,
        "source": file_path,
        "chunk": chunk_index,
        "filename": file_info.get('name', ''),
        "file_size": file_info.get('size', 0),
        "last_modified": file_info.get('last_modified', ''),
        "mime_type": get_mime_type(file_path),
        "indexed_at": datetime.now().isoformat(),
        "id": doc_id
    }

    # Return new document with updated metadata
    return Document(
        page_content=doc.page_content,
        metadata=metadata
    )

def process_file(file_path: str) -> List[Document]:
    """
    Process a single document file with contextual chunking.

    Args:
        file_path: Path to the document file

    Returns:
        List of processed Document chunks with context

    Raises:
        ValueError: If the file doesn't exist or is unsupported
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} does not exist")

    # Get file extension to verify support
    file_extension = os.path.splitext(file_path)[1].lower()
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.csv', '.html', '.htm']

    if file_extension not in supported_extensions:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_extensions)}")

    try:
        # Initialize document processor
        processor = ArabicEnglishDocumentProcessor(file_path)

        # Load the document
        raw_docs = processor.load_document()

        # Process each document with contextual splitting
        all_chunks = []

        for raw_doc in raw_docs:
            text = raw_doc.page_content
            lang = detect_language(text)

            if lang == "unknown":
                logger.warning(f"Document {file_path} has unknown language. Skipping.")
                continue

            # Create semantic splitter with language-appropriate settings
            contextual_splitter = ContextualDocumentSplitter(language=lang)

            # Create a document with appropriate metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "language": lang,
                    "filename": processor.file_name
                }
            )

            # Split with contextual awareness
            chunks = contextual_splitter.split_documents([doc])

            # Enrich with context from adjacent chunks
            enriched_chunks = enrich_chunks_with_context(chunks)

            all_chunks.extend(enriched_chunks)

        logger.info(f"Processed file {file_path}: {len(all_chunks)} chunks created")
        return all_chunks

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise

def process_folder(folder_path: str, file_names: Optional[List[str]] = None) -> List[Document]:
    """
    Process specific documents in a folder, creating preprocessed chunks.
    If file_names is provided, only process those files. Otherwise, process all supported files.
    Skips files that don't exist or aren't supported without raising errors.

    Args:
        folder_path: Path to the folder containing documents
        file_names: Optional list of specific file names to process (not full paths)

    Returns:
        List of processed Document chunks from specified files
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a valid directory")

    all_chunks = []
    processed_files = 0
    skipped_files = 0

    # Supported extensions
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.csv', '.html', '.htm']

    # Get the files to process
    if file_names:
        # Process only specified files
        files = [os.path.join(folder_path, f) for f in file_names]
        logger.info(f"Requested to process {len(files)} specific files in folder {folder_path}")
    else:
        # Process all files in the directory (excluding subdirectories)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))]
        logger.info(f"Found {len(files)} files in folder {folder_path}")
    for file_path in files:
        # Skip if file doesn't exist
        if not os.path.isfile(file_path):
            logger.info(f"Skipping non-existent file: {file_path}")
            skipped_files += 1
            continue

        file_extension = os.path.splitext(file_path)[1].lower()

        # Skip unsupported file types
        if file_extension not in supported_extensions:
            logger.info(f"Skipping unsupported file: {file_path}")
            skipped_files += 1
            continue

        try:
            # logger.info(f"Processing file: {file_path}")
            processor = ArabicEnglishDocumentProcessor(file_path)
            chunks = processor.load_and_process()
            all_chunks.extend(chunks)
            processed_files += 1
            # logger.info(f"Processed {file_path}: extracted {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            skipped_files += 1

    # logger.info(f"Folder processing complete: {processed_files} files processed, "
    #             f"{skipped_files} files skipped, {len(all_chunks)} total chunks extracted")

    return all_chunks

def get_document_chunks(input_source: str, filenames: List[str] = None) -> List[Document]:
    """
    Process specified documents in a folder with contextual chunking.

    Args:
        input_source: Path to the folder containing documents
        filenames: List of specific filenames to process. If None, all files will be processed.

    Returns:
        List of processed Document chunks with context
    """
    # Fix 1: Use input_source consistently (not folder_path)
    if not os.path.isdir(input_source):
        raise ValueError(f"The path {input_source} is not a valid directory")

    # Fix 2: Define supported_extensions function or import it
    def supported_extensions():
        """Return a list of supported file extensions"""
        return ['.pdf', '.docx', '.txt', '.md', '.html', '.csv', '.json', '.xml']

    # If no specific filenames provided, process all documents
    if filenames is None:
        documents = process_folder(input_source)
    else:
        # Process only the specified files
        documents = []
        for filename in filenames:
            file_path = os.path.join(input_source, filename)
            if os.path.isfile(file_path):
                # Determine file type and process accordingly
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in supported_extensions():
                    doc = process_file(file_path)
                    if isinstance(doc, list):
                        documents.extend(doc)
                    else:
                        documents.append(doc)
                else:
                    logger.warning(f"Skipping unsupported file: {filename}")
            else:
                logger.warning(f"File not found: {file_path}")

    # Apply contextual processing to loaded documents
    contextual_splitter = ContextualDocumentSplitter()
    chunks = contextual_splitter.split_documents(documents)

    # Enrich with context
    enriched_chunks = enrich_chunks_with_context(chunks)

    logger.info(f"Processed {len(documents)} files with contextual chunking: {len(enriched_chunks)} chunks created")

    return enriched_chunks

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Process documents with contextual chunking')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing documents')
    parser.add_argument('--files', '-f', nargs='+', help='Specific files to process (if not provided, all files will be processed)')
    args = parser.parse_args()

    # Process the folder
    try:
        logger.info(f"Starting to process folder with contextual chunking: {args.folder_path}")

        # Check if specific files were provided
        if args.files:
            logger.info(f"Processing specific files: {', '.join(args.files)}")
            documents = get_document_chunks(input_source=args.folder_path, filenames=args.files)
        else:
            logger.info("Processing all files in the folder")
            documents = get_document_chunks(input_source=args.folder_path)

        if not documents:
            logger.warning("No documents were found or processed.")
            print("No documents were found or processed.")
        else:
            # Print summary statistics
            logger.info(f"Processing complete. Results:")
            logger.info(f"Total chunks processed: {len(documents)}")

            # Count documents by language
            arabic_docs = sum(1 for doc in documents if doc.metadata.get('language') == 'ar')
            english_docs = sum(1 for doc in documents if doc.metadata.get('language') == 'en')

            logger.info(f"Arabic chunks: {arabic_docs}")
            logger.info(f"English chunks: {english_docs}")

            print(f"\nSuccessfully processed {len(documents)} chunks with contextual awareness")
            print(f"Arabic chunks: {arabic_docs}, English chunks: {english_docs}")

            # Show up to first 3 chunks
            for i in range(min(3, len(documents))):
                print(f"Chunk {i+1}: {documents[i].page_content[:150]}..." if len(documents[i].page_content) > 150 else documents[i].page_content)

    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")
        print(f"Error: {str(e)}")

# To run this script move into backend\app\rag folder then run command given below:
# python .\contextual_chunking.py ..\..\storage\documents\user_123456
