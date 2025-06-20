"""
RAG Prompts

This module contains prompt templates for the RAG system,
optimized for different languages and query types.
"""
from typing import List

from langchain_core.documents import Document


def format_document_context(docs: List[Document], query_lang: str = "en") -> str:
    """
    Format a list of documents into a well-structured context string.

    Args:
        docs: List of documents to format
        query_lang: Language of the query

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, doc in enumerate(docs):
        # Extract useful metadata
        doc_lang = doc.metadata.get("language", "unknown")
        filename = doc.metadata.get("filename", "unknown_file")
        source = doc.metadata.get("source", "")

        # Format header based on query language
        if query_lang == "ar":
            header = f"المستند {i+1} [الملف: {filename}, اللغة: {doc_lang}]:"
        else:
            header = f"Document {i+1} [File: {filename}, Language: {doc_lang}]:"

        # Add source if available and not redundant with filename
        if source and filename not in source:
            if query_lang == "ar":
                header += f" [المصدر: {source}]"
            else:
                header += f" [Source: {source}]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(context_parts)


def summarize_documents_prompt(context: str, query: str, query_lang: str = "ar") -> str:
    """
    Get a prompt for explaining/summarizing all documents.

    Args:
        context: Formatted document context
        query: Original query
        query_lang: Language of the query

    Returns:
        Complete prompt string
    """
    if query_lang == "ar":
        return f"""أنت مساعد ذكي يتحدث العربية بطلاقة ولديك معرفة عميقة بالنصوص القانونية والإدارية.
**تعليمات حاسمة: لا تقم بإنشاء أي معلومات أو تفسيرات أو تفسيرات غير موجودة بشكل مباشر أو يمكن استنتاجها بشكل صريح من المحتوى الموجود في المستندات المقدمة. إذا لم يكن من الممكن الإجابة على سؤال من خلال المستندات، فاذكر أن المعلومات غير موجودة في النص المقدم.**
مهمتك هي تلخيص وشرح مجموعة من المستندات المرفقة بشكل واضح ومنظم. هذه المستندات قد تحتوي على معلومات تنظيمية أو إجرائية أو قانونية.

قم بتحليل المستندات وتقديم:
1. نظرة عامة على نوع المستندات وموضوعها الرئيسي
2. المعلومات الأساسية والمهمة في كل مستند
3. شرح للمصطلحات المتخصصة إن وجدت
4. ترابط المعلومات بين المستندات المختلفة (إن وجد)

إذا كانت المستندات بلغة غير العربية، قم بشرح محتواها بالعربية.

المستندات:
{context}

ملاحظة هامة: يجب أن تجيب باللغة العربية الفصحى بشكل واضح ومهيكل. قسّم إجابتك إلى فقرات منظمة لتسهيل الفهم.
تجاهل أي أخطاء تقنية أو تنسيقية في المستندات وركز على المحتوى.
"""
    else:
        return f"""You are a knowledgeable AI assistant with deep expertise in legal and administrative texts.
**Crucial Instruction: Do NOT generate any information, explanations, or interpretations that are not directly present or explicitly inferable from the content within the provided documents. If a question cannot be answered from the documents, state that the information is not found in the provided text.**
Your task is to summarize and explain a set of attached documents clearly and in an organized manner. These documents may contain regulatory, procedural, or legal information.

Analyze the documents and provide:
1. An overview of the document types and their main subject
2. The essential and important information in each document
3. Explanations of specialized terms if any
4. Connections between information in different documents (if any)

If the documents are in a language other than English, explain their content in English.

Documents:
{context}

Answer in a clear, professional, and comprehensive style. Organize your response in well-structured paragraphs for easy understanding.
"""


def qa_answer_prompt(context: str, query: str, query_lang: str = "en") -> str:
    """
    Get a prompt for answering specific questions.

    Args:
        context: Formatted document context
        query: Original query
        query_lang: Language of the query

    Returns:
        Complete prompt string
    """
    if query_lang == "ar":
        return f"""أنت مساعد ذكي متخصص بالإجابة عن الأسئلة باللغة العربية مع دقة عالية.

سأزودك بمجموعة من المستندات كسياق، وسؤال يحتاج إلى إجابة. استخدم المعلومات الموجودة في المستندات لإعداد إجابتك.

إرشادات مهمة:
- قدّم إجابة مفصلة ومنظمة باللغة العربية الفصحى
- إذا لم تتوفر معلومات كافية في المستندات للإجابة عن السؤال، اذكر ذلك بوضوح: "المستندات المتاحة لا تحتوي على معلومات كافية للإجابة على هذا السؤال"
- إذا كانت المستندات بلغة غير العربية، استخرج المعلومات وأجب بالعربية
- التزم بالمعلومات المذكورة في المستندات فقط ولا تختلق أي معلومات إضافية
- اذكر من أي مستند حصلت على المعلومات
- تجاهل أي أخطاء تقنية أو تنسيقية في المستندات وركز على المحتوى
الالتزام الصارم بالمعلومات:

التقيد الصارم والدقيق بالمعلومات المنصوص عليها صراحة في الوثائق. لا تقم بتلفيق أو افتراض أو إضافة أي معلومات خارجية مهما كانت.
المستندات:
{context}

السؤال: {query}

الإجابة (باللغة العربية):"""
    else:
        return f"""Final Combined Prompt:

You are an intelligent assistant specialized in answering questions in English with high accuracy.

I will provide you with a set of documents as context and a question that needs answering. Use only the information in the documents to prepare your answer.

Important guidelines:

Provide a balanced response that is neither too brief nor excessively long. Aim for a comprehensive answer that fully addresses the question while staying concise and focused on the most relevant information.

If there is not enough information in the documents to answer the question, clearly state: "The available documents do not contain sufficient information to answer this question."

If the documents are in a language other than English, extract the relevant information and respond in English.

Strict Adherence to Information:

Strictly and precisely adhere to the information explicitly stated in the documents. Do not fabricate, assume, or add any external information whatsoever.

Use natural, conversational language while maintaining accuracy and precision.

Documents:
{context}

Question:
{query}

Answer:"""


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Currently only supports English and Arabic.

    Args:
        text: Input text to detect language

    Returns:
        Language code ("en" or "ar")
    """
    # Simple detection based on Arabic characters
    arabic_chars = [chr(code) for code in range(0x0600, 0x06FF)]
    arabic_count = sum(1 for char in text if char in arabic_chars)

    # If more than 10% of characters are Arabic, consider it Arabic
    if arabic_count / max(len(text), 1) > 0.1:
        return "ar"
    else:
        return "en"


def enhance_query_prompt(query: str, query_lang: str = "en") -> str:
    """
    Generate a prompt to enhance/expand the original user query.

    Args:
        query: Original user query
        query_lang: Language of the query

    Returns:
        Complete prompt string for query enhancement
    """
    if query_lang == "ar":
        return f"""أنت مساعد ذكي متخصص في تحسين وتوسيع الاستعلامات البحثية.

الاستعلام الأصلي هو: "{query}"

قم بتوسيع هذا الاستعلام لتحسين نتائج البحث من خلال:
1. إضافة مرادفات أو مصطلحات بديلة
2. تحديد المفاهيم الرئيسية والتعبير عنها بطرق مختلفة
3. إضافة سياق إضافي قد يكون مفيدًا للبحث

قدم الاستعلام المحسّن كنص واحد دون تعليقات أو شروحات إضافية. حافظ على الاستعلام المحسّن باللغة العربية.
"""
    else:
        return f"""You are an intelligent assistant specialized in enhancing and expanding search queries.

The original query is: "{query}"

Expand this query to improve search results by:
1. Adding synonyms or alternative terms
2. Identifying key concepts and expressing them in different ways
3. Adding additional context that might be useful for retrieval

Provide the enhanced query as a single text without additional comments or explanations. Keep the enhanced query in English.
"""


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Currently only supports English and Arabic.

    Args:
        text: Input text to detect language

    Returns:
        Language code ("en" or "ar")
    """
    # Simple detection based on Arabic characters
    arabic_chars = [chr(code) for code in range(0x0600, 0x06FF)]
    arabic_count = sum(1 for char in text if char in arabic_chars)

    # If more than 10% of characters are Arabic, consider it Arabic
    if arabic_count / max(len(text), 1) > 0.1:
        return "ar"
    else:
        return "en"


def generate_query_enhancement_prompt(query: str, query_lang: str = "en") -> str:
    """
    Generate a prompt to enhance/expand the original user query.

    Args:
        query: Original user query
        query_lang: Language of the query

    Returns:
        Complete prompt string for query enhancement
    """
    if query_lang == "ar":
        return f"""أنت مساعد ذكي متخصص في تحسين وتوسيع الاستعلامات البحثية.

الاستعلام الأصلي هو: "{query}"

قم بتوسيع هذا الاستعلام لتحسين نتائج البحث من خلال:
1. إضافة مرادفات أو مصطلحات بديلة
2. تحديد المفاهيم الرئيسية والتعبير عنها بطرق مختلفة
3. إضافة سياق إضافي قد يكون مفيدًا للبحث

قدم الاستعلام المحسّن كنص واحد دون تعليقات أو شروحات إضافية. حافظ على الاستعلام المحسّن باللغة العربية.
"""
    else:
        return f"""You are an intelligent assistant specialized in enhancing and expanding search queries.

The original query is: "{query}"

Expand this query to improve search results by:
1. Adding synonyms or alternative terms
2. Identifying key concepts and expressing them in different ways
3. Adding additional context that might be useful for retrieval

Provide the enhanced query as a single text without additional comments or explanations. Keep the enhanced query in English.
"""
