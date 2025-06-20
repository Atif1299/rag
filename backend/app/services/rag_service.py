import os
from fastapi.responses import StreamingResponse
from ..rag.rag_pipeline import EnhancedRAGPipeline
from langchain_core.messages import AIMessage

def initialize_pipeline():
    return EnhancedRAGPipeline(
        collection_name="my_collection",
        model_name="tgi",
        base_url="https://m0vtsu71q6nl17e8.us-east-1.aws.endpoints.huggingface.cloud/v1/",  # Note new default URL
        api_key=os.environ.get("HF_API_KEY"),  # Will use HF_API_TOKEN or OPENAI_API_KEY if not provided
        max_tokens=1024,  # New default is larger
        # Additional parameters you may want to configure:
        dense_top_k=15,    # Number of dense retrieval results
        sparse_weight=0.3, # Weight for BM25 scores (0.0-1.0)
        final_k=7,         # Final number of documents after filtering
        embeddings_model="Snowflake/snowflake-arctic-embed-l-v2.0",  # New default embedding model
        milvus_host="localhost",  # Milvus connection details
        milvus_port=19530,
        enable_query_enhancement=True  # Can disable if not needed
    )

async def process_user_query(query: str, pipeline: EnhancedRAGPipeline):
    return await pipeline.process_query(query=query)

async def stream_query_response(query: str, pipeline: EnhancedRAGPipeline):
    result = await pipeline.process_query(query=query, stream_output=True)

    if "error" in result:
        raise Exception(result["error"])

    context = result.get("streaming_context", {})
    if not context:
        return {"content": str(result.get("response", "No response generated"))}

    async def response_generator():
        try:
            chat_completion = pipeline.client.chat.completions.create(
                model=context["model"],
                messages=[{"role": "user", "content": context["prompt"]}],
                max_tokens=context["max_tokens"],
                stream=True
            )

            for message in chat_completion:
                chunk = message.choices[0].delta.content
                if chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )
