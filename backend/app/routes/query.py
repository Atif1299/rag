from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from ..services.rag_service import initialize_pipeline, process_user_query, stream_query_response

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    stream: bool = False

@router.post("/query")
async def get_query_response(request: QueryRequest):
    try:
        pipeline = initialize_pipeline()

        if request.stream:
            return await stream_query_response(request.query, pipeline)
        else:
            result = await process_user_query(request.query, pipeline)

            # Handle different response formats
            if isinstance(result, dict):
                if "response" in result:
                    response = result["response"]
                    if hasattr(response, "content"):
                        return {"content": response.content}
                    elif isinstance(response, str):
                        return {"content": response}
                    else:
                        return {
                            "content": str(response)
                        }
                else:
                    return {
                        "content": str(result)
                    }
            else:
                return {
                    "content": str(result)
                }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


