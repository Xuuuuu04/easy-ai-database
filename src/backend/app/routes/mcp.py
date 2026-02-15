from __future__ import annotations

from typing import Generator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..state import mcp_authorized, mcp_protocol_server, mcp_response_headers

router = APIRouter()


@router.get("/mcp/v1")
def mcp_stream(request: Request) -> Response:
    if not mcp_authorized(request):
        return Response(status_code=401, headers=mcp_response_headers())

    def _events() -> Generator[str, None, None]:
        yield ": mcp stream established\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            **mcp_response_headers(),
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/mcp/v1")
async def mcp_http(request: Request) -> Response:
    if not mcp_authorized(request):
        return Response(status_code=401, headers=mcp_response_headers())

    try:
        payload = await request.json()
    except Exception:
        error_payload = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"},
        }
        return JSONResponse(
            error_payload, status_code=400, headers=mcp_response_headers()
        )

    if isinstance(payload, list):
        responses = []
        for item in payload:
            response = mcp_protocol_server.handle_message(item)
            if response is not None:
                responses.append(response)
        if not responses:
            return Response(status_code=202, headers=mcp_response_headers())
        return JSONResponse(responses, headers=mcp_response_headers())

    response = mcp_protocol_server.handle_message(payload)
    if response is None:
        return Response(status_code=202, headers=mcp_response_headers())
    return JSONResponse(response, headers=mcp_response_headers())
