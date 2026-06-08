"""WebSocket handler for real-time training updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.events import event_bus

ws_router = APIRouter()


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections for real-time events."""
    await websocket.accept()
    event_bus.register_ws(websocket)

    try:
        await websocket.send_json({
            "event": "connected",
            "data": {"message": "Connected to AITrainerUltra"},
        })

        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "event": "echo",
                "data": {"message": data},
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        event_bus.unregister_ws(websocket)
