"""AITrainerUltra - 多模型AI训练框架入口点"""

import uvicorn
from backend.api.server import app

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
