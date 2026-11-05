"""Run the FastAPI server."""
import uvicorn
from config.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
