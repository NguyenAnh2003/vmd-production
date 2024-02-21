from fastapi import FastAPI
from setup.setup_app import setup_app
import uvicorn
from core.mylogger import setup_logger

# setup logger
_logger = setup_logger("./core/logs/app_run.log", location="main")
_logger.getLogger("main")

""" setup app and run """
app = setup_app() #

if __name__ == "__main__":
    _logger.log(_logger.INFO, "App running")
    uvicorn.run("main:app", host="127.0.0.1", port=8001, log_level="info", reload=True,
                reload_delay=0.5, use_colors=True)