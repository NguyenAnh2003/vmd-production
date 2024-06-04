from fastapi import FastAPI
from app.setup_app import setup_app
import uvicorn

""" setup app and run """
app = setup_app() #

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True,
                reload_delay=0.5, use_colors=True)