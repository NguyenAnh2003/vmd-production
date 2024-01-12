from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from speech_api.routes import router
def setup_app():
    """ setup app including CORS config and router
    :return: app(FastAPI)
    """
    try:
        app = FastAPI() # app dec
        origin = "http://localhost:3000" # request resoure
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origin,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(router) # include router
    except Exception as e:
        raise HTTPException(status_code=500, detail="Server error")