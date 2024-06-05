from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vmd_app.routes import router

def setup_app() -> FastAPI:
    """ setup vmd_app including CORS config and router
    :return: vmd_app(FastAPI)
    """
    try:
        app = FastAPI() # vmd_app dec
        origin = "http://localhost:3000" # request resoure
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origin,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(router=router) # include router
        return app # return vmd_app
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error with message: {format(str(e))}")