from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from myapp.routes import router


def setup_app() -> FastAPI:
    """setup app including CORS config and router
    :return: app(FastAPI)
    """
    try:
        app = FastAPI()  # app dec

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router=router)  # include router

        return app  # return app

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Server error with message: {format(str(e))}"
        )
