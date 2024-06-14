import io
from fastapi import APIRouter, status, UploadFile, Form, File, HTTPException
from app.services import vmd_service


# router define
router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
def index():
    return {"message": "Hello word"}


@router.post("/danangvsr/vmd", status_code=status.HTTP_200_OK)
def vmd_route(file: UploadFile = File(...), text_target: str = Form(...)):
    file_data = file.file  # file content
    result = vmd_service(media=file_data, text=text_target)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error"
        )

    return {"message": result}
