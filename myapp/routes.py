import io
from fastapi import APIRouter, status, UploadFile, Form, File
from services import vmd_service

# router define
router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
def index():
    return {"message": "Hello word"}


@router.post("/danangvsr/vmd", status_code=status.HTTP_200_OK)
def vmd_route(file: UploadFile = File(...), text_target: str = Form(...)):
    """input of correction route is "wav" file and text"""
    file_data = file.file  # file content
    result = vmd_service(media=file_data, text=text_target)
    return result
