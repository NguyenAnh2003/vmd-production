from fastapi import APIRouter, HTTPException, status, \
    UploadFile, Form, File
from speech_api.services import correcting_service
""" modules """

# router define
router = APIRouter()

@router.get('/', status_code=status.HTTP_200_OK)
def index():
    return {"message": "Hello word"}


@router.post('/danangvsr/vmd', status_code=status.HTTP_200_OK)
def correction_route(file: bytes = File(...), text_target: str = Form(...)):
    text, audio = correcting_service(file, text_target)
    """ input of correction route is "wav" file and text """
    return {"target": text, "file": audio}


@router.post('/test/upload', status_code=status.HTTP_200_OK)
def test_route(file: UploadFile):
    size = file.size
    name = file.filename
    result = {size, name}
    return {"Response": result}
