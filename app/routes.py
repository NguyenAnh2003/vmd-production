import io

from fastapi import APIRouter, status, \
    UploadFile, Form, File
from app.services import correcting_service
import torchaudio

""" modules """

# router define
router = APIRouter()


@router.get('/', status_code=status.HTTP_200_OK)
def index():
    return {"message": "Hello word"}


@router.post('/danangvsr/vmd', status_code=status.HTTP_200_OK)
def correction_route(file: UploadFile = File(...), text_target: str = Form(...),
                     username: str = Form(...), country: str = Form(...), age: int = Form(...)):
    file_data = file.file # file content
    print(f"Filename: {file.filename} Username: {username} Country: {country} Age: {age}")
    result = correcting_service(media=file_data, file_name=str(file.filename), text=text_target,
                                username=username, country=country, age=age)
    """ input of correction route is "wav" file and text """
    # return {"target": text, "file": audio}
    return result


@router.post('/test/upload', status_code=status.HTTP_200_OK)
async def test_route(file: UploadFile = File(...), text_target: str = Form(...)):
    file_data = await file.read()
    array, _ = torchaudio.load(io.BytesIO(file_data))
    result = "success"
    return {"Response": text_target, "Audio shape": array.shape}
