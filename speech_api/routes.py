from fastapi import APIRouter, HTTPException, status

router = APIRouter()

@router.get('/', status_code=status.HTTP_200_OK)
def index():
    return {"message": "Hello word"}

@router.post('/danangvsr/vmd', status_code=status.HTTP_200_OK)
def correction_route():
    """ input of correction route is "wav" file
    the service responsible for dealing with wav file
    :return the output contain phonemes each one will be associated with a tag
    that labeled T(true), F(false)
    """
    return