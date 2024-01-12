from pydantic import BaseModel
from fastapi import File, UploadFile, Form
from typing import List
class CorrectionBody(BaseModel):
    """ body includes text(target)
    And file (.wav) extension for model to process """
    text: Form
    media: UploadFile = File(...)
