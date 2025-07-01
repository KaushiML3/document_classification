from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse , JSONResponse,HTMLResponse
from pydantic import BaseModel


import uvicorn
import cv2  
import tempfile
import shutil
import os
import warnings
import base64
import numpy as np
from pathlib import Path

from app.src.model_loader import vit_loader,vgg_loader
from app.src.logger import setup_logger


warnings.filterwarnings("ignore")


app=FastAPI(title="Document_Classifire",
    description="FastAPI",
    version="0.115.4")

# Allow all origins (replace * with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

@app.get("/")
async def root():
  return {"Fast API":"API is woorking"}


# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = all logs, 1 = filter out info, 2 = filter out warnings, 3 = filter out errors
warnings.filterwarnings("ignore")

logger = setup_logger()

@app.post("/vit_model")    
async def vit_clf(cut_off:float=0.5,image_file: UploadFile = File(...)):

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Create a temporary file path
        temp_file_path = os.path.join(temp_dir,image_file.filename)
        # Write the uploaded file content to the temporary file
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(image_file.file, temp_file)
        result=vit_loader().predict(image_path=Path(temp_file_path), cut_off=cut_off)
        logger.info(result)

        if result is not None:
            return JSONResponse(content={"status":1,"document_classe":result})
        else:
            return JSONResponse(content={"status":0,"document_classe":None})
    
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(content={"status":0,"error_message":str(e)})




@app.post("/vgg_model")    
async def vgg_clf(image_file: UploadFile = File(...)):

    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Create a temporary file path
        temp_file_path = os.path.join(temp_dir,image_file.filename)
        # Write the uploaded file content to the temporary file
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(image_file.file, temp_file)
        result=vgg_loader().predict(image_path=Path(temp_file_path))
        logger.info(result)

        if result is not None:
            return JSONResponse(content={"status":1,"document_classe":result})
        else:
            return JSONResponse(content={"status":0,"document_classe":None})
    
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(content={"status":0,"document_classe":str(e)})
    
    

