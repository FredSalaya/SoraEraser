import cv2
import numpy as np
import tempfile
import os
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

def remove_temp_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def process_video_frames(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Configure writer
    # 'mp4v' is widely supported by OpenCV's built-in FFmpeg
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    kernel = np.ones((3,3), np.uint8)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect white watermark
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High threshold for white pixels (220-255)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Dilate to cover edges of the logo
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Inpaint
        cleaned_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        
        out.write(cleaned_frame)
        
    cap.release()
    out.release()

@app.post("/clean-video")
async def clean_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Create temp file for input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        shutil.copyfileobj(file.file, tmp_input)
        tmp_input_path = tmp_input.name

    # Create temp file for output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
        tmp_output_path = tmp_output.name
    
    try:
        # Process the video
        process_video_frames(tmp_input_path, tmp_output_path)
        
        # Determine filename for download
        filename = f"clean_{file.filename}"
        
        # Callback to cleanup after response
        def cleanup():
            remove_temp_file(tmp_input_path)
            remove_temp_file(tmp_output_path)
            
        background_tasks.add_task(cleanup)
        
        return StreamingResponse(
            open(tmp_output_path, "rb"), 
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        # Cleanup in case of error
        remove_temp_file(tmp_input_path)
        remove_temp_file(tmp_output_path)
        return {"error": str(e)}
