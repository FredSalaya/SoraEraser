import cv2
import numpy as np
import tempfile
import os
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cairosvg
from PIL import Image
import io

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

def remove_temp_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def load_watermark_from_svg(svg_path: str):
    try:
        # Render SVG to PNG in memory
        png_data = cairosvg.svg2png(url=svg_path)
        
        # Open with PIL to handle layers/alpha easily
        pil_img = Image.open(io.BytesIO(png_data))
        
        # Ensure RGBA
        pil_img = pil_img.convert("RGBA")
        
        # Convert to numpy array (OpenCV format is BGR/BGRA, PIL is RGB/RGBA)
        # We need the Alpha channel for the mask, and the brightness for template matching
        img_np = np.array(pil_img)
        
        # Extract Mask (Alpha channel)
        # Binarize: Any alpha > 0 is part of the mask
        alpha = img_np[:, :, 3]
        _, request_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
        # Create Template for Matching (Grayscale)
        # We want the white logo to be bright, background black
        # Create a black background
        background = np.zeros_like(img_np[:, :, :3])
        
        # Use alpha to composite
        # Where alpha is high, use the pixel color (which is white #FFF)
        # Where alpha is low, keep black
        # Since the logo is white, we can just take the alpha channel as the grayscale representation 
        # of the logo shape for matching purposes.
        template_gray = alpha 
        
        return template_gray, request_mask
        
    except Exception as e:
        print(f"Error loading SVG: {e}")
        return None, None

def process_video_frames(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Configure writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load Watermark Template
    # Assuming whatermark.svg is in the root directory
    svg_path = "whatermark.svg"
    template, wm_mask = load_watermark_from_svg(svg_path)
    
    # Fallback kernel if SVG fails or for additional dilation
    kernel = np.ones((5,5), np.uint8) # Increased kernel size slightly
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        final_mask = None
        
        # If we successfully loaded the SVG, try to find it
        if template is not None and wm_mask is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            # TM_CCOEFF_NORMED is good for lighting invariance
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Threshold for detection
            if max_val > 0.6: # Confidence threshold
                top_left = max_loc
                h, w = template.shape
                
                # Create a mask for the whole frame
                full_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Copy the watermark mask to the detected location
                # Handle boundaries
                y1, y2 = top_left[1], top_left[1] + h
                x1, x2 = top_left[0], top_left[0] + w
                
                # Clip to frame dimensions
                y1_cl, y2_cl = max(0, y1), min(height, y2)
                x1_cl, x2_cl = max(0, x1), min(width, x2)
                
                # Calculate offsets for the mask source
                my1, my2 = y1_cl - y1, (y1_cl - y1) + (y2_cl - y1_cl)
                mx1, mx2 = x1_cl - x1, (x1_cl - x1) + (x2_cl - x1_cl)
                
                if y2_cl > y1_cl and x2_cl > x1_cl:
                     full_mask[y1_cl:y2_cl, x1_cl:x2_cl] = wm_mask[my1:my2, mx1:mx2]
                     final_mask = full_mask

        # Fallback to old method if not found (or combine?)
        # For now, if found, use it. If not, maybe use old method?
        # User said "para poder detectar mejor", implying the old one was bad.
        # But if the watermark is missing in a frame (unlikely) or detection fails,
        # we might want to fallback. However, mixing random thresholding might retain the bad behavior.
        # Let's try: If found, use SVG mask. If not found, do nothing (pass frame).
        # OR: Combine. 
        # Let's perform the thresholding as a backup usually, OR just trust the template.
        # Given "better detect", I will fallback to thresholding if low confidence, 
        # but thresholding was "detecting white pixels".
        
        if final_mask is None:
             # Fallback logic: Simple thresholding (Original)
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY) # Increased threshold slightly
             final_mask = thresh

        # Dilate mask to cover edges
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        
        # Inpaint
        # Radius 3 is typical
        cleaned_frame = cv2.inpaint(frame, final_mask, 3, cv2.INPAINT_TELEA)
        
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
