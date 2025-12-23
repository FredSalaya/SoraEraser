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
    svg_path = "whatermark.svg"
    base_template, _ = load_watermark_from_svg(svg_path)
    
    # Pre-compute scaled templates for robustness
    templates = []
    if base_template is not None:
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            t_h, t_w = base_template.shape
            scaled_w = int(t_w * scale)
            scaled_h = int(t_h * scale)
            # Ensure not too small
            if scaled_w > 10 and scaled_h > 10:
                resized_t = cv2.resize(base_template, (scaled_w, scaled_h))
                templates.append((resized_t, scale))
    
    # Kernel for dilation
    kernel = np.ones((5,5), np.uint8)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        final_mask = np.zeros((height, width), dtype=np.uint8)
        found_match = False
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Multi-scale Search
        best_match = None
        best_val = -1
        
        for template, scale in templates:
            # Skip if template is larger than frame
            if template.shape[0] > height or template.shape[1] > width:
                continue
                
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_val:
                best_val = max_val
                best_match = (max_loc, template)
        
        # Check if our best match is good enough
        # Threshold 0.45 is a balanced choice
        if best_val > 0.45 and best_match:
            top_left, match_template = best_match
            h, w = match_template.shape
            
            y1, y2 = top_left[1], top_left[1] + h
            x1, x2 = top_left[0], top_left[0] + w
            
            y1_cl, y2_cl = max(0, y1), min(height, y2)
            x1_cl, x2_cl = max(0, x1), min(width, x2)
            
            if y2_cl > y1_cl and x2_cl > x1_cl:
                # ROI Masking
                roi_gray = gray_frame[y1_cl:y2_cl, x1_cl:x2_cl]
                
                # Blur specifically to remove noise text/edges
                roi_blur = cv2.GaussianBlur(roi_gray, (3,3), 0)
                
                # Threshold to find white content
                _, roi_mask = cv2.threshold(roi_blur, 190, 255, cv2.THRESH_BINARY)
                
                final_mask[y1_cl:y2_cl, x1_cl:x2_cl] = roi_mask
                found_match = True

        # 3. Fallback: only if confidence is low
        if not found_match:
             _, thresh = cv2.threshold(gray_frame, 240, 255, cv2.THRESH_BINARY)
             final_mask = thresh

        # Dilate mask 
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        
        # Inpaint
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
