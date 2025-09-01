import os
import tempfile
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import the existing synchronizer class
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("subtitle_sync", "Subtitle-Sync-Correction.py")
subtitle_sync_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subtitle_sync_module)
SubtitleSynchronizer = subtitle_sync_module.SubtitleSynchronizer

# Import repositioning logic
reposition_spec = importlib.util.spec_from_file_location("reposition_subtitles6", "reposition_subtitles6.py")
reposition_module = importlib.util.module_from_spec(reposition_spec)
reposition_spec.loader.exec_module(reposition_module)


# Pydantic models for API requests/responses
class SyncRequest(BaseModel):
    video_filename: str
    subtitle_filename: str
    output_filename: Optional[str] = None


class RepositionRequest(BaseModel):
    video_filename: str
    subtitle_filename: str
    output_filename: Optional[str] = None


class SyncStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: str
    progress: Optional[int] = None
    result_file: Optional[str] = None


class SyncResult(BaseModel):
    success: bool
    message: str
    output_file: Optional[str] = None
    offset_applied: Optional[float] = None
    processing_time: Optional[float] = None


class GenerationRequest(BaseModel):
    video_filename: str
    subtitle_format: str = "srt"
    whisper_model: str = "base"
    output_filename: Optional[str] = None


# FastAPI app initialization
app = FastAPI(
    title="Subtitle Synchronizer API",
    description="API for synchronizing subtitle files with video audio tracks",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for task management
tasks_status: Dict[str, SyncStatus] = {}
task_results: Dict[str, SyncResult] = {}
executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent processing

# Directory for uploaded files and outputs
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Subtitle Synchronizer API",
        "version": "1.0.0",
        "endpoints": {
            "upload_files": "/upload-files/",
            "sync_subtitles": "/sync/",
            "get_status": "/status/{task_id}",
            "download_result": "/download/{filename}",
            "list_files": "/files/"
        }
    }


@app.post("/upload-files/")
async def upload_files(
    video_file: UploadFile = File(...),
    subtitle_file: UploadFile = File(...)
):
    """Upload video and subtitle files"""
    try:
        # Validate file types
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        subtitle_extensions = {'.srt', '.vtt'}
        
        video_ext = Path(video_file.filename).suffix.lower()
        subtitle_ext = Path(subtitle_file.filename).suffix.lower()
        
        if video_ext not in video_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video file type. Supported: {', '.join(video_extensions)}"
            )
        
        if subtitle_ext not in subtitle_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid subtitle file type. Supported: {', '.join(subtitle_extensions)}"
            )
        
        # Save uploaded files
        video_path = UPLOAD_DIR / video_file.filename
        subtitle_path = UPLOAD_DIR / subtitle_file.filename
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        with open(subtitle_path, "wb") as buffer:
            shutil.copyfileobj(subtitle_file.file, buffer)
        
        return {
            "message": "Files uploaded successfully",
            "video_file": video_file.filename,
            "subtitle_file": subtitle_file.filename,
            "video_size": video_path.stat().st_size,
            "subtitle_size": subtitle_path.stat().st_size
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


@app.post("/sync/", response_model=Dict[str, str])
async def sync_subtitles(
    request: SyncRequest,
    background_tasks: BackgroundTasks
):
    """Start subtitle synchronization process"""
    try:
        # Validate input files exist
        video_path = UPLOAD_DIR / request.video_filename
        subtitle_path = UPLOAD_DIR / request.subtitle_filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")
        
        if not subtitle_path.exists():
            raise HTTPException(status_code=404, detail=f"Subtitle file not found: {request.subtitle_filename}")
        
        # Generate task ID and output filename
        import uuid
        task_id = str(uuid.uuid4())
        
        if request.output_filename:
            output_filename = request.output_filename
        else:
            # Generate default output filename
            base_name = Path(request.subtitle_filename).stem
            extension = Path(request.subtitle_filename).suffix
            output_filename = f"{base_name}_synced{extension}"
        
        output_path = OUTPUT_DIR / output_filename
        
        # Initialize task status
        tasks_status[task_id] = SyncStatus(
            task_id=task_id,
            status="pending",
            message="Task queued for processing"
        )
        
        # Start background task
        background_tasks.add_task(
            process_sync_task,
            task_id,
            str(video_path),
            str(subtitle_path),
            str(output_path)
        )
        
        return {
            "task_id": task_id,
            "message": "Synchronization task started",
            "output_filename": output_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting sync task: {str(e)}")


async def process_sync_task(task_id: str, video_path: str, subtitle_path: str, output_path: str):
    """Background task for processing subtitle synchronization"""
    import time
    start_time = time.time()
    
    try:
        # Update status to processing
        tasks_status[task_id].status = "processing"
        tasks_status[task_id].message = "Initializing synchronizer..."
        
        # Create synchronizer instance
        synchronizer = SubtitleSynchronizer(video_path, subtitle_path, output_path)
        
        # Update progress
        tasks_status[task_id].message = "Extracting audio from video..."
        tasks_status[task_id].progress = 20
        
        # Run synchronization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def run_sync():
            try:
                return synchronizer.synchronize()
            except Exception as e:
                raise e
        
        # Execute synchronization
        tasks_status[task_id].message = "Analyzing audio and synchronizing..."
        tasks_status[task_id].progress = 60
        
        success = await loop.run_in_executor(executor, run_sync)
        
        processing_time = time.time() - start_time
        
        if success:
            # Task completed successfully
            tasks_status[task_id].status = "completed"
            tasks_status[task_id].message = "Synchronization completed successfully"
            tasks_status[task_id].progress = 100
            tasks_status[task_id].result_file = Path(output_path).name
            
            # Store result
            task_results[task_id] = SyncResult(
                success=True,
                message="Subtitles synchronized successfully",
                output_file=Path(output_path).name,
                processing_time=processing_time
            )
        else:
            # Task completed but no sync needed
            tasks_status[task_id].status = "completed"
            tasks_status[task_id].message = "Subtitles were already well synchronized"
            tasks_status[task_id].progress = 100
            
            task_results[task_id] = SyncResult(
                success=True,
                message="Subtitles were already well synchronized",
                processing_time=processing_time
            )
            
    except Exception as e:
        # Task failed
        tasks_status[task_id].status = "failed"
        tasks_status[task_id].message = f"Synchronization failed: {str(e)}"
        
        task_results[task_id] = SyncResult(
            success=False,
            message=f"Error during synchronization: {str(e)}",
            processing_time=time.time() - start_time
        )


@app.post("/reposition/", response_model=Dict[str, str])
async def reposition_subtitles(request: RepositionRequest, background_tasks: BackgroundTasks):
    """Start subtitle repositioning task (after sync)"""
    try:
        video_path = UPLOAD_DIR / request.video_filename
        subtitle_path = OUTPUT_DIR / request.subtitle_filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file not found: {request.video_filename}")
        if not subtitle_path.exists():
            raise HTTPException(status_code=404, detail=f"Subtitle file not found: {request.subtitle_filename}")

        # Run repositioning in background
        def run_reposition():
            try:
                result_file = reposition_module.process_subtitle(str(video_path), str(subtitle_path))
                # Move result to outputs folder if needed
                result_path = Path(result_file)
                output_path = OUTPUT_DIR / result_path.name
                if result_path != output_path:
                    shutil.move(result_file, output_path)
                return output_path.name
            except Exception as e:
                raise e

        loop = asyncio.get_event_loop()
        output_filename = await loop.run_in_executor(executor, run_reposition)

        if output_filename:
            return {
                "success": "true",
                "message": "Subtitle repositioning completed",
                "output_filename": output_filename
            }
        else:
            raise HTTPException(status_code=500, detail="Subtitle repositioning failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during repositioning: {str(e)}")


@app.post("/upload-video/")
async def upload_video(video_file: UploadFile = File(...)):
    """Upload video file for subtitle generation"""
    try:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_ext = Path(video_file.filename).suffix.lower()
        if video_ext not in video_extensions:
            raise HTTPException(400, f"Invalid video file type. Supported: {', '.join(video_extensions)}")
        video_path = UPLOAD_DIR / video_file.filename
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        return {
            "message": "Video uploaded successfully",
            "filename": video_file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


# Generation task storage (simple in-memory for now)
generation_tasks = {}

@app.post("/generate/")
async def start_generation(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Start subtitle generation task"""
    try:
        video_path = UPLOAD_DIR / request.video_filename
        if not video_path.exists():
            raise HTTPException(404, f"Video file not found: {request.video_filename}")
        valid_models = ["tiny", "base", "small", "medium", "large"]
        valid_formats = ["srt", "vtt"]
        if request.whisper_model not in valid_models:
            raise HTTPException(400, f"Invalid model. Supported: {', '.join(valid_models)}")
        if request.subtitle_format not in valid_formats:
            raise HTTPException(400, f"Invalid format. Supported: {', '.join(valid_formats)}")
        if request.output_filename:
            output_filename = request.output_filename
        else:
            base_name = Path(request.video_filename).stem
            output_filename = f"{base_name}_subtitles.{request.subtitle_format}"
        task_id = str(uuid.uuid4())
        generation_tasks[task_id] = {
            "task_id": task_id,
            "status": "started",
            "message": "Task started",
            "progress": 0,
            "start_time": time.time()
        }
        def run_generation_task():
            try:
                import whisper
                generation_tasks[task_id]["status"] = "processing"
                generation_tasks[task_id]["message"] = "Loading Whisper model and transcribing..."
                generation_tasks[task_id]["progress"] = 10
                # Load Whisper model
                model = whisper.load_model(request.whisper_model)
                generation_tasks[task_id]["progress"] = 30
                # Transcribe audio
                result = model.transcribe(str(video_path))
                generation_tasks[task_id]["progress"] = 80
                # Convert to subtitle format
                if request.subtitle_format == "vtt":
                    subtitle_content = "WEBVTT\n\n"
                    for segment in result["segments"]:
                        start = segment["start"]
                        end = segment["end"]
                        text = segment["text"].strip()
                        subtitle_content += f"{int(start//60):02d}:{int(start%60):02d}.{int((start%1)*1000):03d} --> {int(end//60):02d}:{int(end%60):02d}.{int((end%1)*1000):03d}\n{text}\n\n"
                else:
                    subtitle_content = ""
                    for i, segment in enumerate(result["segments"], 1):
                        start = segment["start"]
                        end = segment["end"]
                        text = segment["text"].strip()
                        subtitle_content += f"{i}\n{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d} --> {int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}\n{text}\n\n"
                output_path = OUTPUT_DIR / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(subtitle_content.strip())
                generation_tasks[task_id]["progress"] = 100
                generation_tasks[task_id].update({
                    "status": "completed",
                    "message": "Subtitles generated successfully",
                    "result_file": output_filename,
                    "processing_time": time.time() - generation_tasks[task_id]["start_time"],
                    "segments_count": len(result["segments"])
                })
            except Exception as e:
                generation_tasks[task_id].update({
                    "status": "failed",
                    "message": str(e),
                    "processing_time": time.time() - generation_tasks[task_id]["start_time"]
                })
        threading.Thread(target=run_generation_task, daemon=True).start()
        return {
            "task_id": task_id,
            "message": "Subtitle generation task started",
            "output_filename": output_filename,
            "subtitle_format": request.subtitle_format,
            "whisper_model": request.whisper_model
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start generation: {str(e)}")


@app.get("/generate/status/{task_id}")
async def get_generation_status(task_id: str):
    """Get generation task status"""
    if task_id not in generation_tasks:
        raise HTTPException(404, "Task not found")
    return generation_tasks[task_id]

@app.get("/generate/result/{task_id}")
async def get_generation_result(task_id: str):
    """Get generation task result"""
    if task_id not in generation_tasks:
        raise HTTPException(404, "Task not found")
    task = generation_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(400, "Task not completed")
    return {
        "success": True,
        "message": task["message"],
        "output_file": task["result_file"],
        "processing_time": task.get("processing_time", 0),
        "segments_count": task.get("segments_count", 0)
    }


@app.get("/status/{task_id}", response_model=SyncStatus)
async def get_task_status(task_id: str):
    """Get the status of a synchronization task"""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_status[task_id]


@app.get("/result/{task_id}", response_model=SyncResult)
async def get_task_result(task_id: str):
    """Get the result of a completed synchronization task"""
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task result not found")
    
    return task_results[task_id]

@app.get("/models/")
async def get_available_models():
    """Get available Whisper models"""
    models = [
        {
            "name": "tiny",
            "description": "Fastest, English-only (~39 MB)",
            "languages": "English only"
        },
        {
            "name": "base", 
            "description": "Good balance of speed and accuracy (~74 MB)",
            "languages": "Multilingual"
        },
        {
            "name": "small",
            "description": "Better accuracy, slower (~244 MB)", 
            "languages": "Multilingual"
        },
        {
            "name": "medium",
            "description": "High accuracy, slower (~769 MB)",
            "languages": "Multilingual" 
        },
        {
            "name": "large",
            "description": "Best accuracy, slowest (~1550 MB)",
            "languages": "Multilingual"
        }
    ]
    
    return {
        "models": models,
        "supported_formats": ["srt", "vtt"],
        "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a synchronized subtitle file"""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/files/")
async def list_files():
    """List available files in upload and output directories"""
    try:
        upload_files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
        output_files = [f.name for f in OUTPUT_DIR.iterdir() if f.is_file()]
        
        return {
            "upload_files": upload_files,
            "output_files": output_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.delete("/files/{filename}")
async def delete_file(filename: str, file_type: str = "upload"):
    """Delete a file from upload or output directory"""
    try:
        if file_type == "upload":
            file_path = UPLOAD_DIR / filename
        elif file_type == "output":
            file_path = OUTPUT_DIR / filename
        else:
            raise HTTPException(status_code=400, detail="file_type must be 'upload' or 'output'")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        return {"message": f"File {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.delete("/cleanup/")
async def cleanup_files():
    """Clean up all uploaded and output files"""
    try:
        upload_count = 0
        output_count = 0
        
        # Remove upload files
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                upload_count += 1
        
        # Remove output files
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file():
                file_path.unlink()
                output_count += 1
        
        # Clear task status and results
        tasks_status.clear()
        task_results.clear()
        
        return {
            "message": "Cleanup completed",
            "upload_files_deleted": upload_count,
            "output_files_deleted": output_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


# Health check endpoint
@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_tasks": len([t for t in tasks_status.values() if t.status == "processing"]),
        "total_tasks": len(tasks_status)
    }


if __name__ == "__main__":
    print("Starting Subtitle Synchronizer API server...")
    print("Available endpoints:")
    print("  - Upload files: POST /upload-files/")
    print("  - Sync subtitles: POST /sync/")
    print("  - Check status: GET /status/{task_id}")
    print("  - Download result: GET /download/{filename}")
    print("  - API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
