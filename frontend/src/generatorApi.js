import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'

class SubtitleGeneratorAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Get available Whisper models
  async getAvailableModels() {
    const response = await axios.get(`${this.baseURL}/models/`);
    return response.data;
  }

  // Upload video file for subtitle generation
  async uploadVideo(videoFile, onProgress = null) {
    const formData = new FormData();
    formData.append('video_file', videoFile);

    const config = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    };

    if (onProgress) {
      config.onUploadProgress = (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(percentCompleted);
      };
    }

    const response = await axios.post(`${this.baseURL}/upload-video/`, formData, config);
    return response.data;
  }

  // Start subtitle generation process
  async startGeneration(videoFilename, subtitleFormat = 'srt', whisperModel = 'base', outputFilename = null) {
    const data = {
      video_filename: videoFilename,
      subtitle_format: subtitleFormat,
      whisper_model: whisperModel,
    };

    if (outputFilename) {
      data.output_filename = outputFilename;
    }

    const response = await axios.post(`${this.baseURL}/generate/`, data);
    return response.data;
  }

  // Get task status
  async getTaskStatus(taskId) {
    const response = await axios.get(`${this.baseURL}/generate/status/${taskId}`);
    return response.data;
  }

  // Get task result
  async getTaskResult(taskId) {
    const response = await axios.get(`${this.baseURL}/generate/result/${taskId}`);
    return response.data;
  }

  // Download file
  async downloadFile(filename) {
    const response = await axios.get(`${this.baseURL}/download/${filename}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  // Get file content as text (for preview)
  async getFileContent(filename) {
    const response = await axios.get(`${this.baseURL}/download/${filename}`);
    return response.data;
  }

  // List all files
  async listFiles() {
    const response = await axios.get(`${this.baseURL}/files/`);
    return response.data;
  }

  // Delete file
  async deleteFile(filename, fileType = 'upload') {
    const response = await axios.delete(`${this.baseURL}/files/${filename}`, {
      params: { file_type: fileType }
    });
    return response.data;
  }

  // Cleanup all files
  async cleanup() {
    const response = await axios.delete(`${this.baseURL}/cleanup/`);
    return response.data;
  }

  // Health check
  async healthCheck() {
    try {
      const response = await axios.get(`${this.baseURL}/health/`);
      return response.data;
    } catch (error) {
      throw new Error('Generator API server is not available');
    }
  }

  // Poll task until completion
  async pollTaskUntilComplete(taskId, pollInterval = 3000, maxAttempts = 100) {
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      const status = await this.getTaskStatus(taskId);
      
      if (status.status === 'completed' || status.status === 'failed') {
        return status;
      }
      
      attempts++;
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
    
    throw new Error('Task polling timeout');
  }
}

export default SubtitleGeneratorAPI;
