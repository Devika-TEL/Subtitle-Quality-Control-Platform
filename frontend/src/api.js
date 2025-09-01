import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'


class SubtitleSyncAPI {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Upload video and subtitle files
  async uploadFiles(videoFile, subtitleFile, onProgress = null) {
    const formData = new FormData();
    formData.append('video_file', videoFile);
    formData.append('subtitle_file', subtitleFile);

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

    const response = await axios.post(`${this.baseURL}/upload-files/`, formData, config);
    return response.data;
  }

  // Start synchronization process
  async startSync(videoFilename, subtitleFilename, outputFilename = null) {
    const data = {
      video_filename: videoFilename,
      subtitle_filename: subtitleFilename,
    };

    if (outputFilename) {
      data.output_filename = outputFilename;
    }

    const response = await axios.post(`${this.baseURL}/sync/`, data);
    return response.data;
  }

  // Get task status
  async getTaskStatus(taskId) {
    const response = await axios.get(`${this.baseURL}/status/${taskId}`);
    return response.data;
  }

  // Get task result
  async getTaskResult(taskId) {
    const response = await axios.get(`${this.baseURL}/result/${taskId}`);
    return response.data;
  }

  // Download synchronized subtitle file
  async downloadFile(filename) {
    const response = await axios.get(`${this.baseURL}/download/${filename}`, {
      responseType: 'blob',
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
    
    return true;
  }

  // List all files
  async listFiles() {
    const response = await axios.get(`${this.baseURL}/files/`);
    return response.data;
  }

  // Delete a file
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
    const response = await axios.get(`${this.baseURL}/health/`);
    return response.data;
  }

  // Utility method to poll task status until completion
  async waitForCompletion(taskId, onStatusUpdate = null, pollInterval = 2000) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getTaskStatus(taskId);
          
          if (onStatusUpdate) {
            onStatusUpdate(status);
          }

          if (status.status === 'completed' || status.status === 'failed') {
            resolve(status);
          } else {
            setTimeout(poll, pollInterval);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }

  // Reposition subtitles
  async repositionSubtitles(data) {
    const response = await axios.post(`${this.baseURL}/reposition/`, data);
    return response.data;
  }

  // Format file size
  static formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Format processing time
  static formatProcessingTime(seconds) {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
    }
  }
}

export default SubtitleSyncAPI;
