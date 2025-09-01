import React, { useState, useEffect } from 'react';
import { Upload, Video, FileText, Download, RefreshCw, CheckCircle, XCircle, Clock, Settings, Play } from 'lucide-react';

const SubtitleGenerator = ({ 
  generatorApi, 
  onSubtitleGenerated, 
  showMessage 
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('base');
  const [selectedFormat, setSelectedFormat] = useState('srt');
  const [generatedFile, setGeneratedFile] = useState(null);
  const [isServerAvailable, setIsServerAvailable] = useState(true);

  // Check server availability and load models on component mount
  useEffect(() => {
    checkServerAndLoadModels();
  }, []);

  const checkServerAndLoadModels = async () => {
    try {
      await generatorApi.healthCheck();
      const modelsData = await generatorApi.getAvailableModels();
      setAvailableModels(modelsData.models || []);
      setIsServerAvailable(true);
    } catch (error) {
      setIsServerAvailable(false);
      showMessage('Subtitle Tools API server is not available. Please start it on port 8000.', 'error');
    }
  };

  const handleVideoUpload = (file) => {
    const extension = file.name.toLowerCase().split('.').pop();
    
    if (['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'].includes(extension)) {
      setVideoFile(file);
      showMessage(`Video file selected for subtitle generation: ${file.name}`, 'success');
    } else {
      showMessage(`Unsupported video file type: ${file.name}`, 'error');
    }
  };

  const uploadVideoFile = async () => {
    if (!videoFile) {
      showMessage('Please select a video file first', 'error');
      return false;
    }

    try {
      setUploadProgress(0);
      const result = await generatorApi.uploadVideo(
        videoFile,
        (progress) => setUploadProgress(progress)
      );
      
      showMessage(`Video uploaded successfully: ${result.filename}`, 'success');
      return true;
    } catch (error) {
      showMessage(`Failed to upload video: ${error.message}`, 'error');
      return false;
    }
  };

  const startGeneration = async () => {
    if (!videoFile) {
      showMessage('Please select a video file first', 'error');
      return;
    }

    setIsGenerating(true);
    setGenerationProgress(0);
    setTaskStatus(null);
    setGeneratedFile(null);

    try {
      // First upload the video
      const uploadSuccess = await uploadVideoFile();
      if (!uploadSuccess) {
        setIsGenerating(false);
        return;
      }

      // Start generation
      const result = await generatorApi.startGeneration(
        videoFile.name,
        selectedFormat,
        selectedModel
      );

      setTaskId(result.task_id);
      showMessage(`Subtitle generation started using ${selectedModel} model`, 'info');

      // Poll for status updates
      pollGenerationStatus(result.task_id);
      
    } catch (error) {
      showMessage(`Failed to start generation: ${error.message}`, 'error');
      setIsGenerating(false);
    }
  };

  const pollGenerationStatus = async (taskId) => {
    try {
      const status = await generatorApi.pollTaskUntilComplete(taskId, 3000);
      
      if (status.status === 'completed') {
        const result = await generatorApi.getTaskResult(taskId);
        setGeneratedFile(result.output_file);
        showMessage(`Subtitles generated successfully! ${result.segments_count} segments created.`, 'success');
        
        // Notify parent component about the generated subtitle
        if (onSubtitleGenerated) {
          onSubtitleGenerated(result.output_file, result);
        }
      } else if (status.status === 'failed') {
        showMessage(`Generation failed: ${status.message}`, 'error');
      }
      
      setTaskStatus(status);
      setIsGenerating(false);
      
    } catch (error) {
      showMessage(`Generation failed: ${error.message}`, 'error');
      setIsGenerating(false);
    }
  };

  const downloadGeneratedFile = async () => {
    if (!generatedFile) return;

    try {
      const blob = await generatorApi.downloadFile(generatedFile);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = generatedFile;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      showMessage(`Downloaded: ${generatedFile}`, 'success');
    } catch (error) {
      showMessage(`Failed to download file: ${error.message}`, 'error');
    }
  };

  if (!isServerAvailable) {
    return (
      <div className="card" style={{ backgroundColor: '#fff3cd', borderLeft: '4px solid #ffc107' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '15px' }}>
          <XCircle style={{ color: '#856404', marginRight: '10px' }} size={24} />
          <h3 style={{ color: '#856404', margin: 0 }}>Generator Service Unavailable</h3>
        </div>
        <p style={{ color: '#856404', marginBottom: '15px' }}>
          The Subtitle Tools API server is not running. To use subtitle generation features:
        </p>
        <div style={{ backgroundColor: '#f8f9fa', padding: '15px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.9em', color: '#856404', marginBottom: '15px' }}>
          python subtitle_tools_api.py
        </div>
        <button
          onClick={checkServerAndLoadModels}
          className="btn"
          style={{ background: '#ffc107', display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <RefreshCw size={16} />
          <span>Retry Connection</span>
        </button>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '25px' }}>
        <Play style={{ color: '#4CAF50', marginRight: '12px' }} size={24} />
        <h3 style={{ color: '#333', margin: 0, fontSize: '1.3em' }}>Generate Subtitles from Video</h3>
      </div>

      {/* Video Upload Section */}
      <div style={{ marginBottom: '25px' }}>
        <label style={{ display: 'block', fontWeight: '600', color: '#333', marginBottom: '10px' }}>
          Select Video File
        </label>
        <div 
          className="upload-area" 
          style={{ 
            border: videoFile ? '2px dashed #4CAF50' : '2px dashed #ccc',
            backgroundColor: videoFile ? 'rgba(76, 175, 80, 0.05)' : 'transparent'
          }}
        >
          <input
            type="file"
            accept=".mp4,.avi,.mov,.mkv,.wmv,.flv,.webm"
            onChange={(e) => e.target.files[0] && handleVideoUpload(e.target.files[0])}
            className="file-input"
            id="video-upload"
          />
          <label htmlFor="video-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <div className="upload-icon">
              <Video size={48} />
            </div>
            <div className="upload-text">
              {videoFile ? videoFile.name : 'Click to select video file or drag and drop'}
            </div>
            <div className="upload-subtext">
              Supports: MP4, AVI, MOV, MKV, WMV, FLV, WebM
            </div>
          </label>
        </div>

        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="progress-container">
            <div className="progress-text" style={{ textAlign: 'left', marginBottom: '5px' }}>
              <span>Uploading... {uploadProgress}%</span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Generation Settings */}
      <div className="file-details">
        <div>
          <label style={{ display: 'block', fontWeight: '600', color: '#333', marginBottom: '8px' }}>
            Whisper Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isGenerating}
            style={{
              width: '100%',
              padding: '10px 12px',
              border: '2px solid #ddd',
              borderRadius: '8px',
              fontSize: '0.95em',
              backgroundColor: isGenerating ? '#f5f5f5' : 'white',
              cursor: isGenerating ? 'not-allowed' : 'pointer'
            }}
          >
            {availableModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name} - {model.description}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontWeight: '600', color: '#333', marginBottom: '8px' }}>
            Output Format
          </label>
          <select
            value={selectedFormat}
            onChange={(e) => setSelectedFormat(e.target.value)}
            disabled={isGenerating}
            style={{
              width: '100%',
              padding: '10px 12px',
              border: '2px solid #ddd',
              borderRadius: '8px',
              fontSize: '0.95em',
              backgroundColor: isGenerating ? '#f5f5f5' : 'white',
              cursor: isGenerating ? 'not-allowed' : 'pointer'
            }}
          >
            <option value="srt">SRT</option>
            <option value="vtt">VTT</option>
          </select>
        </div>
      </div>

      {/* Generation Button */}
      <button
        onClick={startGeneration}
        disabled={!videoFile || isGenerating}
        className="btn"
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px',
          padding: '15px 20px',
          backgroundColor: (!videoFile || isGenerating) ? '#ccc' : '#4CAF50',
          cursor: (!videoFile || isGenerating) ? 'not-allowed' : 'pointer',
          opacity: (!videoFile || isGenerating) ? 0.6 : 1,
          marginTop: '20px'
        }}
      >
        {isGenerating ? (
          <>
            <RefreshCw className="loading-spinner" size={20} />
            <span>Generating Subtitles...</span>
          </>
        ) : (
          <>
            <Play size={20} />
            <span>Generate Subtitles</span>
          </>
        )}
      </button>

      {/* Generation Status */}
      {taskStatus && (
        <div className="result-item" style={{ marginTop: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
            {taskStatus.status === 'completed' ? (
              <CheckCircle style={{ color: '#4CAF50', marginRight: '8px' }} size={20} />
            ) : taskStatus.status === 'failed' ? (
              <XCircle style={{ color: '#f44336', marginRight: '8px' }} size={20} />
            ) : (
              <Clock style={{ color: '#ff9800', marginRight: '8px' }} size={20} />
            )}
            <span style={{ fontWeight: '600', textTransform: 'capitalize' }}>{taskStatus.status}</span>
          </div>
          
          <p style={{ color: '#666', fontSize: '0.95em', marginBottom: '10px' }}>{taskStatus.message}</p>
          
          {taskStatus.progress && (
            <div style={{ marginTop: '15px' }}>
              <div className="progress-text" style={{ marginBottom: '5px' }}>
                <span>Progress: {taskStatus.progress}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ 
                    width: `${taskStatus.progress}%`,
                    background: 'linear-gradient(90deg, #4CAF50, #81C784)'
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Generated File Download */}
      {generatedFile && taskStatus?.status === 'completed' && (
        <div className="result-item" style={{ marginTop: '20px', backgroundColor: '#d4edda', borderLeft: '4px solid #4CAF50' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <FileText style={{ color: '#4CAF50', marginRight: '8px' }} size={20} />
              <span style={{ fontWeight: '600', color: '#155724' }}>
                Generated: {generatedFile}
              </span>
            </div>
            <button
              onClick={downloadGeneratedFile}
              className="btn btn-secondary"
              style={{ 
                padding: '8px 12px', 
                fontSize: '0.9em',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <Download size={16} />
              <span>Download</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SubtitleGenerator;
