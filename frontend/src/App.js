import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Video, FileText, Download, Trash2, RefreshCw, CheckCircle, XCircle, Clock, Sidebar, Eye, EyeOff, Play, Zap } from 'lucide-react';
import SubtitleSyncAPI from './api';
import SubtitleGeneratorAPI from './generatorApi';
import FileManagerSidebar from './components/FileManagerSidebar';
import SubtitlePreview from './components/SubtitlePreview';
import SubtitleGenerator from './components/SubtitleGenerator';

const api = new SubtitleSyncAPI();
const generatorApi = new SubtitleGeneratorAPI();

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [subtitleFile, setSubtitleFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [taskStatus, setTaskStatus] = useState(null);
  const [taskResult, setTaskResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [files, setFiles] = useState({ upload_files: [], output_files: [] });
  const [showSidebar, setShowSidebar] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [originalSubtitleContent, setOriginalSubtitleContent] = useState('');
  const [syncedSubtitleContent, setSyncedSubtitleContent] = useState('');
  const [activeTab, setActiveTab] = useState('sync'); // 'sync' or 'generate'

  // Repositioning state
  const [isRepositioning, setIsRepositioning] = useState(false);
  const [repositionResult, setRepositionResult] = useState(null);
  const [repositionError, setRepositionError] = useState('');

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      const extension = file.name.toLowerCase().split('.').pop();
      
      if (['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'].includes(extension)) {
        setVideoFile(file);
        showMessage(`Video file selected: ${file.name}`, 'success');
      } else if (['srt', 'vtt'].includes(extension)) {
        setSubtitleFile(file);
        showMessage(`Subtitle file selected: ${file.name}`, 'success');
      } else {
        showMessage(`Unsupported file type: ${file.name}`, 'error');
      }
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
      'text/*': ['.srt', '.vtt']
    },
    multiple: true
  });

  const showMessage = (msg, type) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => {
      setMessage('');
      setMessageType('');
    }, 5000);
  };

  const handleSubtitleGenerated = async (outputFile, result) => {
    try {
      // Load the generated subtitle content for preview
      const content = await loadSubtitleContent(outputFile);
      setOriginalSubtitleContent(content);
      
      // Refresh file list to show the new file
      await loadFileList();
      
      // Switch to sync tab for further processing if needed
      showMessage(`Subtitle generated successfully! Switch to "Sync Subtitles" tab to fine-tune timing.`, 'success');
    } catch (error) {
      console.error('Error handling generated subtitle:', error);
    }
  };

  const readSubtitleFile = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  const loadSubtitleContent = async (filename) => {
    try {
      const response = await fetch(`http://localhost:8000/download/${filename}`);
      const content = await response.text();
      return content;
    } catch (error) {
      console.error('Error loading subtitle content:', error);
      return '';
    }
  };

  const uploadFiles = async () => {
    if (!videoFile || !subtitleFile) {
      showMessage('Please select both video and subtitle files', 'error');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    try {
      // Read original subtitle content
      const originalContent = await readSubtitleFile(subtitleFile);
      setOriginalSubtitleContent(originalContent);

      const result = await api.uploadFiles(
        videoFile,
        subtitleFile,
        (progress) => setUploadProgress(progress)
      );
      
      showMessage('Files uploaded successfully!', 'success');
      loadFileList();
    } catch (error) {
      showMessage(`Upload failed: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const startSynchronization = async () => {
    if (!videoFile || !subtitleFile) {
      showMessage('Please upload files first', 'error');
      return;
    }

    setIsProcessing(true);
    setTaskStatus(null);
    setTaskResult(null);

    try {
      const result = await api.startSync(videoFile.name, subtitleFile.name);
      setTaskId(result.task_id);
      showMessage('Synchronization started!', 'info');

      // Poll for status updates
      await api.waitForCompletion(
        result.task_id,
        (status) => {
          setTaskStatus(status);
        },
        2000
      );

      // Get final result
      const finalResult = await api.getTaskResult(result.task_id);
      setTaskResult(finalResult);
      
      if (finalResult.success) {
        showMessage('Synchronization completed successfully!', 'success');
        
        // Load synced subtitle content if available
        if (finalResult.output_file) {
          const syncedContent = await loadSubtitleContent(finalResult.output_file);
          setSyncedSubtitleContent(syncedContent);
          setShowPreview(true); // Automatically show preview after sync
        }
        
        loadFileList();
      } else {
        showMessage(`Synchronization failed: ${finalResult.message}`, 'error');
      }
    } catch (error) {
      showMessage(`Synchronization failed: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadFile = async (filename) => {
    try {
      await api.downloadFile(filename);
      showMessage(`Downloaded: ${filename}`, 'success');
    } catch (error) {
      showMessage(`Download failed: ${error.message}`, 'error');
    }
  };

  const deleteFile = async (filename, fileType) => {
    try {
      await api.deleteFile(filename, fileType);
      showMessage(`Deleted: ${filename}`, 'success');
      loadFileList();
    } catch (error) {
      showMessage(`Delete failed: ${error.message}`, 'error');
    }
  };

  const loadFileList = async () => {
    try {
      const fileList = await api.listFiles();
      setFiles(fileList);
    } catch (error) {
      console.error('Failed to load file list:', error);
    }
  };

  const clearFiles = () => {
    setVideoFile(null);
    setSubtitleFile(null);
    setTaskId(null);
    setTaskStatus(null);
    setTaskResult(null);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'processing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="w-5 h-5 text-yellow-500" />;
    }
  };

  // Load file list on component mount
  React.useEffect(() => {
    loadFileList();
  }, []);

  // Handler to trigger repositioning
  const handleReposition = async () => {
    if (!taskResult || !taskResult.output_file || !videoFile) {
      showMessage('Synchronization must be completed first', 'error');
      return;
    }
    setIsRepositioning(true);
    setRepositionResult(null);
    setRepositionError('');
    try {
      const response = await api.repositionSubtitles({
        video_filename: videoFile.name,
        subtitle_filename: taskResult.output_file,
      });
      setRepositionResult(response.output_filename);
      showMessage('Subtitle repositioning completed!', 'success');
      loadFileList();
    } catch (error) {
      setRepositionError(error.response?.data?.detail || error.message);
      showMessage(`Repositioning failed: ${error.response?.data?.detail || error.message}`, 'error');
    } finally {
      setIsRepositioning(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Toggle Button */}
      <button 
        className="sidebar-toggle"
        onClick={() => setShowSidebar(!showSidebar)}
      >
        <Sidebar className="w-5 h-5" />
        File Manager
      </button>

      {/* File Manager Sidebar */}
      <FileManagerSidebar 
        isOpen={showSidebar}
        onClose={() => setShowSidebar(false)}
        files={files}
        onRefresh={loadFileList}
        onDownload={downloadFile}
        onDelete={deleteFile}
      />

      {/* Main Content */}
      <div className={`main-content ${showSidebar ? 'with-sidebar' : ''}`}>
        <div className="container">
      {/* Header */}
      <div className="header">
        <h1>🎬 Subtitle Tools</h1>
        <p>Generate and synchronize subtitle files with video audio tracks</p>
      </div>

      {/* Tab Navigation */}
      <div className="tab-container" style={{ marginBottom: '20px' }}>
        <div className="tab-nav">
          <button
            onClick={() => setActiveTab('generate')}
            className={`tab-button ${activeTab === 'generate' ? 'active' : ''}`}
          >
            <Play className="w-4 h-4 mr-2" />
            Generate Subtitles
          </button>
          <button
            onClick={() => setActiveTab('sync')}
            className={`tab-button ${activeTab === 'sync' ? 'active' : ''}`}
          >
            <Zap className="w-4 h-4 mr-2" />
            Sync Subtitles
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'generate' ? (
        // Subtitle Generation Tab
        <SubtitleGenerator
          generatorApi={generatorApi}
          onSubtitleGenerated={handleSubtitleGenerated}
          showMessage={showMessage}
        />
      ) : (
        // Subtitle Synchronization Tab
        <>
      {/* File Upload Section */}
      <div className="card">
        <h2 style={{ marginBottom: '20px', color: '#333' }}>Upload Files</h2>
        
        <div 
          {...getRootProps()} 
          className={`upload-area ${isDragActive ? 'dragover' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="upload-icon">
            <Upload />
          </div>
          <div className="upload-text">
            {isDragActive ? 'Drop files here...' : 'Drag & drop files here, or click to select'}
          </div>
          <div className="upload-subtext">
            Supports: MP4, AVI, MOV, MKV (video) • SRT, VTT (subtitles)
          </div>
        </div>

        {/* Selected Files Display */}
        {(videoFile || subtitleFile) && (
          <div className="file-info">
            <h4>Selected Files</h4>
            <div className="file-details">
              {videoFile && (
                <div>
                  <Video className="w-4 h-4 inline mr-2" />
                  <strong>Video:</strong> {videoFile.name}
                  <br />
                  <small>Size: {SubtitleSyncAPI.formatFileSize(videoFile.size)}</small>
                </div>
              )}
              {subtitleFile && (
                <div>
                  <FileText className="w-4 h-4 inline mr-2" />
                  <strong>Subtitle:</strong> {subtitleFile.name}
                  <br />
                  <small>Size: {SubtitleSyncAPI.formatFileSize(subtitleFile.size)}</small>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Upload Progress */}
        {isUploading && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <div className="progress-text">
              Uploading... {uploadProgress}%
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="file-actions">
          <button 
            className="btn" 
            onClick={uploadFiles}
            disabled={!videoFile || !subtitleFile || isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload Files'}
          </button>
          
          <button 
            className="btn btn-secondary" 
            onClick={startSynchronization}
            disabled={!videoFile || !subtitleFile || isProcessing || isUploading}
          >
            {isProcessing ? (
              <>
                <span className="loading-spinner"></span>
                Processing...
              </>
            ) : (
              'Start Synchronization'
            )}
          </button>

          <button 
            className="btn" 
            onClick={clearFiles}
            style={{ background: '#ff5722' }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Status Messages */}
      {message && (
        <div className={`status-message status-${messageType}`}>
          {message}
        </div>
      )}

      {/* Processing Status */}
      {taskStatus && (
        <div className="card">
          <h3 style={{ marginBottom: '15px', color: '#333' }}>
            {getStatusIcon(taskStatus.status)}
            <span style={{ marginLeft: '10px' }}>Processing Status</span>
          </h3>
          
          <div className="status-message status-info">
            <strong>Status:</strong> {taskStatus.status}<br />
            <strong>Message:</strong> {taskStatus.message}
          </div>

          {taskStatus.progress !== null && (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${taskStatus.progress}%` }}
                ></div>
              </div>
              <div className="progress-text">
                Progress: {taskStatus.progress}%
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {taskResult && (
        <div className="card">
          <h3 style={{ marginBottom: '15px', color: '#333' }}>
            {taskResult.success ? (
              <CheckCircle className="w-6 h-6 text-green-500 inline mr-2" />
            ) : (
              <XCircle className="w-6 h-6 text-red-500 inline mr-2" />
            )}
            Synchronization Result
          </h3>

          <div className={`status-message ${taskResult && taskResult.success ? 'status-success' : 'status-error'}`}>
            {taskResult && taskResult.message}
          </div>

          {/* Always show download/preview if output_file is present, regardless of message */}
          {taskResult && taskResult.output_file && (
            <div className="result-item">
              <div className="result-header">
                <div className="result-title">
                  📄 {taskResult.output_file}
                </div>
              </div>
              
              <div className="result-details">
                {taskResult.processing_time && (
                  <div>Processing time: {SubtitleSyncAPI.formatProcessingTime(taskResult.processing_time)}</div>
                )}
                {taskResult.offset_applied && (
                  <div>Offset applied: {taskResult.offset_applied.toFixed(2)}s</div>
                )}
              </div>

              <div className="file-actions">
                <button 
                  className="btn btn-secondary"
                  onClick={() => downloadFile(taskResult.output_file)}
                >
                  <Download className="w-4 h-4 inline mr-2" />
                  Download Synchronized Subtitles
                </button>
                <button 
                  className="btn btn-primary ml-2"
                  onClick={() => setShowPreview(true)}
                >
                  <Eye className="w-4 h-4 inline mr-2" />
                  Preview Synchronized Subtitles
                </button>
                {/* Reposition Subtitles Button */}
                {taskResult.success && !isRepositioning && (
                  <button
                    className="btn btn-primary ml-2"
                    style={{ background: '#2196f3' }}
                    onClick={handleReposition}
                  >
                    <RefreshCw className="w-4 h-4 inline mr-2" />
                    Reposition Subtitles
                  </button>
                )}
                {isRepositioning && (
                  <button
                    className="btn btn-primary ml-2"
                    disabled
                  >
                    <span className="loading-spinner"></span>
                    Repositioning...
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Reposition Result Display */}
      {repositionResult && (
        <div className="card">
          <h3 style={{ marginBottom: '15px', color: '#333' }}>
            <CheckCircle className="w-6 h-6 text-blue-500 inline mr-2" />
            Subtitle Repositioning Result
          </h3>
          <div className="result-item">
            <div className="result-title">📄 {repositionResult}</div>
            <br></br>
            <button
              className="btn btn-secondary"
              onClick={() => downloadFile(repositionResult)}
            >
              <Download className="w-4 h-4 inline mr-2" />
              Download Repositioned Subtitles
            </button>
          </div>
        </div>
      )}

      {/* Reposition Error Display */}
      {repositionError && (
        <div className="status-message status-error">
          {repositionError}
        </div>
      )}

      {/* Subtitle Preview */}
      {showPreview && (originalSubtitleContent || syncedSubtitleContent) && (
        <SubtitlePreview 
          originalContent={originalSubtitleContent}
          syncedContent={syncedSubtitleContent}
          onClose={() => setShowPreview(false)}
        />
      )}
        </>
      )}
      </div>
    </div>
    </div>
  );
}

export default App;
