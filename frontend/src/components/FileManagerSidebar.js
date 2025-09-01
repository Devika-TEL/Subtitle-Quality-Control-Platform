import React from 'react';
import { Download, Trash2, RefreshCw, X, Folder, FileText } from 'lucide-react';
import './FileManagerSidebar.css';

const FileManagerSidebar = ({ isOpen, onClose, files, onRefresh, onDownload, onDelete }) => {
  return (
    <>
      {/* Overlay */}
      {isOpen && <div className="sidebar-overlay" onClick={onClose}></div>}
      
      {/* Sidebar */}
      <div className={`file-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <h3>
            <Folder className="w-5 h-5 inline mr-2" />
            File Manager
          </h3>
          <button className="close-btn" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="sidebar-content">
          <button 
            className="refresh-btn"
            onClick={onRefresh}
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Refresh Files
          </button>

          {/* Upload Files Section */}
          {files.upload_files.length > 0 && (
            <div className="file-section">
              <h4 className="section-title">
                <FileText className="w-4 h-4 inline mr-2" />
                Uploaded Files ({files.upload_files.length})
              </h4>
              <div className="file-list">
                {files.upload_files.map((filename) => (
                  <div key={filename} className="file-item">
                    <div className="file-info">
                      <div className="file-name" title={filename}>
                        {filename}
                      </div>
                      <div className="file-type">
                        {filename.split('.').pop().toLowerCase()}
                      </div>
                    </div>
                    <button 
                      className="delete-btn"
                      onClick={() => onDelete(filename, 'upload')}
                      title="Delete file"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Output Files Section */}
          {files.output_files.length > 0 && (
            <div className="file-section">
              <h4 className="section-title">
                <FileText className="w-4 h-4 inline mr-2" />
                Output Files ({files.output_files.length})
              </h4>
              <div className="file-list">
                {files.output_files.map((filename) => (
                  <div key={filename} className="file-item">
                    <div className="file-info">
                      <div className="file-name" title={filename}>
                        {filename}
                      </div>
                      <div className="file-type synchronized">
                        {filename.includes('synced') ? 'synced' : filename.split('.').pop().toLowerCase()}
                      </div>
                    </div>
                    <div className="file-actions">
                      <button 
                        className="download-btn"
                        onClick={() => onDownload(filename)}
                        title="Download file"
                      >
                        <Download className="w-4 h-4" />
                      </button>
                      <button 
                        className="delete-btn"
                        onClick={() => onDelete(filename, 'output')}
                        title="Delete file"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {files.upload_files.length === 0 && files.output_files.length === 0 && (
            <div className="empty-state">
              <Folder className="w-12 h-12 text-gray-300 mb-4" />
              <p className="text-gray-500">No files uploaded yet</p>
              <p className="text-gray-400 text-sm">Upload some files to get started</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default FileManagerSidebar;
