import React, { useState, useEffect } from 'react';
import { X, Clock, ArrowRight, Download } from 'lucide-react';
import './SubtitlePreview.css';

const SubtitlePreview = ({ originalContent, syncedContent, onClose }) => {
  const [originalSubtitles, setOriginalSubtitles] = useState([]);
  const [syncedSubtitles, setSyncedSubtitles] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    if (originalContent) {
      setOriginalSubtitles(parseSubtitleContent(originalContent));
    }
    if (syncedContent) {
      setSyncedSubtitles(parseSubtitleContent(syncedContent));
    }
  }, [originalContent, syncedContent]);

  const parseSubtitleContent = (content) => {
    const subtitles = [];
    
    if (content.trim().startsWith('WEBVTT')) {
      // Parse VTT format
      const lines = content.split('\n');
      let i = 0;
      let index = 1;
      
      while (i < lines.length) {
        const line = lines[i].trim();
        const timeMatch = line.match(/(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/);
        
        if (timeMatch) {
          const startTime = timeMatch[0].split(' --> ')[0];
          const endTime = timeMatch[0].split(' --> ')[1];
          
          // Collect subtitle text
          const textLines = [];
          i++;
          while (i < lines.length && lines[i].trim()) {
            textLines.push(lines[i].trim());
            i++;
          }
          
          if (textLines.length > 0) {
            subtitles.push({
              index,
              startTime,
              endTime,
              text: textLines.join('\n')
            });
            index++;
          }
        }
        i++;
      }
    } else {
      // Parse SRT format
      const blocks = content.split(/\n\s*\n/);
      
      blocks.forEach(block => {
        const lines = block.trim().split('\n');
        if (lines.length >= 3) {
          const index = parseInt(lines[0]);
          const timeLine = lines[1];
          const text = lines.slice(2).join('\n');
          
          if (timeLine.includes('-->')) {
            const [startTime, endTime] = timeLine.split(' --> ');
            subtitles.push({
              index,
              startTime: startTime.trim(),
              endTime: endTime.trim(),
              text
            });
          }
        }
      });
    }
    
    return subtitles;
  };

  const formatTimeDifference = (original, synced) => {
    if (!original || !synced) return '';
    
    const parseTime = (timeStr) => {
      const parts = timeStr.replace(',', '.').split(':');
      return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    };
    
    const originalSeconds = parseTime(original);
    const syncedSeconds = parseTime(synced);
    const diff = syncedSeconds - originalSeconds;
    
    if (Math.abs(diff) < 0.05) return 'No change';
    
    const sign = diff > 0 ? '+' : '';
    return `${sign}${diff.toFixed(2)}s`;
  };

  const maxSubtitles = Math.max(originalSubtitles.length, syncedSubtitles.length);

  return (
    <div className="subtitle-preview-overlay">
      <div className="subtitle-preview-modal">
        <div className="preview-header">
          <h3>
            <Clock className="w-5 h-5 inline mr-2" />
            Subtitle Comparison
          </h3>
          <button className="close-preview-btn" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="preview-content">
          <div className="comparison-container">
            {/* Original Subtitles Column */}
            <div className="subtitle-column">
              <h4 className="column-title">
                📄 Original Subtitles
                <span className="subtitle-count">({originalSubtitles.length})</span>
              </h4>
              <div className="subtitle-list">
                {originalSubtitles.map((sub, index) => (
                  <div 
                    key={`original-${index}`}
                    className={`subtitle-item ${selectedIndex === index ? 'selected' : ''}`}
                    onClick={() => setSelectedIndex(index)}
                  >
                    <div className="subtitle-header">
                      <span className="subtitle-index">#{sub.index}</span>
                      <span className="subtitle-time">
                        {sub.startTime} → {sub.endTime}
                      </span>
                    </div>
                    <div className="subtitle-text">{sub.text}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Arrow Separator */}
            <div className="separator">
              <ArrowRight className="w-6 h-6 text-blue-500" />
              <div className="sync-label">SYNCHRONIZED</div>
            </div>

            {/* Synced Subtitles Column */}
            <div className="subtitle-column">
              <h4 className="column-title">
                ✅ Synchronized Subtitles
                <span className="subtitle-count">({syncedSubtitles.length})</span>
              </h4>
              <div className="subtitle-list">
                {syncedSubtitles.map((sub, index) => {
                  const originalSub = originalSubtitles[index];
                  const timeDiff = originalSub ? 
                    formatTimeDifference(originalSub.startTime, sub.startTime) : '';
                  
                  return (
                    <div 
                      key={`synced-${index}`}
                      className={`subtitle-item synced ${selectedIndex === index ? 'selected' : ''}`}
                      onClick={() => setSelectedIndex(index)}
                    >
                      <div className="subtitle-header">
                        <span className="subtitle-index">#{sub.index}</span>
                        <span className="subtitle-time">
                          {sub.startTime} → {sub.endTime}
                        </span>
                        {timeDiff && (
                          <span className={`time-diff ${timeDiff === 'No change' ? 'no-change' : 'changed'}`}>
                            {timeDiff}
                          </span>
                        )}
                      </div>
                      <div className="subtitle-text">{sub.text}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="preview-stats">
            <div className="stat-item">
              <span className="stat-label">Total Subtitles:</span>
              <span className="stat-value">{maxSubtitles}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Synchronized:</span>
              <span className="stat-value">{syncedSubtitles.length}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Changes Applied:</span>
              <span className="stat-value">
                {syncedSubtitles.filter((sub, index) => {
                  const original = originalSubtitles[index];
                  return original && original.startTime !== sub.startTime;
                }).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubtitlePreview;
