# Subtitle Synchronizer React Frontend

A modern, responsive React frontend for the Subtitle Synchronizer API. This web application provides an intuitive interface for uploading video and subtitle files, monitoring synchronization progress, and downloading results.

## Features

- **Drag & Drop Interface**: Intuitive file upload with drag-and-drop support
- **Real-time Progress**: Live progress tracking for uploads and synchronization
- **File Management Sidebar**: Dedicated sidebar for managing uploaded and output files
- **Subtitle Preview**: Side-by-side comparison of original and synchronized subtitles
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Status Monitoring**: Real-time updates on synchronization progress
- **Error Handling**: Comprehensive error messages and user feedback
- **Modern UI**: Clean, gradient-based design with smooth animations

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- FastAPI backend running on `http://localhost:8000`

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Development

Start the development server:
```bash
npm start
```

The application will open in your browser at `http://localhost:3000`.

## Building for Production

Create a production build:
```bash
npm run build
```

The built files will be in the `build` directory.

## Project Structure

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── components/
│   │   ├── FileManagerSidebar.js    # File management sidebar
│   │   ├── FileManagerSidebar.css   # Sidebar styles
│   │   ├── SubtitlePreview.js       # Subtitle comparison view
│   │   └── SubtitlePreview.css      # Preview styles
│   ├── App.js             # Main application component
│   ├── index.js           # React entry point
│   ├── index.css          # Global styles
│   └── api.js             # API client for backend communication
├── package.json           # Dependencies and scripts
└── README.md             # This file
```

## Key Components

### App.js
The main application component that handles:
- File selection and upload
- Synchronization process management
- Progress monitoring
- File management operations
- User interface state
- Sidebar and preview management

### FileManagerSidebar.js
A slide-out sidebar component for file management:
- Lists uploaded and output files
- Provides download and delete actions
- Shows file types and status
- Responsive design for mobile

### SubtitlePreview.js
A modal component for subtitle comparison:
- Side-by-side view of original and synchronized subtitles
- Highlights timing differences
- Shows synchronization statistics
- Supports both SRT and VTT formats

### api.js
API client class that provides methods for:
- File uploads with progress tracking
- Starting synchronization tasks
- Polling task status
- Downloading results
- File management operations

## Features in Detail

### File Upload
- Supports drag-and-drop for multiple files
- Validates file types (video: mp4, avi, mov, mkv, wmv, flv, webm; subtitles: srt, vtt)
- Shows upload progress with animated progress bar
- Displays file information (name, size)
- Automatically reads subtitle content for preview

### Synchronization Process
- One-click synchronization start
- Real-time status updates
- Progress tracking with percentage
- Background processing with status polling
- Detailed results display
- Automatic preview after completion

### File Management Sidebar
- Slide-out sidebar accessible from top-right button
- Organized display of uploaded and output files
- Color-coded file status (original vs synchronized)
- Quick download and delete actions
- File type indicators and counts
- Mobile-responsive design

### Subtitle Preview
- Modal overlay with side-by-side comparison
- Original subtitles on the left, synchronized on the right
- Timing difference indicators for each subtitle
- Clickable subtitle selection for easy navigation
- Summary statistics (total, synchronized, changes)
- Support for both SRT and VTT formats
- Scrollable lists for long subtitle files

### User Experience
- Responsive design for all screen sizes
- Loading states and spinners
- Success/error message notifications
- Intuitive icons and visual feedback
- Smooth animations and transitions
- Keyboard navigation support

## API Integration

The frontend communicates with the FastAPI backend through the following endpoints:

- `POST /upload-files/` - File uploads
- `POST /sync/` - Start synchronization
- `GET /status/{task_id}` - Check progress
- `GET /result/{task_id}` - Get results
- `GET /download/{filename}` - Download files
- `GET /files/` - List files
- `DELETE /files/{filename}` - Delete files

## Customization

### Styling
Modify `src/index.css` to customize:
- Color scheme and gradients
- Layout and spacing
- Animations and transitions
- Responsive breakpoints

### API Configuration
Update `src/api.js` to:
- Change the backend URL
- Modify request/response handling
- Add authentication headers
- Customize error handling

### UI Components
Extend `src/App.js` to:
- Add new features
- Modify the user interface
- Change the workflow
- Add additional file types

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Development Tips

1. **Hot Reload**: The development server automatically reloads when you make changes
2. **Console Debugging**: Check browser console for API errors and debug information
3. **Network Tab**: Monitor API requests in browser developer tools
4. **React DevTools**: Install React Developer Tools browser extension for debugging

## Production Deployment

### Using Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /path/to/build;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Using PM2 with Serve
```bash
npm install -g serve pm2
npm run build
pm2 start "serve -s build -l 3000" --name subtitle-frontend
```

### Environment Variables
Create a `.env` file for production configuration:
```
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_MAX_FILE_SIZE=100MB
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the FastAPI backend has proper CORS configuration
2. **API Connection**: Verify the backend is running on the correct port
3. **File Upload Failures**: Check file size limits and supported formats
4. **Build Errors**: Clear node_modules and reinstall dependencies

### Debug Mode
Add debug logging by modifying `api.js`:
```javascript
// Add to API calls
console.log('API Request:', endpoint, data);
console.log('API Response:', response);
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Subtitle Synchronizer application suite.
