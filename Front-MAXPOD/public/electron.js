const path = require('path');
const { app, BrowserWindow } = require('electron');
const isDev = require('electron-is-dev');
const express = require('express');
const expressApp = express();
const multer = require('multer');
const cors = require('cors');

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  );

  if (isDev) {
    win.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Express server setup
expressApp.use(express.json());
expressApp.use(cors());

// Configure multer for file upload
const upload = multer({ dest: 'uploads/' });

expressApp.get('/api/hello', (req, res) => {
  res.json({ message: 'Hello from Express!' });
});

expressApp.post('/api/upload', upload.array('files'), (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).send('No files were uploaded.');
  }
  res.send('Files uploaded successfully');
});

expressApp.listen(3001, () => {
  console.log('Express server running on port 3001');
});