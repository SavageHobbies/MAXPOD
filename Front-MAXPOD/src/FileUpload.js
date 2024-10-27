import React, { useState } from 'react';

function FileUpload({ onUploadComplete }) {
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleFileChange = (event) => {
    setSelectedFiles(Array.from(event.target.files));
  };

  const handleUpload = () => {
    // Here you would typically send the files to your server
    // For now, we'll just pass the file names to the parent component
    const fileNames = selectedFiles.map(file => file.name);
    onUploadComplete(fileNames);
  };

  return (
    <div>
      <h3>Upload Your Design Files</h3>
      <input type="file" multiple onChange={handleFileChange} />
      <p>Selected Files: {selectedFiles.length}</p>
      <ul>
        {selectedFiles.map((file, index) => (
          <li key={index}>{file.name}</li>
        ))}
      </ul>
      <button onClick={handleUpload}>Upload Files</button>
    </div>
  );
}

export default FileUpload;
