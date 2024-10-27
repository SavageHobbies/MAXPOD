import React, { useState } from 'react';
import './App.css';
import AIPatternGenerator from './components/AIPatternGenerator';

const mockupCategories = [
  'blankets', 'hoodies', 'mugs', 'ornaments', 'phone', 'pillows', 
  'puzzles', 'sweatshirts', 'totes', 'tshirt', 'tumblers'
];

function App() {
  const [activeTab, setActiveTab] = useState('mockups');
  const [step, setStep] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [thumbnails, setThumbnails] = useState([]);
  const [selectedThumbnails, setSelectedThumbnails] = useState([]);
  const [uploadedDesigns, setUploadedDesigns] = useState([]);
  const [error, setError] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedMockups, setGeneratedMockups] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchThumbnails = async (category) => {
    try {
      setIsLoading(true);
      const response = await fetch(`https://printify.trendsetterz.buzz/get_thumbs.php?category=${category}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      if (Array.isArray(data)) {
        setThumbnails(data);
        setError(null);
      } else if (data.error) {
        throw new Error(data.error);
      } else {
        throw new Error('Invalid data received from server');
      }
    } catch (e) {
      setError(`Failed to fetch thumbnails: ${e.message}`);
      setThumbnails([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCategoryChange = (e) => {
    setSelectedCategory(e.target.value);
    setSelectedThumbnails([]);
    setStep(2);
    fetchThumbnails(e.target.value);
  };

  const handleThumbnailClick = (thumbnail) => {
    if (selectedThumbnails.includes(thumbnail)) {
      setSelectedThumbnails(selectedThumbnails.filter(t => t !== thumbnail));
    } else if (selectedThumbnails.length < 10) {
      setSelectedThumbnails([...selectedThumbnails, thumbnail]);
    }
  };

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files);
    setIsLoading(true);
    try {
      const newDesigns = await Promise.all(files.map(async (file) => {
        const preview = await createTransparentPreview(file);
        
        // Get a pre-signed URL from your backend
        const response = await fetch('https://843b52youc.execute-api.us-east-2.amazonaws.com/Production/get-upload-url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ fileName: file.name }),
        });
        const { uploadUrl } = await response.json();
  
        // Upload the file to S3
        await fetch(uploadUrl, {
          method: 'PUT',
          body: file,
        });
  
        return { file: { name: file.name }, preview };
      }));
      setUploadedDesigns(newDesigns);
      setStep(3);
    } catch (error) {
      console.error("Error processing designs:", error);
      setError("Failed to process designs. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };
  
  const createTransparentPreview = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          
          // Clear the canvas with a transparent background
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          // Draw the image
          ctx.drawImage(img, 0, 0);
          
          resolve(canvas.toDataURL('image/png'));
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  };

  const handleGenerateMockups = async () => {
    if (selectedThumbnails.length > 0 && uploadedDesigns.length > 0) {
      setIsGenerating(true);
      try {
        const generatedMockups = await generateMockups(selectedThumbnails, uploadedDesigns, selectedCategory);
        setGeneratedMockups(generatedMockups);
        setIsGenerating(false);
        setStep(4);
      } catch (error) {
        console.error("Error generating mockups:", error);
        setError("Failed to generate mockups. Please try again.");
        setIsGenerating(false);
      }
    }
  };
  
  const generateMockups = async (templates, designs, category) => {
    try {
      const response = await fetch('http://localhost:8080/generate_mockups', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          templates: templates,
          designs: designs.map(d => d.file.name),
          category: category
        }),
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      const result = await response.json();
      return result.mockups;
    } catch (error) {
      console.error("Error generating mockups:", error);
      throw error;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>MAX POD</h1>
        <nav>
          <button 
            className={`tab-button ${activeTab === 'mockups' ? 'active' : ''}`}
            onClick={() => setActiveTab('mockups')}
          >
            Mockup Generator
          </button>
          <button 
            className={`tab-button ${activeTab === 'ai' ? 'active' : ''}`}
            onClick={() => setActiveTab('ai')}
          >
            AI Pattern Generator
          </button>
        </nav>
      </header>

      <main>
        {activeTab === 'ai' ? (
          <AIPatternGenerator />
        ) : (
          <>
            {isLoading ? (
              <div className="loading-container">
                <div className="loading-robot">🤖</div>
                <p>Hold on, Max is fetching designs...</p>
              </div>
            ) : error ? (
              <div className="error-message">
                <p>{error}</p>
              </div>
            ) : (
              <>
                {step === 1 && (
                  <section className="product-selection">
                    <h2>Select Product Category</h2>
                    <select 
                      value={selectedCategory} 
                      onChange={handleCategoryChange}
                    >
                      <option value="">Select a category</option>
                      {mockupCategories.map(category => (
                        <option key={category} value={category}>{category}</option>
                      ))}
                    </select>
                  </section>
                )}

                {step >= 2 && step < 4 && (
                  <section className="design-upload">
                    <h2>Upload Designs for {selectedCategory}</h2>
                    <input 
                      type="file" 
                      multiple 
                      onChange={handleFileChange} 
                      accept="image/*"
                      id="fileInput"
                      style={{display: 'none'}}
                    />
                    <label htmlFor="fileInput" className="custom-file-upload">
                      Choose Files
                    </label>
                    {uploadedDesigns.length > 0 && (
                      <span>{uploadedDesigns.length} files</span>
                    )}
                    {uploadedDesigns.length > 0 && (
                      <div className="uploaded-designs">
                        <h3>Uploaded Designs:</h3>
                        <div className="design-grid">
                          {uploadedDesigns.map((design, index) => (
                            <div key={index} className="design-item">
                              <img src={design.preview} alt={`Design ${index + 1}`} />
                              <p>{design.file.name}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </section>
                )}

                {step === 3 && selectedCategory && (
                  <section className="thumbnail-selection">
                    <h3>Select up to 10 templates for {selectedCategory}</h3>
                    <div className="thumbnail-grid">
                      {thumbnails.map((thumbnail, index) => (
                        <div 
                          key={index} 
                          className={`thumbnail-item ${selectedThumbnails.includes(thumbnail) ? 'selected' : ''}`}
                          onClick={() => handleThumbnailClick(thumbnail)}
                        >
                          <img 
                            src={`https://printify.trendsetterz.buzz/mockups/${selectedCategory}/${thumbnail}`} 
                            alt={`${selectedCategory} thumbnail ${index + 1}`}
                          />
                        </div>
                      ))}
                    </div>
                    <p>Selected: {selectedThumbnails.length} / 10</p>
                    {selectedThumbnails.length > 0 && (
                      <button 
                        onClick={handleGenerateMockups} 
                        disabled={isGenerating}
                        className="generate-button"
                      >
                        {isGenerating ? "Generating..." : "Generate Mockups"}
                      </button>
                    )}
                  </section>
                )}

                {isGenerating && (
                  <div className="loading-overlay">
                    <div className="loading-content">
                      <div className="loading-robot">🤖</div>
                      <p>Please wait while Max generates your mockups...</p>
                    </div>
                  </div>
                )}

                {step === 4 && generatedMockups.length > 0 && (
                  <section className="generated-mockups">
                    <h2>Generated Mockups:</h2>
                    <div className="mockup-grid">
                      {generatedMockups.map((mockup) => (
                        <div key={mockup.id} className="mockup-item">
                          <div className="mockup-image-container">
                            <img src={`https://max-pod-designs.s3.amazonaws.com/${mockup.mockupKey}`} alt={`Template ${mockup.templateName}`} className="mockup-image" />
                          </div>
                          <p>Template: {mockup.templateName}</p>
                          <p>Design: {mockup.designName}</p>
                        </div>
                      ))}
                    </div>
                  </section>
                )}
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
