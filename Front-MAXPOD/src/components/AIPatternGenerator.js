import React, { useState } from 'react';
import './AIPatternGenerator.css';

const AIPatternGenerator = () => {
  const [idea, setIdea] = useState('');
  const [patterns, setPatterns] = useState(3);
  const [selectedModel, setSelectedModel] = useState('nemotron-mini');
  const [generatedPatterns, setGeneratedPatterns] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const llmOptions = {
    'nemotron-mini': 'Nemotron Mini',
    'llama2': 'Llama 2',
    'gpt-4': 'GPT-4',
    'gpt-3.5-turbo': 'GPT-3.5 Turbo',
    'claude-3-opus': 'Claude 3 Opus',
    'claude-3-sonnet': 'Claude 3 Sonnet',
    'gemini-pro': 'Gemini Pro',
    'mistral': 'Mistral'
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8080/process_patterns', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patterns: parseInt(patterns),
          idea: idea,
          llm_config: selectedModel
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedPatterns(data.patterns);
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to generate patterns: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="ai-pattern-generator">
      <h2>AI Pattern Generator</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="llm-model">LLM Model:</label>
          <select
            id="llm-model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {Object.entries(llmOptions).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="patterns">Number of Patterns:</label>
          <input
            type="number"
            id="patterns"
            min="1"
            max="10"
            value={patterns}
            onChange={(e) => setPatterns(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label htmlFor="idea">Your Idea:</label>
          <textarea
            id="idea"
            value={idea}
            onChange={(e) => setIdea(e.target.value)}
            placeholder="Enter your design idea..."
            required
          />
        </div>

        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate Patterns'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {generatedPatterns.length > 0 && (
        <div className="generated-patterns">
          <h3>Generated Patterns</h3>
          <div className="patterns-grid">
            {generatedPatterns.map((pattern, index) => (
              <div key={index} className="pattern-card">
                <h4>{pattern.product_name}</h4>
                <p><strong>Text:</strong> {pattern.tshirt_text}</p>
                <p><strong>Description:</strong> {pattern.description}</p>
                <div className="tags">
                  {pattern.marketing_tags.map((tag, tagIndex) => (
                    <span key={tagIndex} className="tag">{tag}</span>
                  ))}
                </div>
                {pattern.image_ids && (
                  <div className="pattern-images">
                    {pattern.image_ids.map((imageId, imgIndex) => (
                      <img 
                        key={imgIndex}
                        src={`/api/images/${imageId}`}
                        alt={`Pattern ${index + 1} - Image ${imgIndex + 1}`}
                      />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AIPatternGenerator;