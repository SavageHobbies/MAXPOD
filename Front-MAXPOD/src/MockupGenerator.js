import React from 'react';
import './MockupGenerator.css';

function MockupGenerator({ selectedCategory, onGenerateMockups }) {
  return (
    <div className="mockup-generator">
      <h3>Mockup Generator for {selectedCategory}</h3>
      <button 
        onClick={onGenerateMockups}
        className="generate-button"
      >
        Generate Mockups
      </button>
    </div>
  );
}

export default MockupGenerator;
