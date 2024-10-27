import React from 'react';
import './GeneratedMockups.css';

function GeneratedMockups({ mockups }) {
  return (
    <div className="generated-mockups">
      <h2>Generated Mockups</h2>
      <div className="mockup-grid">
        {mockups.map((mockup, index) => (
          <div key={index} className="mockup-item">
            <img
              src={`https://max-pod-designs.s3.amazonaws.com/${mockup.mockupKey}`}
              alt={`Mockup ${index + 1}`}
              className="mockup-image"
            />
            <p>Template: {mockup.templateName}</p>
            <p>Design: {mockup.designName}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GeneratedMockups;
