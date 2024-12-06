
export function FeatureShowcase() {
  return (
    <div className="feature-showcase">
      <div className="feature-column">
        <div className="feature-card">
          <div className="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/><path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/></svg>
          </div>
          <div className="feature-title">Hand Gesture Recognition</div>
          <div className="feature-message">Detected: "Peace" gesture</div>
          <div className="feature-description">
            <h3>Intuitive Hand Controls</h3>
            <p>Implement natural hand gestures for seamless navigation and control in your applications</p>
          </div>
        </div>
        
        <div className="feature-card">
          <div className="feature-image">
            <div className="feature-overlay">Body pose detected: "T-pose"</div>
          </div>
          <div className="feature-description">
            <h3>Full-Body Pose Estimation</h3>
            <p>Track and analyze full-body movements for immersive gaming and fitness applications</p>
          </div>
        </div>
      </div>

      <div className="feature-column">
        <div className="feature-card">
          <div className="feature-header">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
            </div>
            <div>
              <div className="feature-title">Facial Expression Analysis</div>
              <div className="feature-subtitle">Real-time emotion detection</div>
            </div>
          </div>
          <div className="feature-content">
            <p>Detected emotions:</p>
            <ul>
              <li>Joy: 85%</li>
              <li>Surprise: 10%</li>
              <li>Neutral: 5%</li>
            </ul>
          </div>
          <div className="feature-description">
            <h3>Emotion Recognition</h3>
            <p>Enhance user experience by responding to facial expressions and emotions in real-time</p>
          </div>
        </div>

        <div className="feature-card">
          <div className="feature-message outgoing">Motion path recorded</div>
          <div className="feature-message incoming">
            <div className="feature-icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            </div>
            Analyzing movement patterns...
          </div>
          <div className="feature-description">
            <h3>Advanced Motion Tracking</h3>
            <p>Capture and analyze complex movements for sports analysis, rehabilitation, and more</p>
          </div>
        </div>
      </div>
    </div>
  )
}

