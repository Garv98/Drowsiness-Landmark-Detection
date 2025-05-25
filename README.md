<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
</head>
<body>
  <h1>Drowsiness Landmark Detection</h1>
  <p>A Streamlit-based application for detecting driver drowsiness using computer vision and a deep learning model. It supports real-time webcam monitoring and video file uploads, providing engaging alerts to help keep drivers awake and safe on the road.</p>
  <hr />

  <div class="section">
    <h2>🚀 Features</h2>
    <ul>
      <li><strong>Real-Time Detection</strong>: Monitor webcam feed and overlay visual/tactile alerts when sustained drowsiness is detected.</li>
      <li><strong>Video File Prediction</strong>: Upload an MP4 video to analyze drowsiness and download/save the processed output.</li>
      <li><strong>Engaging Alerts</strong>:
        <ul>
          <li>Audio alarms (WAV or TTS fallback)</li>
          <li>Confetti animation on detection</li>
          <li>Random trivia tips in the sidebar</li>
          <li>Reaction-test and mirror-check prompts to confirm wakefulness</li>
        </ul>
      </li>
      <li><strong>Performance Optimizations</strong>:
        <ul>
          <li>Frame-based threshold to reduce false positives from blinks</li>
          <li>Lower-resolution webcam capture for smoother streaming</li>
        </ul>
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>📂 Repository Structure</h2>
    <pre class="repo-structure">
Drowsiness-Landmark-Detection/
├── Assets/
│   ├── Logo.png
│   └── alarm.wav            # optional alarm sound
├── drowsiness_detector/
│   ├── app.py               # main Streamlit application
│   ├── core.py              # model loading & processing utilities
│   └── main.py              # script for training and evaluation
├── Data/
    ├── drowsiness-prediction-dataset/
        ├── 0 FaceImages/
            ├── Active Subjects
            └── Fatigue Subjects
    └── landmarks/
        ├── Active Subjects
        └── Fatigue Subjects
├── Models                
└── requirements.txt         # Python dependencies
    </pre>
  </div>

  <div class="section">
    <h2>🖥️ Setup & Commands</h2>
    <ol>
      <li><strong>Clone the repository</strong>
<pre><code>git clone https://github.com/Garv98/Drowsiness-Landmark-Detection.git
cd Drowsiness-Landmark-Detection</code></pre>
      </li>
      <li><strong>Create & activate a virtual environment</strong> (recommended)
 <pre><code>python -m venv venv
./venv/Scripts/Activate.ps1        # Windows</code></pre>
      </li>
      <li><strong>Install dependencies</strong>
<pre><code>pip install -r requirements.txt</code></pre>
      </li>
      <li><strong>Configure Kaggle API (optional)</strong>
<pre><code># macOS/Linux:
export KAGGLE_API=&lt;your_kaggle_key&gt;
# Windows:
set KAGGLE_API=&lt;your_kaggle_key&gt;</code></pre>
      </li>
      <li><strong>Download datasets</strong>
<pre><code>python drowsiness_detector/main.py --setup --download_datasets</code></pre>
      </li>
      <li><strong>Preprocess & train model</strong>
<pre><code>python drowsiness_detector/main.py --train_model "Data/drowsiness-prediction-dataset/0 FaceImages" --preprocess_data --epochs 5</code></pre>
        <p>If already preprocessed:</p>
<pre><code>python drowsiness_detector/main.py --train_model Data/landmarks --epochs 5</code></pre>
      </li>
      <li><strong>Launch the Streamlit app</strong>
<pre><code>streamlit run drowsiness_detector/app.py</code></pre>
      </li>
    </ol>
  </div>



  <div class="section">
    <h2>🌐 Live Demo (Optional)</h2>
    <p>To deploy on Streamlit Cloud:</p>
    <ol>
      <li>Push your code to GitHub.</li>
      <li>Go to <a href="https://streamlit.io/cloud">Streamlit Cloud</a> and sign in.</li>
      <li>Click "New app", connect your repo, and select <code>drowsiness_detector/app.py</code> as the entry point.</li>
      <li>Hit "Deploy"!</li>
    </ol>
  </div>
