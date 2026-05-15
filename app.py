"""
Student Placement Predictor — Flask Web App
Run: python app.py  →  open http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import pickle, numpy as np

app = Flask(__name__)

with open('model/placement_model.pkl', 'rb') as f:
    saved = pickle.load(f)

pipeline = saved['pipeline']
le       = saved['label_encoder']
FEATURES = saved['features']

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Placement Predictor</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f5f4fb; color: #222; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .card { background: #fff; border-radius: 16px; padding: 2rem; width: 100%; max-width: 520px; box-shadow: 0 4px 24px rgba(75,0,130,0.10); }
    h1 { font-size: 1.4rem; color: #4B0082; margin-bottom: 4px; }
    p.sub { font-size: 0.85rem; color: #888; margin-bottom: 1.5rem; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field.full { grid-column: 1 / -1; }
    label { font-size: 12px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
    input, select { border: 1px solid #ddd; border-radius: 8px; padding: 9px 12px; font-size: 14px; width: 100%; outline: none; transition: border 0.15s; }
    input:focus, select:focus { border-color: #4B0082; }
    button { margin-top: 1.2rem; width: 100%; background: #4B0082; color: #fff; border: none; border-radius: 10px; padding: 12px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
    button:hover { background: #3a006b; }
    .result { margin-top: 1.2rem; border-radius: 10px; padding: 1rem 1.2rem; display: none; }
    .placed   { background: #e8f5e9; border: 1px solid #81c784; color: #2e7d32; }
    .notplaced{ background: #fce4ec; border: 1px solid #e57373; color: #c62828; }
    .result-title { font-size: 1.1rem; font-weight: 700; }
    .result-prob  { font-size: 0.88rem; margin-top: 4px; }
    .bar-wrap { margin-top: 10px; background: #eee; border-radius: 20px; height: 8px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 20px; transition: width 0.6s ease; }
    .tips { margin-top: 10px; font-size: 0.82rem; line-height: 1.6; }
  </style>
</head>
<body>
<div class="card">
  <h1>🎓 Placement Predictor</h1>
  <p class="sub">Enter your profile to predict placement probability</p>

  <div class="form-grid">
    <div class="field">
      <label>CGPA (4.0 – 10.0)</label>
      <input type="number" id="cgpa" min="4" max="10" step="0.01" value="7.5">
    </div>
    <div class="field">
      <label>Aptitude Score (%)</label>
      <input type="number" id="aptitude" min="0" max="100" value="70">
    </div>
    <div class="field">
      <label>Internships</label>
      <select id="internships">
        <option value="0">0</option>
        <option value="1" selected>1</option>
        <option value="2">2</option>
        <option value="3">3+</option>
      </select>
    </div>
    <div class="field">
      <label>Projects</label>
      <select id="projects">
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2" selected>2</option>
        <option value="3">3</option>
        <option value="4">4+</option>
      </select>
    </div>
    <div class="field">
      <label>Tech Skills (count)</label>
      <input type="number" id="skills" min="1" max="20" value="6">
    </div>
    <div class="field">
      <label>Active Backlogs</label>
      <select id="backlogs">
        <option value="0" selected>0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3">3+</option>
      </select>
    </div>
    <div class="field">
      <label>Communication (1–5)</label>
      <input type="range" id="communication" min="1" max="5" value="3" oninput="document.getElementById('comm_val').textContent=this.value">
      <span id="comm_val" style="font-size:13px;color:#4B0082;font-weight:600;">3</span>
    </div>
    <div class="field">
      <label>Branch</label>
      <select id="branch">
        <option value="CS">CS</option>
        <option value="IT">IT</option>
        <option value="ECE">ECE</option>
        <option value="Mech">Mech</option>
        <option value="Civil">Civil</option>
      </select>
    </div>
  </div>

  <button onclick="predict()">Predict My Chances</button>

  <div class="result" id="result">
    <div class="result-title" id="result-title"></div>
    <div class="result-prob" id="result-prob"></div>
    <div class="bar-wrap"><div class="bar-fill" id="bar"></div></div>
    <div class="tips" id="tips"></div>
  </div>
</div>

<script>
async function predict() {
  const data = {
    cgpa: parseFloat(document.getElementById('cgpa').value),
    internships: parseInt(document.getElementById('internships').value),
    projects: parseInt(document.getElementById('projects').value),
    skills: parseInt(document.getElementById('skills').value),
    backlogs: parseInt(document.getElementById('backlogs').value),
    communication: parseInt(document.getElementById('communication').value),
    aptitude_score: parseFloat(document.getElementById('aptitude').value),
    branch: document.getElementById('branch').value
  };

  const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data) });
  const json = await res.json();

  const box  = document.getElementById('result');
  const prob = Math.round(json.probability * 100);
  box.style.display = 'block';
  box.className = 'result ' + (json.placed ? 'placed' : 'notplaced');
  document.getElementById('result-title').textContent = json.placed ? '✅ Likely to be Placed!' : '⚠️ Placement Risk Detected';
  document.getElementById('result-prob').textContent  = `Placement probability: ${prob}%`;
  document.getElementById('bar').style.width = prob + '%';
  document.getElementById('bar').style.background = json.placed ? '#4caf50' : '#e57373';
  document.getElementById('tips').innerHTML = json.tips.map(t => `• ${t}`).join('<br>');
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    branch_enc = le.transform([d['branch']])[0]
    X = np.array([[
        d['cgpa'], d['internships'], d['projects'], d['skills'],
        d['backlogs'], d['communication'], d['aptitude_score'], branch_enc
    ]])
    prob   = float(pipeline.predict_proba(X)[0][1])
    placed = prob >= 0.5

    tips = []
    if d['cgpa'] < 7.0:     tips.append("Improve CGPA — aim for 7.5+ for most companies.")
    if d['internships'] == 0:tips.append("Do at least 1 internship — it's the top placement factor.")
    if d['backlogs'] > 0:   tips.append("Clear all backlogs — many companies have a strict 0-backlog policy.")
    if d['skills'] < 5:     tips.append("Add more tech skills: Python, SQL, or a framework relevant to your branch.")
    if d['communication'] < 3: tips.append("Work on communication — join a debate club or practice group discussions.")
    if d['aptitude_score'] < 60: tips.append("Practice aptitude tests on platforms like IndiaBix or PrepInsta.")
    if not tips:             tips.append("Great profile! Keep networking and apply early.")

    return jsonify({'placed': placed, 'probability': round(prob, 4), 'tips': tips})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
