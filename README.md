FlowSense 🚦

Intelligent Traffic Command Center
A Neuro-Symbolic Approach to Proactive Traffic Management

👥 The Team

Aman Yadav (Team Leader)

Rahul

Aryan Chaudhary

🛑 The Problem: Event-Driven Congestion

Urban traffic networks are highly sensitive to disruptions. Whether it's a planned event (political rallies, festivals, construction) or an unplanned incident (vehicle breakdowns, sudden gatherings), they create severe, localized traffic breakdowns.

Why It’s Hard Today:

Unquantified Impact: Event impact is not quantified in advance.

Experience-Driven: Resource deployment relies on human intuition rather than data.

No Post-Event Learning: Cities repeat mistakes because there is no automated system to learn from past responses.

💡 Our Solution: A 4-Layer Neuro-Symbolic Hybrid AI

FlowSense solves this by forecasting event-related traffic impact and recommending optimal manpower, barricading, and diversion plans instantly.

Layer 1: NLP Classification (Neuro)
Extracts root causes from raw, unstructured dispatch text using a tuned TF-IDF Vectorizer and Random Forest Classifier. Maps vague reports (e.g., "crater in road") to strict categories (e.g., "Pothole").

Layer 2: Expert Logic Engine (Symbolic)
Hard-coded city-planning rules override AI bias and ensure logical outcomes. (e.g., Ensuring only broken-down buses/trucks trigger high priority, while downgrading small cars).

Layer 3: Impact Forecasting
Predicts incident clearance time based on historical traffic patterns, using an Outlier Capping Algorithm (capping predictions at 3x the historical median) to handle extreme anomalies.

Layer 4: Prescriptive Dashboard
Outputs an exact tactical deployment plan: Manpower limits, specific hardware/barricading needs, and upstream diversion strategies.

📊 The Data Foundation

Grounded in reality, the FlowSense engine is trained directly on the provided Astram Dataset, encompassing 8,173 real traffic events across 54 police stations.

🛠️ Tech Stack & Files

Backend (app.py): Flask Server, Pandas, NumPy, Scikit-Learn (RandomForest, TF-IDF).

Frontend (index.html): HTML, Tailwind CSS, FontAwesome.

Dataset: Astram event data_anonymized.csv

🚀 How to Run the Project Locally

Install Dependencies:
Ensure you have Python installed, then run:

pip install flask pandas numpy scikit-learn


Run the Server:

python app.py


Access the Dashboard:
Open your web browser and go to: http://localhost:8080
