# FlatZ Recommendation API

A recommendation service for FlatZ residents, providing personalized homefeeds, recording feedback, and giving explanations.

---

## Tech Stack
- Python 3.10+
- FastAPI
- SQLite
- Pandas, NumPy
- Scikit-learn (CF, TF-IDF)
- Sentence Transformers (content embeddings)
- Uvicorn

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd flatz__backend



### 2. Create & Activate Virtual Environment
python -m venv .venv
# Windows PowerShell
& .\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate


### 3 install dependencies
pip install -r requirements.txt

###4. Prepare Database
Place CSV files in flatZ_data/:

users_realistic.csv
items_realistic.csv
interactions_realistic.csv
top5_recommendations.csv

### 5. Create and load the database:

python create_db.py

### 6.Running the API
uvicorn main:app --reload
Swagger docs: http://127.0.0.1:8000/docs


###7 Api endpoints
Homefeed:
GET /v1/reco/homefeed?user_id=<id>&top_n=5

Feedback:
POST /v1/reco/feedback?user_id=<id>&item_id=<id>&feedback_type=like

Explanations:
GET /v1/reco/explanations?user_id=<id>&top_n=5

## Testing the FlatZ API

# Homefeed
curl "http://127.0.0.1:8000/v1/reco/homefeed?user_id=1&top_n=5"

# Feedback
curl -X POST "http://127.0.0.1:8000/v1/reco/feedback?user_id=1&item_id=2&feedback_type=like"

# Explanations
curl "http://127.0.0.1:8000/v1/reco/explanations?user_id=1&top_n=5"


### PowerShell Examples

```powershell
# ----------------- Homefeed -----------------
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/reco/homefeed?user_id=1&top_n=5"
$response.recommendations | ForEach-Object { "$($_.rank): $($_.title) - Score: $($_.score)" }

# ----------------- Feedback -----------------
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/reco/feedback?user_id=1&item_id=2&feedback_type=like"
$response | Format-Table

# ----------------- Explanations -----------------
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/reco/explanations?user_id=1&top_n=5"
$response.explanations | ForEach-Object { "$($_.rank): $($_.title) - Reason: $($_.reason)" }


### Generate Top-5 Recommendations CSV

from recommender import FlatZRecommender

r = FlatZRecommender(db_path="sqlite:///flatz.db")
r.save_top5_for_all_users(top_n=5, file_path="top5_recommendations.csv")


