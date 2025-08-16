from fastapi import FastAPI, Query
from recommender import FlatZRecommender

app = FastAPI(title="FlatZ Recommendation API")

# ----------------- Initialize recommender -----------------
# This will precompute embeddings and matrices once at startup
recommender = FlatZRecommender(db_path="sqlite:///flatz.db")

@app.get("/")
async def root():
    return {"message": "FlatZ Recommendation API is running"}

@app.get("/v1/reco/homefeed")
async def homefeed(
    user_id: int = Query(..., description="User ID"),
    top_n: int = Query(5, description="Number of recommendations")
):
    recs = recommender.recommend_homefeed(user_id, top_n)
    return {"user_id": user_id, "recommendations": recs}

@app.post("/v1/reco/feedback")
async def feedback(
    user_id: int = Query(..., description="User ID"),
    item_id: int = Query(..., description="Item ID"),
    feedback_type: str = Query("like", description="Feedback type: view/like/save/contact")
):
    result = recommender.record_feedback(user_id, item_id, feedback_type)
    return {"status": result["status"], "user_id": user_id, "item_id": item_id, "feedback_type": feedback_type}

@app.get("/v1/reco/explanations")
async def explanations(
    user_id: int = Query(..., description="User ID"),
    top_n: int = Query(5, description="Number of explanations")
):
    recs_with_reason = recommender.get_explanations(user_id, top_n)
    return {"user_id": user_id, "explanations": recs_with_reason}
