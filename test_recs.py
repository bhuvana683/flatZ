#test recs
from recommender import FlatZRecommender

recommender = FlatZRecommender(db_path="sqlite:///flatz.db")
recs = recommender.recommend_homefeed(user_id=1, top_n=5)

for r in recs:
    print(r['title'], r['score'])

