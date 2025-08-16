import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util

class FlatZRecommender:
    def __init__(self, db_path="sqlite:///flatz.db"):
        self.engine = create_engine(db_path)

        # Load tables
        self.users = pd.read_sql("SELECT * FROM users", self.engine)
        self.items = pd.read_sql("SELECT * FROM items", self.engine)
        self.interactions = pd.read_sql("SELECT * FROM interactions", self.engine)

        # Preprocess interactions
        self.interactions['interaction_type'] = self.interactions['interaction_type'].fillna('view')
        self.interactions['timestamp'] = pd.to_datetime(self.interactions['timestamp'], errors='coerce')
        self.interactions['interaction_weight'] = self.interactions['interaction_type'].map({
            'view': 1, 'like': 2, 'save': 3, 'contact': 4
        }).fillna(1)

        # User-item matrix
        self.user_item_matrix = self.interactions.pivot_table(
            index='user_id', columns='item_id', values='interaction_weight', fill_value=0
        )

        # Collaborative filtering: user-user similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(self.user_similarity, 0)
        self.predicted_ratings = np.dot(self.user_similarity, self.user_item_matrix)

        # Content-based: TF-IDF
        self.items['content'] = self.items['title'].astype(str) + " " + self.items['amenities'].astype(str)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.item_tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.items['content'])
        self.content_similarity = cosine_similarity(self.item_tfidf_matrix)

        # Content-based: SentenceTransformer embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.items['embedding'] = self.items['content'].apply(lambda x: self.model.encode(x, convert_to_tensor=True))

        # Popularity
        self.popularity = self.interactions.groupby('item_id')['interaction_weight'].sum()

        # Recent popularity (last 30 days)
        recent_threshold = datetime.now() - timedelta(days=30)
        recent_interactions = self.interactions[self.interactions['timestamp'] >= recent_threshold]
        self.recent_popularity = recent_interactions.groupby('item_id')['interaction_weight'].sum()

    # ----------------- Homefeed -----------------
    def recommend_homefeed(self, user_id, top_n=5):
        if user_id not in self.user_item_matrix.index:
            return self.recommend_cold_start(top_n)

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        cf_scores = self.predicted_ratings[user_idx]

        # Content-based: TF-IDF
        liked_items = self.user_item_matrix.loc[user_id]
        liked_item_ids = liked_items[liked_items > 0].index
        if len(liked_item_ids) > 0:
            liked_indices = [self.items.index[self.items['item_id'] == iid][0] for iid in liked_item_ids]
            content_scores = self.content_similarity[:, liked_indices].mean(axis=1)
        else:
            content_scores = np.zeros(len(self.items))

        # SentenceTransformer embeddings (fixed, no stack_embeddings)
        if len(liked_item_ids) > 0:
            liked_embeddings = self.items[self.items['item_id'].isin(liked_item_ids)]['embedding'].tolist()
            embedding_scores = np.array([
                np.mean([util.cos_sim(row, e).item() for e in liked_embeddings]) if liked_embeddings else 0
                for row in self.items['embedding']
            ])
        else:
            embedding_scores = np.zeros(len(self.items))

        # Popularity normalization
        popularity_scores = self.popularity.reindex(self.items['item_id'], fill_value=0).values
        popularity_scores = popularity_scores / (popularity_scores.max() + 1e-6)
        recent_scores = self.recent_popularity.reindex(self.items['item_id'], fill_value=0).values
        recent_scores = recent_scores / (recent_scores.max() + 1e-6)

        # Hybrid final score
        final_scores = (
            0.35*cf_scores + 
            0.25*content_scores + 
            0.15*embedding_scores + 
            0.15*popularity_scores + 
            0.10*recent_scores
        )

        # Exclude already interacted items
        interacted_mask = self.user_item_matrix.loc[user_id] > 0
        final_scores[interacted_mask.values] = -1

        # Top-N selection
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        recommended_items = self.items.iloc[top_indices].copy()
        recommended_items['score'] = final_scores[top_indices]
        recommended_items['rank'] = range(1, len(top_indices)+1)

        # Community filtering
        if 'community' in self.users.columns and 'community' in recommended_items.columns:
            user_community = self.users.loc[self.users['user_id']==user_id, 'community'].values[0]
            temp = recommended_items[recommended_items['community'] == user_community]
            if not temp.empty:
                recommended_items = temp

        # Low-quality filtering
        if 'description' in recommended_items.columns:
            recommended_items = recommended_items[recommended_items['description'].notna()]
        if 'popularity' in recommended_items.columns:
            recommended_items = recommended_items[recommended_items['popularity'] > 0]

        return recommended_items[['item_id', 'title', 'score', 'rank']].to_dict(orient='records')

    # ----------------- Cold start -----------------
    def recommend_cold_start(self, top_n=5):
        score = 0.7*self.popularity.reindex(self.items['item_id'], fill_value=0).values + \
                0.3*self.recent_popularity.reindex(self.items['item_id'], fill_value=0).values
        top_indices = np.argsort(score)[::-1][:top_n]
        recommended_items = self.items.iloc[top_indices].copy()
        recommended_items['score'] = score[top_indices]
        recommended_items['rank'] = range(1, len(top_indices)+1)
        return recommended_items[['item_id', 'title', 'score', 'rank']].to_dict(orient='records')

    # ----------------- Feedback -----------------
    def record_feedback(self, user_id, item_id, feedback_type='like'):
        insert_query = text("""
            INSERT INTO interactions (user_id, item_id, interaction_type, timestamp)
            VALUES (:user_id, :item_id, :interaction_type, :ts)
        """)
        with self.engine.begin() as conn:
            conn.execute(insert_query, {
                "user_id": user_id,
                "item_id": item_id,
                "interaction_type": feedback_type,
                "ts": datetime.now()
            })

        # Update local matrix
        weight_map = {'view':1,'like':2,'save':3,'contact':4}
        value = weight_map.get(feedback_type,1)
        if item_id not in self.user_item_matrix.columns:
            self.user_item_matrix[item_id] = 0
            self.predicted_ratings = np.dot(self.user_similarity, self.user_item_matrix)
        if user_id not in self.user_item_matrix.index:
            self.user_item_matrix.loc[user_id] = 0
            self.predicted_ratings = np.dot(self.user_similarity, self.user_item_matrix)
        self.user_item_matrix.at[user_id, item_id] = value
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        self.predicted_ratings[user_idx] = np.dot(self.user_similarity[user_idx], self.user_item_matrix)
        return {"status": "success"}

    # ----------------- Explanations -----------------
    def get_explanations(self, user_id, top_n=5):
        recs = self.recommend_homefeed(user_id, top_n)
        explanations = []
        for item in recs:
            explanations.append({
                "item_id": item['item_id'],
                "title": item['title'],
                "rank": item['rank'],
                "reason": "Hybrid score: CF + TF-IDF + Embedding + Popularity + Recent interactions"
            })
        return explanations

    # ----------------- Save top5 for all users -----------------
    def save_top5_for_all_users(self, top_n=5, file_path=r"D:\flatZ_data\top5_recommendations.csv"):
        all_recs = []
        for uid in self.user_item_matrix.index:
            recs = self.recommend_homefeed(uid, top_n)
            for item in recs:
                all_recs.append({
                    'user_id': uid,
                    'item_id': item['item_id'],
                    'rank': item['rank'],
                    'score': item['score']
                })
        df = pd.DataFrame(all_recs)
        df.to_csv(file_path, index=False)
        print(f"✅ Top-{top_n} recommendations saved to {file_path}")
        return df
