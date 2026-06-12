"""
app.py
------
Flask backend for the Student Startup Success Predictor.

The model is trained ONCE via train_model.py and loaded at startup here.
Users submit their business data and receive:
  - Their success chance as a percentage
  - The top 3 factors most affecting that result
  - Data saved automatically to MySQL database
"""

import os
import numpy as np
import pandas as pd
import joblib
import pymysql
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Friendly display names (shown to users on the web-app) ───────────────────
FEATURE_DISPLAY_NAMES = {
    'team_size'                     : 'Team Size',
    'avg_team_experience'           : 'Average Team Experience (Years)',
    'innovation_score'              : 'Innovation Score',
    'market_readiness_level'        : 'Market Readiness Level',
    'competition_awards'            : 'Number of Competition Awards',
    'business_model_score'          : 'Business Model Score',
    'technology_maturity'           : 'Technology Maturity Level',
    'ecosystem_support'             : 'Mentorship & Incubation Support',
    'startup_age'                   : 'Startup Age (Years)',
    'funding_amount_usd_log'        : 'Funding Amount (USD)',
    'institution_type_Private'      : 'Institution Type: Private',
    'institution_type_Public'       : 'Institution Type: Public',
    'institution_type_Technical'    : 'Institution Type: Technical',
    'project_domain_EdTech'         : 'Project Domain: EdTech',
    'project_domain_FinTech'        : 'Project Domain: FinTech',
    'project_domain_GreenTech'      : 'Project Domain: GreenTech',
    'project_domain_HealthTech'     : 'Project Domain: HealthTech',
}

def get_display_name(col: str) -> str:
    return FEATURE_DISPLAY_NAMES.get(col, col.replace('_', ' ').title())


# ── Database Connection Configuration ───────────────────────────────────────
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',             # استبدل باسم المستخدم الخاص بك في قاعدة البيانات
        password='',             # استبدل بكلمة المرور الخاصة بك
        db='startup_db', # استبدل باسم قاعدة البيانات لديك
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


# ── Load model at startup (trained once via train_model.py) ──────────────────
MODEL_PATH   = 'model.pkl'
SCALER_PATH  = 'scaler.pkl'
COLUMNS_PATH = 'feature_columns.pkl'

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
    raise RuntimeError(
        "Model files not found. Please run `python train_model.py` first."
    )

model           = joblib.load(MODEL_PATH)
scaler          = joblib.load(SCALER_PATH)
feature_columns = joblib.load(COLUMNS_PATH)

print("Model loaded successfully.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a user's business data and returns their success chance + top 3 factors,
    and stores inputs and predictions into MySQL.

    Expected JSON body:
    {
        "project_id": 1,
        "team_size": 4,
        "avg_team_experience": 2.5,
        "innovation_score": 0.75,
        "funding_amount_usd": 50000,
        "mentorship_support": 1,
        "incubation_support": 0,
        "market_readiness_level": 3,
        "competition_awards": 1,
        "business_model_score": 0.8,
        "technology_maturity": 3,
        "year": 2022,
        "institution_type": "Public",
        "project_domain": "FinTech"
    }
    """
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'error': 'No data provided.'}), 400

    project_id = payload.get('project_id')
    if not project_id:
        return jsonify({'error': 'Missing project_id in request payload.'}), 400

    try:
        df = pd.DataFrame([payload])

        # ── Apply the exact same feature engineering as train_model.py ──────
        df['ecosystem_support'] = df['mentorship_support'] + df['incubation_support']
        df = df.drop(columns=['mentorship_support', 'incubation_support'])

        df['startup_age'] = 2026 - df['year']
        df = df.drop(columns=['year'])

        df['funding_amount_usd_log'] = np.log1p(df['funding_amount_usd'])
        df = df.drop(columns=['funding_amount_usd'])

        df = pd.get_dummies(df, columns=['institution_type', 'project_domain'], drop_first=True)

        # Align with training columns — fill any missing dummy columns with 0
        df = df.reindex(columns=feature_columns, fill_value=0)

        # ── Predict ──────────────────────────────────────────────────────────
        X_scaled   = scaler.transform(df)
        raw_prob   = float(model.predict_proba(X_scaled)[0][1])
        # Clip to 5%-95% — the dataset is highly separable so raw probabilities
        # push to 0% or 100% for most inputs. Clipping keeps output human-readable
        # without affecting the model's classification accuracy.
        clipped_prob   = float(np.clip(raw_prob, 0.05, 0.95))
        success_chance = round(clipped_prob * 100, 2)

        # ── Top 3 factors specific to this user's input ──────────────────────
        # contribution = coefficient x scaled_feature_value
        # This reflects how much each feature actually pushed the result
        # for this specific user, not just the global model weights
        contributions = model.coef_[0] * X_scaled[0]

        contrib_df = pd.DataFrame({
            'feature'     : feature_columns,
            'contribution': contributions,
            'abs_contrib' : np.abs(contributions)
        }).sort_values('abs_contrib', ascending=False)

        top_3 = []
        for _, row in contrib_df.head(3).iterrows():
            top_3.append({
                'display_name': get_display_name(row['feature']),
                'direction'   : 'Increases success chance' if row['contribution'] > 0 else 'Decreases success chance',
                'strength'    : 'High'   if row['abs_contrib'] > 1.5 else
                                'Medium' if row['abs_contrib'] > 0.7 else 'Low',
                'impact_score': round(float(row['abs_contrib']), 4)
            })

        verdict_text = 'Likely to Succeed' if success_chance >= 50 else 'At Risk of Failure'

        # ── Save to Database ──────────────────
        connection = get_db_connection()
        with connection.cursor() as cursor:
            # 1. Save inputs to project_details table
            sql_details = """
                INSERT INTO project_details (
                    project_id, team_size, avg_team_experience, innovation_score, 
                    market_readiness_level, competition_awards, business_model_score, 
                    technology_maturity, mentorship_support, incubation_support, 
                    funding_amount_usd, institution_type, project_domain, year
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql_details, (
                project_id, payload.get('team_size'), payload.get('avg_team_experience'),
                payload.get('innovation_score'), payload.get('market_readiness_level'),
                payload.get('competition_awards'), payload.get('business_model_score'),
                payload.get('technology_maturity'), payload.get('mentorship_support'),
                payload.get('incubation_support'), payload.get('funding_amount_usd'),
                payload.get('institution_type'), payload.get('project_domain'), payload.get('year')
            ))

            # 2. Save prediction results to prediction_results table
            sql_results = """
                INSERT INTO prediction_results (project_id, success_chance, verdict, top_3_factors)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql_results, (
                project_id, success_chance, verdict_text, json.dumps(top_3)
            ))
            
            connection.commit()
        connection.close()

        return jsonify({
            'success_chance'       : f"{success_chance}%",
            'success_chance_raw'   : success_chance,
            'verdict'              : verdict_text,
            'top_3_success_factors': top_3
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)