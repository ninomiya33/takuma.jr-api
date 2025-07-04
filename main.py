from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import os
import openai
import numpy as np

# .env 読み込み & APIキー取得
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("✅ OPENAI_API_KEY:", api_key)

# OpenAIクライアント（v1系）
from openai import OpenAI
client = OpenAI()  # api_keyは環境変数から自動取得

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 追加: CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # フロントのURL（本番時は本番URLも追加）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 全国基準値の読み込み
baseline_df = pd.read_csv("national_baseline.csv")
baseline_df["age"] = baseline_df["age"].astype(int)
baseline_df["gender"] = baseline_df["gender"].str.strip()
baseline_df["test"] = baseline_df["test"].str.strip()

print(baseline_df["age"].unique())
print(baseline_df["gender"].unique())
print(baseline_df["test"].unique())

# 入力データモデル
class PhysicalInput(BaseModel):
    age: int
    gender: str
    run50m: float
    shuttle_run: int
    jump: float
    sit_up: int
    sit_and_reach: float

@app.post("/advice")
def generate_advice(data: PhysicalInput):
    try:
        scores = {}
        test_map = {
            "run50m": "50m_run",
            "shuttle_run": "shuttle_run",
            "jump": "jump",
            "sit_up": "sit_up",
            "sit_and_reach": "sit_and_reach"
        }
        for key, test_name in test_map.items():
            base = baseline_df[
                (baseline_df["age"] == data.age) &
                (baseline_df["gender"] == data.gender) &
                (baseline_df["test"] == test_name)
            ]
            print(f"DEBUG: age={data.age}, gender={data.gender}, test={test_name}, hit={len(base)}")
            if base.empty:
                raise HTTPException(status_code=404, detail=f"{test_name} のベースラインデータが見つかりません")
            m = base["average"].values[0]
            s = base["average"].std()
            user_val = getattr(data, key)
            # --- ここを修正 ---
            if s == 0 or np.isnan(s):
                score = 50  # 標準偏差0やNaNなら偏差値50固定
            else:
                score = 50 + 10 * (user_val - m) / s
            scores[key] = round(score, 1)

        # OpenAI にアドバイスを生成させる
        prompt = (
            f"以下は小学生の体力テスト結果です。優しく前向きなアドバイスをしてください。\n\n"
            f"年齢: {data.age}歳\n性別: {data.gender}\n"
            f"50m走: {data.run50m}秒\nシャトルラン: {data.shuttle_run}回\n"
            f"立ち幅跳び: {data.jump}cm\n上体起こし: {data.sit_up}回\n"
            f"長座体前屈: {data.sit_and_reach}cm\n"
            f"\n"
            f"【出力フォーマット例】\n"
            f"---\n"
            f"■ 50m走\nアドバイス本文\n\n"
            f"■ シャトルラン\nアドバイス本文\n\n"
            f"■ 立ち幅跳び\nアドバイス本文\n\n"
            f"■ 上体起こし\nアドバイス本文\n\n"
            f"■ 長座体前屈\nアドバイス本文\n\n"
            f"■ 総合コメント\n全体への応援やまとめ\n"
            f"---\n"
            f"このように、各項目ごとに「■」で始めて改行し、読みやすく出力してください。"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたは小学生にやさしく体力アドバイスするスポーツコーチです。"},
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "scores": scores,
            "advice": response.choices[0].message.content
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# テスト用エンドポイント
@app.get("/test")
def test_endpoint():
    return {"message": "API is working"}

# 例としてのデータ
example_data = {
  "age": 10,
  "gender": "boy",
  "run50m": 9.0,
  "shuttle_run": 50,
  "jump": 150.0,
  "sit_up": 20,
  "sit_and_reach": 35.0
}

@app.get("/example")
def example_endpoint():
    return example_data

@app.post("/growth")
def predict_growth(data: PhysicalInput):
    """
    3ヶ月間のスコア推移（成長予測）と、1.08倍を目指すための目標値を返すAPI
    """
    try:
        # 50m走は小さいほど良いので0.92倍、他は1.08倍
        target = {}
        for k, v in data.dict().items():
            if k in ["age", "gender"]:
                continue
            if k == "run50m":
                target[k] = v * 0.92  # タイムは減らす
            else:
                target[k] = v * 1.08  # 回数や距離は増やす

        # 3ヶ月間の成長を直線的に予測
        months = [1, 2, 3]
        growth = {}
        for key in target.keys():
            start = getattr(data, key)
            end = target[key]
            growth[key] = [round(start + (end - start) * m / 3, 2) for m in months]

        return {
            "start": {k: getattr(data, k) for k in target.keys()},
            "target": {k: round(v, 2) for k, v in target.items()},
            "growth_prediction": growth
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
