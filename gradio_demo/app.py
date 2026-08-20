"""Standalone Gradio demo for the fraud detection model.

Loads a locally-bundled model (exported once from the MLflow registry) —
no dependency on a live MLflow server or the production API, so this can
be deployed independently (e.g. Hugging Face Spaces).
"""
from pathlib import Path

import gradio as gr
import mlflow
import pandas as pd  # type: ignore
import spaces

MODEL_PATH = Path(__file__).parent / "model"
model = mlflow.sklearn.load_model(str(MODEL_PATH))

COUNTRY_CODES = [
    "AR",
    "AU",
    "BR",
    "CA",
    "CH",
    "CL",
    "CO",
    "ES",
    "FR",
    "GB",
    "GT",
    "IT",
    "KR",
    "MX",
    "PT",
    "TR",
    "UA",
    "US",
    "UY",
]

@spaces.GPU
def predict(a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, s, monto, q, r):
    """Predict fraud probability for a transaction.

    Args:
        a, b, c, d, e, f, g, h, i: Numeric features.
        j: Country code (string).
        l, m, n, o, p: Numeric features.
        s: Numeric feature (range: -1 to 99.97).
        monto: Transaction amount.
        q, r: Numeric features.

    Returns:
        str: Prediction result with fraud probability and emoji indicator.
    """
    row = pd.DataFrame(
        [
            {
                "A": a,
                "B": b,
                "C": c,
                "D": d,
                "E": e,
                "F": f,
                "G": g,
                "H": h,
                "I": i,
                "J": j,
                "L": l,
                "M": m,
                "N": n,
                "O": o,
                "P": p,
                "S": s,
                "Monto": monto,
                "Q": q,
                "R": r,
            }
        ]
    )
    row["C_is_null"] = int(pd.isna(c))

    proba = model.predict_proba(row)[0, 1]
    label = "🚨 Likely fraud" if proba >= 0.5 else "✅ Likely legitimate"
    return f"{label}  —  fraud probability: {proba:.1%}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="A - Range seen on training: 0 - 30"),
        gr.Number(label="B - Range seen on training: -1 - 20"),
        gr.Number(label="C - Range seen on training: 0 - 617324. Could be null."),
        gr.Number(label="D - Range seen on training: 0 - 180"),
        gr.Number(label="E - Range seen on training: 0 - 45"),
        gr.Number(label="F - Range seen on training: 0 - 1"),
        gr.Number(label="G - Range seen on training: 0 - 1"),
        gr.Number(label="H - Range seen on training: 0 - 21"),
        gr.Number(label="I - Range seen on training: 0 - 24"),
        gr.Dropdown(COUNTRY_CODES, label="J (country code)"),
        gr.Number(label="L - Range seen on training: 0 - 7"),
        gr.Number(label="M - Range seen on training: 1 - 13"),
        gr.Number(label="N - Range seen on training: 1 - 10"),
        gr.Number(label="O - Range seen on training: 0 - 3"),
        gr.Number(label="P - Range seen on training: 1 - 41"),
        gr.Number(label="S - Range seen on training: -1 - 99.97"),
        gr.Number(label="Monto - Range seen on training: 0.05 - 998.11"),
        gr.Number(label="Q - Range seen on training: 0 - 984.42"),
        gr.Number(label="R - Range seen on training: 0 - 984.44"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fraud Detection Demo",
    description=(
        "LightGBM model trained on anonymized transaction features. "
        "All value range shown are for guidance, input value is not strictly in that range. "
        "See the full project (training, MLflow tracking, API, Docker, CI/CD) "
        "on GitHub: https://github.com/pechandiat/fraud_det"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
