import base64
import io
import os

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)


# ============================================================
# Configuration
# ============================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "ridwanFatur98/invoice-doc-to-text",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Invoice Document OCR API",
    version="1.0.0",
)


# ============================================================
# Load model
# ============================================================

config = VisionEncoderDecoderConfig.from_pretrained(
    MODEL_ID
)

processor = DonutProcessor.from_pretrained(
    MODEL_ID
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_ID,
    config=config,
)

model.to(DEVICE)
model.eval()


# ============================================================
# Vertex AI request schema
# ============================================================

class Instance(BaseModel):
    image: str


class PredictRequest(BaseModel):
    instances: list[Instance]


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": str(DEVICE),
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: PredictRequest):

    predictions = []

    try:
        for instance in request.instances:

            # ------------------------------------------------
            # Decode base64 image
            # ------------------------------------------------

            image_bytes = base64.b64decode(
                instance.image
            )

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            pixel_values = processor(
                image,
                return_tensors="pt",
            ).pixel_values.to(DEVICE)

            # ------------------------------------------------
            # Decoder input
            # ------------------------------------------------

            task_start_token = "<parsing>"

            decoder_input_ids = processor.tokenizer(
                task_start_token,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(DEVICE)

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            with torch.inference_mode():

                generated_ids = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=300,
                    bad_words_ids=[
                        [processor.tokenizer.unk_token_id]
                    ],
                )

            # ------------------------------------------------
            # Decode output
            # ------------------------------------------------

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]

            output = processor.token2json(
                generated_text
            )

            predictions.append({
                "output": output,
                "generated_text": generated_text,
            })

        # ----------------------------------------------------
        # Vertex AI response format
        # ----------------------------------------------------

        return {
            "predictions": predictions
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )