import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from app.model import LiveRiskPipeline

app = FastAPI(title="CARLA Road Safety Risk Service Gateway")

# Separate pipeline instances so the HTTP endpoint's frame counter never
# collides with a concurrent WebSocket session's counter.
http_pipeline = LiveRiskPipeline(model_path="yolov8n.pt", config_path="best_xgboost_model.json")
ws_pipeline   = LiveRiskPipeline(model_path="yolov8n.pt", config_path="best_xgboost_model.json")


@app.post("/predict/frame")
async def predict_frame(ego_speed_ms: float = 0.0, scenario: str = "normal", file: UploadFile = File(...)):
    """Receives a single image snapshot array and returns the predicted risk classification level."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame image data payload."}

    # /predict/frame serves independent, unrelated single-image requests
    # (e.g. one-off uploads in the dashboard's "Image" tab). Reset the
    # rolling smoothing/hysteresis state first so a previous, unrelated
    # image's history can never leak into this prediction.
    http_pipeline.reset_history()
    prediction = http_pipeline.predict_risk(frame, ego_speed_ms, scenario)
    return {"filename": file.filename, "prediction": prediction}


@app.websocket("/live/carla")
async def live_carla_stream(websocket: WebSocket, scenario: str = "normal"):
    """
    Accepts real-time live video streams from a CARLA simulation client node,
    tracking historical velocity variations over the shared session.
    """
    await websocket.accept()
    ws_pipeline.reset_history()  # Flush rolling metrics on fresh connection setup
    print("New CARLA client stream connection initiated.")

    try:
        while True:
            # Receive JSON containing control parameters and image data.
            # Supports both base64-encoded strings (preferred, efficient) and
            # raw uint8 lists (legacy) for the 'image_bytes' field.
            data = await websocket.receive_json()

            ego_speed = float(data.get("ego_speed_ms", 0.0))
            frame_scenario = data.get("scenario", scenario) 

            raw_field = data["image_bytes"]
            if isinstance(raw_field, str):
                # Base64-encoded JPEG (preferred path — ~4× less data over wire)
                img_bytes = base64.b64decode(raw_field)
            else:
                # Legacy: raw uint8 list
                img_bytes = bytes(raw_field)

            raw_img = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)

            if frame is not None:
                prediction = ws_pipeline.predict_risk(frame, ego_speed, frame_scenario)
                await websocket.send_json({
                    "status": "processing",
                    "prediction": prediction
                })
    except WebSocketDisconnect:
        print("CARLA simulation client dropped connection stream.")
    finally:
        ws_pipeline.reset_history()