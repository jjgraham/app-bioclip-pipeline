import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from bioclip.predict import TreeOfLifeClassifier
from bioclip import Rank
from pywaggle import Waggle  # Standard Sage data publishing library

app = Flask(__name__)

# --- 1. Initialize Models ---
print("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8x.pt")  # High-accuracy YOLO for 4K detection bounding boxes

print("Loading SAM (Segment Anything Model)...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
sam_predictor = SamPredictor(sam)

print("Loading BioCLIP...")
classifier = TreeOfLifeClassifier()

# --- 2. Configure PyBioCLIP Plausible Taxa Subset ---
# Define the subset of taxa relevant to the H02A site (e.g., California Vineyard ecosystem)
PLAUSIBLE_TAXA = {
    Rank.CLASS: ["Aves", "Mammalia", "Insecta"],
    Rank.ORDER: ["Squamata", "Rodentia", "Passeriformes", "Lepidoptera"],
    Rank.FAMILY: ["Cervidae", "Canidae", "Leporidae", "Vitaceae"]
}

print("Generating BioCLIP Tree-of-Life taxonomic subset filters...")
# Combine filters into a single boolean mask to restrict the Tree of Life embeddings
combined_filter = None
for rank, names in PLAUSIBLE_TAXA.items():
    for name in names:
        try:
            # Generate the filter for specific valid TOL values
            taxa_mask = classifier.create_taxa_filter(rank=rank, value=name)
            if combined_filter is None:
                combined_filter = taxa_mask
            else:
                combined_filter = np.logical_or(combined_filter, taxa_mask)
        except ValueError:
            print(f"Warning: Taxon '{name}' at rank '{rank}' not found in TOL labels.")

if combined_filter is not None:
    # Limit model predictions exclusively to this subset array
    classifier.apply_filter(combined_filter)
    print("Successfully restricted BioCLIP predictions to plausible local taxa subset.")

# --- 3. Webhook Endpoint for UVC-AI-Pro ---
@app.route("/webhook/smart-detection", methods=["POST"])
def handle_detection():
    # Handle incoming multi-part image or JSON payload from the camera
    if "image" not in request.files:
        return jsonify({"error": "No image data found in webhook payload"}), 400
        
    file = request.files["image"]
    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 4K base image matrix
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step A: YOLOv8 Bounding Box Generation
    yolo_results = yolo_model(img)
    
    detections = []
    for result in yolo_results:
        for box in result.boxes:
            # Extract box coordinates (xyxy)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            
            # Step B: SAM High-Precision Segment Masking
            sam_predictor.set_image(img_rgb)
            masks, scores, _ = sam_predictor.predict(
                box=xyxy,
                multimask_output=False
            )
            mask = masks[0]
            
            # Isolate the detected object using the SAM mask mask
            isolated_obj = img_rgb.copy()
            isolated_obj[~mask] = 0 # Black out the background
            
            # Crop to bounding box to reduce BioCLIP frame canvas size
            x1, y1, x2, y2 = xyxy
            cropped_obj = isolated_obj[y1:y2, x1:x2]
            
            # Save crop temporarily or pass directly as a temporary image file
            temp_path = "/tmp/crop.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(cropped_obj, cv2.COLOR_RGB2BGR))

            # Step C: Subset-Constrained BioCLIP Classification
            # Predicts over the restricted Tree of Life hierarchy defined above
            predictions = classifier.predict_image(temp_path)
            
            # Parse top BioCLIP predictions
            top_prediction = predictions[0] if predictions else {}
            
            detections.append({
                "bbox": xyxy.tolist(),
                "yolo_conf": conf,
                "taxon": top_prediction.get("species", top_prediction.get("genus", "Unknown")),
                "common_name": top_prediction.get("common_name", "Unknown"),
                "bioclip_score": float(top_prediction.get("score", 0.0))
            })
            
            # --- 4. Publish metrics to Sage Continuum Node Stream ---
            with Waggle() as waggle:
                # Log the prediction details out to the time-series stream
                waggle.plugin.publish("env.bioclip.detection", top_prediction.get("species", "Unknown"), meta={
                    "common_name": top_prediction.get("common_name", "Unknown"),
                    "confidence": str(top_prediction.get("score", 0.0))
                })

    return jsonify({"status": "success", "processed_detections": detections}), 200

if __name__ == "__main__":
    # Internal edge app deployment port
    app.run(host="0.0.0.0", port=5000)
