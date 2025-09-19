import tensorflow as tf
import numpy as np
import io

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("./../model/model.h5")
class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_img_bytes(image):
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img = img.resize((32, 32))
    arr = np.array(img).astype("float32") / 255.
    return np.expand_dims(arr, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        file = request.files["file"].read()
    else:
        file = request.get_data()
    
    try:
        preds = model.predict(preprocess_img_bytes(file))
        class_id = int(tf.argmax(preds, axis=-1)[0])
        confidence = float(np.max(preds, axis=-1)[0])
        return jsonify({
            "class_id": class_id,
            "class_name":class_name[class_id],
            "confidence": confidence
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)