import os
import base64
from io import BytesIO
from flask import Flask, request, render_template
from config import Config
from utils.predictor import predict_mushroom_from_stream

app = Flask(__name__)
app.config.from_object(Config)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file and allowed_file(file.filename):
            try:
                file_content = file.read()
                image_stream = BytesIO(file_content)

                predictions = predict_mushroom_from_stream(image_stream)

                encoded_img = base64.b64encode(file_content).decode("utf-8")
                ext = str(file.filename).rsplit(".", 1)[1].lower()
                mime = "png" if ext == "png" else "jpeg"
                image_data = f"data:image/{mime};base64,{encoded_img}"

                return render_template(
                    "index.html",
                    filename=file.filename,
                    predictions=predictions,
                    image_data=image_data,
                )
            except Exception as e:
                return render_template(
                    "index.html", error=f"Prediction error: {str(e)}"
                )
        else:
            return render_template("index.html", error="File type not allowed")

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
