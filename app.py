"""Flask application serving the hand digit image recognition web UI."""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Predict the digit in the uploaded image and render the result page."""
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))

    # Any failure (missing file, unreadable image, ...) must show a friendly error.
    except Exception:  # pylint: disable=broad-exception-caught
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
