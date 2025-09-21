from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
import numpy as np

app = Flask(__name__)

# Load the model (includes preprocessing)
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['input_text']

        # If your model includes a TextVectorization layer, you can directly pass text
        pred_prob = model.predict([text])[0][0]  # [text] is a batch of 1

        # Binary sentiment: 0=negative, 1=positive
        prediction = "Positive" if pred_prob > 0.5 else "Negative"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
