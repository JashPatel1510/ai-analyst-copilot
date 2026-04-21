MODEL = None
MODEL_META = None


from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")




#analyze 
from modules.preprocess import load_data, clean_data
from modules.eda import generate_summary, create_plots
from modules.insights import generate_insights
from modules.model import train_model


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)


    df = load_data(filepath)
    df = clean_data(df)


    summary = generate_summary(df)
    plots = create_plots(df)
    insights = generate_insights(df)
    model_result = train_model(df)
    
    
    global MODEL, MODEL_META

    MODEL_META = train_model(df)
    MODEL = MODEL_META["model"]
    
    
    return render_template(
        "dashboard.html",
        summary=summary,
        plots=plots,
        insights=insights,
        model_result=model_result
    )


@app.route("/predict", methods=["POST"])
def predict():
    global MODEL, MODEL_META

    if MODEL is None:
        return "Model not trained yet"

    input_data = request.form.to_dict()

    import pandas as pd

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert numeric values
    for col in input_df.columns:
        try:
            input_df[col] = float(input_df[col])
        except:
            pass

    # One-hot encoding to match training
    input_df = pd.get_dummies(input_df)

    # Align with training features
    for col in MODEL_META["features"]:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[MODEL_META["features"]]

    prediction = MODEL.predict(input_df)[0]

    # Decode label if needed
    if MODEL_META["encoder"] is not None:
        prediction = MODEL_META["encoder"].inverse_transform([int(prediction)])[0]

    return f"Prediction: {prediction}"


if __name__ == "__main__":
    app.run()