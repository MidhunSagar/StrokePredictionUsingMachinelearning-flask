from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and components
stroke_model = joblib.load("model.joblib")

# Category mappings from form (numeric) → model (string)
gender_map = {
    "0": "Female",
    "1": "Male"
}

ever_married_map = {
    "0": "No",
    "1": "Yes"
}

work_type_map = {
    "0": "children",
    "1": "Govt_job",
    "2": "Never_worked",
    "3": "Private",
    "4": "Self-employed"
}

residence_map = {
    "0": "Rural",
    "1": "Urban"
}

smoking_map = {
    "0": "formerly smoked",
    "1": "never smoked",
    "2": "smokes",
    "3": "Unknown"
}


# def predict_input(single_input):
#     input_df = pd.DataFrame([single_input])
#     encoded_cols = stroke_model["encoded_cols"]
#     numeric_cols = stroke_model["numeric_cols"]
#     preprocessor = stroke_model["preprocessor"]

#     # Apply preprocessing (assumes preprocessor expects only encoded_cols)
#     input_df[encoded_cols] = preprocessor.transform(input_df)

#     X = input_df[numeric_cols + encoded_cols]
#     prediction = stroke_model["model"].predict(X)
#     return prediction

# Prediction helper function

# def predict_input(single_input):
#     input_df = pd.DataFrame([single_input])
    
#     # Extract components
#     preprocessor = stroke_model["preprocessor"]
#     model = stroke_model["model"]
#     numeric_cols = stroke_model["numeric_cols"]
#     encoded_cols = stroke_model["encoded_cols"]

#     # Apply preprocessing
#     encoded_data = preprocessor.transform(input_df)
#     encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

#     # Combine numerical + encoded
#     X = pd.concat([input_df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

#     # Predict
#     prediction = model.predict(X)
#     return prediction

# Prediction helper function
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    
    preprocessor = stroke_model["preprocessor"]
    model = stroke_model["model"]
    numeric_cols = stroke_model["numeric_cols"]
    encoded_cols = stroke_model["encoded_cols"]

    encoded_data = preprocessor.transform(input_df)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

    X = pd.concat([input_df[numeric_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return prediction, prob





@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Retrieve and map form data
            single_input = {
                "gender": gender_map[request.form["gender"]],
                "age": float(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "ever_married": ever_married_map[request.form["ever_married"]],
                "work_type": work_type_map[request.form["work_type"]],
                "Residence_type": residence_map[request.form["residence_type"]],
                "avg_glucose_level": float(request.form["avg_glucose_level"]),
                "bmi": float(request.form["bmi"]),
                "smoking_status": smoking_map[request.form["smoking_status"]]
            }
            #Map work type
            work_type_mapping ={
            "Government job":"Govt_job",
            "Children": "children",
            "Never Worked": "Never_worked",
            "Private": "Private",
            }

            # Make prediction
            prediction, prob = predict_input(single_input)
            result = "Likely" if prob > 0.2 else "Not Likely"



            # prediction = predict_input(single_input)
            # result = "Likely" if prediction[0] == 1 else "Not Likely"
            

            # return render_template("result.html", result=result)
            return render_template("result.html", result=result, prob=round(prob, 3))  

        except Exception as e:
            return render_template("result.html", result=f"Error: {e}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
