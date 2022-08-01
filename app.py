import numpy as np
from flask import Flask, request, jsonify, render_template,redirect
import pickle as pkl

app = Flask(__name__)

# crop_knn_model = pkl.load(open('models/Crop/knn_pipeline.pkl',"rb"))
crop_rf_model  = pkl.load(open('models/Crop/rf_pipeline.pkl',"rb"))
# crop_xgb_model = pkl.load(open('models/Crop/xgb_pipeline.pkl',"rb"))

crop_labels = pkl.load(open('models/Crop/label_dictionary.pkl',"rb"))

fert_rf_model = pkl.load(open('models/Fertilizer/rf_pipeline.pkl',"rb"))
fert_svm_model = pkl.load(open('models/Fertilizer/svm_pipeline.pkl',"rb"))
# fert_xgb_model = pkl.load(open('models/Fertilizer/xgb_pipeline.pkl',"rb"))

fertilizer_dict = pkl.load(open('models/Fertilizer/fertilizer_dict.pkl',"rb"))
soil_type_dict = pkl.load(open('models/Fertilizer/soil_type_dict.pkl',"rb"))
crop_type_dict = pkl.load(open('models/Fertilizer/crop_type_dict.pkl',"rb"))

def predict_crop(X):
    # knn_prediction = crop_labels[crop_knn_model.predict(X)[0]]
    rf_prediction = crop_labels[crop_rf_model.predict(X)[0]]
    # xgb_prediction = crop_labels[crop_xgb_model.predict(X)]

    return rf_prediction

def predict_fert(X):
    rf_prediction = fertilizer_dict[fert_rf_model.predict(X)[0]]
    svm_prediction = fertilizer_dict[fert_svm_model.predict(X)[0]]
    # xgb_prediction = fertilizer_dict[fert_xgb_model.predict(X)]

    return (rf_prediction,svm_prediction)


@app.route("/<name>")
def home(name):
    if name=="index":
        return render_template("index.html")
    elif name=="crop":
        return render_template("crop.html")
    else:
        print(soil_type_dict,crop_type_dict)
        return render_template("fertilizer.html",soil_types = soil_type_dict,crop_types=crop_type_dict)


@app.route("/predict/<name>",methods=["GET","POST"])
def predict(name):
    ## Crop prediction
    if name=="crop":
        if request.method=="POST":
            input_values = list(request.form.to_dict().values())
            input_values = list(map(float,input_values))
            X = [input_values]
            result = predict_crop(X)
            return render_template('result.html',input_values = input_values,result = result)
        return redirect('/crop')

    ## Fertilizer prediction    
    else:
        if request.method=="POST":
            input_values = list(request.form.to_dict().values())
            input_values = list(map(float,input_values))
            X = [input_values]
            result = predict_fert(X)
            return render_template('result.html',input_values = input_values,result = result)
        return redirect('/fertilizer')



if __name__=="__main__":
    app.run(debug=True)
