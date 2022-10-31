import pickle as pkl
import json 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

crop_rf_model = None
crop_labels = None

fert_rf_model = None
fert_svm_model = None

fertilizer_dict = None
soil_type_dict = None
crop_type_dict = None

MODEL = None

class_names = None

#load_saved_artifacts
def load_saved_artifacts():
	print("Loading... saved artifacts.....")

	global crop_rf_model
	global crop_labels
	global fert_rf_model
	global fert_svm_model
	global fertilizer_dict
	global soil_type_dict
	global crop_type_dict
	global MODEL
	global class_names

	class_names = ['bacterial_leaf_blight',
					'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast',
					'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

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

	MODEL = tf.keras.models.load_model("./EfficientNetB0")

	print("Loaded Artifacts done...")

#predict_crop
def predict_crop(X):
	rf_prediction = crop_labels[crop_rf_model.predict(X)[0]]
	return rf_prediction

def predict_fert(X):
    rf_prediction = fertilizer_dict[fert_rf_model.predict(X)[0]]
    svm_prediction = fertilizer_dict[fert_svm_model.predict(X)[0]]
    # xgb_prediction = fertilizer_dict[fert_xgb_model.predict(X)]

    return rf_prediction

def predict_disease():
	img_file = tf.keras.utils.load_img("./input_image.jpg",target_size=(224,224))
	img_arr = img_to_array(img_file)
	img_f = tf.expand_dims(img_arr,0)
	preds = MODEL.predict(img_f)
	output = class_names[np.argmax(preds)]
	confidence = np.round(preds.max(),3)*100
	return output,confidence



