from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, render_template,request, jsonify, send_from_directory
import io
import werkzeug
import datetime
import os

# initialize our Flask application and the Keras model
app = Flask(__name__)

model = ResNet50(weights="imagenet")
print("loaded model")

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

def create_new_folder(local_dir):
	newpath = local_dir
	if not os.path.exists(newpath):
		os.makedirs(newpath)
		
	return newpath


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS


def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")
	print("loaded model")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/save', methods = ['POST'])
def save():
	data = {"success": False}

	if request.method == 'POST' and request.files['image']:
		img = request.files['image']
		filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(img.filename)
		img_name = werkzeug.secure_filename(filename_)
		create_new_folder(app.config['UPLOAD_FOLDER'])
		saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
		img.save(saved_path)

		#image = open_oriented_im(saved_path)
		# preprocess the image and prepare it for classification
		image = Image.open(saved_path)
		image = prepare_image(image, target=(224, 224))


		preds = model.predict(image)
		results = imagenet_utils.decode_predictions(preds)
		data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
		for (imagenetID, label, prob) in results[0]:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)

			# indicate that the request was a success
		data["success"] = True

		# return the data dictionary as a JSON response
		return jsonify(data)


		#return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
	else:
		return "Where is the image?"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if request.method == "POST" and request.files['image']:
		imagefile = request.files["image"].read()
		image = Image.open(io.BytesIO(imagefile))


		# preprocess the image and prepare it for classification
		image = prepare_image(image, target=(224, 224))

		# classify the input image and then initialize the list
		# of predictions to return to the client
		preds = model.predict(image)
		results = imagenet_utils.decode_predictions(preds)
		data["predictions"] = []

		# loop over the results and add them to the list of
		# returned predictions
		for (imagenetID, label, prob) in results[0]:
			r = {"label": label, "probability": float(prob)}
			data["predictions"].append(r)

		# indicate that the request was a success
		data["success"] = True

		print(data)

	# return the data dictionary as a JSON response
	return jsonify(data)

if __name__ == "__main__":
	print("START FLASK")
	#load_model()

	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
