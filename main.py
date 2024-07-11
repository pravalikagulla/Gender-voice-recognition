from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    audiofile = request.files['hello.wav']
    audio_path = "hello.wav/" + audiofile.filename
    audiofile.save(audio_path)

    my_prediction = classifier.predict(audio_path)

    if my_prediction == 0:
        res_value = "male"
    else:
        res_value = "female"
    return render_template('index.html', prediction_text='gender is {}'.format(res_value))


if __name__ == '__main__':
    app.run(debug=True)
