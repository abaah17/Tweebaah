import pickle
import re
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model("model/modela.h5")
# loading
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# label2int = {'0': 0, '1': 1}
# int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 22


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', str(sentence))
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', str(sentence))
    sentence = re.sub(r'\s+', ' ', str(sentence))

    return sentence


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', str(text))


def get_predictions(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH, padding='post')
    # get the prediction
    prediction = model.predict(sequence)[0]
    # one-hot encoded vector, revert using np.argmax
    return np.argmax(prediction)


def show_predict(tweet):
    prediction = get_predictions(tweet)
    #     tweet_type = ""
    #     if (prediction == '1'):
    #         tweet_type = "The prediction is " + str(prediction) + " Is a positive tweet"
    #     else:
    #         tweet_type = "The prediction is " + str(prediction) + " It is a negative "
    #
    return prediction


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/take', methods=['GET', 'POST'])
def take():
    return render_template('index.html', methods=['GET', 'POST'])


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form['text-message']
        prediction = ""
        if data == prediction:
            prediction = 'Got None'
            return render_template('results.html', prediction=prediction)
        else:
            # print(prediction)
            prediction = show_predict(data)
            # print(prediction)
            return render_template('results.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
