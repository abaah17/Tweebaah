import pickle
import re
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

model = load_model("model/modelt.h5")
# loading
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

label2int = {'0': 0, '1': 1}
int2label = {0: '0', 1: '1'}
SEQUENCE_LENGTH = 23


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
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = request.form.get('message')
    prediction = ""
    if data is None:
        prediction = 'Got None'
    else:
        # model.predict.predict returns a dictionarys
        prediction = show_predict(data)
    # return json.dumps(str(prediction))
    # print(prediction)

    return render_template('results.html', prediction=prediction)



    # return render_template('main.html', original_input={
    #     "email_text": data, },
    #
    #                        prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
