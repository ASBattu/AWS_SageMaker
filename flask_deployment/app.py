# from flask import Flask, render_template, request
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import re, string
# import numpy as np
# import tensorflow as tf
# import pickle

# #Import stopwords library
# from nltk.corpus import stopwords
# stoplist = set(stopwords.words("english"))

# #Load locally saved model
# bert_model = TFBertForSequenceClassification.from_pretrained(r".\My_trained_bert_model")
# # bert_model = pickle.load(open('trained_model.pkl','rb'))
# app = Flask(__name__)

# # Load the BERT Tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# def preprocess_text(text):
#     # Remove http / https links
#     text = re.sub(r'http\S+|https\S+', '', text)
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Remove numbers
#     text = re.sub(r'\d+', '', text)
#     # Remove stopwords
#     text = ' '.join(word for word in text.split() if word not in stoplist)
#     # Remove extra whitespaces
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text



# @app.route('/')
# def index():
#     return render_template('index.html')



# @app.route('/predict',methods=['POST'])
# def predict_sentiment():
#     input_text = request.form.get('txt_input')

#     #Preprocess input text
#     cleaned_text = preprocess_text(input_text)

#     #Tokenize cleaned text
#     encoded_input = tokenizer(
#         cleaned_text,
#         max_length=18,
#         padding="max_length",
#         truncation=True,
#         return_tensors='tf'
#     )

#     #Get text tensor values
#     input_ids = encoded_input["input_ids"]
#     attention_mask = encoded_input["attention_mask"]
#     token_type_ids = encoded_input["token_type_ids"]

#     #Predict sentiment label for input
#     predictions = bert_model.predict([input_ids, attention_mask, token_type_ids])
#     logits = predictions.logits[0]
#     probabilities = tf.nn.softmax(logits)
#     predicted_label = tf.argmax(probabilities).numpy()
#     confidence_level = np.max(probabilities)

#     # Map the predicted label to its corresponding sentiment category
#     sentiment_categories = ["Negative", "Positive"]
#     predicted_sentiment = sentiment_categories[predicted_label]

#     # sentiment = f'Predicted sentiment is: {predicted_sentiment} with a confidence level of {confidence_level}'
#     result = str(predicted_sentiment)
#     result = [str(predicted_sentiment),str(confidence_level)]
#     # return [str(predicted_sentiment),str(confidence_level)]
#     return render_template('index.html',result=result)

# if __name__ == '__main__':
#     app.run(debug=True)
#     # app.run(host='0.0.0.0',port=8080)


from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification
import re
import string
import numpy as np

# Import stopwords library
from nltk.corpus import stopwords
stoplist = set(stopwords.words("english"))

# Load locally saved model
bert_model = TFBertForSequenceClassification.from_pretrained(r".\My_trained_bert_model")
# bert_model = BertForSequenceClassification.from_pretrained(r".\My_trained_bert_model",from_tf = True)
app = Flask(__name__)

# Load the BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    # Remove http / https links
    text = re.sub(r'http\S+|https\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stoplist)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    input_text = request.form.get('txt_input')

    # Preprocess input text
    cleaned_text = preprocess_text(input_text)

    # Tokenize cleaned text
    encoded_input = tokenizer(
        cleaned_text,
        max_length=18,
        padding="max_length",
        truncation=True,
        return_tensors='np'  # Use 'np' for NumPy arrays
    )

    # Get text tensor values
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    token_type_ids = encoded_input["token_type_ids"]

    # Predict sentiment label for input
    with np.errstate(divide='ignore'):  # Ignore division by zero warning
        logits = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    predicted_label = np.argmax(probabilities)
    confidence_level = np.max(probabilities)

    # Map the predicted label to its corresponding sentiment category
    sentiment_categories = ["Negative", "Positive"]
    predicted_sentiment = sentiment_categories[predicted_label]

    result = [str(predicted_sentiment), str(confidence_level)]
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=8080)