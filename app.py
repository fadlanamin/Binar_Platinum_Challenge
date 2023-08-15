import re
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

from LSTM_function import text_cleansing, model_lstm, lstm_upload
from Neural_Network_function import neural_network_model, neural_network_upload, text_cleansing

app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info= {
        'title': LazyString(lambda: 'API for Sentiment Analysis'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Binar Platinum Challenge : API for Sentiment Analysis'),
        'Author': LazyString(lambda: 'Kelompok 4 : Ahmad Fadlan Amin & Susilawaty Chen'),
    },
    host= LazyString(lambda: request.host) 
)

swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': '/docs/'

}

swagger = Swagger(app, template=swagger_template, 
                  config=swagger_config)


# Homepage
@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Binar Platinum Challenge : Senitment Analysis",
        "author": "Ahmad Fadlan Amin & Susilawaty"
    }
    return jsonify(welcome_msg)


#LSTM Model - Text input
@swag_from('docs/lstm_model_input.yml', methods=['POST'])
@app.route('/lstm_model_input', methods=['POST'])
def lstm_form():

    # Get text from input user
    raw_text = request.form["raw_text"]
    clean_text = text_cleansing(raw_text)

    # LSTM text
    results = model_lstm(clean_text)
    result_response = {"text_clean": clean_text, "results": results}
    return jsonify(result_response)

@swag_from('docs/lstm_upload.yml', methods=['POST'])
@app.route('/lstm_upload', methods=['POST'])
def LSTM_upload():

    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']

    # Read csv file to dataframe the analyize the sentiment
    df_lstm = lstm_upload(uploaded_file)
    result_response = df_lstm.T.to_dict()
    
    return jsonify(result_response)


#Neural Network Model - Text input
@swag_from('docs/nn_model_input.yml', methods=['POST'])
@app.route('/nn_model_input', methods=['POST'])
def neural_network_form():

    # Get text from input user
    raw_text = request.form["raw_text"]
    clean_text = text_cleansing(raw_text)

    # Neural Network text
    results = neural_network_model(clean_text)
    result_response = {"text_clean": clean_text, "results": results}

    return jsonify(result_response)


@swag_from('docs/nn_upload.yml', methods=['POST'])
@app.route('/nn_upload', methods=['POST'])
def Neural_Network_upload():

    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']

    # Read csv file to dataframe the analyize the sentiment
    df_nn = neural_network_upload(uploaded_file)
    result_response = df_nn.T.to_dict()

    return jsonify(result_response)


if __name__ == '__main__':
    app.run()