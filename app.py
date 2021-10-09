import numpy as np
from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/',methods=['POST'])
def ml_api():
    data=json.loads(request.data)
    final_features = [np.array(list(data.values()))]
    prediction = model.predict(final_features)
    output = prediction[0]
    return jsonify({"output" : int(output)})

if __name__ == "__main__":
    app.run(debug=True)
