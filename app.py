from flask import Flask, jsonify, render_template, request
from model_parser import get_model_structure
from chat import get_model_layers_info

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/model/<model_name>', methods=['GET'])
def model(model_name):
    input_shape_str = request.args.get('shape')

    if not input_shape_str:
        return jsonify({'error': 'Missing input shape'}), 400 

    try:
        input_shape = tuple(map(int, input_shape_str.strip("[]").split(',')))
    except ValueError:
        return jsonify({'error': 'Invalid input shape format. Use: 1,3,224,224'}), 400

    structure = get_model_structure(model_name, input_shape)
    
    if 'error' in structure:
        return jsonify(structure), 404

    return jsonify(structure)

@app.route("/chat", methods=["POST"])
def explain_model():
    data = request.get_json()
    info = data.get("info")

    response = get_model_layers_info(info)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)