from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify({'message': 'Hello, World!'})
#    return {'message': 500}
#    return 'Hello, World!'



@app.route('/health', methods=['GET','POST'])
def health():
    if request.method == 'GET':
        return jsonify(status='OK', method= "GET"), 200
    
    if request.method == 'POST':
        return jsonify(status='OK', method= "POST"), 200

if __name__ == '__main__':
    app.run()