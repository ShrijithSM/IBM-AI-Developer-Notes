from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/health', methods=['GET','POST'])
def health():
    if request.method == 'GET':
        return jsonify({'status': 'healthy'})



if __name__ == '__main__':
    app.run()