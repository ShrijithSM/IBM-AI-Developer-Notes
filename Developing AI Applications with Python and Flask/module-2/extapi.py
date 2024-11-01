from flask import Flask
import requests

app = Flask(__name__)

@app.route('/')
def get_authod():
    res = requests.get('https://openlibrary.org/search/authors.json?q=tolkien&mode=everything')
    
    if res.status_code == 200:
        return res.json()
    elif res.status_code == 404:
        return {"message": "Not Found"}, 404
    else:
        return {"message": "Internal Server Error"}, 500
    
if __name__ == '__main__':
    app.run()