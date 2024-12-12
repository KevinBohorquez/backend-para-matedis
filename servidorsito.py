from flask import Flask
import os

app = Flask(__name__)

@app.route('/prueba')
def hola_mundo():
    return 'Hola Mundo desde /prueba'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
