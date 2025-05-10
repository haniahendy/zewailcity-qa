from flask import Flask, request, jsonify, render_template
from model import QASystem
from data import qa_pairs

app = Flask(__name__)
qa_system = QASystem("siamese_model.pth")
qa_system.load_data(qa_pairs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa_system.find_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)