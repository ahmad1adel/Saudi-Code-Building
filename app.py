from flask import Flask, render_template, request, url_for
from model import get_answer, analyze_image_and_get_answer
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # for display in browser
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = None
    image_uploaded = False
    image_filename = None

    if request.method == 'POST':
        # Check for image upload
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image']
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            try:
                result = analyze_image_and_get_answer(image_path)
                answer = result['result']
                image_uploaded = True
                image_filename = filename
            except Exception as e:
                answer = f"❌ Error analyzing image: {str(e)}"

        # Otherwise, handle text question
        else:
            question = request.form.get('question', '').strip()
            if question:
                try:
                    result = get_answer(question)
                    answer = result['result']
                except Exception as e:
                    answer = f"❌ Error processing question: {str(e)}"
            else:
                answer = "⚠️ Please enter a question or upload an image."

    return render_template(
        'index.html',
        question=question,
        answer=answer,
        image_uploaded=image_uploaded,
        image_filename=image_filename
    )

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True)
