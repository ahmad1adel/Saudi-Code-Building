from flask import Flask, render_template, request, url_for, send_file
from model import get_answer, analyze_image_and_get_answer
import os
from werkzeug.utils import secure_filename
from fpdf import FPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==== Function to generate English PDF ====
def generate_pdf(data, image_path=None):
    print("\n📝 Generating PDF...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Inspection Results", ln=True, align="C")

    if image_path and os.path.exists(image_path):
        print(f"🖼️ Adding image to PDF: {image_path}")
        pdf.image(image_path, x=10, y=30, w=100)
        pdf.ln(85)

    # Table Header
    headers = ["Item", "Status", "Notes / SBC Reference"]
    col_width = pdf.w / 3.2
    row_height = 10

    print("📊 Adding table headers...")
    for header in headers:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(col_width, row_height, header, border=1)
    pdf.ln(row_height)

    # Table Rows
    print("📊 Adding table rows...")
    for row in data:
        print(f"➡️ Row: {row['item']} | {row['status']} | {row['notes'][:50]}...")
        pdf.set_font("Arial", size=10)
        pdf.cell(col_width, row_height, row["item"][:40], border=1)
        pdf.cell(col_width, row_height, row["status"][:40], border=1)
        pdf.multi_cell(col_width, row_height, row["notes"][:100], border=1)

    filename = "inspection_results_en.pdf"
    pdf.output(filename)
    print(f"✅ PDF saved as {filename}")
    return filename

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = None
    image_uploaded = False
    image_filename = None
    structured_results = None  # English only

    if request.method == 'POST':
        print("\n📩 POST request received.")
        # ===== Image Upload =====
        if 'image' in request.files and request.files['image'].filename.strip() != '':
            image = request.files['image']
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            print(f"🖼️ Image uploaded: {filename}")

            try:
                print("🔍 Analyzing image...")
                result = analyze_image_and_get_answer(image_path)
                structured_results = result["structured"]  # English only
                print(f"✅ Analysis completed. Found {len(structured_results)} items.")
                image_uploaded = True
                image_filename = filename
            except Exception as e:
                answer = f"❌ Error analyzing image: {str(e)}"
                print(answer)

        # ===== Text Question =====
        else:
            question = request.form.get('question', '').strip()
            print(f"💬 Text question received: {question}")
            if question:
                try:
                    result = get_answer(question)
                    answer = result['result']
                    print("✅ Answer generated:")
                    print(answer)
                except Exception as e:
                    answer = f"❌ Error processing question: {str(e)}"
                    print(answer)
            else:
                answer = "⚠️ Please enter a question or upload an image."
                print(answer)

    return render_template(
        'index.html',
        question=question,
        answer=answer,
        image_uploaded=image_uploaded,
        image_filename=image_filename,
        structured_results=structured_results
    )

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    print(f"\n⬇️ Downloading PDF for image: {filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = analyze_image_and_get_answer(image_path)
    data = result["structured"]
    pdf_file = generate_pdf(data, image_path=image_path)
    print("📤 Sending PDF file to user.")
    return send_file(pdf_file, as_attachment=True)

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    app.run(debug=True)
