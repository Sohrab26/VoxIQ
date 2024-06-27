from flask import Flask, render_template, request, redirect, url_for
import torch

# Import your model setup file here, make sure it does not have the same name as any Flask module or your script
from vox import SpeechClassifier, preprocess_user_audio  # Adjust this according to your actual file names and contents

app = Flask(__name__, template_folder='/Users/bakhodirulugov/Desktop/Backend/templates')

# Load your model (Assume it's already trained and saved)
model = SpeechClassifier()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the file temporarily
            file_path = 'temp_audio.wav'
            file.save(file_path)

            # Process the audio file
            input_tensor = preprocess_user_audio(file_path)
            prediction = model(input_tensor.unsqueeze(0))  # Make sure the input tensor shape matches model's expectation
            predicted_class = torch.argmax(prediction).item()

            # Return results
            return render_template('result.html', predicted_class=predicted_class)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
