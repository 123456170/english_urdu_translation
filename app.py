import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ur"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

def translate_en_to_ur(text):
    if not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output

demo = gr.Interface(
    fn=translate_en_to_ur,
    inputs=gr.Textbox(lines=4, placeholder="Enter English text here..."),
    outputs=gr.Textbox(lines=4, label="Urdu Translation"),
    title="English to Urdu Translator",
    description="Offline English → Urdu translation using Hugging Face Transformers (no API key required)."
)

if __name__ == "__main__":
    demo.launch()
