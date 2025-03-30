import gradio as gr
from transformers import pipeline
# Import the DeepL translate function
from utils import translate as deepl_translate

# Initialize the Whisper ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype="auto",
    device="cuda:0"
)

# Initialize the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")


async def transcribe_and_translate(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please record an audio before submitting your request.")

    # Transcribe the audio
    transcription = pipe(inputs, generate_kwargs={
                         "task": "transcribe"}, return_timestamps=True)["text"]

    # Translate using the TranslationPipeline
    pipeline_translation = translator(transcription)[0]["translation_text"]

    # Translate using the DeepL API
    deepl_translation = await deepl_translate(transcription)

    return transcription, pipeline_translation, deepl_translation

demo = gr.Interface(
    fn=transcribe_and_translate,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath"),
    ],
    outputs=[
        "text",  # Output for transcription
        "text",  # Output for TranslationPipeline translation
        "text"   # Output for DeepL translation
    ],
    title="Whisper Large V3: Transcribe and Compare Translations",
    description=(
        "Transcribe long-form microphone or audio inputs and compare translations between the Helsinki-NLP model and DeepL API. "
        "Demo uses the checkpoint [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and 🤗 Transformers "
        "to transcribe audio files of arbitrary length, and translates the text using both methods."
    ),
    allow_flagging="never",
)

demo.launch()
