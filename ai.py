import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import os
import re
import serpapi

# Disable TensorFlow oneDNN optimizations to avoid cuFFT/cuDNN/cuBLAS registration errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)  # Use second voice if available
tts_engine.setProperty('rate', 140)  # Slower for clarity
tts_engine.setProperty('pitch', 1.2)  # Slightly higher pitch (if supported)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize LLM (Flan-T5-Small for faster response)
llm = pipeline("text2text-generation", model="google/flan-t5-small")

# SerpAPI key
SERPAPI_KEY = "234778ab6ec36756598f2704040a41f3b6b88f8b6072970148cdd8e6e776e55c"

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_for_wake_word():
    """Listen for the wake word 'Ashu'."""
    with sr.Microphone() as source:
        print("Listening for 'Ashu'...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        for _ in range(3):
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                if text == "ashu":
                    speak("Hey, I'm here! What's up?")
                    return True
                return False
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                speak("Sorry, I couldn't connect to the speech service.")
                return False
            except sr.WaitTimeoutError:
                continue
        return False

def listen_for_question():
    """Listen for a question or command."""
    with sr.Microphone() as source:
        print("Listening for your question...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        for _ in range(3):
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                question = recognizer.recognize_google(audio).lower()
                print(f"You said: {question}")
                if question == "bye ashu":
                    speak("Okay Ashu bye")
                    return None
                return question
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that. Could you repeat?")
                continue
            except sr.RequestError:
                speak("Sorry, I couldn't connect to the speech service.")
                return None
            except sr.WaitTimeoutError:
                continue
        return None

def web_search(query):
    """Search the web using SerpAPI and return a concise answer."""
    try:
        client = serpapi.Client(api_key=SERPAPI_KEY)
        results = client.search(q=query, num=1)
        if 'answer_box' in results:
            return results['answer_box'].get('answer', 'No clear answer found.')
        elif 'organic_results' in results and len(results['organic_results']) > 0:
            return results['organic_results'][0].get('snippet', 'No clear answer found.')
        return "Sorry, I couldn't find an answer online."
    except Exception:
        return "Sorry, web search failed. Try again later."

def get_llm_response(question):
    """Get a response from the LLM or web search."""
    # Check for basic math questions (e.g., "2 plus 2")
    math_pattern = r"(\d+)\s*(plus|minus|times|divided by)\s*(\d+)"
    match = re.match(math_pattern, question)
    if match:
        num1, op, num2 = match.groups()
        num1, num2 = int(num1), int(num2)
        if op == "plus":
            return f"{num1 + num2}"
        elif op == "minus":
            return f"{num1 - num2}"
        elif op == "times":
            return f"{num1 * num2}"
        elif op == "divided by":
            return f"{num1 / num2 if num2 != 0 else 'Cannot divide by zero'}"

    # Try LLM first
    response = llm(question, max_length=50, num_return_sequences=1)[0]['generated_text']
    # If LLM response is too short or vague, use web search
    if len(response) < 10 or response.lower() in ["i don't know", "not sure", "unknown"]:
        speak("Let me check online...")
        response = web_search(question)
    return response

def main():
    speak("Ashu is ready! Say my name to wake me up.")
    while True:
        if listen_for_wake_word():
            while True:
                question = listen_for_question()
                if question is None:
                    break
                speak("Let me think...")
                response = get_llm_response(question)
                print(f"Ashu's response: {response}")
                speak(response)

if __name__ == "__main__":
    main()