import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox
import threading

# Function to list available microphones
def list_microphones():
    recognizer = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    return mic_list

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    mic_list = list_microphones()
    
    if not mic_list:
        messagebox.showerror("Error", "No microphones detected. Please check your microphone connection.")
        return

    # Print available microphones for debugging
    print("Available Microphones:")
    for index, name in enumerate(mic_list):
        print(f"Index {index}: {name}")

    # Select the first available microphone (or specify an index)
    device_index = 0  # Change this if you need to select a specific microphone
    with sr.Microphone(device_index=device_index) as source:
        try:
            label_status.config(text="Listening...")
            
            # Adjust recognizer sensitivity to ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust duration as needed
            print("Listening...")  # Debugging output
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            label_status.config(text="Processing...")

            # Recognize speech
            text = recognizer.recognize_google(audio, language="en-US", show_all=False)
            print(f"Recognition result: {text}")  # Debugging output
            entry_result.delete(0, tk.END)  # Clear previous result
            entry_result.insert(0, text)   # Display recognized text
            label_status.config(text="Done!")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            messagebox.showerror("Error", "Could not understand the audio.")
            label_status.config(text="Ready")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition.")
            messagebox.showerror("Error", "Could not request results from Google Speech Recognition.")
            label_status.config(text="Ready")
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", str(e))
            label_status.config(text="Ready")

# Function to clear the text box
def clear_text():
    entry_result.delete(0, tk.END)

# Function to run the speech recognition in a separate thread
def start_listening_thread():
    # Start a new thread for speech recognition
    threading.Thread(target=recognize_speech, daemon=True).start()

# Initialize GUI
root = tk.Tk()
root.title("Speech Recognition Tool")
root.geometry("400x200")
root.resizable(False, False)

# Title Label
label_title = tk.Label(root, text="Speech Recognition Tool", font=("Arial", 16, "bold"))
label_title.pack(pady=10)

# Status Label
label_status = tk.Label(root, text="Ready", font=("Arial", 12), fg="green")
label_status.pack(pady=5)

# Recognized Text Entry
entry_result = tk.Entry(root, font=("Arial", 14), width=30)
entry_result.pack(pady=10)

# Buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_start = tk.Button(frame_buttons, text="Start Listening", font=("Arial", 12), command=start_listening_thread)
btn_start.grid(row=0, column=0, padx=5)

btn_clear = tk.Button(frame_buttons, text="Clear", font=("Arial", 12), command=clear_text)
btn_clear.grid(row=0, column=1, padx=5)

btn_exit = tk.Button(frame_buttons, text="Exit", font=("Arial", 12), command=root.quit)
btn_exit.grid(row=0, column=2, padx=5)

# Run the GUI event loop
root.mainloop()
