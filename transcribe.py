import whisper
import sys
from pathlib import Path
from pytube import YouTube
import moviepy.editor as mp
import os
import yt_dlp
import threading
from tqdm import tqdm
import time
import tkinter as tk
from tkinter import ttk, filedialog
import re
import queue
from tkinter import font  # Imported for font customization
import google.generativeai as genai  # Imported for Gemini API integration
import yaml
import markdown
from tkinterweb import HtmlFrame  # Import HtmlFrame for rendering HTML content

# Define a CSS template for styling the HTML content
CSS_TEMPLATE = """
<style>
    body {
        font-size: 16px;                 /* Adjusts the base font size */
        font-family: Arial, sans-serif;  /* Sets the font family */
        color: #333333;                  /* Sets the text color */
        line-height: 1.6;                /* Adjusts the line height for readability */
        margin: 10px;                    /* Adds margin around the content */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333333;                  /* Sets the color for headings */
    }
    p {
        margin-bottom: 10px;             /* Adds space below paragraphs */
    }
    ul, ol {
        margin-left: 20px;               /* Indents lists */
    }
</style>
"""

# Load configuration from config.yaml
config_path = Path('config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Accessing the configuration settings
gemini_api_key = config['gemini']['api_key']
temp_dir = config['directories']['temp_dir']
output_dir = config['directories']['output_dir']
whisper_model_size = config['whisper']['model_size']

# Define a class to redirect stdout to the log callback
class QueueWriter:
    """A writer class that redirects stdout to a thread-safe queue."""
    def __init__(self, log_callback):
        self.log_callback = log_callback

    def write(self, message):
        """Write a message to the log callback if it's not just a newline."""
        if message.strip():
            self.log_callback(message)

    def flush(self):
        """Flush method required for file-like objects."""
        pass

def download_youtube_audio(url, temp_dir="temp"):
    """Download audio from a video with progress tracking."""
    Path(temp_dir).mkdir(exist_ok=True)
    
    try:
        # yt-dlp options with progress hooks
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'progress_hooks': [download_progress_hook],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            wav_path = Path(temp_dir) / f"{video_id}.wav"
            
            if not wav_path.exists():
                raise FileNotFoundError(f"Generated audio file not found at {wav_path}")
                
            return str(wav_path)
            
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")  # This will be captured by QueueWriter
        raise

def download_progress_hook(d):
    """Progress hook for yt-dlp to report download status."""
    # Get the app instance's update_status if available
    log_callback = getattr(app, 'update_status_threadsafe', print) if 'app' in globals() else print
    
    if d['status'] == 'downloading':
        try:
            # Strip ANSI escape codes for cleaner output
            percent_str = re.sub(r'\x1b\[[0-9;]*m', '', d['_percent_str'])
            percent = float(percent_str.replace('%', '').strip())
            speed = d.get('speed', 0)
            if speed:
                speed_str = f"{speed/1024/1024:.2f} MiB/s"
                log_callback(f"Downloading: {percent:.1f}% at {speed_str}")
            else:
                log_callback(f"Downloading: {percent:.1f}%")
        except Exception as e:
            log_callback(f"Error calculating progress: {str(e)}")
    elif d['status'] == 'finished':
        log_callback('Download complete, converting...')

def extract_audio_from_video(video_path, temp_dir="temp"):
    """Extract audio from a video file with progress tracking."""
    video_path = Path(video_path).resolve()
    Path(temp_dir).mkdir(exist_ok=True)
    wav_path = Path(temp_dir) / f"{video_path.stem}.wav"
    
    try:
        video = mp.VideoFileClip(str(video_path))
        # Use tqdm for a simple progress bar
        with tqdm(total=100, desc="Extracting audio") as pbar:
            video.audio.write_audiofile(
                str(wav_path),
                logger=None  # Disable moviepy's logger to prevent conflict
            )
            pbar.n = 100  # Indicate completion
            pbar.refresh()
        video.close()
        
    except OSError as e:
        print(f"Error: Could not open video file. Please check if the file exists and the path is correct.")
        print(f"Full path being used: {video_path}")
        print(f"Detailed error: {str(e)}")
        raise
    
    return str(wav_path)

def transcribe_audio(input_path, log_callback=print, transcript_callback=None):
    """Main transcription function orchestrating the transcription process."""
    try:
        # Load the Whisper model
        log_callback("Loading Whisper model...")
        model = load_whisper_model()

        # Process the input to obtain the audio path
        audio_path = process_input(input_path, log_callback)

        # Redirect stdout to capture verbose logs from Whisper
        redirect_stdout(log_callback)

        # Perform transcription
        result = perform_transcription(model, audio_path, log_callback)

        # Restore original stdout after transcription
        restore_stdout()

        # Process the transcription results
        transcript = process_transcription_result(result, log_callback, transcript_callback)

        # Save the transcription with speaker information
        save_transcription_with_speakers(input_path, result)

        return "\n".join(transcript)

    except Exception as e:
        log_callback(f"Error during transcription: {str(e)}")
        restore_stdout()
        raise e

def load_whisper_model():
    """Load and return the Whisper model."""
    return whisper.load_model("small")

def process_input(input_path, log_callback):
    """Determine the input type and extract audio accordingly."""
    input_path = input_path.strip("'\"").replace('\\', '')

    if input_path.startswith(('http://', 'https://', 'www.')):
        log_callback("Processing URL...")
        return download_youtube_audio(input_path)
    elif input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        log_callback("Processing video file...")
        return extract_audio_from_video(input_path)
    else:
        return input_path

def perform_transcription(model, audio_path, log_callback):
    """Perform transcription on the given audio file using the Whisper model."""
    try:
        log_callback("\nStarting transcription...")
        start_time = time.time()

        # Perform transcription with verbose=True to get real-time logs
        result = model.transcribe(
            audio_path,
            verbose=True,
        )

        # Calculate and display total time
        total_time = time.time() - start_time
        log_callback(f"\nTranscription completed in {total_time:.1f} seconds")

        return result

    except Exception as e:
        log_callback(f"Error during transcription: {str(e)}")
        raise e

def process_transcription_result(result, log_callback, transcript_callback):
    """Process the transcription result and log the output."""
    try:
        log_callback("\nTranscription:")
        transcript = []
        for segment in result["segments"]:
            timestamp = format_timestamp(segment["start"])
            text = segment["text"]
            log_message = f"[{timestamp}] {text}"
            log_callback(log_message)
            if transcript_callback:
                transcript_callback(text)
            transcript.append(log_message)
        return transcript

    except Exception as e:
        log_callback(f"Error during processing transcription: {str(e)}")
        raise e

def redirect_stdout(log_callback):
    """Redirect stdout to a QueueWriter for logging."""
    global original_stdout_backup
    original_stdout_backup = sys.stdout
    sys.stdout = QueueWriter(log_callback)

def restore_stdout():
    """Restore the original stdout."""
    global original_stdout_backup
    if 'original_stdout_backup' in globals():
        sys.stdout = original_stdout_backup

def save_transcription_with_speakers(input_path, result):
    """Save transcription to a text file with timestamps."""
    # Use output directory from GUI if available
    if hasattr(app, 'output_dir') and app.output_dir:
        output_folder = app.output_dir
    else:
        if input_path.startswith(('http://', 'https://', 'www.')):
            output_folder = Path('output')
        else:
            output_folder = Path(input_path).parent / "output"
    
    if 'youtu.be' in input_path:
        video_id = input_path.split('/')[-1]
    elif 'youtube.com' in input_path:
        video_id = input_path.split('v=')[-1].split('&')[0]
    else:
        video_id = Path(input_path).stem
    
    output_folder.mkdir(exist_ok=True)
    output_path = output_folder / f"{video_id}_transcription.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            start = format_timestamp(segment["start"])
            text = segment["text"]
            f.write(f"[{start}] {text}\n")
    
    print(f"Transcription saved to: {output_path}")

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

class TranscriptionApp:
    """GUI application for transcribing audio from YouTube URLs or local files."""
    def __init__(self, root):
        self.root = root
        self.root.title("Universal Transcription App")
        self.root.geometry("800x900")  # Increased height to accommodate new sections
        
        # Set default output directory to Downloads/Transcriptions
        downloads_path = str(Path.home() / "Downloads" / "Transcriptions")
        self.output_dir = Path(downloads_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transcript = ""  # Initialize transcript storage
        # Initialize variables to store HTML content
        self.meeting_minutes_html_content = ""
        self.key_figures_html_content = ""
        self.roadmap_html_content = ""
        self.todos_html_content = ""
        self.list_steps_html_content = ""
        self.priorities_html_content = ""
        self.followup_html_content = ""  # Initialize Follow-Up content storage
        
        self.setup_ui()
        
        # Initialize a queue for thread-safe logging
        self.log_queue = queue.Queue()
        self.root.after(100, self.process_log_queue)
        
    def setup_ui(self):
        """Set up the user interface components with an enhanced layout."""
        
        # Increase the window size for a more spacious layout
        window_size = config['ui']['window_size']
        width = window_size['width']
        height = window_size['height']
        self.root.geometry(f"{width}x{height}")
        
        # Create a Canvas and a vertical Scrollbar for the main frame
        main_canvas = tk.Canvas(self.root, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        # Create a frame inside the Canvas with padding
        self.main_frame = tk.Frame(main_canvas, padx=20, pady=20)
        main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        # Bind the configuration to update the scrollregion
        self.main_frame.bind("<Configure>", lambda event, canvas=main_canvas: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # -------------------- Input Section --------------------
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding=(20, 10))
        input_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

        # video URL input
        ttk.Label(input_frame, text="Video URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(input_frame, textvariable=self.url_var, width=80)
        self.url_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Local file input
        ttk.Label(input_frame, text="Local File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_var = tk.StringVar()
        self.file_entry = ttk.Entry(input_frame, textvariable=self.file_var, width=80)
        self.file_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse("file")).grid(row=1, column=3, padx=5, pady=5)
        
        # Output folder
        ttk.Label(input_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar(value=str(self.output_dir))
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_var, width=80)
        self.output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse("directory")).grid(row=2, column=3, padx=5, pady=5)
        
        # Load existing transcription
        ttk.Label(input_frame, text="Load Transcription:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.transcription_var = tk.StringVar()
        self.transcription_entry = ttk.Entry(input_frame, textvariable=self.transcription_var, width=80)
        self.transcription_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        ttk.Button(input_frame, text="Browse", command=lambda: self.browse("transcription")).grid(row=3, column=3, padx=5, pady=5)
        
        # -------------------- Action Buttons --------------------
        button_frame = ttk.Frame(self.main_frame, padding=(20, 10))
        button_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Transcribe button
        self.transcribe_btn = ttk.Button(button_frame, text="Transcribe", command=self.start_transcription)
        self.transcribe_btn.grid(row=0, column=0, padx=10, pady=5)
        
        # Copy Transcript button
        self.copy_btn = ttk.Button(button_frame, text="Copy Transcript", command=self.copy_transcript_to_clipboard)
        self.copy_btn.grid(row=0, column=1, padx=10, pady=5)
        self.copy_btn.state(['disabled'])  # Disabled until transcript is available
        
        # -------------------- Separator --------------------
        ttk.Separator(self.main_frame, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        # -------------------- Status Log Section --------------------
        log_frame = ttk.LabelFrame(self.main_frame, text="Status Log", padding=(20, 10))
        log_frame.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        self.status_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.status_text.pack(fill="both", expand=True)
        
        # -------------------- AI Generation Buttons --------------------
        ai_buttons_frame = ttk.LabelFrame(self.main_frame, text="AI Generation", padding=(20, 10))
        ai_buttons_frame.grid(row=4, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Meeting Minutes button
        self.meeting_minutes_btn = ttk.Button(ai_buttons_frame, text="Meeting Minutes", command=self.generate_meeting_minutes)
        self.meeting_minutes_btn.grid(row=0, column=0, padx=10, pady=5)
        self.meeting_minutes_btn.state(['disabled'])  # Disabled until transcript is available
        
        # Key Figures button
        self.key_figures_btn = ttk.Button(ai_buttons_frame, text="Key Figures", command=self.generate_key_figures)
        self.key_figures_btn.grid(row=0, column=1, padx=10, pady=5)
        self.key_figures_btn.state(['disabled'])  # Disabled until transcript is available
        
        # Roadmap button
        self.roadmap_btn = ttk.Button(ai_buttons_frame, text="Roadmap", command=self.generate_roadmap)
        self.roadmap_btn.grid(row=0, column=2, padx=10, pady=5)
        self.roadmap_btn.state(['disabled'])  # Disabled until transcript is available
        
        # To-do's button
        self.todos_btn = ttk.Button(ai_buttons_frame, text="To-do's", command=self.generate_todos)
        self.todos_btn.grid(row=0, column=3, padx=10, pady=5)
        self.todos_btn.state(['disabled'])  # Disabled until transcript is available
        
        # List Steps button
        self.list_steps_btn = ttk.Button(ai_buttons_frame, text="List Steps", command=self.generate_list_steps)
        self.list_steps_btn.grid(row=0, column=4, padx=10, pady=5)
        self.list_steps_btn.state(['disabled'])  # Disabled until transcript is available
        
        # Priorities button
        self.priorities_btn = ttk.Button(ai_buttons_frame, text="Priorities", command=self.generate_priorities)
        self.priorities_btn.grid(row=0, column=5, padx=10, pady=5)
        self.priorities_btn.state(['disabled'])  # Disabled until transcript is available
        
        # Follow-Up button
        self.followup_btn = ttk.Button(ai_buttons_frame, text="Follow-Up", command=self.generate_followup)
        self.followup_btn.grid(row=0, column=6, padx=10, pady=5)
        self.followup_btn.state(['disabled'])  # Disabled until transcript is available
        
        # -------------------- Tabbed Content Section --------------------
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=5, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # Create frames for each tab
        self.meeting_minutes_tab = ttk.Frame(self.notebook)
        self.key_figures_tab = ttk.Frame(self.notebook)
        self.roadmap_tab = ttk.Frame(self.notebook)
        self.todos_tab = ttk.Frame(self.notebook)
        self.list_steps_tab = ttk.Frame(self.notebook)
        self.priorities_tab = ttk.Frame(self.notebook)
        self.followup_tab = ttk.Frame(self.notebook)

        # Add frames to notebook
        self.notebook.add(self.meeting_minutes_tab, text='Meeting Minutes')
        self.notebook.add(self.key_figures_tab, text='Key Figures')
        self.notebook.add(self.roadmap_tab, text='Roadmap')
        self.notebook.add(self.todos_tab, text='To-do\'s')
        self.notebook.add(self.list_steps_tab, text='List Steps')
        self.notebook.add(self.priorities_tab, text='Priorities')
        self.notebook.add(self.followup_tab, text='Follow-Up')

        # Set up content in each tab
        # Meeting Minutes tab
        self.meeting_minutes_html = HtmlFrame(self.meeting_minutes_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.meeting_minutes_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_meeting_minutes_btn = ttk.Button(
            self.meeting_minutes_tab, 
            text="Copy Meeting Minutes", 
            command=self.copy_meeting_minutes_to_clipboard
        )
        self.copy_meeting_minutes_btn.pack(pady=5)
        self.copy_meeting_minutes_btn.state(['disabled'])

        # Key Figures tab
        self.key_figures_html = HtmlFrame(self.key_figures_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.key_figures_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_key_figures_btn = ttk.Button(
            self.key_figures_tab, 
            text="Copy Key Figures", 
            command=self.copy_key_figures_to_clipboard
        )
        self.copy_key_figures_btn.pack(pady=5)
        self.copy_key_figures_btn.state(['disabled'])

        # Roadmap tab
        self.roadmap_html = HtmlFrame(self.roadmap_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.roadmap_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_roadmap_btn = ttk.Button(
            self.roadmap_tab, 
            text="Copy Roadmap", 
            command=self.copy_roadmap_to_clipboard
        )
        self.copy_roadmap_btn.pack(pady=5)
        self.copy_roadmap_btn.state(['disabled'])

        # To-do's tab
        self.todos_html = HtmlFrame(self.todos_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.todos_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_todos_btn = ttk.Button(
            self.todos_tab, 
            text="Copy To-do's", 
            command=self.copy_todos_to_clipboard
        )
        self.copy_todos_btn.pack(pady=5)
        self.copy_todos_btn.state(['disabled'])

        # List Steps tab
        self.list_steps_html = HtmlFrame(self.list_steps_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.list_steps_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_list_steps_btn = ttk.Button(
            self.list_steps_tab, 
            text="Copy List Steps", 
            command=self.copy_list_steps_to_clipboard
        )
        self.copy_list_steps_btn.pack(pady=5)
        self.copy_list_steps_btn.state(['disabled'])

        # Priorities tab
        self.priorities_html = HtmlFrame(self.priorities_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.priorities_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_priorities_btn = ttk.Button(
            self.priorities_tab, 
            text="Copy Priorities", 
            command=self.copy_priorities_to_clipboard
        )
        self.copy_priorities_btn.pack(pady=5)
        self.copy_priorities_btn.state(['disabled'])

        # Follow-Up tab
        self.followup_html = HtmlFrame(self.followup_tab, horizontal_scrollbar="auto", messages_enabled=False, height=150)
        self.followup_html.pack(fill="both", expand=True, padx=5, pady=5)
        self.copy_followup_btn = ttk.Button(
            self.followup_tab, 
            text="Copy Follow-Up", 
            command=self.copy_followup_to_clipboard
        )
        self.copy_followup_btn.pack(pady=5)
        self.copy_followup_btn.state(['disabled'])

        # Configure grid to make notebook expand with window
        self.main_frame.rowconfigure(5, weight=1)
        for col in range(4):
            self.main_frame.columnconfigure(col, weight=1)

        # Add padding to all children of main_frame for consistent spacing
        for child in self.main_frame.winfo_children():
            if isinstance(child, ttk.Notebook):
                child.grid_configure(padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            else:
                child.grid_configure(padx=5, pady=5)
        
    def browse(self, dialog_type="file"):
        """Open a file or directory dialog based on the dialog_type."""
        if dialog_type == "file":
            filename = filedialog.askopenfilename(
                filetypes=[
                    ("Media Files", "*.mp4 *.avi *.mov *.mkv *.mp3 *.wav"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                self.file_var.set(filename)
        elif dialog_type == "directory":
            folder = filedialog.askdirectory()
            if folder:
                self.output_var.set(folder)
                # Update transcription dialog's initialdir when output folder changes
                self.transcription_var.set("")  # Clear any existing transcription path
        elif dialog_type == "transcription":
            # Use output folder as initial directory if set
            initial_dir = self.output_var.get() if self.output_var.get() else None
            filename = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                self.transcription_var.set(filename)
                self.load_transcription(filename)

    def load_transcription(self, filepath):
        """Load an existing transcription file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clear existing transcript and status
            self.transcript = ""
            self.status_text.delete(1.0, tk.END)
            
            # Update the transcript and status
            self.transcript = content
            self.status_text.insert(tk.END, content)
            
            # Enable relevant buttons
            self.copy_btn.state(['!disabled'])
            self.meeting_minutes_btn.state(['!disabled'])
            self.key_figures_btn.state(['!disabled'])
            self.roadmap_btn.state(['!disabled'])
            self.todos_btn.state(['!disabled'])
            self.list_steps_btn.state(['!disabled'])
            self.priorities_btn.state(['!disabled'])
            self.followup_btn.state(['!disabled'])
            
            self.update_status_threadsafe("Transcription file loaded successfully!")
            
        except Exception as e:
            self.update_status_threadsafe(f"Error loading transcription file: {str(e)}")

    def update_status_threadsafe(self, message):
        """Add a message to the log queue in a thread-safe manner."""
        self.log_queue.put(message)
        
    def process_log_queue(self):
        """Process messages in the log queue and update the Text widget."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.status_text.insert(tk.END, message + "\n")
                self.status_text.see(tk.END)
                
                # Check if the message is part of the transcript
                if message.startswith("["):
                    # Extract the text after the timestamp
                    transcript_line = message.split("] ", 1)[1]
                    self.transcript += transcript_line + "\n"
                    
                    # Enable the copy button if not already
                    if not self.copy_btn.instate(['!disabled']):
                        self.copy_btn.state(['!disabled'])
        except queue.Empty:
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.process_log_queue)
            
    def start_transcription(self):
        """Start the transcription process in a separate thread."""
        # If a transcription is loaded, don't proceed with new transcription
        if self.transcription_var.get().strip():
            self.update_status_threadsafe("Error: Please clear the loaded transcription before starting a new transcription")
            return
            
        # Reset transcript
        self.transcript = ""
        self.copy_btn.state(['disabled'])  # Disable copy button until new transcript is available
        self.status_text.delete(1.0, tk.END)  # Clear previous logs
        
        # Reset generated sections
        self.meeting_minutes_html_content = ""
        self.key_figures_html_content = ""
        self.roadmap_html_content = ""
        self.todos_html_content = ""
        self.list_steps_html_content = ""
        self.priorities_html_content = ""
        self.followup_html_content = ""  # Reset Follow-Up content storage
        
        # Clear HtmlFrame widgets using .load_html("") instead of .set_content("")
        self.meeting_minutes_html.load_html("")  # Clear Meeting Minutes section
        self.key_figures_html.load_html("")       # Clear Key Figures section
        self.roadmap_html.load_html("")           # Clear Roadmap section
        self.todos_html.load_html("")             # Clear To-do's section
        self.list_steps_html.load_html("")         # Clear List Steps section
        self.priorities_html.load_html("")         # Clear Priorities section
        self.followup_html.load_html("")            # Clear Follow-Up section
        
        # Disable copy buttons for generated sections
        self.copy_meeting_minutes_btn.state(['disabled'])
        self.copy_key_figures_btn.state(['disabled'])
        self.copy_roadmap_btn.state(['disabled'])
        self.copy_todos_btn.state(['disabled'])
        self.copy_list_steps_btn.state(['disabled'])
        self.copy_priorities_btn.state(['disabled'])
        self.copy_followup_btn.state(['disabled'])
        
        # Validate input
        url = self.url_var.get().strip()
        file_path = self.file_var.get().strip()
        
        if not url and not file_path:
            self.update_status_threadsafe("Error: Please provide either a URL or select a local file")
            return
            
        if url and file_path:
            self.update_status_threadsafe("Error: Please provide either a URL or a local file, not both")
            return
            
        # Disable inputs during transcription
        self.transcribe_btn.state(['disabled'])
        
        # Start transcription in a separate thread
        threading.Thread(target=self.run_transcription, daemon=True).start()
        
    def run_transcription(self):
        """Run the transcription process and handle GUI updates."""
        try:
            # Determine the input path
            input_path = self.url_var.get().strip() or self.file_var.get().strip()
            self.output_dir = Path(self.output_var.get())
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            self.update_status_threadsafe("Starting transcription...")
            
            # Call transcribe_audio with the thread-safe log callback and transcript callback
            transcribed_text = transcribe_audio(
                input_path, 
                log_callback=self.update_status_threadsafe,
                transcript_callback=self.collect_transcript
            )
            
            self.update_status_threadsafe("Transcription completed successfully!")
            self.update_status_threadsafe(f"Output saved to: {self.output_dir}")
            
            # Enable the generation buttons now that transcript is available
            self.meeting_minutes_btn.state(['!disabled'])
            self.key_figures_btn.state(['!disabled'])
            self.roadmap_btn.state(['!disabled'])
            self.todos_btn.state(['!disabled'])
            self.list_steps_btn.state(['!disabled'])
            self.priorities_btn.state(['!disabled'])
            self.followup_btn.state(['!disabled'])
            
        except Exception as e:
            self.update_status_threadsafe(f"Error: {str(e)}")
        finally:
            # Re-enable inputs
            self.transcribe_btn.state(['!disabled'])
        
    def collect_transcript(self, text):
        """Collect transcript text."""
        self.transcript += text + "\n"
        # Enable the copy button as transcript is being collected
        if not self.copy_btn.instate(['!disabled']):
            self.copy_btn.state(['!disabled'])
        
    def copy_transcript_to_clipboard(self):
        """Copy the transcript to the system clipboard."""
        try:
            if self.transcript.strip():
                self.root.clipboard_clear()
                self.root.clipboard_append(self.transcript)
                self.update_status_threadsafe("Transcript copied to clipboard.")
                print("Transcript copied to clipboard.")  # Debugging statement
            else:
                self.update_status_threadsafe("No transcript available to copy.")
                print("No transcript available to copy.")  # Debugging statement
        except Exception as e:
            self.update_status_threadsafe(f"Error copying transcript: {str(e)}")
            print(f"Error copying transcript: {str(e)}")  # Debugging statement
    
    def copy_meeting_minutes_to_clipboard(self):
        self.copy_section_to_clipboard("Meeting Minutes", self.meeting_minutes_html_content)
    
    def copy_key_figures_to_clipboard(self):
        self.copy_section_to_clipboard("Key Figures", self.key_figures_html_content)
    
    def copy_roadmap_to_clipboard(self):
        self.copy_section_to_clipboard("Roadmap", self.roadmap_html_content)
    
    def copy_todos_to_clipboard(self):
        self.copy_section_to_clipboard("To-do's", self.todos_html_content)
    
    def copy_list_steps_to_clipboard(self):
        self.copy_section_to_clipboard("List Steps", self.list_steps_html_content)
    
    def copy_priorities_to_clipboard(self):
        self.copy_section_to_clipboard("Priorities", self.priorities_html_content)
    
    def copy_followup_to_clipboard(self):
        """Copy the Follow-Up content to the system clipboard."""
        self.copy_section_to_clipboard("Follow-Up", self.followup_html_content)
    
    def generate_meeting_minutes(self):
        """Generate Meeting Minutes using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.meeting_minutes_btn.state(['disabled'])
        self.meeting_minutes_html.load_html("")  # Clear previous summary
        
        # Define the prompt with the transcript included
        prompt = (
            "Generate meeting minutes in the language of the transcript - please keep the language of the meeting transcript. "
            "The meeting minutes should follow the following structure:\n"
            "Summary of today's meeting:\n"
            "Bullet list of topics discussed, be brief but detailed:\n"
            "   - Keep it simple, avoid using people's names, just make a list of the topics that were discussed & add some detail whenever some is present in the meeting notes.\n"
            f"{self.transcript}"
        )
        
        # Start generation in a separate thread with the updated prompt
        threading.Thread(
            target=self.generate_section, 
            args=("Meeting Minutes", self.meeting_minutes_html, prompt), 
            daemon=True
        ).start()
        
    def generate_key_figures(self):
        """Generate Key Figures using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.key_figures_btn.state(['disabled'])
        self.key_figures_html.load_html("")  # Clear previous content
        
        # Template prompt for Key Figures
        prompt = "Extract the key figures from the following transcript, please keep the language of the meeting transcript:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("Key Figures", self.key_figures_html, prompt), daemon=True).start()
        
    def generate_roadmap(self):
        """Generate Roadmap using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.roadmap_btn.state(['disabled'])
        self.roadmap_html.load_html("")  # Clear previous content
        
        # Template prompt for Roadmap
        prompt = "Create a roadmap for my Odoo implementation project based on the following transcript, please keep the language of the meeting transcript:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("Roadmap", self.roadmap_html, prompt), daemon=True).start()
        
    def generate_todos(self):
        """Generate To-do's using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.todos_btn.state(['disabled'])
        self.todos_html.load_html("")  # Clear previous content
        
        # Template prompt for To-do's
        prompt = "List the to-do's from the following transcript, be brief, direct, but detailed, please keep the language of the meeting transcript:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("To-do's", self.todos_html, prompt), daemon=True).start()
        
    def generate_list_steps(self):
        """Generate List Steps using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.list_steps_btn.state(['disabled'])
        self.list_steps_html.load_html("")  # Clear previous content
        
        # Template prompt for List Steps
        prompt = "List all the steps described in the following text, in the form of a guide or tutorial:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("List Steps", self.list_steps_html, prompt), daemon=True).start()
        
    def generate_priorities(self):
        """Generate Priorities using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.priorities_btn.state(['disabled'])
        self.priorities_html.load_html("")  # Clear previous content
        
        # Template prompt for Priorities
        prompt = "Identify and list elements by prioritization ranking from the following transcript, group them by high priority, medium priority, low priority and nice-to-haves:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("Priorities", self.priorities_html, prompt), daemon=True).start()
        
    def generate_followup(self):
        """Generate Follow-Up items using the Gemini API."""
        # Disable the button to prevent multiple clicks
        self.followup_btn.state(['disabled'])
        self.followup_html.load_html("")  # Clear previous content
        
        # Template prompt for Follow-Up
        prompt = "Generate a list of items to follow-up on from the following transcript: open or unanswered questions, things that will require some more research, please keep the language of the meeting transcript:\n\n" + self.transcript
        threading.Thread(target=self.generate_section, args=("Follow-Up", self.followup_html, prompt), daemon=True).start()
        
    def generate_section(self, section_name, html_widget, prompt):
        """Generate a specific section using the Gemini API and update the corresponding HtmlFrame widget."""
        try:
            # Configure the Gemini API with the API key from config.yaml
            genai.configure(api_key=gemini_api_key)
            
            # Initialize the Gemini model
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Generate the content with streaming
            response = model.generate_content(prompt, stream=True)
            
            self.update_status_threadsafe(f"Generating {section_name.lower()}...")
            
            generated_content = ""
            html_content = ""
            
            # Stream the content and update the HtmlFrame widget
            for chunk in response:
                if hasattr(chunk, 'text'):
                    content_chunk = chunk.text
                    generated_content += content_chunk  # Append to the storage
                    html_content += content_chunk      # Accumulate HTML content
                    
                    # Convert markdown to HTML
                    html_converted = markdown.markdown(html_content)
                    
                    # Prepend the CSS styles to the HTML content
                    final_html = CSS_TEMPLATE + html_converted
                    
                    # Update HtmlFrame with new content
                    html_widget.load_html(final_html)
                else:
                    # Log if the chunk does not contain text
                    self.update_status_threadsafe(f"Received a chunk without text for {section_name}.")
            
            # After streaming is complete, assign the generated content to the respective variable
            if section_name == "Meeting Minutes":
                self.meeting_minutes_html_content = generated_content.strip()
                self.update_status_threadsafe("Meeting Minutes generated successfully.")
                self.copy_meeting_minutes_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "Key Figures":
                self.key_figures_html_content = generated_content.strip()
                self.update_status_threadsafe("Key Figures generated successfully.")
                self.copy_key_figures_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "Roadmap":
                self.roadmap_html_content = generated_content.strip()
                self.update_status_threadsafe("Roadmap generated successfully.")
                self.copy_roadmap_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "To-do's":
                self.todos_html_content = generated_content.strip()
                self.update_status_threadsafe("To-do's generated successfully.")
                self.copy_todos_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "List Steps":
                self.list_steps_html_content = generated_content.strip()
                self.update_status_threadsafe("List Steps generated successfully.")
                self.copy_list_steps_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "Priorities":
                self.priorities_html_content = generated_content.strip()
                self.update_status_threadsafe("Priorities generated successfully.")
                self.copy_priorities_btn.state(['!disabled'])  # Enable copy button
            elif section_name == "Follow-Up":
                self.followup_html_content = generated_content.strip()
                self.update_status_threadsafe("Follow-Up generated successfully.")
                self.copy_followup_btn.state(['!disabled'])  # Enable copy button
            
        except Exception as e:
            # Log any exceptions that occur during generation
            self.update_status_threadsafe(f"Error generating {section_name.lower()}: {str(e)}")
        finally:
            # Re-enable the button after completion
            if section_name == "Meeting Minutes":
                self.meeting_minutes_btn.state(['!disabled'])
            elif section_name == "Key Figures":
                self.key_figures_btn.state(['!disabled'])
            elif section_name == "Roadmap":
                self.roadmap_btn.state(['!disabled'])
            elif section_name == "To-do's":
                self.todos_btn.state(['!disabled'])
            elif section_name == "List Steps":
                self.list_steps_btn.state(['!disabled'])
            elif section_name == "Priorities":
                self.priorities_btn.state(['!disabled'])
            elif section_name == "Follow-Up":
                self.followup_btn.state(['!disabled'])
    
    def copy_section_to_clipboard(self, section_name, content):
        """Copy the specified section's content to the system clipboard."""
        try:
            if content.strip():
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self.update_status_threadsafe(f"{section_name} copied to clipboard.")
                print(f"{section_name} copied to clipboard.")  # Debugging statement
            else:
                self.update_status_threadsafe(f"No {section_name} available to copy.")
                print(f"No {section_name} available to copy.")  # Debugging statement
        except Exception as e:
            self.update_status_threadsafe(f"Error copying {section_name}: {str(e)}")
            print(f"Error copying {section_name}: {str(e)}")  # Debugging statement

# Modify the main block to support both CLI and GUI
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: Transcribe the provided audio path
        audio_path = sys.argv[1]
        transcribe_audio(audio_path)
    else:
        # GUI mode: Launch the transcription application
        root = tk.Tk()
        app = TranscriptionApp(root)
        root.mainloop()
