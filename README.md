<div align="center">

# 🔥 Breaking-/-Cut 🔥

**The Ultimate AI-Powered Story-to-Video Engine**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/)

---

**Breaking-/-Cut** is a high-fidelity automated video generation platform that transforms simple text descriptions into cinematic storytelling experiences. By orchestrating LLMs for narrative, neural TTS for narration, and generative AI for visuals, it produces full-length videos complete with transitions and synchronized subtitles.

[Report Bug](https://github.com/Sumit444-commits/breakingcut/issues) · [Request Feature](https://github.com/Sumit444-commits/breakingcut/issues)

</div>

---

## 🚀 Key Features

*   **Generative Narrative Engine:** Utilizes `Gemini-2.0-Flash` to expand short ideas into structured, cinematic story scripts.
*   **Neural Text-to-Speech:** Integrated `Microsoft SpeechT5` with HifiGan vocoders to generate high-quality, expressive voiceovers for every sentence.
*   **AI Visual Synthesis:** Automatically generates photorealistic image prompts and visual assets using Gemini's image generation models.
*   **Cinematic Transitions:** Programmatic video assembly using `OpenCV` and `MoviePy`, featuring smooth fade transitions between AI-generated frames.
*   **Automated Transcription & Subtitling:** Powered by `AssemblyAI` to generate frame-perfect `.srt` files and overlay them with custom-styled typography.
*   **Containerized Architecture:** Fully Dockerized setup ensuring consistent environment parity across development and production.

---

## 🛠 Technology Stack

### Core Technologies
| Category | Tools & Libraries |
| :--- | :--- |
| **Frontend / UI** | Streamlit 1.43.2, Custom CSS |
| **Generative AI** | Google Gemini (Text & Image Gen), Hugging Face Transformers |
| **Audio Synthesis** | SpeechT5, SoundFile, Pydub, SpeechRecognition |
| **Video Engineering** | MoviePy 2.0.0.dev2, OpenCV (cv2) 4.11.0, Imageio |
| **NLP & Transcription** | AssemblyAI, NLTK, PySRT |
| **DevOps** | Docker, Dev Containers |

---

## 📂 Directory Structure

```ascii
breakingcut/
├── .devcontainer/           # Remote development configuration
├── Text_files/              # Persistent storage for generated assets
│   ├── images_prompts.txt   # AI-generated visual prompts
│   ├── story.txt            # Narrative text output
│   └── subtitle.srt         # Synchronized caption data
├── app.py                   # Main Streamlit application orchestration
├── Dockerfile               # Production-ready container specification
├── requirements.txt         # Python dependency manifest
├── packages.txt             # System-level dependencies (apt-get)
├── style.css                # Custom UI styling and branding
├── audio_duration.py        # Utility for audio temporal analysis
├── audios_to_audio.py       # Audio concatenation logic
├── i_to_v.py                # Image-to-Video assembly module
├── image_gen.py             # Image generation interface
└── add_subtitles.py         # Subtitle overlay engineering
```

---

## ⚙️ Getting Started

### Prerequisites

*   Python 3.11+
*   FFmpeg (required for video/audio processing)
*   Google Gemini API Key
*   AssemblyAI API Key

### Environment Setup

Create a `.streamlit/secrets.toml` file or set environment variables:

```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY"

[assemblyai]
api_key = "YOUR_ASSEMBLY_AI_KEY"
```

### Local Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sumit444-commits/breakingcut.git
    cd breakingcut
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

### Docker Deployment

```bash
docker build -t breakingcut .
docker run -p 8501:8501 breakingcut
```

---

## 👤 Author

**Sumit Sharma**
*   **GitHub:** [@Sumit444-commits](https://github.com/Sumit444-commits)
*   **Portfolio:** [sumit-portfolio.free.nf](https://sumit-portfolio.free.nf)
*   **LinkedIn:** [Sumit Sharma](https://www.linkedin.com/in/sumit-sharma-a0b2c7)
*   **Email:** sumit8444061@gmail.com

---

Designed with ❤️ Autome
