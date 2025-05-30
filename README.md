# Delphi-Bot: AI-Powered Expert Interview & Factors Catalog Generator 🤖

Welcome to the Delphi-Bot! This application utilizes a multi-agent AI system (powered by OpenAI's GPT models via the `agents` SDK) to simulate/conduct expert interviews, extract key insights, and compile them into a structured "Faktorenkatalog" (influence factor catalog). It's designed to support the Delphi method for foresight and strategic planning.

## 🌟 Features

*   **Versatile Study Setup:** Configure new Delphi studies with different topics, target years, and geographical scopes.
*   **Dual Interview Modes (Exploratory Phase):**
    *   **AI Persona Simulation:** The bot simulates interviews with AI-generated expert personas.
    *   **Human as Interviewee (Text Input):** Allows a real user to be the expert, answering questions from the AI Interviewer.
*   **Voice Interaction (for Human Interviewee Mode):**
    *   **Text-to-Speech (TTS):** AI Interviewer's questions can be spoken aloud (options: Google TTS, OpenAI TTS).
    *   **Speech-to-Text (STT):** Human users can dictate their answers (options: Google Web Speech, OpenAI STT).
*   **AI-Driven Structure Discovery:** In exploratory interviews, the AI Summarizer proposes a thematic structure (Systemebenen and Faktorname) based on the interview content.
*   **Human-in-the-Loop Refinement:** Users can review and edit the AI-proposed structure and AI-formalized interview guides.
*   **Structured Interview Phase:** Conducts focused interviews using the defined thematic structure with (currently) AI personas.
*   **Automated Catalog Generation:** AI agents synthesize insights from multiple interviews into a final, downloadable Faktorenkatalog in Markdown format.
*   **Token & Cost Tracking:** Provides an estimate of OpenAI API token usage and costs for each session.
*   **User-Friendly Web Interface:** Built with Streamlit for easy local interaction.

## 📋 Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Python:** Version 3.9 or higher (tested up to 3.12).
    *   👉 [Download Python](https://www.python.org/downloads/)
*   **pip:** The Python package manager (usually comes with Python).
*   **Git:** For cloning the repository.
    *   👉 [Download Git](https://git-scm.com/downloads)
*   **OpenAI API Key:** **Required** for all AI agent functionality (both chat completions and OpenAI TTS/STT).
    *   👉 [OpenAI Platform](https://platform.openai.com/)
*   **(Optional for OpenAI TTS Playback) FFmpeg (with ffplay):** The `LocalAudioPlayer` used by the OpenAI TTS feature relies on `ffplay` being installed and accessible in your system's PATH for direct audio playback. If not found, OpenAI TTS will generate audio data but might not play it automatically, or `st.audio` could be used as a fallback if implemented.
    *   👉 [Download FFmpeg](https://ffmpeg.org/download.html) (ensure to add its `bin` directory to your PATH).

## ⚙️ Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone [URL_OF_YOUR_GITHUB_REPO]
    cd [REPO_NAME]
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    *   **Windows (Command Prompt/PowerShell):**
        ```cmd
        .\venv\Scripts\activate
        ```
    *   **macOS / Linux (Bash/Zsh):**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in the root of your project with the following content:
    ```txt
    streamlit
    openai
    agents-sdk # Or the specific name if it's just 'agents'
    tiktoken
    gTTS
    SpeechRecognition
    PyAudio
    # elevenlabs # Only if you re-add ElevenLabs support
    # requests # Only if you re-add direct ElevenLabs HTTP calls
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Troubleshooting `PyAudio`:** Installation can be tricky.
        *   Linux: `sudo apt-get install portaudio19-dev python3-pyaudio`
        *   macOS: `brew install portaudio && pip install pyaudio`
        *   Windows: May require "Microsoft Visual C++ Build Tools". Try `pip install pipwin` then `pipwin install pyaudio`.

4.  **Configure Environment Variables:**
    The application requires your OpenAI API key. Set it as an environment variable named `OPENAI_API_KEY`.
    *   **Windows (PowerShell - current session):**
        ```powershell
        $env:OPENAI_API_KEY = "sk-YOUR_OPENAI_API_KEY_HERE"
        ```
    *   **Windows (Command Prompt - current session):**
        ```cmd
        set OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY_HERE
        ```
    *   **macOS / Linux (Bash/Zsh - current session):**
        ```bash
        export OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
        ```
    *   For persistent setting, add it to your system's environment variables or your shell's profile file (e.g., `.bashrc`, `.zshrc`, PowerShell profile).
    *   **(Optional) ElevenLabs API Key:** If you re-enable ElevenLabs functionality, set `ELEVEN_API_KEY` similarly.

## 🚀 Running the Application

1.  Ensure your virtual environment is activated.
2.  Ensure your `OPENAI_API_KEY` (and other keys if used) environment variable is set in your current terminal session.
3.  Navigate to the project's root directory (where `app.py` is located).
4.  Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
5.  The application will open in your default web browser, typically at `http://localhost:8501`.

## 📖 Using the App - Workflow

1.  **Configure Study (Sidebar):**
    *   Define the "Overall Study Topic," "Target Year," "Geographical Scope," "Key Objectives," and "Persona Requirements Guidance."
    *   Select the "Exploratory Interview Mode":
        *   `AI Persona Simulation`: The bot interviews an AI-generated expert.
        *   `Human as Interviewee (Text Input)`: You act as the expert.
    *   If Human mode, enable Voice Output/Input and select TTS/STT providers as desired.
    *   Set "Max Interview Turns" and "Target # of Structured Interviews."
    *   Click "**Set Study & Start New Run**" to initialize or reset with new settings.

2.  **Phase 1: Exploratory Interview:**
    *   If Human mode, provide your optional expert profile details.
    *   Click "**Run Exploratory Interview Round**."
    *   Interact (if Human mode) or observe (if AI mode).
    *   Once done, the AI's proposed thematic structure appears under "AI-Proposed Thematic Structure (Review & Edit)."
    *   **Edit this summary directly in the text area.** This is your chance to guide the AI.
    *   Click "**Confirm Edited Summary & AI Formalize Guides**."

3.  **Phase 1.5: AI Formalizes Guides & User Review:**
    *   The AI converts your (edited) summary into formal "Interview Guide Structure" and "Catalog Output Guidance."
    *   Review these AI-formalized guides. You can "**Edit Guides ✒️**," make changes, and then "**✅ Save Edited Guides**."
    *   Once satisfied, click "**➡️ Proceed w/ Current Guides**."

4.  **Phase 2: Structured Interview Round(s):**
    *   The finalized guides are displayed.
    *   Click "**Run Structured Interview #[X] (AI Persona)**" for each targeted interview. (Human mode for structured rounds is a future enhancement).
    *   Transcripts and AI summaries for each structured interview will be displayed.

5.  **Phase 3: Final Catalog Generation:**
    *   Once all targeted structured interviews are complete, click "**Generate Final Faktorenkatalog**."
    *   The AI will attempt to synthesize all collected summaries into a final report.
    *   View the result and use the "**Download Faktorenkatalog (.md)**" button.

6.  **Metrics:** Token usage and estimated costs are updated in the "View Session Token Usage & Estimated Cost" expander in the main area.

## 🛠️ Project Structure
