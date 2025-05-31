# Delphi-Bot: AI-Powered Expert Interview & Factors Catalog Generator 🤖

Welcome to the Delphi-Bot! This application utilizes a multi-agent AI system (powered by OpenAI's GPT models via the OpenAI Agents SDK) to simulate/conduct expert interviews, extract key insights, and compile them into a structured "Faktorenkatalog" (influence factor catalog). It's designed to support the Delphi method for foresight and strategic planning.

## 🌟 Features

*   **Versatile Study Setup:** Configure new Delphi studies with different topics, target years, and geographical scopes.
*   **Dual Interview Modes (Exploratory Phase):**
    *   **AI Persona Simulation:** The bot simulates interviews with AI-generated expert personas.
    *   **Human as Interviewee (Text Input):** Allows a real user to be the expert, answering questions from the AI Interviewer.
*   **Voice Interaction (for Human Interviewee Mode):**
    *   **Text-to-Speech (TTS):** AI Interviewer's questions can be spoken aloud (options: Google TTS, OpenAI TTS).
    *   **Speech-to-Text (STT):** Human users can dictate their answers (options: Google Web Speech).
*   **AI-Driven Structure Discovery:** In exploratory interviews, the AI Summarizer proposes a thematic structure (Systemebenen and Faktorname) based on the interview content.
*   **Human-in-the-Loop Refinement:** Users can review and edit the AI-proposed structure and AI-formalized interview guides.
*   **Structured Interview Phase:** Conducts focused interviews using the defined thematic structure with (currently) AI personas.
*   **Automated Catalog Generation:** AI agents synthesize insights from multiple interviews into a final, downloadable Faktorenkatalog in Markdown format.
*   **Token & Cost Tracking:** Provides an estimate of OpenAI API token usage and costs for each session.
*   **User-Friendly Web Interface:** Built with Streamlit for easy local interaction.

## 📋 Prerequisites

Before you begin, ensure you have the following installed and configured:

*   **Python:** Version 3.9 or higher (tested up to 3.12 recommended).
    *   👉 [Download Python](https://www.python.org/downloads/)
*   **OpenAI API Key:** **Required** for all AI agent functionality (both chat completions and OpenAI TTS/STT).
    *   👉 [OpenAI Platform](https://platform.openai.com/)

## ⚙️ Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Jsiekma/DelphiBot.git
    cd DelphiBot
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
    *(Your terminal prompt should change to indicate the active environment, e.g., `(venv)`)*

3.  **Install Dependencies:**
    Create a `requirements.txt` file in the root of your project with the following content:
    ```txt
    streamlit
    openai
    openai-agents
    tiktoken
    gTTS
    SpeechRecognition
    PyAudio
    ```
    Then, in your **activated virtual environment**, run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    The application requires your OpenAI API key. Set it as an environment variable named `OPENAI_API_KEY`.

    *   **Windows (Persistent - User Level, New Terminal Needed):**
        Open Command Prompt (cmd.exe) and run:
        ```cmd
        setx OPENAI_API_KEY "sk-YOUR_OPENAI_API_KEY_HERE"
        ```
        Then **close and reopen your terminal window** for the change to take effect.
    *   **Windows (PowerShell - Current Session Only):**
        ```powershell
        $env:OPENAI_API_KEY = "sk-YOUR_OPENAI_API_KEY_HERE"
        ```
    *   **Windows (Command Prompt - Current Session Only):**
        ```cmd
        set OPENAI_API_KEY=sk-YOUR_OPENAI_API_KEY_HERE
        ```
    *   **macOS / Linux (Bash/Zsh - Persistent):**
        Add the following line to your shell's profile file (e.g., `~/.bashrc`, `~/.zshrc`, `~/.profile`):
        ```bash
        export OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
        ```
        Then, either source the file (e.g., `source ~/.bashrc`) or open a new terminal window.
    *   **macOS / Linux (Bash/Zsh - Current Session Only):**
        ```bash
        export OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
        ```

## 🚀 Running the Application

1.  Ensure your virtual environment is **activated**.
2.  Ensure your `OPENAI_API_KEY` environment variable is set and accessible in your current terminal session.
3.  Navigate to the project's root directory (where `app.py` and `delphibot_engine.py` are located).
4.  Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
5.  The application will open in your default web browser.

## 📖 Using the App - Workflow

1.  **Configure Study (Sidebar):**
    *   Define the "Overall Study Topic," "Target Year," "Geographical Scope," "Key Objectives," and "Persona Requirements Guidance." The app loads with "Future of Daily Newspapers" as an example.
    *   Select the "Exploratory Interview Mode":
        *   `AI Persona Simulation`: The bot interviews an AI-generated expert.
        *   `Human as Interviewee (Text Input)`: You (or another user) act as the expert.
    *   If Human mode, enable "Voice Output" and/or "Voice Input" and select TTS/STT providers (Google or OpenAI).
    *   Set "Max Interview Turns" (per interview session) and the "Target # of Structured Interviews" for Phase 2.
    *   Click "**Set Study & Start New Run**" to initialize/reset with the new settings.

2.  **Phase 1: Exploratory Interview:**
    *   If Human mode and you wish to provide context about yourself, fill in the optional "Your Expert Profile" fields *before* starting the round.
    *   Click "**Run Exploratory Interview Round**."
    *   Interact by typing or speaking your answers (if Human mode and voice input enabled), or observe (if AI mode). The AI Interviewer's questions can be spoken aloud if enabled.
    *   Once the interview turns are complete, the AI Summarizer will propose a thematic structure. This appears under "AI-Proposed Thematic Structure (Review & Edit)."
    *   **Review and edit this summary directly in the text area.** Your edits are crucial for guiding the AI.
    *   Click "**Confirm Edited Summary & AI Formalize Guides**."

3.  **Phase 1.5: AI Formalizes Guides & User Review:**
    *   The AI Manager agent converts your (edited) summary into more formal "Interview Guide Structure" and "Catalog Output Guidance" strings.
    *   Review these AI-formalized guides. You can click "**Edit Guides ✒️**," make changes, and then "**✅ Save Edited Guides**" or "**❌ Cancel Edits**."
    *   Once satisfied, click "**➡️ Proceed w/ Current Guides**" to lock in these guides for the structured interview phase.

4.  **Phase 2: Structured Interview Round(s):**
    *   The finalized guides are displayed.
    *   Click "**Run Structured Interview #[X] (AI Persona)**" for each targeted interview. (Currently, structured interviews use AI personas; human mode for structured rounds is a future enhancement).
    *   Transcripts and AI summaries (now following the defined structure) for each structured interview will be displayed as they complete.

5.  **Phase 3: Final Catalog Generation:**
    *   Once all targeted structured interviews are complete, the "**Generate Final Faktorenkatalog**" button will become active (if summaries are available).
    *   Click it to have the AI `CatalogWriterAgent` attempt to synthesize all collected structured summaries into a final report.
    *   View the result and use the "**Download Faktorenkatalog (.md)**" button.

6.  **Metrics:** Token usage and estimated costs are updated in the "View Session Token Usage & Estimated Cost" expander in the main area after AI operations.
