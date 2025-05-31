# app.py
import streamlit as st
import json
from delphibot_engine import (
    perform_study_phase,
    formalize_structure_from_exploratory_summary,
    generate_final_catalog_from_summaries,
    reset_session_tokens_for_engine,
    ManagerAgent, 
    InterviewerAgent,
    SummarizerAgent,
    _run_agent_internal,
    INPUT_PRICE_PER_MILLION_TOKENS,
    OUTPUT_PRICE_PER_MILLION_TOKENS,
    PREDEFINED_PERSONAS_NEWSPAPER_TOPIC, 
    MAX_INTERVIEW_TURNS_DEFAULT,
    session_input_tokens, 
    session_output_tokens
)
from typing import Any, Dict, List, Optional

# --- VOICE IMPORTS ---
from gtts import gTTS # For Google TTS
import speech_recognition as sr
import io 
import asyncio
from openai import AsyncOpenAI # For OpenAI TTS
from openai.helpers import LocalAudioPlayer # For OpenAI TTS playback
import os

# --- Initialize OpenAI Async Client for TTS (once per session) ---
if 'openai_async_client' not in st.session_state:
    try:
        st.session_state.openai_async_client = AsyncOpenAI() 
        print("ENGINE: AsyncOpenAI client for TTS initialized.")
    except Exception as e:
        st.session_state.openai_async_client = None
        print(f"ENGINE WARNING: Could not initialize AsyncOpenAI client for TTS: {e}")

# --- Initialize STT Recognizer and Microphone (once per session) ---
if 'recognizer' not in st.session_state: st.session_state.recognizer = sr.Recognizer()
if 'microphone' not in st.session_state:
    try: st.session_state.microphone = sr.Microphone(); print("ENGINE: Microphone initialized.")
    except Exception as e: st.session_state.microphone = None; print(f"ENGINE WARNING: Mic init failed: {e}.")

# --- ASYNC HELPER TO PLAY OPENAI TTS STREAM ---
async def play_openai_tts_stream_async(client: AsyncOpenAI, text: str, voice: str):
    if not client: st.warning("OpenAI TTS client not available."); return
    try:
        async with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice=voice, input=text, response_format="pcm",
        ) as response:
            print(f"ENGINE: OpenAI TTS - Streaming audio with voice '{voice}'...")
            await LocalAudioPlayer().play(response)
        print("ENGINE: OpenAI TTS - Speech finished.")
    except Exception as e: st.error(f"OpenAI TTS Streaming Error: {e}"); print(f"ENGINE ERROR: OpenAI TTS Streaming: {e}")

# --- UNIFIED TTS CONTROLLER FUNCTION ---
def speak_text_controller(text_to_speak: str):
    provider = st.session_state.get('tts_provider_selection', "Google TTS (free)") # Use a new key for provider selection
    if not st.session_state.get("enable_voice_output", False) or not text_to_speak: return

    print(f"ENGINE: TTS ({provider}) - Preparing: '{text_to_speak[:50]}...'")
    try:
        if provider == "Google TTS (free)":
            with st.spinner(f"gTTS generating audio..."):
                tts = gTTS(text=text_to_speak, lang='de', slow=False)
                audio_fp = io.BytesIO(); tts.write_to_fp(audio_fp); audio_fp.seek(0)
            st.audio(audio_fp, format="audio/mp3") 
            print(f"ENGINE: TTS (gTTS) - Audio player rendered.")
        
        elif provider == "OpenAI TTS":
            openai_client = st.session_state.get("openai_async_client")
            selected_openai_voice = st.session_state.get("openai_tts_voice_selection", "alloy")
            if openai_client:
                with st.spinner(f"OpenAI TTS generating audio..."):
                    asyncio.run(play_openai_tts_stream_async(openai_client, text_to_speak, selected_openai_voice))
            else:
                st.warning("OpenAI TTS client not initialized. Cannot play audio.")
                print("ENGINE ERROR: OpenAI TTS client not ready.")
        
        # Add ElevenLabs here if you re-integrate it, using a similar pattern
        # elif provider == "ElevenLabs":
            # ... elevenlabs logic ...

    except Exception as e: st.error(f"TTS Error ({provider}): {e}"); print(f"ENGINE ERROR: TTS failed ({provider}): {e}")


# --- STT Function (recognize_speech_from_mic_sr - keep as is) ---
def recognize_speech_from_mic_sr() -> str:
    # ... (Your existing STT function from the last complete app.py version)
    recognizer = st.session_state.get("recognizer")
    microphone = st.session_state.get("microphone")
    if not recognizer or not microphone: st.warning("STT components not ready."); return ""
    transcribed_text = ""
    with microphone as source:
        recognizer.pause_threshold = 1.5 
        st.toast("Adjusting for ambient noise...", icon="🤫")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("Listening... Speak clearly (up to ~25s). Pauses will end recording.", icon="🎤")
            print("ENGINE: STT Listening...")
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=25) 
        except sr.WaitTimeoutError: st.warning("No speech detected to start."); return ""
        except Exception as e: st.error(f"Mic error: {e}"); return ""
    if audio:
        try:
            print("ENGINE: STT Recognizing...")
            with st.spinner("Transcribing your speech..."):
                 transcribed_text = recognizer.recognize_google(audio, language="de-DE")
            st.success("Speech recognized!"); print(f"ENGINE: STT Recognized: '{transcribed_text}'")
        except sr.RequestError as e: st.error(f"Google API error; {e}")
        except sr.UnknownValueError: st.warning("Google Speech Recognition could not understand audio.")
    return transcribed_text

# --- Function for Speech-to-Text using OpenAI Whisper API ---
def recognize_speech_from_mic_openai() -> str:
    microphone = st.session_state.get("microphone")
    # Use the standard OpenAI client for STT
    openai_s2t_client = st.session_state.get("openai_client") 

    if not microphone:
        st.warning("Microphone not initialized for STT.")
        return ""
    if not openai_s2t_client:
        st.warning("OpenAI client not initialized. Cannot use OpenAI STT.")
        print("ENGINE ERROR: OpenAI client for STT is None.")
        return ""

    transcribed_text = ""
    recognizer = st.session_state.get("recognizer", sr.Recognizer()) # Still use sr for mic access
    
    with microphone as source:
        recognizer.pause_threshold = 1.5 
        st.toast("Adjusting for ambient noise...", icon="🤫")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            st.toast("Listening for OpenAI STT... Speak clearly.", icon="🎤")
            print("ENGINE: STT (OpenAI) Listening...")
            # Capture audio for a reasonable duration for OpenAI STT
            # The API has a 25MB file limit. Short recordings are fine.
            audio_data_sr = recognizer.listen(source, timeout=10, phrase_time_limit=30) # Allow longer speech
        except sr.WaitTimeoutError: st.warning("No speech detected to start."); return ""
        except Exception as e: st.error(f"Mic error: {e}"); return ""
            
    if audio_data_sr:
        try:
            print("ENGINE: STT (OpenAI) Transcribing...")
            with st.spinner("OpenAI transcribing your speech..."):
                # Get audio bytes in WAV format (OpenAI STT supports wav)
                audio_bytes_wav = audio_data_sr.get_wav_data()
                
                # The OpenAI API expects a file-like object.
                # We need to create one from the bytes.
                audio_file_like = io.BytesIO(audio_bytes_wav)
                # OpenAI SDK needs the file to have a name when passed as a tuple for multipart form data
                # (filename, file_object, content_type)
                # For passing directly to client.audio.transcriptions.create, it might infer or not need content_type
                
                # Using the client.audio.transcriptions.create method
                transcription = openai_s2t_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", # Or "gpt-4o-transcribe" or "whisper-1"
                    file=audio_file_like, # Pass the BytesIO object
                    # file=("audio.wav", audio_file_like, "audio/wav"), # More explicit for some SDK versions/methods
                    language="de", # Optional: can help if language is known
                    response_format="text" # Get plain text directly
                )
                # The result for response_format="text" is directly the string
                transcribed_text = transcription if isinstance(transcription, str) else ""

            if transcribed_text:
                st.success("OpenAI STT: Speech recognized!")
                print(f"ENGINE: STT (OpenAI) Recognized: '{transcribed_text}'")
            else:
                st.warning("OpenAI STT returned no text or failed.")

        except Exception as e: 
            st.error(f"Error during OpenAI STT processing: {e}")
            print(f"ENGINE ERROR: OpenAI STT Processing: {e}")
    
    return transcribed_text

# --- Streamlit Page Configuration & Session State Initialization (Comprehensive) ---
# (This block is the same as your last one, ensure all keys are present)
if 'run_id' not in st.session_state: st.session_state.run_id = 0 
if 'current_phase' not in st.session_state: st.session_state.current_phase = "initial_setup" 
# ... (ALL other session state initializations from your last full app.py) ...
# ADD/Ensure these for TTS provider selection:
if 'tts_provider_selection' not in st.session_state: st.session_state.tts_provider_selection = "Google TTS (free)"
if 'openai_tts_voice_selection' not in st.session_state: st.session_state.openai_tts_voice_selection = "alloy"
# Typo fix for formalized guides from a previous version
if 'ai_formalized_interview_guide' not in st.session_state: st.session_state.ai_formalized_interview_guide = ""
if 'ai_formalized_catalog_guide' not in st.session_state: st.session_state.ai_formalized_catalog_guide = ""
if 'exploratory_transcript' not in st.session_state: st.session_state.exploratory_transcript = []
if 'selected_persona_expl_dict' not in st.session_state: st.session_state.selected_persona_expl_dict = {}
if 'selected_persona_name_expl' not in st.session_state: st.session_state.selected_persona_name_expl = "N/A"
if 'exploratory_summary_proposed_structure' not in st.session_state: st.session_state.exploratory_summary_proposed_structure = ""
if 'user_confirmed_edited_exploratory_summary' not in st.session_state: st.session_state.user_confirmed_edited_exploratory_summary = ""
if 'exploratory_interview_turn_count' not in st.session_state: st.session_state.exploratory_interview_turn_count = 0
if 'current_interviewer_question' not in st.session_state: st.session_state.current_interviewer_question = ""
if 'human_answer_input' not in st.session_state: st.session_state.human_answer_input = ""
if 'human_expert_name_title_input' not in st.session_state: st.session_state.human_expert_name_title_input = ""
if 'human_expert_role_input' not in st.session_state: st.session_state.human_expert_role_input = ""
if 'human_expert_expertise_input' not in st.session_state: st.session_state.human_expert_expertise_input = ""
if 'human_expert_perspective_input' not in st.session_state: st.session_state.human_expert_perspective_input = ""
if 'user_edited_interview_guide' not in st.session_state: st.session_state.user_edited_interview_guide = ""
if 'user_edited_catalog_guide' not in st.session_state: st.session_state.user_edited_catalog_guide = ""
if 'editing_formalized_guides' not in st.session_state: st.session_state.editing_formalized_guides = False
if 'num_structured_interviews_target' not in st.session_state: st.session_state.num_structured_interviews_target = 1 
if 'structured_interview_results_list' not in st.session_state: st.session_state.structured_interview_results_list = []
if 'personas_used_in_study' not in st.session_state: st.session_state.personas_used_in_study = []
if 'final_catalog_output' not in st.session_state: st.session_state.final_catalog_output = ""
if 'tokens_input' not in st.session_state: st.session_state.tokens_input = 0
if 'tokens_output' not in st.session_state: st.session_state.tokens_output = 0
if 'error_message' not in st.session_state: st.session_state.error_message = None
if 'max_turns_per_interview_gui' not in st.session_state: st.session_state.max_turns_per_interview_gui = MAX_INTERVIEW_TURNS_DEFAULT 
if 'metrics_expanded' not in st.session_state: st.session_state.metrics_expanded = True 
if 'enable_voice_output' not in st.session_state: st.session_state.enable_voice_output = False
if 'enable_voice_input' not in st.session_state: st.session_state.enable_voice_input = False
if 'study_context' not in st.session_state: 
    st.session_state.study_context = {}

if 'openai_client' not in st.session_state: # General OpenAI client
    try:
        st.session_state.openai_client = OpenAI() # Uses OPENAI_API_KEY env var
        print("ENGINE: Standard OpenAI client initialized.")
    except Exception as e:
        st.session_state.openai_client = None
        print(f"ENGINE WARNING: Could not initialize standard OpenAI client: {e}")


# --- Callback function for "Submit My Answer" button ---
# (This is your existing, corrected callback from the previous version)
def process_human_answer_and_advance():
    if st.session_state.human_answer_input.strip():
        st.session_state.exploratory_transcript.append({
            "question": st.session_state.current_interviewer_question,
            "answer": st.session_state.human_answer_input.strip()
        })
        st.session_state.exploratory_interview_turn_count += 1
        st.session_state.current_interviewer_question = "" 
        if st.session_state.exploratory_interview_turn_count < st.session_state.max_turns_per_interview_gui:
            st.session_state.current_phase = "exploratory_human_awaits_question"
        else: 
            st.session_state.current_phase = "exploratory_processing_human_transcript"
        st.session_state.human_answer_input = "" 
    else: pass


# --- HELPER FUNCTION TO DISPLAY METRICS ---
# (This is your existing function)
def display_token_cost_metrics():
    if st.session_state.study_context.get("OverallStudyTopic"):
        with st.expander("View Session Token Usage & Estimated Cost", expanded=st.session_state.metrics_expanded):
            if st.session_state.tokens_input == 0 and st.session_state.tokens_output == 0:
                st.caption("No tokens used yet in this run/phase.")
            else:
                cost_input = (st.session_state.tokens_input / 1_000_000) * INPUT_PRICE_PER_MILLION_TOKENS
                cost_output = (st.session_state.tokens_output / 1_000_000) * OUTPUT_PRICE_PER_MILLION_TOKENS
                total_cost_usd = cost_input + cost_output
                usd_to_eur_rate = 0.88 
                total_cost_eur = total_cost_usd * usd_to_eur_rate
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(label="Input Tokens", value=f"{st.session_state.tokens_input:,}")
                with col2: st.metric(label="Output Tokens", value=f"{st.session_state.tokens_output:,}")
                with col3: st.metric(label="Total Est. Cost (EUR)", value=f"€{total_cost_eur:.5f}")

# --- Default Detailed Values ---
# (Your DEFAULT_... constants)
DEFAULT_NEWSPAPER_TOPIC = "Die Zukunft der Tageszeitung in Deutschland bis 2047"
DEFAULT_NEWSPAPER_TARGET_YEAR = 2047
DEFAULT_NEWSPAPER_GEO_SCOPE = "Deutschland"
DEFAULT_NEWSPAPER_OBJECTIVES = "Identifizierung und Strukturierung der Schlüsselfaktoren, Herausforderungen, Chancen und Trends, welche die Entwicklung der Tageszeitung in Deutschland bis 2047 maßgeblich beeinflussen werden, um strategische Implikationen für Verlage und Medienhäuser abzuleiten."
DEFAULT_NEWSPAPER_PERSONA_REQS = """Benötigt werden Experten-Personas mit vielfältigem Hintergrund relevant zur Medienlandschaft und Printmedien (z.B. Journalisten, Medienwissenschaftler, Verlagsmanager, Digitalexperten, Leservertreter, junge Mediennutzer), die Einsichten zu technologischem Wandel, Lesegewohnheiten, Geschäftsmodellen, journalistischer Qualität und gesellschaftlicher Rolle bis 2047 bieten können."""

# --- Sidebar ---
with st.sidebar:
    st.header("Delphi Study Configuration")
    # ... (Study Definition Inputs - topic, target_year, etc. - as before) ...
    topic_val = st.session_state.study_context.get("OverallStudyTopic", DEFAULT_NEWSPAPER_TOPIC)
    topic = st.text_input("Overall Study Topic:", value=topic_val, key=f"topic_input_sidebar_{st.session_state.run_id}")
    target_year_val = st.session_state.study_context.get("TargetYear", DEFAULT_NEWSPAPER_TARGET_YEAR)
    target_year = st.number_input("Target Year:", min_value=2025, max_value=2100, value=int(target_year_val), key=f"year_input_sidebar_{st.session_state.run_id}")
    geo_scope_val = st.session_state.study_context.get("GeographicalScope", DEFAULT_NEWSPAPER_GEO_SCOPE)
    geo_scope = st.text_input("Geographical Scope:", value=geo_scope_val, key=f"geo_input_sidebar_{st.session_state.run_id}")
    objectives_val = st.session_state.study_context.get("KeyObjectives_Wofuer", DEFAULT_NEWSPAPER_OBJECTIVES)
    objectives = st.text_area("Key Objectives (Wofür?):", value=objectives_val, height=150, key=f"obj_input_sidebar_{st.session_state.run_id}")
    persona_reqs_val = st.session_state.study_context.get("PersonaRequirementsGuidance", DEFAULT_NEWSPAPER_PERSONA_REQS)
    persona_reqs = st.text_area("Persona Requirements Guidance:", value=persona_reqs_val, height=200, key=f"persona_req_input_sidebar_{st.session_state.run_id}")

    st.markdown("---")
    st.radio("Exploratory Interview Mode:", options=["AI Persona Simulation", "Human as Interviewee (Text Input)"], key="interview_mode")
    
    if st.session_state.interview_mode == "Human as Interviewee (Text Input)":
        st.checkbox("Enable Voice Output (Interviewer AI speaks)", key="enable_voice_output")
        if st.session_state.enable_voice_output:
            # TTS Provider Selection
            tts_options = ["Google TTS (free)"]
            if st.session_state.get("openai_async_client"): tts_options.append("OpenAI TTS")
            # if st.session_state.get("elevenlabs_client"): tts_options.append("ElevenLabs") # If you add 11L TTS back

            current_tts_provider = st.session_state.get("tts_provider_selection", tts_options[0])
            if current_tts_provider not in tts_options: current_tts_provider = tts_options[0]
            try: current_tts_idx = tts_options.index(current_tts_provider)
            except ValueError: current_tts_idx = 0
            
            st.selectbox("TTS Voice Provider:", options=tts_options, index=current_tts_idx, key="tts_provider_selection")

            if st.session_state.tts_provider_selection == "OpenAI TTS" and st.session_state.get("openai_async_client"):
                openai_voice_options = ["alloy", "ash", "echo", "fable", "nova", "onyx", "shimmer"]
                st.selectbox("OpenAI TTS Voice:",options=openai_voice_options,key="openai_tts_voice_selection")
            # elif st.session_state.tts_provider_selection == "ElevenLabs":
            #     st.text_input("ElevenLabs Voice ID:", key="elevenlabs_voice_id_input")
        
        if st.session_state.get("microphone"):
            st.checkbox("Enable Voice Input (Record your answer)", key="enable_voice_input")
            if st.session_state.enable_voice_input:
                # --- NEW STT PROVIDER SELECTION ---
                stt_options = ["Google Web Speech"]
                if st.session_state.get("openai_client"): # Check for the standard OpenAI client
                    stt_options.append("OpenAI STT (Whisper based)")
                # Add ElevenLabs STT option if you implement it and have its client
                # if st.session_state.get("elevenlabs_client_for_stt"): # or just elevenlabs_client if it handles both
                #     stt_options.append("ElevenLabs STT")

                current_stt_provider = st.session_state.get("stt_provider", stt_options[0])
                if current_stt_provider not in stt_options: current_stt_provider = stt_options[0]
                try: current_stt_idx = stt_options.index(current_stt_provider)
                except ValueError: current_stt_idx = 0; 
                
                st.selectbox(
                    "STT Provider:", 
                    options=stt_options, 
                    index=current_stt_idx,
                    key="stt_provider" # This session state variable will store the choice
                )
                if st.session_state.stt_provider == "OpenAI STT":
                    st.caption("Using OpenAI for Speech-to-Text.")
                # --- END STT PROVIDER SELECTION ---
        else:
            st.caption("🎤 Voice input disabled (Mic issue).")
            st.session_state.enable_voice_input = False
            
    st.markdown("---")
    st.slider("Max Interview Turns (per interview):", min_value=1, max_value=10, key="max_turns_per_interview_gui")
    st.number_input("Target # of Structured Interviews:", min_value=1, max_value=10, step=1, key="num_structured_interviews_target")

    if st.button("Set Study & Start New Run", key="update_settings_btn"):
        st.session_state.run_id += 1 
        st.session_state.study_context = {
            "OverallStudyTopic": topic, "TargetYear": int(target_year), "GeographicalScope": geo_scope,
            "KeyObjectives_Wofuer": objectives, "PersonaRequirementsGuidance": persona_reqs,
            "PredefinedPersonas": PREDEFINED_PERSONAS_NEWSPAPER_TOPIC, 
            "InterviewGuideExploratoryPrompt": "Conduct an open-ended, exploratory interview on the OverallStudyTopic...",
            "SummarizerGuidanceExploratory": "This is an initial exploratory interview for the StudyTopic. Analyze the transcript to identify 4-6 MAJOR THEMATIC CATEGORIES...",
            "InterviewGuideStructure_DEFINED": None, "DesiredOutputCatalogStructureGuidance_DEFINED": None 
        }
        keys_to_reset_to_empty_list = ['exploratory_transcript', 'structured_interview_results_list', 'personas_used_in_study']
        keys_to_reset_to_empty_string = ['selected_persona_name_expl', 'exploratory_summary_proposed_structure', 
                                         'user_confirmed_edited_exploratory_summary', 'current_interviewer_question', 'human_answer_input', 
                                         'human_expert_name_title_input', 'human_expert_role_input', 'human_expert_expertise_input',
                                         'human_expert_perspective_input', 'ai_formalized_interview_guide', 'ai_formalized_catalog_guide', 
                                         'user_edited_interview_guide', 'user_edited_catalog_guide', 'final_catalog_output']
        for key_to_reset in keys_to_reset_to_empty_list: st.session_state[key_to_reset] = []
        for key_to_reset in keys_to_reset_to_empty_string: st.session_state[key_to_reset] = ""
        st.session_state.selected_persona_expl_dict = {}; st.session_state.exploratory_interview_turn_count = 0
        st.session_state.editing_formalized_guides = False; st.session_state.tokens_input = 0
        st.session_state.tokens_output = 0; st.session_state.error_message = None
        st.session_state.current_phase = "initial_setup" 
        st.session_state.question_just_spoken = False
        st.success("Study settings updated. Ready for new run."); st.rerun()

# --- Main Area ---

display_token_cost_metrics()
if not st.session_state.study_context.get("OverallStudyTopic"):
    st.info("👈 Configure study settings and click 'Set Study & Start New Run'."); st.stop()
st.markdown(f"**Current Exploratory Interview Mode:** `{st.session_state.interview_mode}`")
if st.session_state.interview_mode == "Human as Interviewee (Text Input)":
    voice_output_status = 'Enabled' if st.session_state.enable_voice_output else 'Disabled'
    mic_ok = st.session_state.get("microphone") is not None
    voice_input_status = 'Enabled' if st.session_state.enable_voice_input and mic_ok else 'Disabled'
    if not mic_ok and st.session_state.enable_voice_input: voice_input_status += " (Mic Error)"
    st.markdown(f"**Voice Output:** `{voice_output_status}` | **Voice Input:** `{voice_input_status}`")
st.markdown("---")

st.header("Phase 1: Exploratory Interview & AI Structure Proposal")
if st.session_state.current_phase == "initial_setup":
    if st.session_state.interview_mode == "Human as Interviewee (Text Input)":
        with st.expander("Your Expert Profile (Optional)", expanded=True):
            st.text_input("Your Name/Title:", key="human_expert_name_title_input")
            st.text_input("Your Role:", key="human_expert_role_input")
            st.text_area("Your Key Expertise Areas:", height=100, key="human_expert_expertise_input")
            st.text_area("Your general perspective:", height=100, key="human_expert_perspective_input")
        st.markdown("---") 
    if st.button("Run Exploratory Interview Round", key=f"start_expl_btn_{st.session_state.run_id}"):
        reset_session_tokens_for_engine(); st.session_state.exploratory_transcript = []
        st.session_state.exploratory_summary_proposed_structure = ""; st.session_state.error_message = None
        st.session_state.exploratory_interview_turn_count = 0; st.session_state.current_interviewer_question = ""
        st.session_state.selected_persona_expl_dict = {}; st.session_state.selected_persona_name_expl = "N/A"
        st.session_state.personas_used_in_study = []
        if st.session_state.interview_mode == "AI Persona Simulation": st.session_state.current_phase = "exploratory_running_ai"; st.rerun() 
        elif st.session_state.interview_mode == "Human as Interviewee (Text Input)":
            name_to_use = st.session_state.human_expert_name_title_input if st.session_state.human_expert_name_title_input else "Human Expert (You)"
            st.session_state.selected_persona_name_expl = name_to_use
            st.session_state.current_phase = "exploratory_human_awaits_question"; st.rerun()

if st.session_state.current_phase == "exploratory_running_ai":
    # ... (Paste your existing AI exploratory running logic) ...
    with st.spinner("Running AI exploratory interview and summarization..."):
        results = perform_study_phase(st.session_state.study_context.copy(), True, st.session_state.max_turns_per_interview_gui)
    st.session_state.exploratory_transcript = results.get("transcript", [])
    st.session_state.exploratory_summary_proposed_structure = results.get("summary", "") 
    st.session_state.user_confirmed_edited_exploratory_summary = st.session_state.exploratory_summary_proposed_structure 
    st.session_state.selected_persona_name_expl = results.get("selected_persona_name", "N/A")
    st.session_state.selected_persona_expl_dict = results.get("selected_persona_dict", {})
    if results.get("selected_persona_dict"): st.session_state.personas_used_in_study.append(results.get("selected_persona_dict"))
    st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
    st.session_state.error_message = results.get("error_message")
    if st.session_state.error_message: st.error(f"Error: {st.session_state.error_message}"); st.session_state.current_phase = "initial_setup"
    else: st.session_state.current_phase = "exploratory_done"; st.success("AI Exploratory round complete!")
    st.rerun() 


if st.session_state.current_phase == "exploratory_human_awaits_question":
    # ... (Paste your existing human_awaits_question logic, calling speak_text_controller after getting question) ...
    st.subheader(f"Exploratory Interview (Expert: {st.session_state.selected_persona_name_expl})")
    st.markdown(f"Turn {st.session_state.exploratory_interview_turn_count + 1} of {st.session_state.max_turns_per_interview_gui}")
    if st.session_state.exploratory_interview_turn_count < st.session_state.max_turns_per_interview_gui:
        if not st.session_state.current_interviewer_question: 
            with st.spinner("Interviewer AI is formulating a question..."):
                human_profile_text_for_prompt = "Human Expert Profile:\n"
                if st.session_state.human_expert_name_title_input: human_profile_text_for_prompt += f"- Name/Title: {st.session_state.human_expert_name_title_input}\n"
                if st.session_state.human_expert_role_input: human_profile_text_for_prompt += f"- Role: {st.session_state.human_expert_role_input}\n"
                if st.session_state.human_expert_expertise_input: human_profile_text_for_prompt += f"- Stated Expertise: {st.session_state.human_expert_expertise_input}\n"
                if st.session_state.human_expert_perspective_input: human_profile_text_for_prompt += f"- Stated Perspective: {st.session_state.human_expert_perspective_input}\n"
                if human_profile_text_for_prompt == "Human Expert Profile:\n": human_profile_text_for_prompt = "Human expert has not provided a specific profile.\n"
                interviewer_prompt = (
                    f"OverallStudyTopic: {st.session_state.study_context['OverallStudyTopic']}\nTargetYear: {st.session_state.study_context['TargetYear']}\n"
                    f"ConversationHistory: {json.dumps(st.session_state.exploratory_transcript, indent=2, ensure_ascii=False)}\n"
                    f"You are conducting an 'exploratory_interview' with a human expert. {human_profile_text_for_prompt}"
                    f"Your general guidance is: \"{st.session_state.study_context.get('InterviewGuideExploratoryPrompt','')}\"\n"
                    f"Based on the history and profile, what is your next question? Output ONLY the question."
                )
                interviewer_response_obj = _run_agent_internal(InterviewerAgent, interviewer_prompt) 
                st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
                if interviewer_response_obj and interviewer_response_obj.final_output:
                    st.session_state.current_interviewer_question = interviewer_response_obj.final_output.strip()
                    st.session_state.current_phase = "human_providing_answer_exploratory" 
                else: st.error("Interviewer AI failed to generate question."); st.session_state.current_phase = "initial_setup"
                st.rerun() 
    else: st.info("Max turns reached. Processing transcript..."); st.session_state.current_phase = "exploratory_processing_human_transcript"; st.rerun()

if st.session_state.current_phase == "human_providing_answer_exploratory":
    st.subheader(f"Exploratory Interview (Expert: {st.session_state.selected_persona_name_expl})")
    st.markdown(f"Turn {st.session_state.exploratory_interview_turn_count + 1} of {st.session_state.max_turns_per_interview_gui}")

    # This condition is key: Only proceed if there's an actual question to answer
    if st.session_state.current_interviewer_question:
        st.markdown(f"**Interviewer AI asks:**")
        st.info(st.session_state.current_interviewer_question) # Display the question text
        
        # Speak only if voice output is enabled AND this specific question hasn't been spoken yet
        if st.session_state.get("enable_voice_output", False) and not st.session_state.get("question_just_spoken", False):
            speak_text_controller(st.session_state.current_interviewer_question)
            st.session_state.question_just_spoken = True 
            # We might not need an immediate rerun here if st.audio for gTTS or 11L stream() doesn't block badly
            # or if the UI updates sufficiently without it. Test this.

        st.markdown("---") # Visual separator before answer area

        # --- STT Button and Text Area for human answer ---
        if st.session_state.get("enable_voice_input", False) and st.session_state.get("microphone"):
            if st.button("🎤 Record Answer", key=f"record_btn_{st.session_state.run_id}_{st.session_state.exploratory_interview_turn_count}"):
                with st.spinner("Listening for your answer..."):
                    transcribed_text = ""
                    if st.session_state.stt_provider == "OpenAI STT (Whisper based)" and st.session_state.get("openai_client"):
                        transcribed_text = recognize_speech_from_mic_openai()
                    else: # Default to Google
                        if st.session_state.stt_provider != "Google Web Speech":
                             st.warning(f"STT Provider '{st.session_state.stt_provider}' selected but not ready, falling back to Google.")
                        transcribed_text = recognize_speech_from_mic_sr()
                if transcribed_text:
                    st.session_state.human_answer_input = transcribed_text # Update the session state for the text_area
                    st.rerun() # Rerun to populate the text_area with transcribed text
        
        # Text area for answer is ALWAYS present in this phase if there's a question
        st.text_area("Your Answer (type, or record then edit):", height=150, key="human_answer_input")
        
        # Submit button with on_click callback
        if st.button("Submit My Answer", 
                     key=f"submit_human_ans_btn_{st.session_state.run_id}_{st.session_state.exploratory_interview_turn_count}",
                     on_click=process_human_answer_and_advance): 
            # This block executes AFTER the on_click callback (if one was triggered).
            # The callback is responsible for changing the phase.
            # If human_answer_input was empty, callback might not have changed phase.
            if not st.session_state.get("human_answer_input","").strip() and st.session_state.current_phase == "human_providing_answer_exploratory":
                 st.warning("Please provide an answer before submitting.")
            # If the callback DID process an answer and changed the phase, Streamlit's rerun (triggered by state change)
            # will take care of moving to the next UI state. No explicit rerun here is needed.
            
    else: 
        # This case means current_phase is "human_providing_answer_exploratory" but there's no question.
        # This implies the "exploratory_human_awaits_question" phase failed to set one.
        st.warning("Waiting for Interviewer AI's question... Attempting to fetch/re-fetch.")
        # To ensure we go back to fetching, we can reset current_interviewer_question and trigger the awaits_question phase
        if st.button("Retry Fetching Question", key=f"retry_fetch_q_in_provide_answer_{st.session_state.run_id}"):
            st.session_state.current_interviewer_question = "" 
            st.session_state.question_just_spoken = False # Allow TTS for new question
            st.session_state.current_phase = "exploratory_human_awaits_question"
            st.rerun()


if st.session_state.current_phase == "exploratory_processing_human_transcript":
    # ... (Paste your existing logic for this phase) ...
    if st.session_state.exploratory_transcript:
        st.subheader("Processing Your Exploratory Interview...");
        with st.spinner("Summarizer AI is proposing a structure based on your interview..."):
            persona_name_for_summary = st.session_state.selected_persona_name_expl
            prompt_for_manager_s5 = (
                f"Current Study Context:\n{json.dumps(st.session_state.study_context, indent=2, ensure_ascii=False)}\n"
                f"Exploratory Interview Transcript (with {persona_name_for_summary}):\n{json.dumps(st.session_state.exploratory_transcript, indent=2, ensure_ascii=False)}\n\n"
                f"This was an EXPLORATORY interview. Instruct SummarizerAgent to perform an 'exploratory_summary' using 'SummarizerGuidanceExploratory'. Output ONLY this instruction."
            )
            manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_s5)
            st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
            if manager_response_obj and manager_response_obj.final_output:
                instruction_for_summarizer = manager_response_obj.final_output
                full_prompt_for_summarizer_human = (
                    f"{instruction_for_summarizer}\n\n"
                    f"OverallStudyTopic: {st.session_state.study_context.get('OverallStudyTopic')}\nTargetYear: {st.session_state.study_context.get('TargetYear')}\n"
                    f"Guidance: {st.session_state.study_context.get('SummarizerGuidanceExploratory')}\n\n"
                    f"Interview Transcript to Summarize:\n```json\n{json.dumps(st.session_state.exploratory_transcript, indent=2, ensure_ascii=False)}\n```\nPlease provide summary."
                )
                summarizer_response_obj = _run_agent_internal(SummarizerAgent, full_prompt_for_summarizer_human)
                st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
                if summarizer_response_obj and summarizer_response_obj.final_output:
                    st.session_state.exploratory_summary_proposed_structure = summarizer_response_obj.final_output
                    st.session_state.user_confirmed_edited_exploratory_summary = st.session_state.exploratory_summary_proposed_structure
                    st.session_state.current_phase = "exploratory_done"; st.success("Summary from your interview ready!")
                else: st.error("Summarizer AI failed."); st.session_state.current_phase = "initial_setup"
            else: st.error("Manager failed to instruct Summarizer."); st.session_state.current_phase = "initial_setup"
        st.rerun()

# --- Display Exploratory Results & Confirm/Formalize Structure ---
# (This block is the same as your last complete version)
if st.session_state.current_phase in ["exploratory_done", "structure_formalizing", "structure_review_edit", "structure_confirmed_for_structured_rounds"]:
    if st.session_state.current_phase not in ["exploratory_running_ai", "exploratory_human_awaits_question", "human_providing_answer_exploratory", "exploratory_processing_human_transcript"]:
        if st.session_state.exploratory_transcript:
            st.subheader(f"Exploratory Interview Transcript (Interviewee: {st.session_state.selected_persona_name_expl})")
            if st.session_state.interview_mode == "AI Persona Simulation" and isinstance(st.session_state.selected_persona_expl_dict, dict) and st.session_state.selected_persona_expl_dict:
                 with st.popover("View AI Persona Details"): st.json(st.session_state.selected_persona_expl_dict)
            elif st.session_state.interview_mode == "Human as Interviewee (Text Input)":
                human_details_provided = any([st.session_state.human_expert_name_title_input, st.session_state.human_expert_role_input, st.session_state.human_expert_expertise_input, st.session_state.human_expert_perspective_input])
                if human_details_provided:
                    with st.popover("View Your Provided Expert Details"):
                        if st.session_state.human_expert_name_title_input: st.markdown(f"**Name/Title:** {st.session_state.human_expert_name_title_input}")
                        if st.session_state.human_expert_role_input: st.markdown(f"**Role:** {st.session_state.human_expert_role_input}")
                        if st.session_state.human_expert_expertise_input: st.markdown(f"**Expertise:** {st.session_state.human_expert_expertise_input}")
                        if st.session_state.human_expert_perspective_input: st.markdown(f"**Perspective:** {st.session_state.human_expert_perspective_input}")
            with st.expander("View Full Exploratory Transcript", expanded=False):
                for i, qa in enumerate(st.session_state.exploratory_transcript):
                    if "event" in qa: st.caption(f"Event: {qa['event']}")
                    else: st.markdown(f"**Q{i+1}:** {qa['question']}\n\n**A{i+1}:** {qa['answer']}")
            st.markdown("---")
    if st.session_state.current_phase == "exploratory_done" or st.session_state.current_phase == "structure_formalizing": 
        if st.session_state.exploratory_summary_proposed_structure:
            st.subheader("AI-Proposed Thematic Structure (Review & Edit)")
            st.session_state.user_confirmed_edited_exploratory_summary = st.text_area(
                "Edit the AI's proposed structure/summary below...", value=st.session_state.exploratory_summary_proposed_structure, 
                height=300, key=f"user_confirmed_edited_exploratory_summary_key_{st.session_state.run_id}")
            if st.button("Confirm Edited Summary & AI Formalize Guides", key=f"confirm_expl_summary_btn_{st.session_state.run_id}"):
                st.session_state.current_phase = "structure_formalizing"; st.rerun()
    if st.session_state.current_phase == "structure_formalizing":
        with st.spinner("AI is formalizing the guides based on your confirmed summary..."):
            formalized_guides = formalize_structure_from_exploratory_summary(st.session_state.study_context, st.session_state.user_confirmed_edited_exploratory_summary)
            st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
        if formalized_guides and formalized_guides.get("InterviewGuideStructure_DEFINED") and formalized_guides.get("DesiredOutputCatalogStructureGuidance_DEFINED"):
            st.session_state.ai_formalized_interview_guide = formalized_guides["InterviewGuideStructure_DEFINED"]
            st.session_state.ai_formalized_catalog_guide = formalized_guides["DesiredOutputCatalogStructureGuidance_DEFINED"]
            st.session_state.user_edited_interview_guide = st.session_state.ai_formalized_interview_guide 
            st.session_state.user_edited_catalog_guide = st.session_state.ai_formalized_catalog_guide
            st.session_state.current_phase = "structure_review_edit"; st.success("AI has formalized the guides. Review/edit below.")
        else: st.error("AI failed to formalize the structure."); st.session_state.current_phase = "exploratory_done" 
        st.rerun()
    if st.session_state.current_phase == "structure_review_edit":
        st.subheader("AI-Formalized Guides (View/Edit)")
        is_disabled_for_editing = not st.session_state.editing_formalized_guides
        st.session_state.user_edited_interview_guide = st.text_area("Interview Guide Structure (Edit if needed):", value=st.session_state.user_edited_interview_guide, height=150, key=f"user_edit_interview_guide_key_{st.session_state.run_id}", disabled=is_disabled_for_editing)
        st.session_state.user_edited_catalog_guide = st.text_area("Catalog Output Guidance (Edit if needed):", value=st.session_state.user_edited_catalog_guide, height=150, key=f"user_edit_catalog_guide_key_{st.session_state.run_id}", disabled=is_disabled_for_editing)
        col1, col2, col3 = st.columns([0.35, 0.35, 0.3])
        with col1:
            if is_disabled_for_editing:
                if st.button("Edit Guides ✒️", use_container_width=True): st.session_state.editing_formalized_guides = True; st.rerun()
            else:
                if st.button("✅ Save Edited Guides", type="primary", use_container_width=True):
                    st.session_state.study_context["InterviewGuideStructure_DEFINED"] = st.session_state.user_edited_interview_guide
                    st.session_state.study_context["DesiredOutputCatalogStructureGuidance_DEFINED"] = st.session_state.user_edited_catalog_guide
                    st.session_state.ai_formalized_interview_guide = st.session_state.user_edited_interview_guide 
                    st.session_state.ai_formalized_catalog_guide = st.session_state.user_edited_catalog_guide
                    st.session_state.editing_formalized_guides = False; st.session_state.current_phase = "structure_confirmed_for_structured_rounds"; st.success("Guides Saved!"); st.rerun()
        with col2:
            if not is_disabled_for_editing:
                if st.button("❌ Cancel Edits", use_container_width=True):
                    st.session_state.user_edited_interview_guide = st.session_state.ai_formalized_interview_guide 
                    st.session_state.user_edited_catalog_guide = st.session_state.ai_formalized_catalog_guide
                    st.session_state.editing_formalized_guides = False; st.rerun()
        with col3:
             if is_disabled_for_editing:
                if st.button("➡️ Proceed w/ Current Guides", use_container_width=True):
                    st.session_state.study_context["InterviewGuideStructure_DEFINED"] = st.session_state.user_edited_interview_guide 
                    st.session_state.study_context["DesiredOutputCatalogStructureGuidance_DEFINED"] = st.session_state.user_edited_catalog_guide
                    st.session_state.editing_formalized_guides = False
                    st.session_state.current_phase = "structure_confirmed_for_structured_rounds"; st.rerun()

# PHASE 2: Structured Interview Round(s)
if st.session_state.current_phase == "structure_confirmed_for_structured_rounds":
    st.markdown("---"); st.header("Phase 2: Structured Interview Round(s)")
    st.markdown("**Finalized Interview Guide (to be used):**"); st.text_area("Finalized Interview Guide Display:", value=st.session_state.study_context.get("InterviewGuideStructure_DEFINED","Not defined."), height=75, disabled=True, key=f"final_guide_disp_{st.session_state.run_id}")
    st.markdown("**Finalized Catalog Guidance (to be used):**"); st.text_area("Finalized Catalog Guidance Display:", value=st.session_state.study_context.get("DesiredOutputCatalogStructureGuidance_DEFINED","Not defined."), height=75, disabled=True, key=f"final_catalog_guide_disp_{st.session_state.run_id}")
    num_done = len(st.session_state.structured_interview_results_list)
    num_target = st.session_state.num_structured_interviews_target
    st.write(f"**Structured Interviews Completed: {num_done} / {num_target}**")
    if num_done < num_target:
        if st.button(f"Run Structured Interview #{num_done + 1} (AI Persona)", key=f"run_struct_int_btn_{st.session_state.run_id}_{num_done}"):
            st.session_state.current_phase = "structured_interview_running"; st.rerun()
    elif num_target > 0 : 
        st.success(f"All {num_target} targeted structured interview round(s) complete!"); st.session_state.current_phase = "structured_interviews_done"; st.rerun()
    else: 
        st.info("Target for structured interviews is 0. Proceed to catalog if exploratory data exists."); st.session_state.current_phase = "structured_interviews_done"; st.rerun()

# PHASE 2.5: Running a Structured Interview
if st.session_state.current_phase == "structured_interview_running":
    num_done_before_this_run = len(st.session_state.structured_interview_results_list)
    with st.spinner(f"Running structured interview round #{num_done_before_this_run + 1}..."):
        current_run_study_context = st.session_state.study_context.copy()
        current_run_study_context["roles_interviewed_so_far"] = [p.get("role_title", p.get("Role", "UnknownRole")) for p in st.session_state.personas_used_in_study if isinstance(p,dict)]
        if not current_run_study_context.get("InterviewGuideStructure_DEFINED"): st.error("Critical Error: Interview Guide Structure is missing!"); st.stop()
        results_structured = perform_study_phase(current_run_study_context, False, st.session_state.max_turns_per_interview_gui)
        st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
        if results_structured.get("error_message"): st.error(f"Error: {results_structured['error_message']}")
        else: 
            st.session_state.structured_interview_results_list.append(results_structured)
            if results_structured.get("selected_persona_dict"): st.session_state.personas_used_in_study.append(results_structured.get("selected_persona_dict"))
            st.success(f"Structured interview round #{len(st.session_state.structured_interview_results_list)} complete!")
        if len(st.session_state.structured_interview_results_list) < st.session_state.num_structured_interviews_target:
            st.session_state.current_phase = "structure_confirmed_for_structured_rounds" 
        else: st.session_state.current_phase = "structured_interviews_done"
        st.rerun()

# Display results of ALL structured interviews
if st.session_state.structured_interview_results_list and st.session_state.current_phase in ["structure_confirmed_for_structured_rounds", "structured_interviews_done", "catalog_generating", "catalog_done"]:
    st.subheader(f"Results from Structured Interview Round(s):")
    for i, interview_data in enumerate(st.session_state.structured_interview_results_list):
        persona_name_struct = interview_data.get('selected_persona_name', 'N/A')
        persona_dict_struct = interview_data.get('selected_persona_dict', {})
        role_title = persona_dict_struct.get('role_title', persona_dict_struct.get('Role',''))
        expander_title = f"Interview #{i+1} (Persona: {persona_name_struct}{f' - {role_title}' if role_title else ''})"
        with st.expander(expander_title, expanded=(i == len(st.session_state.structured_interview_results_list) -1) ):
            if persona_dict_struct:
                with st.popover("View Full Persona JSON"): st.json(persona_dict_struct)
                st.markdown("---")
            st.markdown("**Transcript:**"); 
            if interview_data.get("transcript"):
                for qa_idx, qa in enumerate(interview_data["transcript"]):
                    if "event" in qa: st.caption(f"Event: {qa['event']}")
                    else: st.markdown(f"**Q{qa_idx+1}:** {qa['question']}\n\n**A{qa_idx+1}:** {qa['answer']}")
            st.markdown("**Summary from this round (AI Generated):**"); 
            st.text_area(f"summary_round_struct_{i+1}", value=interview_data.get("summary", ""), height=200, disabled=True, key=f"summary_disp_struct_{st.session_state.run_id}_{i}")
    st.markdown("---")

# PHASE 3: Final Catalog Generation
if st.session_state.current_phase == "structured_interviews_done":
    if st.session_state.structured_interview_results_list or st.session_state.exploratory_summary_proposed_structure : 
        st.header("Phase 3: Final Catalog Generation")
        if st.button("Generate Final Faktorenkatalog", key=f"gen_catalog_btn_{st.session_state.run_id}"):
            st.session_state.current_phase = "catalog_generating"; st.rerun()

if st.session_state.current_phase == "catalog_generating":
    with st.spinner("Aggregating summaries and generating final Faktorenkatalog..."):
        valid_summaries = []
        if st.session_state.structured_interview_results_list:
             valid_summaries = [ f"Summary from interview with {res.get('selected_persona_name', 'Unknown Expert')}:\n{res.get('summary', '')}" 
                for res in st.session_state.structured_interview_results_list if res.get("summary") and res.get("summary").strip()]
        final_exploratory_summary_to_use = st.session_state.user_confirmed_edited_exploratory_summary if st.session_state.user_confirmed_edited_exploratory_summary else st.session_state.exploratory_summary_proposed_structure
        if not valid_summaries and final_exploratory_summary_to_use:
            st.warning("No structured summaries found. Generating catalog from the (edited/confirmed) exploratory summary.")
            valid_summaries = [f"Exploratory Summary (Interviewee: {st.session_state.selected_persona_name_expl}):\n{final_exploratory_summary_to_use}"]
        if not valid_summaries:
            st.error("No valid summaries available to generate catalog."); st.session_state.current_phase = "structured_interviews_done"; st.rerun()
        else:
            aggregated_summaries_text = "\n\n---\nNEXT INTERVIEW SUMMARY:\n---\n\n".join(valid_summaries)
            if not st.session_state.study_context.get("DesiredOutputCatalogStructureGuidance_DEFINED"):
                st.error("Critical Error: Desired Output Catalog Structure Guidance is missing for final catalog generation!")
                st.session_state.current_phase = "structure_review_edit"; st.rerun()
            else:
                final_catalog = generate_final_catalog_from_summaries(st.session_state.study_context, aggregated_summaries_text)
                st.session_state.tokens_input = session_input_tokens; st.session_state.tokens_output = session_output_tokens
                if final_catalog:
                    st.session_state.final_catalog_output = final_catalog
                    st.session_state.current_phase = "catalog_done"; st.success("Final Faktorenkatalog generated!")
                else: st.error("Failed to generate final Faktorenkatalog."); st.session_state.current_phase = "structured_interviews_done"
                st.rerun() 

if st.session_state.current_phase == "catalog_done":
    if st.session_state.final_catalog_output:
        st.subheader("Final Generated Faktorenkatalog"); st.markdown("---")
        st.markdown(st.session_state.final_catalog_output) 
        st.download_button(
            label="Download Faktorenkatalog (.md)",
            data=st.session_state.final_catalog_output,
            file_name=f"Faktorenkatalog_{st.session_state.study_context.get('OverallStudyTopic','Study').replace(' ','_')}.md",
            mime="text/markdown")
    else: st.warning("Final catalog was not generated or is empty.")
    if st.button("Start New Study (Resets Everything)", key=f"final_reset_btn_{st.session_state.run_id}"):
        st.session_state.clear(); st.session_state.run_id = 0; st.session_state.current_phase = "initial_setup"
        st.session_state.study_context = {}; st.session_state.interview_mode = "AI Persona Simulation"
        st.session_state.max_turns_per_interview_gui = MAX_INTERVIEW_TURNS_DEFAULT
        st.session_state.num_structured_interviews_target = 1; st.session_state.metrics_expanded = True
        # Re-initialize specific keys that st.clear() removes but are needed for widgets before first "Set Study"
        # For keys used as widget `key` or `value` directly from session_state
        st.session_state.human_expert_name_title_input = ""
        st.session_state.human_expert_role_input = ""
        st.session_state.human_expert_expertise_input = ""
        st.session_state.human_expert_perspective_input = ""
        st.session_state.user_edited_interview_guide = ""
        st.session_state.user_edited_catalog_guide = ""
        st.session_state.exploratory_summary_proposed_structure ="" 
        st.session_state.human_answer_input = ""
        # Voice toggles - let them persist user's last choice or reset them:
        # st.session_state.enable_voice_output = False 
        # st.session_state.enable_voice_input = False
        # st.session_state.tts_provider = "Google TTS (gTTS)"
        # st.session_state.elevenlabs_voice_id_input = "Rachel"
        st.rerun()
