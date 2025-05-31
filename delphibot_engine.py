# delphibot_engine.py

from agents import Agent, Runner
from typing import Any, Dict, List, Optional
import json
import tiktoken
import asyncio # Make sure asyncio is imported



# --- Configuration & Pricing (Keep these at the top) ---
MAX_INTERVIEW_TURNS_DEFAULT = 3 # Default, can be overridden
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
INPUT_PRICE_PER_MILLION_TOKENS = 0.40
OUTPUT_PRICE_PER_MILLION_TOKENS = 1.60

# --- Helper Function for Token Counting (Keep this as is) ---
def count_tokens(string: Optional[str], model_name: str = MODEL_NAME) -> int:
    if not string: return 0
    try: encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # print(f"Warning: Model '{model_name}' not found by tiktoken. Using 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))

# --- AGENT DEFINITIONS ---
ManagerAgent = Agent(
    name="ManagerAgent",
    instructions="""
    You are the Central Orchestrator for a Delphi Study.
    You will be initialized with a 'StudyContext'. This context indicates the phase and relevant guides.

    Your role:
    1.  **Persona Request:** (As before, including diversity for structured rounds)
    2.  **Interview Initiation/Conduction Instruction:** (As before)
    3.  **Summarization Instruction:** (As before)
    4.  **Structure Formalization:** (As before)
    5.  **Catalog Writing Instruction (MODIFIED FOR SYNTHESIS):**
        -   You will receive 'Aggregated Structured Summaries' which contain summaries from MULTIPLE expert interviews
            on the OverallStudyTopic, all following the same defined structure (Systemebenen and general idea of Faktorname).
        -   Your task is to instruct the CatalogWriterAgent to take these multiple summaries and COMPILE and SYNTHESIZE them
            into ONE coherent, final 'Faktorenkatalog'.
        -   The CatalogWriterAgent should identify common Faktorname across the summaries for each Systemebene.
        -   For each Faktorname, it should synthesize the definitions, dimensions, and trends from the different experts,
            noting any consensus, unique insights, or important variations in perspective.
        -   It must strictly adhere to the 'DesiredOutputCatalogStructureGuidance_DEFINED' from the StudyContext for the final report's structure and style.
        -   Provide the CatalogWriterAgent with the Aggregated Summaries and all necessary guidance from the StudyContext.

    Output ONLY the direct instruction or requested JSON. Be precise.
    """,
    model=MODEL_NAME
)

PersonaManagerAgent = Agent(
    name="PersonaManagerAgent",
    instructions="""
    You are a Persona Manager. You will receive:
    - The OverallStudyTopic.
    - PersonaRequirementsGuidance.
    - (Optionally) A list of PredefinedPersonas.
    - (Optionally, for subsequent interviews) A list of 'roles_or_expertise_already_interviewed'.

    Your task: Provide ONE expert persona JSON highly relevant to the StudyTopic and requirements.
    If 'roles_or_expertise_already_interviewed' is provided and not empty, make a strong effort to select or create a persona
    that offers a *distinctly different perspective or primary area of expertise* than those already covered to ensure diversity.
    Select from predefined if a suitable diverse option exists, otherwise create a new detailed one fitting this need for diversity.
    Output ONLY the persona JSON.
    """,
    model=MODEL_NAME
)

InterviewerAgent = Agent(
    name="InterviewerAgent",
    instructions="""
    You are a professional Interviewer. You will receive:
    - The OverallStudyTopic and TargetYear.
    - A PersonaProfile (JSON of an AI expert OR a text block describing a human expert's profile).
    - The ConversationHistory so far.
    - An indication if this is an 'exploratory_interview' OR a specific 'interview_guide_structure' to follow.

    Your task:
    - **If a PersonaProfile (AI or Human) is provided, use any relevant details (like name, expertise, stance) to help tailor your questions and make the interaction more relevant.**
    - If 'exploratory_interview': Goal is to broadly explore the OverallStudyTopic...
    - If 'interview_guide_structure' is provided: Goal is to cover the sections/themes in that structure...

    In both modes: Refer to ConversationHistory. 
    **Decision to Conclude:** If objectives met or interview unproductive, output: INTERVIEW_COMPLETE. Otherwise, output ONLY your next question.
    """,
    model=MODEL_NAME
)

PersonaResponderAgent = Agent(
    name="PersonaResponderAgent",
    instructions="""
    You are an AI embodying an expert persona. You will receive:
    - A PersonaProfile (JSON) to adopt.
    - The ConversationHistory.
    - The CurrentQuestion from the interviewer.
    Answer the CurrentQuestion from the perspective of the PersonaProfile, considering ConversationHistory. Be concise. Output ONLY the answer.
    """,
    model=MODEL_NAME
)

SummarizerAgent = Agent(
    name="SummarizerAgent",
    instructions="""
    You are a Summarization Specialist. You will receive:
    - The full InterviewTranscript.
    - The OverallStudyTopic and TargetYear.
    - An indication if this was an 'exploratory_summary' OR if you should follow a 'defined_output_structure_guidance'.

    Your task:
    - **If 'exploratory_summary':** Analyze transcript to identify 4-6 MAJOR THEMATIC CATEGORIES (initial 'Systemebenen'). Under each, list specific 'Faktorname' with Definitions, Dimensions, Trends. This summary PROPOSES the structure.
    - **If 'defined_output_structure_guidance' is provided:**
        1. Extract key 'Einflussfaktoren' **directly from the provided InterviewTranscript** that are relevant to the **OverallStudyTopic**.
        2. Structure these factors strictly according to the 'defined_output_structure_guidance' (which specifies the Systemebenen).
        3. For each Faktorname you identify **from the transcript**, detail its Definition/Understanding, Dimensions Discussed, and Trends for the TargetYear **as stated or implied by the interviewee in the transcript.**
        4. Ensure all extracted content is pertinent to the OverallStudyTopic. Do not introduce factors or details not supported by the current transcript.
    Output a well-organized, structured text summary.
    """,
    model=MODEL_NAME
)

CatalogWriterAgent = Agent(
    name="CatalogWriterAgent",
    instructions="""
    You are a highly skilled Catalog Writer and Synthesizer. You will receive:
    1.  'AggregatedSummaries': A collection of structured summaries from MULTIPLE expert interviews on a specific OverallStudyTopic.
        Each summary in the collection follows a common structure (e.g., Systemebenen, Faktorname, Definition, Dimensions, Trends).
    2.  The 'OverallStudyTopic'.
    3.  'DesiredOutputCatalogStructureGuidance': Detailed instructions on the final catalog's formatting, style,
        and the main Systemebenen (section headings) to use.

    Your CRITICAL task is to:
    A.  **Process ALL provided summaries.**
    B.  **Identify common Systemebenen and Faktorname** discussed across the different expert summaries.
    C.  For each common Faktorname within a Systemebene:
        i.  **Synthesize** the 'Definitions/Understanding' provided by the different experts into a comprehensive definition.
        ii. **Synthesize** the 'Dimensions Discussed,' incorporating all relevant aspects mentioned across the summaries.
        iii. **Synthesize** the 'Trends for [TargetYear],' highlighting consensus, key variations, or unique future outlooks from different experts.
    D.  If a factor is only mentioned by one expert but is significant, include it.
    E.  **Compile these synthesized insights into ONE coherent, final 'Faktorenkatalog' section or document.**
    F.  Strictly adhere to the 'DesiredOutputCatalogStructureGuidance' for the overall structure (Systemebenen) and professional report style.
        Ensure clear headings, and well-written, concise paragraphs or bullet points for the synthesized details.
    G.  Your output should be the single, consolidated, and synthesized Faktorenkatalog. DO NOT just concatenate the input summaries.
    """,
    model=MODEL_NAME
)
# --- END OF AGENT DEFINITIONS ---

PREDEFINED_PERSONAS_NEWSPAPER_TOPIC = [
    { "name": "Dr. Johanna Weber", "age": 58, "role_title": "Chefredakteurin", "expertise_areas": ["Digitaljournalismus", "Medienmanagement"]},
    { "name": "Prof. Dr. Klaus Richter", "age": 65, "role_title": "Medienwissenschaftler", "expertise_areas": ["Medienwandel", "Journalismusforschung"]},
    { "name": "Lena Meyer", "age": 22, "role_title": "Medienstudentin", "expertise_areas": ["Mediennutzung junger Zielgruppen", "Social Media News"]}
]

# Global session token counters
session_input_tokens = 0
session_output_tokens = 0

def reset_session_tokens_for_engine():
    global session_input_tokens, session_output_tokens
    session_input_tokens = 0
    session_output_tokens = 0

def _run_agent_internal(agent: Agent, prompt_text: str) -> Any | None:
    global session_input_tokens, session_output_tokens
    current_input_tokens = count_tokens(prompt_text); session_input_tokens += current_input_tokens
    print(f"  ENGINE: (Running Agent: {agent.name}, Input Tokens: {current_input_tokens})")
    result = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: result = Runner.run_sync(agent, prompt_text)
    finally: pass # Loop management for run_sync is usually handled by itself or needs more complex sync context if used heavily
    output_text = result.final_output if result and result.final_output else ""
    current_output_tokens = count_tokens(output_text); session_output_tokens += current_output_tokens
    print(f"  ENGINE: (Agent: {agent.name} completed, Output Tokens: {current_output_tokens})")
    return result

def extract_json_from_response(response_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if not response_str: return None
    core_json_str = ""
    stripped_response = response_str.strip()
    if stripped_response.startswith("{") and stripped_response.endswith("}"): core_json_str = stripped_response
    else:
        if stripped_response.startswith("```json"): json_block_start = stripped_response.find("```json") + len("```json"); json_block_end = stripped_response.rfind("```"); core_json_str = stripped_response[json_block_start:json_block_end].strip() if json_block_start < json_block_end and stripped_response[json_block_start:json_block_end].strip().startswith("{") and stripped_response[json_block_start:json_block_end].strip().endswith("}") else ""
        elif stripped_response.startswith("```"): json_block_start = stripped_response.find("```") + len("```"); json_block_end = stripped_response.rfind("```"); core_json_str = stripped_response[json_block_start:json_block_end].strip() if json_block_start < json_block_end and stripped_response[json_block_start:json_block_end].strip().startswith("{") and stripped_response[json_block_start:json_block_end].strip().endswith("}") else ""
        if not core_json_str: start_index = stripped_response.find("{"); end_index = stripped_response.rfind("}") + 1; core_json_str = stripped_response[start_index:end_index] if start_index != -1 and end_index > start_index else ""
    if not core_json_str: print(f"!ENGINE WARNING: No valid JSON object found in string: '{response_str}'"); return None
    try: return json.loads(core_json_str)
    except json.JSONDecodeError as e: print(f"!ENGINE ERROR: JSON parsing failed. {e.msg}. Raw: '{core_json_str}' from '{response_str}'"); return None

# --- ENGINE FUNCTIONS ---

def _conduct_single_interview(
    study_context_for_interview: Dict,
    selected_persona_dict: Dict,
    is_exploratory: bool,
    max_turns: int
) -> List[Dict[str, str]]:
    local_interview_transcript: List[Dict[str, str]] = []
    print(f"\nENGINE: --- ManagerAgent: Task -> Formulate Interview Start Instruction ---")
    interview_type_guidance_key = 'InterviewGuideExploratoryPrompt' if is_exploratory else 'InterviewGuideStructure_DEFINED'
    interview_type_description = "EXPLORATORY" if is_exploratory else "STRUCTURED (using defined guide)"
    prompt_for_manager_interview_start = (
        f"Current Study Context ({interview_type_description} Phase):\n{json.dumps(study_context_for_interview, indent=2, ensure_ascii=False)}\n"
        f"Selected Persona:\n{json.dumps(selected_persona_dict, indent=2, ensure_ascii=False)}\n\n"
        f"Instruct InterviewerAgent to start the interview. Provide it with:\n"
        f"1. OverallStudyTopic: '{study_context_for_interview['OverallStudyTopic']}'\n"
        f"2. TargetYear: {study_context_for_interview['TargetYear']}\n"
        f"3. The selected PersonaProfile.\n"
        f"4. The guidance from StudyContext's '{interview_type_guidance_key}' for this {interview_type_description.lower()} interview.\n"
        f"Output ONLY the complete instruction for InterviewerAgent."
    )
    manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_interview_start)
    if not (manager_response_obj and manager_response_obj.final_output):
        print(f"!ENGINE ERROR: ManagerAgent failed to instruct Interviewer for {interview_type_description} interview.")
        return local_interview_transcript
    instruction_for_interviewer = manager_response_obj.final_output
    print(f"ENGINE: Manager's instruction for Interviewer ({interview_type_description}):\n{instruction_for_interviewer}")

    current_question = ""
    for turn in range(max_turns):
        print(f"\nENGINE: --- {interview_type_description} Interview - Turn {turn + 1}/{max_turns} ---")
        prompt_for_interviewer_agent: str
        if turn == 0: prompt_for_interviewer_agent = instruction_for_interviewer
        else:
            guide_ref_str = (f"exploratory guidance: '{study_context_for_interview.get('InterviewGuideExploratoryPrompt', '')}'"
                           if is_exploratory
                           else f"defined guide: '{study_context_for_interview.get('InterviewGuideStructure_DEFINED', '')}'")
            prompt_for_interviewer_agent = (
                f"OverallStudyTopic: {study_context_for_interview['OverallStudyTopic']}\nTargetYear: {study_context_for_interview['TargetYear']}\n"
                f"PersonaProfile: {json.dumps(selected_persona_dict, ensure_ascii=False)}\n"
                f"ConversationHistory: {json.dumps(local_interview_transcript, indent=2, ensure_ascii=False)}\n"
                f"You are conducting an {interview_type_description.lower()} interview, following {guide_ref_str}. Ask your next question or output INTERVIEW_COMPLETE."
            )
        interviewer_response_obj = _run_agent_internal(InterviewerAgent, prompt_for_interviewer_agent)
        if not (interviewer_response_obj and interviewer_response_obj.final_output): print(f"!ENGINE ERROR: InterviewerAgent failed turn {turn + 1}."); break
        current_question = interviewer_response_obj.final_output.strip()
        print(f"ENGINE: InterviewerAgent's Question {turn + 1}:\n{current_question}")
        if "INTERVIEW_COMPLETE" in current_question.upper():
            print("ENGINE: InterviewerAgent signaled interview completion."); local_interview_transcript.append({"event": f"INTERVIEW_CONCLUDED_BY_INTERVIEWER_AT_TURN_{turn+1}", "signal": current_question}); break
        
        prompt_for_responder_agent = (
            f"PersonaProfile: {json.dumps(selected_persona_dict, ensure_ascii=False)}\n"
            f"ConversationHistory: {json.dumps(local_interview_transcript, indent=2, ensure_ascii=False)}\n"
            f"CurrentQuestion: '{current_question}'\n\nAnswer as persona. Output ONLY the answer."
        )
        responder_response_obj = _run_agent_internal(PersonaResponderAgent, prompt_for_responder_agent)
        if not (responder_response_obj and responder_response_obj.final_output): print(f"!ENGINE ERROR: PersonaResponderAgent failed turn {turn + 1}."); break
        current_answer = responder_response_obj.final_output.strip()
        print(f"ENGINE: PersonaResponderAgent's Answer {turn + 1}:\n{current_answer}")
        local_interview_transcript.append({"question": current_question, "answer": current_answer})
    print(f"\nENGINE: --- {interview_type_description} Interview Loop Finished. Transcript ({len(local_interview_transcript)} turns). ---")
    return local_interview_transcript


def perform_study_phase(
    study_context: Dict,
    is_exploratory_phase: bool,
    max_interview_turns: int = MAX_INTERVIEW_TURNS_DEFAULT # Make sure MAX_INTERVIEW_TURNS_DEFAULT is defined
) -> Dict[str, Any]:
    """
    Performs one phase of the study:
    - Gets a persona
    - Conducts an interview (exploratory or structured based on is_exploratory_phase and study_context)
    - Summarizes the interview (exploratory or structured based on is_exploratory_phase and study_context)
    Returns a dictionary containing 'transcript', 'summary', 'selected_persona_dict', 'selected_persona_name', 'error_message'.
    Token counts are updated globally via _run_agent_internal.
    """
    phase_results = {
        "transcript": [], 
        "summary": "", 
        "selected_persona_dict": None, 
        "selected_persona_name": "N/A", 
        "error_message": None,
        "error_message_interview_loop": None, # Specific for interview loop errors
        "study_context_used": study_context.copy() # For debugging or context checking
    }
    selected_persona_dict_candidate: Optional[Dict[str, Any]] = None # Renamed to avoid conflict

    # 1. Manager -> Get Persona Instruction
    print(f"\nENGINE: --- ManagerAgent: Task -> Formulate Persona Request for {'Exploratory' if is_exploratory_phase else 'Structured'} Round ---")
    roles_already_covered_prompt_segment = ""
    if not is_exploratory_phase and study_context.get("roles_interviewed_so_far"):
        roles_already_covered_prompt_segment = (
            f"Previously interviewed roles/expertise areas for this study include: "
            f"{json.dumps(study_context['roles_interviewed_so_far'])}. "
            f"Please aim for a persona offering a new/different perspective for this round."
        )

    prompt_for_manager_persona_req = (
        f"Current Study Context:\n{json.dumps(study_context, indent=2, ensure_ascii=False)}\n\n"
        f"You are initiating an interview for the {'Exploratory' if is_exploratory_phase else 'Structured'} Phase. "
        f"{roles_already_covered_prompt_segment}\n"
        f"Formulate instruction for PersonaManagerAgent to get a persona using OverallStudyTopic and "
        f"PersonaRequirementsGuidance. It should consider the list of predefined personas if available in the context. "
        f"Output ONLY this instruction for PersonaManagerAgent."
    )
    manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_persona_req)
    if not (manager_response_obj and manager_response_obj.final_output): 
        phase_results["error_message"] = "ManagerAgent failed PersonaManager instruction."; return phase_results
    instruction_for_persona_manager = manager_response_obj.final_output
    # print(f"ENGINE: Manager's instruction for PersonaManager:\n{instruction_for_persona_manager}") # Optional debug

    # 2. PersonaManager -> Get Persona
    print(f"\nENGINE: --- PersonaManagerAgent: Task -> Provide Persona ---")
    persona_response_obj = _run_agent_internal(PersonaManagerAgent, instruction_for_persona_manager)
    if not (persona_response_obj and persona_response_obj.final_output):
        phase_results["error_message"] = "PersonaManagerAgent failed to provide a persona."; return phase_results
    
    selected_persona_dict_candidate = extract_json_from_response(persona_response_obj.final_output)
    if not selected_persona_dict_candidate:
        phase_results["error_message"] = f"PersonaManagerAgent JSON parsing failed. Raw: '{persona_response_obj.final_output if persona_response_obj else 'No output from PersonaManager'}'."; return phase_results
    
    phase_results["selected_persona_dict"] = selected_persona_dict_candidate
    phase_results["selected_persona_name"] = phase_results["selected_persona_dict"].get("Name", 
                                               phase_results["selected_persona_dict"].get("name", "Unknown Persona"))
    print(f"ENGINE: Successfully parsed persona: {phase_results['selected_persona_name']}")


    # 3. Conduct Interview
    # We need to pass the selected_persona_dict to _conduct_single_interview
    if phase_results["selected_persona_dict"] is None: # Should not happen if parsing was successful
        phase_results["error_message"] = "Selected persona dictionary is None before conducting interview."
        print(f"!ENGINE ERROR: {phase_results['error_message']}")
        return phase_results

    interview_transcript_result = _conduct_single_interview(
        study_context_for_interview=study_context,
        selected_persona_dict=phase_results["selected_persona_dict"], # Pass the actual dict
        is_exploratory=is_exploratory_phase,
        max_turns=max_interview_turns
    )
    phase_results["transcript"] = interview_transcript_result
    # _conduct_single_interview doesn't explicitly return an error, it prints. We check transcript length.
    if not phase_results["transcript"]:
        phase_results["error_message_interview_loop"] = "Interview did not produce a transcript or an error occurred in _conduct_single_interview."
        # No need to set phase_results["error_message"] here as it might overwrite a more specific one from _conduct_single_interview's internals (though it doesn't have one yet)


    # 4. Summarize Interview
    if phase_results["transcript"] and not phase_results.get("error_message_interview_loop"):
        print(f"\nENGINE: --- ManagerAgent: Task -> Formulate Summarizer Instruction ({'Exploratory' if is_exploratory_phase else 'Structured'}) ---")
        summarizer_guidance_key = 'SummarizerGuidanceExploratory' if is_exploratory_phase else 'DesiredOutputCatalogStructureGuidance_DEFINED'
        summarizer_mode_description = "an 'exploratory_summary' to PROPOSE a structure" if is_exploratory_phase else "a 'structured_summary' adhering to the defined output structure"

        prompt_for_manager_summarizer_instr = (
            f"Current Study Context:\n{json.dumps(study_context, indent=2, ensure_ascii=False)}\n"
            f"An Interview Transcript with {phase_results['selected_persona_name']} is ready (length: {len(json.dumps(phase_results['transcript']))} characters).\n\n"
            f"Your task is to formulate a concise, direct instruction for the SummarizerAgent. "
            f"This instruction should tell it to process AN UPCOMING transcript to perform {summarizer_mode_description}. "
            f"The SummarizerAgent will also receive the OverallStudyTopic ('{study_context['OverallStudyTopic']}'), "
            f"TargetYear ({study_context['TargetYear']}), the actual transcript, and guidance from the StudyContext's '{summarizer_guidance_key}'. "
            f"Output ONLY the direct command or introductory framing for the SummarizerAgent, "
            f"NOT the full prompt it will receive."
        )
        manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_summarizer_instr)
        
        if not (manager_response_obj and manager_response_obj.final_output):
            phase_results["error_message"] = "ManagerAgent failed to formulate instruction for Summarizer.";
        else:
            base_instruction_from_manager = manager_response_obj.final_output 
            print(f"ENGINE: Manager's base instruction for Summarizer:\n{base_instruction_from_manager}")

            # --- PYTHON CODE CONSTRUCTS THE FULL PROMPT FOR SUMMARIZER AGENT ---
            full_prompt_for_summarizer = (
                f"{base_instruction_from_manager}\n\n" 
                f"OverallStudyTopic: {study_context.get('OverallStudyTopic')}\n"
                f"TargetYear: {study_context.get('TargetYear')}\n"
                f"Guidance on structure/output (from StudyContext's '{summarizer_guidance_key}'):\n"
                f"{study_context.get(summarizer_guidance_key, 'No specific structural guidance provided.')}\n\n"
                f"**Interview Transcript to Summarize:**\n" 
                f"```json\n{json.dumps(phase_results['transcript'], indent=2, ensure_ascii=False)}\n```\n\n" # Transcript embedded
                f"Please provide the required summary based on ALL the above information, especially focusing on the Interview Transcript and the provided Guidance."
            )
            # --- END OF PROMPT CONSTRUCTION ---
            
            print(f"\nENGINE: --- SummarizerAgent: Task -> Provide Summary ({'Exploratory' if is_exploratory_phase else 'Structured'}) ---")
            
            # +++ START DEBUG BLOCK +++
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! DEBUGGING PROMPT BEING SENT TO SUMMARIZER AGENT !!!")
            transcript_preview = "TRANSCRIPT MISSING/EMPTY IN PHASE_RESULTS"
            if phase_results.get('transcript'):
                transcript_preview = json.dumps(phase_results['transcript'], ensure_ascii=False)[:100] + "..." \
                                     if len(json.dumps(phase_results['transcript'])) > 100 \
                                     else json.dumps(phase_results['transcript'], ensure_ascii=False)

            print(f"!!! IS TRANSCRIPT PRESENT? Preview: {transcript_preview}")
            print(f"!!! LENGTH OF FULL PROMPT TO SUMMARIZER: {len(full_prompt_for_summarizer)}")
            print(f"!!! START OF FULL PROMPT TO SUMMARIZER (first 500 chars):\n{full_prompt_for_summarizer[:500]}")
            print("!!! END OF FULL PROMPT TO SUMMARIZER (last 300 chars):\n{full_prompt_for_summarizer[-300:]}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # +++ END DEBUG BLOCK +++
            
            summarizer_response_obj = _run_agent_internal(SummarizerAgent, full_prompt_for_summarizer) 
            
            if summarizer_response_obj and summarizer_response_obj.final_output:
                phase_results["summary"] = summarizer_response_obj.final_output
            else: 
                phase_results["error_message"] = "SummarizerAgent failed to provide summary."
    elif not phase_results.get("error_message") and not phase_results.get("error_message_interview_loop"): 
         phase_results["error_message"] = "No transcript available to summarize for this round (or previous interview error)."
    
    # If an error message was set during the interview loop, make sure it's the primary one.
    if phase_results.get("error_message_interview_loop") and not phase_results.get("error_message"):
        phase_results["error_message"] = phase_results["error_message_interview_loop"]
    del phase_results["error_message_interview_loop"] # Clean up temporary error key

    return phase_results


def formalize_structure_from_exploratory_summary(study_context: Dict, exploratory_summary: str) -> Optional[Dict[str,str]]:
    print(f"\nENGINE: --- ManagerAgent: Task -> Formalize Discovered Structure from Exploratory Summary ---")
    prompt_for_manager_formalize = (
        f"The following 'Exploratory Summary' was generated for the Study Topic '{study_context['OverallStudyTopic']}' "
        f"(Target Year: {study_context['TargetYear']}). It proposes several thematic categories (Systemebenen), "
        f"some with detailed factors and others noted as having no content from the initial interview:\n\n"
        f"```text\n{exploratory_summary}\n```\n\n"
        f"Your task is to analyze this proposed structure and define a focused and robust set of guides for subsequent structured interviews:\n"
        f"1.  **'InterviewGuideStructure_DEFINED':** Create a concise string that lists the **key thematic categories (Systemebenen)** that should be systematically covered. "
        f"    You might select the most content-rich categories from the proposal, or refine their names for clarity. "
        f"    This string should instruct an Interviewer to cover these selected Systemebenen and probe for Faktorname, Definitions, Dimensions, and Trends within each for the TargetYear.\n"
        f"    Example for a topic: 'Main Systemebenen to cover: 1. Tech Developments (AI, Platforms), 2. Economic Models (Subscriptions, Ads), 3. Audience Behavior (Engagement, Personalization). Probe factors within each.'\n\n"
        f"2.  **'DesiredOutputCatalogStructureGuidance_DEFINED':** Create a concise string for the Summarizer and CatalogWriter. "
        f"    This should specify that the final catalog be structured by the Systemebenen you just defined in point 1. "
        f"    For each Faktorname under those Systemebenen, it should detail: Definition/Understanding, Dimensions Discussed, and Trends for {study_context['TargetYear']}. "
        f"    Mention a professional report style.\n\n"
        f"Output ONLY a JSON object with keys 'InterviewGuideStructure_DEFINED' and 'DesiredOutputCatalogStructureGuidance_DEFINED'. "
        f"Focus on creating a practical and effective guide based on the exploratory findings."
    )
    manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_formalize) # ManagerAgent does this
    if manager_response_obj and manager_response_obj.final_output:
        formalized_guides_dict = extract_json_from_response(manager_response_obj.final_output)
        if formalized_guides_dict and \
           formalized_guides_dict.get("InterviewGuideStructure_DEFINED") and \
           formalized_guides_dict.get("DesiredOutputCatalogStructureGuidance_DEFINED"):
            print("ENGINE: ManagerAgent successfully formalized structure.")
            return formalized_guides_dict
        else: 
            print(f"!ENGINE ERROR: ManagerAgent did not return both defined guides in JSON. Raw: {manager_response_obj.final_output}")
            # Fallback: try to extract Systemebenen directly from summary if Manager fails to produce JSON
            # This is a more complex parsing task we can add later if needed. For now, rely on Manager.
    else: 
        print("!ENGINE ERROR: ManagerAgent failed to formalize structure.")
    return None

# The generate_final_catalog_from_summaries function needs to correctly use the
# DesiredOutputCatalogStructureGuidance_DEFINED from the study_context.
# Its prompt to ManagerAgent should emphasize this.
def generate_final_catalog_from_summaries(study_context: Dict, aggregated_summaries: str) -> Optional[str]:
    print(f"\nENGINE: --- ManagerAgent: Task -> Formulate FINAL CatalogWriter Instruction (for Synthesis) ---")
    
    # Ensure the definitive structure guide is passed
    defined_catalog_structure_guidance = study_context.get('DesiredOutputCatalogStructureGuidance_DEFINED', 
                                                           study_context.get('CatalogWriterGuidanceExploratory', # Fallback if structured not set
                                                                             "Structure as a professional catalog."))

    prompt_for_manager_final_cw = (
        f"Current Study Context:\n{json.dumps(study_context, indent=2, ensure_ascii=False)}\n" # Contains defined guides
        f"You have received Aggregated Structured Summaries from expert interviews on the OverallStudyTopic "
        f"'{study_context.get('OverallStudyTopic')}'. These summaries should align with a defined structure.\n"
        f"Aggregated Summaries:\n```text\n{aggregated_summaries}\n```\n\n"
        f"Your task is to instruct the CatalogWriterAgent to take these summaries, "
        f"SYNTHESIZE the insights, and compile the FINAL 'Faktorenkatalog'. "
        f"The CatalogWriterAgent MUST use the following 'DesiredOutputCatalogStructureGuidance_DEFINED' "
        f"from the StudyContext for the final report's structure and style:\n"
        f"'{defined_catalog_structure_guidance}'\n"
        f"Emphasize the need for synthesis of information for common factors across different summaries. "
        f"Output ONLY the direct instruction for the CatalogWriterAgent."
    )
    manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_final_cw)
    if manager_response_obj and manager_response_obj.final_output:
        instruction_for_final_catalogwriter = manager_response_obj.final_output
        
        # Python code now constructs the full prompt for CatalogWriter, ensuring data is present
        full_prompt_for_catalogwriter = (
            f"{instruction_for_final_catalogwriter}\n\n"
            f"OverallStudyTopic: {study_context.get('OverallStudyTopic')}\n"
            f"DesiredOutputCatalogStructureGuidance (use this for final structure and style):\n{defined_catalog_structure_guidance}\n\n"
            f"AggregatedSummaries to process and synthesize:\n{aggregated_summaries}\n\n"
            f"Compile the final, synthesized Faktorenkatalog based on ALL the above."
        )
        print(f"ENGINE: Manager's instruction + Full Context for Final CatalogWriter:\n{full_prompt_for_catalogwriter[:1000]}...") # Print snippet
        
        print(f"\nENGINE: --- CatalogWriterAgent: Task -> Provide Final Synthesized Catalog ---")
        final_catalog_obj = _run_agent_internal(CatalogWriterAgent, full_prompt_for_catalogwriter)
        if final_catalog_obj and final_catalog_obj.final_output:
            print(f"ENGINE: CatalogWriterAgent FINAL Output generated.")
            return final_catalog_obj.final_output
        else: print("!ENGINE ERROR: CatalogWriterAgent failed final output.")
    else: print("!ENGINE ERROR: ManagerAgent failed to instruct final CatalogWriter.")
    return None


# --- Example of how app.py might call these (for testing the engine directly) ---
if __name__ == "__main__":
    print("--- Direct Engine Test Start ---")
    reset_session_tokens_for_engine()
    
    # 1. Define Initial Study Context (Exploratory)
    current_study_context = {
        "OverallStudyTopic": "Die Zukunft der Tageszeitung in Deutschland bis 2047", "TargetYear": 2047,
        "GeographicalScope": "Deutschland", 
        "KeyObjectives_Wofuer": "Identifizierung und Strukturierung der Schlüsselfaktoren...",
        "PersonaRequirementsGuidance": "Experten-Personas mit vielfältigem Hintergrund relevant zur Medienlandschaft...",
        "PredefinedPersonas": PREDEFINED_PERSONAS_NEWSPAPER_TOPIC, 
        "InterviewGuideExploratoryPrompt": "Conduct an open-ended, exploratory interview on the OverallStudyTopic...",
        "SummarizerGuidanceExploratory": "This is an initial exploratory interview for the StudyTopic. Analyze the transcript to identify 4-6 MAJOR THEMATIC CATEGORIES that emerged...",
        # These are None initially for an exploratory phase
        "InterviewGuideStructure_DEFINED": None, 
        "DesiredOutputCatalogStructureGuidance_DEFINED": None,
        "roles_interviewed_so_far": [] 
    }

    # 2. Run Exploratory Phase
    print("\n\n========== RUNNING EXPLORATORY PHASE ==========")
    exploratory_results = perform_study_phase(current_study_context, is_exploratory_phase=True)
    
    all_interview_summaries_for_catalog = [] # To collect all summaries

    if exploratory_results.get("error_message"): 
        print(f"EXPLORATORY PHASE FAILED: {exploratory_results['error_message']}")
    else:
        print(f"\nExploratory Interview with {exploratory_results.get('selected_persona_name')} complete.")
        exploratory_summary = exploratory_results.get('summary', '')
        print(f"Proposed Structure by Summarizer:\n{exploratory_summary}")

        # Add exploratory summary to collection (optional, could be kept separate)
        # all_interview_summaries_for_catalog.append(f"=== Insights from Exploratory Interview with {exploratory_results.get('selected_persona_name')} (Proposed Structure for Context):\n{exploratory_summary}")

        # 3. Formalize Structure (AI Step)
        if exploratory_summary:
            print("\n\n========== FORMALIZING STRUCTURE FROM EXPLORATORY SUMMARY ==========")
            formalized_guides = formalize_structure_from_exploratory_summary(current_study_context, exploratory_summary)
            if formalized_guides:
                current_study_context["InterviewGuideStructure_DEFINED"] = formalized_guides.get("InterviewGuideStructure_DEFINED")
                current_study_context["DesiredOutputCatalogStructureGuidance_DEFINED"] = formalized_guides.get("DesiredOutputCatalogStructureGuidance_DEFINED")
                current_study_context["InterviewGuideExploratoryPrompt"] = None # No longer needed for this topic
                current_study_context["SummarizerGuidanceExploratory"] = None # No longer needed
                print("\n--- Study Context Updated with AI-Formalized Structure for Structured Phase ---")
                print(f"Defined Interview Guide:\n{current_study_context['InterviewGuideStructure_DEFINED']}")
                print(f"Defined Catalog Guidance:\n{current_study_context['DesiredOutputCatalogStructureGuidance_DEFINED']}")
                
                # Update roles interviewed for the next round
                if exploratory_results.get("selected_persona_dict"):
                    role = exploratory_results["selected_persona_dict"].get("role_title", exploratory_results["selected_persona_dict"].get("Role", "UnknownRole"))
                    if role not in current_study_context["roles_interviewed_so_far"]:
                        current_study_context["roles_interviewed_so_far"].append(role)

                # 4. Run Structured Phase (Example with ONE more interview)
                print("\n\n========== RUNNING STRUCTURED PHASE (Example with 1 interview) ==========")
                structured_phase_results = perform_study_phase(current_study_context, is_exploratory_phase=False) # Now it's structured
                
                if structured_phase_results.get("error_message"):
                    print(f"STRUCTURED PHASE INTERVIEW FAILED: {structured_phase_results['error_message']}")
                elif structured_phase_results.get("summary"):
                    print(f"\nStructured Interview with {structured_phase_results.get('selected_persona_name')} complete.")
                    print(f"Summary from Structured Interview:\n{structured_phase_results.get('summary')}")
                    all_interview_summaries_for_catalog.append(
                        f"=== Insights from Structured Interview with {structured_phase_results.get('selected_persona_name')}:\n{structured_phase_results.get('summary')}"
                    )
                else:
                    print("Structured interview ran but produced no summary.")

            else:
                print("!ENGINE ERROR: Could not AI-formalize structure from exploratory summary. Cannot proceed to structured phase with AI guide.")
        else:
            print("No exploratory summary to formalize structure from.")

    # 5. Generate Final Catalog (using whatever summaries were collected)
    if all_interview_summaries_for_catalog:
        print("\n\n========== GENERATING FINAL CATALOG ==========")
        # Ensure the context has the DEFINED catalog guidance for the CatalogWriter
        if not current_study_context.get("DesiredOutputCatalogStructureGuidance_DEFINED"):
            print("!ENGINE WARNING: DesiredOutputCatalogStructureGuidance_DEFINED is missing. CatalogWriter might use generic guidance.")
            current_study_context["DesiredOutputCatalogStructureGuidance_DEFINED"] = "Compile a professional factors catalog based on the provided summaries, using the main thematic sections present in the summaries."


        final_catalog = generate_final_catalog_from_summaries(
            current_study_context, 
            "\n\n".join(all_interview_summaries_for_catalog)
        )
        if final_catalog:
            print("\n\n===== FINAL GENERATED CATALOG (from Test Harness) =====")
            print(final_catalog)
        else:
            print("!ENGINE ERROR: Could not generate final catalog in test harness.")
    else:
        print("No summaries available to generate final catalog in test harness.")


    # --- Final Token Count and Cost ---
    print(f"\n--- Session Summary (Direct Run) ---")
    print(f"Total Input Tokens: {session_input_tokens}")
    print(f"Total Output Tokens: {session_output_tokens}")
    cost_input = (session_input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION_TOKENS
    cost_output = (session_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION_TOKENS
    total_cost = cost_input + cost_output
    print(f"Estimated Cost for Input Tokens: ${cost_input:.6f}")
    print(f"Estimated Cost for Output Tokens: ${cost_output:.6f}")
    print(f"Total Estimated Session Cost: ${total_cost:.6f}")
    usd_to_eur_rate = 0.88 
    total_cost_eur = total_cost * usd_to_eur_rate
    print(f"Total Estimated Session Cost (EUR): €{total_cost_eur:.6f} (at rate 1 USD = {usd_to_eur_rate:.4f} EUR)")

    print(f"\n--- Engine Direct Run Finished ---")
