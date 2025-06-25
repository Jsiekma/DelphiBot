# delphibot_engine.py

from agents import Agent, Runner
from typing import Any, Dict, List, Optional
import json
import tiktoken
import asyncio



# --- Configuration & Pricing ---
MAX_INTERVIEW_TURNS_DEFAULT = 3 # Default, can be overridden
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
INPUT_PRICE_PER_MILLION_TOKENS = 0.40
OUTPUT_PRICE_PER_MILLION_TOKENS = 1.60

# --- Helper Function for Token Counting ---
def count_tokens(string: Optional[str], model_name: str = MODEL_NAME) -> int:
    if not string: return 0
    try: encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
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

# <<< --- HEBEL 1: ANPASSUNG PERSONA MANAGER AGENT --- >>>
PersonaManagerAgent = Agent(
    name="PersonaManagerAgent",
    instructions="""
    You are a Persona Manager. You will receive:
    - The OverallStudyTopic.
    - PersonaRequirementsGuidance.
    - (Optionally) A list of PredefinedPersonas.
    - (Optionally) A list of 'roles_or_expertise_already_interviewed'.

    Your CRITICAL task is to provide ONE expert persona JSON that is not only relevant but also adds a UNIQUE and STRONG perspective. The expert need not be someone highly conventionally qualified, they could also be ordinary people offering their perspective on the topic at hand.
    If 'roles_or_expertise_already_interviewed' is not provided or empty, create a persona that does **not have any extremely unique or radical viewpoints**, but rather an expert who can broadly cover all of the topic at hand.
    If 'roles_or_expertise_already_interviewed' is provided and not empty, your primary goal is to create a persona with a *radically different viewpoint or primary focus*.
    Do not just change the job title; change the core perspective. For example, if you've interviewed a tech-optimist, create a strong tech-skeptic or a regulator focused only on risks.
    
    To ensure diversity, consider these axes:
    - Optimism vs. Pessimism regarding the technology's future.
    - Focus: Technical implementation vs. Social impact vs. Economic shifts vs. Ethical risks.
    - Background: Academic vs. Corporate vs. Governmental vs. Activist.

    Incorporate a specific, even slightly extreme, "stance" or "key belief" into the persona's profile.
    Output ONLY the persona JSON.
    """,
    model=MODEL_NAME
)

# <<< --- HEBEL 1: ANPASSUNG INTERVIEWER AGENT --- >>>
InterviewerAgent = Agent(
    name="InterviewerAgent",
    instructions="""
    You are a professional, sharp-witted Interviewer. You will receive:
    - The OverallStudyTopic and TargetYear.
    - A PersonaProfile (JSON of an AI expert OR a text block for a human).
    - The ConversationHistory.
    - Guidance on the interview type.

    Your task:
    - **CRITICAL: Analyze the PersonaProfile for any specific 'stance', 'role', or 'key beliefs'. You MUST use these details to ask targeted, probing, and sometimes challenging questions.** Do not ask generic questions.
    - If 'exploratory_interview': Goal is to broadly explore the OverallStudyTopic in Order to make it easy to identify broad categories of factors. 
    - If the persona is a 'tech-optimist', ask them about the potential downsides they might be ignoring. If they are a 'regulator', ask them how innovation can still thrive under their proposed rules.
    - Your goal is to extract the unique, specialized knowledge that ONLY this specific persona would have.
    - If 'interview_guide_structure' is provided: Your goal is to **aggressively populate the provided System Levels with numerous, diverse factors**. Your task is to probe the expert to identify as many distinct influence factors as possible, while asking questions relating to the experts characteristics. This **does not mean asking specific questions about one system level at a time**. Connect the system levels and ask more general questions that can be used to populate multiple system levels.  Ask follow-up questions to uncover different facets, sub-topics, and a wide variety of factors. Do not be satisfied with just one or two factors per level. Your primary objective in this phase is to generate **a large quantity and diversity of factors** for the final catalog. Try to identify at least 6-8 factors per system level.
    - In both modes: Refer to ConversationHistory. Do the interview in German.
    
    Decision to Conclude: If the persona's unique perspective is fully explored or the interview is unproductive, output: INTERVIEW_COMPLETE. Otherwise, output ONLY your next question.
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
    - **If 'exploratory_summary':** Your goal is to propose a high-level system model. Analyze the transcript to identify 4-6 abstract, overarching **System Levels** (e.g., for a newspaper topic, levels could be 'Publishing Sector', 'Society', 'Media Technology', 'Regulation'). System levels are often general market dynamics and structures, like the providers and users of services. These levels should be broad categories, not specific key factors. For each proposed System Level, provide a brief one-sentence description of what it encompasses. **Do not** list specific 'Faktorname' or their details (Definitions, Dimensions, Trends) at this stage. Your output should be a list of these coarse system levels and their descriptions. This proposes the high-level structure for the next phase.
    - **If 'defined_output_structure_guidance' is provided:** Your goal is to capture every influence factor discussed in the interview. 1. Meticulously extract **all** 'Einflussfaktoren' mentioned in the **InterviewTranscript** that are relevant to the **OverallStudyTopic**. 2. Structure these factors strictly according to the 'defined_output_structure_guidance' (which specifies the Systemebenen). It is expected that there will be many factors for each Systemebene. 3. For each Faktorname you identify from the transcript, detail its Definition/Understanding, Dimensions Discussed, and Trends for the TargetYear **as stated or implied by the interviewee in the transcript.** 4. Ensure your output is comprehensive and captures the **full breadth of factors** discussed, as the goal of this phase is to maximize the number of identified factors.
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
    B.  **Identify common and very similar Systemebenen and Faktorname** discussed across the different expert summaries. Try to identify many common Factors (preferably 6-10)
    C.  For each common or very similar Faktorname within a Systemebene:
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
        # Simplified event loop handling for Streamlit compatibility
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = Runner.run_sync(agent, prompt_text)
    except Exception as e:
        print(f"!ENGINE ERROR during agent run: {e}")
    finally:
        loop.close()

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
    max_interview_turns: int = MAX_INTERVIEW_TURNS_DEFAULT
) -> Dict[str, Any]:
    """
    Performs one phase of the study. This version uses the direct-prompting method.
    """
    phase_results = {
        "transcript": [], 
        "summary": "", 
        "selected_persona_dict": None, 
        "selected_persona_name": "N/A", 
        "error_message": None,
        "error_message_interview_loop": None,
        "study_context_used": study_context.copy()
    }
    selected_persona_dict_candidate: Optional[Dict[str, Any]] = None

    # 1. PersonaManager -> Get Persona (Instruction built directly in Python)
    print(f"\nENGINE: --- Building direct instruction for PersonaManagerAgent ---")

    roles_already_covered_prompt_segment = ""
    roles_list = [
        role for role in study_context.get("roles_interviewed_so_far", []) 
        if role and role != "UnknownRole"
    ]
    if not is_exploratory_phase and roles_list:
        roles_already_covered_prompt_segment = (
            f"\n**IMPORTANT: This is a list of 'roles_or_expertise_already_interviewed'**\n"
            f"Experts with the following roles/expertise areas have already been interviewed: {json.dumps(roles_list, ensure_ascii=False)}.\n"
            f"Your task is to select or create a persona that offers a *distinctly different perspective* or a different primary area of expertise "
            f"to maximize thematic diversity. DO NOT select a persona whose main role is already covered in the list above."
        )

    instruction_for_persona_manager = (
        f"OverallStudyTopic: {study_context.get('OverallStudyTopic')}\n\n"
        f"PersonaRequirementsGuidance: {study_context.get('PersonaRequirementsGuidance')}\n\n"
        f"PredefinedPersonas (for you to choose from if a suitable, diverse option exists): {json.dumps(study_context.get('PredefinedPersonas', []), ensure_ascii=False)}\n"
        f"{roles_already_covered_prompt_segment}\n\n"
        f"Based on all the information above, provide ONE suitable expert persona. "
        f"Output ONLY the final persona JSON object."
    )

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
    if phase_results["selected_persona_dict"] is None:
        phase_results["error_message"] = "Selected persona dictionary is None before conducting interview."
        print(f"!ENGINE ERROR: {phase_results['error_message']}")
        return phase_results

    interview_transcript_result = _conduct_single_interview(
        study_context_for_interview=study_context,
        selected_persona_dict=phase_results["selected_persona_dict"],
        is_exploratory=is_exploratory_phase,
        max_turns=max_interview_turns
    )
    phase_results["transcript"] = interview_transcript_result
    if not phase_results["transcript"]:
        phase_results["error_message_interview_loop"] = "Interview did not produce a transcript or an error occurred in _conduct_single_interview."
        

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

            full_prompt_for_summarizer = (
                f"{base_instruction_from_manager}\n\n" 
                f"OverallStudyTopic: {study_context.get('OverallStudyTopic')}\n"
                f"TargetYear: {study_context.get('TargetYear')}\n"
                f"Guidance on structure/output (from StudyContext's '{summarizer_guidance_key}'):\n"
                f"{study_context.get(summarizer_guidance_key, 'No specific structural guidance provided.')}\n\n"
                f"**Interview Transcript to Summarize:**\n" 
                f"```json\n{json.dumps(phase_results['transcript'], indent=2, ensure_ascii=False)}\n```\n\n"
                f"Please provide the required summary based on ALL the above information, especially focusing on the Interview Transcript and the provided Guidance."
            )
            
            print(f"\nENGINE: --- SummarizerAgent: Task -> Provide Summary ({'Exploratory' if is_exploratory_phase else 'Structured'}) ---")
            
            summarizer_response_obj = _run_agent_internal(SummarizerAgent, full_prompt_for_summarizer) 
            
            if summarizer_response_obj and summarizer_response_obj.final_output:
                phase_results["summary"] = summarizer_response_obj.final_output
            else: 
                phase_results["error_message"] = "SummarizerAgent failed to provide summary."
    elif not phase_results.get("error_message") and not phase_results.get("error_message_interview_loop"): 
         phase_results["error_message"] = "No transcript available to summarize for this round (or previous interview error)."
    
    if phase_results.get("error_message_interview_loop") and not phase_results.get("error_message"):
        phase_results["error_message"] = phase_results["error_message_interview_loop"]
    del phase_results["error_message_interview_loop"]

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
    manager_response_obj = _run_agent_internal(ManagerAgent, prompt_for_manager_formalize)
    if manager_response_obj and manager_response_obj.final_output:
        formalized_guides_dict = extract_json_from_response(manager_response_obj.final_output)
        if formalized_guides_dict and \
           formalized_guides_dict.get("InterviewGuideStructure_DEFINED") and \
           formalized_guides_dict.get("DesiredOutputCatalogStructureGuidance_DEFINED"):
            print("ENGINE: ManagerAgent successfully formalized structure.")
            return formalized_guides_dict
        else: 
            print(f"!ENGINE ERROR: ManagerAgent did not return both defined guides in JSON. Raw: {manager_response_obj.final_output}")
    else: 
        print("!ENGINE ERROR: ManagerAgent failed to formalize structure.")
    return None

def generate_final_catalog_from_summaries(study_context: Dict, aggregated_summaries: str) -> Optional[str]:
    print(f"\nENGINE: --- ManagerAgent: Task -> Formulate FINAL CatalogWriter Instruction (for Synthesis) ---")
    
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