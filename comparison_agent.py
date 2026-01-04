from utils.llm_wrapper import run_qwen2vl

def run_comparison_agent(defect_text, reference_image, user_description, cropped_user_image):
    prompt = f"""
    You are a specialized Visual QA Agent for e-commerce returns.
    
    ### TASK:
    Determine if the "User's Item" is the SAME MODEL as the "Reference Image".
    
    ### INPUTS:
    1. USER COMPLAINT: {user_description}
    2. VISUAL ANALYSIS OF USER ITEM: {defect_text}
    3. IMAGES: Provided below.

    ### INSTRUCTIONS:
    - Ignore lighting, background, or minor wear.
    - Focus on: Brand Logo, Shape, Buttons/Ports placement, and distinctive design features.
    
    ### OUTPUT DECISION RULES:
    1. If features match -> Output: "VERDICT: MATCH"
    2. If it is a different product/brand/model -> Output: "VERDICT: WRONG PRODUCT"
    
    ### FINAL RESPONSE FORMAT:
    Start immediately with the verdict. Example: "VERDICT: MATCH. Reasoning: The logo and shape align perfectly."
    """

    response = run_qwen2vl(prompt, [cropped_user_image, reference_image])
    return response

