import json
import re
import ollama  

def determine_required_views(product_category: str):
    
    prompt = f"""
    You are an expert quality assurance inspector. 
    For the product category: "{product_category}", list exactly 2 to 4 essential camera angles/views needed to inspect it for defects properly.
    
    Rules:
    1. Output MUST be a valid JSON list of strings.
    2. Do NOT write markdown formatting (like ```json).
    3. Do NOT add any conversational text. Just the list. 
    
    Example output:
    ["Front View", "Back View", "Label Details", etc.]
    """
    
    try:
        # call Ollama 
        response = ollama.chat(
            model='phi3', 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0} 
        )
        
        content = response['message']['content']
        
       
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            views = json.loads(json_str)
            return views
        else:
            print(f"Warning: Could not parse JSON from Ollama response: {content}")
            return ["Front View", "Back View", "Close-up of Defect"]

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        # Erorr handling 
        return ["Front View", "Back View", "Close-up of Defect"]


# --- n-test ---
if __name__ == "__main__":
    # negarb
    category = "Running Shoes"
    print(f"Testing for {category}...")
    print(determine_required_views(category))