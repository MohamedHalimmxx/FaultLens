from utils.llm_wrapper import run_qwen2vl

def run_vision_agent(cropped_image_path, product_description):
    prompt = f"""
You are a professional product quality inspector.
Analyze the product in the image.
Description: {product_description}
Explain defect type, location, and severity in plain language.
"""
    return run_qwen2vl(prompt, image_paths=cropped_image_path)


