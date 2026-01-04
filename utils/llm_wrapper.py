import subprocess


def run_qwen2vl(prompt, image_paths=None):
    """
    Run local Ollama model Qwen2.5VL
    Args:
        prompt: str
        image_paths: str or list of str, optional, if you want to pass images
    Returns:
        str: model output
    """
    cmd = ["ollama", "run", "qwen2.5vl:7b"]

    if image_paths:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        cmd += image_paths

    cmd.append(prompt)

    result = subprocess.run(cmd, capture_output=True, text=False)
    stdout_decoded = result.stdout.decode('utf-8', errors='replace')
    return stdout_decoded.strip()

#def run_llm_vision(prompt):
#    output = text_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
#    return output[0]["generated_text"]