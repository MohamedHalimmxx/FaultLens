import gradio as gr
from langgraph_flow import run_langgraph_pipeline
from category_agent import determine_required_views
import sqlite3
import time

# --- Logic Functions ---

def get_product_requirements(category_name):
    """
    AI to determine required views based on category
    """
    if not category_name:
        return [gr.update(visible=True, value="Please enter a category.")] + [gr.update(visible=False)] * 4 + [gr.update(visible=False)]
    
    required_views = determine_required_views(category_name) 
    
    updates = [gr.update(visible=False, value="")] 
    
    image_updates = []
    stored_views = []
    
    for i in range(4):
        if i < len(required_views):
            view_label = required_views[i]
            image_updates.append(gr.update(visible=True, label=f"Upload: {view_label}"))
            stored_views.append(view_label)
        else:
            image_updates.append(gr.update(visible=False, value=None))
    
    form_visibility = gr.update(visible=True)
    return updates + image_updates + [form_visibility, stored_views]


def process_final_submission(order_id, user_desc, view_names, img1, img2, img3, img4):
    """
    collect images and start initial analysis
    """
    images_list = [img1, img2, img3, img4]
    user_images_dict = {}
    missing_files = False
    
    for i, view_label in enumerate(view_names):
        if images_list[i] is None:
            missing_files = True
            break
        user_images_dict[view_label] = images_list[i]
        
    if not order_id or not user_desc or missing_files:
        yield gr.update(visible=True), gr.update(visible=False), "Please upload all required photos and fill in Order ID & Description.", []
        return

    # --- Database ---
    reference_image_path = None
    try:
        with sqlite3.connect("database.db") as conn:
            c = conn.cursor() 
            row = c.execute("SELECT reference_image FROM orders WHERE order_id=?", (order_id,)).fetchone()
            if row:
                reference_image_path = row[0]
            else:
                yield gr.update(visible=True), gr.update(visible=False), "Order ID not found.", []
                return
    except Exception as e:
        yield gr.update(visible=True), gr.update(visible=False), f"DB Error: {e}", []
        return
    

    # ---Switch to Chat View---
    current_history = [{"role": "user", "content": f"Order: {order_id}\nIssue: {user_desc}"}]
    current_history.append({"role": "assistant", "content": "Analyzing and comparing images..."})
    yield gr.update(visible=False), gr.update(visible=True), "", current_history

    # ---Run Pipeline---
    try:
        outputs = run_langgraph_pipeline(
            order_id=order_id, 
            user_images_dict=user_images_dict, 
            reference_image_path=reference_image_path, 
            user_description=user_desc
        )
        initial_reply = outputs['Policy']['policy_decision']
        current_history[-1]["content"] = initial_reply
        yield gr.update(visible=False), gr.update(visible=True), "", current_history
    except Exception as e:
        current_history[-1]["content"] = f"System Error: {e}"
        yield gr.update(visible=False), gr.update(visible=True), "", current_history


def add_user_message(user_msg, history):
    if not user_msg.strip():
        return "", history
    history.append({"role": "user", "content": user_msg})
    return "", history


def chat_response(history, order_id):
    if not history or history[-1]["role"] != "user":
        yield history
        return

    user_msg = history[-1]["content"]
    try:
        outputs = run_langgraph_pipeline(order_id=order_id, user_images_dict=None, reference_image_path=None, user_description=user_msg)
        ai_reply = outputs['Policy']['policy_decision']
    except Exception as e:
        ai_reply = f"Error: {e}"

    history.append({"role": "assistant", "content": ""})
    for char in ai_reply:
        history[-1]["content"] += char
        time.sleep(0.005) 
        yield history


# --- UI ---
with gr.Blocks(title="FaultLens AI") as demo:
    stored_order_id = gr.State()
    stored_view_names = gr.State([]) 
    gr.Markdown("# FaultLens AI")
    
    # --- Stage 1: Category & Uploads ---
    with gr.Column(visible=True) as inspection_stage:
        with gr.Row():
            cat_input = gr.Textbox(label="Product Category", placeholder="e.g. Smartphone, Running Shoes...")
            cat_btn = gr.Button("Get Requirements", variant="secondary")
        
        cat_status = gr.Markdown("", visible=False)
        
        with gr.Row():
            img1 = gr.Image(type="filepath", visible=False, label="Image 1")
            img2 = gr.Image(type="filepath", visible=False, label="Image 2")
            img3 = gr.Image(type="filepath", visible=False, label="Image 3")
            img4 = gr.Image(type="filepath", visible=False, label="Image 4")
        
        with gr.Column(visible=False) as details_section:
            gr.Markdown("### Enter Complaint Details")
            with gr.Row():
                order_input = gr.Textbox(label="Order ID", placeholder="e.g. 12345")
                desc_input = gr.Textbox(label="Description", placeholder="Describe the defect...", lines=2)
            
            submit_btn = gr.Button("Start Analysis", variant="primary", size="lg")
            error_output = gr.Markdown("")

    # --- Stage 2: Chat ---
    with gr.Column(visible=False) as chat_stage:
        gr.Markdown("Chat with FaultLens AI")
        chatbot = gr.Chatbot(label="Conversation History", height=500, show_label=False)
        
        with gr.Row():
            msg_input = gr.Textbox(placeholder="Have questions? Ask me here", scale=4, container=False)
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        reset_btn = gr.Button("Back to Start", size="sm")


    # --- Event Listeners ---
    cat_btn.click(
        fn=get_product_requirements,
        inputs=[cat_input],
        outputs=[cat_status, img1, img2, img3, img4, details_section, stored_view_names]
    )
    
    submit_btn.click(
        fn=process_final_submission,
        inputs=[order_input, desc_input, stored_view_names, img1, img2, img3, img4],
        outputs=[inspection_stage, chat_stage, error_output, chatbot]
    ).then(
        fn=lambda x: x, inputs=[order_input], outputs=[stored_order_id]
    )

    chat_event = {
        "fn": add_user_message,
        "inputs": [msg_input, chatbot],
        "outputs": [msg_input, chatbot],
        "queue": False
    }

    msg_input.submit(**chat_event).then(
        fn=chat_response, 
        inputs=[chatbot, stored_order_id],
        outputs=[chatbot],
        show_progress="minimal"
        )
    
    
    send_btn.click(**chat_event).then(
        fn=chat_response,
        inputs=[chatbot, stored_order_id], 
        outputs=[chatbot], 
        show_progress="minimal"
        )

    reset_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False), "", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)),
        outputs=[inspection_stage, chat_stage, error_output, chatbot, img1, img2, img3, img4, details_section]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())