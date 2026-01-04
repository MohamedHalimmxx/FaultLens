import sqlite3
import os
import time 
from concurrent.futures import ThreadPoolExecutor 
from typing import TypedDict, Optional, Dict, List 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from yolov8_crop import run_yolo_crop
from vision_agent import run_vision_agent
from comparison_agent import run_comparison_agent
from policy_agent import run_policy_agent
from db_ops import insert_reference_image

# --- Database Setup ---
conn = sqlite3.connect("database.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    reference_image TEXT,
    user_image TEXT 
)
""")
conn.commit()
#########################Database
insert_reference_image(
    order_id=909,
    image_path="images\\ref.jpg"
)

insert_reference_image(
    order_id=999,
    image_path="images\\Smart_ref.jpg"
)

# --- State Definition ---
class OrderState(TypedDict):
    order_id: int
    reference_image: str
    user_images: Optional[Dict[str, str]] 
    user_description: str
    cropped_images: Optional[Dict[str, object]] 
    defect_text: Optional[str]
    comparison_text: Optional[str]
    policy_text: Optional[str]


# --- Nodes (The Logic) ---

def yolo_crop_node(state: OrderState):
    """
    YOLO Crop
    """
    if not state.get('user_images'):
        print("--- Skipping YOLO (Chat Mode) ---")
        return {}
    
    image_items = list(state['user_images'].items())
    print(f"--- Starting Sequential YOLO on {len(image_items)} images ---")
    start_time = time.time()
    
    cropped_results = {}

    for view_name, img_path in image_items:
        if img_path:
            try:
                print(f"Processing view: {view_name}...")
                cropped = run_yolo_crop(img_path)
                if cropped is not None:
                    cropped_results[view_name] = cropped
            except Exception as e:
                print(f" Error processing {view_name}: {e}")

    print(f"---YOLO Finished in {time.time() - start_time:.2f} seconds---")     
    return {"cropped_images": cropped_results}


def vision_node(state: OrderState):
    """
    Vision Analysis
    """
    if not state.get('cropped_images'):
        print("--- Vision Node: No images to analyze ---")
        return {}
        
    if state.get('defect_text') and not state.get('user_images'):
        return {} 

    print(f"---Starting Parallel Vision Analysis on {len(state['cropped_images'])} images---")
    start_time = time.time()
    
    cropped_items = list(state['cropped_images'].items())

    def process_single_analysis(item):
        view_name, cropped_img = item
        analysis = run_vision_agent(cropped_img, state['user_description'])
        return f"View [{view_name}]: {analysis}"

    with ThreadPoolExecutor(max_workers=3) as executor:
        reports = list(executor.map(process_single_analysis, cropped_items))
    
    combined_defect_text = "\n".join(reports)
    
    print(f"---Vision Finished in {time.time() - start_time:.2f} seconds---")
    return {"defect_text": combined_defect_text}


def comparison_node(state: OrderState):
    """Comparison Node"""
    if not state.get('cropped_images'):
        print("--- Comparison Node: No images found, skipping... ---")
        return {} 
    
    if state.get('comparison_text') and not state.get('user_images'):
        print("---Comparison Node: Using existing memory---")
        return {}

    print(f"---Running Comparison Agent---")
    
    all_cropped_images = list(state['cropped_images'].values())
    
    enhanced_defect_context = f"""
    The following analysis is derived from {len(all_cropped_images)} different angles of the same product. 
    Please verify if these attributes match the single reference image provided.
    ---
    DETAILED ANALYSIS:
    {state.get('defect_text', '')}
    """

    comparison_text = run_comparison_agent(
        cropped_user_image=all_cropped_images[0],  
        reference_image=state['reference_image'],
        defect_text=enhanced_defect_context, 
        user_description=state['user_description']
    )
    
    return {"comparison_text": comparison_text}


def policy_node(state: OrderState):
    print("---Running Policy Agent---")
    
    if "WRONG_PRODUCT_CONFIRMED" in state.get('comparison_text', ''):
        return {
            "policy_text": "Based on the images provided, the product does not match our records for this Order ID."
        }
    
    policies_dir = "policies" 
    combined_policies = ""

    if os.path.exists(policies_dir):
        for file in os.listdir(policies_dir):
            if file.endswith(".txt"):
                with open(os.path.join(policies_dir, file), "r", encoding="utf-8") as f: 
                     combined_policies += f.read() + "\n---\n"

    policy_decision = run_policy_agent(
        comparison_text=state['comparison_text'],
        defect_text=state['defect_text'],
        user_description=state['user_description'],
        policies_combined_text=combined_policies
    )
    return {"policy_text": policy_decision}


def route_start(state: OrderState):
    """Route starting point"""
    if not state.get('user_images'):
        print("--- ROUTING TO POLICY (CHAT MODE) ---")
        return "policy_decision"
    else:
        print("--- ROUTING TO YOLO (INITIAL INSPECTION) ---")
        return "yolo_crop"


# ---- Build the Graph ---- #
workflow = StateGraph(OrderState)
workflow.add_node("yolo_crop", yolo_crop_node)
workflow.add_node("vision_analysis", vision_node)
workflow.add_node("comparison", comparison_node)
workflow.add_node("policy_decision", policy_node)

workflow.set_conditional_entry_point(
    route_start,
    {
        "yolo_crop": "yolo_crop",
        "policy_decision": "policy_decision"
    }
)
workflow.add_edge("yolo_crop", "vision_analysis")
workflow.add_edge("vision_analysis", "comparison")
workflow.add_edge("comparison", "policy_decision")
workflow.add_edge("policy_decision", END) 

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def run_langgraph_pipeline(order_id, user_images_dict, reference_image_path, user_description):
    
    config = {"configurable": {"thread_id": str(order_id)}}
    initial_inputs = {
        "order_id": order_id,
        "user_images": user_images_dict, 
        "reference_image": reference_image_path,
        "user_description": user_description,
        "cropped_images": None,
        "defect_text": "",
        "comparison_text": "",
        "policy_text": ""
    }

    result = app.invoke(initial_inputs, config=config)
    return {
        "Vision": {"defect_text": result.get("defect_text")},
        "Comparison": {"comparison_text": result.get("comparison_text")},
        "Policy": {"policy_decision": result.get("policy_text")}
    }