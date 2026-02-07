"""
ProductMatchPro - Hugging Face Spaces Entry Point
Combines the distributor app and admin console into a single tabbed interface.

For local development, you can run each app separately:
  - Distributor: python distributor_app.py (port 7860)
  - Admin: python admin_app.py (port 7861)

Environment variables for auth (optional, recommended for HF Spaces):
  - ADMIN_USERNAME / ADMIN_PASSWORD: Protects the entire combined app
"""

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

from distributor_app import distributor_ui
from admin_app import admin_ui, MAX_UPLOAD_SIZE_MB

with gr.Blocks(title="ProductMatchPro - Hydraulic Product Cross-Reference") as app:
    gr.Markdown("# ProductMatchPro - Hydraulic Product Cross-Reference")
    with gr.Tabs():
        with gr.Tab("Product Finder"):
            distributor_ui.render()
        with gr.Tab("Admin Console"):
            admin_ui.render()

if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.getenv("PORT", "7860")),
        "max_file_size": f"{MAX_UPLOAD_SIZE_MB}mb",
    }
    # Enable auth if credentials are set
    admin_user = os.getenv("ADMIN_USERNAME")
    admin_pass = os.getenv("ADMIN_PASSWORD")
    if admin_user and admin_pass:
        launch_kwargs["auth"] = (admin_user, admin_pass)
    app.launch(**launch_kwargs)
