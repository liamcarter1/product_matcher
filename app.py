"""
ProductMatchPro - Hugging Face Spaces Entry Point
Combines the distributor app and admin console into a single tabbed interface.

For local development, you can run each app separately:
  - Distributor: python distributor_app.py (port 7860)
  - Admin: python admin_app.py (port 7861)
"""

import gradio as gr

from distributor_app import distributor_ui
from admin_app import admin_ui

with gr.Blocks(title="ProductMatchPro - Hydraulic Product Cross-Reference") as app:
    gr.Markdown("# ProductMatchPro - Hydraulic Product Cross-Reference")
    with gr.Tabs():
        with gr.Tab("Product Finder"):
            distributor_ui.render()
        with gr.Tab("Admin Console"):
            admin_ui.render()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
