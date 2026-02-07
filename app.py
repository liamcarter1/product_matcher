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

app = gr.TabbedInterface(
    interface_list=[distributor_ui, admin_ui],
    tab_names=["Product Finder", "Admin Console"],
    title="ProductMatchPro - Hydraulic Product Cross-Reference",
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
