---
title: ProductMatchPro
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
---

# ProductMatchPro - Hydraulic Product Cross-Reference

A RAG-powered application for cross-referencing competitor hydraulic products against your company's catalogue.

## Features

- **Distributor Product Finder** - Chat interface where distributors enter a competitor model code and get the best equivalent with a confidence score
- **Admin Console** - Upload PDF catalogues and user guides, manage products, review feedback
- **Smart Matching** - 12-dimension weighted scoring across pressure, flow, coil voltage, mounting, spool type, and more
- **Fuzzy Model Code Lookup** - Partial codes accepted (e.g. "DG4V-3" finds "DG4V-3-2A-M-U-H7-60")
- **75% Confidence Threshold** - Below threshold directs distributors to contact their sales representative

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key (required for PDF extraction and query parsing):
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. Run the combined app:
   ```bash
   python app.py
   ```

   Or run each app separately:
   ```bash
   python distributor_app.py  # Port 7860
   python admin_app.py        # Port 7861
   ```

## Architecture

- **LangGraph** StateGraph with MemorySaver for conversation persistence
- **Numpy-based vector store** with sentence-transformers embeddings + cross-encoder reranking
- **SQLite** for structured product data, model code patterns, confirmed equivalents, and feedback
- **Gradio** for both the distributor and admin interfaces
