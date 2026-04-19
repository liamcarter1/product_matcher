/**
 * Danfoss RAG Chat Widget
 * Embeddable chatbot for Danfoss distributor websites
 *
 * Usage:
 * <script
 *   src="danfoss-chat-widget.js"
 *   data-api-url="https://your-api.com"
 *   data-primary-color="#E2000F"
 *   data-title="Danfoss Product Assistant">
 * </script>
 */

(function() {
    'use strict';

    // Configuration from script attributes
    const scriptTag = document.currentScript;
    const config = {
        apiUrl: scriptTag?.getAttribute('data-api-url') || 'http://localhost:8000',
        primaryColor: scriptTag?.getAttribute('data-primary-color') || '#E2000F',
        title: scriptTag?.getAttribute('data-title') || 'Danfoss Product Assistant',
        position: scriptTag?.getAttribute('data-position') || 'bottom-right'
    };

    // Session management
    const SESSION_KEY = 'danfoss_chat_session_id';
    const HISTORY_KEY = 'danfoss_chat_history';

    function getSessionId() {
        let sessionId = localStorage.getItem(SESSION_KEY);
        if (!sessionId) {
            sessionId = 'sess_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem(SESSION_KEY, sessionId);
        }
        return sessionId;
    }

    function getStoredHistory() {
        try {
            const stored = localStorage.getItem(HISTORY_KEY);
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            return [];
        }
    }

    function saveHistory(messages) {
        try {
            // Keep only last 50 messages
            const toSave = messages.slice(-50);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(toSave));
        } catch (e) {
            // Storage might be full
        }
    }

    // Inject CSS
    function injectStyles() {
        const css = `
            .danfoss-chat-widget {
                --primary-color: ${config.primaryColor};
                --secondary-color: #333333;
                --bg-color: #FFFFFF;
                --text-color: #1A1A1A;
                --border-radius: 12px;
                --shadow: 0 4px 20px rgba(0, 0, 0, 0.15);

                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }

            .danfoss-chat-button {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background-color: var(--primary-color);
                border: none;
                cursor: pointer;
                box-shadow: var(--shadow);
                display: flex;
                align-items: center;
                justify-content: center;
                transition: transform 0.2s, box-shadow 0.2s;
                z-index: 999998;
            }

            .danfoss-chat-button:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 25px rgba(0, 0, 0, 0.2);
            }

            .danfoss-chat-button svg {
                width: 28px;
                height: 28px;
                fill: white;
            }

            .danfoss-chat-window {
                position: fixed;
                bottom: 90px;
                right: 20px;
                width: 380px;
                height: 520px;
                background: var(--bg-color);
                border-radius: var(--border-radius);
                box-shadow: var(--shadow);
                display: none;
                flex-direction: column;
                overflow: hidden;
                z-index: 999999;
            }

            .danfoss-chat-window.open {
                display: flex;
            }

            .danfoss-chat-header {
                background: var(--primary-color);
                color: white;
                padding: 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }

            .danfoss-chat-header-title {
                font-weight: 600;
                font-size: 16px;
            }

            .danfoss-chat-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                padding: 4px;
                font-size: 20px;
                line-height: 1;
            }

            .danfoss-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .danfoss-chat-message {
                max-width: 85%;
                padding: 10px 14px;
                border-radius: 12px;
                word-wrap: break-word;
            }

            .danfoss-chat-message.user {
                align-self: flex-end;
                background: var(--primary-color);
                color: white;
                border-bottom-right-radius: 4px;
            }

            .danfoss-chat-message.assistant {
                align-self: flex-start;
                background: #f0f0f0;
                color: var(--text-color);
                border-bottom-left-radius: 4px;
            }

            .danfoss-chat-confidence {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                font-size: 11px;
                margin-top: 6px;
                padding: 2px 8px;
                border-radius: 10px;
                background: rgba(0, 0, 0, 0.1);
            }

            .danfoss-confidence-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }

            .danfoss-confidence-high {
                background-color: #22c55e;
            }

            .danfoss-confidence-medium {
                background-color: #eab308;
            }

            .danfoss-confidence-low {
                background-color: #ef4444;
            }

            .danfoss-chat-disclaimer {
                font-size: 11px;
                color: #666;
                margin-top: 8px;
                padding: 8px;
                background: #fff3cd;
                border-radius: 6px;
                border-left: 3px solid #ffc107;
            }

            .danfoss-chat-input-area {
                padding: 12px;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 8px;
            }

            .danfoss-chat-input {
                flex: 1;
                padding: 10px 14px;
                border: 1px solid #e0e0e0;
                border-radius: 20px;
                outline: none;
                font-size: 14px;
            }

            .danfoss-chat-input:focus {
                border-color: var(--primary-color);
            }

            .danfoss-chat-send {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: var(--primary-color);
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .danfoss-chat-send:disabled {
                background: #ccc;
                cursor: not-allowed;
            }

            .danfoss-chat-send svg {
                width: 18px;
                height: 18px;
                fill: white;
            }

            .danfoss-typing-indicator {
                display: flex;
                gap: 4px;
                padding: 12px 14px;
            }

            .danfoss-typing-dot {
                width: 8px;
                height: 8px;
                background: #999;
                border-radius: 50%;
                animation: danfoss-typing 1.4s infinite ease-in-out;
            }

            .danfoss-typing-dot:nth-child(1) { animation-delay: 0s; }
            .danfoss-typing-dot:nth-child(2) { animation-delay: 0.2s; }
            .danfoss-typing-dot:nth-child(3) { animation-delay: 0.4s; }

            @keyframes danfoss-typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-4px); }
            }

            .danfoss-chat-welcome {
                text-align: center;
                color: #666;
                padding: 20px;
            }

            .danfoss-chat-welcome h3 {
                margin: 0 0 8px 0;
                color: var(--text-color);
            }

            .danfoss-quick-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 12px;
                justify-content: center;
            }

            .danfoss-quick-action {
                padding: 6px 12px;
                border: 1px solid var(--primary-color);
                border-radius: 16px;
                background: white;
                color: var(--primary-color);
                font-size: 12px;
                cursor: pointer;
                transition: background 0.2s, color 0.2s;
            }

            .danfoss-quick-action:hover {
                background: var(--primary-color);
                color: white;
            }

            /* Image upload button */
            .danfoss-image-btn {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: none;
                border: 1px solid #e0e0e0;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                transition: border-color 0.2s, color 0.2s;
                flex-shrink: 0;
            }

            .danfoss-image-btn:hover {
                border-color: var(--primary-color);
                color: var(--primary-color);
            }

            /* Image preview */
            .danfoss-image-preview {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border-top: 1px solid #e0e0e0;
                background: #f9f9f9;
            }

            .danfoss-image-preview-img {
                width: 48px;
                height: 48px;
                object-fit: cover;
                border-radius: 6px;
                border: 1px solid #ddd;
            }

            .danfoss-image-preview-remove {
                background: none;
                border: none;
                font-size: 20px;
                color: #999;
                cursor: pointer;
                margin-left: auto;
                padding: 0 4px;
            }

            .danfoss-image-preview-remove:hover {
                color: var(--primary-color);
            }

            /* Nameplate confirmation card */
            .danfoss-nameplate-card {
                background: white !important;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 14px !important;
                max-width: 95% !important;
            }

            .danfoss-nameplate-title {
                font-weight: 600;
                font-size: 13px;
                margin-bottom: 10px;
                color: var(--secondary-color);
            }

            .danfoss-nameplate-field {
                margin-bottom: 8px;
            }

            .danfoss-nameplate-field label {
                display: block;
                font-size: 11px;
                color: #666;
                margin-bottom: 2px;
            }

            .danfoss-nameplate-input {
                width: 100%;
                padding: 6px 10px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 13px;
                box-sizing: border-box;
            }

            .danfoss-nameplate-input:focus {
                outline: none;
                border-color: var(--primary-color);
            }

            .danfoss-nameplate-specs {
                font-size: 12px;
                color: #555;
                margin-bottom: 10px;
                padding: 6px 8px;
                background: #f5f5f5;
                border-radius: 6px;
            }

            .danfoss-nameplate-spec {
                margin-bottom: 2px;
            }

            .danfoss-nameplate-actions {
                display: flex;
                gap: 8px;
                margin-top: 10px;
            }

            .danfoss-nameplate-confirm {
                flex: 1;
                padding: 8px 12px;
                background: var(--primary-color);
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
            }

            .danfoss-nameplate-confirm:hover {
                opacity: 0.9;
            }

            .danfoss-nameplate-cancel {
                padding: 8px 12px;
                background: none;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 12px;
                color: #666;
                cursor: pointer;
            }

            .danfoss-nameplate-cancel:hover {
                border-color: #999;
            }

            /* Mobile responsive */
            @media (max-width: 480px) {
                .danfoss-chat-window {
                    width: calc(100% - 20px);
                    height: calc(100% - 100px);
                    right: 10px;
                    bottom: 80px;
                    border-radius: 12px;
                }

                .danfoss-chat-button {
                    bottom: 15px;
                    right: 15px;
                }
            }
        `;

        const style = document.createElement('style');
        style.textContent = css;
        document.head.appendChild(style);
    }

    // Create widget HTML
    function createWidget() {
        const widget = document.createElement('div');
        widget.className = 'danfoss-chat-widget';
        widget.innerHTML = `
            <button class="danfoss-chat-button" aria-label="Open chat">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
                </svg>
            </button>
            <div class="danfoss-chat-window">
                <div class="danfoss-chat-header">
                    <span class="danfoss-chat-header-title">${config.title}</span>
                    <button class="danfoss-chat-close" aria-label="Close chat">&times;</button>
                </div>
                <div class="danfoss-chat-messages">
                    <div class="danfoss-chat-welcome">
                        <h3>Welcome!</h3>
                        <p>I can help you find Danfoss parts and answer technical questions.</p>
                        <div class="danfoss-quick-actions">
                            <button class="danfoss-quick-action" data-query="What Danfoss parts do you have?">Browse parts</button>
                            <button class="danfoss-quick-action" data-query="How do I find a replacement part?">Find replacement</button>
                        </div>
                    </div>
                </div>
                <div class="danfoss-image-preview" style="display:none;">
                    <img class="danfoss-image-preview-img" src="" alt="Selected image" />
                    <button class="danfoss-image-preview-remove" aria-label="Remove image">&times;</button>
                </div>
                <div class="danfoss-chat-input-area">
                    <input type="file" class="danfoss-image-file-input" accept="image/*" capture="environment" style="display:none;" />
                    <button class="danfoss-image-btn" aria-label="Upload nameplate image">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" width="20" height="20">
                            <path fill="currentColor" d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4z"/>
                            <path fill="currentColor" d="M9 2L7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/>
                        </svg>
                    </button>
                    <input type="text" class="danfoss-chat-input" placeholder="Ask about Danfoss products..." />
                    <button class="danfoss-chat-send" aria-label="Send message">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(widget);
        return widget;
    }

    // Chat functionality
    class DanfossChatWidget {
        constructor(widget) {
            this.widget = widget;
            this.button = widget.querySelector('.danfoss-chat-button');
            this.window = widget.querySelector('.danfoss-chat-window');
            this.closeBtn = widget.querySelector('.danfoss-chat-close');
            this.messagesContainer = widget.querySelector('.danfoss-chat-messages');
            this.input = widget.querySelector('.danfoss-chat-input');
            this.sendBtn = widget.querySelector('.danfoss-chat-send');
            this.imageBtn = widget.querySelector('.danfoss-image-btn');
            this.fileInput = widget.querySelector('.danfoss-image-file-input');
            this.imagePreview = widget.querySelector('.danfoss-image-preview');
            this.imagePreviewImg = widget.querySelector('.danfoss-image-preview-img');
            this.imagePreviewRemove = widget.querySelector('.danfoss-image-preview-remove');

            this.sessionId = getSessionId();
            this.messages = getStoredHistory();
            this.isLoading = false;
            this.selectedImageFile = null;

            this.bindEvents();
            this.restoreHistory();
        }

        bindEvents() {
            this.button.addEventListener('click', () => this.toggle());
            this.closeBtn.addEventListener('click', () => this.close());
            this.sendBtn.addEventListener('click', () => this.handleSend());
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleSend();
                }
            });

            // Image upload
            this.imageBtn.addEventListener('click', () => this.fileInput.click());
            this.fileInput.addEventListener('change', (e) => this.handleImageSelect(e));
            this.imagePreviewRemove.addEventListener('click', () => this.clearSelectedImage());

            // Quick action buttons
            this.widget.querySelectorAll('.danfoss-quick-action').forEach(btn => {
                btn.addEventListener('click', () => {
                    const query = btn.getAttribute('data-query');
                    this.input.value = query;
                    this.handleSend();
                });
            });
        }

        toggle() {
            this.window.classList.toggle('open');
            if (this.window.classList.contains('open')) {
                this.input.focus();
            }
        }

        close() {
            this.window.classList.remove('open');
        }

        restoreHistory() {
            if (this.messages.length > 0) {
                // Clear welcome message
                this.messagesContainer.innerHTML = '';

                // Restore messages
                this.messages.forEach(msg => {
                    this.addMessageToUI(msg.role, msg.content, msg.confidence, msg.confidence_level, msg.disclaimer);
                });
            }
        }

        addMessageToUI(role, content, confidence = null, confidenceLevel = null, disclaimer = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `danfoss-chat-message ${role}`;

            let html = `<div class="message-content">${this.escapeHtml(content)}</div>`;

            if (role === 'assistant' && confidence !== null) {
                const dotClass = `danfoss-confidence-${confidenceLevel || 'medium'}`;
                html += `
                    <div class="danfoss-chat-confidence">
                        <span class="danfoss-confidence-dot ${dotClass}"></span>
                        <span>${Math.round(confidence)}% confidence</span>
                    </div>
                `;
            }

            if (disclaimer) {
                html += `<div class="danfoss-chat-disclaimer">${this.escapeHtml(disclaimer)}</div>`;
            }

            messageDiv.innerHTML = html;
            this.messagesContainer.appendChild(messageDiv);
            this.scrollToBottom();
        }

        showTypingIndicator() {
            const typing = document.createElement('div');
            typing.className = 'danfoss-chat-message assistant danfoss-typing-indicator';
            typing.id = 'danfoss-typing';
            typing.innerHTML = `
                <span class="danfoss-typing-dot"></span>
                <span class="danfoss-typing-dot"></span>
                <span class="danfoss-typing-dot"></span>
            `;
            this.messagesContainer.appendChild(typing);
            this.scrollToBottom();
        }

        hideTypingIndicator() {
            const typing = document.getElementById('danfoss-typing');
            if (typing) {
                typing.remove();
            }
        }

        scrollToBottom() {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        handleImageSelect(e) {
            const file = e.target.files[0];
            if (!file) return;

            // Validate file size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('Image is too large. Maximum size is 10MB.');
                this.fileInput.value = '';
                return;
            }

            // Validate file type
            const allowed = ['image/jpeg', 'image/png', 'image/webp'];
            if (!allowed.includes(file.type)) {
                alert('Unsupported image type. Please use JPEG, PNG, or WebP.');
                this.fileInput.value = '';
                return;
            }

            this.selectedImageFile = file;
            const url = URL.createObjectURL(file);
            this.imagePreviewImg.src = url;
            this.imagePreview.style.display = 'flex';
        }

        clearSelectedImage() {
            this.selectedImageFile = null;
            this.fileInput.value = '';
            this.imagePreview.style.display = 'none';
            this.imagePreviewImg.src = '';
        }

        handleSend() {
            if (this.selectedImageFile) {
                this.sendImageMessage();
            } else {
                this.sendMessage();
            }
        }

        async sendImageMessage() {
            if (this.isLoading || !this.selectedImageFile) return;

            // Clear welcome message if first message
            if (this.messages.length === 0) {
                const welcome = this.messagesContainer.querySelector('.danfoss-chat-welcome');
                if (welcome) welcome.remove();
            }

            // Add user message with image indicator
            this.messages.push({ role: 'user', content: '[Nameplate image uploaded]' });
            this.addMessageToUI('user', '[Nameplate image uploaded]');

            // Show loading
            this.isLoading = true;
            this.sendBtn.disabled = true;
            this.showTypingIndicator();

            const formData = new FormData();
            formData.append('image', this.selectedImageFile);
            formData.append('session_id', this.sessionId);

            this.clearSelectedImage();

            try {
                const response = await fetch(`${config.apiUrl}/api/chat/image`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({}));
                    throw new Error(err.detail || 'Failed to analyze image');
                }

                const data = await response.json();
                this.hideTypingIndicator();

                if (data.session_id) {
                    this.sessionId = data.session_id;
                    localStorage.setItem(SESSION_KEY, this.sessionId);
                }

                // If nameplate data was extracted, show confirmation card
                if (data.parsed_nameplate) {
                    this.showNameplateConfirmation(data.parsed_nameplate, data);
                } else {
                    // Fallback: show response directly
                    this.addAssistantResponse(data);
                }

                saveHistory(this.messages);

            } catch (error) {
                this.hideTypingIndicator();
                this.addMessageToUI(
                    'assistant',
                    'Sorry, I could not analyze the image. Please try a clearer photo.',
                    null, 'low', null
                );
                console.error('Danfoss Chat Error:', error);
            } finally {
                this.isLoading = false;
                this.sendBtn.disabled = false;
            }
        }

        showNameplateConfirmation(nameplate, fullResponse) {
            const card = document.createElement('div');
            card.className = 'danfoss-chat-message assistant danfoss-nameplate-card';

            const manufacturer = nameplate.manufacturer || 'Unknown';
            const modelNumber = nameplate.model_number || 'Unknown';

            let specsHtml = '';
            if (nameplate.specifications && Object.keys(nameplate.specifications).length > 0) {
                specsHtml = '<div class="danfoss-nameplate-specs">';
                for (const [key, val] of Object.entries(nameplate.specifications)) {
                    specsHtml += `<div class="danfoss-nameplate-spec"><strong>${this.escapeHtml(key)}:</strong> ${this.escapeHtml(val)}</div>`;
                }
                specsHtml += '</div>';
            }

            card.innerHTML = `
                <div class="danfoss-nameplate-title">Nameplate Data Extracted</div>
                <div class="danfoss-nameplate-field">
                    <label>Manufacturer</label>
                    <input type="text" class="danfoss-nameplate-input" data-field="manufacturer" value="${this.escapeHtml(manufacturer)}" />
                </div>
                <div class="danfoss-nameplate-field">
                    <label>Model Number</label>
                    <input type="text" class="danfoss-nameplate-input" data-field="model_number" value="${this.escapeHtml(modelNumber)}" />
                </div>
                ${specsHtml}
                <div class="danfoss-nameplate-actions">
                    <button class="danfoss-nameplate-confirm">Search for Danfoss Equivalent</button>
                    <button class="danfoss-nameplate-cancel">Cancel</button>
                </div>
            `;

            this.messagesContainer.appendChild(card);
            this.scrollToBottom();

            // Confirm button — send edited values as a normal chat message
            card.querySelector('.danfoss-nameplate-confirm').addEventListener('click', () => {
                const mfr = card.querySelector('[data-field="manufacturer"]').value.trim();
                const model = card.querySelector('[data-field="model_number"]').value.trim();

                const parts = [mfr, model].filter(Boolean);
                const query = parts.length > 0
                    ? `Find Danfoss equivalent for ${parts.join(' ')}`
                    : `Find Danfoss equivalent for: ${nameplate.raw_text.slice(0, 200)}`;

                card.remove();
                this.input.value = query;
                this.sendMessage();
            });

            // Cancel button
            card.querySelector('.danfoss-nameplate-cancel').addEventListener('click', () => {
                card.remove();
            });
        }

        addAssistantResponse(data) {
            this.messages.push({
                role: 'assistant',
                content: data.response,
                confidence: data.confidence,
                confidence_level: data.confidence_level,
                disclaimer: data.disclaimer
            });
            this.addMessageToUI(
                'assistant', data.response, data.confidence,
                data.confidence_level, data.disclaimer
            );
        }

        async sendMessage() {
            const message = this.input.value.trim();
            if (!message || this.isLoading) return;

            // Clear welcome message if first message
            if (this.messages.length === 0) {
                const welcome = this.messagesContainer.querySelector('.danfoss-chat-welcome');
                if (welcome) welcome.remove();
            }

            // Add user message
            this.messages.push({ role: 'user', content: message });
            this.addMessageToUI('user', message);
            this.input.value = '';

            // Show loading
            this.isLoading = true;
            this.sendBtn.disabled = true;
            this.showTypingIndicator();

            try {
                const response = await fetch(`${config.apiUrl}/api/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        session_id: this.sessionId
                    })
                });

                if (!response.ok) {
                    throw new Error('Failed to get response');
                }

                const data = await response.json();

                // Add assistant message
                this.messages.push({
                    role: 'assistant',
                    content: data.response,
                    confidence: data.confidence,
                    confidence_level: data.confidence_level,
                    disclaimer: data.disclaimer
                });

                this.hideTypingIndicator();
                this.addMessageToUI(
                    'assistant',
                    data.response,
                    data.confidence,
                    data.confidence_level,
                    data.disclaimer
                );

                // Update session ID if returned
                if (data.session_id) {
                    this.sessionId = data.session_id;
                    localStorage.setItem(SESSION_KEY, this.sessionId);
                }

                // Save history
                saveHistory(this.messages);

            } catch (error) {
                this.hideTypingIndicator();
                this.addMessageToUI(
                    'assistant',
                    'Sorry, I encountered an error. Please try again.',
                    null,
                    'low',
                    null
                );
                console.error('Danfoss Chat Error:', error);
            } finally {
                this.isLoading = false;
                this.sendBtn.disabled = false;
            }
        }
    }

    // Initialize widget when DOM is ready
    function init() {
        injectStyles();
        const widget = createWidget();
        new DanfossChatWidget(widget);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
