import gradio as gr
from huggingface_hub import InferenceClient

# Logic: Connects to the AI
def respond(message, history, system_message, max_tokens, temperature, top_p, hf_token: gr.OAuthToken = None):
    if hf_token is None or hf_token.token is None:
        yield "### 🌿 Welcome to Serenity\n\nPlease **Login with Hugging Face** in the sidebar to begin."
        return

    client = InferenceClient(token=hf_token.token, model="Qwen/Qwen2.5-7B-Instruct")
    
    messages = [{"role": "system", "content": system_message}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        for message_chunk in client.chat_completion(
            messages, 
            max_tokens=max_tokens, 
            stream=True, 
            temperature=temperature, 
            top_p=top_p
        ):
            token = message_chunk.choices[0].delta.content or ""
            response += token
            yield response
    except Exception as e:
        yield f"**I'm having a moment of silence.** (Error: {str(e)})"

# Fixed CSS: Forces Black Text on White Bubbles for visibility
CSS = """
.gradio-container { background: linear-gradient(135deg, #f0f4f0 0%, #f7f3f0 100%) !important; font-family: 'DM Sans', sans-serif !important; }
#header { text-align: center; padding: 20px; border-bottom: 1px solid #dce8dc; }
#header h1 { color: #5c7a5c !important; font-size: 2.8rem !important; font-weight: 300; margin: 0; }

/* Bot Message Styling: White Background, Black Text */
.message.bot { 
    background-color: #ffffff !important; 
    color: #000000 !important; 
    border: 1px solid #e0e0e0 !important; 
    border-radius: 15px 15px 15px 2px !important;
}
.message.bot p, .message.bot span { color: #000000 !important; }

/* User Message Styling: Sage Green Background, White Text */
.message.user { 
    background-color: #5c7a5c !important; 
    color: #ffffff !important; 
    border-radius: 15px 15px 2px 15px !important; 
}
.message.user p, .message.user span { color: #ffffff !important; }

footer { text-align: center; margin-top: 20px; color: #9e9286; font-size: 0.85rem; }
"""

with gr.Blocks(title="Health.ai Serenity") as demo:
    with gr.Sidebar():
        gr.Markdown("### 🌿 Serenity Control")
        gr.LoginButton()
        gr.Markdown("---")
        sys_msg = gr.Textbox(
            value="You are Serenity, an empathetic mental health AI. Provide support and active listening.", 
            label="System Message"
        )
        tkns = gr.Slider(1, 2048, 512, label="Max Tokens")
        temp = gr.Slider(0.1, 2.0, 0.7, label="Creativity")
        top_p = gr.Slider(0.1, 1.0, 0.95, label="Focus")

    with gr.Column():
        gr.HTML("<div id='header'><h1>🌿 Serenity</h1><p>Your empathetic wellness companion.</p></div>")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[sys_msg, tkns, temp, top_p],
            cache_examples=False,
            examples=[["I'm feeling very stressed."], ["Breathing exercise?"], ["Motivation?"]]
        )
    gr.HTML("<footer>Designed for DevelopersHub AI/ML Internship · Task 5 · 2026</footer>")

if __name__ == "__main__":
    demo.launch(css=CSS)