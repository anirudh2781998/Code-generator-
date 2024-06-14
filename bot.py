from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
# Load the tokenizer and model
model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_code(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return code



def chat_with_bot(prompt):
    return generate_code(prompt)

# Create the Gradio interface
iface = gr.Interface(
    fn=chat_with_bot,
    inputs="text",
    outputs="text",
    title="Code Generation Chatbot",
    description="Ask me to generate code for you!"
)

# Launch the interface
iface.launch()