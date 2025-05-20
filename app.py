import streamlit as st
import torch
from transformers import pipeline

# Initialize the pipeline once (cache it)
@st.cache_resource(show_spinner=False)
def load_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return pipe

def getLLamaresponse(input_text, no_words, blog_style):
    pipe = load_model()

    # Prepare chat messages for pirate style blog generation
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that writes blogs in the style of a {blog_style}.",
        },
        {
            "role": "user",
            "content": f"Write a blog on the topic: '{input_text}'. Keep it under {no_words} words.",
        },
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate response with sampling parameters for diversity
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    generated_text = outputs[0]["generated_text"]

    # Remove prompt from output to keep only the generated part
    generated_blog = generated_text[len(prompt):].strip()
    return generated_blog

# Streamlit UI
st.set_page_config(page_title="Generate Blogs 🤖", page_icon="🤖", layout="centered")
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")
col1, col2 = st.columns(2)
with col1:
    no_words = st.text_input("No. of Words", value="200")
with col2:
    blog_style = st.selectbox("Writing Style for", ["Researchers", "Data Scientist", "Common People"], index=0)

submit = st.button("Generate")

if submit:
    if not input_text.strip():
        st.warning("Please enter a blog topic.")
    else:
        with st.spinner("Generating..."):
            blog = getLLamaresponse(input_text, no_words, blog_style)
            st.subheader("Generated Blog:")
            st.write(blog)
