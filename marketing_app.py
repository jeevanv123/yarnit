import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

st.title("Creative Marketing Post Generator")

# Input fields for format and topic
format_input = st.text_input("Enter the format (e.g., Email, Social Media Post):")
topic_input = st.text_input("Enter the topic:")

if st.button("Generate Post"):
    if format_input.strip() == "":
        st.warning("Please enter the format.")
    elif topic_input.strip() == "":
        st.warning("Please enter the topic.")
    else:
        # Call OpenAI's completion endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": f""""You are a creative marketing manager for your company. 
                    Your goal is to generate a captivating post that will intrigue and engage our audience. 
                    Think outside the box and showcase the innovative potential in a way that sparks curiosity and excitement. 
                    Remember to highlight unique features, benefits, and the impact. Be bold, creative, and imaginative in your approach. 
                    Let's inspire our audience with a post they won't forget!
                    Format: {format_input}. Topic: {topic_input}"""
                }
            ],
            temperature=1,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Output the generated post
        if response.choices:
            generated_post = response.choices[0].message.content
            st.text_area("Generated Post:", value=generated_post, height=1000)
        else:
            st.error("Failed to generate post. Please try again.")
