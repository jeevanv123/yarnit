from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI()


@app.route('/marketing', methods=['POST'])
def generate_post():
    data = request.get_json()
    format_text = data.get('format')
    topic = data.get('topic')

    if format_text is None or topic is None:
        return jsonify({'error': 'Format and topic must be provided.'}), 400

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": """"You are a creative marketing manager for your company. 
                Your goal is to generate a captivating posts that will intrigue and engage our audience. 
                Think outside the box and showcase the innovative potential in a way that sparks curiosity and excitement. 
                Remember to highlight unique features, benefits, and the impact. Be bold, creative, and imaginative in your approach. 
                Let's inspire our audience with a post they won't forget!
                I will provide you the format of the post and the topic."""
            },
            {
                "role": "user",
                "content": f"Format: {format_text}. Topic: {topic}"
            }
        ],
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    generated_post = response.choices[0].message.content
    return jsonify({'text': generated_post})


if __name__ == '__main__':
    app.run(debug=True,port=7000)

