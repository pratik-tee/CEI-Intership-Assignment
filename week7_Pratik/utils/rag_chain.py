import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class RAGChain:

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    def generate_answer(self, question, retrieved_chunks):

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are an AI assistant.

Answer ONLY from the given context.

If the answer is not present, say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        return response.text