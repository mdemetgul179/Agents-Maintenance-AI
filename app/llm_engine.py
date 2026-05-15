import ollama

def analyze_ticket(user_issue, retrieved_docs):

    context = "\n\n".join(retrieved_docs)

    prompt = f"""
    You are an industrial maintenance AI assistant.

    User Issue:
    {user_issue}

    Similar Maintenance Tickets:
    {context}

    Analyze the issue and provide:

    1. Possible root cause
    2. Recommended actions
    3. Priority level
    4. Safety recommendation

    Keep the response concise and professional.
    """

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]