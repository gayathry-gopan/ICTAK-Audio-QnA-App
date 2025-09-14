# This Flask application serves as a backend API for the Q&A model.
# It receives questions from a frontend application, processes them
# using the transformers pipeline, and returns the answer.

# Prerequisites: You need to install the following libraries first:
# pip install Flask transformers torch
# You may also need pip install flask_cors if you run into CORS issues.

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
# Enable CORS to allow the frontend HTML file to communicate with this server.
CORS(app)

# Load the question-answering pipeline globally once to avoid
# reloading the model on every request, which is very slow.
try:
    qa_pipeline = pipeline(
        "question-answering", model="distilbert-base-cased-distilled-squad"
    )
    print("Q&A model loaded successfully.")
except Exception as e:
    print(f"Error loading the Q&A model: {e}")
    qa_pipeline = None

# A more structured and comprehensive knowledge base
course_details = {
    "data science": {
        "name": "Certified Specialist in Data Science & Analytics",
        "duration": "6 months, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.",
        "fees": "The fees can cost between Rs. 15,000 to Rs. 25,000."
    },
    "artificial intelligence": {
        "name": "Certified Specialist in Artificial Intelligence & Machine Learning",
        "duration": "6 months, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.",
        "fees": "The fees can cost between Rs. 15,000 to Rs. 25,000."
    },
    "full stack development": {
        "name": "Certified Specialist in Full Stack Development (MERN)",
        "duration": "6 months, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.",
        "fees": "The fees can cost between Rs. 15,000 to Rs. 25,000."
    },
    "cyber security": {
        "name": "Certified Cyber Security Analyst",
        "duration": "6 months, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.",
        "fees": "The fees can cost between Rs. 15,000 to Rs. 25,000."
    },
    "sdet": {
        "name": "Certified Specialist in SDET",
        "duration": "6 months, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.",
        "fees": "The fees can cost between Rs. 15,000 to Rs. 25,000."
    }
}


# Create a comprehensive context string from the structured data
ict_academy_context = """
ICT Academy of Kerala is a Social Enterprise created in a Public Private Partnership (PPP) model.
It is supported by the Government of India and the Government of Kerala.
The Academy provides various courses in emerging technologies like Data Science, Artificial Intelligence,
Machine Learning, Cloud Computing, and more. Their courses are designed for students and working professionals.
The programs offered are Certified Specialist in Data Science & Analytics, Certified Specialist in Artificial Intelligence & Machine Learning, Certified Specialist in Full Stack Development (MERN), Certified Cyber Security Analyst, and Certified Specialist in SDET.
The fees for the courses vary depending on the program. For example, the Certified Specialist programs can cost between Rs. 15,000 to Rs. 25,000. The exact fees are available on their official website.
The duration of the courses also varies. The Certified Specialist programs are typically 6 months long, which includes 3 months of theoretical training and 3 months of hands-on projects and internship.
ICT Academy of Kerala's main office is located in Thiruvananthapuram, Kerala.
"""


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handles the question-answering request.
    It takes a JSON payload with a 'question' field,
    uses the Q&A model to find the answer in the context,
    and returns the answer as a JSON response.
    """
    if qa_pipeline is None:
        return jsonify({"error": "Q&A model is not loaded."}), 503

    data = request.get_json()
    user_question = data.get("question", "")
    user_question_lower = user_question.lower()

    if not user_question:
        return jsonify({"error": "No question provided."}), 400
    
    # Debugging: Print the user's question to the terminal
    print(f"Received question: {user_question}")

    # Check for specific course names in the question
    for course_key, details in course_details.items():
        if course_key in user_question_lower:
            # Check for "other than" or "except for"
            if "other than" in user_question_lower or "except for" in user_question_lower:
                all_courses = [c_details['name'] for c_details in course_details.values()]
                other_courses = [course for course in all_courses if course != details['name']]
                answer = f"The other courses offered are: {', '.join(other_courses)}."
                return jsonify({"answer": answer})
            else:
                answer = f"The {details['name']} is a {details['duration']} program. The fees are {details['fees']}."
                return jsonify({"answer": answer})
    
    # A simple, hard-coded check for the specific programs question.
    if "programs offered" in user_question_lower or "courses offered" in user_question_lower:
        course_list = [details['name'] for details in course_details.values()]
        answer = "The programs offered are: " + ", ".join(course_list) + "."
        return jsonify({"answer": answer})
    
    # Add a hard-coded check for duration questions.
    if "duration" in user_question_lower or "how long" in user_question_lower:
        answer = "The Certified Specialist programs are 6 months long."
        return jsonify({"answer": answer})
    
    # Add a hard-coded check for fees and cost questions.
    if "fees" in user_question_lower or "cost" in user_question_lower or "how much" in user_question_lower:
        answer = "The fees for Certified Specialist programs are between Rs. 15,000 to Rs. 25,000."
        return jsonify({"answer": answer})

    try:
        # Use the loaded pipeline to get the answer from the context.
        result = qa_pipeline(question=user_question, context=ict_academy_context)
        answer = result["answer"]
        print(f"Pipeline result: {result}") # Debugging: Print the full pipeline result
        return jsonify({"answer": answer})
    except Exception as e:
        # This will catch and print the specific error to your terminal
        print(f"Error processing question: {e}")
        return jsonify({"error": "I'm sorry, an error occurred while processing your request. Please try again later."}), 500


if __name__ == "__main__":
    # To run the app, use: python ict_academy_app.py
    # Or in production, use a WSGI server like Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)
