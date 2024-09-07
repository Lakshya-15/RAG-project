from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    """
    Test the query and validation for Monopoly rules.
    """
    try:
        assert query_and_validate(
            question="How much total money does a player start with in Monopoly? (Answer with the number only)",
            expected_response="$1500",
        )
    except Exception as e:
        print(f"An error occurred in test_monopoly_rules: {e}")


def test_ticket_to_ride_rules():
    """
    Test the query and validation for Ticket to Ride rules.
    """
    try:
        assert query_and_validate(
            question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
            expected_response="10 points",
        )
    except Exception as e:
        print(f"An error occurred in test_ticket_to_ride_rules: {e}")


def query_and_validate(question: str, expected_response: str):
    """
    Query the RAG system and validate the response against the expected response.

    Args:
        question (str): The question to query the RAG system.
        expected_response (str): The expected response to validate against.

    Returns:
        bool: True if the actual response matches the expected response, False otherwise.
    """
    try:
        response_text = query_rag(question)
        prompt = EVAL_PROMPT.format(
            expected_response=expected_response, actual_response=response_text
        )

        model = Ollama(model="mistral")
        evaluation_results_str = model.invoke(prompt)
        evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

        print(prompt)

        if "true" in evaluation_results_str_cleaned:
            # Print response in Green if it is correct.
            print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return True
        elif "false" in evaluation_results_str_cleaned:
            # Print response in Red if it is incorrect.
            print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
            return False
        else:
            raise ValueError(
                f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
            )
    except Exception as e:
        print(f"An error occurred in query_and_validate: {e}")
        return False
