#!/usr/bin/env python
import sys
from newsletter.newsletter_crew import NewsLetterCrew


def run():
    inputs = {
        'topic': 'direct preference optimization'
    }
    result = NewsLetterCrew().crew().kickoff(inputs=inputs)
    print(result)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'topic': 'direct preference optimization'
    }
    try:
        NewsLetterCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
