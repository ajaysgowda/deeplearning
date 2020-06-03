"""
main script
central script to avoid any path issues
"""

from open_ai_gym.cartpole.naive_dqn.dqn_agent import run_dqn_training


def invalid_input():
    """
    Invalid input message
    """

    print("Invalid input")


CHOICES = {
        "1": {
            "name": "Train Naive DQN Agent",
            "func": run_dqn_training
            },

        }
INVALID = {
        "name": "Invalid",
        "func": invalid_input
        }

if __name__ == "__main__":
    print(" Please choose one:")

    for item in CHOICES:
        print(f"{item}:  {CHOICES[item]['name']}")

    user_choice = CHOICES.get(input(), INVALID)
    user_choice["func"]()
