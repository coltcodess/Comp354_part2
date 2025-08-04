"""
experiments.py

A script for conducting training and testing experiments with the DQN agent
on the LunarLander-v3 environment. Includes support for dynamic hyperparameter
selection and model management.

Author: AF
Created: 06-28-2025

Description:
    - Allows user to run full training + testing experiments.
    - Interactively sets number of neurons and layers.
    - Supports overwriting previous models and cleans up temporary model files on exit.
    - Contains demo mode for quick agent evaluation.

Functions:
    - demo(): Run pre-trained agent in human-rendered environment.
    - train(): Train a new model with specified hyperparameters.
    - test(): Evaluate a trained model.
    - run_new_experiment(): Main experiment workflow with user input.
    - cleanup(): Removes the temporary model file if it exists.
"""



import DQNAgent
from train_agent import TrainAgent
from hparams import HParams
import gymnasium as gym
import os

new_model_path = None

def demo():
    """
    A quick demonstration to be run as a stand alone.
    Set the default neurons and number of layers before executing.
    """
    hp = HParams()
    hp.set_n_layers(1)
    hp.set_n_neurons(128)
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent.DQNAgent(env=env, model_class=r"dqnModel_lunar_lander.pt", hparams=hp)
    agent.test_demonstration()

def train(overwrite_bool, hparams = None):

    """
    Fuction that runs one training experiement with custom number of neurons and layers.
    Args:
        overwrite_bool (boolean): boolean to overwrite previous model
        hparams (HParam): HParam object with default values or set by user
    return: model_path (str)
    """

    global new_model_path
    num_episodes = 50
    trainer = TrainAgent(overwrite=overwrite_bool, hparams=hparams)
    trainer.train_agent()
    return trainer.get_model_path()

def test(new_model_path, hparams = None):

    """
    Function that runs a test specifically using custom hyperparameters set by a user.
    This function runs automatically after the train function.
    Args:
        new_model_path (str): string indicating the new model location
        hparams (HParams): HParams file indicating the hyperparameters of the saved model
    return: void
    """
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent.DQNAgent(env=env, model_class=new_model_path,hparams=hparams)
    agent.test_demonstration()

def run_new_experiment():
    """
    Function that runs a full experiment with training and testing.
    It takes user input for number of neurons and layers.

    """
    global new_model_path
    overwrite_bool = True  # Default to overwrite if none exists
    # needs to be asked first. If num neurons and num layers are changed the pre-existing model will not work.
    # so if a new model is trained with new hparams will overwrite the old one
    if new_model_path is not None:
        print("You already trained a model. To train another you must overwrite it.")
        overwrite = input("Overwrite existing model? (y/n): ").strip().lower()
        overwrite_bool = overwrite == "y"

        if not overwrite_bool:
            print("Returning to main menu...")
            return
        
    hp = HParams()

    # Input handling for the number of neurons
    while True:
        min_val_neurons = 1
        max_val_neurons = 256

        print("You chose to train a new model. Please choose:")
        neurons = input("number of neurons ( max 256, default 128): ").strip()

        if neurons.isdigit() and (min_val_neurons <= int(neurons) <= max_val_neurons):
            hp.set_n_neurons(int(neurons))
            break
        else:
            print("Invalid number of neurons. Please enter between 1 - 256.\n")

    # Input handling for the number of layers
    while True:
        min_val_layers = 1
        max_val_layers = 10

        layers = input("number of hidden layers: an integer smaller than 10 (default 1): ").strip()

        if layers.isdigit() and (min_val_layers <= int(layers) <= max_val_layers):
            hp.set_n_layers(int(layers))
            break
        else:
            print("Invalid number of layers. Please enter between 1 - 10.\n")

    new_model_path = train(overwrite_bool, hparams=hp)
    print("Close the graph windows to proceed to testing phase and see the aircraft land.")
    if new_model_path:
        print("Now testing...")
        test(new_model_path, hparams=hp)

def cleanup():
    """
    Deletes the temporary model file if it was created.
    """
    global new_model_path
    if new_model_path and os.path.exists(new_model_path):
        os.remove(new_model_path)
        print(f"Deleted temporary model file: {new_model_path}")
