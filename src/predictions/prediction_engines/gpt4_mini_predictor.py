"""
gpt4_mini_predictor.py

Warning:
- Using the OpenAI API will incur costs. Make sure to set usage limits and monitor usage to avoid unexpected charges.

This module provides a predictor that uses OpenAI's GPT-4 Mini model to generate predictions for NBA games.
It consists of a class to:
- Generate pre-game predictions.
- Generate current predictions.

Classes:
- GPT4MiniPredictor: Predictor that uses GPT-4 Mini model to generate predictions.

Methods:
- make_pre_game_predictions(game_ids): Generates pre-game predictions for the given game IDs.
- prepare_prompt(user_message): Prepares the prompt for the GPT-4 Mini model.
- make_gpt4o_request(messages): Makes a request to the GPT-4 Mini model and returns the response.
- load_pre_game_data(game_ids): Loads pre-game data for the given game IDs.
- make_current_predictions(game_ids): Generates current predictions for the given game IDs.
- load_current_game_data(game_ids): Loads current game data for the given game IDs.

Usage:
- Typically used as part of the prediction generation process in the prediction_manager module.
- Can be instantiated and used to generate predictions for specified game IDs.

Example:
    predictor = GPT4MiniPredictor()
    pre_game_predictions = predictor.make_pre_game_predictions(game_ids)
    current_predictions = predictor.make_current_predictions(game_ids)
"""

import json

from openai import OpenAI
from pydantic import BaseModel

from src.predictions.prediction_utils import (
    calculate_home_win_prob,
    load_current_game_data,
    update_predictions,
)
from src.predictions.prompt_data import load_prompt_data

client = OpenAI()


class TeamOutcome(BaseModel):
    score: int


class GameOutcome(BaseModel):
    home_team: TeamOutcome
    away_team: TeamOutcome


class GPT4MiniPredictor:
    """
    Predictor that uses OpenAI's GPT-4 Mini model to generate predictions.
    """

    def __init__(self, model_paths=None):
        self.model_paths = model_paths or []

    def make_pre_game_predictions(self, game_ids):
        """
        Generate pre-game predictions using GPT-4 Mini.

        Parameters:
        game_ids (list): A list of game IDs.

        Returns:
        dict: A dictionary of predictions, including predicted scores and win probabilities for each game.
        """
        if not game_ids:
            return {}

        predictions = {}
        games = self.load_pre_game_data(game_ids)

        for game_id in game_ids:
            user_message = games[game_id]
            messages = self.prepare_prompt(user_message)
            response_dict = self.make_gpt4o_request(messages)
            home_score = response_dict["home_team"]["score"]
            away_score = response_dict["away_team"]["score"]
            home_win_prob = calculate_home_win_prob(home_score, away_score)

            predictions[game_id] = {
                "pred_home_score": float(home_score),
                "pred_away_score": float(away_score),
                "pred_home_win_pct": float(home_win_prob),
                "pred_players": {"home": {}, "away": {}},
            }

        return predictions

    def prepare_prompt(self, user_message):
        system_message = (
            "You are an expert NBA game outcome predictor. "
            "Please predict the final outcomes for this game including the home and away team scores. "
        )
        user_message = json.dumps(user_message)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return messages

    def make_gpt4o_request(self, messages):
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=GameOutcome,
            logprobs=True,
            top_logprobs=20,
            n=1,
        )

        response_dict = response.choices[0].message.parsed.model_dump()

        return response_dict

    def load_pre_game_data(self, game_ids):
        return load_prompt_data(game_ids)

    def make_current_predictions(self, game_ids):
        if not game_ids:
            return {}

        games = self.load_current_game_data(game_ids)
        current_predictions = update_predictions(games)
        return current_predictions

    def load_current_game_data(self, game_ids):
        return load_current_game_data(game_ids, predictor_name="GPT4Mini")
