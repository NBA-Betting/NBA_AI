# NBA_AI
Using Generative AI tools to explain and predict the outcomes of NBA games.



## Explain
Take in play by play data of arbitrary length and generate a summary of the game. This could be a summary of the game, a summary of a player's performance, or a summary of a team's performance.

```
def explain(pbp_logs, prompts):
    """
    Generate a summary of the game using the play by play logs.

    If no prompts are given, generate a general summary of the game action that the logs include.
    If prompts are given, generate a summary of the game action that the logs include that is relevant to the prompts.
    Examples of prompts include:
    - "What was the most exciting part of the game?"
    - "What was the most exciting part of the game for the Lakers?"
    - "What was the most exciting part of the game for LeBron James?"
    - "How many points did LeBron James score?"
    - "Who was the best player in the game?" 
    """

    pass
```


## Predict
Take in play by play data of arbitrary length and generate a prediction of the outcome of the game. This could be a prediction of the final score, a prediction of the winner, or a prediction of the final score of a player or team.

```
def predict(pbp_logs, game_id, prompts):
    """
    Generate a prediction of the game outcome using the play by play logs.

    If no game_id is given, generate a predictions for the game in the logs.
    If a game_id is given, generate a prediction for the game with that id.

    If no prompts are given, generate a general prediction of the game outcome that the logs include.
    If prompts are given, generate a prediction of the game outcome that the logs include that is relevant to the prompts.
    Examples of prompts include:
    - "What will be the final score of the game?"
    - "Who will win the game?"
    - "What will be the final score of the game for the Lakers?"
    - "How many points will Lebron score in the game?"
    - "Who will be the best player in the game?" 
    - "What will happen on the next play?"
    """

    pass
```

## Generate
Take in play by play data or a game id and generate play by play logs.

```
def generate(pbp_logs, game_id, log_count):
    """
    Generate play by play logs based of the play by play logs given.
    
    If no game_id is given, generate play by play logs for the game in the logs.
    If a game_id is given, generate play by play logs for the game with that id.

    If no log_count is given, generate a single play by play log.
    If a log_count is given, generate that many play by play logs.
    """

    pass
```

