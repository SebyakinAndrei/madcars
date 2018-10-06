# Madcars Reinforcement Learning Solution
## Project structure
1) Runners - main source
2) to_send - solution ready to send to the site (https://aicups.ru/)
3) smartguy - compiled c++11 baseline for windows
4) memes - contains original meme about my worthless solution (Russian)

## How it works
**Localrunner -> RLClient (custom Client) [rl_env.py] -> RLEnv (object that incapsulates methods of RL env. that needs RL agent) [my_strategy.py] -> Strategy (func. run_strategy) [my_strategy.py] (inference) [rl_env.py] (training)**

Strategies are communicating with Client with queues, so it's faster and more debug-friendly solution in compared to stdin/stdoud. We also sending and recieving model weights and other information with queues while training.

## How you can run it
### Flags
- -rl_env - flag for freezing one match config and play it every time
  - -n - the number of times that match runs
  - -env_car - Car used in matches
  - -env_map - Map used in matches
  - -dump - Enables dump of every match (for pre-training agents). Dumps are stored in ```Runners/dumps/<env_map>_<env_car>/dump_<randnum>.txt```
- -reset_every - If used, enables self-play mode (can run only if -f and -s is "agent") and reset the bad model weights to weights of the better model after a number of episodes chosen (look at rl_env.py -> run_strategy for details)
- -resume - the number of epoch to resume model weights. Loads weights from ```Runners/weights/<env_map>_<env_car>/duel_dqn_<resume>.h5f```
- -train1 - enables train mode for the first agent
- -train2 - enables train mode for the second agent
- -dueling1 - enables dueling model (addition for dqn that prevents from overfitting) for first double dqn agent
- -dueling2 - enables dueling model (addition for dqn that prevents from overfitting) for second double dqn agent
- --nodraw - runs localrunner without displaying anything

### Examples
```python localrunner.py -f "agent" -s "agent" -rl_env -env_car Buggy -env_map IslandMap -n 5000 -reset_every 10``` - Self-play config, for test only, models are empty (lol)

```python localrunner.py -f "agent" -s "../smartguy/main.exe" -rl_env -env_car SquareWheelsBuggy -env_map IslandMap -n 500000 -train1 --nodraw``` - Train RL agent against smartguy

