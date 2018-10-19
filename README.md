# Solve CartPole using Hill Climbing Algorithms
This repo contains the code to solve openai gym env "CartPole-v0" using variations of hill climbing algorithms.

- [x] Vanilla Hill Climbing
- [x] Steepest Ascend
- [x] Simulated Annealing
- [x] Adaptive Noise Scaling

# Getting Started

## Docker
The easiest way to get started is to use docker.

Build Image
```
docker build -t hill_climb:v0.1 .
```

Access terminal to execute scripts
```
docker run -it -p 8080:8080 -v$(pwd):/app hill_climb:v0.1 /bin/bash
```

Run notebook
```
docker run -it -p 8080:8080 -v$(pwd):/app hill_climb:v0.1
```

## Pip

Or you could just pip install the requirements and run it locally in the environment of your preference.

```
pip install -r requirements.txt
```


# Running Scripts

We can train the agent using any of the above listed policy. The rewards list and final policy is pickled and stored under runs/<policy_name>/timestamp directory.

```
cd src
python main.py --help
```

Example: Train an agent using simulated annealing policy for 1000 episodes, 100 timesteps and goal score of 200. The environment is considered solved when the agent is able to get an average reward greater than goal score over 100 consecutive episodes.
```
python main.py steepest 1000 100 200
```


# Multiple Runs

For my post, I used 1000 training runs for each method to perform the comparisons.

E.g To collect data for vanilla hill climbing policy

```
python main.py vanilla 1000 1000 200 --runs 1000
```