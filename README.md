# TetrisRL
Tetris Deep Reinforcement Learning 

My implementation of the DQN paper as it relates to Tetris. 

When I embarked on the "TetrisRL" project, I aimed to combine my fascination with Tetris and the cutting-edge techniques of Deep Reinforcement Learning (DRL). This project utilizes Deep Q-Learning (DQN) to train an AI to play Tetris. The repository includes essential scripts, such as DQ_Agent.py for defining the DQN agent and tetris_gym_env.py for creating the gym environment where the agent learns and plays Tetris.

The journey began with setting up the environment and understanding the complex interactions within the code. Writing this in my second term of my first year, it was a leap that proved to be incredibly challenging as many of the prerequisites required for this project (machine learning methods, a sufficient understanding of linear algera and math in multi-dimentionality, and interactions with pytorch in general) were not quite set in place. In fact, the mere reading of the original DQN paper proved to be a hard challenge in and of itself. 

That all being said, the mere concept of reinforcement learning became progressively more and more interesting to me as the project went on. The concepts of learning from experiences, learning about setting up environments in a markovian way, and more or less everything about this project became more interesting than any of the courses I was in. 

The code for the environment itself was more or less inspired by Tech With Tim's implementation of Tetris, with my changes to his code being more or less in the form of transforming his code to one that would work under OpenAI's Gym Environments paradigms. 

After that, much of the struggles in this project came from messing with reward functions and hyperparameters, as well as updates to the environment to let the agent have an easier time generalizing. 

