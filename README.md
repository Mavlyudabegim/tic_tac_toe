# Tic-Tac-Toe with Q-Learning

An agent that learns to play Tic-Tac-Toe **from scratch**, using tabular **Q-learning** and **self-play**. It is never told the rules of good play. It only gets a reward at the end of each game and learns from that.

After about **5,000 training games (a few seconds)** the agent plays **perfectly**:

| Test                                                 | Result                   |
| ---------------------------------------------------- | ------------------------ |
| Optimal move in all 4,520 possible positions         | **100%**                 |
| vs perfect (minimax) player, 1,000 games as X / as O | **0 losses** (all draws) |
| vs random player, 1,000 games as X                   | 96.9% wins, 0 losses     |
| vs random player, 1,000 games as O                   | 74.5% wins, 0 losses     |

![Web UI]
<img width="1104" height="768" alt="Screenshot 2026-10-18 at 19 43 34" src="https://github.com/user-attachments/assets/227386f8-392d-4044-9a65-468d7e8b8ed5" />

## 1. The task

Tic-Tac-Toe is a two-player game on a 3×3 board. X moves first, and the players take turns. Three in a row (horizontal, vertical or diagonal) wins. A full board with no line is a draw.

The game is **solved**: if both players play perfectly, every game ends in a **draw**. So a perfect agent must:

- **never lose**, whether it plays X or O, against anyone
- **win whenever the opponent makes a mistake**

**Goal of the project:** let an agent discover this perfect strategy by itself, through reinforcement learning, and **measure honestly** whether it succeeded.

In reinforcement-learning terms:

| RL concept | In Tic-Tac-Toe                                                   |
| ---------- | ---------------------------------------------------------------- |
| State _s_  | the board: 9 cells, each `+1` (X), `−1` (O) or `0` (empty)       |
| Action _a_ | the empty cell to play in (0–8)                                  |
| Reward _r_ | `+1` for a winning move, `0` for a draw, nothing during the game |
| Episode    | one full game                                                    |

```
cell numbers:   0 | 1 | 2
                3 | 4 | 5
                6 | 7 | 8
```

---

## 2. Algorithm

### 2.1 Q-learning

The agent keeps a **Q-table**: for every board _s_ and move _a_, a number **Q(s, a)** that estimates how good that move is (+1 = sure win, 0 = draw, −1 = sure loss).

After each move, it updates the table:

```
Q(s, a) ← Q(s, a) + α · ( target − Q(s, a) )
```

The table starts with all zeros. Thousands of these small updates turn it into an accurate map of good and bad moves.

### 2.2 The target: two-player version ("negamax")

Tic-Tac-Toe has **two players**. After I move, it is my **opponent's** turn, and **whatever is best for them is worst for me**. So the target is:

```
target =  +1                          if my move wins the game
           0                          if my move ends in a draw
          −γ · max Q(s′, a′)          otherwise
                 a′
```

where `s′` is the board **from the opponent's side** after my move. The minus sign turns "opponent's best value" into "my value".

- This is the standard Q-learning target (`r + γ · max Q`), written for a zero-sum game. It is known as the **negamax** form.
- Every move gets a correct update straight away. There is no need to wait for the opponent's reply or to hand out rewards by hand.

### 2.3 One table for both players

Before the board is looked up, it is multiplied by the player to move, so **"my stones" are always +1** and the opponent's are −1. The same Q-table can then play both X and O, and the agent learns from both sides of every game.

### 2.4 Self-play training

The agent plays **against itself**. Each training game:

1. Starting from the empty board, the agent picks moves with the **ε-greedy** rule: with probability **ε** a random move (to explore), otherwise the move with the highest Q-value.
2. When the game ends, the moves are updated **from last to first** (backward updates). The final result then travels back through the whole game in one pass.

The target uses `max`, so this is **off-policy** learning: random exploration moves do not corrupt the values.

### 2.5 Parameters

| Symbol | Name             | Value                                     | Meaning                                                                                 |
| ------ | ---------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| **α**  | learning rate    | 0.5 → 0.05 (linear decay)                 | how big each update step is                                                             |
| **γ**  | discount factor  | 0.9                                       | future rewards count a bit less, so the agent prefers **fast wins** and **slow losses** |
| **ε**  | exploration rate | 1.0 → 0.05 over the first 70% of training | how often a random move is played during training                                       |
|        | training games   | 50,000 (default)                          | ~10 seconds on a laptop                                                                 |

γ = 0.9 is why the web page shows **−0.90** on some cells. It means "if you play here, the opponent wins on the very next move": −1, discounted by one step.

---

## 3. Symmetry

### 3.1 The idea

These four boards are **the same position**, just rotated:

```
 X | . | .      . | . | X      . | . | .      . | . | .
 . | . | .      . | . | .      . | . | .      . | . | .
 . | . | .      . | . | .      . | . | X      X | . | .
```

Whatever is the best reply in one of them, the **rotated** reply is best in the others.

A square board has **8 symmetries**: 4 rotations (0°, 90°, 180°, 270°), each with or without a **mirror** (left↔right flip).

- **Without symmetry**, plain Q-learning treats these as different boards and must learn each one separately.
- **With symmetry**, before every Q-table lookup:
  1. all 8 versions of the board are generated
  2. one fixed "standard" version is chosen (the one that sorts first)
  3. the move is rotated or mirrored **the same way**
  4. the **normal Q-learning update** is applied to that one shared entry

All versions of a position therefore share **one row** of the Q-table. Something learned in one orientation is immediately known in all 8. The **Q-learning formula itself does not change**; only the choice of table row does.

|                              | Without symmetry | With symmetry    |
| ---------------------------- | ---------------- | ---------------- |
| Q-table rows                 | 4,520            | **627**          |
| Updates per move             | 1                | 1                |
| Result after enough training | perfect play     | perfect play     |
| Training needed              | ~27,000 games    | **~6,000 games** |

Symmetry is domain knowledge that we give the agent, so it can be switched off (`--no-symmetry`). The comparison below shows that **plain Q-learning also reaches perfect play**; symmetry only makes it about **6× faster**.

---

## 4. Evaluation method

The agent learns only from self-play. To judge it fairly, it is **tested separately** every 2,500 training games, playing **greedily** (no random moves). There are three tests:

1. **Optimal-move accuracy.** For **every one of the 4,520** non-terminal positions, are _all_ the agent's preferred moves truly optimal?
2. **vs perfect player.** 200–1,000 games against a minimax player that never makes a mistake. Any loss shows a real error.
3. **vs random player.** Checks that the agent actually **wins** when the opponent blunders.

The **correct answers** come from a **minimax solver** (`rl/minimax.py`), which computes the true game value of every position. Minimax is used **only for grading**. It is never used in training, and the agent never sees it.

> Why not just report self-play win rates? Because when both sides improve together, the score stays near "all draws" whether the agent is weak or strong. A fixed, perfect examiner is the only reliable measure.

---

## 5. Results and plots

### 5.1 Learning curve (with symmetry, 50,000 games)

<img width="1950" height="546" alt="image" src="https://github.com/user-attachments/assets/ab34fce4-daa6-4b92-a3f4-aaeb6480535f" />

- **Left, optimal-move accuracy:** starts at 29%, because with an empty table all moves look equal and the agent picks at random. It reaches **95% at 2,500 games** and **100% at 5,000 games**, and stays there.
- **Middle, vs random player:** losses fall to **0** almost at once. Wins settle at about **97% as X** and **75–80% as O**.
  - Why lower as O? Moving second is harder, and the agent learned to play perfectly against a _perfect_ opponent. Many moves are equally "safe" (all lead to a draw), and it doesn't specially prefer the ones that trap a weak player.
- **Right, vs perfect player:** losses fall from 77% (as X) and 96% (as O) to **0% by 5,000 games**.

### 5.2 With vs without symmetry (3 seeds each)

<img width="2080" height="572" alt="image" src="https://github.com/user-attachments/assets/e74f6a5d-f8a5-4ad0-8f47-7a54511da077" />

|                   | Games to 99% accuracy | Games to 0 losses vs perfect player | Final accuracy                | Q-table rows |
| ----------------- | --------------------- | ----------------------------------- | ----------------------------- | ------------ |
| **With symmetry** | **4,000**             | **6,000**                           | 100%                          | **627**      |
| Without symmetry  | 24,000                | 27,333                              | 99.6% (→100% with more games) | 4,520        |

Both versions learn the game. Symmetry gets there **about 6× faster** with a **7× smaller table**.

### 5.3 Final evaluation (default model, 50,000 games)

```
$ python -m rl evaluate
Optimal-move accuracy: 100.00%  (all 4,520 positions)
vs_random_as_X       win  969  draw   31  loss    0
vs_random_as_O       win  745  draw  255  loss    0
vs_minimax_as_X      win    0  draw 1000  loss    0
vs_minimax_as_O      win    0  draw 1000  loss    0
```

The agent **never loses**, plays the optimal move in **every** position, and draws every game against perfect play, which is the best possible result in Tic-Tac-Toe.

---

## 6. How to run

```bash
# setup (once)
conda create -n tic_tac_toe python=3.11 -y
conda activate tic_tac_toe
pip install matplotlib pytest

# train: also writes the plot and updates the web page data
python -m rl train
open runs/default/learning_curve.png

# evaluate the trained model
python -m rl evaluate

# play
open web/index.html                 # in the browser
python -m rl play --first           # in the terminal (you are X)

# experiments
python experiments/compare_symmetry.py      # with vs without symmetry (~1 min)
python experiments/compare.py && python experiments/plot_comparison.py   # old vs new

# tests
pytest
```

Useful `train` options: `--episodes`, `--gamma`, `--seed`, `--no-symmetry`, `--online` (update after each move instead of backward), `--no-export`.

### Web UI

`web/index.html` lets you play against the trained agent in the browser:

- choose X or O
- a **Q-value heat map** on each empty cell (★ = the agent's preferred move)
- a bar chart of the AI's last decision
- an **"AI random moves" slider**, which is ε at play time: at 0% the AI is unbeatable, so raise it to give yourself a chance

The page doesn't learn. It reads the trained table from `web/data/qtable.js`, which `train` updates automatically. The JavaScript doesn't use `innerHTML`, and the page sets a strict Content-Security-Policy and checks all of its data before using it.

---

## 7. Project structure

```
rl/
  game.py        rules, board helpers, all 5,478 reachable positions
  symmetry.py    the 8 board symmetries, standard form, move mapping
  agents.py      QAgent (Q-learning), RandomAgent, MinimaxAgent
  minimax.py     exact solver, used only for evaluation
  train.py       self-play loop, ε/α schedules, periodic evaluation
  evaluate.py    optimal-move accuracy, matches vs other players
  plots.py       learning-curve figure
  export.py      writes the Q-table for the web page (validated + escaped)
  cli.py         commands: train / evaluate / play / export
web/
  index.html     page layout
  css/style.css  styles
  js/engine.js   rules, symmetry, Q-table lookup (no learning)
  js/app.js      user interface
  data/qtable.js trained Q-table (generated)
experiments/     symmetry comparison, old-vs-new comparison
tests/           unit and end-to-end tests
results/         figures used in this README
```

## References

- R. S. Sutton & A. G. Barto, _Reinforcement Learning: An Introduction_, 2nd ed., MIT Press, 2018. Ch. 1 (Tic-Tac-Toe example, Ex. 1.2 "Symmetries") and Ch. 6.5 (Q-learning). [Free online](http://incompleteideas.net/book/the-book-2nd.html)
- E. van der Pol, D. Worrall, H. van Hoof, F. Oliehoek, M. Welling, _MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning_, NeurIPS 2020. [arXiv:2006.16908](https://arxiv.org/abs/2006.16908)

## License

MIT
