import random

from rl.agents import MinimaxAgent, QAgent
from rl.evaluate import match, optimal_move_accuracy
from rl.game import EMPTY_BOARD, all_reachable_states, play, winner
from rl.minimax import optimal_moves, value
from rl.symmetry import (INV, PERMS, action_from_canonical,
                         action_to_canonical, canonical, transform)
from rl.train import TrainConfig, train


def test_state_counts():
    states = all_reachable_states()
    assert len(states) == 5478                                   # known result
    assert sum(winner(b) is None for b in states) == 4520


def test_winner():
    assert winner((1, 1, 1, -1, -1, 0, 0, 0, 0)) == 1
    assert winner((1, -1, 1, 1, -1, -1, -1, 1, 1)) == 0
    assert winner(EMPTY_BOARD) is None


def test_symmetry_group():
    assert len(set(PERMS)) == 8
    b = (1, 0, -1, 0, 1, 0, 0, 0, -1)
    for t in range(8):
        for a in range(9):
            # the piece at original cell a shows up at INV[t][a] after transforming
            assert transform(b, t)[INV[t][a]] == b[a]
            assert action_from_canonical(action_to_canonical(a, t), t) == a
    # all 8 images share one canonical form
    assert len({canonical(transform(b, t))[0] for t in range(8)}) == 1


def test_minimax_empty_board_is_draw():
    assert value(EMPTY_BOARD) == 0
    assert set(optimal_moves(EMPTY_BOARD)) == set(range(9))  # every opening draws


def test_update_learns_immediate_win():
    agent = QAgent(alpha=1.0)
    b = (1, 1, 0, -1, -1, 0, 0, 0, 0)  # X to move, cell 2 wins
    agent.update(b, 1, 2)
    assert agent.q_values(b, 1)[2] == 1.0
    assert agent.greedy_moves(b, 1) == [2]


def test_symmetric_positions_share_values():
    agent = QAgent(alpha=1.0)
    b = (1, 1, 0, -1, -1, 0, 0, 0, 0)
    agent.update(b, 1, 2)
    rotated = transform(b, 2)
    win_cell = INV[2][2]
    assert agent.q_values(rotated, 1)[win_cell] == 1.0


def test_training_reaches_perfect_play():
    agent, hist = train(TrainConfig(episodes=15_000, eval_every=15_000, eval_games=50), verbose=False)
    assert optimal_move_accuracy(agent) == 1.0
    mm = MinimaxAgent(random.Random(0))
    assert match(agent, mm, 100, 1)["loss"] == 0
    assert match(agent, mm, 100, -1)["loss"] == 0
    assert len(agent.Q) == 627                                   # canonical decision states


def test_save_load(tmp_path):
    agent = QAgent()
    agent.update(play(EMPTY_BOARD, 4, 1), -1, 0)
    agent.save(tmp_path / "q.pkl")
    again = QAgent.load(tmp_path / "q.pkl")
    assert again.Q == agent.Q


def test_export_web(tmp_path):
    import json
    from rl.export import export_web
    agent, _ = train(TrainConfig(episodes=3_000, eval_every=3_000, eval_games=10), verbose=False)
    agent.save(tmp_path / "q.pkl")
    out = tmp_path / "data" / "qtable.js"
    meta = export_web(tmp_path / "q.pkl", out)
    text = out.read_text()
    assert text.count("window.TTT_QTABLE = ") == 1
    payload = text.split("window.TTT_QTABLE = ", 1)[1].rstrip().rstrip(";")
    data = json.loads(payload)
    assert data["meta"] == meta and len(data["Q"]) == len(agent.Q)


def test_export_escapes_script_breaking_characters():
    from rl.export import to_safe_js_json
    evil = {"x": "</script><script>alert(1)</script>&\u2028"}
    out = to_safe_js_json(evil)
    assert "<" not in out and ">" not in out and "&" not in out
    import json
    assert json.loads(out) == evil              # still the same data after escaping


def test_export_rejects_corrupted_table():
    import pytest
    from rl.export import build_payload
    agent = QAgent()
    agent.Q[(2, 0, 0, 0, 0, 0, 0, 0, 0)] = [0.0] * 9          # 2 is not a valid cell
    with pytest.raises(ValueError):
        build_payload(agent)
    agent = QAgent()
    agent.Q[(0,) * 9] = [float("nan")] + [0.0] * 8           # NaN is not a valid Q-value
    with pytest.raises(ValueError):
        build_payload(agent)