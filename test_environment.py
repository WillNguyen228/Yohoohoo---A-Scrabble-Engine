"""Run from the repository root: python -m unittest scrabble.test_environment."""

import csv
import json
import tempfile
import unittest
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from .benchmark import run_benchmark
from .environment import (
    ALPHABET, DIRECTIONS, DISTRIBUTION, Move, ScrabbleEnv, greedy_action, load_words,
)


class ScrabbleTests(unittest.TestCase):
    def make_env(self, words=("AT", "CAT", "CATS", "BAT", "BAR", "CAR", "ART", "RAT")):
        env = ScrabbleEnv(words=words)
        env.reset(seed=1)
        self.addCleanup(env.close)
        return env

    def set_position(self, env, rack, entries=(), bag=None):
        env.board.fill(0)
        env.board_blanks.fill(0)
        env.racks[0] = list(rack)
        for row, col, letter in entries:
            env.board[row, col] = ALPHABET.index(letter.upper()) + 1
            env.board_blanks[row, col] = int(letter.islower())
        env.bag = list("EEEEEEEEEE" if bag is None else bag)
        env._moves = None

    def physical_key(self, env, move):
        dr, dc = DIRECTIONS[move.direction]
        return tuple(sorted(
            (move.row + i * dr, move.col + i * dc, letter, i in move.blanks)
            for i, letter in enumerate(move.word)
            if not env.board[move.row + i * dr, move.col + i * dc]
        ))

    def brute_force(self, env):
        """Independent enumeration; shared validator is tested by rule fixtures."""
        found = {}
        for word in env.words:
            for direction, (dr, dc) in enumerate(DIRECTIONS):
                for row in range(15 - dr * (len(word) - 1)):
                    for col in range(15 - dc * (len(word) - 1)):
                        new_indices = [
                            i for i in range(len(word))
                            if not env.board[row + i * dr, col + i * dc]
                        ]
                        if not new_indices or len(new_indices) > 7:
                            continue
                        for flags in product((False, True), repeat=len(new_indices)):
                            blanks = tuple(i for i, flag in zip(new_indices, flags) if flag)
                            if len(blanks) > env.racks[env.current_player].count("?"):
                                continue
                            move = Move("place", row, col, direction, word, blanks)
                            try:
                                score, _ = env._evaluate(move)
                            except ValueError:
                                continue
                            found[self.physical_key(env, move)] = score
        return found

    def test_seed_observation_and_inventory(self):
        env = self.make_env()
        obs, info = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(len(env.bag), 86)
        self.assertEqual(Counter(env.bag + env.racks[0] + env.racks[1]), DISTRIBUTION)
        other = self.make_env()
        obs2, _ = other.reset(seed=42)
        for key in obs:
            np.testing.assert_equal(obs[key], obs2[key])
        self.assertEqual(info["current_player"], 0)
        obs["board"][7, 7] = 26
        obs["scores"][0] = 999
        self.assertEqual(env.board[7, 7], 0)
        self.assertEqual(env.scores[0], 0)

    def test_opening_center_and_premium(self):
        env = self.make_env()
        self.set_position(env, "CATDEFG")
        _, reward, terminated, truncated, info = env.step(Move("place", 7, 6, 0, "CAT"))
        self.assertEqual(reward, 10)
        self.assertEqual(info["move_score"], 10)
        self.assertEqual(env.current_player, 1)
        self.assertFalse(terminated or truncated)
        self.assertEqual(len(env.racks[0]), 7)

    def test_invalid_moves_are_atomic(self):
        env = self.make_env()
        self.set_position(env, "CATDEFG")
        board = env.board.copy()
        racks = [rack.copy() for rack in env.racks]
        bag = env.bag.copy()
        rng = repr(env.np_random.bit_generator.state)
        bad_moves = [
            Move("place", 0, 0, 0, "CAT"),
            Move("place", 7, 7, 0, "BAT"),
            Move("place", 7, 14, 0, "CAT"),
            Move("place", 7, 7, 0, "ZZZ"),
            Move("place", 7, 7, 0, "CAT", (0,)),
            Move("exchange", exchange="ZZ"),
            Move("exchange"), Move("unknown"),
        ]
        for move in bad_moves:
            with self.subTest(move=move):
                with self.assertRaises(ValueError):
                    env.step(move)
                np.testing.assert_equal(env.board, board)
                self.assertEqual(env.racks, racks)
                self.assertEqual(env.bag, bag)
                self.assertEqual(env.turn, 0)
                self.assertEqual(repr(env.np_random.bit_generator.state), rng)

    def test_cross_words_and_single_tile_dedup(self):
        env = self.make_env(words=("AT", "CAT"))
        self.set_position(env, "TCDEFGH", ((6, 7, "A"), (7, 6, "A")))
        # T forms AT horizontally and vertically: center doubles both words.
        score, placed = env._evaluate(Move("place", 7, 6, 0, "AT"))
        self.assertEqual(score, 8)
        self.assertEqual(len(placed), 1)
        target = ((7, 7, "T", False),)
        generated = [
            item for item in env.legal_moves()
            if item.move.kind == "place" and self.physical_key(env, item.move) == target
        ]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].score, 8)

    def test_bad_cross_and_disconnected_and_partial_words(self):
        env = self.make_env(words=("AT", "CAT", "CATS", "BAT"))
        self.set_position(env, "BATSEFG", ((7, 6, "C"), (7, 7, "A"), (7, 8, "T")))
        for move in (
            Move("place", 6, 6, 0, "BAT"),  # Invalid BC, AA, TT crosses.
            Move("place", 1, 1, 0, "BAT"),
            Move("place", 7, 7, 0, "AT"),  # Omits leading C.
            Move("place", 7, 6, 0, "CAT"),  # No new tiles.
            Move("place", 7, 6, 0, "BAT"),  # Overwrites C.
        ):
            with self.subTest(move=move), self.assertRaises(ValueError):
                env.step(move)
        # Existing center premium is not applied again; S is on a normal square.
        self.assertEqual(env._evaluate(Move("place", 7, 6, 0, "CATS"))[0], 6)

    def test_blanks_stay_zero_and_have_distinct_actions(self):
        env = self.make_env(words=("AT", "CAT"))
        self.set_position(env, "C?TDEFG")
        _, reward, _, _, _ = env.step(Move("place", 7, 6, 0, "CAT", (1,)))
        self.assertEqual(reward, 8)
        self.assertEqual(env.board_blanks[7, 7], 1)
        env.racks[1] = list("TCDEFGH")
        # A is an existing blank: AT scores only T on a normal square at (8, 7).
        self.assertEqual(env._evaluate(Move("place", 7, 7, 1, "AT"))[0], 1)
        self.set_position(env, "A?TDEFG")
        env.current_player = 0
        moves = [item for item in env.legal_moves() if item.move.kind == "place"
                 and item.move.word == "AT" and item.move.row == 7 and item.move.col == 7
                 and item.move.direction == 0]
        self.assertEqual({item.move.blanks for item in moves}, {(), (0,), (1,)})

    def test_bingo_and_word_multipliers(self):
        env = self.make_env(words=("READING",))
        self.set_position(env, "READING")
        # D on center, other tiles on normal squares: 9 * 2 + 50.
        self.assertEqual(env._evaluate(Move("place", 7, 4, 0, "READING"))[0], 68)
        self.set_position(env, "READING", ((0, 7, "R"),))
        # Double word (1,1) and double letter (3,3) are independent squares.
        self.assertEqual(env.word_multiplier[0, 0], 3)
        self.assertEqual(env.word_multiplier[1, 1], 2)
        self.assertEqual(env.letter_multiplier[5, 5], 3)

    def test_multiple_word_multipliers_and_cross_letter_premium(self):
        env = self.make_env(words=("CATSATAT", "CAT", "AT"))
        self.set_position(env, "CTSATAT", ((0, 1, "A"),))
        # Two triple words, double-letter S, seven new tiles: 11 * 9 + 50.
        self.assertEqual(env._evaluate(Move("place", 0, 0, 0, "CATSATAT"))[0], 149)
        self.set_position(env, "CATDEFG", ((4, 5, "A"),))
        # T on triple letter scores in both CAT (7) and AT (4).
        self.assertEqual(env._evaluate(Move("place", 5, 3, 0, "CAT"))[0], 11)
        self.set_position(env, "CA?DEFG", ((4, 5, "A"),))
        self.assertEqual(env._evaluate(Move("place", 5, 3, 0, "CAT", (2,)))[0], 5)
        self.assertEqual(np.count_nonzero(env.word_multiplier == 3), 8)
        self.assertEqual(np.count_nonzero(env.word_multiplier == 2), 17)
        self.assertEqual(np.count_nonzero(env.letter_multiplier == 3), 12)
        self.assertEqual(np.count_nonzero(env.letter_multiplier == 2), 24)
        self.assertFalse(np.any((env.word_multiplier > 1) & (env.letter_multiplier > 1)))

    def test_exchange_subsets_and_bag_threshold(self):
        env = self.make_env()
        self.set_position(env, "AABCDEF", bag="GGGGGGG")
        exchanges = [item.move.exchange for item in env.legal_moves()
                     if item.move.kind == "exchange"]
        self.assertEqual(len(exchanges), 3 * 2 ** 5 - 1)
        before = Counter(env.bag + env.racks[0])
        _, reward, _, _, _ = env.step(Move("exchange", exchange="AA"))
        self.assertEqual(reward, 0)
        self.assertEqual(env.racks[0].count("A"), 0)
        self.assertEqual(env.racks[0].count("G"), 2)
        self.assertEqual(Counter(env.bag + env.racks[0]), before)
        env.bag = list("EEEEEE")
        env._moves = None
        self.assertFalse(any(item.move.kind == "exchange" for item in env.legal_moves()))
        with self.assertRaises(ValueError):
            env.step(Move("exchange", exchange=env.racks[1][0]))

    def test_rack_out_adjustments_and_reward(self):
        env = self.make_env(words=("AT",))
        self.set_position(env, "AT", bag="")
        env.racks[1] = list("Z?")
        observation, reward, terminated, truncated, info = env.step(
            Move("place", 7, 7, 0, "AT")
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 14)
        self.assertEqual(env.scores.tolist(), [14, -10])
        self.assertEqual(info["score_adjustments"].tolist(), [10, -10])
        self.assertEqual(info["end_reason"], "rack_out")
        self.assertTrue(env.observation_space.contains(observation))
        with self.assertRaises(RuntimeError):
            env.step(Move())
        with self.assertRaises(RuntimeError):
            env.legal_moves()

    def test_six_scoreless_turns_and_turn_limit(self):
        env = self.make_env()
        env.racks = [list("A?"), list("Z")]
        for _ in range(5):
            _, _, terminated, _, _ = env.step(Move())
            self.assertFalse(terminated)
        _, reward, terminated, truncated, info = env.step(Move())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -10)
        self.assertEqual(env.scores.tolist(), [-1, -10])
        self.assertEqual(info["end_reason"], "six_scoreless_turns")
        limited = ScrabbleEnv(words=("AT",), max_turns=1)
        self.addCleanup(limited.close)
        limited.reset(seed=1)
        _, reward, terminated, truncated, info = limited.step(Move())
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 0)
        self.assertEqual(info["end_reason"], "turn_limit")

    def test_zero_point_placement_counts_as_scoreless(self):
        env = self.make_env(words=("AT",))
        self.set_position(env, "??", bag="EEEEEEEE")
        env.step(Move("place", 7, 7, 0, "AT", (0, 1)))
        self.assertEqual(env.scoreless_turns, 1)

    def test_gym_actions_round_trip_and_padding(self):
        env = self.make_env(words=("AT",))
        self.set_position(env, "A?TDEFG")
        for item in env.legal_moves():
            action = item.move.to_action()
            self.assertTrue(env.action_space.contains(action))
            self.assertEqual(env._decode(action), item.move)
        action = Move("place", 7, 7, 0, "AT").to_action()
        action["word"][3] = 1
        with self.assertRaises(ValueError):
            env.step(action)

    def test_generator_matches_exhaustive_oracle(self):
        for entries, rack in (
            ((), "C?T"),
            (((7, 6, "C"), (7, 7, "A"), (7, 8, "T")), "BS?"),
            (((6, 7, "A"), (7, 6, "A")), "T?"),
            (((0, 0, "A"), (0, 1, "T")), "C?S"),
            (((14, 13, "A"), (14, 14, "T")), "C?S"),
        ):
            with self.subTest(entries=entries, rack=rack):
                env = self.make_env(words=("AT", "CAT", "CATS", "BAT", "BATS", "AS", "SAT"))
                self.set_position(env, rack, entries)
                actual = {
                    self.physical_key(env, item.move): item.score
                    for item in env.legal_moves() if item.move.kind == "place"
                }
                self.assertEqual(actual, self.brute_force(env))

    def test_generator_oracle_on_reachable_midgame_positions(self):
        env = self.make_env(words=("AT", "CAT", "CATS", "BAT", "BATS", "AS", "SAT"))
        self.set_position(env, "CATSB??", bag="AAAAAAAATTTTTTTTCCCCSSSSBBBB")
        env.racks[1] = list("CATSB??")
        random = np.random.default_rng(42)
        for _ in range(8):
            placements = [item for item in env.legal_moves() if item.move.kind == "place"]
            actual = {self.physical_key(env, item.move): item.score for item in placements}
            self.assertEqual(actual, self.brute_force(env))
            action = placements[int(random.integers(len(placements)))].move if placements else Move()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_greedy_chooses_highest_score_then_exchange_then_pass(self):
        env = self.make_env(words=("AT", "CAT"))
        self.set_position(env, "CATDEFG")
        move = env._decode(greedy_action(env))
        self.assertEqual(
            env._evaluate(move)[0],
            max(item.score for item in env.legal_moves() if item.move.kind == "place"),
        )
        self.set_position(env, "ZZZZZZZ")
        self.assertEqual(env._decode(greedy_action(env)).exchange, "ZZZZZZZ")
        self.set_position(env, "ZZZZZZZ", bag="")
        self.assertEqual(env._decode(greedy_action(env)).kind, "pass")

    def test_seeded_self_play_conserves_tiles_and_scores(self):
        env = self.make_env()
        other = self.make_env()
        env.reset(seed=17)
        other.reset(seed=17)
        for _ in range(env.max_turns):
            action = greedy_action(env)
            other_action = greedy_action(other)
            for key in action:
                np.testing.assert_equal(action[key], other_action[key])
            player = env.current_player
            old_scores = env.scores.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            other.step(other_action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(reward, int(env.scores[player] - old_scores[player]))
            expected_delta = info["score_adjustments"].copy()
            expected_delta[player] += info["move_score"]
            np.testing.assert_equal(env.scores - old_scores, expected_delta)
            board_tiles = [
                "?" if env.board_blanks[r, c] else ALPHABET[int(env.board[r, c]) - 1]
                for r, c in zip(*np.nonzero(env.board))
            ]
            self.assertEqual(
                Counter(env.bag + env.racks[0] + env.racks[1] + board_tiles),
                DISTRIBUTION,
            )
            if terminated or truncated:
                break
        self.assertTrue(terminated or truncated)

    def test_dictionary_validation_and_lifecycle(self):
        env = ScrabbleEnv(words=("AT",))
        self.addCleanup(env.close)
        with self.assertRaises(RuntimeError):
            env.step(Move())
        for words in ((), ("a",), ("at",), ("C4T",)):
            with self.assertRaises(ValueError):
                ScrabbleEnv(words=words)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "words.txt"
            source.write_text("# teaching\nat CAT\nAT\n", encoding="utf-8")
            self.assertEqual(load_words(source), frozenset(("AT", "CAT")))
            source.write_text("A1", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid word"):
                load_words(source)
        self.assertGreater(len(load_words()), 1000)

    def test_benchmark_output_and_measured_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_benchmark(episodes=1, seed=2, max_turns=2, output=directory)
            self.assertEqual(report["episodes"][0]["turns"], 2)
            self.assertTrue(report["episodes"][0]["truncated"])
            self.assertGreater(report["summary"]["python_peak_bytes"], 0)
            self.assertGreater(report["summary"]["process_lifetime_peak_bytes"], 0)
            saved = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(saved, report)
            with (Path(directory) / "turns.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(
                    int(row["legal_actions_including_pass"]),
                    int(row["legal_placements"]) + int(row["legal_exchanges"]) + 1,
                )
                self.assertGreater(float(row["generation_seconds"]), 0)
                self.assertGreater(int(row["process_rss_bytes"]), 0)


if __name__ == "__main__":
    unittest.main()
