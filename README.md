# Scrabble RL environment

A two-player, English-tile Scrabble environment using the same Gymnasium
`reset()` / `step()` loop as the CartPole starter. No GPU or PyTorch is needed.
Includes legal-move generation, a greedy self-play baseline, tests, and measured
runtime/memory results.

## Run in `rl_demo`

From the **repository root** in PowerShell:

```powershell
conda activate rl_demo
python -m pip install -r .\scrabble\requirements.txt
python -m unittest scrabble.test_environment -v
python -m scrabble.benchmark --episodes 3 --seed 42
```

Alternatively, prefix Python commands with `conda run -n rl_demo`, without
activating the environment. The dependencies are local to this subproject;
the root Atari dependency file is unchanged.

Show the ASCII board after each turn:

```powershell
python -m scrabble.benchmark --episodes 1 --seed 42 --render
```

Lowercase board letters represent blanks; `?` is an unassigned rack blank.
Rows and columns are zero-based.

## Minimal interaction loop

```python
from scrabble import ScrabbleEnv, greedy_action

env = ScrabbleEnv(max_turns=200, render_mode="ansi")
observation, info = env.reset(seed=42)
terminated = truncated = False

while not (terminated or truncated):
    player = observation["player"]
    legal = env.legal_moves()  # Tuple of ScoredMove(move, score).
    action = greedy_action(env)
    observation, reward, terminated, truncated, info = env.step(action)
    print(player, info["move"], reward, info["scores"])

print(env.render())
env.close()
```

**One step is one player's turn.** The returned observation belongs to the
*next* player, including after the last turn. The baseline controls both players;
there is no hidden opponent policy inside `step()`.

## Rules and dictionary scope

- Standard 15x15 board, standard letter/word premiums, standard 100-tile English
  distribution, two blanks, seven-tile racks, and a 50-point seven-tile bonus.
- An opening play covers the center. Subsequent plays connect to the board.
  Every main/cross word must be in the configured dictionary.
- Blanks have zero face value forever. Premiums apply only to newly placed tiles,
  including each word formed by that tile.
- Exchange any nonempty rack subset only with at least seven tiles in the bag.
  Draw replacements **before** returning the exchanged tiles, then shuffle.
- A pass, exchange, or zero-point placement counts as a scoreless turn.
- No challenge/adjudication phase, clock, networking, or tournament word list.

The bundled [words.txt](./words.txt) contains **4,166 teaching words**, not an
official or exhaustive Scrabble lexicon. Legality, move counts, and playing
strength depend on this choice. Short games or blocked boards are possible.
Supply an appropriately licensed lexicon for realistic experiments:

```powershell
python -m scrabble.benchmark --word-list "C:\data\words.txt" --episodes 3
```

Or use `ScrabbleEnv(word_list=path)` / `ScrabbleEnv(words={"AT", "CAT", ...})`.
Files accept whitespace-separated words and full-line `#` comments. File words
are uppercased, duplicates removed, and invalid tokens rejected explicitly.
Words must have 2..15 ASCII letters; headers, definitions, punctuation, and
one-letter words are not supported. The `words=` API requires uppercase already.

## State and observations

The **full internal state** is the board, board blank flags, both racks, ordered
tile bag, both scores, player to move, turn count, scoreless-turn count, terminal
flags, and seeded random-generator state. The lexicon/premiums are fixed rules.
Given this full state and an action, the transition is deterministic.

The Gymnasium `spaces.Dict` observation intentionally hides the opponent's rack
contents and the bag contents/order, so an individual player's problem is
**partially observable**.

| Key | Shape / values | Meaning |
|---|---|---|
| `board` | `(15, 15)`, int8, 0..26 | 0 = empty, 1..26 = A..Z |
| `board_blanks` | `(15, 15)`, binary | A placed tile is a zero-valued blank |
| `rack` | `(27,)`, integer counts 0..7 | Current player's counts in `ABCDEFGHIJKLMNOPQRSTUVWXYZ?` order |
| `scores` | `(2,)`, int32 | Absolute player-0 and player-1 scores |
| `player` | 0 or 1 | Player whose turn it is |
| `opponent_rack_size` | 0..7 | Opponent's tile count only |
| `bag_size` | 0..100 | Number of undrawn tiles |
| `scoreless_turns` | 0..6 | Consecutive zero-score turns |
| `turn` | 0..`max_turns` | Turns already completed |

Arrays returned in observations are copies. `reset(seed=...)` returns
`(observation, info)` and shuffles/deals reproducibly. Unsupported nonempty reset
options are rejected.

## Actions and legal-move generation

The fixed `spaces.Dict` action representation is:

| Key | Values | Meaning |
|---|---|---|
| `kind` | 0, 1, 2 | Place, exchange, pass |
| `row`, `col` | 0..14 | Main word's starting square |
| `direction` | 0, 1 | Horizontal, vertical |
| `word` | `(15,)`, codes 0..26 | Complete main word, A=1..Z=26; trailing zero padding |
| `blanks` | `(15,)`, binary | Word positions using **new** blank tiles |
| `exchange` | `(27,)`, counts 0..7 | Rack tiles to exchange, alphabet-plus-`?` order |

Placement uses coordinates, direction, word, and blanks. Exchange uses only
exchange counts; pass uses only kind. Irrelevant fields are ignored. Direct
`Move` objects are also accepted as a convenience:

```python
from scrabble import Move

# These are action-format examples, not necessarily legal in a random position.
place = Move("place", row=7, col=6, direction=0, word="CAT", blanks=(1,))
exchange = Move("exchange", exchange="AE?")
pass_turn = Move()
gym_action = place.to_action()
```

Main words include letters already on the board. Blank indices refer to positions
in that complete word; they cannot designate existing tiles. `?` is never a
character in a placed word: choose its represented letter and mark its position.
Dictionary/rack/connectivity violations raise `ValueError` **without changing
the board, scores, bag, racks, turn, or RNG state**.

`action_space.sample()` samples the **syntactic** action space, not necessarily
legal Scrabble plays. Select from `env.legal_moves()` instead; generic algorithms
that require every sampled action to be valid need a legal-action adapter.

`legal_moves()` returns all distinct legal physical placements (with immediate
scores), all distinct nonempty exchange multisets when allowed, and one pass.
A trie restricts dictionary prefixes, rack counts restrict tile use, and
perpendicular cross checks prune illegal combinations. Reachable empty anchors
restrict starting positions. Blank assignments are separate actions because
they change scores and future board values. Equivalent single-tile horizontal
and vertical descriptions are deduplicated. There is **no top-k or score cutoff**.
Results are cached until `step()` or `reset()`.

Internal board/rack/bag fields are useful for debugging but must not be mutated
by agents: doing so bypasses tile invariants and cache invalidation. Public
observations and legal moves are sufficient for normal agent interaction.

## Transition, reward, and episode boundaries

For acting player `p`, a valid turn:

1. Validates the entire action without state changes.
2. Places and scores tiles and draws to seven; or exchanges; or passes.
3. Updates the scoreless-turn counter and applies any endgame adjustments.
4. Advances the turn, switches players, and returns
   `(observation, reward, terminated, truncated, info)`.

**Reward** is `score_after[p] - score_before[p]`: immediate placement points,
plus that acting player's endgame adjustment. Pass/exchange normally yield zero.
This is not a win/loss reward or a zero-sum score-difference reward. At game end,
the opponent's penalty changes its score too, but is not retroactively added to
that opponent's earlier rewards; use `info["score_adjustments"]` if implementing
a different objective.

- **Rack out:** when a player uses all its tiles with an empty bag, subtract the
  opponent's rack face value from its score and add that amount to the finisher.
- **Six consecutive scoreless turns:** subtract each player's own rack face value.
- **Turn limit:** `truncated=True` at `max_turns`; do not apply endgame penalties.
  Natural termination takes precedence when both happen on the same turn.

`info` includes scores, current player, dictionary size, and (after a step) the
acting player, decoded move, placement score, score adjustments, and end reason.
At episode end, `winner` is 0, 1, or `None` for a tie; at truncation this only
compares the unfinished scores. Stepping/generating moves before reset or after
the episode ends raises `RuntimeError`.

### Learning-agent integration

Unlike CartPole's small fixed `Discrete` space, Scrabble has a large,
state-dependent action set. The greedy policy chooses the highest-scoring
placement, breaks ties by deterministic generator order, exchanges the whole
rack if no placement exists and exchange is allowed, and otherwise passes.
It does not optimize future rack value or game outcome.

For a DQN-style learner, score the legal candidate moves with a shared Q-network
and maximize over those candidates; do not treat a candidate's list index as a
stable action ID across states. Store the move and the next legal candidates
with replay data. The next observation is an opponent turn, so directly copying
CartPole's single-agent Bellman target is not correct. Use a fixed-opponent
wrapper that advances through the opponent turn, or an explicit turn-aware
two-player learning formulation. No neural-network training is included here.

## Measurements

[benchmark.py](./benchmark.py) writes:

- [results/turns.csv](./results/turns.csv): placement count, exchange count, total
  legal actions including pass, generation/selection/step/turn timings, selected
  move, points, scores, Python memory, RSS, and process peak for every turn.
- [results/summary.json](./results/summary.json): configuration and versions,
  per-episode length/scores/termination/runtime/memory, and aggregate statistics.

The checked-in sample uses greedy vs greedy, seeds **42, 43, 44**, the bundled
4,166-word lexicon, and `max_turns=200` in `rl_demo`. All three games terminated
naturally, rather than hitting the limit.

| Measurement | Sample result |
|---|---:|
| Episode lengths | 38, 33, 29 turns |
| Mean episode length | 33.33 turns |
| Mean legal placements per turn | 151.70 |
| Mean legal-move generation time | 0.209 seconds |
| Mean episode runtime | 7.02 seconds |
| Peak traced Python memory | 8.63 MiB |
| Process lifetime peak working set | 52.15 MiB |

These are illustrative local measurements, **not performance guarantees**.
The benchmark enables `tracemalloc` during timings, which adds overhead. Python
peaks include live setup allocations and retained measurement rows; they are not
isolated per-turn allocation deltas. Windows working set / Linux RSS includes
the interpreter and native allocations, and process peak is a lifetime
high-water mark. Episode runtime includes reset, bookkeeping, and memory
sampling, but excludes setup and final file writes. Turn runtime excludes memory
sampling, CSV row construction, and optional rendering. Large dictionaries and
blanks can increase legal counts, runtime, and memory substantially.

Use `--output "C:\path\results"` to preserve a separate run. Rerunning with the
default output overwrites the two sample measurement files.

## Validation

[test_environment.py](./test_environment.py) covers seeding/observation spaces,
tile conservation through full self-play, rewards and score deltas, standard
premiums and seven-tile bonuses, cross words, blanks, exchanges and their bag
threshold, invalid-action atomicity, action encoding, endgame rules, truncation,
greedy choices, and actual benchmark output shapes.

The move generator is compared against a separate exhaustive enumeration of
dictionary words, board starts, directions, and blank assignments on small
dictionaries, including edge positions and reachable midgame positions. That
oracle shares the placement validator; independent hand-calculated scoring and
invalid-play fixtures test the validator itself.
