# Future Strategy: Advanced Adaptive Architecture (League Ready)
**Status:** Planned / Concept
**Goal:** Enable a fully trained agent to dominate in a competitive league format by combining **Meta-Learning** (Opponent Profiling), **LoRA Adaptation** (Strategy Switching), and **RAG** (Tactical Recovery).

## 1. The Core Concept
The system is designed to handle three levels of adaptation:
1.  **Pre-Game (Strategic):** Select the best "Mindset" (LoRA) based on the specific opponent's history.
2.  **Mid-Game (Tactical):** Use the specialized LoRA to execute the chosen strategy (e.g., "Anti-Zerg Rush").
3.  **Emergency (Recovery):** If the strategy fails (Value drops), fallback to "Supervisor Mode" (Vector DB) to retrieve a specific bailout plan from a winning replay.

---

## 2. Architecture Components

### A. League Manager (The Meta-Controller)
This component sits outside the game loop and manages the agent's long-term knowledge of opponents.
*   **Technology:** Relational Database (SQLite).
*   **Data:** Stores Matchup Stats (`Opponent_ID` vs `LoRA_Strategy` -> `WinRate`).
*   **Logic (Bandit Algorithm):**
    *   **Exploitation (80%):** Choose the LoRA with the highest win rate against this specific `Opponent_ID`.
    *   **Exploration (20%):** Try a random LoRA or the "Standard" model to verify if better strategies exist.
*   **Output:** Selects the `Selected_LoRA` to load for the upcoming match.

### B. Adaptive Agent (The Mind)
Wraps the base neural network and manages dynamic weight loading.
*   **Base Model:** A frozen, general-purpose PPO agent (Trained on standard play).
*   **Adapters (LoRA):** Lightweight (~2MB) parameter sets trained for specific scenarios:
    *   `lora_standard.pt`: General balanced play.
    *   `lora_rush_defense.pt`: Trained to over-react to early scouting.
    *   `lora_macro_greed.pt`: Trained to prioritize economy.
*   **Mechanism:** At game start, the `League Manager` injects the chosen LoRA into the Base Model using `peft` (Parameter-Efficient Fine-Tuning).

### C. The Supervisor (The Safety Net / RAG)
A "Ghost in the Machine" that activates only when the agent is failing.
*   **Trigger:** Internal Value Function ($V$) drops below threshold (e.g., $V < 0.3$).
*   **Mechanism:**
    1.  **Disengage LoRA:** Disable the specialized strategy (it's failing).
    2.  **Engage RAG:** Query the Vector Database using a **30-second Trajectory Key** (GRU Hidden States).
    3.  **Retrieve:** Find a "Winning Script" (Sequence of 100+ actions) from a historical game that matches the current crisis.
    4.  **Autopilot:** Lock controls to the retrieved script until the situation stabilizes.

---

## 3. Workflow Diagram

1.  **Match Found:** Opponent is `Bot_X`.
2.  **League Manager:** Queries DB. "We have a 10% winrate vs `Bot_X` with `Macro`, but 80% with `Cheese`. Selecting `Cheese_LoRA`."
3.  **Game Start:** Agent loads `Base_Model` + `Cheese_LoRA`.
4.  **Mid-Game:**
    *   *Scenario A (Winning):* Value stays high. Agent executes `Cheese_LoRA`.
    *   *Scenario B (Failing):* Opponent counters the cheese. Value drops to 0.2.
        *   **Switch:** Disable LoRA. Enable Supervisor.
        *   **Retrieve:** DB finds a "Transition to Defense" plan from Replay #55.
        *   **Execute:** Agent follows the plan to recover.
5.  **Post-Game:** Update SQLite: `Bot_X` + `Cheese_LoRA` = `Result`.

---

## 4. League Implementation Constraints (SC2 AI Arena)
The SC2 AI Arena terminates processes immediately after a match. Post-game replay analysis is impossible.
**Solution:** The "Live-to-Persistent" Workflow using the `./data/` directory.

### A. Live Recording (RAM)
*   The agent cannot rely on parsing replay files later.
*   **Mechanism:** The `Supervisor` maintains a strictly bounded `HistoryBuffer` in RAM during the match.
    *   Stores: `(Trajectory_Key, Action_Sequence, Value_Estimate)`.

### B. The "Commit" (Game End)
*   **Trigger:** `on_end(game_result)` callback.
*   **Logic:**
    *   If `Result == WIN`: The RAM buffer represents a winning strategy.
    *   **Action:** Serialize the buffer and **append** it to a lightweight queue file: `./data/new_memories.jsonl`.
    *   *Constraint:* Must be extremely fast (milliseconds) to avoid server timeout/kill.

### C. The "Ingestion" (Game Start)
*   **Trigger:** `__init__` or `on_start` (Setup Phase).
*   **Logic:**
    1.  Check `./data/new_memories.jsonl`.
    2.  If data exists:
        *   Parse the JSONL.
        *   **Vectorize & Index:** Add entries to the local Vector Database (also stored in `./data/chromadb`).
        *   **Clear:** Empty the JSONL file.
    3.  **Load:** Load the updated Vector DB and SQLite Matchup DB into memory.

---

## 5. Implementation Strategy

### Phase 1: Offline Data & RAG (The Supervisor)
*   **Goal:** Build the "Safety Net" first.
*   **Task:** Implement the `SupervisorAgent` wrapper and the Vector DB ingestion pipeline for winning replays.
*   **Ref:** See `conductor/tracks/strategy_rag_agent.md` (Previous version).

### Phase 2: LoRA Training (The Specialist)
*   **Goal:** Create the "Mindsets".
*   **Task:**
    1.  Freeze the main PPO model.
    2.  Create specialized Gym environments (e.g., "Defend Cannon Rush").
    3.  Train LoRA adapters on these envs.

### Phase 3: League Integration (The Meta)
*   **Goal:** Tie it all together.
*   **Task:** Implement `src/utils/league_manager.py` with SQLite and UCB logic.

---

## 5. Technical Details: Trajectory Keys
*   **Problem:** Matching a single frame is brittle.
*   **Solution:** Use the **GRU Hidden State** as the search key.
*   **Why:** In a recurrent network, the hidden state $h_t$ mathematically represents the compressed history of the game up to time $t$. It is the perfect "Context Vector" for retrieval.
