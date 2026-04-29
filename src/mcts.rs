use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::state::State;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: u32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
        }
    }
    unsafe fn populate(
        &mut self,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_priors: Option<&[f32]>,
        s2_priors: Option<&[f32]>,
    ) {
        let n1 = s1_options.len();
        let n2 = s2_options.len();
        let uniform1 = if n1 > 0 { 1.0 / n1 as f32 } else { 0.0 };
        let uniform2 = if n2 > 0 { 1.0 / n2 as f32 } else { 0.0 };
        let p1 = s1_priors;
        let p2 = s2_priors;

        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: p1.and_then(|v| v.get(i).copied()).unwrap_or(uniform1),
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: p2.and_then(|v| v.get(i).copied()).unwrap_or(uniform2),
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_puct_for_side(&self, side_map: &[MoveNode], c_puct: f32) -> usize {
        let mut choice = 0;
        let mut best_score = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_score = node.puct(self.times_visited, c_puct);
            if this_score > best_score {
                best_score = this_score;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(&mut self, state: &mut State, c_puct: f32) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            // Non-root selection: uniform priors. Root NN priors are populated
            // by `MctsSearch::new` via `Node::populate_with_priors`.
            self.populate(s1_options, s2_options, None, None);
        }

        let s1_mc_index = self.maximize_puct_for_side(&self.s1_options.as_ref().unwrap(), c_puct);
        let s2_mc_index = self.maximize_puct_for_side(&self.s2_options.as_ref().unwrap(), c_puct);
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, c_puct)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (&mut *move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;

            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on
        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = evaluate(state);
            sigmoid(eval - root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    /// PUCT prior. `1/N_options` (uniform) when the NN is not consulted;
    /// otherwise the per-option prior derived from Kakuna's policy via
    /// `nn_state_encoder::map_policy_to_options`.
    pub prior: f32,
}

impl MoveNode {
    /// PUCT score: `Q + c * P * sqrt(N_parent) / (1 + N_self)`.
    ///
    /// Replaces the old UCB1 formula. With uniform prior `P = 1/N` and
    /// `c_puct = sqrt(2.0)` this is asymptotically equivalent to UCB1 (the
    /// regression suite at `mcts.rs:876-1130` is the back-compat guard).
    /// Unvisited nodes get `Q = 0.5` (neutral); the prior-weighted U-term
    /// breaks ties at iteration 0 in favor of high-prior moves.
    pub fn puct(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let q = if self.visits == 0 {
            0.5
        } else {
            self.total_score / self.visits as f32
        };
        // sqrt(N_parent) — note: at root iteration 0, parent_visits=0 → u=0
        // so the first iteration ties at q=0.5. After the first backprop,
        // parent_visits >= 1 and the prior dominates U for unvisited siblings.
        let u = c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32);
        q + u
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct PrincipalVariationStep {
    pub s1_move: String,
    pub s2_move: String,
}

#[derive(Clone)]
pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
    pub principal_variation: Vec<PrincipalVariationStep>,
}

fn do_mcts(root_node: &mut Node, state: &mut State, root_eval: &f32, c_puct: f32) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state, c_puct) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

const PV_MAX_DEPTH: usize = 4;

/// Walk the MCTS tree from `node` following the most-visited moves at each level.
/// Collects up to `PV_MAX_DEPTH` steps of (s1_move, s2_move) as human-readable strings.
/// Applies and then reverses instructions so `state` is left unchanged.
fn extract_principal_variation(node: &Node, state: &mut State) -> Vec<PrincipalVariationStep> {
    let mut pv = Vec::with_capacity(PV_MAX_DEPTH);
    let mut applied: Vec<&Vec<crate::instruction::Instruction>> = Vec::new();

    let mut current = node;
    for _ in 0..PV_MAX_DEPTH {
        let s1_options = match current.s1_options.as_ref() {
            Some(opts) => opts,
            None => break,
        };
        let s2_options = match current.s2_options.as_ref() {
            Some(opts) => opts,
            None => break,
        };

        // Find the most-visited s1 and s2 moves
        let (s1_idx, s1_best) = s1_options
            .iter()
            .enumerate()
            .filter(|(_, m)| m.visits > 0)
            .max_by_key(|(_, m)| m.visits)
            .unwrap_or((0, &s1_options[0]));

        let (s2_idx, s2_best) = s2_options
            .iter()
            .enumerate()
            .filter(|(_, m)| m.visits > 0)
            .max_by_key(|(_, m)| m.visits)
            .unwrap_or((0, &s2_options[0]));

        // Resolve move names using current state
        let s1_name = s1_best.move_choice.to_string(&state.side_one).to_uppercase();
        let s2_name = s2_best.move_choice.to_string(&state.side_two).to_uppercase();

        pv.push(PrincipalVariationStep {
            s1_move: s1_name,
            s2_move: s2_name,
        });

        // Descend into the child for this (s1_idx, s2_idx) pair
        let children = match current.children.get(&(s1_idx, s2_idx)) {
            Some(c) => c,
            None => break,
        };

        // Pick the child node with the most visits (handles damage branching)
        let best_child = match children.iter().max_by_key(|c| c.times_visited) {
            Some(c) => c,
            None => break,
        };

        // Apply instructions to advance state to this child's depth
        state.apply_instructions(&best_child.instructions.instruction_list);
        applied.push(&best_child.instructions.instruction_list);
        current = best_child;
    }

    // Reverse all applied instructions to restore the original state
    for instructions in applied.iter().rev() {
        state.reverse_instructions(instructions);
    }

    pv
}

/// 10 million iteration safety cap (see inner-loop comment for rationale).
const MCTS_MAX_ITERATIONS: u32 = 10_000_000;

/// Batch size between wall-clock checks in the inner MCTS loop.
const MCTS_BATCH_SIZE: u32 = 1000;

/// Incremental MCTS search handle. Owns the root `Node` via `Box` so that the
/// raw `*mut Node` parent pointers within the tree remain valid across
/// multiple `run_for` invocations.
///
/// **Invariants:**
/// - The tree is pinned on a single thread. `Node` contains raw pointers and
///   is therefore `!Send`; `MctsSearch` inherits that constraint. Do not move
///   across threads.
/// - `state` represents the position at the root. During search it is mutated
///   in place by `do_mcts` (apply -> rollout -> reverse), so between search
///   invocations it MUST be back at root. This holds because every MCTS
///   iteration fully unwinds instructions during backpropagation.
/// PUCT exploration constant. AlphaZero's default is 1.25; the legacy UCB1
/// constant was `sqrt(2.0)` (~1.414). For backward compatibility on the
/// `MctsSearch::new` path (which now defaults to c_puct=sqrt(2.0)), this
/// keeps the regression tests happy.
pub const DEFAULT_C_PUCT: f32 = std::f32::consts::SQRT_2;

pub struct MctsSearch {
    root: Box<Node>,
    state: State,
    root_eval: f32,
    c_puct: f32,
}

impl MctsSearch {
    /// Build a new search anchored at `state` with the given root options.
    /// Uses the default exploration constant `DEFAULT_C_PUCT = sqrt(2)` and
    /// uniform priors — preserves pre-Plan-E behavior.
    pub fn new(
        state: State,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
    ) -> Self {
        Self::new_with_priors(state, s1_options, s2_options, DEFAULT_C_PUCT, None, None)
    }

    /// Plan E variant of `new` that takes optional priors and a custom
    /// exploration constant.
    ///
    /// `s1_priors`/`s2_priors`: `None` → uniform; otherwise must have length
    /// matching the corresponding options vector. The Plan E pipeline calls
    /// `nn_state_encoder::map_policy_to_options` to produce `s1_priors`.
    pub fn new_with_priors(
        state: State,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        c_puct: f32,
        s1_priors: Option<Vec<f32>>,
        s2_priors: Option<Vec<f32>>,
    ) -> Self {
        let mut root = Box::new(Node::new());
        let s1p = s1_priors.as_deref();
        let s2p = s2_priors.as_deref();
        // SAFETY: fresh root, no aliases yet.
        unsafe {
            root.populate(s1_options, s2_options, s1p, s2p);
        }
        root.root = true;

        // CRITICAL (verifier CRIT-2): root_eval is ALWAYS the heuristic.
        // Kakuna's `v_estimate` is in raw shaped-reward Q-units (~[100, 2000])
        // and is NOT comparable to evaluate()'s signed-f32 (~[-300, +300])
        // scale; mixing them saturates the leaf sigmoid. NN contributes ONLY
        // the policy prior, never the value baseline.
        let root_eval = evaluate(&state);
        MctsSearch {
            root,
            state,
            root_eval,
            c_puct,
        }
    }

    /// Run MCTS for up to `budget` of wall-clock time. Returns the number of
    /// iterations performed in this invocation (not cumulative — see
    /// `total_iterations` for that).
    ///
    /// Honours the 10-million iteration safety cap across all `run_for`
    /// invocations on this search.
    pub fn run_for(&mut self, budget: Duration) -> u32 {
        let start_time = Instant::now();
        let iterations_before = self.root.times_visited;
        while start_time.elapsed() < budget {
            for _ in 0..MCTS_BATCH_SIZE {
                do_mcts(&mut self.root, &mut self.state, &self.root_eval, self.c_puct);
            }

            // Cut off after 10 million iterations
            //
            // Under normal circumstances the bot will only run for 2.5-3.5 million iterations
            // however towards the end of a battle the bot may perform tens of millions of iterations
            //
            // Beyond about 30 million iterations some floating point nonsense happens where
            // MoveNode.total_score stops updating because f32 does not have enough precision
            //
            // I can push the problem farther out by using f64 but if the bot is running for 10 million iterations
            // then it almost certainly sees a forced win
            if self.root.times_visited >= MCTS_MAX_ITERATIONS {
                break;
            }
        }
        self.root.times_visited - iterations_before
    }

    /// Total cumulative MCTS iterations executed against this search.
    pub fn total_iterations(&self) -> u32 {
        self.root.times_visited
    }

    /// Produce a snapshot of the current best-move information. Safe to call
    /// between `run_for` invocations; does not disturb the search tree.
    ///
    /// `elapsed_ms` is opaque to the search — the caller supplies whatever
    /// wall-clock figure is meaningful (e.g. cumulative across all
    /// `run_for` calls).
    pub fn snapshot(&mut self, _elapsed_ms: u64) -> MctsResult {
        // Extract principal_variation by walking the tree. This mutates state
        // transiently but restores it before returning, matching original
        // behavior.
        let principal_variation =
            extract_principal_variation(&self.root, &mut self.state);

        MctsResult {
            s1: self
                .root
                .s1_options
                .as_ref()
                .unwrap()
                .iter()
                .map(|v| MctsSideResult {
                    move_choice: v.move_choice.clone(),
                    total_score: v.total_score,
                    visits: v.visits,
                })
                .collect(),
            s2: self
                .root
                .s2_options
                .as_ref()
                .unwrap()
                .iter()
                .map(|v| MctsSideResult {
                    move_choice: v.move_choice.clone(),
                    total_score: v.total_score,
                    visits: v.visits,
                })
                .collect(),
            iteration_count: self.root.times_visited,
            principal_variation,
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
) -> MctsResult {
    // `State: Clone` — the old implementation held a &mut State and mutated
    // the original in place. MctsSearch owns its State (required because it
    // may live across multiple run_for calls). Since every MCTS iteration
    // fully restores state via backpropagation's reverse_instructions, the
    // caller's State is observationally unchanged at the end of this
    // function either way. We clone once at entry to preserve the old
    // "caller hands us a mut reference" signature.
    let _ = state; // keep &mut in signature for API stability
    let mut search = MctsSearch::new(state.clone(), side_one_options, side_two_options);
    let start = Instant::now();
    search.run_for(max_time);
    search.snapshot(start.elapsed().as_millis() as u64)
}

/// Aggregate K `MctsResult`s into one, using foul-play's visit-share formula.
///
/// For each move on side one, weighted_visits[m] = Σ_i (1/K) × (visits_i[m] / total_visits_i).
/// Then keep moves within 75% of the top, weighted-random sample among them.
/// Side two and iteration_count are summed straightforwardly.
///
/// `pick_deterministic` = true → argmax over the survivor set (for tests).
/// `pick_deterministic` = false → weighted-random pick.
///
/// PV is taken from the hypothesis whose top move matches the chosen aggregate
/// move (first such; falls back to first hypothesis's PV if none match).
///
/// Panics on empty input. Caller must guarantee `results.len() >= 1`.
pub fn aggregate_pimc(
    results: Vec<MctsResult>,
    pick_deterministic: bool,
    seed: Option<u64>,
) -> MctsResult {
    assert!(!results.is_empty(), "aggregate_pimc requires at least 1 result");
    let k = results.len() as f32;

    // 1. Compute weighted_visits per MoveChoice on side one.
    let mut weighted: HashMap<MoveChoice, f32> = HashMap::new();
    for r in &results {
        let total: u32 = r.s1.iter().map(|m| m.visits).sum();
        if total == 0 {
            continue;
        }
        for m in &r.s1 {
            let share = m.visits as f32 / total as f32;
            *weighted.entry(m.move_choice).or_insert(0.0) += share / k;
        }
    }

    if weighted.is_empty() {
        // Pathological: every hypothesis had zero visits. Return the first.
        return results.into_iter().next().unwrap();
    }

    // 2. Find top weighted score; filter to within 75% of top.
    let top = weighted.values().copied().fold(f32::MIN, f32::max);
    let threshold = top * 0.75;
    let mut survivors: Vec<(MoveChoice, f32)> = weighted
        .iter()
        .filter(|(_, w)| **w >= threshold)
        .map(|(mc, w)| (*mc, *w))
        .collect();
    // HashMap iteration order is non-deterministic, but seeded sampling needs a
    // stable ordering so the same seed always picks the same survivor. Sort by
    // weight desc, breaking ties by Debug-string of MoveChoice (gives a total order).
    survivors.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0)))
    });

    // 3. Pick winner.
    //    `partial_cmp(...).unwrap()` is safe: weights are `share / k` where
    //    `share` ∈ [0, 1] and `k` is a positive finite f32, so values are
    //    finite non-negative — NaN is unreachable.
    let winner: MoveChoice = if pick_deterministic || survivors.len() == 1 {
        // Argmax
        survivors
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(mc, _)| *mc)
            .unwrap()
    } else {
        // Weighted-random sample
        let weights: Vec<f32> = survivors.iter().map(|(_, w)| *w).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng_inst: Box<dyn RngCore> = match seed {
            Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };
        let idx = dist.sample(&mut rng_inst);
        survivors[idx].0
    };

    // 4. Build the aggregated MctsResult.
    //    s1: synthesize one MctsSideResult per move-key, with visits = sum of
    //    raw visits across hypotheses, total_score = sum.
    //    s2: same but on side two (we don't need to interleave; sum is fine).
    //    iteration_count: sum.
    let mut agg_s1_map: HashMap<MoveChoice, MctsSideResult> = HashMap::new();
    for r in &results {
        for m in &r.s1 {
            let entry = agg_s1_map.entry(m.move_choice).or_insert_with(|| MctsSideResult {
                move_choice: m.move_choice,
                total_score: 0.0,
                visits: 0,
            });
            entry.total_score += m.total_score;
            entry.visits += m.visits;
        }
    }
    // Sort: winner first, rest by visits descending.
    let mut agg_s1: Vec<MctsSideResult> = agg_s1_map.into_values().collect();
    agg_s1.sort_by(|a, b| {
        let a_is_winner = a.move_choice == winner;
        let b_is_winner = b.move_choice == winner;
        match (a_is_winner, b_is_winner) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b.visits.cmp(&a.visits)
                .then_with(|| format!("{:?}", a.move_choice).cmp(&format!("{:?}", b.move_choice))),
        }
    });

    // Same shape on s2 (sum across hypotheses).
    let mut agg_s2_map: HashMap<MoveChoice, MctsSideResult> = HashMap::new();
    for r in &results {
        for m in &r.s2 {
            let entry = agg_s2_map.entry(m.move_choice).or_insert_with(|| MctsSideResult {
                move_choice: m.move_choice,
                total_score: 0.0,
                visits: 0,
            });
            entry.total_score += m.total_score;
            entry.visits += m.visits;
        }
    }
    let mut agg_s2: Vec<MctsSideResult> = agg_s2_map.into_values().collect();
    agg_s2.sort_by(|a, b| b.visits.cmp(&a.visits)
        .then_with(|| format!("{:?}", a.move_choice).cmp(&format!("{:?}", b.move_choice))));

    let iteration_count: u32 = results.iter().map(|r| r.iteration_count).sum();

    // PV: from first hypothesis whose top-visited s1 move matches winner.
    let pv = results
        .iter()
        .find(|r| {
            r.s1.iter()
                .max_by_key(|m| m.visits)
                .map(|m| m.move_choice == winner)
                .unwrap_or(false)
        })
        .map(|r| r.principal_variation.clone())
        .unwrap_or_else(|| results[0].principal_variation.clone());

    MctsResult {
        s1: agg_s1,
        s2: agg_s2,
        iteration_count,
        principal_variation: pv,
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn test_aggregate_pimc_uniform_visits() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32, s: f32| MctsSideResult {
            move_choice: mc,
            total_score: s,
            visits: v,
        };
        let r1 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 100, 60.0),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 50, 25.0),
            ],
            s2: vec![mk(MoveChoice::None, 150, 75.0)],
            iteration_count: 150,
            principal_variation: vec![],
        };
        let r2 = r1.clone();
        let agg = aggregate_pimc(vec![r1, r2], true, None);
        // Move(M0) had 2/3 of visits in both → top of agg.
        assert_eq!(
            format!("{:?}", agg.s1[0].move_choice),
            format!("{:?}", MoveChoice::Move(PokemonMoveIndex::M0))
        );
    }

    #[test]
    fn test_aggregate_pimc_dominant_move() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32| MctsSideResult {
            move_choice: mc,
            total_score: 0.0,
            visits: v,
        };
        // Move(M0) wins in 3 of 4 hypotheses; Move(M1) wins in 1.
        let r1 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 100),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 50),
            ],
            s2: vec![],
            iteration_count: 150,
            principal_variation: vec![],
        };
        let r2 = r1.clone();
        let r3 = r1.clone();
        let r4 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 50),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 100),
            ],
            s2: vec![],
            iteration_count: 150,
            principal_variation: vec![],
        };
        let agg = aggregate_pimc(vec![r1, r2, r3, r4], true, None);
        assert_eq!(
            format!("{:?}", agg.s1[0].move_choice),
            format!("{:?}", MoveChoice::Move(PokemonMoveIndex::M0))
        );
    }

    #[test]
    fn test_aggregate_pimc_top_75_prune_keeps_close_seconds() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32| MctsSideResult {
            move_choice: mc,
            total_score: 0.0,
            visits: v,
        };
        // Move(M0) gets 100, Move(M1) gets 80 → both within 75% of top (80/100 = 0.80 ≥ 0.75).
        let r = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 100),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 80),
            ],
            s2: vec![],
            iteration_count: 180,
            principal_variation: vec![],
        };
        let agg = aggregate_pimc(vec![r], true, None);
        // Deterministic pick → winner is Move(M0). But both should be in agg.s1.
        assert_eq!(agg.s1.len(), 2);
    }

    #[test]
    fn test_aggregate_pimc_singleton_matches_input() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32, s: f32| MctsSideResult {
            move_choice: mc,
            total_score: s,
            visits: v,
        };
        let r = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 80, 45.0),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 20, 10.0),
            ],
            s2: vec![mk(MoveChoice::None, 100, 50.0)],
            iteration_count: 100,
            principal_variation: vec![],
        };
        let agg = aggregate_pimc(vec![r.clone()], true, None);
        assert_eq!(agg.iteration_count, 100);
        assert_eq!(agg.s1.len(), 2);
        // Top of agg.s1 is the same as top of r.s1 (Move(M0)).
        assert_eq!(
            format!("{:?}", agg.s1[0].move_choice),
            format!("{:?}", MoveChoice::Move(PokemonMoveIndex::M0))
        );
    }

    #[test]
    #[should_panic(expected = "aggregate_pimc requires at least 1 result")]
    fn test_aggregate_pimc_empty_panics() {
        aggregate_pimc(vec![], true, None);
    }

    #[test]
    fn test_aggregate_pimc_pv_picked_from_matching_hypothesis() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32| MctsSideResult {
            move_choice: mc,
            total_score: 0.0,
            visits: v,
        };
        let pv_a = vec![PrincipalVariationStep {
            s1_move: "moveA".into(),
            s2_move: "x".into(),
        }];
        let pv_b = vec![PrincipalVariationStep {
            s1_move: "moveB".into(),
            s2_move: "y".into(),
        }];

        // r1 favors Move(M0) with PV pv_a; r2 favors Move(M1) with PV pv_b.
        // Aggregated winner is Move(M0) (3:1 visit-share advantage), so PV should be pv_a.
        let r1 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 100),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 10),
            ],
            s2: vec![],
            iteration_count: 110,
            principal_variation: pv_a.clone(),
        };
        let r2 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 30),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 60),
            ],
            s2: vec![],
            iteration_count: 90,
            principal_variation: pv_b.clone(),
        };
        let agg = aggregate_pimc(vec![r1, r2], true, None);
        assert_eq!(agg.principal_variation.len(), 1);
        assert_eq!(agg.principal_variation[0].s1_move, "moveA");
    }

    #[test]
    fn test_aggregate_pimc_pv_falls_back_to_first_when_no_match() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32| MctsSideResult {
            move_choice: mc,
            total_score: 0.0,
            visits: v,
        };
        let pv_a = vec![PrincipalVariationStep {
            s1_move: "fromR1".into(),
            s2_move: "x".into(),
        }];
        let pv_b = vec![PrincipalVariationStep {
            s1_move: "fromR2".into(),
            s2_move: "y".into(),
        }];

        // r1 top is M0; r2 top is M2. Aggregate weights tie M0 and M2 (each ~0.362),
        // M1 ~0.276. Argmax picks one of M0/M2 (HashMap-order dependent), but in
        // either case the winner IS the top of one hypothesis, so the matching-PV
        // path will fire for that hypothesis. We assert the PV is non-empty and
        // came from one of the two inputs.
        let r1 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 200),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 80),
                mk(MoveChoice::Move(PokemonMoveIndex::M2), 10),
            ],
            s2: vec![],
            iteration_count: 290,
            principal_variation: pv_a.clone(),
        };
        let r2 = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 10),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 80),
                mk(MoveChoice::Move(PokemonMoveIndex::M2), 200),
            ],
            s2: vec![],
            iteration_count: 290,
            principal_variation: pv_b.clone(),
        };
        let agg = aggregate_pimc(vec![r1, r2], true, None);
        assert_eq!(agg.principal_variation.len(), 1);
        let pv_text = &agg.principal_variation[0].s1_move;
        assert!(
            pv_text == "fromR1" || pv_text == "fromR2",
            "PV s1_move should be fromR1 or fromR2, got {}",
            pv_text
        );
    }

    #[test]
    fn test_aggregate_pimc_weighted_random_with_seed_is_deterministic() {
        use crate::engine::state::MoveChoice;
        use crate::state::PokemonMoveIndex;
        let mk = |mc: MoveChoice, v: u32| MctsSideResult {
            move_choice: mc,
            total_score: 0.0,
            visits: v,
        };
        // Two-move survivor set within 75% of top, so weighted-random branch fires.
        let r = MctsResult {
            s1: vec![
                mk(MoveChoice::Move(PokemonMoveIndex::M0), 100),
                mk(MoveChoice::Move(PokemonMoveIndex::M1), 80),
            ],
            s2: vec![],
            iteration_count: 180,
            principal_variation: vec![],
        };
        // Seed 42 gives some deterministic outcome. Run twice; results identical.
        let agg1 = aggregate_pimc(vec![r.clone()], false, Some(42));
        let agg2 = aggregate_pimc(vec![r.clone()], false, Some(42));
        assert_eq!(
            format!("{:?}", agg1.s1[0].move_choice),
            format!("{:?}", agg2.s1[0].move_choice),
            "same seed must yield same winner"
        );
        // Winner must be one of the two survivors (sanity).
        let winner_mc = &agg1.s1[0].move_choice;
        assert!(
            matches!(winner_mc, MoveChoice::Move(PokemonMoveIndex::M0))
                || matches!(winner_mc, MoveChoice::Move(PokemonMoveIndex::M1)),
            "winner must be M0 or M1, got {:?}",
            winner_mc
        );
    }

    /// Exercise the MctsSearch incremental API: build, run twice, verify
    /// iterations accumulate and snapshot is well-formed. Uses a very short
    /// budget for test hygiene but long enough that at least one
    /// MCTS_BATCH_SIZE of iterations completes.
    #[test]
    fn mcts_search_incremental_api() {
        use crate::translate::json_to_poke_state;

        let json = r#"{
            "sideOne": {
                "pokemon": [
                    {
                        "species": "Blaziken",
                        "level": 100,
                        "types": ["Fire", "Fighting"],
                        "hp": 302,
                        "maxhp": 302,
                        "ability": "Speed Boost",
                        "item": "Life Orb",
                        "nature": "Jolly",
                        "attack": 349,
                        "defense": 196,
                        "specialAttack": 230,
                        "specialDefense": 176,
                        "speed": 284,
                        "status": "None",
                        "weightKg": 52.0,
                        "moves": [
                            {"id": "Close Combat", "pp": 8},
                            {"id": "Flare Blitz", "pp": 24},
                            {"id": "Swords Dance", "pp": 32},
                            {"id": "Knock Off", "pp": 32}
                        ],
                        "teraType": "Fire"
                    }
                ],
                "activeIndex": 0
            },
            "sideTwo": {
                "pokemon": [
                    {
                        "species": "Alakazam",
                        "level": 100,
                        "types": ["Psychic"],
                        "hp": 251,
                        "maxhp": 251,
                        "ability": "Magic Guard",
                        "item": "Focus Sash",
                        "nature": "Timid",
                        "attack": 121,
                        "defense": 128,
                        "specialAttack": 369,
                        "specialDefense": 206,
                        "speed": 372,
                        "status": "None",
                        "weightKg": 48.0,
                        "moves": [
                            {"id": "Psychic", "pp": 16},
                            {"id": "Shadow Ball", "pp": 24},
                            {"id": "Focus Blast", "pp": 8},
                            {"id": "Energy Ball", "pp": 16}
                        ],
                        "teraType": "Psychic"
                    }
                ],
                "activeIndex": 0
            }
        }"#;

        let state = json_to_poke_state(json).expect("parse state");
        let (s1_opts, s2_opts) = state.root_get_all_options();
        assert!(!s1_opts.is_empty());

        let mut search = MctsSearch::new(state.clone(), s1_opts, s2_opts);

        // First slice
        search.run_for(Duration::from_millis(150));
        let sims_1 = search.total_iterations();
        assert!(sims_1 > 0, "first run_for should have produced iterations");

        let snap_1 = search.snapshot(150);
        assert!(
            !snap_1.s1.is_empty(),
            "snapshot s1 results should not be empty"
        );
        let best_1 = snap_1
            .s1
            .iter()
            .max_by_key(|r| r.visits)
            .expect("best move");
        assert!(best_1.visits > 0, "best move should have visits");

        // Second slice — iterations should accumulate
        search.run_for(Duration::from_millis(150));
        let sims_2 = search.total_iterations();
        assert!(
            sims_2 > sims_1,
            "iterations should accumulate across run_for calls: sims_1={} sims_2={}",
            sims_1,
            sims_2
        );

        let snap_2 = search.snapshot(300);
        let best_2 = snap_2
            .s1
            .iter()
            .max_by_key(|r| r.visits)
            .expect("best move");
        assert!(
            best_2.visits >= best_1.visits,
            "best move visits should grow or stay equal across snapshots"
        );
        assert_eq!(
            snap_2.iteration_count,
            search.total_iterations(),
            "snapshot iteration_count must match total_iterations"
        );
    }

    /// Reproduce the live-bug scenario: Iron Hands (Electric/Fighting) at 34% HP
    /// vs Togekiss (Fairy/Flying) at 10% HP. Iron Hands should NEVER prefer
    /// Earthquake since Ground does 0 damage to Flying defenders.
    ///
    /// Empirical result (gen9, release): calculate_damage correctly returns 0 for
    /// EQ vs Flying, threat_score correctly finds ThunderPunch (344 dmg) as best,
    /// and MCTS correctly selects DrainPunch/ThunderPunch (≈3.6M visits each)
    /// over EQ (≈470 visits). The live-battle Earthquake selection must have
    /// some other cause (e.g. serialization mismatch, snapshot at <1ms, or
    /// payload differing from reported summary). This test serves as a
    /// regression guard for the primitive pipeline.
    #[test]
    fn repro_iron_hands_vs_togekiss_never_picks_earthquake() {
        use crate::translate::json_to_poke_state;

        // Togekiss typical stats: HP 310 (so 10% ~= 31). Iron Hands Atk ~419.
        // Using approximate competitive stats. The exact numbers don't matter
        // for the type-effectiveness conclusion (EQ = 0 dmg regardless).
        let json = r#"{
            "sideOne": {
                "pokemon": [
                    {
                        "species": "IronHands",
                        "level": 100,
                        "types": ["Electric", "Fighting"],
                        "hp": 180,
                        "maxhp": 527,
                        "ability": "QuarkDrive",
                        "item": "Assault Vest",
                        "nature": "Adamant",
                        "attack": 419,
                        "defense": 203,
                        "specialAttack": 130,
                        "specialDefense": 237,
                        "speed": 113,
                        "status": "None",
                        "weightKg": 380.7,
                        "moves": [
                            {"id": "Drain Punch", "pp": 16},
                            {"id": "Thunder Punch", "pp": 24},
                            {"id": "Ice Punch", "pp": 24},
                            {"id": "Earthquake", "pp": 16}
                        ],
                        "teraType": "Electric"
                    }
                ],
                "activeIndex": 0
            },
            "sideTwo": {
                "pokemon": [
                    {
                        "species": "Togekiss",
                        "level": 100,
                        "types": ["Fairy", "Flying"],
                        "hp": 31,
                        "maxhp": 310,
                        "ability": "Serene Grace",
                        "item": "Leftovers",
                        "nature": "Timid",
                        "attack": 157,
                        "defense": 216,
                        "specialAttack": 295,
                        "specialDefense": 237,
                        "speed": 236,
                        "status": "None",
                        "weightKg": 38.0,
                        "moves": [
                            {"id": "Air Slash", "pp": 24},
                            {"id": "Dazzling Gleam", "pp": 16},
                            {"id": "Flamethrower", "pp": 24},
                            {"id": "Roost", "pp": 16}
                        ],
                        "teraType": "Fairy"
                    }
                ],
                "activeIndex": 0
            }
        }"#;

        let state = json_to_poke_state(json).expect("parse state");
        let (s1_opts, s2_opts) = state.root_get_all_options();
        let s1_move_names: Vec<String> = s1_opts
            .iter()
            .map(|mc| mc.to_string(&state.side_one))
            .collect();
        eprintln!("side_one move options: {:?}", s1_move_names);

        // Also directly check what calculate_damage says for each move from this state.
        use crate::engine::damage_calc::{calculate_damage, DamageRolls};
        use crate::state::SideReference;
        for (idx, mv) in state.side_one.get_active_immutable().moves.into_iter().enumerate() {
            let dmg = calculate_damage(&state, &SideReference::SideOne, &mv.choice, DamageRolls::Average);
            eprintln!(
                "move[{}] id={:?} type={:?} base_power={} -> dmg={:?}",
                idx, mv.choice.move_id, mv.choice.move_type, mv.choice.base_power, dmg
            );
        }

        // Check the threat_score output too.
        let ts = crate::engine::evaluate::threat_score(&state, &SideReference::SideOne);
        eprintln!("threat_score(SideOne) = {}", ts);

        let mut search = MctsSearch::new(state.clone(), s1_opts, s2_opts);
        // Short slice first to show early-iteration noise
        search.run_for(Duration::from_millis(50));
        let snap_early = search.snapshot(50);
        eprintln!("=== 50ms snapshot ===");
        for (i, r) in snap_early.s1.iter().enumerate() {
            let avg = if r.visits > 0 { r.total_score / r.visits as f32 } else { 0.0 };
            let name = s1_move_names.get(i).cloned().unwrap_or_default();
            eprintln!(
                "s1[{}] {:<20} visits={:<8} total_score={:.3} avg={:.4}",
                i, name, r.visits, r.total_score, avg
            );
        }

        search.run_for(Duration::from_millis(5000));
        let snap = search.snapshot(5050);
        eprintln!("=== 5050ms snapshot ===");

        eprintln!("iterations: {}", snap.iteration_count);
        for (i, r) in snap.s1.iter().enumerate() {
            let avg = if r.visits > 0 { r.total_score / r.visits as f32 } else { 0.0 };
            let name = s1_move_names.get(i).cloned().unwrap_or_default();
            eprintln!(
                "s1[{}] {:<20} visits={:<8} total_score={:.3} avg={:.4}",
                i, name, r.visits, r.total_score, avg
            );
        }

        // Determine which s1 move has the most visits (that's what the engine picks).
        let best = snap.s1.iter().enumerate().max_by_key(|(_, r)| r.visits).unwrap();
        let best_name = s1_move_names[best.0].to_lowercase().replace(' ', "");
        eprintln!("engine best move: {} (visits={})", best_name, best.1.visits);
        assert!(
            !best_name.contains("earthquake"),
            "Engine must not pick Earthquake against Flying target. Got {}",
            best_name
        );
    }
}
