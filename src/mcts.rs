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
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(&self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.maximize_ucb_for_side(&self.s2_options.as_ref().unwrap());
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
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
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
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

pub struct PrincipalVariationStep {
    pub s1_move: String,
    pub s2_move: String,
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
    pub principal_variation: Vec<PrincipalVariationStep>,
}

fn do_mcts(root_node: &mut Node, state: &mut State, root_eval: &f32) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
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
pub struct MctsSearch {
    root: Box<Node>,
    state: State,
    root_eval: f32,
}

impl MctsSearch {
    /// Build a new search anchored at `state` with the given root options.
    pub fn new(
        state: State,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
    ) -> Self {
        let mut root = Box::new(Node::new());
        // SAFETY: fresh root, no aliases yet.
        unsafe {
            root.populate(s1_options, s2_options);
        }
        root.root = true;

        let root_eval = evaluate(&state);
        MctsSearch {
            root,
            state,
            root_eval,
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
                do_mcts(&mut self.root, &mut self.state, &self.root_eval);
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

#[cfg(test)]
mod stream_tests {
    use super::*;

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
}
