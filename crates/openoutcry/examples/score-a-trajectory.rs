//! The end-to-end ecosystem path: **env → trajectory → SharpeBench score.**
//!
//! Run an agent in OpenOutcry while capturing only its raw decisions, then hand the
//! trajectory to a *separate verifier* that recomputes the submission from those
//! decisions + the frozen data alone (the agent cannot lie about its returns), and
//! finally score it with SharpeBench's `CompositeScore`.
//!
//! `cargo run -p openoutcry --example score-a-trajectory`

use openoutcry::{
    replay_submission, run_backtest_capture, AgentTrajectory, BuyAndHold, CostModel, Dataset,
    Window, CONTRACT_VERSION,
};
use sharpebench_core::{score_agent, ScoreConfig};

fn main() {
    let data = Dataset::synthetic(4, 160, 7);
    let window = Window {
        start: 20,
        end: 160,
    };
    let costs = CostModel::default();

    // 1) env → trajectory: run the agent with capture — the artifact holds ONLY the
    //    raw per-step decisions + the window/seed coordinates, no self-reported metric.
    let (_run, run_traj) = run_backtest_capture(&data, &mut BuyAndHold, window, 1, costs);
    let trajectory = AgentTrajectory {
        agent_id: "buy-and-hold".to_string(),
        runs: vec![run_traj],
        in_sample_trials: 0,
    };

    // 2) verify: recompute the scoreable submission from the decisions + frozen data
    //    alone. Tamper with the trajectory and this recomputes to different returns.
    let submission = replay_submission(&data, &trajectory, costs);

    // 3) score: deflated Sharpe / pass^k / process gate — the SharpeBench verdict.
    let score = score_agent(&submission, &ScoreConfig::default());

    println!(
        "OpenOutcry contract v{CONTRACT_VERSION} — scored '{}':",
        submission.agent_id
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&score).expect("score serializes")
    );
}
