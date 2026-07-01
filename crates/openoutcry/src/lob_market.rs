//! Deterministic continuous-double-auction limit-order book (M3) — price-time priority.
//!
//! The bar-level and endogenous-clearing markets ([`crate::market`]) model price as a
//! scalar that aggregate flow nudges. This module is the microstructure-faithful sibling:
//! a real CDA matching engine with a resting book, FIFO time priority per price level, and
//! partial fills. It is the M3 market surface.
//!
//! ## Determinism is the contract
//!
//! Reference CDA engines key their books on `Decimal`/float prices and break ties with
//! `random.shuffle`; neither is byte-identical across runtimes. Here:
//!
//! - **Prices are integer ticks** (`i64`), so the book keys are exact and order across
//!   Rust/WASM/Python is total and identical. `tick_size` is a display scalar only — it
//!   never keys the book.
//! - **Time priority is an explicit FIFO** `VecDeque` per price; there is no shuffle. The
//!   resting order id is the pre-call `next_order_id`, a monotone counter, so ids are a
//!   pure function of the submission sequence.
//! - **The batch [`OrderBook::step`] folds a bar's orders in canonical order** (sorted by
//!   agent, then submission index) before touching the book, so however a parallel
//!   collector assembles the actions, the matched tape is identical.
//! - The fill tape carries **only integers** (tick, qty, ids, agents); the golden FNV-1a
//!   test pins it. Derived observation scalars ([`LadderSnapshot::mid`] / `microprice` /
//!   `queue_imbalance`) use only `mul/add/div` over integer inputs — no `ln`/`exp`/`sqrt`
//!   — matching the sibling market's cross-runtime arithmetic discipline.
//!
//! ## Leak-free invariant
//!
//! [`OrderBook::step`] consumes a whole bar's orders and matches them against the resting
//! book and each other in canonical order; it never reads a peer's *pending* order out of
//! sequence. A fill price reflects the resting liquidity an aggressor crosses — the
//! price-discovery channel of a real book, not an information leak.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// Order side. Serialized lowercase for the JSON boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

/// A resting order on one side of the book at one price level. Its price and side are
/// carried by its location in the book; the struct holds identity, owner, and live size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestingOrder {
    pub id: u64,
    pub agent: usize,
    pub qty: u64,
}

/// One agent order. `Limit`/`Market` carry their own side; `Cancel`/`Modify` reference a
/// resting order by id (side and price are recovered from the book).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderKind {
    Limit {
        side: Side,
        price_tick: i64,
        qty: u64,
    },
    Market {
        side: Side,
        qty: u64,
    },
    Cancel {
        id: u64,
    },
    Modify {
        id: u64,
        new_qty: u64,
    },
}

/// A single match event: `qty` traded at the resting `price_tick`, between the aggressor
/// (`taker_*`) and the resting order it crossed (`maker_*`). All fields are integers, so
/// the tape is byte-identical across runtimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fill {
    pub price_tick: i64,
    pub qty: u64,
    pub maker_id: u64,
    pub maker_agent: usize,
    pub taker_agent: usize,
    pub taker_side: Side,
}

/// Top-N order-book observation: the bid/ask ladders (`[price_tick, qty]`, best first) and
/// derived microstructure scalars. `mid`, `microprice`, and `queue_imbalance` are in tick
/// units, computed from integer prices/sizes with `mul/add/div` only.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LadderSnapshot {
    /// `[price_tick, qty]` per bid level, best (highest) first.
    pub bids: Vec<[i64; 2]>,
    /// `[price_tick, qty]` per ask level, best (lowest) first.
    pub asks: Vec<[i64; 2]>,
    /// `(best_bid + best_ask) / 2` in tick units (one-sided or `0.0` when degenerate).
    pub mid: f64,
    /// Size-weighted mid `(bid*ask_qty + ask*bid_qty) / (bid_qty + ask_qty)`, tick units.
    pub microprice: f64,
    /// `(bid_qty - ask_qty) / (bid_qty + ask_qty)` over the best levels, in `[-1, 1]`.
    pub queue_imbalance: f64,
}

/// The cost of walking the book to fill a target size, as a read-only impact query (the
/// book is never mutated). `avg_px_tick` is the size-weighted fill price in tick units;
/// `slippage_ticks` is its absolute distance from the touched best price; `filled_qty` is
/// how much the visible depth could actually fill (`< qty` when the book runs dry). All
/// derived from integer prices/sizes with `mul/add/div` only, matching the ladder scalars.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SweepCost {
    /// Size-weighted average fill price in tick units (`0.0` when nothing fills).
    pub avg_px_tick: f64,
    /// `|avg_px_tick - best|` in ticks against the near touch (`0.0` when nothing fills).
    pub slippage_ticks: f64,
    /// Size actually fillable from visible depth (`<= qty`).
    pub filled_qty: u64,
}

/// A price-time-priority continuous-double-auction book. `bids`/`asks` are keyed by integer
/// price tick; each level is a FIFO [`VecDeque`] (earliest resting order at the front).
pub struct OrderBook {
    bids: BTreeMap<i64, VecDeque<RestingOrder>>,
    asks: BTreeMap<i64, VecDeque<RestingOrder>>,
    tick_size: f64,
    next_order_id: u64,
}

impl OrderBook {
    /// An empty book with the given display `tick_size` (price = `price_tick * tick_size`).
    pub fn new(tick_size: f64) -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            tick_size,
            next_order_id: 0,
        }
    }

    /// The display tick size (price-per-tick); never used to key the book.
    pub fn tick_size(&self) -> f64 {
        self.tick_size
    }

    /// The id the next resting limit will receive — read it before [`Self::process_limit`]
    /// to learn the id a (partially or fully resting) order will carry for later
    /// cancel/modify. Every limit submission consumes exactly one id.
    pub fn next_order_id(&self) -> u64 {
        self.next_order_id
    }

    /// Best bid price tick (highest), if any.
    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    /// Best ask price tick (lowest), if any.
    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    /// Submit a limit order: cross the opposite side from the best price while the resting
    /// price is marketable (asks `<= price_tick` for a buy, bids `>= price_tick` for a
    /// sell), generating a [`Fill`] against each resting order touched (decrementing the
    /// FIFO head), then rest any remainder at `price_tick`. Consumes one order id.
    pub fn process_limit(
        &mut self,
        side: Side,
        price_tick: i64,
        qty: u64,
        agent: usize,
    ) -> Vec<Fill> {
        let id = self.next_order_id;
        self.next_order_id += 1;
        let (fills, remaining) = self.match_against(side, Some(price_tick), qty, agent);
        if remaining > 0 {
            let level = match side {
                Side::Buy => self.bids.entry(price_tick).or_default(),
                Side::Sell => self.asks.entry(price_tick).or_default(),
            };
            level.push_back(RestingOrder {
                id,
                agent,
                qty: remaining,
            });
        }
        fills
    }

    /// Submit a market order: cross the opposite side from the best price until filled or
    /// the book is empty. Never rests; an empty opposite side is a no-op (empty `Vec`).
    pub fn process_market(&mut self, side: Side, qty: u64, agent: usize) -> Vec<Fill> {
        self.match_against(side, None, qty, agent).0
    }

    /// Walk the opposite side from the best price, filling FIFO. `limit` is `Some(tick)`
    /// for a limit (stop when the resting price stops crossing) or `None` for a market
    /// (cross unconditionally). Returns the fills and the unfilled remainder.
    fn match_against(
        &mut self,
        side: Side,
        limit: Option<i64>,
        qty: u64,
        agent: usize,
    ) -> (Vec<Fill>, u64) {
        let mut remaining = qty;
        let mut fills = Vec::new();
        while remaining > 0 {
            let best = match side {
                Side::Buy => self.asks.keys().next().copied(),
                Side::Sell => self.bids.keys().next_back().copied(),
            };
            let Some(price) = best else { break };
            if let Some(lim) = limit {
                let crosses = match side {
                    Side::Buy => price <= lim,
                    Side::Sell => price >= lim,
                };
                if !crosses {
                    break;
                }
            }
            let book = match side {
                Side::Buy => &mut self.asks,
                Side::Sell => &mut self.bids,
            };
            let level = book.get_mut(&price).unwrap();
            while remaining > 0 {
                let Some(head) = level.front_mut() else { break };
                let traded = remaining.min(head.qty);
                let maker_id = head.id;
                let maker_agent = head.agent;
                head.qty -= traded;
                remaining -= traded;
                if head.qty == 0 {
                    level.pop_front();
                }
                fills.push(Fill {
                    price_tick: price,
                    qty: traded,
                    maker_id,
                    maker_agent,
                    taker_agent: agent,
                    taker_side: side,
                });
            }
            if level.is_empty() {
                book.remove(&price);
            }
        }
        (fills, remaining)
    }

    /// Cancel the resting order `id`, freeing its size and preserving the queue position of
    /// every other order at its level. Returns whether an order was found and removed.
    pub fn cancel_order(&mut self, id: u64) -> bool {
        let Some((side, price)) = self.locate(id) else {
            return false;
        };
        let book = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        let level = book.get_mut(&price).unwrap();
        let pos = level.iter().position(|o| o.id == id).unwrap();
        level.remove(pos);
        if level.is_empty() {
            book.remove(&price);
        }
        true
    }

    /// Resize the resting order `id`. A size **decrease** (or no change) keeps its queue
    /// position (the real-book rule — you only gave up size); an **increase** loses
    /// priority and re-queues at the back of the same price level. `new_qty == 0` cancels.
    /// Returns whether an order was found.
    pub fn modify_order(&mut self, id: u64, new_qty: u64) -> bool {
        if new_qty == 0 {
            return self.cancel_order(id);
        }
        let Some((side, price)) = self.locate(id) else {
            return false;
        };
        let book = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        let level = book.get_mut(&price).unwrap();
        let pos = level.iter().position(|o| o.id == id).unwrap();
        if new_qty <= level[pos].qty {
            level[pos].qty = new_qty;
        } else {
            let mut ord = level.remove(pos).unwrap();
            ord.qty = new_qty;
            level.push_back(ord);
        }
        true
    }

    /// Find the `(side, price_tick)` of resting order `id`, or `None` if absent.
    fn locate(&self, id: u64) -> Option<(Side, i64)> {
        for (&price, level) in &self.bids {
            if level.iter().any(|o| o.id == id) {
                return Some((Side::Buy, price));
            }
        }
        for (&price, level) in &self.asks {
            if level.iter().any(|o| o.id == id) {
                return Some((Side::Sell, price));
            }
        }
        None
    }

    /// Process a whole bar's orders in canonical order — sorted by agent, then submission
    /// index — so a reordered (e.g. parallel-collected) batch yields the identical tape.
    pub fn step(&mut self, orders: &[(usize, OrderKind)]) -> Vec<Fill> {
        let mut idx: Vec<usize> = (0..orders.len()).collect();
        idx.sort_by_key(|&i| (orders[i].0, i));
        let mut fills = Vec::new();
        for i in idx {
            let (agent, kind) = orders[i];
            match kind {
                OrderKind::Limit {
                    side,
                    price_tick,
                    qty,
                } => fills.extend(self.process_limit(side, price_tick, qty, agent)),
                OrderKind::Market { side, qty } => {
                    fills.extend(self.process_market(side, qty, agent))
                }
                OrderKind::Cancel { id } => {
                    self.cancel_order(id);
                }
                OrderKind::Modify { id, new_qty } => {
                    self.modify_order(id, new_qty);
                }
            }
        }
        fills
    }

    /// Top-`levels` ladder + derived microstructure scalars (the book observation). Bids
    /// are highest-first, asks lowest-first; per-level qty sums the FIFO at that price.
    pub fn depth_ladder(&self, levels: usize) -> LadderSnapshot {
        let bids: Vec<[i64; 2]> = self
            .bids
            .iter()
            .rev()
            .take(levels)
            .map(|(&p, q)| [p, level_qty(q) as i64])
            .collect();
        let asks: Vec<[i64; 2]> = self
            .asks
            .iter()
            .take(levels)
            .map(|(&p, q)| [p, level_qty(q) as i64])
            .collect();

        let best_bid = self.bids.iter().next_back();
        let best_ask = self.asks.iter().next();
        let (mid, microprice, queue_imbalance) = match (best_bid, best_ask) {
            (Some((&bp, bq)), Some((&ap, aq))) => {
                let bqty = level_qty(bq) as f64;
                let aqty = level_qty(aq) as f64;
                let total = bqty + aqty;
                let mid = (bp + ap) as f64 / 2.0;
                let micro = (bp as f64 * aqty + ap as f64 * bqty) / total;
                let imb = (bqty - aqty) / total;
                (mid, micro, imb)
            }
            (Some((&bp, _)), None) => (bp as f64, bp as f64, 1.0),
            (None, Some((&ap, _))) => (ap as f64, ap as f64, -1.0),
            (None, None) => (0.0, 0.0, 0.0),
        };
        LadderSnapshot {
            bids,
            asks,
            mid,
            microprice,
            queue_imbalance,
        }
    }

    /// Single-price call-auction uncross: the batch open/close clearing the continuous book
    /// lacks. Folds both resting ladders to the price that maximizes matched volume —
    /// cumulative demand (bids at/above the candidate) vs cumulative supply (asks at/below),
    /// matched = the lesser. Ties break by minimum `|demand - supply|` imbalance, then lower
    /// tick. Read-only and integer-only, so it never touches the book or the fill tape.
    /// Returns `Some((clearing_tick, matched_qty))`, or `None` when nothing crosses (the book
    /// is uncrossed — `best_bid < best_ask` — so no price clears positive volume).
    pub fn uncross(&self) -> Option<(i64, u64)> {
        let mut candidates: Vec<i64> = self.bids.keys().chain(self.asks.keys()).copied().collect();
        candidates.sort_unstable();
        candidates.dedup();

        // (matched, imbalance, tick) — maximize matched, then minimize imbalance, then tick.
        let mut best: Option<(u64, u64, i64)> = None;
        for &p in &candidates {
            let demand: u64 = self.bids.range(p..).map(|(_, l)| level_qty(l)).sum();
            let supply: u64 = self.asks.range(..=p).map(|(_, l)| level_qty(l)).sum();
            let matched = demand.min(supply);
            if matched == 0 {
                continue;
            }
            let imbalance = demand.abs_diff(supply);
            let better = match best {
                None => true,
                Some((bm, bi, bt)) => {
                    matched > bm
                        || (matched == bm && imbalance < bi)
                        || (matched == bm && imbalance == bi && p < bt)
                }
            };
            if better {
                best = Some((matched, imbalance, p));
            }
        }
        best.map(|(matched, _, tick)| (tick, matched))
    }

    /// Read-only walk-the-book cost of filling `qty` on `side` against the opposite ladder
    /// (a buy consumes asks lowest-first, a sell consumes bids highest-first), without
    /// mutating the book. Accumulates notional as exact integer `price_tick * qty` and
    /// reports the size-weighted average, slippage from the near touch, and the size the
    /// visible depth could fill. An empty or exhausted book yields a partial (or zero) fill.
    pub fn sweep_cost(&self, side: Side, qty: u64) -> SweepCost {
        let mut remaining = qty;
        let mut filled: u64 = 0;
        let mut notional: i128 = 0;
        let best = match side {
            Side::Buy => self.best_ask(),
            Side::Sell => self.best_bid(),
        };
        let mut consume = |price: i64, level: &VecDeque<RestingOrder>| {
            if remaining == 0 {
                return;
            }
            let take = remaining.min(level_qty(level));
            notional += price as i128 * take as i128;
            filled += take;
            remaining -= take;
        };
        match side {
            Side::Buy => {
                for (&price, level) in self.asks.iter() {
                    consume(price, level);
                }
            }
            Side::Sell => {
                for (&price, level) in self.bids.iter().rev() {
                    consume(price, level);
                }
            }
        }
        let avg_px_tick = if filled > 0 {
            notional as f64 / filled as f64
        } else {
            0.0
        };
        let slippage_ticks = match best {
            Some(b) if filled > 0 => (avg_px_tick - b as f64).abs(),
            _ => 0.0,
        };
        SweepCost {
            avg_px_tick,
            slippage_ticks,
            filled_qty: filled,
        }
    }
}

/// Total resting size at one price level (sum of the FIFO's order qtys).
fn level_qty(level: &VecDeque<RestingOrder>) -> u64 {
    level.iter().map(|o| o.qty).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dependency-free FNV-1a/64 — the same fingerprint [`crate::scenario_gen`] uses to pin
    /// cross-runtime serialization determinism without adding a hash crate.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Golden fingerprint of a fixed scripted order sequence's resulting fill tape. The tape
    /// is all integers, so this value must reproduce on any runtime (the wasm/python crates
    /// can assert the same number).
    const GOLDEN_TAPE_FNV1A: u64 = 0x8bbc_a7c3_2cea_d625;

    #[test]
    fn time_priority_earlier_order_fills_first() {
        let mut b = OrderBook::new(1.0);
        let a_id = b.next_order_id();
        b.process_limit(Side::Sell, 100, 5, 0);
        let _b_id = b.next_order_id();
        b.process_limit(Side::Sell, 100, 5, 1);
        // A buy that takes 1 share hits the earliest resting ask (agent 0).
        let fills = b.process_limit(Side::Buy, 100, 1, 2);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].maker_id, a_id);
        assert_eq!(fills[0].maker_agent, 0);
    }

    #[test]
    fn partial_fill_decrements_the_head() {
        let mut b = OrderBook::new(1.0);
        let a_id = b.next_order_id();
        b.process_limit(Side::Sell, 100, 10, 0);
        let fills = b.process_market(Side::Buy, 4, 1);
        assert_eq!(fills, vec![single(100, 4, a_id, 0, 1, Side::Buy)]);
        // 6 left at the head; the next taker keeps hitting the same maker.
        let more = b.process_market(Side::Buy, 6, 2);
        assert_eq!(more, vec![single(100, 6, a_id, 0, 2, Side::Buy)]);
        assert!(b.best_ask().is_none());
    }

    #[test]
    fn crossing_limit_matches_then_rests_remainder() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Sell, 100, 3, 0);
        // Buy 8 @ 100: 3 cross the resting ask, 5 rest as the new best bid.
        let rest_id = b.next_order_id();
        let fills = b.process_limit(Side::Buy, 100, 8, 1);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].qty, 3);
        assert_eq!(b.best_bid(), Some(100));
        assert!(b.best_ask().is_none());
        let ladder = b.depth_ladder(1);
        assert_eq!(ladder.bids, vec![[100, 5]]);
        // The resting remainder carries the id we previewed.
        assert!(b.cancel_order(rest_id));
        assert!(b.best_bid().is_none());
    }

    #[test]
    fn market_order_walks_multiple_levels() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Sell, 100, 2, 0);
        b.process_limit(Side::Sell, 101, 2, 1);
        b.process_limit(Side::Sell, 102, 2, 2);
        let fills = b.process_market(Side::Buy, 5, 9);
        let prices: Vec<i64> = fills.iter().map(|f| f.price_tick).collect();
        let qtys: Vec<u64> = fills.iter().map(|f| f.qty).collect();
        assert_eq!(prices, vec![100, 101, 102]);
        assert_eq!(qtys, vec![2, 2, 1]);
        // One share left at 102.
        assert_eq!(b.depth_ladder(3).asks, vec![[102, 1]]);
    }

    #[test]
    fn cancel_removes_and_frees_the_level() {
        let mut b = OrderBook::new(1.0);
        let id = b.next_order_id();
        b.process_limit(Side::Buy, 99, 7, 0);
        assert!(b.cancel_order(id));
        assert!(b.best_bid().is_none());
        // A market sell now finds nothing to hit.
        assert!(b.process_market(Side::Sell, 1, 1).is_empty());
        // Cancelling a stale id is a no-op.
        assert!(!b.cancel_order(id));
    }

    #[test]
    fn modify_decrease_keeps_increase_loses_priority() {
        let mut b = OrderBook::new(1.0);
        let a_id = b.next_order_id();
        b.process_limit(Side::Buy, 100, 10, 0);
        let b_id = b.next_order_id();
        b.process_limit(Side::Buy, 100, 10, 1);
        // Decrease A: it keeps the front of the queue.
        assert!(b.modify_order(a_id, 5));
        assert_eq!(b.process_market(Side::Sell, 1, 9)[0].maker_id, a_id);
        // Increase A: it loses priority and re-queues behind B.
        assert!(b.modify_order(a_id, 8));
        assert_eq!(b.process_market(Side::Sell, 1, 9)[0].maker_id, b_id);
    }

    #[test]
    fn step_is_canonical_order_deterministic() {
        let seed_book = || {
            let mut b = OrderBook::new(1.0);
            b.process_limit(Side::Sell, 105, 100, 7);
            b.process_limit(Side::Buy, 95, 100, 7);
            b
        };
        let batch: Vec<(usize, OrderKind)> = vec![
            (
                0,
                OrderKind::Limit {
                    side: Side::Buy,
                    price_tick: 105,
                    qty: 5,
                },
            ),
            (
                1,
                OrderKind::Limit {
                    side: Side::Sell,
                    price_tick: 95,
                    qty: 7,
                },
            ),
            (
                2,
                OrderKind::Limit {
                    side: Side::Buy,
                    price_tick: 106,
                    qty: 3,
                },
            ),
        ];
        let mut reversed = batch.clone();
        reversed.reverse();

        let mut forward = seed_book();
        let mut backward = seed_book();
        let f1 = forward.step(&batch);
        let f2 = backward.step(&reversed);
        assert_eq!(
            serde_json::to_string(&f1).unwrap(),
            serde_json::to_string(&f2).unwrap(),
            "a re-sorted batch must yield the identical tape"
        );
        assert!(!f1.is_empty());
    }

    #[test]
    fn empty_book_market_order_is_a_noop() {
        let mut b = OrderBook::new(1.0);
        assert!(b.process_market(Side::Buy, 10, 0).is_empty());
        assert!(b.process_market(Side::Sell, 10, 0).is_empty());
        let ladder = b.depth_ladder(5);
        assert!(ladder.bids.is_empty() && ladder.asks.is_empty());
        assert_eq!(ladder.mid, 0.0);
        assert_eq!(ladder.microprice, 0.0);
        assert_eq!(ladder.queue_imbalance, 0.0);
    }

    #[test]
    fn ladder_derives_mid_microprice_and_imbalance() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Buy, 100, 6, 0);
        b.process_limit(Side::Sell, 102, 2, 1);
        let l = b.depth_ladder(1);
        assert_eq!(l.bids, vec![[100, 6]]);
        assert_eq!(l.asks, vec![[102, 2]]);
        assert_eq!(l.mid, 101.0);
        // Size-weighted toward the larger (bid) side, i.e. above the mid.
        assert_eq!(l.microprice, (100.0 * 2.0 + 102.0 * 6.0) / 8.0);
        assert_eq!(l.queue_imbalance, (6.0 - 2.0) / 8.0);
    }

    /// A single-fill convenience for the assertions above.
    fn single(
        price_tick: i64,
        qty: u64,
        maker_id: u64,
        maker_agent: usize,
        taker_agent: usize,
        taker_side: Side,
    ) -> Fill {
        Fill {
            price_tick,
            qty,
            maker_id,
            maker_agent,
            taker_agent,
            taker_side,
        }
    }

    /// A fixed scripted sequence exercising rest / cross / market / cancel / modify, whose
    /// resulting tape pins the golden hash.
    fn scripted_tape() -> Vec<Fill> {
        let mut b = OrderBook::new(0.01);
        let mut tape = Vec::new();
        let batch: Vec<(usize, OrderKind)> = vec![
            (
                2,
                OrderKind::Limit {
                    side: Side::Sell,
                    price_tick: 102,
                    qty: 4,
                },
            ),
            (
                0,
                OrderKind::Limit {
                    side: Side::Sell,
                    price_tick: 101,
                    qty: 5,
                },
            ),
            (
                1,
                OrderKind::Limit {
                    side: Side::Sell,
                    price_tick: 101,
                    qty: 3,
                },
            ),
            (
                3,
                OrderKind::Limit {
                    side: Side::Buy,
                    price_tick: 100,
                    qty: 6,
                },
            ),
        ];
        tape.extend(b.step(&batch));
        let modify_id = b.next_order_id();
        b.process_limit(Side::Buy, 100, 4, 5);
        b.modify_order(modify_id, 2);
        tape.extend(b.process_limit(Side::Buy, 102, 10, 6));
        tape.extend(b.process_market(Side::Sell, 7, 7));
        tape
    }

    #[test]
    fn golden_tape_hash_is_stable() {
        let json = serde_json::to_string(&scripted_tape()).unwrap();
        assert_eq!(fnv1a(json.as_bytes()), GOLDEN_TAPE_FNV1A);
    }

    #[test]
    fn scripted_tape_is_reproducible() {
        assert_eq!(scripted_tape(), scripted_tape());
    }

    /// Rest an order directly into the book (bypassing matching) so a *crossed* book can be
    /// constructed for the call-auction uncross — the continuous API never rests a cross.
    fn rest(book: &mut OrderBook, side: Side, price: i64, qty: u64, agent: usize) {
        let id = book.next_order_id;
        book.next_order_id += 1;
        let level = match side {
            Side::Buy => book.bids.entry(price).or_default(),
            Side::Sell => book.asks.entry(price).or_default(),
        };
        level.push_back(RestingOrder { id, agent, qty });
    }

    #[test]
    fn uncross_picks_the_volume_maximizing_tick() {
        let mut b = OrderBook::new(1.0);
        for (p, q) in [(102, 5), (101, 5), (100, 5)] {
            rest(&mut b, Side::Buy, p, q, 0);
        }
        for (p, q) in [(100, 5), (101, 5), (102, 5)] {
            rest(&mut b, Side::Sell, p, q, 1);
        }
        // demand>=p vs supply<=p: 5/15/10 at 100/.., peaks at 101 (10 vs 10 -> matched 10).
        assert_eq!(b.uncross(), Some((101, 10)));
    }

    #[test]
    fn uncross_returns_none_on_an_uncrossed_book() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Buy, 99, 10, 0);
        b.process_limit(Side::Sell, 101, 10, 1);
        assert_eq!(b.uncross(), None);
    }

    #[test]
    fn uncross_tie_breaks_by_lower_tick_when_imbalance_is_equal() {
        let mut b = OrderBook::new(1.0);
        rest(&mut b, Side::Buy, 101, 10, 0);
        rest(&mut b, Side::Buy, 100, 10, 0);
        rest(&mut b, Side::Sell, 100, 10, 1);
        rest(&mut b, Side::Sell, 101, 10, 1);
        // p=100: 20 vs 10 (imb 10); p=101: 10 vs 20 (imb 10). Both match 10 -> lower tick.
        assert_eq!(b.uncross(), Some((100, 10)));
    }

    #[test]
    fn uncross_tie_breaks_by_min_imbalance_over_tick() {
        let mut b = OrderBook::new(1.0);
        rest(&mut b, Side::Buy, 102, 10, 0);
        rest(&mut b, Side::Buy, 100, 15, 0);
        rest(&mut b, Side::Sell, 100, 10, 1);
        rest(&mut b, Side::Sell, 102, 5, 1);
        // p=100: 25 vs 10 -> match 10, imb 15. p=102: 10 vs 15 -> match 10, imb 5.
        // Equal match; min imbalance wins the higher tick over the lower one.
        assert_eq!(b.uncross(), Some((102, 10)));
    }

    #[test]
    fn sweep_cost_walks_asks_and_leaves_the_book_unchanged() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Sell, 100, 2, 0);
        b.process_limit(Side::Sell, 101, 2, 1);
        b.process_limit(Side::Sell, 102, 2, 2);
        let before = b.depth_ladder(3);
        // Buy 5: 2@100 + 2@101 + 1@102 = 504 ticks / 5 = 100.8 avg, best 100 -> 0.8 slippage.
        let c = b.sweep_cost(Side::Buy, 5);
        assert_eq!(c.filled_qty, 5);
        assert_eq!(c.avg_px_tick, 504.0 / 5.0);
        assert_eq!(c.slippage_ticks, (504.0_f64 / 5.0 - 100.0).abs());
        // Read-only: the book (and its tape-relevant state) is byte-identical afterward.
        assert_eq!(b.depth_ladder(3), before);
        assert_eq!(b.best_ask(), Some(100));
    }

    #[test]
    fn sweep_cost_walks_bids_for_a_sell() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Buy, 100, 2, 0);
        b.process_limit(Side::Buy, 99, 2, 1);
        b.process_limit(Side::Buy, 98, 2, 2);
        // Sell 5: 2@100 + 2@99 + 1@98 = 496 / 5 = 99.2 avg, best 100 -> 0.8 slippage.
        let c = b.sweep_cost(Side::Sell, 5);
        assert_eq!(c.filled_qty, 5);
        assert_eq!(c.avg_px_tick, 496.0 / 5.0);
        assert_eq!(c.slippage_ticks, (100.0_f64 - 496.0 / 5.0).abs());
        assert_eq!(b.best_bid(), Some(100));
    }

    #[test]
    fn sweep_cost_partial_fills_when_depth_runs_dry() {
        let mut b = OrderBook::new(1.0);
        b.process_limit(Side::Sell, 101, 6, 0);
        let c = b.sweep_cost(Side::Buy, 100);
        assert_eq!(c.filled_qty, 6);
        assert_eq!(c.avg_px_tick, 101.0);
        assert_eq!(c.slippage_ticks, 0.0);
    }

    #[test]
    fn sweep_cost_on_an_empty_book_fills_nothing() {
        let b = OrderBook::new(1.0);
        let c = b.sweep_cost(Side::Buy, 10);
        assert_eq!(c.filled_qty, 0);
        assert_eq!(c.avg_px_tick, 0.0);
        assert_eq!(c.slippage_ticks, 0.0);
    }
}
