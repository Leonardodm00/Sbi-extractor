# Handoff: parameters that are DRAWN but never READ

**Date:** 2026-08-21
**Target repo for the changes described here:** `Astro-Neuron-Network`
(`HPC_main_sweep.py`, `submit_sweep_mixed.sh`, `launch_campaign.sh`)
**Repo already changed in response:** `Sbi-extractor` (branch
`feat/real-arm-parity`) -- see section 6.

## Abstract

A campaign sweep writes a per-simulation record of the parameters it sampled.
That record is currently a log of what the sweep **drew**, not of what the
simulator **read**. Under `--conn_rule weibull` the outer loop still draws
`conn_prob ~ U(0.1, 0.6)` per topology and stores it in every `iter_*.npz`,
but the connectivity is built from the Weibull kernel and the resulting edge
list is passed to the network builder explicitly, so the branch that consumes
`conn_prob` never executes. Downstream, the SBI export read that column as a
component of the inference target $\theta$, where a causally inert coordinate
is at best wasted flow capacity and at worst a silent corruption of every
posterior diagnostic. This document records the evidence, states the general
principle, and specifies what to change in the simulator's launch and
recording scripts so that the stored record distinguishes *sampled*,
*consumed*, and *fixed* parameters. It covers the `conn_prob` case in full and
the general mechanism; it does **not** re-derive the SBI export pipeline, and
it does **not** address the separate `CONN_PERIODIC` discrepancy beyond
flagging it in section 5.

## 1. Notation and symbols

| Symbol | Name / meaning | Type & domain | Units | First used in Sec.  |
|---|---|---|---|---|
| $\theta$ | inference target (label vector) for one simulation | $\theta \in \Theta \subseteq \mathbb{R}^{p}$ | mixed (per axis) | section 2 |
| $\theta_j$ | the $j$-th coordinate of $\theta$ | $\theta_j \in \mathbb{R}$, $j \in \{1,\dots,p\}$ | per axis | section 2 |
| $\theta_{\rm topo}$ | the topology-level sub-vector of $\theta$, drawn once per topology | $\theta_{\rm topo} \in \mathbb{R}^{p_{\rm topo}}$ | mixed | section 4 |
| $p$ | dimension of $\theta$ | $p \in \mathbb{N}$ | dimensionless | section 2 |
| $p_{\rm topo}$ | number of topology-level axes retained in $\theta$ | $p_{\rm topo} \in \mathbb{N}$ | dimensionless | section 4 |
| $x$ | observable (pooled IFR window) for one simulation | $x \in \mathbb{R}^{W}$ | Hz | section 2 |
| $W$ | window length in samples | $W \in \mathbb{N}$ | samples | section 2 |
| $p(x \mid \theta)$ | simulator likelihood, implicit (sampling only) | conditional density on $\mathbb{R}^{W}$ for each fixed $\theta$ | dimensionless density | section 2 |
| $p(\theta)$ | prior over the sampled parameters | density on $\Theta$ | dimensionless density | section 2 |
| $p(\theta \mid x)$ | posterior | conditional density on $\Theta$ for each fixed $x$ | dimensionless density | section 2 |
| $q_\phi(\theta \mid x)$ | learned posterior approximation (normalizing flow) | conditional density on $\Theta$ for each fixed $x$, $\phi$ the network weights | dimensionless density | section 2 |
| $\phi$ | flow parameters (network weights) | $\phi \in \mathbb{R}^{n_\phi}$ | dimensionless | section 2 |
| $d$ | inter-neuron distance | $d \in [0,\infty)$ | um | section 3 |
| $p_{\rm conn}(d)$ | distance-dependent connection probability (Weibull kernel) | $p_{\rm conn}: [0,\infty) \to [0,1]$ | dimensionless | section 3 |
| $p_0$ (`p0_conn`) | kernel amplitude, $p_{\rm conn}(0)$ | $p_0 \in (0,1]$ | dimensionless | section 3 |
| $d_0$ (`d0_conn`) | kernel characteristic length | $d_0 \in (0,\infty)$ | um | section 3 |
| $\beta$ (`beta_conn`) | kernel stretch exponent | $\beta \in (0,\infty)$ | dimensionless | section 3 |
| `conn_prob` | flat (distance-independent) connection probability | $\in [0,1]$ | dimensionless | section 3 |
| $\delta(\cdot)$ | Dirac delta | generalised function on $\mathbb{R}$ | inverse of its argument's units | section 2 |
| $N_{\rm topo}$ | number of independent topology draws in an export | $N_{\rm topo} \in \mathbb{N}$ | dimensionless | section 4 |
| $n_{\rm sim}$ | number of simulated rows in an export | $n_{\rm sim} \in \mathbb{N}$ | dimensionless | section 4 |
| $\mathcal{U}(a,b)$ | uniform distribution on $[a,b]$ | distribution on $\mathbb{R}$ | -- | section 3 |

### Conventions

Axis names in `typewriter` are the literal keys as they appear in
`iter_*.npz` and in `get_Synparam`'s dictionary; the same quantity in
mathematical notation uses the corresponding Greek/Latin symbol
(`p0_conn` $\equiv p_0$). "Drawn" always means sampled by the sweep's
`master_rng`; "read" always means consumed by code that affects the state of
the Brian2 network. Distances are in micrometres throughout, matching the
simulator. Conditioning is written out in full: $p(\theta_j \mid x)$, never
$p(\theta_j)$, unless the independence is the point being made.

## 2. Glossary

Ordered by first appearance, because the concepts build on each other.

- **Simulation-based inference (SBI).** Bayesian inference where $p(x \mid \theta)$
  can be sampled but not evaluated. The simulator is used as a black-box
  sampler and a neural network learns the mapping from observable to
  posterior. Operative from section 2 onward.
- **Neural posterior estimation (NPE).** The SBI variant used here: a
  conditional density estimator $q_\phi(\theta \mid x)$ is trained on pairs
  $(\theta^{(i)}, x^{(i)})$ so that it approximates $p(\theta \mid x)$ for
  each fixed $x$. section 2.
- **Normalizing flow.** The density estimator behind $q_\phi(\theta \mid x)$:
  an invertible map from a simple base distribution to the target, whose
  density follows by the change-of-variables formula. Requires the target to
  have a density with respect to Lebesgue measure -- the reason a delta prior
  is fatal. section 2.
- **Causally inert parameter.** *Term of art introduced in this document.* A
  parameter that is sampled and recorded but never influences the simulator
  state, so $p(x \mid \theta)$ does not depend on it. Note the everyday
  reading of "unused parameter" suggests it is *constant*; an inert parameter
  is typically **not** constant, which is exactly why it evades detection. section 2.
- **Delta prior / degenerate axis.** An axis on which the prior is
  $p(\theta_j) = \delta(\theta_j - c)$, i.e. the value never varied. section 2.
- **Prior predictive.** The distribution of $x$ obtained by drawing
  $\theta \sim p(\theta)$ and then $x \sim p(x \mid \theta)$. What the
  misspecification gate compares against real recordings. section 4.
- **Simulation-based calibration (SBC).** A diagnostic that checks whether
  posterior ranks are uniform under the prior. Passes trivially on an inert
  axis, which is why inertness silently degrades diagnostics rather than
  breaking them. section 2.
- **Weibull / stretched-exponential connection kernel.** The distance-dependent
  connectivity rule $p_{\rm conn}(d) = p_0 \exp[-(d/d_0)^{\beta}]$, selected by
  `--conn_rule weibull`. section 3.
- **`active_indices`.** The list of registry indices the sweep actually varied,
  written into `manifest.json` by the sweep itself. The export follows it
  rather than re-deriving the sweep group. section 6.

## 3. The specific case: `conn_prob` under `--conn_rule weibull`

**What this section establishes:** that `conn_prob` is drawn, recorded, and
never read, by naming the exact branch where the causal path terminates.

The chain, verified by reading the sources (not inferred):

1. `launch_campaign.sh` sets `CONN_RULE="weibull"` and passes it to every
   worker via `qsub -v`.
2. `submit_sweep_mixed.sh` passes `--conn_prob_lo 0.1 --conn_prob_hi 0.6` to
   `HPC_main_sweep.py` **unconditionally**, i.e. regardless of `CONN_RULE`.
3. `HPC_main_sweep.py` (outer topology loop) draws
   `conn_prob = master_rng.uniform(conn_prob_lo, conn_prob_hi)`, i.e.
   $\texttt{conn\_prob} \sim \mathcal{U}(0.1, 0.6)$, once per topology, and
   writes it into every `iter_*.npz` for that topology.
4. The same file passes both `conn_prob_=conn_prob` **and**
   `connections=[topo['S_i'], topo['S_j']]` into `Neuronal_Network(...)`.
5. In `ASD_fun_BD_cpp.py`, `Neuronal_Network` branches on the *type* of
   `connections`:
   - `connections is True` -> `S.connect(p=params_Syn['conn_prob'], condition='i != j')`
     -- the **only** read of that value anywhere in the module;
   - `isinstance(connections, list)` -> `S.connect(i=Source, j=Target)`.
   The sweep always supplies a list, so the second branch runs and the first
   never does.
6. `conn_prob_` does reach `get_Synparam(synapse_type=..., conn_prob=conn_prob_)`,
   but there it only overwrites the dictionary entry `'conn_prob': 0.107`.
   No weight normalisation, no in-degree scaling, no second consumer.

Therefore, for every campaign run with `--conn_rule weibull`,
$p(x \mid \theta)$ is independent of `conn_prob`.

**Measured, on the eight campaigns `campaign_cadex_rho1300v{1,2,3,4,5,7,8,9}`:**
546 distinct values of `conn_prob` across 546 sampled topologies, spanning
approximately $[0.1005, 0.6]$ -- consistent with $\mathcal{U}(0.1,0.6)$ -- and
`conn_rule = weibull`, `conn_periodic = False` uniformly across all eight.
`conn_prob`, `p0_conn`, `d0_conn`, `beta_conn` share identical per-campaign
distinct-counts (229, 229, 383, 95, 237, 26, 161, 153), confirming all four are
drawn together, once per topology.

**Why a variance scan cannot find this.** The column varies perfectly well;
it is uninformative about $x$, not constant. Only an audit of the code path
distinguishes the two cases. Plainly: the log records that a dial was turned,
but the dial was not connected to anything.

## 4. Why it matters for inference

**What this section establishes:** the two distinct failure modes, and why the
inert one is the more dangerous of the pair.

**Constant axis (delta prior).** If $\theta_j = c$ for all rows, then
$p(\theta_j) = \delta(\theta_j - c)$, and for each fixed $x$ the true posterior
marginal is that same delta. It has no density with respect to Lebesgue
measure, so a flow can only approach it by driving its log-density to
$+\infty$ along that coordinate; the Jacobian term degenerates and training
destabilises. This is loud: it shows up as an assertion failure or a diverging
loss.

**Causally inert axis.** If $p(x \mid \theta)$ does not depend on $\theta_j$,
then by Bayes' theorem, for each fixed $x$,
$$p(\theta_j \mid x) = p(\theta_j) \qquad \text{(1)}$$
i.e. the posterior equals the prior on that axis. Nothing crashes -- equation
(1) *is* the correct answer, and a well-trained flow will learn it. The damage
is silent:

- flow capacity is spent modelling a coordinate that carries no information;
- SBC passes trivially on that axis, so a calibration check that "passes" is
  partly measuring nothing;
- posterior contraction is zero by construction on that axis, which will look
  like a modelling failure to anyone reading the diagnostics later;
- `minimum_detectable_shift` along that axis is meaningless.

**A structural note on effective sample size.** $\theta_{\rm topo}$ is drawn
once per topology, so rows within a topology share it exactly. With
$N_{\rm topo} = 546$ and $n_{\rm sim} \approx 3 \times 10^{5}$, the independent
sample size along the topology axes is $N_{\rm topo}$, not $n_{\rm sim}$ --
roughly 560 repeats per draw. Any power calculation that counts rows (the MMD
gate's $n_{\rm sim} \geq 4 n_{\rm real}$ floor included) overstates the
evidence available about those axes by nearly three orders of magnitude. This
is a property of the sweep design, not a bug, but it must be stated wherever
those axes are interpreted.

## 5. What to change in `Astro-Neuron-Network`

**What this section establishes:** the concrete edits, in priority order.

The governing principle: **record what was consumed, not merely what was
sampled**, and where the two differ, say so in the record itself. Deleting the
value is *not* sufficient on its own -- a silently absent key is
indistinguishable from a key that a future reader forgot to write.

### 5.1 Do not draw an inert parameter (preferred)

In `HPC_main_sweep.py`, make the `conn_prob` draw conditional on the
connectivity rule:

- under `--conn_rule flat`: draw and record as today (it is causally live);
- under `--conn_rule weibull`: do not draw it at all.

Consequence: the sweep's `master_rng` consumption changes, so **seeds will no
longer reproduce earlier campaigns bit-for-bit.** This is the single reason
the change cannot simply be back-applied to existing output; see section 5.4.

### 5.2 Record a consumed-parameter declaration

Whatever is drawn, write into `manifest.json` (and mirror into `job_args.json`)
an explicit block naming, per run:

- `swept_axes`: the axes drawn from a non-degenerate prior;
- `consumed_axes`: the subset the simulator actually reads under the active
  rule set;
- `fixed_axes`: name -> value for parameters deliberately held constant,
  with the value recorded (this is what lets a later reader distinguish
  "fixed at 0.107" from "absent by oversight");
- `inert_axes`: name -> reason, for anything drawn but not consumed, should
  section 5.1 not be applied.

`consumed_axes` is the field the SBI export should ultimately read instead of
inferring the axis set. It is cheap to write and it is the only place the
information exists at all: it is known at launch time and unrecoverable
afterwards without re-reading the simulator source.

### 5.3 Stop writing NaN-valued kernel keys under the flat rule

`p0_conn`/`d0_conn`/`beta_conn` are currently written as NaN when
`--conn_rule flat`. Prefer omitting them and declaring them in `fixed_axes`
or `inert_axes`; a NaN in a numeric column propagates silently through
anything that does not explicitly check for it.

### 5.4 Do not retro-fit existing campaigns

Existing `iter_*.npz` files keep their `conn_prob` entry. The exclusion is
handled on the consumer side (section 6), which is reversible and leaves the raw
record untouched. Rewriting the archive would destroy the evidence that the
draw happened, which is itself worth keeping.

### 5.5 Separately: the `CONN_PERIODIC` discrepancy

`launch_campaign.sh` sets `CONN_PERIODIC=0` directly beneath a comment stating
it is *"ON, per the decision to train on periodic boundaries (cleaner
plateaus, larger size-invariant region)"*. The measured `job_args.json` for all
eight campaigns confirms `conn_periodic = False`. Either the comment or the
value is wrong. This is constant across the campaign set, so it is not a
$\theta$ axis and does not affect the label spec -- but if the intended design
was minimum-image distances and the runs used finite boundaries, that is a
simulator-versus-intent gap sitting inside precisely what the misspecification
gate is meant to detect. Resolve and record which was intended.

## 6. What has already been changed in `Sbi-extractor`

**What this section establishes:** the consumer-side handling, so the two
repos can be reconciled later.

- `preflight_label_axes.py` (new). Scans every campaign once, reports per-axis
  distinct counts and ranges, drops zero-variance axes **automatically**, and
  accepts `--exclude NAME=REASON` for varying-but-inert axes (a reason is
  mandatory). Refuses to write if `--require-conn-rule` is violated, if an
  excluded name never appears in the data, or if no axis would remain. Writes
  `artifacts/label_axes.json` plus a `.sha256`.
- `sbi_labels.py`. `TOPOLOGY_AXES` is no longer hardcoded into the spec:
  `build_label_spec` takes `topology_axes` and `excluded_axes`, and topology
  bounds are looked up **by name** rather than by position, so removing an
  axis cannot shift a column onto the wrong prior interval.
- `example_export.py`. Reads only the axes the frozen spec asks for; takes
  `--label_axes`; records `topology_axes`, `excluded_axes` and $p$ in every
  sidecar; and the JSON-null kernel-bounds bug is fixed at source
  (`job_args.get(k) is not None`, since `k in job_args` is true for a null).
- `export_embeddings.py`. Assertion A4 now separates **fatal** constancy (a
  run-args axis with no variation) from **structural** constancy (a
  topology-level axis in a shard covering one topology, which is expected and
  now warns instead of aborting). The population-level check belongs to the
  preflight, which is the only component that sees all campaigns at once.

Resulting label dimension for these eight campaigns: $p = 26$
(23 `active_indices` + 3 kernel axes), down from 27.

## 7. Summary of results

1. `conn_prob` is drawn per topology from $\mathcal{U}(0.1,0.6)$ and written to
   every `iter_*.npz`, but under `--conn_rule weibull` it is never read: the
   explicit edge list selects `S.connect(i=Source, j=Target)` and the branch
   `S.connect(p=params_Syn['conn_prob'], ...)` is dead. (section 3)
2. All eight campaigns used `conn_rule = weibull`, so the inertness holds
   uniformly across the export set; under `flat` it would **not**. (section 3)
3. A causally inert axis satisfies $p(\theta_j \mid x) = p(\theta_j)$ for each
   fixed $x$ -- equation (1), section 4 -- so it fails silently rather than loudly,
   unlike a constant axis.
4. A variance scan cannot detect inertness; only a code-path audit can. (section 3)
5. The independent sample size along topology axes is $N_{\rm topo} = 546$,
   not $n_{\rm sim} \approx 3\times10^5$. (section 4)
6. The simulator should record `swept`/`consumed`/`fixed`/`inert` axis
   declarations at launch time, since that information is unrecoverable from
   the output afterwards. (section 5.2)

## 8. Open points, caveats, and assumptions

- **Assumed without exhaustive proof:** that `get_Synparam`'s `conn_prob`
  entry has no consumer other than the single `S.connect(p=...)` call. This
  was checked by grepping every occurrence of `conn_prob` in
  `ASD_fun_BD_cpp.py` and `HPC_main_sweep.py`; a consumer reached by
  string-built attribute access or by a Brian2 equation referencing the name
  indirectly would not have been caught.
- **Not audited:** whether any of the 23 `active_indices` run-args axes is
  itself inert under `MODE=Neuronal`. The launcher comment states that
  `neuron_synapse` sweeps **24** axes, while `manifest.json` reports **23**
  `active_indices` -- an unexplained discrepancy of one, plausibly one of the
  "fixed for simplicity" parameters. This should be resolved the same way:
  by audit, not by variance scan.
- **Not addressed:** the `CONN_PERIODIC` comment-versus-value contradiction
  (section 5.5), beyond flagging it.
- **Regime of validity:** every claim about `conn_prob` inertness is
  conditional on `--conn_rule weibull` **and** on the sweep passing an
  explicit edge list. A campaign launched by any other route inherits
  `submit_sweep_mixed.sh`'s own default, `CONN_RULE="flat"`, under which
  `conn_prob` is causally live.
- **Consequence of section 5.1 left unresolved:** removing the draw shifts the RNG
  stream, so post-change campaigns are not seed-comparable with pre-change
  ones. Whether to accept that break, or to preserve the draw and discard the
  value, is a decision not yet taken.

## 9. References

All source-level claims in section 3 and section 5 come from reading these files directly at
the revisions checked out on 2026-08-21; nothing in this document is stated
from memory of the code:

- `Astro-Neuron-Network`: `hpc/Phenomenological_finalv1/HPC_main_sweep.py`,
  `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` (branch `main`).
- `Astro-Neuron-Network`: `submit_sweep_mixed.sh`, `launch_campaign.sh`
  (provided directly).
- `Sbi-extractor`: `sbi_labels.py`, `example_export.py`,
  `export_embeddings.py` (branch `feat/real-arm-parity`).
- Measurements in section 3 and section 4 are from scans run on the cluster over
  `campaign_cadex_rho1300v{1,2,3,4,5,7,8,9}` on 2026-08-21.

No external literature was consulted for this document; the inference-side
statements in section 4 are standard properties of Bayesian conditioning and
normalizing-flow density estimation, stated here as reasoning rather than
cited to a source.
