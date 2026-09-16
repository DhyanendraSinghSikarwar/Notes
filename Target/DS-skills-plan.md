# The ₹60 LPA Data Scientist: Country-Wise Skills Map & Learning Plan

**Compiled:** 16 September 2026 **Target band:** ₹60,00,000 INR ≈ **USD 68K** ≈ SGD 92K ≈ GBP 53K ≈ EUR 62K ≈ CHF 60K **Scope:** India, Singapore, Indonesia, Thailand, USA, UK, Germany, Netherlands, Switzerland, Nordics, Ireland **Domains covered:** Fintech · E-commerce & Marketplace · Healthcare

> **How to use this file.** Part I tells you *what each market and company actually asks for*. Part II is the full skill taxonomy with sub-topics. Part III is the domain overlay. Part IV is the execution plan — what to learn, where, how much per week, and what to build.
> ****A caution about the salary figures.** These come from salary aggregators and published job postings, not from verified offer data. Treat them as bands, not quotes. Levels.fyi and Glassdoor self-reported data skews high; local salary guides skew low. Where I read a posting in full, it is cited by name.

---

# PART I — COUNTRY-WISE & COMPANY-WISE REQUIREMENTS

## 0. The one-paragraph summary

₹60 LPA is a **different job in every market**. In India it is increasingly a *GenAI* job — LLMs, RAG, agents, LLMOps named explicitly in postings. In Singapore and Indonesia it is a *causal inference and production* job — uplift modelling, switchback tests, models you own in production. In the USA it is a *product analytics + experimentation* job at the ₹60 LPA-equivalent level, which there is merely mid-level. In Europe it is a *domain + regulation* job, where GDPR, model governance and industry specifics carry as much weight as modelling. Optimise for one and you are a mediocre fit for the others; the only skill that pays everywhere is **production ownership**.

---

## 1. INDIA — Bangalore, Hyderabad, Gurugram, Pune

### Band reality

| Level | Years | Typical CTC | Notes |
| --- | --- | --- | --- |
| Senior DS | 5–8 | ₹25–45 LPA | The crowded middle |
| **Senior DS at top-tier / Staff DS** | **7–12** | **₹45–70 LPA** | **Target band** |
| Principal / Lead IC | 10–14+ | ₹50–90 LPA | At enterprise firms can mean architecture, not modelling |
| Head of DS / Director | 12+ | ₹60 LPA – ₹1 Cr+ | People + strategy |

At and above ₹45 LPA, **30–50% of CTC is stock and variable**. A pure-cash offer at ₹60 LPA is rare outside a few funded startups paying above market to close.

### Companies that reach the band, and what each actually wants

#### Tier A — MNC Global Capability Centres (GCCs)

**Google, Microsoft, Amazon, Meta, Walmart Global Tech, Uber, Salesforce, Adobe, Atlassian, Intuit, LinkedIn, Oracle, Expedia, Target**

- **What they test:** the *global* bar, localised. Product sense, metric definition, experimentation design, SQL depth, and a coding round that is genuinely LeetCode-medium.
- **Distinctive:** Amazon runs its Leadership Principles bar-raiser regardless of function — behavioural answers must be STAR-formatted with real metrics. Google runs a separate "analytical reasoning" round.

**Skills weighted heaviest — in priority order**

**1. SQL** — *used for:* the screening round and the analytics round; you write live while they watch.

> Sub-topics: window functions (`ROW_NUMBER`, `LAG`/`LEAD`, `SUM() OVER`, frame clauses) · self-joins · CTEs · cohort and retention queries · funnel queries with step ordering · sessionisation · date arithmetic across time zones · `GROUPING SETS`/`ROLLUP` · anti-joins for "users who did X but not Y" · deduplication patterns · query-plan reasoning when they ask "make this faster"

**2. Experimentation & A/B testing at scale** — *used for:* the dedicated experimentation round; they will hand you a half-broken test and ask what is wrong with it.

> Sub-topics: hypothesis formulation · **minimum detectable effect and power calculation** · sample-size determination · randomisation unit choice (user vs session vs device) · **statistical vs practical significance** · **sample ratio mismatch (SRM)** · the peeking problem and sequential testing · multiple-comparison correction (Bonferroni, Benjamini-Hochberg) · **CUPED variance reduction** · novelty and primacy effects · guardrail metrics · network interference · long-run holdouts

**3. Statistics & inference** — *used for:* the stats round, and as a cross-examination inside every other round.

> Sub-topics: conditional probability and Bayes · common distributions and where each arises · CLT and its limits · confidence intervals (and what they do *not* mean) · Type I/II error and power · **hypothesis testing choice** (t-test, chi-square, Mann-Whitney, proportions z-test) · bootstrap and permutation tests · linear and logistic regression assumptions and diagnostics · multicollinearity · Simpson's paradox · survivorship and selection bias

**4. Product sense & metric definition** — *used for:* the product/case round; heavily weighted at Meta, Google, Uber, LinkedIn.

> Sub-topics: choosing the **OEC (overall evaluation criterion)** · north-star metric and its input tree · leading vs lagging indicators · gameable-metric detection · **diagnosing a metric drop** (segment → seasonality → instrumentation → external) · opportunity sizing before building · trade-off framing (engagement vs retention vs trust) · defining success for a feature that has never shipped

**5. Python** — *used for:* the coding round and the take-home.

> Sub-topics: pandas `groupby().transform()`, `merge` semantics, `MultiIndex`, window ops · NumPy vectorisation and broadcasting · scikit-learn `Pipeline` and `ColumnTransformer` · writing testable functions · complexity awareness · clean, readable code under time pressure

**6. DSA (the coding screen)** — *used for:* the elimination round. It is not optional even for analytics roles here.

> Sub-topics: hash maps · two pointers · sliding window · string manipulation · binary search · sorting with custom comparators · BFS/DFS on grids and graphs · heaps/top-K · basic DP (climbing stairs, coin change, LIS) · stating time and space complexity out loud

**7. One ML specialisation** — *used for:* the depth round; increasingly required even on product-DS ladders.

> Sub-topics (pick one lane): **ranking** — learning-to-rank, CTR prediction, NDCG · **forecasting** — hierarchical, seasonality, backtesting · **risk** — imbalanced learning, calibration, cost-sensitive thresholds · plus, for all lanes: leakage detection, cross-validation design, feature importance and SHAP, baseline-first discipline

**8. Behavioural & communication** — *used for:* the bar-raiser / hiring-manager round, which can sink an otherwise clean loop.

> Sub-topics: **STAR structure with a quantified outcome in every story** · Amazon's 16 Leadership Principles mapped to your own stories · conflict and disagreement narratives · a failure story with a real lesson · answer-first (top-down) structuring · 90-second answer discipline · data storytelling to a non-technical stakeholder

#### Tier B — Indian consumer tech & fintech

**Flipkart, Swiggy, Zomato, Meesho, PhonePe, Razorpay, CRED, Navi, Zepto, Rapido, ShareChat, Dream11, Groww, Zerodha**

- **What they test:** business impact under ambiguity, marketplace economics, and speed. Case rounds are common and heavily weighted.
- **Read in full — Swiggy, Staff Data Scientist, Bengaluru:** asks **8–12 years**. Staff titles here are genuinely senior.

**Skills weighted heaviest — in priority order**

**1. Marketplace & business case reasoning** — *used for:* the case round, which carries the most weight here and is often the differentiator between offer levels.

> Sub-topics: two-sided marketplace dynamics (which side is constrained) · unit economics — contribution margin, take rate, CAC, LTV, payback period · liquidity, fill rate, match rate · **cannibalisation across channels and categories** · supply-demand imbalance diagnosis · sizing a market or an intervention with Fermi reasoning · knowing when the answer is "do not build this"

**2. Pricing & incentives** — *used for:* the core problem at Swiggy, Zomato, Rapido, Zepto, Meesho.

> Sub-topics: **price elasticity estimation from observational data** (and the endogeneity problem) · surge and dynamic pricing · personalised pricing and its fairness limits · **discount and coupon incrementality** · **uplift modelling for voucher targeting** (S/T/X-learners, causal forests, Qini curves) · budget allocation across incentive levers · markdown optimisation · promotion cannibalisation

**3. Demand forecasting & operations** — *used for:* quick-commerce and delivery roles specifically.

> Sub-topics: hierarchical forecasting and reconciliation · intermittent demand (Croston) · new-SKU and dark-store cold start · promotional lift decomposition · seasonality and Indian festival calendars · **rider/driver supply forecasting** · ETA prediction · batching and last-mile routing (VRP) · inventory: safety stock, reorder points, newsvendor · **backtesting with rolling origin**

**4. Recommendation & ranking** — *used for:* Flipkart, Myntra, Meesho, ShareChat, Dream11 feed and catalogue teams.

> Sub-topics: retrieval → ranking → re-ranking architecture · **two-tower / dual-encoder retrieval** · ANN indexes (HNSW, IVF-PQ) · learning-to-rank (LambdaMART, pairwise vs listwise) · CTR/CVR models (Wide & Deep, DeepFM, DCN, DIN) · **position and popularity bias correction** · cold start for new sellers and SKUs · diversity and multi-objective ranking · **NDCG, MAP, MRR, Recall@K**

**5. Fraud & credit risk** *(fintech subset: PhonePe, Razorpay, CRED, Navi, Groww, Slice)* — *used for:* the domain round.

> Sub-topics: **WOE binning and Information Value** · logistic scorecards and why they beat GBDTs in regulated lending · PD/LGD/EAD and expected loss · **KS statistic, Gini, PSI** · reject inference · vintage and cohort loss curves · alternative data (CIBIL, bank-statement cashflow, UPI behaviour, device and telco signals) · **graph analytics for fraud rings** · velocity features · extreme class imbalance (0.01–0.5% positive rate) · sub-100ms real-time scoring · **RBI digital lending guidelines** and explainability obligations

**6. Causal inference** — *used for:* growth, retention and incentive teams; asked less formally than in Singapore but decisive when it comes up.

> Sub-topics: **difference-in-differences** and parallel trends · **geo-experiments and holdout design** · propensity score matching and IPTW · instrumental variables · regression discontinuity · **switchback tests** where treatment leaks across the marketplace · synthetic control for city-level rollouts · confounding by indication

**7. SQL & Python at speed** — *used for:* the screening round; the bar is lower than Tier A but the time limit is tighter.

> Sub-topics: same SQL sub-topics as Tier A, with extra weight on **cohort retention, funnel and sessionisation queries** · pandas fluency for a 2-hour take-home · fast, defensible EDA · chart choice that carries an argument

**8. Production ownership** — *used for:* the staff-level round; the single most common reason a senior candidate is levelled down.

> Sub-topics: model deployment as an API or scheduled job · **Airflow DAG design**, idempotency, backfills · monitoring and drift alerting · retraining cadence · on-call reality · A/B rollout and rollback of a model

#### Tier C — Enterprise & GenAI-titled roles

**SAP Labs, Brillio, Wipro, TCS Research, Fractal Analytics, Tiger Analytics, Mu Sigma, EXL, PhysicsWallah, Jio, Reliance**

- **Read in full — SAP Labs India, Principal Data Scientist / AI Application Development Expert, Bangalore:** requires **14+ years**, and names: RAG, AI agents & multi-agent systems, embeddings & semantic search, fine-tuning, prompt optimisation, **LLMOps and observability**; LangChain, LlamaIndex, Semantic Kernel, Hugging Face, OpenAI APIs; Pinecone, Weaviate, ChromaDB, FAISS; TensorFlow, PyTorch, scikit-learn; Spark/Databricks; Kubernetes, Docker, CI/CD for AI; enterprise & solution architecture; responsible AI and governance; executive-level communication.
- **Read in full — PhysicsWallah, Principal Data Scientist GenAI, Bangalore:** **6–10 years**; Generative AI, LLMs, Machine Learning, PyTorch, Python, TensorFlow.
- **Warning:** "Principal" at an enterprise firm often means *solution architect*. Check whether the role builds models or builds slideware.

**Skills weighted heaviest — in priority order**

**1. Retrieval-Augmented Generation (RAG)** — *used for:* the flagship use case at nearly every enterprise GenAI role — "chat with our documents".

> Sub-topics: chunking strategies (fixed, semantic, recursive, **parent-document/hierarchical**) · embedding model selection and benchmarking · **vector databases — Pinecone, Weaviate, ChromaDB, FAISS** (SAP names all four) · **hybrid search** (BM25 + dense) and reciprocal rank fusion · **cross-encoder rerankers** · query transformation (HyDE, multi-query, step-back) · context assembly and token budgeting · **citation grounding** and hallucination mitigation · **retrieval evaluation treated separately from generation evaluation**

**2. LLM evaluation** — *used for:* the round that separates people who have shipped from people who have demoed. Most candidates fail here.

> Sub-topics: **golden eval-set construction** · **LLM-as-judge** with rubric design, and its position bias and self-preference bias · **RAGAS-style metrics** — faithfulness, answer relevance, context precision, context recall · regression suites and eval-driven development · offline-to-online correlation · human review workflows · **red-teaming: prompt injection, jailbreaks, data exfiltration** · A/B testing an LLM feature in production

**3. Agents & orchestration** — *used for:* SAP explicitly names "AI agents/multi-agent systems"; this is the newest and least-supplied skill.

> Sub-topics: **tool/function calling** and tool-schema design · **ReAct**, plan-and-execute, reflection patterns · **multi-agent coordination** and handoff protocols · memory (short-term, long-term, episodic) · **Model Context Protocol (MCP)** · frameworks — **LangChain, LlamaIndex, Semantic Kernel** (all three named by SAP), LangGraph, DSPy · failure containment: timeouts, retries, **loop detection, cost ceilings** · tracing with Langfuse or Phoenix

**4. Fine-tuning & model adaptation** — *used for:* the "can you go below the API" test.

> Sub-topics: **PEFT — LoRA, QLoRA**, adapters, prefix tuning · full fine-tuning vs PEFT trade-offs · instruction dataset construction and cleaning · **synthetic data generation and its failure modes** · preference tuning (DPO, ORPO) · **prompt optimisation** (named by SAP) · quantisation for serving (INT8, 4-bit) · knowing **when not to fine-tune** — usually try retrieval and prompting first

**5. LLMOps & observability** — *used for:* named verbatim in the SAP posting; the enterprise differentiator over startup GenAI work.

> Sub-topics: prompt versioning and management · **caching — exact, semantic, prefix/KV** · serving with vLLM, TGI, SGLang; continuous batching, PagedAttention · **token accounting and cost-per-query observability** · latency budgets and streaming · guardrails: input/output filtering, **PII redaction**, policy enforcement · Langfuse, Phoenix, LangSmith · fallback and degradation paths when a provider fails

**6. Deep learning frameworks** — *used for:* PhysicsWallah names PyTorch *and* TensorFlow; enterprise roles often inherit legacy TF codebases.

> Sub-topics: **PyTorch** — `Dataset`/`DataLoader`, custom modules, hooks, `torch.compile`, debugging NaNs and shape bugs · **TensorFlow/Keras** — enough to read and maintain existing models · **transformers**: self-attention, multi-head attention, positional encodings · Hugging Face `transformers`, `peft`, `datasets` · mixed precision, gradient accumulation and checkpointing · inference optimisation: ONNX, TensorRT, distillation

**7. Enterprise & solution architecture** — *used for:* the round that makes this a "Principal" title rather than a senior one.

> Sub-topics: distributed systems and microservices · **Kubernetes** — deployments, services, HPA, resource limits · Docker multi-stage builds · **CI/CD for AI workflows** · cloud AI platforms — **AWS, Azure, GCP, SAP BTP** · integration with enterprise systems (SAP, Salesforce, ServiceNow) · multi-tenancy and data isolation · build-vs-buy reasoning · cost modelling for an AI programme

**8. Big data & the classical ML floor** — *used for:* the enterprise reality that most delivered value still comes from tabular models.

> Sub-topics: **Spark/Databricks** — partitioning, shuffle, skew, broadcast joins, AQE, UDF cost · Delta Lake and the lakehouse pattern · scikit-learn pipelines · **XGBoost/LightGBM** tuning and calibration · time-series forecasting for enterprise planning · feature stores and offline/online parity

**9. Responsible AI & governance** — *used for:* the compliance round, which is real at SAP, TCS, Wipro and any firm selling into Europe.

> Sub-topics: bias evaluation and fairness metrics · toxicity and safety benchmarks · **model cards and datasheets** · audit trails and lineage · **EU AI Act** high-risk obligations · **NIST AI RMF**, ISO 42001 · data residency and sovereignty · explainability for regulated decisions

**10. Executive communication** — *used for:* named explicitly by SAP; at Principal level this is assessed as hard as the technical rounds.

> Sub-topics: **answer-first structuring** · one-pager and architecture-decision-record writing · translating model uncertainty into business risk language · roadmap and investment cases with cost numbers · stakeholder management across business and IT · presenting to a steering committee · saying no with a documented reason

### India — the distinctive signal

**GenAI is the loudest ask in India and nowhere else at this band.** Both GenAI-heavy postings read in full were Bangalore. Reported premium for demonstrated RAG/fine-tuning experience: **20–40% over a generalist DS** at the same experience level.

### India — what will block you

- No production ownership. Notebook-only portfolios cap out around ₹30 LPA.
- Service-company background without a product-company stint — clearable, but expect a level-down on the first jump.
- Weak DSA. Product companies still run a coding round.

---

## 2. SINGAPORE

### Band reality

| Level | Typical TC (SGD) | INR equivalent |
| --- | --- | --- |
| Data Scientist | 85–120K | ₹55–78 LPA |
| **Senior DS** | **130–190K** | **₹85 LPA – ₹1.25 Cr** |
| Staff / Lead | 190–280K+ | ₹1.25–1.8 Cr |

**₹60 LPA-equivalent (SGD 92K) is entry-to-mid here, not senior.** If you are targeting Singapore, aim higher than your India number.

### Companies and what each wants

#### Grab

**Read in full — Senior Data Scientist, Ads & Demand Optimization, Singapore:**

- **Experience:** minimum **4 years** in ads ranking, recommendation systems, uplift modelling, or marketplace optimisation
- **Required methods:** causal inference, **uplift modelling**, experimental design, **ads auction theory**, mathematical optimisation
- **Required tools:** Python, SQL, **Spark**
- **Experimental frameworks named:** A/B testing, **switchback testing**, **bandit algorithms**
- **Metrics named:** CTR, CVR, incremental GMV
- **Soft:** "translate complex concepts and bring structural clarity to open-ended product problems"

Other Grab DS tracks hiring at this level: Dynamic Pricing, Search & Personalisation, GrabMaps, GrabX.

**Skills weighted heaviest — in priority order**

**1. Causal inference** — *used for:* the gatekeeping technical round. Named as a hard requirement, not a preference.

> Sub-topics: potential outcomes framework, SUTVA, ignorability · **DAGs, confounders, colliders, backdoor criterion** · difference-in-differences and parallel trends · **instrumental variables** and weak-instrument diagnosis · regression discontinuity (sharp and fuzzy) · propensity scores, IPTW, **doubly robust estimation** · **Double/Debiased ML** · sensitivity analysis for unobserved confounding · knowing when a randomised test is impossible and what you do instead

**2. Uplift / heterogeneous treatment effect modelling** — *used for:* the core modelling problem in ads and incentive allocation. Named verbatim in the posting.

> Sub-topics: **S-learner, T-learner, X-learner, R-learner** · **causal forests** and DR-learner · **Qini curves and uplift curves** for evaluation (not AUC) · policy learning — converting CATE estimates into a targeting rule · budget-constrained assignment · treatment-effect heterogeneity across segments · libraries: **EconML, CausalML, DoWhy**

**3. Experimental design** — *used for:* the round where they hand you a marketplace and ask how you would test in it.

> Sub-topics: randomisation unit choice under **interference** · **switchback testing** — why it exists, cycle-length selection, carryover (named in the posting) · **cluster randomisation** and geo-splits · **bandit algorithms** — epsilon-greedy, Thompson sampling, **contextual bandits** (named in the posting) · MDE and power under clustering · **CUPED** and stratification · guardrail metrics and OEC selection · SRM detection · interleaving for ranking tests

**4. Ads auction theory & mathematical optimisation** — *used for:* the ads-specific depth round. This is a genuinely scarce skill.

> Sub-topics: **second-price, GSP and VCG auctions** · reserve prices and floor optimisation · **bid optimisation and budget pacing** · incrementality-aware bidding · constrained optimisation (LP, ILP, Lagrangian relaxation) · **assignment and matching problems** · marketplace equilibrium reasoning · revenue vs advertiser-ROI vs user-experience trade-offs

**5. Ranking & recommendation** — *used for:* the domain qualifier — the posting requires prior work in ads ranking, recommenders, uplift or marketplace optimisation.

> Sub-topics: **CTR and CVR prediction** (named in the posting) · retrieval → ranking → re-ranking pipeline · two-tower models and ANN retrieval · learning-to-rank · **calibration for auctions** — a miscalibrated CTR model breaks bidding · multi-objective ranking · position-bias correction · cold start

**6. Metric design** — *used for:* the round that reveals whether you think like a business owner.

> Sub-topics: **CTR, CVR, incremental GMV** (the three named) · incremental vs observed conversion · defining the OEC when revenue and user experience conflict · metric sensitivity and variance · guardrails that block a launch · attribution windows · **proving incrementality rather than correlation**

**7. Python, SQL, Spark** — *used for:* the coding round and the scale conversation.

> Sub-topics: **Spark** (named explicitly) — DataFrame API, partitioning, shuffle, **skew handling**, broadcast joins, AQE, UDF cost, window functions at scale · SQL window functions and cohort queries · pandas and NumPy fluency · scikit-learn pipelines · writing code someone else can run

**8. Structured communication** — *used for:* the behavioural round; the posting asks for it in its own words.

> Sub-topics: "**bring structural clarity to open-ended product problems**" — framing before solving · translating causal-inference caveats into product language · cross-functional work with Product and Engineering · answer-first structuring · defending a design when challenged mid-answer

#### Sea Group / Shopee, TikTok / ByteDance, Coupang

- Ranking, retrieval and ads. Expect a **ML system design** round and a coding round.

**Skills weighted heaviest — in priority order**

**1. ML system design** — *used for:* the round that sets your level. Failing it is the most common reason for a downlevel here.

> Sub-topics: the 11-step framework (objective → framing → metrics → data → features → model → training → serving → evaluation → monitoring → scale) · **latency budgets under 100ms** · throughput and QPS estimation · offline/online feature parity · shadow deployment and canary rollout · cost at 10× traffic · **failure modes and rollback plans** · canonical designs: feed ranking, search ranking, ad click prediction, short-video recommendation

**2. Retrieval (candidate generation)** — *used for:* the first stage of every ranking system they run.

> Sub-topics: **two-tower / dual-encoder models** · negative sampling strategies (in-batch, hard negatives) · **ANN indexes — HNSW, IVF-PQ, ScaNN** · embedding dimensionality and quantisation trade-offs · collaborative filtering (ALS, BPR) · session-based and sequential retrieval · multi-channel retrieval blending

**3. Ranking models** — *used for:* the modelling depth round.

> Sub-topics: **learning-to-rank** — pointwise, pairwise (RankNet), listwise (LambdaMART) · **CTR architectures — Wide & Deep, DeepFM, DCN, DIN** · feature crossing and embedding tables · multi-task ranking (engagement + revenue + satisfaction) · **calibration** — Platt scaling, isotonic regression · **NDCG, MAP, MRR, Recall@K, Hit Rate**

**4. Real-time serving & infrastructure** — *used for:* the scale round; these companies serve billions of requests.

> Sub-topics: **feature stores** and point-in-time correctness · feature freshness and streaming updates (**Kafka, Flink**) · model serving (Triton, TorchServe, TensorRT) · **batching, caching, GPU utilisation** · embedding serving and index refresh · A/B infrastructure and traffic splitting

**5. Coding (DSA)** — *used for:* the elimination round. ByteDance in particular runs a genuinely hard algorithmic screen.

> Sub-topics: arrays and hashing · two pointers, sliding window · binary search · trees and graphs, BFS/DFS, topological sort · heaps and top-K · **dynamic programming** — these companies do ask real DP · complexity analysis stated aloud

**6. Bias correction & feedback loops** — *used for:* the senior-level depth question that juniors cannot answer.

> Sub-topics: **position bias** and inverse propensity weighting · popularity bias and long-tail coverage · selection bias in logged feedback · **counterfactual learning-to-rank** · exploration policies to avoid model collapse · diversity and serendipity objectives · filter-bubble mitigation

#### GoTo Group (Singapore entity), Gojek

- **Read in full — Senior Manager Data Scientist, Pricing:** **8–10 years**; dynamic pricing engines for ride-hailing; Python, SQL, stats/ML fundamentals "with projects demonstrating practical application". **Preferred:** econometrics, causal inference, simulation, real-time ML systems, Jupyter/Git/Docker, GCP/AWS/AliCloud, modern data stack, open-source contributions.

**Skills weighted heaviest — in priority order**

**1. Dynamic pricing** — *used for:* the domain round; prior ride-hailing or equally dynamic-sector experience is asked for by name.

> Sub-topics: **surge pricing and supply-demand balancing** · price elasticity under endogeneity · geo-temporal demand modelling · **ETA and travel-time prediction** as a pricing input · driver-rider matching and its interaction with price · fairness and regulatory constraints on surge · cross-subsidy between segments · long-run vs short-run elasticity

**2. Econometrics** — *used for:* the preferred qualification that separates candidates at this level.

> Sub-topics: **endogeneity and simultaneity** — why naive price-demand regression is wrong · **instrumental variables and 2SLS** · panel data and **fixed effects** · demand system estimation · structural vs reduced-form modelling · discrete choice models (logit, nested logit) · elasticity identification from natural experiments

**3. Simulation** — *used for:* the preferred qualification most candidates have never done; a strong differentiator.

> Sub-topics: **agent-based marketplace simulation** · Monte Carlo for policy evaluation · discrete-event simulation for matching and dispatch · counterfactual policy testing before a live rollout · calibrating a simulator against real data · using simulation to size an experiment you cannot run

**4. Causal inference** — *used for:* validating pricing changes where a clean A/B is often impossible.

> Sub-topics: **switchback and time-sliced designs** for marketplace interference · difference-in-differences across cities · synthetic control for city-level rollouts · **geo-experiments** · IPTW and doubly robust estimation · network interference between drivers and riders

**5. Real-time ML systems** — *used for:* the preferred qualification; pricing must score in milliseconds.

> Sub-topics: **Kafka and Flink** streaming · online feature computation and freshness · feature stores with online/offline parity · low-latency model serving · fallback pricing when the model is unavailable · monitoring a model that updates continuously

**6. Modern data stack & engineering workflow** — *used for:* the preferred list — Jupyter, **Git, Docker**, GCP/AWS/AliCloud, modern data stack.

> Sub-topics: Git branching and PR discipline · **Docker** and reproducible environments · cloud ML platforms (GCP Vertex, AWS SageMaker) · **dbt** transformation layer · Airflow orchestration · CI/CD for model pipelines · **open-source contributions and a public portfolio** (named as preferred)

**7. Python, SQL & ML fundamentals with evidence** — *used for:* the posting asks for fundamentals "with projects demonstrating practical application" — the projects are the point.

> Sub-topics: pandas, NumPy, scikit-learn fluency · SQL window functions and cohort analysis · regression, GBDTs, calibration, cross-validation design · leakage detection · **a portfolio where each project names the decision it informed**

**8. Leadership & cross-functional influence** — *used for:* this is a **Senior Manager** title; scope is assessed as heavily as modelling.

> Sub-topics: owning a pricing roadmap · mentoring and design review · communicating to cross-functional teams · negotiating trade-offs with Product, Ops and Finance · hiring and calibrating a bar · written decision documents

#### Banks & finance: DBS, OCBC, UOB, GIC, Temasek, Stripe SG, Airwallex, Nium, Thunes

- Model risk governance, MAS regulatory expectations, explainability. Slower, more documented, comparable pay at senior level.

**Skills weighted heaviest — in priority order**

**1. Model risk governance** — *used for:* the round that has no equivalent in consumer tech, and which almost no consumer-tech candidate can pass.

> Sub-topics: **model development documentation standards** · **independent model validation** — what a validator looks for · ongoing performance monitoring and annual review · model inventory and tiering · **champion-challenger frameworks** · conceptual soundness testing · outcome analysis and benchmarking · limitations and compensating controls

**2. Regulatory frameworks** — *used for:* the compliance interview; naming the right framework unprompted is a strong signal.

> Sub-topics: **MAS FEAT principles** — Fairness, Ethics, Accountability, Transparency (Singapore-specific) · **MAS Notice 637** and Basel III/IV capital models · **SR 11-7** for US-linked entities · **IFRS 9 expected credit loss** staging and lifetime PD · AML/CFT transaction monitoring obligations · sanctions screening · data residency and cross-border transfer rules

**3. Explainability** — *used for:* the technical round; in regulated finance a black box is a blocker, not a trade-off.

> Sub-topics: **SHAP** — global and local, and its known pitfalls · LIME · **counterfactual explanations** · **monotonic constraints in GBDTs** · scorecard-style logistic regression and why it survives · reason-code generation · partial dependence and ALE plots · stability of explanations across retraining

**4. Credit & market risk modelling** — *used for:* the domain round at banks and lenders.

> Sub-topics: **PD, LGD, EAD** and expected loss · **WOE binning and Information Value** · application vs behavioural scorecards · **KS, Gini, PSI** · reject inference · vintage and roll-rate analysis · survival analysis for time-to-default · stress testing and scenario analysis · **fair lending and disparate impact testing**

**5. Fraud & financial crime** — *used for:* the payments-side roles (Stripe SG, Airwallex, Nium, Thunes).

> Sub-topics: rules-engine and ML hybrids · **velocity features and time-window aggregations** · device fingerprinting and behavioural biometrics · account takeover vs synthetic identity vs first-party fraud · **graph analytics for rings** · extreme class imbalance · **false-positive cost modelling** — in AML the false positives are the whole problem · adversarial drift · **authorisation-rate optimisation** and decline recovery

**6. Statistics & time series** — *used for:* the quantitative round, which is more classical than in tech.

> Sub-topics: regression diagnostics and assumption testing · **GLMs** for counts and durations · **survival analysis** · time-series: stationarity, ARIMA/SARIMAX, cointegration · backtesting discipline · **calibration over discrimination** — a well-ranked but miscalibrated model fails a capital calculation

**7. Documentation & stakeholder discipline** — *used for:* the culture-fit round; the pace and the paper trail are the job.

> Sub-topics: writing a model development document a validator can approve · audit-ready lineage and versioning · presenting to a model risk committee · working with Compliance, Legal and Internal Audit · change management and sign-off processes · **reproducibility as a regulatory requirement, not a nicety**

### Singapore — the distinctive signal

**Causal inference is the gatekeeper, not GenAI.** Not one Singapore posting read in full named RAG, agents or fine-tuning. They named uplift modelling, switchbacks, bandits and auction theory. If you prepare for Singapore the way you prepare for India, you will fail the technical round.

### Singapore — what will block you

- Employment Pass. Singapore's COMPASS framework scores your salary, qualifications, and the employer's workforce diversity. Below-market salary offers fail the points test — which perversely means **lowballing yourself blocks the visa**.
- No equity literacy. Understand RSU vesting and refreshers before negotiating.

---

## 3. INDONESIA — Jakarta

### Band reality

| Level | Typical (IDR/year) | USD |
| --- | --- | --- |
| Data Scientist | 180–400M | $11–25K |
| Senior DS | 400–700M | $25–43K |
| **Staff / Lead / Manager** | **900M – 1.5B** | **$55–90K** |

**USD 68K requires staff or manager level at a GoTo-tier firm.** Local-payroll senior roles do not reach it.

### Companies

#### GoTo Group / Gojek / Tokopedia

- **Read in full — Senior Data Scientist, Jakarta:** **3–5 years** (note: "Senior" here sits a level below its Singapore equivalent). Python, SQL, basic visualisation, DS/ML fundamentals, statistical analysis and **experimentation design**, and the ability to take "Data Science models to production". **Bonus:** ML engineering, operations research, GCS/AWS/Azure, **Flink**, **Airflow**, Git/CI-CD, **dbt**, clear technical documentation.
- **Read in full — Senior Data Scientist, KYC, Jakarta:** **5+ years** deep learning on **computer vision, voice and text**; **PyTorch**, Python, **C++**, inference engines like **NCNN**; GCP or AWS; models "successfully deployed and used in production"; Android/iOS working knowledge a plus; **Master's or PhD preferred**.

**Skills weighted heaviest — in priority order**

**1. Taking models to production** — *used for:* stated as a core requirement at "Senior" (3–5 yrs), and as "deployed and used in production" in the KYC role. This is the hard filter.

> Sub-topics: packaging a model as a service (FastAPI, Docker) · batch vs real-time vs streaming inference · **Airflow DAG design**, idempotency, retries, backfills (Airflow named as bonus) · model registry and versioning · **CI/CD for ML** (Git/CI-CD named as bonus) · rollback procedure · on-call ownership · **a live endpoint you can point at**, not a notebook

**2. Deep learning across modalities** *(KYC track)* — *used for:* identity verification — face match, liveness, document OCR, voice.

> Sub-topics: **computer vision** — face detection and recognition, liveness/anti-spoofing, document OCR and forgery detection, image quality assessment · **speech** — speaker verification, anti-spoofing · **text** — document field extraction, name matching across scripts · **PyTorch** craft: custom `Dataset`, augmentation, debugging shape and NaN bugs · transfer learning from pretrained backbones · handling extreme class imbalance in fraud-labelled identity data

**3. Inference optimisation & edge deployment** — *used for:* the KYC posting names **C++ and NCNN** — this runs on a phone, not a server.

> Sub-topics: **NCNN** and mobile inference runtimes · ONNX export and graph simplification · **quantisation (INT8, 4-bit), pruning, distillation** · TensorRT and TorchScript · latency and memory budgets on low-end Android devices · **C++** enough to read and modify an inference path · Android/iOS integration basics (named as a plus) · accuracy-vs-latency trade-off curves

**4. Experimentation design** — *used for:* named explicitly even in the 3–5 year Senior role.

> Sub-topics: hypothesis formulation and metric selection · sample size and MDE · randomisation unit under marketplace interference · **switchback tests** for allocation and pricing · guardrail metrics · SRM detection · reading a test that disagrees with intuition

**5. Modern data stack** — *used for:* the "bonus" list that is effectively assumed — **Flink, Airflow, dbt, Git/CI-CD, GCS/AWS/Azure**.

> Sub-topics: **dbt** — models, refs, sources, tests, snapshots, macros · **Airflow** — DAGs, sensors, SLAs, backfills · **Flink** / Spark Structured Streaming — event time, watermarks, late data · BigQuery/Snowflake cost and partitioning · cloud storage and IAM basics · **clear technical documentation** (named explicitly in the posting)

**6. Operations research & optimisation** — *used for:* the "bonus" qualification that matters most for allocation and dispatch teams.

> Sub-topics: linear and integer programming · **assignment and matching problems** (driver-order, courier-batch) · vehicle routing (VRP) and batching · queueing theory for supply-demand · constrained allocation under budget · heuristics and metaheuristics when exact solvers are too slow · tools: OR-Tools, PuLP, Gurobi

**7. Python, SQL & DS fundamentals** — *used for:* the screening round.

> Sub-topics: pandas and NumPy fluency · SQL joins, window functions, cohort and funnel queries · **statistical analysis** — hypothesis testing, regression, distributions · scikit-learn pipelines and cross-validation · **basic visualisation** (named in the posting) · framing a business problem "as a technical problem that can be solved using data and math/stats/ML" — the posting's own words

**8. Communication & documentation** — *used for:* named in both postings; unusually prominent for this level.

> Sub-topics: **clear, concise technical documentation** (named twice) · conveying analytical solutions to business teams · independent problem framing without a spec · written design proposals · working in English across a regional team

#### Traveloka, Bukalapak, Blibli, Grab Indonesia, Shopee Indonesia

- Recommendations, pricing, supply-demand matching.

**Skills weighted heaviest — in priority order**

**1. Recommendation & personalisation** — *used for:* the primary revenue surface at all five.

> Sub-topics: retrieval → ranking → re-ranking · two-tower retrieval and ANN indexes · learning-to-rank and CTR/CVR models · **cold start for new SKUs, new sellers and new users** — acute in fast-growing markets · session-based recommendation for low-frequency categories (travel) · multi-objective ranking: relevance vs margin vs delivery feasibility · **NDCG, Recall@K, MRR**

**2. Pricing & revenue management** — *used for:* travel and marketplace dynamics; Traveloka in particular is a revenue-management shop.

> Sub-topics: **price elasticity estimation** and endogeneity · dynamic pricing and inventory-constrained pricing (hotel rooms, flight seats) · **markdown and clearance optimisation** · promotion and voucher incrementality · **uplift modelling for targeted discounts** · competitor price monitoring and response · bundling and ancillary pricing

**3. Supply-demand matching & forecasting** — *used for:* the operational core of Grab and Shopee Indonesia.

> Sub-topics: geo-temporal demand forecasting · **hierarchical forecasting with reconciliation** · courier/driver supply prediction · **ETA prediction** · batching and routing · warehouse and dark-store allocation · surge and incentive design · Ramadan/Lebaran seasonality — an Indonesia-specific modelling problem

**4. Experimentation in a marketplace** — *used for:* the design round.

> Sub-topics: interference between buyers and sellers · **switchback and geo-experiments** · cluster randomisation · CUPED variance reduction · guardrails against cannibalisation · long-run holdouts for incentive spend

**5. Production & data stack** — *used for:* the same bar as GoTo; these companies hire from the same pool.

> Sub-topics: deployment, monitoring, retraining · Airflow, dbt, Spark · feature stores and freshness · cloud platform depth in one of GCP/AWS/Azure

**6. Business framing in a price-sensitive market** — *used for:* the case round.

> Sub-topics: unit economics under thin margins · **take-rate and subsidy trade-offs** · COD (cash-on-delivery) and its effect on cancellation and fraud modelling · tier-2/tier-3 city expansion dynamics · logistics cost as a first-class model input

#### Fintech: Xendit, DANA, OVO, Kredivo, Amartha, Akulaku

- Credit scoring for thin-file populations, alternative data, fraud rings, collections optimisation.

**Skills weighted heaviest — in priority order**

**1. Credit scoring for thin-file borrowers** — *used for:* the defining problem of Indonesian lending, where most applicants have no bureau history.

> Sub-topics: **application vs behavioural scorecards** · **WOE binning and Information Value** · logistic scorecards and explainability · **PD/LGD/EAD**, expected loss · **reject inference** (parcelling, augmentation) · vintage and roll-rate analysis · **KS, Gini, PSI** · approval-rate vs bad-rate trade-off curves · **swap-set analysis** when replacing a model · IFRS 9 staging

**2. Alternative data** — *used for:* the skill that substitutes for a missing credit bureau.

> Sub-topics: **bank-statement cashflow underwriting** · e-wallet and transaction behaviour features · **device and telco signals** — SIM tenure, top-up patterns, handset tier · app-installed lists and their privacy limits · utility and e-commerce repayment history · psychometric scoring · social and referral graph features (Amartha's group-lending model) · **feature stability and drift when a data vendor changes**

**3. Fraud & financial crime** — *used for:* the fastest-moving problem; fraud rings adapt weekly.

> Sub-topics: **graph analytics and community detection for rings** · shared-attribute linking (device, IP, bank account, address) · **velocity features** across time windows · synthetic identity vs first-party vs account-takeover fraud · **sub-100ms real-time scoring** · extreme imbalance (0.01–0.5%) · **adversarial drift** and rapid retraining · rules + ML hybrid architectures · AML transaction monitoring and false-positive reduction

**4. Collections optimisation** — *used for:* a textbook uplift problem that most candidates have never framed correctly.

> Sub-topics: **uplift modelling for treatment assignment** — who to call, who to SMS, who to leave alone · S/T/X-learners and **Qini evaluation** · contact-channel and timing optimisation · roll-rate and cure-rate modelling · **cost-sensitive thresholds** — collection cost vs recovered amount · fairness and conduct constraints on collections practice

**5. Regulation** — *used for:* the compliance round; OJK rules have tightened sharply for digital lending.

> Sub-topics: **OJK digital lending regulations** and interest-rate caps · data-privacy limits on contact-list and location access · **PDP Law** (Indonesia's data protection law) · consent management · adverse-action explainability · fair-lending analogues · audit trail and model documentation

**6. Real-time ML engineering** — *used for:* fraud and underwriting decisions made during a checkout flow.

> Sub-topics: **Kafka/Flink** streaming features · online feature store with point-in-time correctness · latency budgets and fallback rules · shadow deployment before a live decision model · monitoring with delayed labels — you learn a loan defaulted months later

### Indonesia — the distinctive signal

**Production deployment is stated as a hard requirement even at "Senior" (3–5 yrs).** And the modern data stack — dbt, Airflow, Flink — appears as "bonus" so consistently that it is effectively assumed.

---

## 4. THAILAND — Bangkok

### Band reality

USD 50–90K, concentrated at a handful of global-HQ employers. Local Thai firms pay far less.

### Companies

#### Agoda — the dominant employer at this band

- Hires **Senior**, **Lead/Staff**, and a dedicated **Staff/Lead LLM Data Scientist** track, plus Senior/Staff Data Engineer roles.
- Roles are **"Bangkok based, relocation provided"** — meaning you compete against a global applicant pool, in English, and they will move you.
- Culture is heavily experimentation-driven; expect deep A/B testing and metric-design rounds.

**Skills weighted heaviest — in priority order**

**1. Experimentation at scale** — *used for:* the defining round. Agoda runs one of the largest continuous-testing programmes in travel.

> Sub-topics: **A/B test design end to end** — hypothesis, randomisation unit, MDE, duration · **statistical vs practical significance** · **sample ratio mismatch** · sequential testing and the peeking problem · **multiple comparisons across thousands of concurrent tests** · **CUPED and stratification** for variance reduction · interaction effects between simultaneously running tests · novelty and primacy effects · guardrail metrics and automatic stopping · **long-run holdouts** for cumulative effect

**2. Metric design** — *used for:* the round that reveals seniority; travel metrics are unusually delayed and noisy.

> Sub-topics: choosing the **OEC** when booking, cancellation and lifetime value conflict · **delayed-conversion metrics** — a booking today is a stay in three months · metric sensitivity and variance reduction · proxy metrics and their validation · gameable-metric detection · cancellation, no-show and refund as guardrails · **attribution across a long, multi-device funnel**

**3. SQL & large-scale data** — *used for:* the technical screen; Agoda's data volume is genuinely large.

> Sub-topics: window functions, CTEs, cohort and funnel queries · sessionisation across devices · query optimisation and partition pruning · Spark DataFrame API, skew and shuffle · columnar formats and cost control

**4. Statistics & causal inference** — *used for:* the depth round.

> Sub-topics: hypothesis testing choice and power analysis · **bootstrap and permutation tests** · regression diagnostics · **difference-in-differences** and synthetic control for market-level changes · propensity scores and IPTW where randomisation is impossible · heterogeneous treatment effects for personalisation

**5. Machine learning for travel** — *used for:* the applied-modelling tracks (ranking, pricing, supply).

> Sub-topics: **hotel and flight ranking** — learning-to-rank, personalisation · demand forecasting with strong seasonality and event effects · **price and availability prediction** · fraud and chargeback modelling · cancellation prediction · cold start for new properties · **calibration** for ranking-to-price consistency

**6. LLM systems** *(the Staff/Lead LLM Data Scientist track)* — *used for:* a newer, separately-hired lane; less competition than the general DS track.

> Sub-topics: RAG over property, review and policy content · **evaluation harnesses and golden sets** · fine-tuning with LoRA/QLoRA · multilingual handling across Asian markets · content generation with factual grounding · **cost-per-query and latency budgets** · guardrails and red-teaming

**7. Communication in a global, English-first team** — *used for:* the behavioural round. These are relocation roles; you compete with a global pool.

> Sub-topics: written analysis and experiment write-ups · presenting a result that contradicts a stakeholder's expectation · answer-first structure · working asynchronously across nationalities and time zones · **relocation readiness** as an explicit interview topic

#### Lazada Thailand, Shopee Thailand, LINE MAN Wongnai, SCBX, Ascend Money / TrueMoney, Bitkub

- E-commerce ranking, food-delivery logistics, digital lending, crypto exchange risk.

**Skills weighted heaviest — in priority order**

**1. Ranking & search** *(Lazada, Shopee)* — retrieval → ranking → re-ranking · two-tower models and ANN · learning-to-rank and CTR/CVR · position-bias correction · cold start · NDCG and Recall@K · **Thai-language query understanding**, tokenisation and transliteration

**2. Logistics & delivery optimisation** *(LINE MAN Wongnai)* — demand and courier-supply forecasting · **ETA prediction** · batching and vehicle routing · dispatch and assignment optimisation · surge and incentive design · **switchback tests** for dispatch changes

**3. Digital lending & credit risk** *(SCBX, Ascend Money / TrueMoney)* — WOE/IV and scorecards · PD/LGD/EAD · **alternative data — e-wallet and telco behaviour** · KS/Gini/PSI · reject inference · **Bank of Thailand digital lending rules** · collections uplift modelling

**4. Crypto & exchange risk** *(Bitkub)* — market-microstructure basics and order-book features · **anomaly and wash-trading detection** · on-chain analytics and address clustering · **AML for virtual assets — Travel Rule compliance** · liquidity and slippage modelling · extreme-tail risk and volatility modelling

**5. Cross-cutting production skills** — deployment and monitoring · Airflow/dbt/Spark · real-time serving for fraud and dispatch · SQL and Python fluency · working in Thai-language business contexts with English technical documentation

### Thailand — the distinctive signal

One employer dominates. Target Agoda specifically, prepare for experimentation at scale, and treat the LLM DS track as a separate, newer opening.

---

## 5. UNITED STATES

### Band reality

| Level | Years | Base | **Total comp** |
| --- | --- | --- | --- |
| Entry / Junior | 0–2 | $95–120K | $110–145K |
| Mid | 2–5 | $120–155K | $150–210K |
| **Senior** | **5–8** | **$160–210K** | **$230–330K** |
| **Staff / Principal** | **8+** | **$200–260K** | **$300–500K+** |
| DS Manager / Lead | 6+ | $190–250K | $280–450K |

**USD 68K (₹60 LPA) is below entry level in the US market.** If the US is your target, the question is not "how do I reach 60 LPA" but "how do I reach $230K+".

**Top-paying cities (2026):** San Francisco $172K avg (+30% vs national) · Remote-US $159K (+24%) · New York $137K (+12%) · Los Angeles $134K · Seattle $134K · Boston $132K.

**Highest-paying industries:** finance, healthcare and adtech — "due to regulatory requirements and data complexity". Fintech fraud-detection specialists out-earn marketing-focused DS at the same experience.

### Companies and what each wants

#### Meta — the archetype of the US "Product Data Scientist"

Assessed areas:

- **Data manipulation:** merging, filtering, finding insight — SQL, Python or R
- **Programming:** code efficiency, query and algorithm trade-offs
- **Statistics:** hypothesis testing, regression modelling
- **Experimentation:** A/B design, **statistical vs practical significance**
- **Research design:** experiment design and causal inference for product features
- **Metrics definition:** defining goals and success metrics for a feature
- **Product sense:** business objectives and user impact
- **Data storytelling** and **proven executive-level communication**

This is the most copied job spec in the industry — Airbnb, Lyft, DoorDash, Pinterest, Reddit, Instacart and Robinhood all run close variants.

**Skills weighted heaviest — in priority order**

**1. Product sense** — *used for:* the round Meta weights most heavily and the one most international candidates fail.

> Sub-topics: **defining goals and success metrics** for a feature (the posting's own phrase) · choosing the OEC when metrics conflict · **north-star metric and its input tree** · leading vs lagging indicators · **diagnosing a metric drop** — segment, seasonality, instrumentation, external shock · opportunity sizing before building · trade-offs across engagement, retention, revenue and trust · recommending *not* to ship · knowing Meta's actual product surfaces and their metrics

**2. Experimentation & research design** — *used for:* the dedicated round; "causal inference for product features" is named.

> Sub-topics: A/B design end to end · **statistical vs practical significance** (named explicitly) · MDE and power calculation · randomisation unit and **network interference** — critical at a social company · **cluster randomisation** for social graphs · sequential testing and peeking · multiple comparisons · **CUPED** · novelty and primacy effects · **quasi-experiments when randomisation is impossible** — DiD, synthetic control, RDD, IV, propensity scores · holdouts and long-run measurement

**3. Statistics** — *used for:* the analytical round, and as cross-examination everywhere else.

> Sub-topics: **hypothesis testing** — choosing the right test and defending it · **regression modelling** and its assumptions · confidence intervals and their misreadings · distributions and CLT limits · bootstrap and permutation tests · logistic regression and odds ratios · **Simpson's paradox**, selection and survivorship bias · variance, sensitivity and noise in metrics

**4. Data manipulation (SQL / Python / R)** — *used for:* the technical screen; "merging datasets, filtering data to find insights" in the posting's words.

> Sub-topics: **window functions and frame clauses** · self-joins and anti-joins · CTEs · **cohort, retention and funnel queries** · sessionisation · deduplication · date arithmetic and time zones · pandas `groupby().transform()`, `merge` semantics, reshaping · **writing a query top-down while narrating your reasoning**

**5. Programming efficiency** — *used for:* the round the posting calls "code efficiency and trade-offs in query/algorithm design".

> Sub-topics: complexity analysis of your own query or script · **query-plan reasoning** — why this join is slow · vectorisation over loops · sampling and approximation when exact is too expensive · **trade-off articulation**: readability vs speed vs cost · basic DSA: hashing, two pointers, sorting, binary search, BFS/DFS

**6. Data storytelling** — *used for:* the presentation round; named explicitly in the posting.

> Sub-topics: **answer-first (top-down) structure** · one chart per claim · visual encoding that carries the argument · translating uncertainty into decision language · **writing the recommendation, not just the finding** · handling a challenge to your conclusion mid-presentation

**7. Executive-level communication & influence** — *used for:* the hiring-manager round; "proven executive-level communication skills" is a stated requirement.

> Sub-topics: influencing a product roadmap without authority · **partnering with cross-functional teams** (posting's phrase) · written memos a VP can act on · disagreeing with a PM productively · 90-second behavioural answers with quantified outcomes · STAR stories mapped to scope and ambiguity

#### Google, Microsoft, Amazon, Apple, Netflix

- Google: separate analytical-reasoning round; strong stats bar.
- Amazon: Leadership Principles are half the loop; economist and applied-scientist tracks pay above DS.
- Netflix: very small, very senior team; experimentation and causal inference specialists.

**Skills weighted heaviest — in priority order**

**1. Analytical reasoning** *(Google's distinct round)* — structured problem decomposition · **estimation and Fermi problems** grounded in real data · choosing an approach and defending why not the alternatives · reasoning under missing information · stating assumptions explicitly and testing their sensitivity

**2. Statistics depth** *(Google, Netflix)* — the highest statistical bar of the group: distribution theory · **estimator properties — bias, variance, consistency** · maximum likelihood · Bayesian inference, priors and credible intervals · **experiment analysis with heavy-tailed metrics** · quantile regression · bootstrap variants

**3. Causal inference** *(Netflix especially)* — Netflix's DS function is essentially a causal-inference team: **potential outcomes and DAGs** · quasi-experimental methods when a test is impossible · **heterogeneous treatment effects** for personalisation · long-run and cumulative-effect measurement · interference in a shared-account product · **incrementality of content spend**

**4. Leadership Principles behavioural depth** *(Amazon)* — this is half the loop, not a formality: **STAR with a quantified result in every story** · 5–6 distinct stories mapped across the 16 principles · **Dive Deep** — you must know the numbers behind your own project · Disagree and Commit · Ownership beyond your remit · a genuine failure with a documented lesson · **written narrative skill** for Amazon's six-pager culture

**5. Applied-science modelling** *(Amazon, Apple, Microsoft — the higher-paying tracks)* — **ML system design** · forecasting and optimisation at supply-chain scale · **econometrics** for the Amazon Economist ladder · deep learning depth for Applied Scientist roles · publication record helps materially at Apple and Microsoft Research

**6. Scale and infrastructure literacy** — distributed compute (Spark, internal equivalents) · **petabyte-scale query cost reasoning** · streaming and near-real-time metrics · reproducible pipelines · working within a large internal tooling ecosystem you must learn fast

#### Fintech: Stripe, Block, Capital One, SoFi, Chime, Plaid, Ramp, Brex, Affirm

- **Model risk management under SR 11-7** is a named requirement at banks and larger fintechs — model development documentation, independent validation, ongoing monitoring.
- Fair lending, adverse action reasoning, ECOA/Reg B explainability.

**Skills weighted heaviest — in priority order**

**1. Model risk management under SR 11-7** — *used for:* the governance round; named as a requirement, and the hardest skill for a consumer-tech candidate to fake.

> Sub-topics: **model development documentation** — what must be in it · **independent validation** — conceptual soundness, outcome analysis, benchmarking · **ongoing monitoring** and annual review cadence · model inventory and risk tiering · **champion-challenger** frameworks · limitations, assumptions and compensating controls · effective challenge · model change management and re-approval triggers

**2. Fair lending & adverse action** — *used for:* the compliance round; **ECOA / Regulation B** is named.

> Sub-topics: **adverse action reason codes** — you must be able to say why someone was declined · **disparate impact testing** and the four-fifths rule · **proxy discrimination** — how a facially neutral feature becomes a protected-class proxy · BISG proxy methodology for race estimation · less-discriminatory-alternative search · **FCRA** obligations on credit data · fairness metrics and their mutual incompatibility

**3. Explainability** — *used for:* the technical round; regulation makes this non-negotiable.

> Sub-topics: **SHAP** — global, local, and its known pitfalls with correlated features · LIME · **monotonic constraints in GBDTs** — how to keep a tree model compliant · **WOE scorecards** and why logistic regression survives in lending · counterfactual explanations · **explanation stability across retraining** · partial dependence and ALE

**4. Credit risk modelling** — *used for:* the domain round at Capital One, SoFi, Chime, Affirm.

> Sub-topics: application vs behavioural scorecards · **PD, LGD, EAD**, expected loss · **WOE binning and Information Value** · **KS statistic, Gini, PSI** · reject inference · vintage, roll-rate and cohort loss curves · **CECL** lifetime expected credit loss · survival analysis for time-to-default · swap-set analysis · **approval-rate vs bad-rate frontier**

**5. Fraud & payments modelling** — *used for:* the domain round at Stripe, Block, Plaid, Ramp, Brex.

> Sub-topics: **real-time scoring under 100ms** · velocity and time-window aggregation features · **graph analytics for fraud rings** · device fingerprinting and behavioural biometrics · account takeover vs synthetic identity vs first-party fraud · **extreme class imbalance** and cost-sensitive thresholds · **authorisation-rate optimisation** and decline recovery · adversarial drift and rapid retraining · AML transaction monitoring and false-positive reduction

**6. Production engineering** — *used for:* fintech DS roles are unusually engineering-heavy.

> Sub-topics: real-time feature stores and point-in-time correctness · streaming (Kafka, Flink) · model serving with strict latency SLAs · **shadow deployment before a live money decision** · monitoring with **delayed labels** — default outcomes arrive months later · rollback under regulatory scrutiny

**7. Statistics & calibration** — *used for:* the quantitative round.

> Sub-topics: **calibration over discrimination** — a miscalibrated PD breaks pricing and capital · Platt scaling and isotonic regression · **Brier score** · confidence intervals on loss estimates · stress testing and scenario analysis · A/B testing a credit policy safely

#### Healthcare: Tempus, Flatiron Health, Verily, Komodo Health, Oscar, Devoted, Medeloop

- **Read in full — Healthcare Data Scientist (RWD), Medeloop:** SQL and Python required (R welcome); **R and/or SAS**; Spark/PySpark/BigQuery/Snowflake nice-to-have; **ICD, CPT, RxNorm** coding systems; **claims and EHR data**; **RWD/RWE methodology**; biostatistics and epidemiological methods; observational-data bias assessment; cohort modelling and patient-journey analysis; treatment pattern and outcomes evaluation; hands-on AI/ML with evidence; ability to review AI agent outputs; research-grade analysis and **publication experience**.

**Skills weighted heaviest — in priority order**

**1. Healthcare data literacy** — *used for:* the screening conversation. Without the vocabulary you cannot pass round one, regardless of modelling skill.

> Sub-topics: **ICD-9/10/11** diagnoses · **CPT/HCPCS** procedures · **RxNorm** drugs · LOINC labs · SNOMED CT clinical terms (the posting names ICD, CPT and RxNorm explicitly) · **claims data vs EHR data** — what each can and cannot answer · registry and wearable data · **FHIR and HL7** interoperability · **OMOP Common Data Model** and OHDSI tooling · coding drift and site heterogeneity · **enrolment gaps and continuous-eligibility windows** in claims

**2. RWD / RWE methodology** — *used for:* the core competency the posting asks for a track record in.

> Sub-topics: **cohort definition and phenotyping algorithms** · inclusion/exclusion and index-date selection · **target trial emulation** · comparative effectiveness research · **treatment pattern and line-of-therapy analysis** (named in the posting) · **patient journey analysis** (named) · external control arms for single-arm trials · FDA RWE framework and EMA guidance · **STROBE and RECORD reporting standards**

**3. Biostatistics & epidemiology** — *used for:* the methods round; this is a statistics job first and an ML job second.

> Sub-topics: **survival analysis** — Kaplan-Meier, log-rank, **Cox proportional hazards**, competing risks, time-varying covariates · study designs: cohort, case-control, nested case-control, cross-sectional · **propensity scores, IPTW, doubly robust, marginal structural models** · **the three classic traps: confounding by indication, immortal time bias, selection bias** (the posting asks for observational-data bias assessment) · missing data — MICE, multiple imputation, sensitivity analysis · power for clinical endpoints · meta-analysis

**4. SQL, Python, R and SAS** — *used for:* the technical screen. Note the unusual stack — **R and/or SAS** are asked for, unlike in tech.

> Sub-topics: **SQL** for cohort construction over huge claims tables — window functions, date-interval logic, episode grouping · **Python** for analysis pipelines · **R** — `survival`, `tidyverse`, `MatchIt` · **SAS** — still the regulatory-submission standard at pharma · **Spark/PySpark, BigQuery, Snowflake** (named as nice-to-have) · "high-quality, scalable code" — the posting's own phrase

**5. Clinical ML** — *used for:* the modelling round; "AI/ML experience required, with clear evidence of hands-on use".

> Sub-topics: risk prediction — readmission, sepsis, deterioration, no-show · **clinical NLP** — de-identification, concept extraction, **negation detection**, clinical BERT, LLMs on notes · medical imaging and DICOM handling · **calibration matters more than discrimination** in clinical deployment · **fairness across demographic groups** — documented harms here are severe · prospective validation · **reviewing AI agent outputs and model reasoning** (named in the posting)

**6. Regulation & privacy** — *used for:* the compliance conversation.

> Sub-topics: **HIPAA** — PHI, **Safe Harbor vs Expert Determination** de-identification, BAAs · **FDA SaMD** and the Predetermined Change Control Plan · IRB and ethics approval · GDPR special-category data for EU work · limited data sets and data use agreements · **payer vs provider vs pharma** — three buyers, three different questions

**7. Research-grade communication** — *used for:* the round that has no tech equivalent; **publication experience is explicitly requested**.

> Sub-topics: writing a manuscript or study report · **STROBE-compliant reporting** · presenting to clinicians and biostatisticians · **customer-facing partnership** with biopharma (named in the posting) · defending a methods choice to a reviewer · translating a clinical question into an analysable cohort definition

#### E-commerce & marketplace: Amazon, Instacart, DoorDash, Wayfair, eBay, Etsy, Shopify

- Demand forecasting, assortment, pricing, logistics, search ranking, incrementality of ad spend.

**Skills weighted heaviest — in priority order**

**1. Demand forecasting** — *used for:* the domain round; the highest-value forecasting problems in industry sit here.

> Sub-topics: **hierarchical forecasting and reconciliation** across SKU, store and region · intermittent demand (**Croston**) · **new-product cold start** · promotional lift decomposition · seasonality, holidays and calendar effects · **backtesting with rolling origin** and horizon-aware metrics · quantile forecasts and **pinball loss** for inventory decisions · M5-style competition techniques

**2. Search & recommendation ranking** — *used for:* the largest revenue surface at Amazon, eBay, Etsy, Wayfair.

> Sub-topics: **query understanding** — intent classification, spell correction, query expansion · retrieval → ranking → re-ranking · two-tower retrieval and ANN · **learning-to-rank on implicit feedback** · **position-bias correction and counterfactual LTR** · multi-objective ranking (relevance vs margin vs delivery speed vs seller fairness) · complementary vs substitute products · **NDCG, MAP, Recall@K**

**3. Pricing & promotions** — *used for:* the margin lever.

> Sub-topics: **price elasticity estimation under endogeneity** · dynamic and personalised pricing plus its fairness limits · **markdown optimisation** and clearance · **promotion incrementality** — how much would have happened anyway · competitor price monitoring and response · **uplift modelling for coupon targeting** · cannibalisation across channels and categories

**4. Logistics & operations** — *used for:* Instacart, DoorDash and Amazon's largest DS teams.

> Sub-topics: **ETA prediction** · batching and **vehicle routing (VRP)** · shopper/dasher supply forecasting and incentive design · warehouse slotting and pick-path optimisation · **inventory: safety stock, reorder points, newsvendor** · assortment and space optimisation · capacity planning · **marketplace matching under supply constraints**

**5. Incrementality & marketing measurement** — *used for:* the ad-spend round; named as a distinct competency.

> Sub-topics: **geo-experiments and holdout tests** for channel incrementality · **marketing mix modelling (MMM)** and its post-ATT revival · multi-touch attribution and its biases · **LTV prediction** — BG/NBD, Gamma-Gamma, DNN-based · CAC and payback modelling · **retail media ad ranking and auction mechanics** · churn and win-back targeting

**6. Experimentation under interference** — *used for:* the design round; marketplaces break naive A/B tests.

> Sub-topics: **buyer-seller and supply-side interference** · **switchback tests** for dispatch and pricing · cluster and geo randomisation · CUPED · guardrails against cannibalisation · **long-run holdouts for incentive spend** · SRM detection

**7. SQL, Python and production** — *used for:* the screen and the level-setting conversation.

> Sub-topics: SQL at warehouse scale — window functions, cohort and funnel queries, cost-aware querying · pandas and scikit-learn fluency · **Spark** for large jobs · Airflow orchestration · deploying a forecast or ranking model and owning its monitoring · **feature stores and offline/online parity**

### USA — the distinctive signal

The US splits the role in two, and they interview differently:

- **Product/Analytics DS** — SQL, stats, experimentation, product sense, communication. Little ML.
- **ML Engineer / Applied Scientist** — modelling depth, coding, ML system design. Pays more.

Pick one before you prepare. Preparing for both badly is the most common failure.

### USA — what will block you

- **Visa.** H-1B is a lottery; O-1 requires a documented record; cap-exempt (universities, some non-profits) and L-1 (internal transfer) are the reliable routes. Many companies will not sponsor for DS.
- Weak coding. Even "analytics" roles run a coding screen.

---

## 6. UNITED KINGDOM — London, Cambridge, Manchester, Edinburgh

### Band reality

| Level | Typical |
| --- | --- |
| Junior | £40K |
| Mid / Specialist | \~£112K |
| Senior / Lead | £130–260K (frontier AI labs and quant firms reach much higher) |

₹60 LPA ≈ £53K — roughly a **mid-level London** salary.

### Companies and what each wants

#### Tier A — AI labs: Google DeepMind, Anthropic (London), Stability, Wayve

Research-grade; a publication record is effectively expected.

**Skills weighted heaviest**

**1. Research depth** — reading and reproducing papers · **a publication record at NeurIPS/ICML/ICLR** or equivalent · formulating a novel research question · ablation design and honest negative results · **arXiv fluency in your sub-field**

**2. Deep learning fundamentals** — **transformer internals** — attention, positional encodings, normalisation placement · optimisation dynamics, learning-rate schedules, loss landscapes · **scaling laws** · pretraining vs instruction tuning vs preference tuning (RLHF, DPO) · architecture ablation reasoning

**3. Large-scale training engineering** — **distributed training — DDP, FSDP/ZeRO, tensor and pipeline parallel** · mixed precision, gradient accumulation and checkpointing · throughput and MFU optimisation · debugging a training run that silently diverges · JAX and/or PyTorch at depth

**4. Evaluation & safety** — benchmark design and contamination · **red-teaming and adversarial evaluation** · interpretability methods · alignment techniques · **honest reporting of capability limits**

**5. Mathematics** — linear algebra, probability, optimisation, information theory at derivation level, not recall level

#### Tier B — Fintech & quant: Revolut, Monzo, Starling, Wise, Checkout.com; Marshall Wace, Man Group, G-Research

Quant firms pay multiples of the standard DS market and interview as a different profession.

**Skills weighted heaviest — consumer fintech (Revolut, Monzo, Starling, Wise, Checkout)**

**1. Fraud & financial crime** — real-time scoring · **velocity features** · graph analytics for rings · account takeover and authorised push payment (APP) fraud · **AML transaction monitoring and false-positive reduction** · sanctions screening · **UK APP-fraud reimbursement rules** — a live regulatory driver

**2. Credit risk & affordability** — **FCA affordability assessment rules** · **Consumer Duty** obligations on outcomes and fair value · WOE scorecards, PD/LGD/EAD · **open banking cashflow underwriting** · KS/Gini/PSI · arrears and forbearance modelling

**3. Model governance** — **FCA/PRA model risk expectations (SS1/23)** · model documentation and independent validation · **UK GDPR Article 22** — automated decision-making and the right to an explanation · **ICO guidance on AI and data protection** · explainability with SHAP and monotonic constraints

**4. Growth & experimentation** — A/B testing on financial products with regulatory constraints · CUPED · uplift modelling for offer targeting · LTV and unit-economics modelling

**Skills weighted heaviest — quant (G-Research, Man Group, Marshall Wace, Jane Street London)**

**1. Mathematics & probability at competition level** — measure-theoretic probability · **stochastic calculus and Itô's lemma** · linear algebra and numerical methods · optimisation theory · **brainteaser-style probability under time pressure**

**2. Statistical learning for noisy signals** — **signal-to-noise realities of financial data** · overfitting control and multiple-testing correction across thousands of candidate signals · **time-series cross-validation with embargo and purging** · regime detection · feature decay

**3. Backtesting rigour** — **look-ahead and survivorship bias** · transaction costs, slippage and market impact · capacity constraints · Sharpe, drawdown, turnover · **out-of-sample discipline**

**4. Low-latency engineering** — **C++** at production level · vectorisation and cache-aware code · market-data handling · **order-book microstructure**

#### Tier C — E-commerce & consumer: Deliveroo, Ocado Technology, ASOS, Trainline, Depop

**Skills weighted heaviest**

**1. Logistics & fulfilment optimisation** *(Deliveroo, Ocado)* — **rider supply forecasting and dispatch** · ETA prediction · batching and routing (VRP) · **Ocado's warehouse robotics scheduling and pick optimisation** · capacity planning · **switchback tests** for dispatch changes

**2. Demand forecasting** — hierarchical and SKU-level forecasting · **fashion demand and size-curve forecasting** *(ASOS)* · promotional lift · **returns prediction** — a defining ASOS problem · intermittent demand · rolling-origin backtesting

**3. Ranking & personalisation** — search and feed ranking · learning-to-rank · **visual and multimodal similarity for fashion** · cold start · NDCG and Recall@K

**4. Pricing & experimentation** — elasticity estimation · **dynamic rail pricing** *(Trainline)* · markdown optimisation · A/B and geo-experiments · uplift modelling for promotions

#### Tier D — Health: BenevolentAI, Genomics England, NHS analytics

**Skills weighted heaviest**

**1. Biostatistics & epidemiology** — **survival analysis and competing risks** · cohort and case-control design · propensity scores and IPTW · **confounding by indication and immortal time bias** · missing data and multiple imputation

**2. Health data standards** — **SNOMED CT and ICD-10** (the NHS standard) · **OMOP CDM** · **NHS Digital / OpenSAFELY-style secure data environments** · primary-care (CPRD) and hospital (HES) datasets · linkage across datasets

**3. Genomics & bioinformatics** *(Genomics England, BenevolentAI)* — variant calling and annotation pipelines · **GWAS and polygenic risk scores** · multi-omics integration · knowledge-graph reasoning over biomedical literature *(BenevolentAI's core method)*

**4. Governance** — **UK GDPR special-category data** · **Caldicott Principles** and NHS information governance · research ethics and data access committees · **publication and peer review**

#### Tier E — Big tech London: Meta, Google, Amazon, Microsoft, Apple, Snowflake, Databricks

Same loops as the US offices — see §5 for the full Meta and Google breakdowns. Locally distinct: **UK GDPR and EU AI Act** exposure appears in privacy and responsible-AI rounds, and London teams are often EMEA-scoped, so **multi-market and multi-language analysis** comes up more than in a US-only role.

### UK — the distinctive signal

- **FCA/PRA model governance** for finance; **UK GDPR + ICO guidance** on automated decision-making everywhere.
- Quant finance (G-Research, Man, Marshall Wace, Jane Street London) is a genuinely separate market paying multiples of standard DS — but requires competition-grade maths and coding.

### UK — visa

Skilled Worker visa is routine and many employers sponsor. **Global Talent visa** (endorsed by Tech Nation successor bodies or via academic route) needs no job offer — the single best route for a strong profile.

---

## 7. GERMANY — Berlin, Munich, Hamburg

### Band reality

| Level | Typical (EUR) |
| --- | --- |
| Junior | €50–65K |
| Mid | €70–85K |
| **Senior** | **€115–150K+** |

₹60 LPA ≈ €62K — a **junior-to-mid** German salary.

### Companies and what each wants

#### Tier A — E-commerce: Zalando, Otto, About You, Delivery Hero (HQ Berlin), HelloFresh, Flink

**Skills weighted heaviest — in priority order**

**1. Experimentation at scale** — *used for:* Zalando and Delivery Hero both run mature in-house experimentation platforms; this is the decisive round.

> Sub-topics: A/B design end to end, MDE and power · **CUPED and stratified variance reduction** · **sample ratio mismatch** · sequential testing and peeking · multiple comparisons across concurrent tests · **switchback tests** for delivery dispatch *(Delivery Hero, Flink)* · guardrail metrics and OEC selection · **metric sensitivity for low-frequency purchase categories**

**2. Fashion & catalogue ML** *(Zalando, Otto, About You)* — **size and fit prediction** · **returns prediction and returns-cost modelling** — the defining economics problem in European fashion e-commerce · visual and multimodal similarity search · outfit and complementary-item recommendation · **cold start for seasonal assortments** · attribute extraction from product images and text

**3. Demand forecasting & supply chain** — hierarchical forecasting with reconciliation · **seasonal and weather-driven demand** · new-product cold start · **meal-kit and grocery perishability constraints** *(HelloFresh, Flink)* · safety stock and newsvendor · quantile forecasts and pinball loss · dark-store and micro-fulfilment allocation

**4. Ranking & personalisation** — retrieval → ranking → re-ranking · two-tower models and ANN · learning-to-rank on implicit feedback · **position-bias correction** · multi-objective ranking (relevance vs margin vs return rate) · NDCG and Recall@K

**5. Causal inference & marketing measurement** — **geo-experiments and holdout design** · marketing mix modelling · **incrementality of discount and CRM spend** · uplift modelling for voucher targeting · difference-in-differences across markets · LTV modelling

**6. Production & data stack** — deployment and monitoring · **Airflow, dbt, Spark/Databricks** · feature stores · AWS or GCP depth · **GDPR-compliant data handling** and consent-aware feature use

#### Tier B — Mobility & industrial: BMW, Mercedes-Benz, Bosch, Siemens, Volkswagen (CARIAD), Porsche Digital

This cluster values a **completely different stack** from consumer tech — and faces far less competition.

**Skills weighted heaviest — in priority order**

**1. Time-series & sensor analytics** — *used for:* the core of industrial DS.

> Sub-topics: **high-frequency multivariate sensor data** · signal processing — filtering, FFT, wavelets · resampling and alignment across sensors · **change-point and anomaly detection** · feature extraction from telemetry (tsfresh-style) · **CAN bus and vehicle telemetry** data structures

**2. Predictive maintenance & reliability** — **remaining useful life (RUL) estimation** · **survival analysis and Weibull reliability modelling** · failure-mode classification with very few positive examples · **cost-of-false-alarm modelling** — a false maintenance call is expensive · degradation modelling · warranty-claim analytics

**3. Optimisation & operations research** — production scheduling and line balancing · **supply-chain and multi-echelon inventory optimisation** · routing and logistics · **constraint programming and MILP** · simulation of manufacturing processes · tools: OR-Tools, Gurobi, CPLEX

**4. Computer vision for industrial use** — **visual quality inspection and defect detection** · anomaly detection with very few defect samples · **perception stack basics for autonomous driving** *(CARIAD, Bosch)* · sensor fusion (camera, radar, lidar) · **edge inference** and embedded deployment constraints

**5. Simulation & digital twins** — physics-informed machine learning · surrogate modelling of expensive simulations · **digital twin calibration against real telemetry** · design-of-experiments for engineering tests

**6. Domain and process context** — **ISO 26262 functional safety** basics for automotive · manufacturing process vocabulary (OEE, takt time, yield) · **working with engineers rather than product managers** · German language is a real advantage here, unlike in Berlin tech

#### Tier C — Fintech: N26, Trade Republic, Solaris, Raisin, Scalable Capital

**Skills weighted heaviest**

**1. Fraud & AML** — real-time transaction scoring · **velocity and device features** · graph-based ring detection · **BaFin AML expectations** and transaction-monitoring tuning · false-positive reduction · SEPA and instant-payment fraud patterns

**2. Credit risk & regulatory capital** — **IFRS 9 expected credit loss** staging · PD/LGD/EAD · **SCHUFA and German bureau data** · WOE scorecards · affordability assessment · **EBA guidelines on loan origination**

**3. Model governance under EU rules** — **BaFin / EBA model requirements** · **EU AI Act — credit scoring is explicitly high-risk** · GDPR Article 22 automated decision-making · model documentation and validation · **explainability as a legal obligation**

**4. Investment & trading analytics** *(Trade Republic, Scalable Capital)* — portfolio risk and factor modelling · **MiFID II suitability and appropriateness** · execution quality analysis · client-behaviour and churn modelling

#### Tier D — Enterprise & SaaS: SAP, Celonis, Personio

**Skills weighted heaviest**

**1. Process mining** *(Celonis — the category leader, headquartered in Munich)* — **event-log mining and process discovery** · conformance checking and variant analysis · **root-cause analysis of process deviations** · throughput and bottleneck modelling · simulation of process changes

**2. Enterprise GenAI** *(SAP)* — the same stack as the India SAP breakdown in §1: **RAG, agents, LLMOps, vector databases, LangChain/LlamaIndex/Semantic Kernel** · plus **EU AI Act compliance** and data-residency constraints, which weigh more heavily in the German entity

**3. B2B analytics** — **product-led growth metrics** — activation, expansion, net revenue retention · churn prediction with tiny positive-class counts and long cycles · **account-level rather than user-level modelling** · HR analytics and fairness constraints *(Personio)* · usage-based pricing analytics

**4. Enterprise architecture & integration** — Kubernetes, Docker, CI/CD · multi-tenancy and data isolation · **SAP BTP / cloud platform depth** · integration with ERP and HR systems

#### Tier E — Big tech: Amazon Berlin/Munich, Google Munich, Microsoft, Apple Munich

Same loops as the US offices (see §5). Munich in particular hosts **hardware, systems and privacy engineering** teams, so expect more **on-device ML, efficiency and differential-privacy** content than in a US product-DS loop.

### Germany — the distinctive signal

- **Zalando and Delivery Hero run heavy experimentation platforms** — A/B testing at scale, causal inference, and metric design are core.
- Industrial DS (Bosch, Siemens, BMW) values **time-series, sensor data, predictive maintenance and optimisation** far more than GenAI.
- German is not required at most tech firms, but is a real advantage in industrial and Mittelstand roles.

### Germany — visa

**EU Blue Card** — the most accessible skilled route in Europe. Lower salary threshold for shortage occupations (IT qualifies), path to permanent residence in 21–33 months.

---

## 8. NETHERLANDS — Amsterdam, Eindhoven, Utrecht

### Band reality

| Level | Typical (EUR) |
| --- | --- |
| Junior | €60K |
| Mid | €80–138K |
| Senior | €183K+ (top of market) |

### Companies and what each wants

#### Tier A — Booking.com

The most experimentation-mature company in Europe; runs thousands of concurrent tests. Their DS interviews are experimentation-first.

**Skills weighted heaviest — in priority order**

**1. Experimentation design & analysis** — *used for:* the round that decides the outcome. Nowhere in Europe weights this more.

> Sub-topics: A/B design end to end · **MDE, power and duration planning** · **statistical vs practical significance** · **sample ratio mismatch detection** · **interaction effects between thousands of concurrent tests** — a Booking-specific problem · sequential testing and peeking · multiple-comparison control at scale · **CUPED and stratification** · novelty and primacy effects · guardrails and automated stopping rules · **long-run and cumulative-effect holdouts**

**2. Metric design under delayed, noisy conversion** — *used for:* the seniority discriminator; travel metrics resolve months later.

> Sub-topics: choosing the **OEC** when bookings, cancellations and lifetime value conflict · **delayed conversion** — a click today is a stay in six months · **cancellation and no-show as guardrails** · proxy-metric construction and validation · metric variance and sensitivity · **attribution across a long, multi-device, multi-session funnel**

**3. Causal inference** — *used for:* the cases where randomisation is impossible or contaminated.

> Sub-topics: difference-in-differences and synthetic control for market-level changes · **geo-experiments** · propensity scores and IPTW · instrumental variables · **heterogeneous treatment effects** for personalisation · **interference between supply (properties) and demand (travellers)**

**4. Statistics & SQL depth** — hypothesis-test selection and defence · bootstrap and permutation tests · regression diagnostics · **heavy-tailed metric handling** · SQL window functions, sessionisation, funnel and cohort queries at very large scale

**5. Product sense for a marketplace** — supply-side (property) vs demand-side (traveller) trade-offs · ranking and its effect on partner fairness · **pricing and availability dynamics** · knowing when a statistically significant result is commercially irrelevant

#### Tier B — Fintech & payments: Adyen, Mollie, Bunq

**Skills weighted heaviest**

**1. Payments optimisation** *(Adyen's core)* — **authorisation-rate optimisation** · intelligent payment routing across acquirers · **decline recovery and retry logic** · 3DS and SCA exemption strategy under **PSD2** · chargeback and dispute prediction · **network-token and card-lifecycle modelling**

**2. Fraud & risk** — real-time scoring under strict latency · velocity and device features · **graph-based ring detection** · merchant-level risk scoring and onboarding underwriting · **AML under DNB and EU rules** · false-positive cost modelling

**3. Model governance** — **EU AI Act** obligations · GDPR Article 22 automated decision-making · **DNB / EBA model expectations** · explainability with SHAP and monotonic constraints · model documentation and validation

**4. Scale engineering** — streaming (Kafka, Flink) · sub-100ms serving · feature stores with point-in-time correctness · **monitoring with delayed labels**

#### Tier C — Retail & grocery: Ahold Delhaize, Picnic, Coolblue

**Skills weighted heaviest**

**1. Grocery demand forecasting** — **perishability and waste minimisation** · hierarchical forecasting across SKU, store and region · **promotional lift decomposition** · weather and holiday effects · quantile forecasts and **pinball loss for stock decisions** · new-product cold start

**2. Supply chain & fulfilment optimisation** *(Picnic in particular is an operations-research shop)* — **slot planning and delivery-window optimisation** · vehicle routing (VRP) with time windows · warehouse slotting and pick-path optimisation · **capacity and labour planning** · multi-echelon inventory · MILP and constraint programming

**3. Pricing & assortment** — elasticity estimation · **markdown optimisation for perishables** · assortment and shelf-space optimisation · private-label vs brand substitution modelling · competitor price response

**4. Personalisation & experimentation** — basket and complementary-item recommendation · **repeat-purchase and replenishment prediction** · A/B testing in a low-frequency, high-basket-value setting · CUPED

#### Tier D — Deep tech & health: ASML, Philips

**Skills weighted heaviest**

**1. Semiconductor process analytics** *(ASML)* — **high-dimensional sensor and metrology data** · **yield modelling and root-cause analysis** · statistical process control · **design of experiments** for equipment tuning · anomaly detection on machine telemetry · physics-informed and surrogate modelling · **predictive maintenance and remaining useful life**

**2. Medical device & clinical ML** *(Philips)* — **medical imaging** — segmentation, classification, DICOM handling · patient monitoring and deterioration prediction · **calibration over discrimination** in clinical deployment · fairness across demographic groups · **FDA SaMD and EU MDR** regulatory pathways · clinical validation study design

**3. Regulated engineering discipline** — **IEC 62304** software lifecycle for medical devices · traceability and design history files · verification and validation · **reproducibility as a regulatory requirement**

#### Tier E — Big tech & trading: Uber Amsterdam, Netflix EMEA, Databricks, Miro, Optiver, IMC

- **Uber / Netflix EMEA:** same loops as their US offices — see §5. EMEA scope means **multi-market and multi-language analysis** and heavier **GDPR and EU AI Act** exposure.
- **Optiver, IMC (market makers):** a different profession — **probability and mental-maths speed tests**, market microstructure, **C++ and low-latency engineering**, options pricing and greeks, **backtesting rigour with look-ahead and survivorship controls**. Pay is well above the DS market; the interview shares almost nothing with it.

### Netherlands — the distinctive signal

**The 30% ruling** — qualifying skilled migrants receive \~30% of salary tax-free for up to 5 years (being phased down; check current rules). This materially changes take-home versus Germany.

Booking.com and Adyen both weight **experimentation design and metric sensitivity** over ML modelling.

---

## 9. SWITZERLAND — Zurich, Lausanne, Basel

### Band reality

| Level | Typical (CHF) |
| --- | --- |
| Junior | CHF 87K |
| Mid | CHF 101–110K |
| **Senior** | **CHF 170–253K+** |

Highest nominal salaries in Europe; also the highest cost of living.

### Companies and what each wants

#### Tier A — Pharma & health: Roche, Novartis, Johnson & Johnson

The largest and least-contested opportunity in Switzerland. Basel is the global centre of pharmaceutical data science.

**Skills weighted heaviest — in priority order**

**1. Biostatistics** — *used for:* the methods round. This is the core discipline, not an adjunct to ML.

> Sub-topics: **clinical trial design** — phase I/II/III, randomisation, blinding, stratification · **sample size and power for clinical endpoints** · **survival analysis** — Kaplan-Meier, log-rank, **Cox proportional hazards**, competing risks, time-varying covariates · **adaptive and group-sequential designs**, alpha spending · **estimands framework (ICH E9(R1))** · multiplicity control across endpoints · **mixed models for repeated measures (MMRM)** · Bayesian trial designs · missing data: MICE, tipping-point and sensitivity analysis

**2. Real-world evidence & causal inference on observational data** — *used for:* the fastest-growing function in pharma.

> Sub-topics: **target trial emulation** · propensity scores, IPTW, **marginal structural models** · **confounding by indication, immortal time bias, selection bias** · **external control arms** for single-arm oncology trials · comparative effectiveness research · claims and EHR data across US/EU sources · **OMOP CDM and OHDSI tooling** · **STROBE and RECORD reporting**

**3. Regulatory statistics & submissions** — *used for:* the discipline that makes this work different from tech DS.

> Sub-topics: **SAS** — still the submission standard, and genuinely required · **CDISC standards — SDTM and ADaM** dataset structures · statistical analysis plan (SAP) authoring · **ICH guidelines** · FDA and EMA submission processes · **21 CFR Part 11** validated computing environments · reproducibility and audit trail as legal requirements · **defending a methods choice to a regulator**

**4. Clinical & biomedical domain literacy** — therapeutic-area vocabulary (oncology, immunology, neuroscience) · **endpoints — OS, PFS, ORR, response criteria** · biomarkers and companion diagnostics · **ICD, MedDRA, WHO Drug** coding dictionaries · drug development lifecycle and its economics

**5. Bioinformatics & multi-omics** *(for research-facing roles)* — genomics pipelines, variant calling and annotation · **differential expression and pathway analysis** · single-cell analysis · **GWAS and polygenic risk scores** · proteomics and multi-omics integration · R/Bioconductor ecosystem

**6. ML with clinical constraints** — **calibration over discrimination** · fairness across demographic groups · interpretability as a hard requirement · prospective validation · **clinical NLP** on trial and safety narratives · **pharmacovigilance signal detection** · small-n, high-dimension modelling

**7. Publication & scientific communication** — peer-reviewed publication record · presenting to clinicians and regulators · **medical writing conventions** · defending results in scientific review

#### Tier B — Big tech: Google Zurich, Apple, Meta, Microsoft, Disney Research

Google Zurich is its largest engineering site outside the US; these loops mirror the US ones (see §5), with local emphases.

**Skills weighted heaviest**

**1. Research-grade ML** — Zurich hosts core research and infrastructure teams: **publication record valued** · deep learning depth, transformers and multimodal models · **privacy-preserving ML — differential privacy, federated learning** (a real Zurich specialisation) · ML system design at global scale

**2. Statistics & experimentation** — the standard Google bar: analytical reasoning round · hypothesis testing and power · experiment design and metric definition · **causal inference for product changes**

**3. Engineering depth** — strong coding (this is an engineering site, not an analytics site) · distributed systems literacy · **large-scale data processing** · C++ or Go exposure helps at Google Zurich specifically

**4. Multi-language and multi-market analysis** — EMEA scope, **GDPR and EU AI Act** exposure, cross-locale metric comparability

#### Tier C — Finance & insurance: UBS, Julius Baer, Swiss Re, Zurich Insurance

**Skills weighted heaviest**

**1. Actuarial & insurance modelling** — **frequency-severity modelling** and GLMs (Poisson, gamma, Tweedie) · **reserving — chain ladder, Bornhuetter-Ferguson** · pricing and technical-premium calculation · **catastrophe and reinsurance modelling** *(Swiss Re's core)* · **extreme value theory** and tail risk · exposure and accumulation modelling

**2. Model risk & regulation** — **FINMA expectations** · **Solvency II / Swiss Solvency Test** internal models · **IFRS 17** insurance contract accounting · Basel III/IV for the banks · model documentation, independent validation, **effective challenge** · **EU AI Act** for insurance pricing

**3. Wealth & market analytics** *(UBS, Julius Baer)* — portfolio risk and factor models · **VaR, expected shortfall, stress testing** · client segmentation and next-best-action · **MiFID II suitability** · AML and sanctions for private banking · churn and asset-outflow prediction

**4. Explainability & fairness in pricing** — **avoiding proxy discrimination in insurance rating** · SHAP and monotonic constraints · GLM-style transparency and why it persists · regulatory justification of rating factors

#### Tier D — Deep tech: ETH Zurich / EPFL spinouts, Scandit, Climeworks

**Skills weighted heaviest**

**1. Applied research translation** — reading and implementing papers · **prototype-to-product engineering** · working directly from academic collaborations · a strong publication or thesis record is often the entry ticket

**2. Computer vision & edge ML** *(Scandit)* — **barcode and text recognition under poor conditions** · on-device inference — **quantisation, pruning, distillation** · ONNX, TensorRT, mobile runtimes · latency and battery budgets · synthetic data generation for training

**3. Physical-systems modelling** *(Climeworks and climate/energy tech)* — **process optimisation and control** · sensor time-series and anomaly detection · **physics-informed machine learning** · surrogate models for expensive simulations · **lifecycle and carbon accounting analytics**

**4. Small-team engineering breadth** — you own the whole stack: data engineering, modelling, deployment, monitoring · **MLOps without an MLOps team** · cloud cost discipline at startup scale

### Switzerland — the distinctive signal

**Pharma is the differentiator.** Roche and Novartis hire at volume for causal inference on observational data, survival analysis, and regulatory-grade statistics — a skill set that transfers poorly from consumer tech and therefore faces less competition.

---

## 10. NORDICS & IRELAND

### Band reality

| Country | Junior | Mid | Senior |
| --- | --- | --- | --- |
| **Denmark** | DKK 400–520K | DKK 600–685K | DKK 1.4–1.6M |
| **Sweden** | SEK 500K | \~SEK 670K | SEK 888K+ |
| **Norway** | NOK 598–800K | — | NOK 1.0–1.28M+ |
| **Finland** | €36K | \~€60K | €72–90K |
| **Ireland** | €60–65K | €75–85K | €125–126K+ |

### Companies and what each wants

#### Sweden — Spotify, Klarna, King, Northvolt, H&M, Truecaller, Einride

**Skills weighted heaviest — in priority order**

**1. Experimentation & product sense** *(Spotify — the canonical European product-DS interview)* — *used for:* the decisive round; their loop mirrors Meta's structure.

> Sub-topics: A/B design, MDE and power · **metric definition and OEC selection for a subscription product** · **statistical vs practical significance** · CUPED · sequential testing · **listening-behaviour metrics** — engagement vs discovery vs retention trade-offs · diagnosing a metric drop · **long-run holdouts for recommendation changes** · product sense on Spotify's actual surfaces

**2. Recommendation & personalisation** *(Spotify, King)* — **sequential and session-based recommendation** · two-tower retrieval and ANN · **audio and content embeddings** · exploration vs exploitation and **bandits for discovery** · **cold start for new artists and new tracks** · diversity, serendipity and filter-bubble mitigation · playlist and editorial-curation modelling · NDCG and Recall@K

**3. Credit risk & fraud** *(Klarna — BNPL)* — **instant underwriting at checkout, sub-second** · thin-file and alternative-data scoring · **BNPL-specific default dynamics** · WOE scorecards, PD/LGD/EAD, KS/Gini/PSI · **Swedish FSA and EU consumer-credit rules** · **EU AI Act — credit scoring is high-risk** · fraud rings and device signals · collections uplift modelling

**4. Games analytics** *(King)* — **player LTV and monetisation modelling** · churn and D1/D7/D30 retention · **level-difficulty tuning and progression modelling** · A/B testing in a live-ops environment · **whale and payer segmentation** · in-game economy balancing and simulation

**5. Industrial & supply-chain ML** *(Northvolt, H&M, Einride)* — **battery-manufacturing yield and process analytics** *(Northvolt)* · **fashion demand and size-curve forecasting, returns prediction** *(H&M)* · **autonomous-freight routing and fleet optimisation** *(Einride)* · sensor time-series and predictive maintenance

**6. Production & privacy** — deployment, monitoring and drift alerting · Airflow, dbt, Spark/Databricks · **GDPR-compliant personalisation and consent-aware features** · GCP or AWS depth

#### Denmark — Novo Nordisk, Maersk, Zendesk, Pleo, Vestas

**Skills weighted heaviest — in priority order**

**1. Biostatistics & clinical data science** *(Novo Nordisk — one of Europe's largest employers of statisticians)* — *used for:* the core hiring need; see the Switzerland pharma breakdown for full depth.

> Sub-topics: **clinical trial design and the estimands framework (ICH E9(R1))** · **survival analysis and MMRM** · adaptive and group-sequential designs · **SAS and CDISC (SDTM/ADaM)** · statistical analysis plan authoring · **real-world evidence and target trial emulation** · diabetes and obesity therapeutic-area literacy · **regulatory submission discipline and 21 CFR Part 11**

**2. Supply chain & logistics optimisation** *(Maersk — global container shipping)* — **network and route optimisation** · **container repositioning and empty-equipment modelling** · **port congestion and vessel ETA prediction** · demand forecasting for freight · **revenue management and dynamic freight pricing** · MILP and large-scale optimisation · simulation of network changes

**3. Industrial & energy analytics** *(Vestas — wind turbines)* — **SCADA and turbine telemetry** time-series · **power-curve modelling and performance loss detection** · **predictive maintenance and remaining useful life** · wind-resource and generation forecasting · **weather-model integration** · anomaly detection with few failure examples

**4. B2B SaaS analytics** *(Zendesk, Pleo)* — **product-led growth metrics** — activation, expansion, net revenue retention · **account-level churn with long cycles and few positives** · usage-based pricing analytics · **support-ticket NLP and deflection modelling** *(Zendesk)* · **spend-anomaly and expense-fraud detection** *(Pleo)*

#### Norway — DNB, Equinor, Cognite, Oda

**Skills weighted heaviest**

**1. Energy & subsurface analytics** *(Equinor, Cognite)* — **industrial IoT and sensor time-series at scale** · **predictive maintenance for offshore assets** · **production optimisation and reservoir modelling** · **physics-informed ML and digital twins** *(Cognite's product category)* · anomaly detection on equipment telemetry · **HSE-critical model reliability** — failures have safety consequences

**2. Banking risk & compliance** *(DNB)* — **IRB credit models under Basel** · **IFRS 9 expected credit loss** · AML and transaction monitoring under Norwegian FSA rules · model validation and governance · **GDPR Article 22** explainability

**3. Grocery & fulfilment** *(Oda)* — **perishable demand forecasting and waste minimisation** · **delivery-slot and route optimisation** · warehouse automation and pick optimisation · replenishment and repeat-purchase prediction

#### Finland — Wolt, Supercell, Nokia, Relex

**Skills weighted heaviest**

**1. Marketplace & delivery optimisation** *(Wolt)* — **courier supply forecasting and dispatch** · **ETA prediction and batching** · surge and incentive design · **switchback tests for dispatch changes** · three-sided marketplace dynamics (customer, courier, merchant) · unit economics per delivery

**2. Games analytics** *(Supercell)* — **LTV and monetisation modelling** · retention curves and cohort analysis · **live-ops A/B testing** · game-economy simulation and balancing · **soft-launch market testing** and generalising from small samples

**3. Retail forecasting & optimisation** *(Relex — a forecasting software company)* — **hierarchical retail forecasting at massive SKU-store scale** · promotional lift modelling · **replenishment and allocation optimisation** · fresh-product and waste modelling · **forecast accuracy metrics and backtesting infrastructure**

**4. Telecom & networks** *(Nokia)* — **network-performance time-series and anomaly detection** · traffic forecasting and capacity planning · **RAN optimisation and self-organising networks** · edge and on-device inference · **5G/6G research-track roles** where a publication record matters

#### Ireland (Dublin) — Google, Meta, Microsoft, Stripe, Intercom, Workday, LinkedIn, Amazon

The EMEA-HQ cluster. The reqs are frequently the *same requisition* as the US one at lower comp — so the skill bar is the US bar.

**Skills weighted heaviest**

**1. The US big-tech loop, unchanged** — see §5 for the full Meta, Google and Amazon breakdowns: **product sense, experimentation, statistics, SQL/Python, data storytelling, executive communication** · Amazon Leadership Principles apply identically in Dublin

**2. Payments & fraud** *(Stripe Dublin)* — **authorisation-rate optimisation** and routing · real-time fraud scoring · **PSD2 / SCA exemption strategy** · chargeback and dispute modelling · merchant underwriting

**3. EMEA-scope analysis** — *used for:* what actually distinguishes a Dublin role from its US twin.

> Sub-topics: **multi-market, multi-language and multi-currency analysis** · locale-level metric comparability · **GDPR as a first-class constraint** — Ireland's DPC is the lead supervisory authority for most US tech firms in the EU · **EU AI Act** readiness · data residency and transfer mechanisms · **privacy-preserving analytics** — differential privacy, aggregation thresholds, consent-aware feature use

**4. B2B SaaS analytics** *(Intercom, Workday, LinkedIn)* — product-led growth and net revenue retention · **account-level churn and expansion modelling** · **support and conversational NLP** *(Intercom)* · **HR and workforce analytics with strict fairness constraints** *(Workday)* · professional-graph and recommendation modelling *(LinkedIn)*

### Nordics — the distinctive signal

- **Spotify** is the canonical European product-DS interview: experimentation, metric design, product sense, ML for personalisation. Their interview guides mirror Meta's structure.
- **Novo Nordisk and Maersk** pay well for domain specialists — pharma statistics and supply-chain optimisation respectively.
- Flat hierarchies, strong work-life norms, high tax. Compare **net**, not gross.

### Ireland — why it matters

Dublin is the **easiest English-speaking EU entry point**: Critical Skills Employment Permit, huge concentration of US tech EMEA HQs, and roles that are often the same req as the US one at lower comp.

---

## 11. Cross-market comparison — where does the same skill pay most?

| Skill | India | Singapore | Indonesia | USA | UK | Germany | Netherlands | Switzerland |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Causal inference / experimentation | Medium | **Critical** | High | **Critical** | High | High | **Critical** | High |
| LLM / GenAI systems | **Critical** | Low–Med | Low–Med | High | High | Medium | Medium | Medium |
| MLOps / production ownership | High | High | **Critical** | High | High | High | High | High |
| Recommender / ranking | High | **Critical** | High | High | Medium | High | High | Low |
| Deep learning (CV/NLP/speech) | High | Medium | High | High | High | High | Medium | High |
| Time-series & forecasting | Medium | High | High | High | Medium | **Critical** | High | Medium |
| Biostatistics / survival analysis | Low | Low | Low | High | Medium | Medium | High | **Critical** |
| Credit risk & model governance | High | High | High | **Critical** | **Critical** | High | High | High |
| Product sense & storytelling | Medium | High | Medium | **Critical** | High | Medium | High | Medium |

Read the columns, not the rows. **Your preparation should be one column deep, not one row wide.**

---

# PART II — THE SKILL TAXONOMY IN DETAIL

Each block below lists the **topic**, its **sub-topics**, and the **bar** — what "good enough for the band" actually means, as opposed to "I have heard of it".

---

## A. Programming & Software Engineering

### A1. Python — production grade

- Core language: comprehensions, generators, decorators, context managers, `dataclasses`, typing/`mypy`
- OOP and composition; when a class earns its place over a function
- Error handling, logging (`structlog`), configuration management (`pydantic-settings`, Hydra)
- Concurrency: `asyncio`, multiprocessing, when each helps and when neither does
- Memory and performance: profiling (`cProfile`, `memory_profiler`, `py-spy`), vectorisation, avoiding copies
- Packaging: `pyproject.toml`, `uv`/`poetry`, virtual environments, publishing an internal package

### A2. Scientific Python stack

- NumPy: broadcasting, strides, views vs copies, `einsum`
- pandas: `MultiIndex`, `groupby().transform()`, window functions, memory dtypes, `merge` semantics
- **Polars / DuckDB** — increasingly expected; out-of-core and lazy evaluation
- scikit-learn: `Pipeline`, `ColumnTransformer`, custom transformers, `cross_val_predict`, calibration

### A3. Software engineering discipline

- Git: branching, rebase vs merge, resolving conflicts, meaningful commits, PR hygiene
- Testing: `pytest`, fixtures, parametrisation, mocking; **testing data and models**, not just code
- Code review: giving and receiving
- CI/CD: GitHub Actions or GitLab CI — lint, test, build, deploy on merge
- Design patterns used in ML code: strategy, factory, repository, dependency injection

### A4. Data structures & algorithms (for the coding screen)

- Arrays, two pointers, sliding window · Hashing · Strings
- Trees, graphs, BFS/DFS, topological sort
- Heaps, binary search, sorting
- Dynamic programming — the common patterns, not the exotica
- Complexity analysis, stated out loud while you solve

**The bar:** you can hand someone a repository, and they can run it, read it, and change it without asking you questions.

---

## B. SQL & Data Modelling

### B1. Query craft

- Joins: inner/left/full/cross/anti/semi, and what each does to row counts
- Window functions: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`/`LEAD`, `SUM() OVER`, frame clauses
- CTEs, recursive CTEs, lateral joins
- Aggregation: `GROUPING SETS`, `ROLLUP`, `CUBE`, `FILTER`
- Date/time arithmetic, time zones, and calendar tables
- `PIVOT`/`UNPIVOT`, arrays and structs, JSON extraction
- Set operations, `QUALIFY`, `DISTINCT ON`

### B2. Performance

- Reading an `EXPLAIN` plan
- Indexes, partitioning, clustering, sort keys
- Predicate pushdown, partition pruning, broadcast vs shuffle joins
- Avoiding full scans, skew handling, cardinality estimation

### B3. Modelling & warehouse design

- Star and snowflake schemas; fact vs dimension tables
- Slowly changing dimensions (Type 1/2/3)
- Grain — the single most important word in data modelling
- Idempotency and backfills
- **dbt**: models, sources, refs, tests, snapshots, exposures, macros
- Data contracts and schema evolution

### B4. Analytical patterns

- Cohort and retention analysis; N-day and rolling retention
- Funnels with step ordering and time windows
- Sessionisation
- Attribution (first/last touch, position-based, data-driven)
- Anomaly detection in metrics

**The bar:** a 200-line SQL query that someone else can read, that runs in seconds, and whose grain you can state in one sentence.

---

## C. Statistics & Probability

### C1. Foundations

- Probability axioms, conditional probability, Bayes' theorem
- Distributions: Bernoulli, binomial, Poisson, exponential, normal, log-normal, beta, gamma — and where each arises
- Expectation, variance, covariance, correlation vs dependence
- Law of large numbers, central limit theorem — and when CLT does *not* save you
- Moment generating functions (light touch)

### C2. Inference

- Estimators: bias, variance, consistency, efficiency
- Maximum likelihood; method of moments
- Confidence intervals — and what they do *not* mean
- Hypothesis testing: null/alternative, Type I and II error, power, effect size
- p-values and their misinterpretations
- Bootstrap and permutation tests
- Bayesian inference: priors, posteriors, credible intervals, conjugacy, MCMC basics

### C3. Regression & modelling

- OLS: assumptions, diagnostics, heteroskedasticity, multicollinearity
- Logistic regression; odds ratios; separation
- GLMs: Poisson, negative binomial, gamma — for counts and durations
- Regularisation: ridge, lasso, elastic net
- Mixed effects / hierarchical models
- Quantile regression

### C4. Multiple comparisons & sequential testing

- Bonferroni, Benjamini-Hochberg FDR
- Peeking problem; always-valid inference; group sequential designs
- Alpha spending functions

**The bar:** someone challenges your test result in a meeting and you can defend the design, the power calculation, and the decision rule without hedging.

---

## D. Causal Inference & Experimentation

*The single highest-leverage block for Singapore, USA, Netherlands.*

### D1. Experiment design

- Randomisation units: user, session, device, cluster — and interference between them
- Sample size and **minimum detectable effect** calculation
- Variance reduction: **CUPED**, stratification, covariate adjustment
- **Switchback tests** (for marketplaces where treatment leaks)
- **Cluster randomisation** and network effects
- Guardrail metrics, OEC (overall evaluation criterion), metric sensitivity
- Novelty and primacy effects; test duration
- Multi-armed bandits, Thompson sampling, contextual bandits
- Interleaving (for ranking systems)
- Holdout groups and long-run measurement

### D2. Quasi-experimental methods

- Difference-in-differences; parallel trends; event studies
- Synthetic control
- Regression discontinuity (sharp and fuzzy)
- Instrumental variables; two-stage least squares; weak instruments
- Propensity score matching, IPTW, doubly robust estimation
- Panel methods and fixed effects

### D3. Modern causal ML

- Potential outcomes framework; SUTVA; ignorability
- DAGs, confounders, colliders, backdoor criterion (Pearl)
- **Uplift / heterogeneous treatment effect modelling**: S-learner, T-learner, X-learner, R-learner
- **Causal forests**, DR-learner, **Double/Debiased ML**
- Qini and uplift curves for evaluation
- Sensitivity analysis for unobserved confounding

### D4. Experimentation platform thinking

- Assignment service, exposure logging, metric computation pipeline
- Sample ratio mismatch (SRM) detection
- Metric definitions as code; metric repositories
- Auto-analysis and the guardrails that stop bad launches

**The bar:** you can take an open-ended product question, choose between a randomised test and an observational design, justify the choice, and say what would make you wrong.

---

## E. Classical Machine Learning

### E1. Supervised learning

- Linear and logistic models; regularisation paths
- Tree ensembles: random forest, **XGBoost / LightGBM / CatBoost** — objectives, tuning, categorical handling
- SVMs and kernels (conceptual)
- Naive Bayes, k-NN
- Calibration: Platt scaling, isotonic regression — critical in risk and ads

### E2. Unsupervised & representation

- Clustering: k-means, DBSCAN, HDBSCAN, hierarchical, GMM
- Dimensionality reduction: PCA, SVD, t-SNE, UMAP
- Anomaly detection: isolation forest, one-class SVM, autoencoders
- Matrix factorisation; embeddings from co-occurrence

### E3. Model evaluation

- Classification: ROC-AUC, PR-AUC, log loss, Brier score, KS statistic, Gini
- Regression: MAE, RMSE, MAPE, pinball loss
- Ranking: NDCG, MAP, MRR, Hit Rate, Recall@K
- Cross-validation variants: stratified, group, **time-series split**
- Leakage detection — the most common silent failure
- Threshold selection tied to business cost

### E4. Feature engineering

- Target/mean encoding with proper out-of-fold construction
- Interaction and polynomial features
- Binning, WOE (weight of evidence) — standard in credit risk
- Missing-value strategies and missingness as signal
- Feature selection: permutation importance, SHAP, mutual information, recursive elimination
- **Feature stores**: offline/online consistency, point-in-time correctness

### E5. Imbalanced & cost-sensitive learning

- Resampling: SMOTE and its discontents
- Class weights, focal loss
- Cost matrices and decision thresholds

### E6. Time series & forecasting

- Decomposition, stationarity, differencing, ACF/PACF
- ARIMA/SARIMAX, exponential smoothing, **Prophet**
- Gradient boosting for forecasting with lag features
- Hierarchical and reconciled forecasts
- Intermittent demand (Croston), cold start
- Backtesting with rolling origin; horizon-aware metrics

**The bar:** you can explain why your model beat the baseline, and what it would take to break it.

---

## F. Deep Learning

### F1. Fundamentals

- Backpropagation and autodiff
- Optimisers: SGD, momentum, Adam, AdamW; learning-rate schedules
- Initialisation, normalisation (batch/layer/group), residual connections
- Regularisation: dropout, weight decay, early stopping, augmentation
- Loss functions and when to write a custom one

### F2. Architectures

- MLPs and tabular deep learning (and why GBDTs usually still win)
- CNNs: convolution, pooling, modern backbones (ResNet, EfficientNet, ConvNeXt)
- RNN/LSTM/GRU (legacy but still asked)
- **Transformers**: self-attention, multi-head attention, positional encodings, encoder/decoder
- Vision transformers; CLIP-style multimodal models
- Graph neural networks (for fraud rings, recommendations)

### F3. Training at scale

- Mixed precision, gradient accumulation, gradient checkpointing
- Distributed training: data parallel, DDP, FSDP/ZeRO, model parallel
- Experiment tracking: **Weights & Biases**, MLflow
- Hyperparameter search: Optuna, Ray Tune

### F4. Inference & optimisation

- Quantisation (INT8, 4-bit), pruning, distillation
- ONNX, TensorRT, **NCNN** (named in the GoTo KYC posting), TorchScript
- Batching, caching, latency budgets, GPU utilisation

### F5. PyTorch craft

- `Dataset`/`DataLoader`, collate functions, samplers
- Custom modules, hooks, `torch.compile`
- Debugging NaNs, exploding gradients, silent shape bugs

**The bar:** you have trained something non-trivial from scratch, and you can say what you would change to halve the inference cost.

---

## G. LLM & Generative AI Systems

*The single highest-leverage block for India.*

### G1. Foundations

- Tokenisation (BPE, SentencePiece), context windows, attention cost
- Pretraining vs instruction tuning vs preference tuning (RLHF, DPO)
- Decoding: temperature, top-k, top-p, beam search, structured/constrained decoding
- Scaling laws, emergent capability claims and their critiques

### G2. Retrieval-Augmented Generation (RAG)

- Chunking strategies: fixed, semantic, recursive, hierarchical/parent-document
- Embedding models; choosing and benchmarking them
- **Vector databases**: FAISS, Pinecone, Weaviate, ChromaDB, pgvector, Qdrant (the SAP posting names four)
- Hybrid search: BM25 + dense; reciprocal rank fusion
- Rerankers (cross-encoders)
- Query transformation: HyDE, multi-query, step-back prompting
- Context assembly, citation grounding, hallucination mitigation
- **Retrieval evaluation as a separate problem from generation evaluation**

### G3. Fine-tuning & adaptation

- Full fine-tuning vs **PEFT**: LoRA, QLoRA, adapters, prefix tuning
- Instruction dataset construction and cleaning
- Synthetic data generation and its failure modes
- DPO/ORPO preference tuning
- When *not* to fine-tune (usually: try retrieval and prompting first)

### G4. Agents & orchestration

- Tool/function calling; schema design for tools
- ReAct, plan-and-execute, reflection patterns
- **Multi-agent systems** (named explicitly by SAP)
- Memory: short-term, long-term, episodic
- **Model Context Protocol (MCP)** and tool interoperability
- Frameworks: **LangChain, LlamaIndex, Semantic Kernel** (SAP names all three), LangGraph, DSPy, Haystack
- Failure containment: timeouts, retries, loop detection, cost caps

### G5. Evaluation — the part most people skip

- Offline: exact match, ROUGE/BLEU (weak), semantic similarity
- **LLM-as-judge**: rubric design, position bias, self-preference bias
- RAG-specific: faithfulness, answer relevance, context precision/recall (RAGAS, TruLens)
- Golden datasets; regression suites; eval-driven development
- Red-teaming: prompt injection, jailbreaks, data exfiltration
- Online: human feedback loops, A/B tests on LLM features

### G6. LLMOps & serving

- Prompt versioning and management
- Caching: exact, semantic, prefix/KV cache
- Serving: vLLM, TGI, SGLang; continuous batching, PagedAttention
- Cost and latency observability; token accounting
- Guardrails: input/output filtering, PII redaction, policy enforcement
- **Observability** (SAP names LLMOps and observability specifically): Langfuse, Phoenix, LangSmith

### G7. Responsible AI

- Bias evaluation, toxicity, safety benchmarks
- Model cards, data sheets, audit trails
- **EU AI Act** obligations for high-risk systems (Europe-critical)
- Governance frameworks: NIST AI RMF, ISO 42001

**The bar:** you have shipped an LLM feature that has an eval suite, a cost ceiling, and a rollback plan — not a demo.

---

## H. Recommender & Ranking Systems

### H1. Retrieval (candidate generation)

- Collaborative filtering: user-based, item-based, ALS, BPR
- **Two-tower / dual-encoder** models
- ANN search: HNSW, IVF-PQ, ScaNN
- Graph-based retrieval; session-based retrieval

### H2. Ranking

- Learning-to-rank: pointwise, pairwise (RankNet), listwise (LambdaMART)
- CTR/CVR prediction: Wide & Deep, DeepFM, DCN, DIN
- Multi-task and multi-objective ranking (engagement vs revenue vs satisfaction)
- Calibration for auctions

### H3. Ads-specific

- **Auction theory**: second-price, GSP, VCG; reserve prices (named by Grab)
- Bid optimisation and budget pacing
- Incrementality measurement; **uplift-based targeting**

### H4. System concerns

- Cold start (user, item, both)
- Position, popularity and selection bias; inverse propensity weighting
- Diversity, serendipity, filter bubbles
- Feedback loops and model collapse
- Real-time feature freshness; latency budgets under 100ms

**The bar:** you can draw the retrieval → ranking → re-ranking pipeline on a whiteboard and defend every latency and quality trade-off in it.

---

## I. MLOps & Production Ownership

*The only block that pays in every market.*

### I1. Packaging & deployment

- Docker: multi-stage builds, slim images, reproducibility
- **Kubernetes**: deployments, services, HPA, resource limits, jobs
- Model serving: FastAPI, BentoML, Seldon, KServe, TorchServe, Triton
- Batch vs real-time vs streaming inference; shadow deployment; canary; blue-green

### I2. Pipelines & orchestration

- **Airflow**, Prefect, Dagster, Kubeflow Pipelines, Metaflow
- DAG design, idempotency, retries, backfills, SLAs
- Data validation in-pipeline: Great Expectations, Pandera, Soda

### I3. Tracking & registry

- **MLflow**: experiments, model registry, stage transitions
- Weights & Biases; DVC for data versioning
- Reproducibility: seeds, environment pinning, lineage

### I4. Monitoring

- Data drift (PSI, KL divergence, KS test), concept drift
- Prediction drift and performance decay with delayed labels
- **Evidently AI**, WhyLabs, Arize, Prometheus + Grafana
- Alerting thresholds that do not cry wolf
- Automated retraining triggers

### I5. Infrastructure literacy

- Cloud: **AWS SageMaker / GCP Vertex AI / Azure ML** — pick one and go deep
- IaC: Terraform basics
- Cost management: spot instances, autoscaling, GPU scheduling
- Secrets management, IAM, least privilege

### I6. Streaming & real-time

- **Kafka**: topics, partitions, consumer groups, exactly-once semantics
- **Flink** / Spark Structured Streaming (Flink named by GoTo)
- Feature stores: Feast, Tecton — online/offline parity, point-in-time joins
- Late-arriving data and watermarks

**The bar:** your model is live, you get alerted when it degrades, and you can roll it back in one command.

---

## J. Data Engineering & Big Data

- **Spark**: RDD vs DataFrame, catalyst optimiser, partitioning, shuffle, skew, broadcast joins, AQE, UDF cost (named by Grab and SAP)
- **Databricks** and the lakehouse pattern; Delta Lake, time travel
- Table formats: Delta, **Iceberg**, Hudi
- Warehouses: Snowflake, BigQuery, Redshift — clustering, micro-partitions, cost control
- File formats: Parquet, ORC, Avro; columnar layout and compression
- **dbt** for transformation (named by GoTo)
- Ingestion: CDC (Debezium), Fivetran, Airbyte
- Data quality, lineage and cataloguing: OpenLineage, DataHub, Amundsen

**The bar:** a 2 TB job that used to take 6 hours takes 40 minutes after your change, and you can explain why.

---

## K. ML System Design

*The round most candidates fail at staff level.*

### K1. The framework

 1. Clarify the business objective and the ML objective — and note where they diverge
 2. Frame as an ML problem; state the alternative framings you rejected
 3. Define metrics: offline proxy, online business, guardrails
 4. Data: sources, labels, volume, freshness, privacy
 5. Features: online/offline parity, leakage risk
 6. Model: baseline first, then complexity, with justification
 7. Training: cadence, compute, validation strategy
 8. Serving: latency budget, throughput, infrastructure
 9. Evaluation: offline → shadow → A/B → rollout
10. Monitoring, failure modes, and the rollback plan
11. Scale, cost, and what breaks at 10×

### K2. Canonical problems to be able to design cold

- Feed ranking · Search ranking · Product recommendations
- Fraud detection · Credit underwriting
- Dynamic pricing · Surge/ETA prediction · Driver-rider matching
- Ad click prediction and bidding
- Content moderation · Spam detection
- Demand forecasting · Inventory allocation
- RAG assistant over enterprise documents
- Churn prediction and retention intervention targeting

**The bar:** you talk for 45 minutes, the interviewer interrupts with constraints, and your design bends without breaking.

---

## L. Product Sense, Communication & Leadership

*Named in \~70% of senior postings. The reason two identical stacks get offers 20 LPA apart.*

### L1. Product sense

- Metric definition: choosing the OEC, avoiding gameable metrics
- North-star metric and its input tree
- Trade-off reasoning: engagement vs retention vs revenue vs trust
- Diagnosing a metric drop: segment, seasonality, instrumentation, external
- Sizing an opportunity before building anything

### L2. Communication

- Structuring an answer top-down (answer first, then support)
- Written memos: the six-pager / one-pager discipline
- Visualisation that encodes the argument, not the data dump
- Translating statistical uncertainty into decision language
- **Executive-level communication** (Meta and SAP both name this explicitly)
- Stakeholder management and saying no with a reason

### L3. Scope & ownership

- Owning a metric rather than a ticket queue
- Writing the roadmap for your area
- Influence without authority across Product and Engineering
- Knowing when the answer is "we should not build this"

### L4. Leadership

- Mentoring and growing junior analysts
- Design and code review as a teaching act
- Hiring: writing the loop, calibrating the bar
- Documentation as leverage

**The bar:** a VP reads your one-pager and makes the decision without a follow-up meeting.

---

# PART III — DOMAIN SKILL STACKS

Domain depth is what separates the ₹35 LPA generalist from the ₹60 LPA specialist. Pick **one** primary domain. The three below cover the majority of roles at this band.

---

## DOMAIN 1 — FINTECH

Highest-paying domain in the US, UK and India. Regulation is the moat: it takes longer to learn, so fewer people have it.

### 1.1 Credit risk & underwriting

**Modelling**

- Application scorecards (new customers) vs behavioural scorecards (existing)
- **Weight of Evidence (WOE) binning** and **Information Value (IV)** — still the industry standard for feature prep
- Logistic regression scorecards — *preferred over GBDTs in regulated lending because they are explainable*
- Reject inference (parcelling, augmentation) — modelling applicants you declined
- PD, LGD, EAD; expected loss = PD × LGD × EAD
- IFRS 9 / CECL expected credit loss staging and lifetime PD
- Vintage analysis and cohort loss curves
- Survival analysis for time-to-default
- Champion-challenger frameworks

**Metrics**

- KS statistic, Gini, ROC-AUC, **PSI** (population stability index) for drift
- Approval rate vs bad rate trade-off curves
- Swap-set analysis when replacing a model

**Alternative data (critical in India and Indonesia)**

- Bureau data: CIBIL, Experian, Equifax (India) · FICO, VantageScore (US)
- Cashflow underwriting from bank statements; account aggregator frameworks
- Device, telco, utility, psychometric and e-commerce signals for thin-file borrowers
- UPI transaction behaviour (India-specific)

### 1.2 Fraud & financial crime

- Rules engines + ML hybrids; why pure ML rarely wins alone
- **Graph analytics for fraud rings** — community detection, GNNs, shared-attribute linking
- Velocity features and time-window aggregations
- Device fingerprinting, behavioural biometrics
- Account takeover vs new-account fraud vs first-party fraud vs synthetic identity
- AML: transaction monitoring, sanctions screening, false-positive reduction
- Extreme class imbalance (fraud rates of 0.01–0.5%)
- Real-time scoring under 100ms; feature freshness
- Adversarial drift — your counterparty adapts to your model

### 1.3 Regulation & governance — the real differentiator

- **SR 11-7** (US Federal Reserve) model risk management: development documentation, **independent validation**, ongoing monitoring — named in US bank and larger fintech postings
- **ECOA / Regulation B** — adverse action reasons; you must be able to say *why* someone was declined
- **Fair lending**: disparate impact testing, proxy discrimination, protected class analysis
- **FCRA** for credit reporting data
- **Basel III/IV** capital models (banks)
- **EU AI Act** — credit scoring is explicitly a high-risk system
- **RBI digital lending guidelines** (India); **MAS FEAT principles** (Singapore); **FCA/PRA** (UK)
- Explainability: SHAP, LIME, counterfactual explanations, monotonic constraints in GBDTs

### 1.4 Other fintech DS areas

- Collections optimisation and treatment assignment (a perfect uplift-modelling problem)
- Pricing and risk-based interest rates
- LTV and CAC modelling for lending books
- Payments: authorisation rate optimisation, routing, decline recovery
- Trading/quant (a separate discipline — stochastic calculus, market microstructure, backtesting rigour)

### 1.5 Fintech company targets

India: PhonePe, Razorpay, CRED, Navi, Groww, Jupiter, Slice, Paytm, Zerodha Singapore: Stripe SG, Airwallex, Nium, Thunes, DBS, GIC Indonesia: Xendit, DANA, Kredivo, Amartha, Akulaku USA: Stripe, Block, Capital One, SoFi, Chime, Affirm, Plaid, Ramp, Brex Europe: Revolut, Monzo, Starling, Wise, Adyen, Klarna, N26, Trade Republic, Checkout.com

---

## DOMAIN 2 — E-COMMERCE & MARKETPLACE

The largest volume of roles at this band across Asia and Europe.

### 2.1 Search, ranking & recommendations

- Query understanding: intent classification, spell correction, query expansion, synonyms
- Retrieval → ranking → re-ranking architecture
- Personalisation vs popularity; exploration vs exploitation
- Learning-to-rank on implicit feedback (clicks, add-to-cart, purchase)
- Position bias correction and counterfactual learning-to-rank
- Multi-objective ranking: relevance vs margin vs seller fairness vs delivery speed
- Complementary vs substitute products; basket analysis
- Cold-start for new sellers and new SKUs

### 2.2 Pricing & promotions

- Price elasticity estimation from observational data (endogeneity is the hard part)
- Dynamic and personalised pricing; fairness and regulatory constraints
- Markdown optimisation and inventory clearance
- **Promotion incrementality** — how much of the discounted sale would have happened anyway
- **Uplift modelling for coupon and voucher targeting** — the canonical marketplace uplift problem
- Competitor price monitoring and response
- Surge pricing and supply-demand balancing (for delivery/ride marketplaces)

### 2.3 Supply chain & operations

- Demand forecasting: hierarchical, intermittent, new-product, promotional lift
- Inventory: safety stock, reorder points, newsvendor problem
- Assortment and space optimisation
- Warehouse slotting, pick-path optimisation
- Last-mile routing (VRP), batching, ETA prediction
- Capacity planning and rider/driver supply forecasting

### 2.4 Growth & marketing analytics

- Marketing mix modelling (MMM) and its revival post-ATT
- Multi-touch attribution and its known biases
- **Geo-experiments** and holdout tests for channel incrementality
- LTV prediction: BG/NBD, Gamma-Gamma, DNN-based
- Churn and win-back targeting
- Referral and network-effect measurement

### 2.5 Marketplace economics

- Two-sided marketplace dynamics; which side is constrained
- Liquidity, match rate, fill rate
- Network effects and **interference in experiments** — why switchback tests exist
- Take-rate optimisation
- Seller quality scoring and trust & safety
- Cannibalisation across channels and categories

### 2.6 E-commerce company targets

India: Flipkart, Meesho, Myntra, Nykaa, Swiggy, Zomato, Zepto, Blinkit SEA: Shopee, Lazada, Tokopedia, Grab, GoTo, Traveloka, Bukalapak USA: Amazon, Instacart, DoorDash, Wayfair, eBay, Etsy, Shopify, Chewy Europe: Zalando, Booking.com, Delivery Hero, Ocado, ASOS, Picnic, Coolblue, Deliveroo

---

## DOMAIN 3 — HEALTHCARE & LIFE SCIENCES

Highest barriers, least competition, strongest position in the USA and Switzerland.

### 3.1 Healthcare data literacy — learn this before any modelling

- **Coding systems: ICD-9/10/11 (diagnoses), CPT/HCPCS (procedures), RxNorm (drugs), LOINC (labs), SNOMED CT (clinical terms)** — all named in the Medeloop posting
- **Claims data** vs **EHR data** vs registry vs wearable — what each can and cannot answer
- **FHIR** and HL7 interoperability standards
- **OMOP Common Data Model** and the OHDSI ecosystem — the standard for multi-source observational research
- Data quality in healthcare: missing not at random, coding drift, site heterogeneity
- Patient journey and episode-of-care construction

### 3.2 Biostatistics & epidemiology

- Study designs: RCT, cohort, case-control, cross-sectional, nested case-control
- **Survival analysis**: Kaplan-Meier, log-rank, Cox proportional hazards, competing risks, time-varying covariates
- **Causal inference on observational data**: propensity scores, IPTW, target trial emulation, marginal structural models
- Confounding by indication, immortal time bias, selection bias — the three classic healthcare traps
- Sample size and power for clinical endpoints
- Missing data: MICE, multiple imputation, sensitivity analysis
- Meta-analysis and evidence synthesis

### 3.3 Real-World Data / Real-World Evidence (RWD/RWE)

- Cohort definition and phenotyping algorithms
- Comparative effectiveness research
- Treatment pattern and line-of-therapy analysis
- External control arms for single-arm trials
- Regulatory-grade evidence: FDA RWE framework, EMA guidance
- Publication: STROBE, RECORD reporting standards — *publication experience is explicitly requested in RWD roles*

### 3.4 Clinical ML

- Risk prediction: readmission, sepsis, deterioration, no-show
- **Clinical NLP**: de-identification, concept extraction, negation detection, clinical BERT models, LLMs on clinical notes
- Medical imaging: segmentation, classification, DICOM handling
- Genomics and multi-omics (a specialisation of its own)
- Model fairness across demographic groups — documented harms here are severe
- **Calibration matters more than discrimination** in clinical deployment
- Prospective validation and clinical trial of the algorithm itself

### 3.5 Regulation & compliance

- **HIPAA** (US): PHI, de-identification (Safe Harbor vs Expert Determination), BAAs
- **GDPR** special category data (EU); **DPDP Act** (India)
- **FDA SaMD** (Software as a Medical Device) and the Predetermined Change Control Plan
- **EU MDR** and AI Act high-risk classification for clinical decision support
- IRB/ethics approval processes
- Payer vs provider vs pharma — three different buyers with three different questions

### 3.6 Healthcare company targets

USA: Tempus, Flatiron Health, Verily, Komodo Health, Oscar Health, Devoted Health, Cedar, Medeloop, Epic, UnitedHealth/Optum Switzerland: Roche, Novartis, J&J Denmark: Novo Nordisk UK: Genomics England, BenevolentAI, NHS analytics India: Practo, HealthifyMe, Qure.ai, Niramai, pharma GCCs (Novartis, Roche, Pfizer Hyderabad/Bangalore)

---

# PART IV — THE LEARNING & PRACTICE PLAN

---

## Step 0 — Three decisions before you start (do this in one sitting)

You cannot learn Part II. Nobody has all of it. The plan only works if you narrow first.

**Decision 1 — Pick your market column**.Look at the cross-market table in Part I §11. Pick **one primary** and one secondary. India-primary means GenAI-heavy. Singapore/Netherlands/US-primary means causal-inference-heavy. This changes \~30% of your plan.

**Decision 2 — Pick your domain**.Fintech, e-commerce, or healthcare. Pick the one closest to your current work — domain credibility is built from shipped work, not courses.

**Decision 3 — Pick your track.**

- **Product / Analytics DS** → heavy on SQL, statistics, experimentation, product sense
- **ML Engineer / Applied Scientist** → heavy on coding, modelling, MLOps, system design
- **Applied / GenAI Scientist** → heavy on deep learning, LLM systems, evaluation

Then run a baseline audit — score yourself honestly on every Part II block (0 = none, 1 = aware, 2 = applied, 3 = shipped and owned). Anything below 2 in your chosen column is a gap. Anything at 3 is your story.

---

## Step 1 — The 12-month architecture

| Phase | Months | Focus | Exit criterion |
| --- | --- | --- | --- |
| **Phase 1 — Repair the floor** | 1–3 | SQL, statistics, Python engineering, DSA | You clear any screening round without preparation |
| **Phase 2 — Build the differentiator** | 4–7 | Your column's critical skill (causal inference *or* LLM systems) + MLOps | One production-grade project, publicly visible |
| **Phase 3 — Add depth & scope** | 8–10 | Domain specialisation + ML system design + a second project | You can design any system in your domain cold |
| **Phase 4 — Convert** | 11–12 | Mocks, applications, referrals, negotiation | Offers |

**Total time budget: 10–14 hours per week.** More than 20 and you will burn out around week 9; fewer than 8 and the compounding never starts. If you have a full-time job, protect two weekday evenings and one weekend block.

**Ratio to hold throughout: 40% building, 40% practising, 20% reading.** Most people invert it — 80% courses, 20% doing — and plateau.

---

## Step 2 — Weekly practice quotas

These are *floors*, not ceilings. Consistency beats volume.

### Phase 1 (months 1–3) — \~12 hrs/week

| Topic | Volume per week | Platform | Time |
| --- | --- | --- | --- |
| SQL problems | **6** (2 easy, 3 medium, 1 hard) | StrataScratch, DataLemur | 2.5 hrs |
| DSA problems | **4** (LeetCode medium) | NeetCode 150, LeetCode | 2.5 hrs |
| Statistics & probability | **5** questions | Interview Query, Ace the Data Science Interview | 1.5 hrs |
| Python refactoring | **1** module rewritten to production standard | your own old code | 2 hrs |
| Reading | **2** engineering blog posts | company eng blogs | 1 hr |
| Project | ongoing | — | 2.5 hrs |

### Phase 2 (months 4–7) — \~14 hrs/week

| Topic | Volume per week | Platform | Time |
| --- | --- | --- | --- |
| SQL (maintenance) | **3** hard problems | StrataScratch | 1 hr |
| DSA (maintenance) | **2** problems | LeetCode | 1 hr |
| **Differentiator** (causal inference *or* LLM) | **1 chapter + 1 notebook implemented** | see §3 below | 4 hrs |
| Experiment / eval design cases | **2** | Interview Query, Exponent | 1.5 hrs |
| MLOps Zoomcamp module | **1** module | DataTalksClub | 2.5 hrs |
| **Project build** | ongoing | — | 4 hrs |

### Phase 3 (months 8–10) — \~14 hrs/week

| Topic | Volume per week | Platform | Time |
| --- | --- | --- | --- |
| **ML system design** | **1** full design, written up | Chip Huyen's repo, Exponent | 3 hrs |
| Product sense / case studies | **2** cases | Exponent, Interview Query, DataInterview | 2 hrs |
| Domain reading (papers, regulation) | **2** papers or 1 regulation doc | arXiv, OHDSI, regulator sites | 2 hrs |
| SQL + DSA (maintenance) | **3** + **2** | — | 2 hrs |
| **Project 2 build** | ongoing | — | 5 hrs |

### Phase 4 (months 11–12) — \~12 hrs/week

| Activity | Volume per week |
| --- | --- |
| **Mock interviews** | **2** (1 technical, 1 behavioural/case) |
| Applications | **8–12** targeted, with a tailored resume line each |
| Referral outreach | **5** genuine conversations |
| Behavioural story polishing | **3** STAR stories written and timed |
| Maintenance drills | 3 SQL + 2 DSA + 1 case |

---

## Step 3 — Skill-by-skill: what to learn, where, and how to practise

### SQL

|  |  |
| --- | --- |
| **Learn** | Mode Analytics SQL Tutorial (free) → *SQL for Data Analysis* (Cathy Tanimura) for window functions and cohort patterns |
| **Practise** | [StrataScratch](https://platform.stratascratch.com/coding) — real company questions, best for analytics depth · [DataLemur](https://datalemur.com/) — free tier, excellent explanations · [LeetCode Database](https://leetcode.com/problemset/database/) — narrower but good for joins/windows |
| **Method** | Solve without running. Write the query, predict the output, *then* run. This is the only way to build the muscle an interview tests. |
| **Weekly** | 6 in Phase 1, dropping to 3 for maintenance |
| **Done when** | You can write a correct window-function query with a self-join in under 8 minutes, talking through it |

### Statistics & probability

|  |  |
| --- | --- |
| **Learn** | *Statistical Rethinking* (McElreath — [free lectures on YouTube](https://github.com/rmcelreath/stat_rethinking_2024)) for intuition · *All of Statistics* (Wasserman) for rigour · StatQuest for gaps |
| **Practise** | [Interview Query](https://www.interviewquery.com/) · *Ace the Data Science Interview* (Nick Singh) probability chapters · Brainstellar puzzles |
| **Method** | For every result, write the one-sentence intuition *and* the formal statement. Interviewers probe the gap between them. |
| **Weekly** | 5 questions |
| **Done when** | You can derive a power calculation on a whiteboard and explain what a confidence interval is *not* |

### Causal inference & experimentation — *the Singapore/US/NL differentiator*

|  |  |
| --- | --- |
| **Learn** | [**Causal Inference: The Mixtape**](https://mixtape.scunning.com/) (free online, Scott Cunningham) — start here · [**Python Causality Handbook**](https://matheusfacure.github.io/python-causality-handbook/) (free, code-first) · [**Brady Neal's Causal Inference course**](https://www.bradyneal.com/causal-inference-course) (free video) · ***Trustworthy Online Controlled Experiments*** (Kohavi, Tang, Xu) — **the single most valuable book for this band** · [Hernán & Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook) (free PDF) for the epidemiology side |
| **Practise** | Re-implement every method on real data: DiD on a policy change, RDD on a threshold, uplift models on the [Criteo uplift dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), CUPED on any A/B log · Read and critique published experiment write-ups from Netflix, Booking.com, Airbnb, Spotify |
| **Method** | For each method: implement from scratch once, then with a library (`EconML`, `DoWhy`, `CausalML`), then write 400 words on when it breaks. |
| **Weekly** | 1 chapter + 1 implemented notebook |
| **Done when** | Given a product question, you can propose the design, the MDE, the variance-reduction strategy, and the thing that would invalidate it |

### LLM & GenAI systems — *the India differentiator*

|  |  |
| --- | --- |
| **Learn** | [**LLM Zoomcamp (DataTalksClub)**](https://github.com/DataTalksClub/llm-zoomcamp) — free, project-based, the best structured start · [Hugging Face LLM Course](https://huggingface.co/learn) · [DeepLearning.AI short courses](https://www.deeplearning.ai/short-courses/) — RAG, agents, evals, fine-tuning · Chip Huyen, *AI Engineering* |
| **Practise** | Build a RAG system on a corpus you actually care about → **then build its eval suite** → then improve retrieval measurably. The eval suite is the part that gets you hired. · Fine-tune a small open model with LoRA on a real task · Build one agent with tool-calling and make it fail gracefully |
| **Method** | Never ship a demo without: a golden eval set, a cost-per-query number, and a documented failure mode. That trio is what a Principal DS posting means by "LLMOps". |
| **Weekly** | 1 module + 4 hrs building |
| **Done when** | You can answer "how do you know your RAG got better?" with numbers, not adjectives |

### MLOps & production — *pays in every market*

|  |  |
| --- | --- |
| **Learn** | [**MLOps Zoomcamp (DataTalksClub)**](https://github.com/DataTalksClub/mlops-zoomcamp) — free, the standard recommendation · [**Made With ML**](https://madewithml.com/) (Goku Mohandas) — free, production-focused · Chip Huyen, *Designing Machine Learning Systems* |
| **Practise** | Take *one* model you have already built and put it fully into production: Docker → FastAPI → CI/CD → MLflow registry → Airflow retraining DAG → Evidently drift monitoring → alert. Do this once, end to end, and you have crossed the line this whole document is about. |
| **Method** | Deploy to a cheap cloud VM or free tier. It must survive a week and you must be able to show the monitoring dashboard. |
| **Weekly** | 1 Zoomcamp module + build time |
| **Done when** | You can demo a live endpoint, show its drift dashboard, and roll back a version while someone watches |

### Machine learning depth

|  |  |
| --- | --- |
| **Learn** | [ML Zoomcamp (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp) for breadth · *The Elements of Statistical Learning* for depth · [fast.ai](https://course.fast.ai/) for deep learning intuition · [Papers with Code](https://paperswithcode.com/) for the current state of a sub-field |
| **Practise** | [Kaggle](https://www.kaggle.com/competitions) — but **read the winning solution write-ups more than you compete**; that is where the technique is · [DrivenData](https://www.drivendata.org/competitions/) for social-impact problems with cleaner framing |
| **Method** | For every Kaggle competition you touch, write a one-page post-mortem: what the winner did that you did not think of. |
| **Weekly** | Folded into project time |

### Data engineering & big data

|  |  |
| --- | --- |
| **Learn** | [Data Engineering Zoomcamp (DataTalksClub)](https://github.com/DataTalksClub/data-engineering-zoomcamp) — free · [dbt Learn](https://learn.getdbt.com/) — free official courses · *Designing Data-Intensive Applications* (Kleppmann) — the book that makes system design rounds easy |
| **Practise** | Build one pipeline: ingest → dbt transform with tests → warehouse → orchestrated by Airflow. Then deliberately break it and fix the backfill. |
| **Weekly** | 1 module in Phase 2 |

### ML system design

|  |  |
| --- | --- |
| **Learn** | [Chip Huyen's ML systems design repo](https://github.com/chiphuyen/machine-learning-systems-design) (free) · [Eugene Yan's applied-ml repo](https://github.com/eugeneyan/applied-ml) — real company write-ups by problem type · *Designing Machine Learning Systems* · Grokking the ML System Design Interview (Educative, paid) |
| **Practise** | One design per week, **written up as a document**, using the 11-step framework in Part II §K1. Then find how a real company solved it (their eng blog) and diff your answer against theirs. |
| **Weekly** | 1 full design |
| **Done when** | You can hold a 45-minute design conversation and adapt when the interviewer adds a constraint |

### Product sense & communication

|  |  |
| --- | --- |
| **Learn** | *Decode and Conquer* (Lewis Lin) for structure · Company eng blogs for metric thinking · Amazon's six-pager format |
| **Practise** | [Exponent](https://www.tryexponent.com/) case library · [Interview Query](https://www.interviewquery.com/) product cases · [DataInterview](https://www.datainterview.com/) guides |
| **Method** | Record yourself answering. Watch it back at 1.5×. Painful, and the fastest improvement available. |
| **Weekly** | 2 cases, one recorded |

### Company engineering blogs — read these weekly, free, and better than most courses

[Netflix Tech Blog](https://netflixtechblog.com/) · [Uber Engineering](https://www.uber.com/blog/engineering/) · [Airbnb Tech](https://medium.com/airbnb-engineering) · [DoorDash Engineering](https://careersatdoordash.com/engineering-blog/) · [Meta Engineering](https://engineering.fb.com/) · [Grab Tech](https://engineering.grab.com/) · [Gojek Engineering](https://www.gojek.io/blog/) · [Booking.com Data Science](https://booking.ai/) · [Spotify R&D](https://engineering.atspotify.com/) · [Zalando Engineering](https://engineering.zalando.com/) · [Pinterest Engineering](https://medium.com/pinterest-engineering) · [LinkedIn Engineering](https://www.linkedin.com/blog/engineering)

---

## Step 4 — Projects: where to find them, and what to build

### Why most portfolios fail

Titanic, Iris, and a churn notebook signal nothing. A hiring manager is looking for **evidence you have operated a system**, not evidence you can call `.fit()`.

### What a portfolio project must have at this band

1. **A real, messy data source** — not a curated CSV
2. **A decision it informs** — stated explicitly
3. **An evaluation that could have failed** — and a documented baseline it beat
4. **Production deployment** — live endpoint, or scheduled pipeline, or both
5. **Monitoring** — a dashboard someone else could read
6. **A written README** that explains the trade-offs, not the code

Three projects with all six beat fifteen notebooks.

### Where to find data and problem ideas

| Source | Link | Best for |
| --- | --- | --- |
| Kaggle Datasets | [kaggle.com/datasets](https://www.kaggle.com/datasets) | Breadth, quick starts |
| Google Dataset Search | [datasetsearch.research.google.com](https://datasetsearch.research.google.com/) | Finding anything specific |
| Hugging Face Datasets | [huggingface.co/datasets](https://huggingface.co/datasets) | NLP, LLM, multimodal |
| data.gov.in | [data.gov.in](https://data.gov.in/) | India public data |
| EU Open Data Portal | [data.europa.eu](https://data.europa.eu/) | Europe |
| data.gov | [data.gov](https://data.gov/) | US federal |
| DrivenData | [drivendata.org](https://www.drivendata.org/competitions/) | Well-framed real problems |
| Awesome Public Datasets | [github.com/awesomedata/awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets) | Curated index |
| **Public APIs** | [github.com/public-apis/public-apis](https://github.com/public-apis/public-apis) | **Building your own dataset — the strongest signal** |
| OHDSI / SynPUF | [ohdsi.org](https://www.ohdsi.org/) | Healthcare, OMOP-standard |
| Criteo Uplift | [ailab.criteo.com](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | Causal / uplift modelling |
| Lending Club / Home Credit | Kaggle | Credit risk |
| IEEE-CIS Fraud | Kaggle | Fraud detection |
| MovieLens / Amazon Reviews | [grouplens.org](https://grouplens.org/datasets/movielens/) | Recommenders |
| M5 / M4 Forecasting | Kaggle | Demand forecasting |

### Project blueprints by domain — build two, properly

**Fintech**

1. **Credit scorecard with governance** — WOE binning, logistic scorecard, KS/Gini/PSI, reject inference, **an SR 11-7-style model documentation pack**, adverse-action reason codes, fairness testing across proxy groups. *The documentation is the differentiator, not the model.*
2. **Real-time fraud pipeline** — Kafka stream → velocity features in a feature store → sub-100ms scoring endpoint → graph-based ring detection → drift monitoring. Simulate adversarial drift and show the model degrading.

**E-commerce / marketplace**3. **Two-stage recommender** — two-tower retrieval + HNSW ANN + LambdaMART re-ranking, NDCG@K evaluation, position-bias correction, served under a latency budget. 4. **Promotion uplift engine** — train S/T/X-learners and a causal forest on Criteo, evaluate with Qini curves, and produce a targeting policy with an estimated incremental profit number. 5. **Experimentation platform (mini)** — assignment service, exposure logging, SRM check, CUPED variance reduction, sequential-testing guardrail, and an auto-generated results page.

**Healthcare**6. **OMOP cohort study** — build a phenotype, emulate a target trial, run propensity-matched survival analysis with competing risks, and write it up to STROBE reporting standards. *This is publishable-shaped work and almost nobody has it in a portfolio*.7. **Clinical NLP pipeline** — de-identification → concept extraction against RxNorm/ICD → negation detection → structured output, with a held-out annotated eval set.

**GenAI (any domain**)8. **Domain RAG assistant with a real eval harness** — hybrid retrieval, reranking, citation grounding, RAGAS-style faithfulness/context-recall metrics, cost-per-query tracking, prompt-injection red-team suite, and a documented A/B comparison of two retrieval strategies. 9. **Agentic workflow with guardrails** — tool-calling agent for a genuine multi-step task, with timeouts, loop detection, cost ceilings, and a trace viewer (Langfuse or Phoenix).

### Contributing to open source — the underrated route

Pick one library you actually use — `scikit-learn`, `MLflow`, `Evidently`, `DoWhy`, `EconML`, `LangChain`, `dbt` — and start with documentation fixes, then bug fixes, then features. Named in the GoTo posting as a preferred qualification ("open-source contributions or public work portfolios"). It is also the cheapest way to get a reference from someone senior outside your company.

---

## Step 5 — The practice protocol (how to practise, per skill type)

Different skills need different practice. Using one method for all of them is why people plateau.

### Type 1 — Recall skills (SQL syntax, statistical definitions, algorithm patterns)

**Method: spaced repetition + timed retrieval**

- Keep an Anki deck or a plain markdown file of everything you got wrong
- Review before learning anything new — 15 minutes at the start of every session
- Every fourth session, do a **timed cold retrieval**: no notes, no autocomplete, clock running
- Anything you fail twice goes to the top of the deck

### Type 2 — Problem-solving skills (DSA, SQL problems, probability)

**Method: attempt → fail → study → re-solve cold**

1. Attempt for 25 minutes. Do not look anything up.
2. If stuck, read *one hint*, not the solution.
3. If still stuck, read the solution and **write down the single insight you were missing**.
4. Close everything. Re-solve from scratch.
5. Log it. **Re-solve it again 7 days later.** If you cannot, it never landed.

The re-solve is where the learning happens. Skipping it is the most common wasted effort in interview prep.

### Type 3 — Design skills (ML system design, experiment design)

**Method: write, then diff against reality**

1. Pick a problem. Set a 45-minute timer. Write the full design as a document.
2. Find how a real company solved it — their engineering blog, a paper, a conference talk.
3. **Diff your design against theirs.** List every consideration they had that you missed.
4. Rewrite your design incorporating those.
5. Two weeks later, design a *different* problem and check whether the missed considerations now appear unprompted.

### Type 4 — Building skills (MLOps, pipelines, LLM systems)

**Method: build → break → fix → document**

- Build the thing end to end, badly, fast. Working beats elegant.
- Then deliberately break it: kill the database, feed it malformed input, 10× the traffic, corrupt the schema.
- Fix each failure and **write down the failure mode**. That list is your interview material.
- Refactor only after it has survived being broken.

### Type 5 — Communication skills (product sense, behavioural, storytelling)

**Method: record, watch, rewrite**

- Answer out loud, recorded, with a timer. Never silently in your head.
- Watch it back. Count filler words. Note where you buried the answer.
- Rewrite the answer top-down: conclusion first, three supports, one caveat.
- Re-record. The second version is always 40% shorter and twice as clear.
- Target: **90 seconds** for a behavioural answer, **4 minutes** for a project walkthrough.

### The weekly review — 30 minutes, every Sunday

- What did I get wrong this week, and which deck did it go into?
- Which quota did I miss, and why?
- What did I build that did not exist last week?
- One thing to drop next week (there is always one)

---

## Step 6 — Becoming interview-ready (months 11–12)

### The readiness bar, honestly stated

You are ready when:

- SQL and DSA screens require no preparation
- You can talk for 45 minutes about a system you built, including what went wrong
- You have two projects with live URLs and readable READMEs
- You can design any canonical problem in your domain cold
- You have five STAR stories that each contain a number

### Mock interviews — two per week, non-negotiable

| Source | Type | Note |
| --- | --- | --- |
| [Pramp / Exponent peer mocks](https://www.tryexponent.com/practice) | Peer, free | Volume; quality varies |
| [interviewing.io](https://interviewing.io/) | Anonymous, with real engineers | Paid, highest signal |
| [IGotAnOffer](https://igotanoffer.com/) | Ex-FAANG coaches | Paid, good for behavioural |
| Colleagues and friends | Free | Best for domain-specific rounds |

Alternate: one technical, one case/behavioural, every week.

### Application strategy — target, do not spray

- **12 tier-1 targets** (your dream companies) — approach only via referral, after preparing
- **20 tier-2** (good companies, realistic) — apply directly, tailored
- **8 tier-3 practice interviews** — deliberately interview at places you would not join, *first*, to calibrate. Your first three interviews will be bad. Spend them where it does not matter.

### Referrals — the single highest-ROI activity

- Alumni are the warmest route. Search your university on LinkedIn, filtered by company.
- Message specifically: name the team, name the req, say in one line why you fit, attach nothing.
- Engage with people's work publicly (their blog posts, their open-source repos) for weeks *before* asking.
- Conference and meetup talks — giving one, even a small one, inverts the dynamic entirely.

### Your public surface — assume they will look

- **GitHub**: pinned repos with READMEs that explain trade-offs. Clean commit history.
- **LinkedIn**: headline names your specialisation, not your title. Experience bullets carry numbers.
- **Resume**: one page until 10 years. Every bullet is *action → method → quantified outcome*. Remove every skill you cannot be grilled on.
- **Writing**: 3–4 technical posts on your differentiator area. This is what makes recruiters approach you instead of the reverse.

### Negotiation

- Never give the first number. "I'd like to understand the band for this level first."
- Collect competing processes deliberately — run them to offer stage in the same 3-week window.
- At this band, negotiate **equity and level**, not base. A level bump is worth more than any base increase.
- Levels.fyi and Blind give you the company's actual band. Use them, but discount the self-reporting bias.

---

## Step 7 — Tracking

Keep one file. Review it every Sunday.

```
## Week 23
Quotas:  SQL 3/3 ✓ | DSA 2/2 ✓ | Causal 1/1 ✓ | MLOps 0/1 ✗ | Project 3.5/4 hrs
Wrong-answer log: 4 new entries (window frame clauses ×2, IV weak instruments ×2)
Built: drift monitoring wired to Slack alerts
Re-solved from week 22: 5/6
Read: Booking.com on SRM detection; Grab on switchback design
Next week: drop reading to 1 post, recover the MLOps module
```

Track four numbers monthly: **quota adherence %**, **re-solve success rate**, **projects shipped**, **mock interviews done**. If quota adherence is below 70% for two consecutive months, your plan is too big — cut it, do not push harder.

---

## Step 8 — Failure modes to avoid

| Failure | What it looks like | Fix |
| --- | --- | --- |
| **Course collecting** | 14 certificates, no deployed system | 40% of time must be building |
| **Preparing for two markets** | Half-ready for India and Singapore, ready for neither | Pick one column |
| **Breadth over depth** | Aware of everything, expert in nothing | Two skills at "shipped and owned" beat eight at "applied" |
| **Notebook portfolio** | `.ipynb` files with no README | Six-point checklist in Step 4 |
| **Skipping the re-solve** | Solved 300 problems, can't solve them again | Re-solve at day 7 |
| **No eval suite on LLM work** | A demo that impresses nobody senior | Golden set + cost + failure modes |
| **Applying before ready** | Burning tier-1 companies early | Practice interviews at tier-3 first |
| **Silent practice** | Never speaking answers out loud | Record everything |
| **Ignoring the visa** | Perfect prep, no route in | Research the visa before the market |
| **Chasing GenAI everywhere** | It's the India signal, not the global one | Match the skill to the column |

---

## Appendix A — Master resource list

**Free courses**

- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) · [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) · [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) · [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) — all from [DataTalks.Club](https://datatalks.club/)
- [Made With ML](https://madewithml.com/) · [fast.ai](https://course.fast.ai/) · [Hugging Face Learn](https://huggingface.co/learn)
- [Causal Inference: The Mixtape](https://mixtape.scunning.com/) · [Python Causality Handbook](https://matheusfacure.github.io/python-causality-handbook/) · [Brady Neal's course](https://www.bradyneal.com/causal-inference-course) · [Causal Inference: What If](https://miguelhernan.org/whatifbook)
- [dbt Learn](https://learn.getdbt.com/)

**Practice platforms**

- [StrataScratch](https://platform.stratascratch.com/coding) · [DataLemur](https://datalemur.com/) · [LeetCode](https://leetcode.com/) · [NeetCode](https://neetcode.io/) · [HackerRank](https://www.hackerrank.com/) · [Interview Query](https://www.interviewquery.com/) · [Exponent](https://www.tryexponent.com/) · [DataInterview](https://www.datainterview.com/)

**Competitions & data**

- [Kaggle](https://www.kaggle.com/) · [DrivenData](https://www.drivendata.org/) · [Papers with Code](https://paperswithcode.com/) · [Hugging Face Datasets](https://huggingface.co/datasets)

**Tooling to have used, not just read about**

- [MLflow](https://mlflow.org/) · [Weights & Biases](https://wandb.ai/) · [Evidently AI](https://www.evidentlyai.com/) · [DoWhy](https://github.com/py-why/dowhy) · [EconML](https://github.com/py-why/EconML) · [CausalML](https://github.com/uber/causalml) · [Optuna](https://optuna.org/) · [Feast](https://feast.dev/)

**Books worth the money, in priority order**

1. *Trustworthy Online Controlled Experiments* — Kohavi, Tang, Xu
2. *Designing Machine Learning Systems* — Chip Huyen
3. *Designing Data-Intensive Applications* — Martin Kleppmann
4. *Causal Inference: The Mixtape* — Scott Cunningham (also free online)
5. *AI Engineering* — Chip Huyen
6. *Ace the Data Science Interview* — Nick Singh & Kevin Huo
7. *Statistical Rethinking* — Richard McElreath
8. *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman (free PDF)

---

## Appendix B — Sources

Job postings read in full (September 2026):

- [Grab — Senior Data Scientist, Ads & Demand Optimization, Singapore](https://www.grab.careers/en/jobs/744000133998534/senior-data-scientist-ads-demand-optimization/)
- [GoTo Group — Senior Manager Data Scientist, Pricing](https://jobs.lever.co/GoToGroup/d8b85b16-1073-4c6a-a349-56ac7010d862)
- [GoTo Group — Senior Data Scientist, Jakarta](https://datasciencejobs.com/jobs/senior-data-scientist-goto-group-indonesia-1/)
- [GoTo Group — Senior Data Scientist, KYC, Jakarta](https://builtin.com/job/senior-data-scientist-kyc/7782914)
- [SAP — Principal Data Scientist / AI Application Development Expert, Bangalore](https://jobs.sap.com/job/Bangalore-Principal-Data-Scientist-AI-Application-Development-Expert-560066/1292489801/)
- [PhysicsWallah — Principal Data Scientist GenAI, Bangalore](https://www.instahyre.com/job-392932-principal-data-scientist-genai-at-physicswallah-bangalore/)
- [Medeloop — Healthcare Data Scientist (RWD)](https://freehire.me/jobs/healthcare-data-scientist-rwd-medeloop-ow62qppr)
- [Agoda — Lead/Staff Data Scientist, Bangkok](https://careersatagoda.com/job/5432621-lead-staff-data-scientist-bangkok-based-relocation-provided/)

Compensation and market data:

- [Data Scientist Salary Guide 2026: Pay by Level and City — KORE1](https://www.kore1.com/data-scientist-salary-guide/) (US)
- [AI Salaries in Europe 2026 — DigitalDefynd](https://digitaldefynd.com/IQ/ai-salaries-in-europe/)
- [What Employers Are Actually Paying Data Engineers and Data Scientists in Southeast Asia in 2026 — High Five](https://highfive.global/global-expansion/what-employers-are-actually-paying-data-engineers-and-data-scientists-in-southeast-asia-in-2026/)
- [Data Scientist Salary in India 2026 — Futurense](https://futurense.com/blog/data-scientist-salary-in-india)
- [Data Science Job Market in India 2026 — DataExpertise](https://www.dataexpertise.in/data-science-job-market-india-2026-salaries-roles-skills/)
- [AI Jobs in India 2026: Complete Hiring Guide — Shifttotech](https://shifttotech.co.in/blog/ai-ml-jobs-india-2025-complete-guide)
- [Data Scientist Salary in Singapore — Vertical Institute](https://verticalinstitute.com/blog/data-scientist-salary-singapore/)
- [Grab Data Scientist Salary — Levels.fyi](https://www.levels.fyi/companies/grab/salaries/data-scientist)
- [Meta Data Scientist Guide 2026 — DataInterview](https://www.datainterview.com/blog/meta-data-scientist-interview)
- [SR 11-7 Model Risk Management and AI](https://go.ai/blog/sr-11-7-model-risk-management-ai)

**Caveat on sample size.** Eight postings read in full is a small sample, and salary aggregators disagree with each other by wide margins. Company lists are compiled from market knowledge, not from verified current openings — verify any specific company's band on Levels.fyi or Glassdoor before you act on it.
