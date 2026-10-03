# Finding Sabah's Missing TB Cases: a detection-gap map and mobile-screening planner

**Track:** Public Health & Epidemiology · **Problem statement:** use data to help communities anticipate
and respond to health threats before they become crises.

**Problem.** Sabah holds ~10% of Malaysia's population but ~20% of its TB notifications (2012–2018).
58% of cases already show moderate or advanced disease on chest X-ray at diagnosis, which points to late
detection (Goroh et al. 2020). The lowest notification rates are in remote interior districts such as
Tongod (53/100k) and Kinabatangan (56/100k). A low number there may mean *missed* cases, not *absent*
ones. Each missed case keeps transmitting for months. The State Health Department has a limited number
of mobile screening units and no systematic way to decide where to send them.

**Innovation & concept.** Most TB dashboards map *notified* cases, which steers resources toward
districts that are already finding their cases. We instead estimate the **detection gap**: how many cases
each district is likely missing given its poverty, migrant share, remoteness and notification rate. We
then convert the gap into an **optimised deployment plan** (which district, how many unit-weeks, expected
cases found vs random allocation). We validate it in simulated worlds where the hidden cases are known.
Those tests also show *when the method fails*, which shapes our roadmap.

**Impact & relevance.** *Users:* Sabah State Health Department TB programme and NGO partners running
active case finding. *If it works:* more cases found per screening week, earlier diagnosis, shorter
infectious periods. In simulation, a 4-unit plan finds 2.4× more hidden cases than random allocation.
*Beneficiaries:* remote, poor and migrant communities with the least access to diagnosis.

**Feasibility & execution (built this weekend).**
- *Data:* 175 district-years digitised from the paper's district maps by colour-sampling against official
  boundaries. Statewide rate reproduces the paper within 1.3%. Combined with DOSM census, DOSM poverty and
  a hand-coded distance-to-specialist-hospital proxy.
- *Model:* negative-binomial GLM with a "good access" counterfactual, plus a transparent weighted index.
  A rule set in advance chose the index, because the access effect was not significant (p = 0.51, 25
  districts).
- *Validation:* synthetic recovery precision@5 = 0.79 (chance 0.20). We also report a 2017–18 backtest
  where we do *not* beat naive persistence.
- *Prototype:* a working Streamlit dashboard with a choropleth, adjustable planner (units, weeks,
  throughput) and forecast with intervals. Every view says whether the data is real or synthetic.

**Ethics & limitations.** Undocumented and stateless people are under-counted in both numerator and
denominator. District aggregates only, no individual data. Screening is delivered by health services and
NGOs with a strict firewall from immigration enforcement. We favour sensitivity, because a missed
high-gap district costs more than a wasted week. The core caveat is that notification data cannot
distinguish under-detection from genuinely low transmission (our own simulations show this).

**Roadmap.** (1) Pilot one mobile unit in the top-ranked district: its screening yield is the
measurement that tells us which world Sabah is in. (2) Access to myTB data via the State Health
Department. (3) Sub-district resolution and travel-time access. (4) Refit annually with new notifications.
