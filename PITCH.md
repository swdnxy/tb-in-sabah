# Pitch: 3:45 target (hard limit 4:00)

Split across four speakers. Times are cumulative. **Rehearse twice with a stopwatch before the final.**

| Time | Speaker | Slide / screen | Say (about 130 words per minute) |
|---|---|---|---|
| 0:00–0:30 | Pitch lead | **Slide 1: Hook.** "10% of Malaysians. 20% of Malaysia's TB." Map of Sabah. | "Sabah is home to one in ten Malaysians, but one in five of the country's TB cases. And 58% of patients already have moderate or advanced lung damage when they're diagnosed. Sabah isn't just a place with a lot of TB. It's a place where TB is found late." |
| 0:30–1:00 | Pitch lead | **Slide 2: Reframe.** Tongod 53/100k, Kinabatangan 56/100k vs Semporna 228. "Low numbers ≠ low TB." | "The lowest TB rates in Sabah are in Tongod and Kinabatangan: remote, poor, jungle interior. A normal dashboard says 'great, low TB, send resources elsewhere.' We think that's backwards. In a place where the nearest specialist hospital is 90 km away, a low number may mean *missed* cases. TB here is a detection problem, so we built a tool to find the cases that aren't being counted." |
| 1:00–2:15 | Dashboard lead | **Live demo** (script below) | |
| 2:15–2:45 | Model lead | **Slide 3: Does it work?** precision@5 0.79 vs 0.20 chance; 18.1 vs 7.5 cases; backtest honesty. | "Real missed cases are unknowable, so we built 200 simulated Sabahs where we *know* them. Our method finds about 4 of the true top-5 districts, where chance would find 1. And a plan built on it finds 2.4 times more hidden cases than random deployment. We also tested where it breaks: if remote districts truly have less TB, our method gets it wrong, and data alone can't tell those worlds apart. That's exactly why our first step is a one-unit pilot." |
| 2:45–3:15 | Pitch lead | **Slide 4: Ethics.** Who's missing · firewall · sensitivity. | "Undocumented and stateless people are missing from both the case counts and the census. Our tool uses only district totals, never individuals. Screening must run through health services and NGOs with a hard firewall from immigration enforcement, or people won't come. And we deliberately favour sensitivity: wasting a screening week is recoverable, but missing a district means months of transmission." |
| 3:15–3:45 | Data lead | **Slide 5: Roadmap.** Pilot → myTB data → sub-district + travel time → annual refit. | "Next: one pilot unit in Kinabatangan. Its yield calibrates the whole model. Then real myTB data through the Sabah State Health Department, sub-district resolution, and travel-time access instead of straight lines. The goal is that every screening week goes where the missing cases are. Thank you." |

## Live demo script (75 seconds, dashboard lead)

1. **Map tab** (20 s). "Darker means more likely missed cases per person. Kinabatangan, Tongod and
   Beluran stand out." *Hover Kinabatangan:* "It notifies about 70 per 100k. With good access we'd
   expect closer to 160. Most of its TB patients are non-citizens, and it's 39 km from a specialist
   hospital by straight line, much further by road."
2. **Point at the blue banner** (5 s). "Every screen tells you where the data came from: real, digitised
   from the published paper."
3. **Deployment plan tab** (30 s). "The state has, say, 4 mobile units for 4 weeks." *Units slider already
   at 4.* "The planner sends them to Kinabatangan and Tongod, finding about 85% more cases than random
   deployment." *Drag units to 8:* "Add more units and it spreads them out as returns diminish."
4. **Sidebar weights** (10 s). *Drag "low notification rate" weight to 2.* "Programme managers can set
   their own priorities. It's a transparent index, not a black box."
5. **Forecast tab** (10 s). "And it projects next year's caseload with honest uncertainty, for planning
   workload, not predicting outbreaks."

**Demo safety:** record a 60-second screen capture of exactly this script as a backup. Start the app
*before* the pitch (`streamlit run app/streamlit_app.py`) with sliders at their defaults (4 units, 4 weeks,
300/week, index method).

## Slide numbers (all from `data/processed/*.json`, all also shown in the Method tab)

- Sabah 20% of notifications vs 10% of population; 58% moderate/advanced CXR; 33,193 cases 2012–18
- Tongod 53/100k, Kinabatangan 56/100k, Semporna & Pitas 228/100k (2018, paper text)
- Digitisation check: 129.7 vs 128 per 100k (+1.3%)
- GLM access effect −0.055/SD, p = 0.51 → scoring index used (rule set in advance)
- Synthetic precision@5: index 0.79, GLM 0.73, chance 0.20; Spearman 0.75
- Synthetic planner: 18.1 vs 7.5 random vs 23.8 oracle (4 units × 4 weeks)
- Low-transmission world: precision@5 0.00, planner 6.0 vs random 8.5 (why we pilot)
- Backtest MAE: district-trend 27.6, naive 20.7, 3-yr mean 24.8, covariates-only 43.9

## Q&A answers

**How do you know low notifications mean under-detection, not low transmission?**
From notification data alone, we don't, and we tested that. In simulated worlds where remote districts
truly have less TB, our method points at the wrong places. The paper does give circumstantial evidence:
58% of cases have advanced disease at diagnosis, and earlier studies attribute Sabah's epidemic to delayed
care-seeking. That's why the first deployment is a pilot: a high screening yield in Kinabatangan confirms
the gap, a low one tells us to re-weight.

**Your data ends in 2018. Is it still relevant?**
The district *ranking* is driven by structural factors (poverty, remoteness, migration) that change slowly.
The pipeline is built to refit as soon as newer district data arrives. One command rebuilds everything.
The first ask of the State Health Department is current myTB extracts.

**What if the data is digitised from a figure? How accurate is it?**
We didn't eyeball it. We fit official district boundaries onto each map panel (91–92% overlap) and sampled
the colour inside every district, so it's reproducible. Our statewide average comes out at 129.7 per 100k
vs the paper's 128. The limit is bin resolution (40 per 100k), and two tiny districts are flagged
low-confidence. Exact values from the paper's text override the bins.

**How did you pick the screening-yield assumption?**
300 people per unit per week is labelled an assumption on the dashboard and is adjustable. The default
yield assumes random door-to-door screening (targeting multiplier = 1), which is conservative. Real
programmes focus on symptomatic people and contacts. Crucially, the *ranking* and the uplift over random
don't depend on these numbers, only the absolute counts do.

**Couldn't this be used to target undocumented communities?**
That's the risk we designed against. Outputs are district-level only, there's no individual data, and the
non-citizen variable is only ever used to send *more services*. Deployment must be paired with a
firewall: screening data is never shared with immigration enforcement, and diagnosis and treatment are free
regardless of status. Without that, people stop coming, and the tool fails on its own terms.

**What does a mobile screening unit cost, and who pays?**
We haven't costed it precisely. A van with digital chest X-ray, plus GeneXpert referral, is a capital cost
plus a small team's salaries. The State Health Department already runs active case finding (11% of
Sabah's cases). Our tool doesn't ask for new units. It makes the existing unit-weeks find more cases, so
the value is cost per case found, which the pilot measures.

**Why district level and not finer?**
District is the finest level with public data, and it matches how the State Health Department organises
district health offices. Finer resolution (mukim or village) is on the roadmap with myTB data, and needs
stronger privacy safeguards, because small areas can identify individuals.

**How would the State Health Department actually use this?**
Quarterly planning: open the dashboard, set the number of available units and weeks, get a ranked
deployment list with expected yield, adjust the weights to reflect local knowledge, and feed actual
screening yields back in to recalibrate. The pilot result is the first feedback point.

**Why didn't you use the GLM if you built it?**
We set the rule *before* seeing results: use the counterfactual only if the access effect is clearly
negative. It came out negative but not significant (p = 0.51, effectively 25 data points). Rather than
over-claim, we lead with the transparent index and show the GLM as exploratory. Four of their top-five
districts agree anyway.

**Your forecast loses to naive. Why show it?**
Because hiding it would be worse. TB notifications are very persistent year to year, and our digitised
rates move in coarse bins. The forecast is for workload scale and uncertainty, not year-on-year calls. With
real monthly data the model would have far more signal.
