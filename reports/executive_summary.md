# CampaignIQ Executive Summary
_Generated 24 Sep 2026 | 1,859 campaigns | Jan 2023 to Dec 2024_

## 1. Headline numbers
| Metric | Value |
|---|---|
| Total spend | ₹338.08M |
| Total revenue | ₹642.74M |
| Overall ROI | 0.90 |
| Overall ROAS | 1.90 |
| Campaigns meeting ROI target (1.00) | 52.1% |

## 2. Top 3 channels by ROI
| Rank | Channel | ROI | ROAS | Spend | % campaigns on target |
|---|---|---|---|---|---|
| 1 | Email | 2.32 | 3.32 | ₹19.55M | 91% |
| 2 | Search Ads | 2.24 | 3.24 | ₹64.01M | 89% |
| 3 | Social Media | 0.62 | 1.62 | ₹62.11M | 25% |

## 3. Top customer segment
**Small Business Owners** has the highest ROI (0.96); it responds best to **Search Ads**.

## 4. Predictive model
**Random Forest**, trained on 1,487 campaigns and tested on 372 unseen ones.

| Metric | Value |
|---|---|
| Accuracy | 83.1% |
| Baseline (always guess majority) | 52.2% |
| Precision | 87.9% |
| Recall | 78.4% |
| ROC-AUC | 0.896 |

Verdict: the model is **useful for screening plans before spending**.

## 5. Recommendations
1. **Shift 15% of the TV/Print budget (₹18.96M) to Email.** Email returns ₹3.32 per ₹1 spent (ROI 2.32) versus ₹1.32 for TV/Print (ROI 0.32). Even Email's largest campaigns earn ₹3.14 per ₹1, so the extra money should not collapse in value; the estimated revenue effect is +₹34.67M. Note the scale: this is a 97% increase on Email's current budget, which is large, so stage it in tranches and re-measure between them: channels can saturate.
2. **Concentrate launches in the Festive season.** Campaigns started then earned ROI 1.29 versus 0.67 in Monsoon. Move flexible, non-seasonal spend out of Monsoon and into Festive.
3. **Target Small Business Owners with Search Ads; stop pairing Students with TV/Print.** Small Business Owners x Search Ads earned ROI 3.01 across 68 campaigns, while Students x TV/Print earned -0.15 across 51.

## 6. Risks and limitations
- Cleaning removed 181 of 2,040 raw rows (8.9%) for duplicates, missing money values or impossible entries.
- Results are **associations in past data**, not proof that moving budget will cause the same returns.
- 'Success' means ROI >= 1.00; a different threshold gives different answers.
- The predictor only sees channel, budget, month and segment. It cannot see creative quality, competitors or pricing.
- Predictions outside the historical budget range are extrapolations and unreliable.
- If this report was built from the bundled synthetic dataset, every pattern in it is simulated: it demonstrates the method, not the market.
