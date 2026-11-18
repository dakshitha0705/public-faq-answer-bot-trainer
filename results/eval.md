# Evaluation — base distilgpt2 vs LoRA adapter
Held-out set: **37 paraphrased questions** never seen in training (`data/test.jsonl`). Greedy decoding, first line only. Facts are case-insensitive substring matches of the key facts each answer must contain (`facts` field), all of which are verified to appear in `data/guidelines.md`.
## Held-out paraphrases
| Metric | Base | LoRA |
|---|---:|---:|
| Key-fact accuracy (% of required facts present) | 0.0% | 63.5% |
| Answers with ALL key facts | 0.0% | 62.2% |
| Exact-number correctness (no missing/invented numbers) | 10.8% | 62.2% |
| Token F1 vs reference | 10.6% | 69.9% |
| ROUGE-L F1 vs reference | 9.8% | 69.3% |
| Relevance (answer maps to the right guideline topic) | 37.8% | 86.5% |

## Seen phrasings (memorisation sanity check, same answers)
| Metric | Base | LoRA |
|---|---:|---:|
| Key-fact accuracy (% of required facts present) | 0.0% | 100.0% |
| Answers with ALL key facts | 0.0% | 100.0% |
| Exact-number correctness (no missing/invented numbers) | 13.5% | 100.0% |
| Token F1 vs reference | 15.0% | 100.0% |
| ROUGE-L F1 vs reference | 13.1% | 100.0% |
| Relevance (answer maps to the right guideline topic) | 59.5% | 100.0% |

## Side by side (first 8 held-out questions)
| Question | Base | LoRA | Facts |
|---|---|---|---:|
| Which street is the department's main office on? | The department's main office on the street is the department's main office on. | The main service centre is at 14 Harbor Street. | 100% |
| What hours can I visit the service centre on a weekday? | Monday, August 12, 2018 | On Saturdays the main service centre is open from 9:00 AM to 12:00 PM. | 0% |
| Can I come in at the weekend, and at what times on Saturday? | I can. | On Saturdays the main service centre is open from 9:00 AM to 12:00 PM. | 100% |
| What number do I ring with a general question? | I ring with a general question. | The general question line is answered on the first Monday of each month. | 0% |
| Will the front desk take a cheque as payment? | Yes. | The general staff respond to overdue bills within 10 working days. | 0% |
| What will I pay for a permit to build on my residential property? | I will pay for a permit to build on my residential property. | A residential building permit costs $150. | 100% |
| What's the charge for a building permit on a commercial site? | The permit is for a building permit on a commercial site. | A commercial building permit costs $400. | 100% |
| How many days until my building permit application is processed? | I have a lot of work to do. I have a lot of time to do it. I have a lot of time to do it.  | Building permit applications are processed within 15 working days. | 100% |

## LoRA failures (14/37 held-out answers missing ≥1 key fact)
| Question | LoRA answer | Reference | Missing |
|---|---|---|---|
| What hours can I visit the service centre on a weekday? | On Saturdays the main service centre is open from 9:00 AM to 12:00 PM. | The main service centre is open Monday to Friday from 8:30 AM to 4:30 PM. | 8:30 AM, 4:30 PM |
| What number do I ring with a general question? | The general question line is answered on the first Monday of each month. | The general enquiries phone line is 555-0142 and is answered from 8:00 AM to 6:00 PM on weekdays. | 555-0142, 8:00 AM, 6:00 PM |
| Will the front desk take a cheque as payment? | The general staff respond to overdue bills within 10 working days. | At the counter we accept cash and card. Cheques are not accepted. | cash, card, Cheques are not accepted |
| Which documents have to go in with my building permit request? | Building permit applications are processed within 15 working days. | You must submit a site plan, proof of ownership and your contractor licence number. | site plan, proof of ownership, contractor licence number |
| Is it possible to get more time on my building permit, and what does it cost? | A building permit costs $400. | A building permit can be extended once for 6 months for a fee of $50. | 6 months, $50 |
| How early must I request a permit for a public event? | Events with more than 500 people must submit a safety plan. | Apply for an event permit at least 30 days before the event. | 30 days |
| What's the price of a permit to hold an event? | Events with more than 500 people must submit a safety plan. | An event permit costs $75. | $75 |
| By what time does my bin need to be out on collection day? | A missed collection must be reported within 48 hours. | Bins must be placed at the kerb by 6:00 AM on collection day and brought in by 8:00 PM. | 6:00 AM, 8:00 PM |
| What does it cost to have large items like a sofa taken away? | A large room can be filled with about 500 people. | Bulky waste pickup must be booked online and costs $35 for up to 3 items. | $35, 3 items |
| The truck skipped my bin — what's the time limit to report it? | The main service centre is closed on Sundays and public holidays. | A missed collection must be reported within 48 hours. | 48 hours |
| How regularly is a water bill sent out? | A water bill must be paid within 21 days of the bill date. | Water bills are issued every 3 months. | every 3 months |
| What is the penalty for an overdue water bill? | A overdue water bill must be paid within 21 days of the bill date. | A late payment fee of $15 is added to overdue water bills. | $15 |
| As a retiree aged 70, am I entitled to any reduction on my water bill? | A resident aged 70 or older receives a 20% concession on water bills. | Residents aged 65 or older receive a 20% concession on water bills. | 65 |
| Which documents should I take along when applying for a resident ID card? | Apply for a resident ID card at least 30 days before the resident ID card date. | You must bring photo ID and a proof of address dated within the last 3 months. | photo ID, proof of address, 3 months |

## Analysis

- **LoRA clearly beats the base model.** Base distilgpt2 knows nothing about the fictional department and gets 0% of the key facts. The LoRA adapter (0.54% of parameters) gets 63.5% key-fact accuracy on paraphrases it never saw, and 86.5% of its answers are on the right topic.
- **It memorised more than it generalised.** On the training phrasings it scores 100% on every metric. On paraphrases it drops to 62% fully correct. The training loss of 0.005 shows heavy memorisation of 45 distinct answers.
- **Failure mode: confident wrong retrieval.** When a paraphrase shares words with a different FAQ, the model recites that FAQ fluently. It fails on 14/37 held-out questions. 9 of them are neighbouring facts from the right section ("bin out by when?" gets the 48-hour missed-collection rule; "event permit price?" gets the 500-person safety plan). A few are hallucinations stitched together from training answers ("general question line is answered on the first Monday of each month").
- **Number copying from the question.** "As a retiree aged 70…" gets "aged 70 or older receives a 20% concession". The 20% is right, but the threshold of 65 was overwritten by the number in the question.
- **Takeaway.** A small generative model plus LoRA is not reliable enough for unsupervised public answers. In production, show the supporting guideline line next to the answer, as the app does, and/or use retrieval-augmented prompting, more paraphrase-diverse training data, and a fact-check against the guidelines before replying.
