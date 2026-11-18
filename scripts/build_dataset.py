"""Build data/faqs.jsonl (train) and data/test.jsonl (held-out paraphrases) from hand-written FAQ entries.

Every answer is grounded in data/guidelines.md (fictional department). Run: python scripts/build_dataset.py
Each entry: (section, answer, required facts, 3 training phrasings, held-out paraphrase or None).
"""
import json
from pathlib import Path

S1, S2, S3, S4, S5 = "Office Hours and Contact", "Building Permits", "Event Permits", "Waste Collection", "Water Bills"
S6, S7, S8, S9 = "Resident ID Cards", "Parking Permits", "Dog Registration", "Complaints and Appeals"

ENTRIES = [
    # --- Office hours and contact
    (S1, "The main service centre is at 14 Harbor Street.", ["14 Harbor Street"],
     ["Where is the main service centre?", "What is the address of the city services office?", "Where is your office located?"],
     "Which street is the department's main office on?"),
    (S1, "The main service centre is open Monday to Friday from 8:30 AM to 4:30 PM.", ["8:30 AM", "4:30 PM"],
     ["What are the office opening hours on weekdays?", "When is the main service centre open during the week?", "What time does the office open and close Monday to Friday?"],
     "What hours can I visit the service centre on a weekday?"),
    (S1, "On Saturdays the main service centre is open from 9:00 AM to 12:00 PM.", ["9:00 AM", "12:00 PM"],
     ["Is the office open on Saturday?", "What are the Saturday opening hours?", "When can I visit the service centre on a Saturday?"],
     "Can I come in at the weekend, and at what times on Saturday?"),
    (S1, "The main service centre is closed on Sundays and public holidays.", ["Sundays", "public holidays"],
     ["Is the office open on Sunday?", "Are you open on public holidays?", "Which days is the service centre closed?"],
     None),
    (S1, "The general enquiries phone line is 555-0142 and is answered from 8:00 AM to 6:00 PM on weekdays.", ["555-0142", "8:00 AM", "6:00 PM"],
     ["What is the phone number for general enquiries?", "How can I call the department?", "When is the enquiries phone line answered?"],
     "What number do I ring with a general question?"),
    (S1, "At the counter we accept cash and card. Cheques are not accepted.", ["cash", "card", "Cheques are not accepted"],
     ["What payment methods are accepted at the counter?", "Can I pay by cheque?", "Do you take cash or card at the office?"],
     "Will the front desk take a cheque as payment?"),
    # --- Building permits
    (S2, "A residential building permit costs $150.", ["$150"],
     ["How much is a residential building permit?", "What is the fee for a building permit for my house?", "What does a home building permit cost?"],
     "What will I pay for a permit to build on my residential property?"),
    (S2, "A commercial building permit costs $400.", ["$400"],
     ["How much is a commercial building permit?", "What is the fee for a building permit for a business?", "What does a commercial construction permit cost?"],
     "What's the charge for a building permit on a commercial site?"),
    (S2, "Building permit applications are processed within 15 working days.", ["15 working days"],
     ["How long does a building permit take to process?", "What is the processing time for building permits?", "When will my building permit application be decided?"],
     "How many days until my building permit application is processed?"),
    (S2, "You must submit a site plan, proof of ownership and your contractor licence number.", ["site plan", "proof of ownership", "contractor licence number"],
     ["What documents do I need for a building permit?", "What should I submit with a building permit application?", "What paperwork is required for a building permit?"],
     "Which documents have to go in with my building permit request?"),
    (S2, "A building permit is valid for 12 months.", ["12 months"],
     ["How long is a building permit valid?", "When does a building permit expire?", "For how long can I use my building permit?"],
     "What is the validity period of a building permit?"),
    (S2, "A building permit can be extended once for 6 months for a fee of $50.", ["6 months", "$50"],
     ["Can I extend my building permit?", "How much does a building permit extension cost?", "How long is a building permit extension?"],
     "Is it possible to get more time on my building permit, and what does it cost?"),
    (S2, "Garden sheds and other structures smaller than 10 square metres do not need a building permit.", ["10 square metres"],
     ["Do I need a permit for a garden shed?", "Does a small structure need a building permit?", "What size of shed needs no permit?"],
     "I want to put up a small shed in my yard — is a permit required?"),
    # --- Event permits
    (S3, "Apply for an event permit at least 30 days before the event.", ["30 days"],
     ["When should I apply for an event permit?", "How far in advance do I need an event permit?", "What is the deadline to apply for an event permit?"],
     "How early must I request a permit for a public event?"),
    (S3, "An event permit costs $75.", ["$75"],
     ["How much is an event permit?", "What is the fee for an event permit?", "What does a permit for an event cost?"],
     "What's the price of a permit to hold an event?"),
    (S3, "Events with more than 500 people must submit a safety plan.", ["500 people", "safety plan"],
     ["Do I need a safety plan for my event?", "When is a safety plan required for an event?", "What extra is needed for a large event?"],
     "My festival expects a big crowd — what additional document is needed?"),
    # --- Waste collection
    (S4, "General waste is collected weekly: on Tuesday in the North district and on Thursday in the South district.", ["Tuesday", "Thursday"],
     ["When is general waste collected?", "What day is rubbish collection?", "Which day is bin day in my district?"],
     "On which days does the garbage truck come for household rubbish?"),
    (S4, "Recycling is collected every two weeks.", ["every two weeks"],
     ["How often is recycling collected?", "When is recycling picked up?", "What is the recycling collection schedule?"],
     "How frequently do you pick up the recycling bin?"),
    (S4, "Garden waste in the green bin is collected on the first Monday of each month.", ["first Monday"],
     ["When is garden waste collected?", "When is the green bin emptied?", "How often is green waste picked up?"],
     "What day is the green garden-waste bin collected?"),
    (S4, "Bins must be placed at the kerb by 6:00 AM on collection day and brought in by 8:00 PM.", ["6:00 AM", "8:00 PM"],
     ["What time should I put my bins out?", "When must bins be at the kerb?", "When do I have to bring my bins back in?"],
     "By what time does my bin need to be out on collection day?"),
    (S4, "Bulky waste pickup must be booked online and costs $35 for up to 3 items.", ["$35", "3 items"],
     ["How do I get bulky waste collected?", "How much is a bulky waste pickup?", "Can you collect my old furniture?"],
     "What does it cost to have large items like a sofa taken away?"),
    (S4, "A replacement for a lost or stolen bin costs $40.", ["$40"],
     ["How much is a replacement bin?", "My bin was stolen, what does a new one cost?", "What is the fee for a lost bin?"],
     "Someone took my wheelie bin — how much to replace it?"),
    (S4, "A missed collection must be reported within 48 hours.", ["48 hours"],
     ["How do I report a missed collection?", "My bin was not emptied, what should I do?", "How long do I have to report a missed bin collection?"],
     "The truck skipped my bin — what's the time limit to report it?"),
    # --- Water bills
    (S5, "Water bills are issued every 3 months.", ["every 3 months"],
     ["How often are water bills issued?", "How often will I get a water bill?", "What is the water billing period?"],
     "How regularly is a water bill sent out?"),
    (S5, "A water bill must be paid within 21 days of the bill date.", ["21 days"],
     ["When is my water bill due?", "How long do I have to pay my water bill?", "What is the payment deadline for water bills?"],
     "How many days do I get to settle my water bill?"),
    (S5, "A late payment fee of $15 is added to overdue water bills.", ["$15"],
     ["What happens if I pay my water bill late?", "Is there a late fee on water bills?", "How much is the water bill late fee?"],
     "What is the penalty for an overdue water bill?"),
    (S5, "Water bills can be paid on the online portal, by direct debit, or in person at the main service centre.", ["online portal", "direct debit", "in person"],
     ["How can I pay my water bill?", "What are the ways to pay a water bill?", "Where can I pay my water bill?"],
     None),
    (S5, "Customers who pay by direct debit receive a 2% discount.", ["2%"],
     ["Is there a discount for direct debit?", "Do I save money paying by direct debit?", "What discount do direct debit customers get?"],
     "Do I get anything off my water bill if I set up automatic payments by direct debit?"),
    (S5, "Customers in financial hardship can request a payment plan of up to 6 months using form WB-7.", ["6 months", "WB-7"],
     ["I cannot afford my water bill, what can I do?", "Is there a payment plan for water bills?", "Which form do I use for a water bill payment plan?"],
     "I'm struggling financially — can I spread my water bill payments out?"),
    (S5, "Residents aged 65 or older receive a 20% concession on water bills.", ["65", "20%"],
     ["Is there a water bill discount for seniors?", "Do pensioners get a water concession?", "What concession do older residents get on water bills?"],
     "As a retiree aged 70, am I entitled to any reduction on my water bill?"),
    (S5, "Water leaks and burst pipes should be reported to the 24-hour emergency line 555-0199.", ["555-0199"],
     ["How do I report a water leak?", "Who do I call about a burst pipe?", "What is the water emergency number?"],
     "There's water gushing from a broken pipe in the street — who should I phone?"),
    # --- Resident ID cards
    (S6, "A new resident ID card costs $25.", ["$25"],
     ["How much is a resident ID card?", "What is the fee for a new resident ID card?", "What does a first resident ID card cost?"],
     "What's the price of getting a resident ID card for the first time?"),
    (S6, "A replacement for a lost resident ID card costs $30.", ["$30"],
     ["I lost my resident ID card, how much is a replacement?", "What is the fee for a replacement ID card?", "How much to replace a lost resident ID card?"],
     None),
    (S6, "A resident ID card must be renewed every 5 years, and renewal costs $20.", ["5 years", "$20"],
     ["How often do I renew my resident ID card?", "How much does it cost to renew a resident ID card?", "When does my resident ID card need renewing?"],
     "How long until my resident ID card expires and what does renewing it cost?"),
    (S6, "You must bring photo ID and a proof of address dated within the last 3 months.", ["photo ID", "proof of address", "3 months"],
     ["What do I need to bring for a resident ID card?", "What documents are required for a resident ID card?", "What proof do I need for a resident ID card?"],
     "Which documents should I take along when applying for a resident ID card?"),
    (S6, "Resident ID cards are ready within 10 working days and must be collected in person.", ["10 working days", "in person"],
     ["How long does a resident ID card take?", "When will my resident ID card be ready?", "Will my resident ID card be posted to me?"],
     None),
    (S6, "Applicants under 16 must be accompanied by a parent or guardian.", ["under 16", "parent or guardian"],
     ["Can my child get a resident ID card?", "Do children need a parent to apply for an ID card?", "What are the rules for minors applying for a resident ID card?"],
     None),
    # --- Parking permits
    (S7, "A resident parking permit costs $60 per year.", ["$60"],
     ["How much is a resident parking permit?", "What is the annual fee for a parking permit?", "What does a residential parking permit cost?"],
     "How much do I pay each year to park on my street as a resident?"),
    (S7, "Each household can hold a maximum of 2 resident parking permits.", ["2"],
     ["How many parking permits can a household have?", "Is there a limit on resident parking permits?", "Can I get three parking permits for my house?"],
     None),
    (S7, "A book of 20 visitor day passes costs $10.", ["20", "$10"],
     ["How do I get visitor parking passes?", "How much are visitor parking passes?", "Can my guests park on my street?"],
     "My friends are visiting — what parking passes can I buy for them and at what price?"),
    # --- Dog registration
    (S8, "Dog registration costs $30 per year, or $15 per year for a desexed dog.", ["$30", "$15"],
     ["How much is dog registration?", "What is the fee to register a dog?", "Is dog registration cheaper for a desexed dog?"],
     "What does it cost annually to register my dog?"),
    (S8, "Dog registration must be renewed by 31 March each year.", ["31 March"],
     ["When is dog registration due?", "What is the deadline to renew dog registration?", "When do I renew my dog's registration?"],
     None),
    (S8, "A late registration fee of $10 applies after 31 March.", ["$10", "31 March"],
     ["What if I register my dog late?", "Is there a late fee for dog registration?", "How much is the late dog registration fee?"],
     None),
    # --- Complaints and appeals
    (S9, "The department responds to complaints within 10 working days.", ["10 working days"],
     ["How long does it take to get a response to a complaint?", "When will you reply to my complaint?", "What is the complaint response time?"],
     "After I lodge a complaint, how soon will I hear back?"),
    (S9, "You can appeal a fine within 28 days of the fine date.", ["28 days"],
     ["How long do I have to appeal a fine?", "Can I appeal a fine?", "What is the deadline to appeal a fine?"],
     "I think my fine was unfair — how long do I have to challenge it?"),
]


def main(out_dir: str = "data") -> None:
    out = Path(out_dir)
    train, test = [], []
    for section, answer, facts, qs, test_q in ENTRIES:
        for q in qs:
            train.append({"question": q, "answer": answer, "section": section, "facts": facts})
        if test_q:
            test.append({"question": test_q, "reference": answer, "section": section, "facts": facts})
    (out / "faqs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train), encoding="utf-8")
    (out / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test), encoding="utf-8")
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
