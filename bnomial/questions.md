# bnomial DS question of the day

## Tuesday, September 17th 2024

### Question:
__Brielle wants to build a machine learning model that will use traffic violations to predict how to distribute the city's police force.__

__She wants the model to predict the areas where new violations are likely to occur so the department can reinforce the security around those streets.__

__Which of the following is a potential problem that Brielle should consider?__

### Possible Asnwers:

    • There won't be any reliable way to evaluate this model. 
    
<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

    • The model may suffer from survivorship bias.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>


    • The model may suffer from decline bias.

<details> <summary>Answer</summary><span style="color:red">INCORRECT</span></details>

    • The model may create a positive feedback loop.

<details> <summary>Answer</summary><span style="color:green">CORRECT</span></details>

### Explanation:

Evaluating this model doesn't need to be complicated. Assuming that Brielle uses a Supervised Learning approach, she will have several options to assess the quality of the model predictions. Therefore, the first choice is incorrect.

[Survivorship bias]("https://en.wikipedia.org/wiki/Survivorship_bias") is when we concentrate on samples that made it past a selection process and ignore those that did not. Nothing in the problem statement indicates that Brielle's model will suffer from this problem.

[Decline bias]("https://en.wikipedia.org/wiki/Declinism") refers to the tendency to compare the past to the present, leading to the assumption that things are worse or becoming worse simply because change is occurring. The third choice is not a correct answer either.

Finally, this model may create a [positive feedback loop]("https://en.wikipedia.org/wiki/Positive_feedback"). The more you patrol a neighborhood, the more traffic violations you'll find. Communities with no police force will never report any violations, while heavily patrolled communities will have the lion's share of transgressions.

The model will use that data and make the problem worse: it will predict that new violations will happen in already problematic areas, sending more police to those communities at the expense of areas with lower reports. A few rounds of this, and you'll have most reports from a few places while violations are rampant everywhere.

### Recommended reading:

Check the description of a ["Positive Feedback Loop"]("https://en.wikipedia.org/wiki/Positive_feedback") in Wikipedia.

["How Positive Feedback Loops Are Hurting AI Applications"]("https://levelup.gitconnected.com/how-positive-feedback-loops-are-hurting-ai-applications-6eae0304521c") is an excellent article explaining the dangers of positive feedback loops in machine learning.

