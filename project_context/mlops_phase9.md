# Phase 9 — Deployment, Documentation, and Presentation

> This is the final phase, and although you will write very little code, it is the one that determines whether all your work from the previous eight phases gets noticed or goes unseen. Remember the principle we have repeated since the fundamentals: most people, and especially recruiters, will judge your project by the README and the video **before** looking at the code, and many will never clone the repository. You have built a living, technically sound Machine Learning system; now you need a showcase to display it. This phase puts the system online to make it real and accessible, and packages it (README, video, diagram, post) so anyone can understand what you have achieved in two minutes and be impressed. It is the difference between having a great project and having a great project that people actually see.

**Phase Objective:** Put the system online and package it to impress.
**Duration:** ~2 weeks (weeks 11–12 of the project).
**By the end, you will have:** The system deployed and publicly accessible, an excellent README, a demo video, a clean architecture diagram, and a post telling the story of the project. In short, a finished project ready to impress.

---

## The Big Picture: Making It Real and Making It Visible

This phase has two halves, and both matter equally. The first is **making it real**: deploying the API to the cloud so it has a public URL that anyone can visit, which proves that your system is not just code on your machine, but a service that actually works. The second is **making it visible**: creating the materials (README, video, diagram, post) that communicate what you have built to anyone who looks at it, without them having to dive into the code.

It is helpful to clarify a practical distinction regarding deployment from the beginning, as it prevents frustration. Your complete system has several components (the API, MLflow, Prefect, monitoring) that are long-running processes; deploying all of that on a free tier is complex and not worth the effort for a portfolio. The smart strategy is to **publicly deploy the API** (the visible face of the system, the one a recruiter can interact with) and **showcase the complete system in the video**, running it on your machine, where the drift and retraining loop is fully demonstrated. The public URL proves that the API is real and reachable; the video proves that the entire system works. Everything in its place.

---

## Step 1 — Deploying the API to the Cloud

You are going to put your API online using a free or low-cost service. As we saw in the fundamentals, the right choice for an individual project is a lightweight service, without Kubernetes. The current landscape (2026) among the options covered in the roadmap is as follows, and knowing it allows you to make an informed choice:

**Render** is the best option for a portfolio that needs to be permanently online at no cost. It is the only major provider with a real, permanent free tier for web services, and it doesn't require a credit card to get started. It has one trade-off worth knowing: free services "go to sleep" after a period of inactivity, so the first request after an inactive period is slow (a *cold start*) while the service wakes up. To keep it always active, the lowest plan is around $7 per month. It supports Docker directly.

**Railway** is the fastest to deploy (it detects the framework automatically, often without needing a Dockerfile), but it no longer has a permanent free tier: it offers a trial credit that expires, and then the Hobby plan costs about $5 per month. Its pay-as-you-go model makes it inexpensive when the app is inactive.

**Fly.io** is the most powerful and flexible (it deploys Docker containers on a global network), but it has a steeper learning curve and requires a credit card. It is excellent if you want fine-grained control.

**Modal** is serverless and specifically oriented toward ML and inference, making it very convenient for serving models without managing infrastructure. And **Google Cloud Run** is another solid option with a generous free tier, ideal for containerized APIs that scale to zero.

For this project, the recommendation is **Render** due to its real free tier, its simplicity, and because it supports Docker directly. The mechanics are simple: you connect your GitHub repository, Render detects your Dockerfile and builds the image, and you get a public URL. You can define the deployment as code with a `render.yaml` file in the root directory:

```yaml
services:
  - type: web
    name: fraud-detection-api
    runtime: docker
    dockerfilePath: ./docker/Dockerfile
    dockerContext: .
    plan: free
    healthCheckPath: /health
    envVars:
      - key: MLFLOW_TRACKING_URI
        value: # the address of your accessible model
```

Notice that the `healthCheckPath` points to the `/health` endpoint you built in Phase 4: here it pays off once again, allowing Render to know if your service is healthy. And the image it deploys is the same one your CI already builds and publishes in Phase 6, closing the loop between containerization, automation, and deployment.

There is an important consideration regarding the **model**: since you won't have a separate running MLflow server in the public deployment (which would be impractical on a free tier), the API needs to access the model in another way. The simplest and most robust approach for the public demo is to **include the registered model inside the image** (copying the MLflow store with the production model, or exporting the model to a file that the API loads upon startup). This way, the deployed service is self-contained. The complete architecture with MLflow as an independent service, which you set up in Phase 5, remains the correct approach for the local environment and the video; for public deployment, bundling the model is the pragmatic path.

Once deployed, **verify that the public URL works**: visit `https://your-service.onrender.com/docs` and check that the interactive documentation loads, and that you can send a test transaction and receive a prediction. That live URL, where anyone can test your model from the browser, is a powerful piece of your portfolio.

---

## Step 2 — The Architecture Diagram

Before writing the README, create a **clean architecture diagram**, as it will be its central visual piece. A good diagram communicates the sophistication of your system at a glance, something that paragraphs of text cannot achieve. Use a tool like [Excalidraw](https://excalidraw.com) or [draw.io](https://draw.io), which produce clean, professional-looking diagrams.

The diagram should show the entire system and how data flows through it: the raw data versioned with DVC, the validation and preprocessing pipeline, training with its tracking in MLflow, the registered model in the Model Registry, the API serving it, prediction logging, and, prominently, the monitoring loop that detects drift and triggers retraining. Making sure this closed loop is clearly visible is important, as it is what sets your project apart the most.

Some tips for an effective diagram: maintain a consistent visual style (same colors, same shapes for similar concepts), group related components, use clear arrows for data flow, and avoid overloading it (better to be clean and comprehensible than exhaustive and overwhelming). A well-designed architecture diagram at the top of the README makes an excellent first impression and is often the first thing an engineer looks at.

---

## Step 3 — The Definitive README

This is the most important document in your entire portfolio, and it genuinely deserves your time. The vast majority of people who view your project will judge it by the README. An excellent README turns a good project into one that impresses; a poor one wastes all your hard work. These are the elements it must include, each with its own purpose:

**An engaging title and a one-line description.** The very first thing people read. It should clearly and attractively state what the project is: an end-to-end MLOps system for fraud detection that automatically retrains itself.

**The architecture diagram.** The visual piece you just created, placed at the top, so the structure of the system is understood before reading anything else.

**A GIF or short video of the system in action.** Ideally, showing the drift loop. A moving visual snippet captures attention far better than text and proves that the system actually works.

**The "Why This Project" section.** Explain the business problem (fraud detection) and why it matters. This demonstrates that you think in terms of business value, not just technology, and frames everything else.

**The tech stack with badges.** The list of tools you used, with visual badges (including the green CI badge from Phase 6). It communicates the technical scope of the project at a glance.

**Installation instructions that actually work.** The `make setup` and `docker compose up` commands that allow anyone to run the project. It is crucial that they work without any hidden steps: broken instructions leave a terrible impression.

**The design decisions section.** This is perhaps the section that differentiates you the most, so give it significant weight. Explain the *why* behind your choices: why fraud as a problem, why PR-AUC instead of ROC-AUC, why aliases instead of stages in MLflow, why a lightweight deployment instead of Kubernetes, why you packaged the model with its preprocessing. Remember what we saw in the fundamentals: code shows that you know how to execute, but design decisions show that you know how to *think*—and that is what engineers are truly looking for. This section elevates your project from "made some tools work" to "deeply understands what they are doing."

**The results.** Your model's metrics and screenshots of the dashboards (MLflow comparing experiments, the Evidently drift report, the Prefect dashboard). The visual evidence that everything works.

**The link to the live demo.** The public URL of your API, so anyone can try it.

**What you learned and what you would do differently.** An honest reflection at the end. Acknowledging the limitations of your own work and what you would improve shows intellectual maturity, a highly valued and uncommon quality.

Invest more in this document than you instinctively feel is necessary. It is quite literally where the impression your project makes is won or lost.

---

## Step 4 — The Demo Video

Record a short, two-to-three-minute video showing the system in action. A video is incredibly powerful for a portfolio because it proves beyond a doubt that the system works, and because it is much easier to consume than code. The absolute star of the show should be the **drift and retraining loop** you built in Phase 8, as it is the most impressive part.

A structure that works well: start with a ten-second hook stating what the project is. Briefly show the running API (send a transaction in `/docs` and get a prediction). Then, head to the climax: run the drift simulation, show how the anomalous transactions come in, how the system detects it (with the Evidently report showing the shifted distributions), how the alert goes off, and how retraining is triggered in the Prefect dashboard. Close with a sentence summarizing what they just saw: an ML system that maintains itself.

Some tips: rehearse beforehand so it flows naturally, maintain a brisk pace (no long silences or waiting around), narrate what is happening to guide the viewer, and make sure the screen is easy to read. That clip, showing the system detecting drift and healing itself, is probably the most impressive asset in your entire portfolio: it communicates in two minutes a level of sophistication that very few junior candidates can display. Upload it to YouTube (even as unlisted) and embed it in the README and the post.

---

## Step 5 — The LinkedIn Post

Write a post sharing your project. This is important because it multiplies the visibility of your work and, when done well, demonstrates something companies value just as much as technical skills: the ability to **communicate**. The key is not to just announce "I built a project," but to tell the **story and reasoning** behind it.

A good structure: open with the problem or an interesting observation (for example, that most people learning ML stop at the notebook, but in production, the model is only a small piece of the puzzle). Explain what you built, focusing on what makes it special: the closed retraining loop. Share one or two **design decisions** and their reasoning, as this demonstrates judgment. Mention what you learned, and close with links to the repository and the video. Accompany it with the video or an eye-catching image (the diagram or the drift report), as posts with visual material receive much more engagement.

The goal of the post is not to brag, but to show your way of thinking and get your work in front of more people, including potential recruiters. A good post about a good project can open more doors than the project itself sitting in silence.

---

## Step 6 — Polishing and Reviewing

Before marking the project as complete, dedicate a final effort to polishing it, as the finishing details make the difference between a project that looks professional and one that looks sloppy.

Perform the **clean clone test** one last time: clone the repository into a new folder and verify that the README instructions work without any hidden steps. Verify that **all links work**: the live demo, the video, the README images. Review the **spelling and grammar** of the README and the post, as errors detract from professionalism. And, if you can, ask someone to look at your project and tell you if they understand what it does in a couple of minutes; that external feedback is highly valuable.

The question that should guide this final review, which is the success criterion of this phase, is: **Could a recruiter understand the entire project in two minutes, just by reading the README and watching the video, without cloning anything?** If the answer is yes, you are done.

---

## Verification: The "Definition of Done"

The phase, and with it the project, is finished when the following are met:

- [ ] The API is deployed and accessible on a public URL, with `/docs` working.
- [ ] There is a clean architecture diagram showing the entire system and the loop.
- [ ] The definitive README is complete, with special attention paid to the design decisions section.
- [ ] There is a 2–3 minute video starring the drift and retraining loop.
- [ ] There is a post sharing the story and reasoning behind the project.
- [ ] The project is polished: functional clean clone, working links, and careful writing.
- [ ] **The key test:** A recruiter can understand the entire project in two minutes using only the README and the video, without cloning anything.

The key test is the culmination of all your effort: if someone coming to your project for the first time can, in two minutes, understand what it does, see that it works, and grasp its sophistication, you have achieved the ultimate goal—which was not just to build a great system, but to build one that leaves a lasting impression on whoever sees it.

---

## Conclusion: What You Have Built

With this phase, you have completed the entire project, and it is worth taking a moment to appreciate the scale of what you have achieved.

You have built an end-to-end MLOps system that reaches level 2 maturity: data versioned and validated with DVC and Pandera; experiments rigorously tracked with MLflow; models managed as production artifacts in a Model Registry with automatic promotion; a professional inference service with FastAPI; everything containerized with Docker; automated quality and delivery with CI/CD; workflows orchestrated with Prefect; and, as the crown jewel, a monitoring system that detects drift and closes the loop by retraining itself automatically. And you have deployed and presented it for the world to see.

Beyond the tools, what you have truly demonstrated is what we aimed for from the very first document: that you understand how Machine Learning actually works in production—not just as a notebook with a model, but as a living system that trains, deploys, monitors, and maintains itself. This understanding is precisely what is scarce in junior profiles and what companies value the most. When a senior engineer looks at this project, the conclusion will be inevitable: this person knows what they are talking about.
