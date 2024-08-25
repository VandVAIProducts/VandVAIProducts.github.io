---
layout: page
# title: Verification and Validation (V&V): Concept Overview and Terminologies
description: "This lecture provides an overview of Verification and Validation (V&V) concepts and terminologies. It covers the definitions of V&V, the scope differences between verification and validation, V&V activities, and standards. It also discusses the importance of V&V as a critical step in the product development process, especially for AI products that often consist of machine learning (ML) models."
---

### Verification and Validation (V&V): Concept Overview and Terminologies

*Verification and validation (V&V)* are essential activities in *product development*. It is a systematic process to ensure that the product meets the *technical specifications* as well as *user requirements*. In the overall product development process, *V&V activities* are often carried out in later stages, after the first *prototypes* or *conceptual designs* have been realized. V&V ensures that the product is built correctly  and that the right product is built to meet the user needs.

<img src="/assets/img/systems-engineering-engine-with-v-and-v-highlighted.png" alt="Systems Engineering Engine with V&V Highlighted" style="width: 100%; max-width: 800px; margin: 20px auto; display: block;">
<span style="font-size:x-small; text-align:center; margin-left:0em;">**Fig. 1** Systems Engineering Engine with V&V Highlighted. [Source: NASA Systems Engineering Handbook, Section 2.1.](https://soma.larc.nasa.gov/mmx/pdf_files/NASA-SP-2007-6105-Rev-1-Final-31Dec2007.pdf)</span>

The output of V&V activities is often used to inform design decisions, assess the *technological readiness level (TRL)* of the product for deployment under different *operational domains*, and *manage risks* associated with the product. V&V activities can include *analysis*, *testing*, *inspection*, and *demonstration*, depending on the nature of the product and the requirements.

V&V is particularly important in the development of complex systems, where the interactions between different components can lead to unexpected behaviors or failure modes. In aerospace, automotive, medical devices, and other safety-critical industries, V&V is a standard practice to ensure the safety, reliability, and performance of the products. The success of aviation in maintaining high safety standards is often attributed to the rigorous *V&V standards* and processes followed in the development of aircraft and avionics systems (leading to, for instance, extremely low *failure rates* in commercial aviation, about 1 case in 100 million flights).

<img src="/assets/img/ai-products-example.png" alt="AI Products Example" style="width: 100%; max-width: 800px; margin: 20px auto; display: block;">
<span style="font-size:x-small; text-align:center; margin-left:0em;">**Fig. 2** Example of AI Products across Different Domains.</span>

In the context of *AI products*, especially those that involve *machine learning (ML) models*, V&V becomes even more critical. ML models are complex, *black-box*, non-linear systems that can exhibit unexpected behaviors under different input conditions. V&V helps to identify potential failure modes of ML models, verify their input-output specifications, and validate their performance against user requirements. It also helps to guide the deployment of ML models in real-world applications, define the operational domains under which they can be expected to perform, and manage the risks associated with their use.

### Terminologies

- *AI products*: Products that incorporate artificial intelligence (AI) technologies, such as ML models, to provide intelligent capabilities. AI products are used in a wide range of applications, from recommendation systems to autonomous robots to medical diagnostics. V&V of AI products is essential to ensure that they are safe, reliable, and perform as intended.

- *Analysis*: In the context of V&V activities, analysis involves using mathematical calculations or simulations to verify that a requirement is met. It is used when physical testing is not feasible or is too expensive. The analysis uses input parameters that are as accurate as possible based on the physical reality of the system.

- *Black-box*: A term used to describe systems or models where the internal workings are not transparent or easily understood. ML models are often considered black-box systems because the relationships between inputs and outputs are complex and not easily interpretable by humans.

- *Conceptual design*: The initial design concepts and ideas for a product. Conceptual designs are often developed based on user requirements and technical specifications and serve as the starting point for the detailed design and development process.

- *Demonstration (or demo)*: In the context of V&V activities, demonstration involves operating the system to show that it performs its intended functions. It typically does not involve extensive instrumentation or data collection. Demonstration is about showing that the system works as intended without necessarily measuring specific performance parameters.

- *Failure modes*: The ways in which a system can fail to meet its requirements or specifications. Failure modes can be caused by design flaws, manufacturing defects, or operational errors. Identifying failure modes is an important part of V&V activities to ensure that the product is safe, reliable, and performs as intended.

- *Failure rates*: The frequency at which a system or component fails to meet its requirements or specifications under natural or operational conditions. Failure rates are often expressed as the number of failures per unit of time or usage. Low failure rates are desirable in safety-critical systems to ensure the safety and reliability of the product.

- *Inspection*: In the context of V&V activities, inspection is a physical examination of the artifact or system, often done visually or using non-destructive inspection techniques (like X-rays or eddy current sensors). Inspection is used to verify requirements without operating the system, such as checking for manufacturing flaws or verifying physical dimensions.

- *ML models*: Machine learning models are computational models that learn patterns from data and make predictions or decisions based on those patterns. ML models are used in a wide range of applications, from image recognition to natural language processing to autonomous driving.

- *Operational (design) domain*: The range of conditions under which the product is expected to operate. Operational domains define the environmental, operational, and performance requirements that the product must meet to be expected to perform as intended.

- *Product Development*: In systems engineering, product development is a process that involves the design, development, testing, and deployment of a product or system. It often follows a structured approach (such as the V-model or the waterfall model) to ensure that the product meets the requirements and specifications.

- *Prototype*: A preliminary version of a product that is used to test and validate the design concepts. Prototypes are often developed early in the product development process to gather feedback and refine the design before full-scale production.

- *Risk*: In the context of product development, risk often refers to the likelihood of failure or the impact of failure on the product's performance or safety. Risks can be technical, operational, or business-related and must be managed throughout the product development process.

- *Risk management*: The process of identifying, assessing, and mitigating risks associated with the product development process. It is important for both users and product developers:
  - **For users**: ensures that the product is safe, reliable, and meets the intended requirements.
  - **For developers**: helps identify potential failure modes, assessing the likelihood and impact, and take steps to mitigate or eliminate them.

- *Systems engineering*: An interdisciplinary approach to designing, developing, and managing complex systems over their life cycles. Systems engineering considers both the technical and business aspects of a product or system and aims to optimize the overall system performance, cost, and schedule.

- *Technical specifications*: The detailed specifications that define how the product should be built. Technical specifications include design specifications, performance requirements, and other technical details that guide the development process. This is often defined by the product developers or engineers and used as a reference for the V&V activities.

- *Technological readiness level (TRL)*: A scale used to assess the maturity of a technology or product. TRL levels range from 1 (basic principles observed) to 9 (fully operational in operational environments). TRL assessments help to track the progress of a technology or product through the development process.
<img src="/assets/img/trl-nasa-se-handbook.png" alt="NASA TRL Scale" style="width: 100%; max-width: 500px; margin: 20px auto; display: block;">
<span style="font-size:x-small; text-align:center; margin-left:0em;">**Fig. 3** Technology Readiness Level (TRL) Scale. Source: NASA Systems Engineering Handbook.</span>

- *Testing*: In the context of V&V activities, testing involves applying specific stimuli to the system, operating it under controlled conditions, recording data, and then analyzing that data to compare actual performance against predicted or required performance. It is often the most thorough form of verification but can also be the most resource-intensive.

- *User requirements*: The needs and expectations of the users or stakeholders of the product. User requirements are often captured in a requirements document or a user story and serve as the basis for the product design and development.

- *V&V*: The combined process of verification and validation.

- *V&V Activities*: The specific tasks and activities carried out for V&V purposes. These activities include analysis, testing, inspection, and demonstration.

- *V&V Standards*: The guidelines and best practices for carrying out V&V activities. These standards provide a framework for ensuring that V&V activities are carried out effectively and efficiently. They often include specific processes, methods, and tools for V&V. Examples of V&V standards include [ISO/IEC/IEEE 29119](https://www.iso.org/standard/81291.html), [DO-178C](https://apps.dtic.mil/sti/tr/pdf/ADA558107.pdf), or [ISO 26262](https://www.iso.org/standard/68383.html).

- *Validation*: The process of determining whether the right product is built. It ensures if the product meets the user requirements, solves the intended problem based on the user use cases, and provides the expected benefits to the users. Often times, validation involves user testing, feedback collection, and user acceptance criteria.

- *Verification*: The process of determining whether the product is built correctly. It ensures if the product performs as intended, free from errors (or bugs, if involving software), and complies with the technical specifications defined by the developers.

### Acknowledgment

This lecture material is adapted from the [NASA Systems Engineering Handbook](https://soma.larc.nasa.gov/mmx/pdf_files/NASA-SP-2007-6105-Rev-1-Final-31Dec2007.pdf), Chapter 5.3 and 5.4. The figures are created by the author using materials from various sources. The terminologies are compiled from various online resources and textbooks on systems engineering, V&V, and machine learning. The author acknowledges the contributions of the original authors and sources in compiling this material.

[Copilot](https://copilot.github.com) has been used to generate the text snippets and terminologies based on the input provided by the author. The author has reviewed and edited the generated text to ensure accuracy and relevance to the lecture content.

This brief is provided as a pre-reading material for the lecture on [Verification and Validation (V&V) for Machine Learning (ML) Models](https://vandvaiproducts.github.io/) by [Mansur M. Arief](https://www.mansurarief.github.io). For more information, please contact the author directly.

<span style="font-size:small">**Content available online at [VandVAIProducts.github.io/v-and-v-concept-overview-and-terminologies](https://vandvaiproducts.github.io/v-and-v-concept-overview-and-terminologies)**.</span>
  