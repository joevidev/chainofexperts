
# Exploring Communicative Mixture of Experts (MoE) Models


## Background
The concept of a Mixture of Experts (MoE) model can be likened to a system where multiple specialized experts process different aspects of a given input, and their outputs are combined to generate a final prediction. This approach is reminiscent of how customer service functions in an industry setting, where an issue is forwarded to different agents, each analyzing and addressing specific aspects before arriving at a resolution.

A fine-grained MoE model differs from a simpler MoE model in that each token is processed by a subset of experts rather than a single expert. In large-scale MoE architectures, multiple experts exist, but only a few are activated per token, ensuring that processing is distributed efficiently. This idea aligns with a scenario where a problem is assigned to a specialized group of experts, who collaborate to resolve it at a granular level.

One analogy that comes to mind is how a token is processed in a feed-forward network with multiple layers, separated by activation functions. If a token passes through multiple experts—say four—it will generate four distinct activations. Instead of selecting just one, the model could average these activations, thereby leveraging multiple perspectives. This is comparable to a customer service system where various experts collaborate before delivering a final response to the customer.

We would like to extend this concept into a research project, where We conduct an experiment to test the efficacy of a communicative MoE model. My plan is to create a minimal prototype, starting with a basic implementation. We intend to generate around 2 test cases, with one set based on a traditional MoE model and another employing a communicative MoE approach. By running these models on a dataset, We aim to observe how communication among experts influences outcomes.

To control whether experts communicate, We will implement a switch within the model's configuration. This will allow me to compare performance between communicative and non-communicative setups. My hypothesis is that a communicative approach may lead to lower loss values, demonstrating improved efficiency when experts collaborate.

The final deliverable for this project will be a simple, well-documented code repository implementation, including instructions on running the experiment. The results will be presented in a structured format, allowing for further analysis and discussion.

## Project Structure

```
CoMoE/
    config/
        r1-tiny-hf/
            config.json
            ...
        r1-hf/
            config.json
            ...
```