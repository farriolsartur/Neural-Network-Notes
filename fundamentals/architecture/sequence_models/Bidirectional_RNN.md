# Bidirectional RNN (BRNN)

## Introduction

Bidirectional Recurrent Neural Networks (BRNNs) are a sophisticated variant of traditional RNNs designed to address a fundamental limitation in sequential input processing. In standard RNNs, the meaning of a word or input element might depend on another that hasn't yet appeared in the sequence. This can lead to incomplete or incorrect interpretations of the input data.

For example, consider the following sentences:
1. "Rex barked loudly at the mailman."
2. "The Rex roared fiercely in the Jurassic period."

In these sentences, the meaning of "Rex" is only fully understood later in the context. BRNNs are designed to capture both past and future context, allowing for more accurate predictions and interpretations.

It's worth noting that the RNN blocks in Bidirectional RNNs can be substituted by more advanced recurrent units such as Gated Recurrent Units (GRU) or Long Short-Term Memory (LSTM) blocks, further enhancing their capability to capture long-term dependencies.

However, BRNNs have a limitation: they require the entire sequence of data to make a prediction. This makes them challenging to use in scenarios where data flows continuously, such as real-time speech-to-text applications.

## Architecture

To understand the architecture of BRNNs, let's first look at the structure of a standard RNN unit:

```mermaid
graph LR
    FDOTS[...] --> A0
    X1[X<sub>t</sub>] --> RNN1((RNN))
    A0[a<sub>t-1</sub>] --> RNN1
    RNN1 --> A1[a<sub>t</sub>]
    RNN1 --> Y1[y<sub>t</sub>]
    
    
    A1 --> DOTS[...]
    
    style RNN1 fill:#f9f,stroke:#333,stroke-width:4px
    style A0 fill:#a9f,stroke:#333,stroke-width:2px
    style A1 fill:#a9f,stroke:#333,stroke-width:2px
```

Now, let's look at the architecture of a Bidirectional RNN:

```mermaid
graph LR



    X1[X<sub>t</sub>]
    Y1[y<sub>t</sub>]

    subgraph Backward
        direction RL
        BDOTS[...] --> B0
        B0[b<sub>t-1</sub>] --> RNN2((RNN))
        RNN2 --> B1[b<sub>t</sub>]
        B1 --> BDOTS2[...]
    end

    subgraph Forward
        direction LR
        FDOTS[...] --> A0
        A0[a<sub>t-1</sub>] --> RNN1((RNN))
        RNN1 --> A1[a<sub>t</sub>]
        A1 --> DOTS[...]
    end


    X1 --> Forward
    X1 --> Backward
    Forward --> Y1
    Backward --> Y1


    style RNN1 fill:#f9f,stroke:#333,stroke-width:4px
    style RNN2 fill:#9ff,stroke:#333,stroke-width:4px
    style A0 fill:#a9f,stroke:#333,stroke-width:2px
    style A1 fill:#a9f,stroke:#333,stroke-width:2px
    style B0 fill:#aff,stroke:#333,stroke-width:2px
    style B1 fill:#aff,stroke:#333,stroke-width:2px
    style X1 fill:#ffa,stroke:#333,stroke-width:2px
    style Y1 fill:#ffa,stroke:#333,stroke-width:2px
```

Note that X<sub>t</sub> should be connected to the RNN units and so should Y<sub>t</sub> but this occasionates a mermaid bug.

In a BRNN, each time step t has both a forward ($a^t_{}$) and backward ($b^t_{}$) hidden state. The output at each time step is typically computed as a function of both these hidden states:

$\hat{y}^t = g(W_y[a^t_{}, b^t_{}])$

Where $g$ is an activation function, $W_y$ is a weight matrix, and $[a^t_{}, b^t_{}]$ represents the concatenation of the forward and backward hidden states.
This architecture allows the network to incorporate information from both past and future context at every time step, leading to more informed predictions.