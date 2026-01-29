\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{SIS}} \cdot \mathcal{L}_{\text{SIS}}
\end{equation}

where $\mathcal{L}_{\text{data}}$ is the data fidelity loss:

\begin{equation}
\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{I}_{t+1}^{(i)} - I_{t+1}^{(i)} \right)^2
\end{equation}

and $\mathcal{L}_{\text{SIS}}$ is the SIS consistency regularization loss:

\begin{equation}
\mathcal{L}_{\text{SIS}} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{I}_{t+1}^{(i)} - I_{t+1}^{\text{SIS},(i)} \right)^2
\end{equation}

The SIS model prediction $I_{t+1}^{\text{SIS}}$ is computed as:

\begin{equation}
I_{t+1}^{\text{SIS}} = I_t + \beta \cdot \frac{(N - I_t) \cdot I_t}{N} - \gamma \cdot I_t
\end{equation}

where $\beta$ is the infection rate, $\gamma$ is the recovery rate, $N$ is the total population (normalized to 1), and $\lambda_{\text{SIS}}$ is the regularization weight (default: 0.1).

