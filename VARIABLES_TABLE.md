\begin{table}[htbp]
\centering
\caption{Variables used for dengue prediction}
\label{tab:variables}
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\textwidth}{lX}
\hline
\textbf{Variable name (number of factors)} & \textbf{Description} \\
\hline

Dengue case data (1) &
Number of confirmed dengue cases aggregated by county and date \\
\hline

Temporal features (1) &
Historical dengue cases in the previous 14 days (sliding window) \\
\hline

Spatial features (1) &
County-level spatial relationships (fully connected graph structure) \\
\hline

Epidemiological parameters (2) &
Infection rate ($\beta$) and recovery rate ($\gamma$) in SIS model \\
\hline

\end{tabularx}
\end{table}

