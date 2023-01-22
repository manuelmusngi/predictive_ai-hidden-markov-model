### hidden markov model

Application of hidden markov model in time series pattern recognition and market regimes.

== Definition ==

$Let <math>X_n</math> and <math>Y_n</math> be discrete-time [[stochastic process]]es and <math>n\geq 1</math>. The pair <math>(X_n,Y_n)</math> is a ''hidden Markov model'' if$

$ * <math>X_n</math> is a [[Markov process]] whose behavior is not directly observable ("hidden");$
$ * <math>\operatorname{\mathbf{P}}\bigl(Y_n \in A\ \bigl|\ X_1=x_1,\ldots,X_n=x_n\bigr)=\operatorname{\mathbf{P}}\bigl(Y_n \in A\ \bigl|\ X_n=x_n\bigr),</math>$
$ :for every <math>n\geq 1,</math> <math>x_1,\ldots, x_n,</math> and every [[Borel set|Borel]] set <math>A</math>.$

Let <math>X_t</math> and <math>Y_t</math> be continuous-time stochastic processes. The pair <math>(X_t,Y_t)</math> is a ''hidden Markov model'' if
*<math>X_t</math> is a Markov process whose behavior is not directly observable ("hidden");
$$*<math>\operatorname{\mathbf{P}}(Y_{t_0} \in A \mid \{X_t \in B_t\}_{ t\leq t_0}) = \operatorname{\mathbf{P}}(Y_{t_0} \in A \mid X_{t_0} \in B_{t_0})</math>,$$

:for every <math> t_0, </math> every Borel set <math> A, </math> and every family of Borel sets <math> \{B_t\}_{t \leq t_0}. </math>

=== Terminology ===
The states of the process <math>X_n</math> (resp. <math>X_t)</math> are called ''hidden states'', and <math>\operatorname{\mathbf{P}}\bigl(Y_n \in A \mid X_n=x_n\bigr)</math> (resp. <math>\operatorname{\mathbf{P}}\bigl(Y_t \in A \mid X_t \in  B_t\bigr))</math> is called ''emission probability'' or ''output probability''.
$


- File Structure
  
  - src/

- Requirements

- References
  - [A Hybrid Learning Approach to Detecting Regime Switches in Financial Markets](https://arxiv.org/abs/2108.05801)
  - [Market Regime Identification Using Hidden Markov Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3406068)
  - [Predicting Daily Probability Distributions of S&P500 Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1288468)
  - [Unraveling S&P500 stock volatility and networks -- An encoding-and-decoding approach](https://arxiv.org/abs/2101.09395)
  - [Hidden Markov Models Applied To Intraday Momentum Trading ](https://arxiv.org/abs/2006.08307)
