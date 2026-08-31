# "Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators

This repository **will provide** resources developed within the following article:

> Krisztian Balog and Arild Michel Bakken. **"Act Like a 5th Grader" is Not Enough: Bounding Knowledge in LLM-Based User Simulators.** In: Findings of the Association for Computational Linguistics: EMNLP 2026. Association for Computational Linguistics. 2026.

## Summary

Large language models (LLMs) are increasingly used to simulate human behavior but frequently fail to exhibit realistic cognitive constraints, suffering from a "superhuman bias." Using a dataset of over 71,000 reading comprehension responses from 2,359 primary-school students (grades 4–6), we demonstrate that standard persona prompting yields near-perfect, deterministic performance, failing to capture the natural variance of developing readers. To address this, we introduce the Cognitively Bounded User Simulator (CBUS), an architectural framework that explicitly models the restricted working memory of young readers through an episodic bottleneck. Within this framework, we formalize two distinct test-taking strategies to emulate different reading behaviors. Our evaluation shows that explicitly modeling cognitive bounds significantly narrows the simulation gap across multiple LLM backbones, demonstrating that enforcing architectural constraints is more effective for high-fidelity simulation than simply scaling raw model capabilities.

## Contents

COMING SOON

## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Balog:2026:EMNLP,
  author =    {Balog, Krisztian and Bakken, Arild Michel},
  title =     {"Act Like a 5th Grader" is Not Enough: Bounding Knowledge in {LLM}-Based User Simulators},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  publisher = {Association for Computational Linguistics},
  year =      {2026}
}
```

## Contact

Should you have any questions, please contact Krisztian Balog at `krisztian.balog`[AT]uis.no (with [AT] replaced by @).
