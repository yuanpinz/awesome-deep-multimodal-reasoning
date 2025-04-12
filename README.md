# Awesome-deep-multimodal-reasoning
Collect the awesome works evolved around reasoning models like O1/R1 in multimodal domain.

> I distinguish between **Visual Understanding** and **Visual Reasoning** based on my personal interpretation - while an ideal model should possess both capabilities, different research efforts tend to emphasize one aspect over the other.
>
> - **Visual Understanding** focuses on image-centric comprehension, typically employing vision-language tasks like RefCOCO (as used in VLM-R1).
> - **Visual Reasoning** emphasizes logical inference where visual information primarily supplements textual context, exemplified by tasks like MathVision.
>
> Vertical Domains like **Robotics/Spatial**, **Medical**, and **Audio** are categorized separately, though their methodologies often overlap with either Visual Understanding or Reasoning approaches.
>
> The **Generation** category presents an interesting intersection, leveraging reasoning-based methods to produce multimodal content.
>
> **Datasets** here refer to aggregated resources not specifically tied to individual projects - primarily including R1/O1 distilled datasets and benchmark collections.
>
> **Infrastructure** encompasses essential libraries and tools for model training.
>
> **Related Collections** serves as a catch-all category for other relevant resources.

### Surveys

- Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey [[paper]](https://arxiv.org/abs/2503.12605) [[code]](https://github.com/yaotingwangofficial/Awesome-MCoT)

### Papers/Projects

#### Visual Understanding

- **[Image]** On the Suitability of Reinforcement Fine-Tuning to Visual Tasks [[paper]](https://arxiv.org/abs/2504.05682)
- **[Image]** OmniCaptioner: One Captioner to Rule Them All [[paper]](https://arxiv.org/abs/2504.07089) [[code]](https://github.com/Alpha-Innovator/OmniCaptioner)
- **[Image]** CrowdVLM-R1: Expanding R1 Ability to Vision Language Model for Crowd Counting using Fuzzy Group Relative Policy Reward [[paper]](https://arxiv.org/abs/2504.03724) [[code]](https://github.com/yeyimilk/CrowdVLM-R1)
- **[Image]** Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme [[Paper]](https://arxiv.org/abs/2504.02587) [[code]](https://github.com/GAIR-NLP/MAYE) [[Datasets]](https://huggingface.co/datasets/ManTle/MAYE)
- **[Image]** Q-Insight: Understanding Image Quality via Visual Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.22679) [[code]](https://github.com/lwq20020127/Q-Insight)
- **[Image]** CLS-RL: Image Classification with Rule-Based Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.16188)
- **[Image]** Grounded Chain-of-Thought for Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2503.12799)
- **[Image]** VisRL: Intention-Driven Visual Perception via Reinforced Reasoning [[paper]](https://arxiv.org/abs/2503.07523) [[code]](https://github.com/zhangquanchen/VisRL)
- **[Image]** DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding [[paper]](https://arxiv.org/abs/2503.12797) [[code]](https://github.com/thunlp/DeepPerception) [[KVG]](https://huggingface.co/datasets/MaxyLee/KVG) 
- **[Image]** Visual Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2503.01785) [[code]](https://github.com/Liuziyu77/Visual-RFT) [[ViRFT]](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df)
- **[Image]** VLM-R1: A stable and generalizable R1-style Large Vision-Language Model [[code]](https://github.com/om-ai-lab/VLM-R1) [[VLM-R1 Data]](https://huggingface.co/datasets/omlab/VLM-R1) 
- **[Image]** Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement [[paper]](https://arxiv.org/abs/2503.06520) [[code]](https://github.com/dvlab-research/Seg-Zero) [ [refCOCOg_2k_840]](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840)
- **[Image]** **[&Reasoning]** Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.07065)
- **[Image]** **[&Reasoning]** Improve Vision Language Model Chain-of-thought Reasoning [[paper]](https://arxiv.org/abs/2410.16198) [[code]](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)
- **[3D Object]** Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning [[code]](https://arxiv.org/abs/2503.06232) [[3D-CoT]](https://huggingface.co/datasets/Battam/3D-CoT)
- **[Video]** VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2504.06958) [[code]](https://github.com/OpenGVLab/VideoChat-R1)
- **[Video]** Video-R1: Towards Super Reasoning Ability in Video Understanding [[paper]](https://arxiv.org/pdf/2503.21776) [[code]](https://github.com/tulerfeng/Video-R1) 
- **[Video]** Open R1 Video [[code]](https://github.com/Wang-Xiaodong1899/Open-R1-Video) [[open-r1-video-4k]](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)
- **[Video]** TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM [[paper]](https://arxiv.org/abs/2503.13377) [[code]](https://github.com/www-Ye/TimeZero)
- **[Video]** CoS: Chain-of-Shot Prompting for Long Video Understanding [[paper]](https://arxiv.org/abs/2502.06428) [[code]](https://github.com/lwpyh/CoS_codes)
- **[Video]** **[WWW2025]** Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning [[paper]](https://openreview.net/forum?id=lq2jDWv3w0#discussion) 
- **[Video]** **[&Reasoning]** VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning [[paper]](https://arxiv.org/abs/2503.13444) [[code]](https://github.com/yeliudev/VideoMind)
- **[Omni]** MM-RLHF: The Next Step Forward in Multimodal LLM Alignment [[paper]](https://arxiv.org/abs/2502.10391) [[code]](https://github.com/Kwai-YuanQi/MM-RLHF) [[MM-RLHF Data]](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) [[MM-RLHF-RewardBench]](https://huggingface.co/datasets/yifanzhang114/MM-RLHF-RewardBench)
- **[Omni]** R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.05379) [[code]](https://github.com/HumanMLLM/R1-Omni)

#### Visual Reasoning

- OThink-MR1: Stimulating multimodal generalized reasoning capabilities via dynamic reinforcement learning [[paper]](https://arxiv.org/abs/2503.16081)
- LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning? [[paper]](https://arxiv.org/abs/2503.19990)
- Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning [[paper]](https://arxiv.org/abs/2503.20752)
- OpenVLThinker: An Early Exploration to Vision-Language Reasoning via Iterative Self-Improvement [[Paper]](https://arxiv.org/abs/2503.17352) [[code]](https://github.com/yihedeng9/OpenVLThinker)
- SciVerse: Unveiling the Knowledge Comprehension and Visual Reasoning of LMMs on Multi-modal Scientific Problems  [[paper]](https://arxiv.org/abs/2503.10627) [[SciVerse](https://huggingface.co/datasets/ZiyuG/SciVerse)] [[LLaVA-NeXT-Interleave-Bench]](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Interleave-Bench)
- R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization [[paper]](https://arxiv.org/abs/2503.12937) [[code]](https://github.com/jingyi0000/R1-VL)
- Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models? [[paper]](https://arxiv.org/abs/2503.06252) [[code]](https://github.com/Quinn777/AtomThink)
- VisualPRM: An Effective Process Reward Model for Multimodal Reasoning [[paper]](https://arxiv.org/abs/2503.10291) [[code]](https://huggingface.co/OpenGVLab/VisualPRM-8B)  [[VisualPRM400K]](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K)
- Kimi k1.5: Scaling Reinforcement Learning with LLMs [[paper]](https://arxiv.org/abs/2501.12599) 
- R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization [[paper]](https://arxiv.org/pdf/2503.10615) [[code]](https://github.com/Fancy-MLLM/R1-Onevision) [[R1-Onevision Data]](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) 
- MMR1: Advancing the Frontiers of Multimodal Reasoning [[code]](https://github.com/LengSicong/MMR1) [[MMR1-Math-RL-Data-v0]](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0)
- LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL [[paper]](https://arxiv.org/abs/2503.07536) [[code]](https://github.com/TideDra/lmm-r1) [[VerMulti]](https://huggingface.co/datasets/VLM-Reasoner/VerMulti) 
- R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model [[paper]](https://arxiv.org/abs/2503.05132) [code](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)
- R1-Vision: Let's first take a look at the image [[code]](https://github.com/yuyq96/R1-Vision/tree/main) [[R1-Vision Data]](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83)
- MM-EUREKA: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning  [[paper]](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf) [[code]](https://github.com/ModalMinds/MM-EUREKA) [[MM-Eureka-Dataset]](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset/tree/main)
- Multimodal Open R1 [[code]](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) [[multimodal-open-r1-8k-verified]](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 
- VL-Thinking: An R1-Derived Visual Instruction Tuning Dataset for Thinkable LVLMs [[code]](https://github.com/UCSC-VLAA/VL-Thinking) [[VL-Thinking]](https://huggingface.co/datasets/UCSC-VLAA/VL-Thinking)
- R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3 [[code]](https://github.com/Deep-Agent/R1-V) [[R1V Training Dataset: CLEVR-70k-Counting]](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train) [[R1V Training Dataset: CLEVR-70k-Complex]](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_70K_Complex) [[R1V Training Dataset: GEOQA-8k]](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K) [[R1-Distilled Visual Reasoning Dataset]](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1)
- LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs [[paper]](https://arxiv.org/abs/2501.06186) [[code]](https://github.com/mbzuai-oryx/LlamaV-o1) [[VRC-Bench]](https://huggingface.co/datasets/omkarthawakar/VRC-Bench) 
- Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2503.06749) [[code]](https://github.com/Osilly/Vision-R1/tree/main)

#### Robotics/Spatial

- Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks [[paper]](https://arxiv.org/abs/2503.21696) [[code]](https://github.com/zwq2018/embodied_reasoner) [[embodied_reasoner]](https://huggingface.co/datasets/zwq2018/embodied_reasoner)
- Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.18013) [[code]](https://github.com/jefferyZhan/Griffon/tree/master/Vision-R1)
- ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos [[paper]](https://arxiv.org/abs/2503.12542) [[code]](https://github.com/WPR001/Ego-ST) [[Ego-ST-bench]](https://huggingface.co/datasets/openinterx/Ego-ST-bench)
- Imagine while Reasoning in Space: Multimodal Visualization-of-Thought [[paper]](https://arxiv.org/abs/2501.07542)
- MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse [[code]](https://github.com/PzySeere/MetaSpatial)
- AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO [[paper]](https://arxiv.org/abs/2502.14669)
- AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning [[paper]](https://arxiv.org/abs/2503.07608) [[code]](https://github.com/hustvl/AlphaDrive)

#### Medical

- GMAI-VL-R1: Harnessing Reinforcement Learning for Multimodal Medical Reasoning [[paper]](https://arxiv.org/abs/2504.01886) [[code]](https://github.com/uni-medical/GMAI-VL-R1)
- PharmAgents: Building a Virtual Pharma with Large Language Model Agents [[paper]](https://arxiv.org/abs/2503.22164)
- Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models [[paper]](https://arxiv.org/abs/2503.13939v2)
- MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning [[paper]](https://arxiv.org/abs/2502.19634v1)
- HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs [[paper]](https://arxiv.org/abs/2412.18925) [[code]](https://github.com/FreedomIntelligence/HuatuoGPT-o1) [[medical-o1-reasoning-SFT]](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **[EMNLP2024]** MedCoT: Medical Chain of Thought via Hierarchical Expert [[paper]](https://aclanthology.org/2024.emnlp-main.962/) [[code]](https://github.com/JXLiu-AI/MedCoT)

#### Audio

- **[&Generation]** Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation [[paper]](https://arxiv.org/abs/2503.19611)
- R1-AQA --- Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering [[paper]](https://arxiv.org/abs/2503.11197) [[code]](https://github.com/xiaomi-research/r1-aqa)
- Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models [[paper]](https://arxiv.org/abs/2503.02318) [[code]](https://github.com/xzf-thu/Audio-Reasoner)

#### Generation

- GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing [[paper]](https://arxiv.org/abs/2503.10639) [[code]](https://github.com/rongyaofang/GoT)
- **[CVPR2025]** Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[paper]](https://arxiv.org/abs/2501.13926) [[code]](https://github.com/ZiyuGuo99/Image-Generation-CoT)
- **[ICLR2025]** Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation [[paper]](https://arxiv.org/abs/2410.10676) [[code]](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open)

#### Benchmarks

- MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2504.05782) [[code]](https://github.com/LanceZPF/MDK12) [[Datasets]](https://github.com/LanceZPF/MDK12#-datasets) [[Leaderboard]](https://github.com/LanceZPF/MDK12#-leaderboard)
- MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models [[paper]](https://arxiv.org/abs/2502.00698) [[code]](https://github.com/AceCHQ/MMIQ)
- MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency [[paper]](https://arxiv.org/abs/2502.09621) [[code]](https://github.com/CaraJ7/MME-CoT)
- ZeroBench: An Impossible* Visual Benchmark for Contemporary Large Multimodal Models [[paper]](https://arxiv.org/pdf/2502.09696) [[code]](https://github.com/jonathan-roberts1/zerobench/) 

### Datasets

- **[Training]** [LLaVA-R1-100k](https://www.modelscope.cn/datasets/modelscope/LLaVA-R1-100k) - LLaVA多模态Reasoning数据集
- **[Benchmarking]** [MMMU-Reasoning-R1-Distill-Validation](https://www.modelscope.cn/datasets/modelscope/MMMU-Reasoning-Distill-Validation) - MMMU-满血版R1蒸馏多模态Reasoning验证集

### Infra

- EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL [[code]](https://github.com/hiyouga/EasyR1)
- verl: Volcano Engine Reinforcement Learning for LLM [[code]](https://github.com/volcengine/verl)
- TRL - Transformer Reinforcement Learning [[code]](https://github.com/huggingface/trl)
- Align-Anything: Training All-modality Model with Feedback [[code]](https://github.com/PKU-Alignment/align-anything)
- R-Chain: A lightweight toolkit for distilling reasoning models [[code]](https://github.com/modelscope/r-chain)

### Related Collections

- [Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs](https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs)
- [yaotingwangofficial/Awesome-MCoT](https://github.com/yaotingwangofficial/Awesome-MCoT)
- [modelscope/awesome-deep-reasoning](https://github.com/modelscope/awesome-deep-reasoning)
- [atfortes/Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning)
- [hijkzzz/Awesome-LLM-Strawberry](https://github.com/hijkzzz/Awesome-LLM-Strawberry)
- [srush/awesome-o1](https://github.com/srush/awesome-o1)

