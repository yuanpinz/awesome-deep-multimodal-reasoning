# Awesome-deep-multimodal-reasoning
Collect the awesome works evolved around reasoning models like O1/R1 in multimodal domain. Welcome to  submit pull requests to contribute if there are omissions in the list.

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

- Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models [[paper]](https://arxiv.org/abs/2505.04921) [[code]](https://github.com/HITsz-TMG/Awesome-Large-Multimodal-Reasoning-Models)
- Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2504.21277)
- Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey [[paper]](https://arxiv.org/abs/2503.12605) [[code]](https://github.com/yaotingwangofficial/Awesome-MCoT)

### Papers/Projects

#### Visual Understanding

- **[Image]** Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning [[paper]](https://arxiv.org/abs/2506.04034) [[code]](https://github.com/IDEA-Research/Rex-Thinker)
- **[Image]** DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models [[paper]](https://arxiv.org/abs/2505.24025) [[code]](https://christinepan881.github.io/DINO-R1/)
- **[Image]** UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.23380) [[code]](https://github.com/showlab/UniRL)
- **[Image]** DIP-R1: Deep Inspection and Perception with RL Looking Through and Understanding Complex Scenes [[paper]](https://arxiv.org/abs/2505.23179)
- **[Image]** VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2505.23504) [[code]](https://github.com/GVCLab/VAU-R1)
- **[Image]** SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.22596)
- **[Image]** OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning [[paper]](https://arxiv.org/abs/2505.22039)
- **[Image]** ACTIVE-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO [[paper]](https://arxiv.org/abs/2505.21457) [[code]](https://github.com/aim-uofa/Active-o3)
- **[Image]** SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards [[paper]](https://arxiv.org/abs/2505.19094) [[code]](https://github.com/justairr/SATORI-R1)
- **[Image]** Align and Surpass Human Camouflaged Perception: Visual Refocus Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2505.19611)
- **[Image]** OpenSeg-R: Improving Open-Vocabulary Segmentation via Step-by-Step Visual Reasoning [[paper]](https://arxiv.org/abs/2505.16974v1) [[code]](https://github.com/Hanzy1996/OpenSeg-R)
- **[Image]** LAD-Reasoner: Tiny Multimodal Models are Good Reasoners for Logical Anomaly Detection [[paper]](https://arxiv.org/abs/2504.12749)
- **[Image]** AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection [[paper]](https://arxiv.org/abs/2504.11914)
- **[Image]** Perception-R1: Pioneering Perception Policy with Reinforcement Learning [[paper]](https://arxiv.org/abs/2504.07954) [[code]](https://github.com/linkangheng/PR1)
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
- **[Image]** VLM-R1: A stable and generalizable R1-style Large Vision-Language Model [[paper]](https://arxiv.org/abs/2504.07615) [[code]](https://github.com/om-ai-lab/VLM-R1) [[VLM-R1 Data]](https://huggingface.co/datasets/omlab/VLM-R1) 
- **[Image]** Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement [[paper]](https://arxiv.org/abs/2503.06520) [[code]](https://github.com/dvlab-research/Seg-Zero) [ [refCOCOg_2k_840]](https://huggingface.co/datasets/Ricky06662/refCOCOg_2k_840)
- **[Image]** **[&Reasoning]** Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.07065) [[code]](https://github.com/ding523/Curr_REFT) [[Curr-ReFT-data]](https://huggingface.co/datasets/ZTE-AIM/Curr-ReFT-data)
- **[Image]** **[&Reasoning]** Improve Vision Language Model Chain-of-thought Reasoning [[paper]](https://arxiv.org/abs/2410.16198) [[code]](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)
- **[3D Object]** Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning [[code]](https://arxiv.org/abs/2503.06232) [[3D-CoT]](https://huggingface.co/datasets/Battam/3D-CoT)
- **[Video]** Chain-of-Thought Textual Reasoning for Few-shot Temporal Action Localization [[paper]](https://arxiv.org/abs/2504.13460)
- **[Video]** FingER: Content Aware Fine-grained Evaluation with Reasoning for AI-Generated Videos [[paper]](https://arxiv.org/abs/2504.10358)
- **[Video]** VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2504.06958) [[code]](https://github.com/OpenGVLab/VideoChat-R1)
- **[Video]** Video-R1: Towards Super Reasoning Ability in Video Understanding [[paper]](https://arxiv.org/pdf/2503.21776) [[code]](https://github.com/tulerfeng/Video-R1) 
- **[Video]** Open R1 Video [[code]](https://github.com/Wang-Xiaodong1899/Open-R1-Video) [[open-r1-video-4k]](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)
- **[Video]** TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM [[paper]](https://arxiv.org/abs/2503.13377) [[code]](https://github.com/www-Ye/TimeZero)
- **[Video]** CoS: Chain-of-Shot Prompting for Long Video Understanding [[paper]](https://arxiv.org/abs/2502.06428) [[code]](https://github.com/lwpyh/CoS_codes)
- **[Video]** **[WWW 2025]** Following Clues, Approaching the Truth: Explainable Micro-Video Rumor Detection via Chain-of-Thought Reasoning [[paper]](https://openreview.net/forum?id=lq2jDWv3w0#discussion) 
- **[Video]** **[&Reasoning]** VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning [[paper]](https://arxiv.org/abs/2503.13444) [[code]](https://github.com/yeliudev/VideoMind)
- **[Omni]** MM-RLHF: The Next Step Forward in Multimodal LLM Alignment [[paper]](https://arxiv.org/abs/2502.10391) [[code]](https://github.com/Kwai-YuanQi/MM-RLHF) [[MM-RLHF Data]](https://huggingface.co/datasets/yifanzhang114/MM-RLHF) [[MM-RLHF-RewardBench]](https://huggingface.co/datasets/yifanzhang114/MM-RLHF-RewardBench)
- **[Omni]** R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.05379) [[code]](https://github.com/HumanMLLM/R1-Omni)

#### Visual Reasoning

- Vision-EKIPL: External Knowledge-Infused Policy Learning for Visual Reasoning [[paper]](https://arxiv.org/abs/2506.06856)
- GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization [[paper]](https://arxiv.org/abs/2506.07160)
- DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO [[paper]](https://arxiv.org/abs/2506.07464) [[code]](https://github.com/mlvlab/DeepVideoR1)
- Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning [[paper]](https://arxiv.org/abs/2506.04559) [[code]](https://github.com/gyhdog99/RACRO2/)
- Seeing the Arrow of Time in Large Multimodal Models [[paper]](https://arxiv.org/abs/2506.03340)
- Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning [[paper]](https://arxiv.org/abs/2506.04207) [[code]](https://github.com/CSfufu/Revisual-R1)
- VisuRiddles: Fine-grained Perception is a Primary Bottleneck for Multimodal Large Language Models in Abstract Visual Reasoning [[paper]](https://arxiv.org/abs/2506.02537) [[code]](https://github.com/yh-hust/VisuRiddles)
- Mixed-R1: Unified Reward Perspective For Reasoning Capability in Multimodal Large Language Models [[paper]](https://arxiv.org/abs/2505.24164) [[code]](https://github.com/xushilin1/mixed-r1)
- ProxyThinker: Test-Time Guidance through Small Visual Reasoners [[paper]](https://arxiv.org/abs/2505.24872) [[code]](https://github.com/MrZilinXiao/ProxyThinker)
- MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.24871) [[project page]](https://modomodo-rl.github.io/)
- Reinforcing Video Reasoning with Focused Thinking [[paper]](https://arxiv.org/abs/2505.24718) [[code]](https://github.com/longmalongma/TW-GRPO)
- Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning [[paper]](https://arxiv.org/abs/2506.04207)
- Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.16965v2) [[code]](https://github.com/Derekkk/Praxis-VLM)
- Grounded Reinforcement Learning for Visual Reasoning [[paper]](http://arxiv.org/abs/2505.23678) [[project page]](https://visually-grounded-rl.github.io/)
- PIXELTHINK: Towards Efficient Chain-of-Pixel Reasoning [[paper]](https://arxiv.org/abs/2505.23727)
- Qwen Look Again: Guiding Vision-Language Reasoning Models to Re-attention Visual Information [[paper]](https://arxiv.org/abs/2505.23558) [[code]](https://github.com/Liar406/Look_Again)
- Fostering Video Reasoning via Next-Event Prediction [[paper]](https://arxiv.org/pdf/2505.22457) [[code]](https://github.com/sail-sg/Video-Next-Event-Prediction)
- Sherlock: Self-Correcting Reasoning in Vision-Language Models [[paper]](https://arxiv.org/abs/2505.22651) [[project page]](https://dripnowhy.github.io/Sherlock/)
- Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start [[paper]](https://arxiv.org/abs/2505.22334) [[code]](https://github.com/waltonfuture/RL-with-Cold-Start)
- Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO [[paper]](https://arxiv.org/abs/2505.22453) [[code]](https://github.com/waltonfuture/MM-UPT)
- TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs [[paper]](https://arxiv.org/abs/2505.20777)
- VerIPO: Cultivating Long Reasoning in Video-LLMs via Verifier-Gudied Iterative Policy Optimization [[paper]](https://arxiv.org/abs/2505.19000) [[code]](https://github.com/HITsz-TMG/VerIPO)
- Foundation Models for Geospatial Reasoning: Assessing Capabilities of Large Language Models in Understanding Geometries and Topological Spatial Relations [[paper]](https://arxiv.org/abs/2505.17136)
- DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.14362v1) [[project page]](https://visual-agent.github.io/)
- Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.15966) [[project page]](https://tiger-ai-lab.github.io/Pixel-Reasoner/)
- Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models [[paper]](https://arxiv.org/abs/2505.16854) [[code]](https://github.com/kokolerk/TON)
- GRIT: Teaching MLLMs to Think with Images [[paper]](https://arxiv.org/abs/2505.15879) [[project page]](https://grounded-reasoning.github.io/)
- R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO [[paper]](https://arxiv.org/abs/2505.16673) [[code]](https://github.com/HJYao00/R1-ShareVL)
- SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward [[paper]](https://arxiv.org/abs/2505.17018) [[code]](https://github.com/kxfan2002/SophiaVL-R1)
- PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models [[paper]](https://arxiv.org/abs/2505.14481)
- Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.14677)
- VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.12081) [[code]](https://github.com/dvlab-research/VisionReasoner)
- UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.14231) [[project page]](https://amap-ml.github.io/UniVG-R1-page/)
- MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision [[paper]](https://arxiv.org/abs/2505.13427) [[code]](https://github.com/ModalMinds/MM-PRM)
- MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO [[paper]](https://arxiv.org/abs/2505.13031) [[code]](https://github.com/EasonXiao-888/MindOmni)
- Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method [[paper]](https://arxiv.org/abs/2505.14197)
- Visual Planning: Let's Think Only with Images [[paper]](https://arxiv.org/abs/2505.11409) [[code]](https://github.com/yix8/VisualPlanning)
- Seeing Beyond the Scene: Enhancing Vision-Language Models with Interactional Reasoning [[paper]](https://arxiv.org/pdf/2505.09118) [[code]](https://github.com/open_upon_acceptance)
- Flash-VL 2B: Optimizing Vision-Language Model Performance for Ultra-Low Latency and High Throughput [[paper]](https://arxiv.org/abs/2505.09498)
- OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.08617) [[code]](https://github.com/zhaochen0110/OpenThinkIMG)
- STOLA: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios [[paper]](https://arxiv.org/abs/2505.04201)
- Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning [[paper]](https://arxiv.org/abs/2505.03318) [[project page]](https://codegoat24.github.io/UnifiedReward/think)
- R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.02835) [[code]](https://github.com/yfzhang114/r1_reward) [[Data]](https://huggingface.co/datasets/yifanzhang114/R1-Reward-RL)
- Fast-Slow Thinking for Large Vision-Language Model Reasoning [[paper]](https://arxiv.org/abs/2504.18458)
- Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning [[paper]](https://arxiv.org/abs/2504.16656v2) [[code]](https://github.com/SkyworkAI/Skywork-R1V) [[Skywork-R1V2-38B]](https://huggingface.co/Skywork/Skywork-R1V2-38B)
- SFT or RL? An Early Investigation into Training R1-Like Reasoning Large Vision-Language Models [[paper]](https://arxiv.org/abs/2504.11468v1) [[code]](https://github.com/UCSC-VLAA/VLAA-Thinking) [[VLAA-Thinking Dataset]](https://huggingface.co/datasets/UCSC-VLAA/VLAA-Thinking)
- NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation [[paper]](https://arxiv.org/abs/2504.13055) [[code]](https://github.com/John-AI-Lab/NoisyRollout)
- VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning [[paper]](https://arxiv.org/abs/2504.08837) [[code]](https://github.com/TIGER-AI-Lab/VL-Rethinker/)
- TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning [[paper]](https://arxiv.org/abs/2504.09641) [[code]](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)
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

- Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning [[paper]](https://arxiv.org/abs/2506.05341) [[project page]](https://directlayout.github.io/)
- Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics [[paper]](https://arxiv.org/abs/2506.00070)
- Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization [[paper]](https://arxiv.org/abs/2505.15660) [[project page]](https://jiaming-zhou.github.io/AGNOSTOS/)
- RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.03238)
- SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning [[paper]](https://arxiv.org/abs/2504.20024) [[project page]](https://spatial-reasoner.github.io/)
- Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning [[paper]](https://arxiv.org/abs/2504.12680)
- Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks [[paper]](https://arxiv.org/abs/2503.21696) [[code]](https://github.com/zwq2018/embodied_reasoner) [[embodied_reasoner]](https://huggingface.co/datasets/zwq2018/embodied_reasoner)
- Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.18013) [[code]](https://github.com/jefferyZhan/Griffon/tree/master/Vision-R1)
- ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos [[paper]](https://arxiv.org/abs/2503.12542) [[code]](https://github.com/WPR001/Ego-ST) [[Ego-ST-bench]](https://huggingface.co/datasets/openinterx/Ego-ST-bench)
- Imagine while Reasoning in Space: Multimodal Visualization-of-Thought [[paper]](https://arxiv.org/abs/2501.07542)
- MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse [[code]](https://github.com/PzySeere/MetaSpatial)
- AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO [[paper]](https://arxiv.org/abs/2502.14669)

#### Medical

- RARL: Improving Medical VLM Reasoning and Generalization with Reinforcement Learning and LoRA under Data and Hardware Constraints [[paper]](https://arxiv.org/abs/2506.06600) [[code]](https://github.com/Hanhpt23/MedicalImagingReasoning)
- MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning [[paper]](https://arxiv.org/abs/2506.00555)
- QoQ-Med: Building Multimodal Clinical Foundation Models with Domain-Aware GRPO Training [[paper]](https://arxiv.org/abs/2506.00711)
- Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling [[paper]](https://arxiv.org/abs/2505.17982) [[code]](https://github.com/bryanwong17/HiVE-MIL)
- Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models [[paper]](https://arxiv.org/abs/2505.13973)
- Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner [[paper]](https://www.arxiv.org/abs/2505.11404)
- VideoPath-LLaVA: Pathology Diagnostic Reasoning Through Video Instruction Tuning [[paper]](https://arxiv.org/abs/2505.04192) [[code]](https://github.com/trinhvg/VideoPath-LLaVA)
- PhysLLM: Harnessing Large Language Models for Cross-Modal Remote Physiological Sensing [[paper]](https://arxiv.org/abs/2505.03621)
- ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification [[paper]](https://arxiv.org/abs/2504.20930)
- Reason Like a Radiologist: Chain-of-Thought and Reinforcement Learning for Verifiable Report Generation [[paper]](https://arxiv.org/abs/2504.18453)
- VLM-based Prompts as the Optimal Assistant for Unpaired Histopathology Virtual Staining [[paper]](https://arxiv.org/abs/2504.15545) 
- Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain [[paper]](https://arxiv.org/abs/2504.13950) [[code]](https://github.com/Qsingle/open-medical-r1)
- How to Detect and Defeat Molecular Mirage: A Metric-Driven Benchmark for Hallucination in LLM-based Molecular Comprehension [[paper]](https://arxiv.org/abs/2504.12314)
- Can DeepSeek Reason Like a Surgeon? An Empirical Evaluation for Vision-Language Understanding in Robotic-Assisted Surgery [[paper]](https://arxiv.org/abs/2503.23130)
- PathVLM-R1: A Reinforcement Learning-Driven Reasoning Model for Pathology Visual-Language Tasks [[paper]](https://arxiv.org/abs/2504.09258)
- GMAI-VL-R1: Harnessing Reinforcement Learning for Multimodal Medical Reasoning [[paper]](https://arxiv.org/abs/2504.01886) [[code]](https://github.com/uni-medical/GMAI-VL-R1)
- PharmAgents: Building a Virtual Pharma with Large Language Model Agents [[paper]](https://arxiv.org/abs/2503.22164)
- Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models [[paper]](https://arxiv.org/abs/2503.13939v2)
- MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning [[paper]](https://arxiv.org/abs/2502.19634v1)
- HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs [[paper]](https://arxiv.org/abs/2412.18925) [[code]](https://github.com/FreedomIntelligence/HuatuoGPT-o1) [[medical-o1-reasoning-SFT]](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- **[EMNLP 2024]** MedCoT: Medical Chain of Thought via Hierarchical Expert [[paper]](https://aclanthology.org/2024.emnlp-main.962/) [[code]](https://github.com/JXLiu-AI/MedCoT)

#### Remote Sensing

- TinyRS-R1: Compact Multimodal Language Model for Remote Sensing [[paper]](https://arxiv.org/abs/2505.12099)
- MilChat: Introducing Chain of Thought Reasoning and GRPO to a Multimodal Small Language Model for Remote Sensing [[paper]](https://arxiv.org/abs/2505.07984)

#### Audio

- Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM? [[paper]](https://arxiv.org/abs/2505.09439)
- WavReward: Spoken Dialogue Models With Generalist Reward Evaluators [[paper]](https://arxiv.org/abs/2505.09558) [[code]](https://github.com/jishengpeng/WavReward)
- **[&Visual]** EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.04623) [[code]](https://github.com/HarryHsing/EchoInk) [[Dataset (AVQA-R1-6K)]](https://huggingface.co/datasets/harryhsing/OmniInstruct_V1_AVQA_R1)
- SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning [[paper]](https://arxiv.org/abs/2504.15900)
- **[&Generation]** Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation [[paper]](https://arxiv.org/abs/2503.19611)
- R1-AQA --- Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering [[paper]](https://arxiv.org/abs/2503.11197) [[code]](https://github.com/xiaomi-research/r1-aqa)
- Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models [[paper]](https://arxiv.org/abs/2503.02318) [[code]](https://github.com/xzf-thu/Audio-Reasoner)

#### Driving

- Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling [[paper]](https://arxiv.org/abs/2505.17659)
- AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving [[paper]](https://arxiv.org/abs/2505.15298)
- HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving [[paper]](https://arxiv.org/abs/2505.15793)
- DSDrive: Distilling Large Language Model for Lightweight End-to-End Autonomous Driving with Unified Reasoning and Planning [[paper]](https://arxiv.org/abs/2505.05360) [[demo]](https://www.youtube.com/watch?v=op8PzQurugY)
- LangCoop: Collaborative Driving with Language [[paper]](https://arxiv.org/abs/2504.13406) 
- AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning [[paper]](https://arxiv.org/abs/2503.07608) [[code]](https://github.com/hustvl/AlphaDrive)

#### GUI

- GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior [[paper]](https://arxiv.org/abs/2506.08012) [[code]](https://penghao-wu.github.io/GUI_Reflection/)
- Look Before You Leap: A GUI-Critic-R1 Model for Pre-Operative Error Diagnosis in GUI Automation [[paper]](https://arxiv.org/abs/2506.04614) [[code]](https://github.com/X-PLUG/MobileAgent/tree/main/GUI-Critic-R1)
- GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents [[paper]](https://arxiv.org/abs/2505.15810)
- UI-R1: Enhancing Efficient Action Prediction of GUI Agents by Reinforcement Learning [[paper]](https://arxiv.org/abs/2503.21620) [[code]](https://github.com/lll6gg/UI-R1)
- GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents [[paper]](https://arxiv.org/abs/2504.10458) [[code]](https://github.com/taco-group/LangCoop)

#### Other Modalities

- **[Tabular]** Multimodal Tabular Reasoning with Privileged Structured Information [[paper]](https://arxiv.org/abs/2506.04088)
- Shadow Wireless Intelligence: Large Language Model-Driven Reasoning in Covert Communications [[paper]](https://arxiv.org/abs/2505.04068)

#### Generation

- **[Image]** Reason-SVG: Hybrid Reward RL for Aha-Moments in Vector Graphics Generation [[paper]()](https://arxiv.org/abs/2505.24499)
- **[Image]** ReasonGen-R1: CoT for Autoregressive Image generation models through SFT and RL [[paper]](https://arxiv.org/abs/2505.24875) [[project page]](https://aka.ms/reasongen)
- **[Video]** Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation [[paper]](https://arxiv.org/abs/2505.21653) [[project page]](https://bwgzk-keke.github.io/DiffPhy/)
- **[Image]** Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment [[paper]](https://arxiv.org/abs/2505.18600) [[project page]](https://bryanswkim.github.io/chain-of-zoom/)
- **[Image]** RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.17540) [[code]](https://github.com/microsoft/DKI_LLM/tree/main/RePrompt)
- **[Video]** InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO [[paper]](https://arxiv.org/abs/2505.17574) [[code]](https://github.com/MAPLE-AIGC/InfLVG)
- **[Image]** Co-Reinforcement Learning for Unified Multimodal Understanding and Generation [[paper]](https://arxiv.org/abs/2505.17534)
- **[Image]** GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning [[paper]](https://arxiv.org/abs/2505.17022) [[code]](https://github.com/gogoduan/GoT-R1)
- **[Image]** Flow-GRPO: Training Flow Matching Models via Online RL [[paper]](https://www.arxiv.org/abs/2505.05470) [[code]](https://github.com/yifan123/flow_grpo)
- **[Traffic]** RIFT: Closed-Loop RL Fine-Tuning for Realistic and Controllable Traffic Simulation [[paper]](https://arxiv.org/abs/2505.03344) [[project page]](https://currychen77.github.io/RIFT/)
- **[Image]** T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT [[paper]](https://arxiv.org/abs/2505.00703) [[code]](https://github.com/CaraJ7/T2I-R1)
- **[Code]** AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers [[paper]](https://arxiv.org/abs/2504.20115) [[code]](https://github.com/shoushouyu/Automated-Paper-to-Code)
- **[Video]** Reasoning Physical Video Generation with Diffusion Timestep Tokens via Reinforcement Learning [[paper]](https://arxiv.org/abs/2504.15932)
- **[Image]** SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL [[paper]](https://arxiv.org/abs/2504.11455)
- **[Image]** GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing [[paper]](https://arxiv.org/abs/2503.10639) [[code]](https://github.com/rongyaofang/GoT)
- **[Image]** **[CVPR 2025]** Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[paper]](https://arxiv.org/abs/2501.13926) [[code]](https://github.com/ZiyuGuo99/Image-Generation-CoT)
- **[Audio]** **[ICLR 2025]** Both Ears Wide Open: Towards Language-Driven Spatial Audio Generation [[paper]](https://arxiv.org/abs/2410.10676) [[code]](https://github.com/PeiwenSun2000/Both-Ears-Wide-Open)

#### Benchmarks

- AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs [[paper]](https://arxiv.org/abs/2506.05328) [[project page]](https://av-reasoner.github.io/)
- **[Video]** MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos [[paper]](https://arxiv.org/abs/2506.04141) [[project page]](https://mmr-v.github.io/)
- **[Medical]** DrVD-Bench: Do Vision-Language Models Reason Like Human Doctors in Medical Image Diagnosis? [[paper]](https://arxiv.org/abs/2505.24173) [[code]](https://github.com/Jerry-Boss/DrVD-Bench)
- **[Medical]** Image Aesthetic Reasoning: A New Benchmark for Medical Image Screening with MLLMs [[paper]](https://arxiv.org/abs/2505.23265)
- **[Medical]** ER-REASON: A Benchmark Dataset for LLM-Based Clinical Reasoning in the Emergency Room [[paper]](https://arxiv.org/abs/2505.22919) [[code]](https://github.com/AlaaLab/ER-Reason)
- OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning [[paper]](https://arxiv.org/abs/2505.17163) [[code]](https://github.com/SCUT-DLVCLab/OCR-Reasoning)
- LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models [[paper]](https://arxiv.org/abs/2505.15616) [[code]](https://github.com/Lens4MLLMs/lens)
- Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans [[paper]](https://www.arxiv.org/abs/2505.11141) [[project page]](https://yansheng-qiu.github.io/human-aligned-bench.github.io/)
- VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models[[paper]](https://arxiv.org/abs/2505.08455) [[project page]](https://pritamsarkar.com/VCRBench/)
- BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning [[paper]](https://arxiv.org/abs/2505.07889) [[code]](https://github.com/YuyangSunshine/bioprotocolbench)
- **[Robotics]** Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks [[paper]](https://openreview.net/pdf?id=z0MM0y20I2)
- VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations for Synthetic Videos [[paper]](https://arxiv.org/abs/2505.01481) [[code]](https://github.com/zli12321/VideoHallu)
- **[ICML 2025]** R-Bench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation [[paper]](https://arxiv.org/abs/2505.02018) [[project page]](https://evalmodels.github.io/rbench/)
- GDI-Bench: A Benchmark for General Document Intelligence with Vision and Reasoning Decoupling [[paper]](https://arxiv.org/abs/2505.00063)
- VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models [[paper]](https://arxiv.org/abs/2504.15279) [[code]](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval) [[Page]](https://visulogic-benchmark.github.io/VisuLogic/) [[Benchmark]](https://huggingface.co/datasets/VisuLogic/VisuLogic) [[Train Data]](https://huggingface.co/datasets/VisuLogic/VisuLogic-Train) [[Train Code]](https://github.com/VisuLogic-Benchmark/VisuLogic-Train)
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

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuanpinz/awesome-deep-multimodal-reasoning&type=Date)](https://www.star-history.com/#yuanpinz/awesome-deep-multimodal-reasoning&Date)


