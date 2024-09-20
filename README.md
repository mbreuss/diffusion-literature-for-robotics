# Diffusion-Literature-for-Robotics

> "Creating noise from data is easy; creating data from noise is generative modeling."

Yang Song in "Score-Based Generative Modeling through Stochastic Differential Equations" [Song et al., 2020](https://arxiv.org/pdf/2011.13456)

This repository offers a brief summary of essential papers and blogs on diffusion models, alongside a categorized collection of robotics diffusion papers and useful code repositories for starting your own diffusion robotics project.

---

# Table of Contents
1. [Learning about Diffusion models](#Learning-about-Diffusion-models)
2. [Diffusion in Robotics](#Diffusion-in-Robotics)

    2.1 [Imitation Learning and Policy Learning](#Imitation-Learning-and-Policy-Learning)

    2.2 [Video Diffusion in Robotics](#Video-Diffusion)

    2.3 [Online RL](#Online-RL)
    
    2.4 [Offline RL](#Offline-RL)

    2.5 [Inverse RL](#Inverse-RL)

    2.6 [World Models](#World-Models)

    2.7 [Task and Motion Planning](#tamp)

    2.8 [Tactile Sensing & Pose Estimation](#Grasping-&-Tactile-Sensing-&-Pose-Estimation)

    2.9 [Robot Design and Development](#Robot-Design-and-Construction)

3. [Code Implementations](#Code-Bases)

4. [Diffusion History](#Diffusion-History)

---

## Learning about Diffusion models 
<a name="Learning-about-Diffusion-models"></a>
While there exist many tutorials for Diffusion models, below you can find an overview of some of the best introduction blog posts and video:

- [What are Diffusion Models?](https://www.youtube.com/watch?v=fbLgFrlTnGU&t=1s): an introduction video, which introduces the general idea of diffusion models and some high-level math about how the model works

- [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) another great video tutorial explaining the math and notation of diffusion models in detail with visual aid

- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/): blog post from the one of the most influential authors in this area, which introduces diffusion models from the score-based perspective 
- [What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): a in-depth blog post about the theory of diffusion models with a general  summary on how diffusion model improved over time 
- [Understanding Diffusion Models](https://arxiv.org/pdf/2208.11970.pdf): an in-depth explanation paper, which explains the diffusion models from both perspectives with detailed derivations

If you don't like reading blog posts and prefer the original papers, below you can find a list with the most important diffusion theory papers:

- Sohl-Dickstein, Jascha, et al. ["Deep unsupervised learning using nonequilibrium thermodynamics."](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf) _International Conference on Machine Learning_. PMLR, 2015.

- Ho, Jonathan, et al. ["Denoising diffusion probabilistic models."](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf) _Advances in Neural Information Processing Systems_ 33 (2020): 6840-6851.
- Song, Yang, et al. ["Score-Based Generative Modeling through Stochastic Differential Equations."](https://arxiv.org/pdf/2011.13456) _International Conference on Learning Representations_. 2020.

- Ho, Jonathan, and Tim Salimans. ["Classifier-Free Diffusion Guidance."](https://arxiv.org/pdf/2207.12598) _NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications_. 2021.

- Karras, Tero, et al. ["Elucidating the Design Space of Diffusion-Based Generative Models."](https://arxiv.org/pdf/2206.00364) _Advances in Neural Information Processing Systems_ 35 (2022)

A general list with all published diffusion papers can be found here: [Whats the score?](https://scorebasedgenerativemodeling.github.io/)

---

## Diffusion in Robotics
<a name="Diffusion-in-Robotics"></a>
Since the modern diffusion models have been around for only 3 years, the literature about diffusion models in the context of robotics is still small, but growing rapidly. Below you can find most robotics diffusion papers, which have been published at conferences or uploaded to Arxiv so far:

---

### Imitation Learning and Policy Learning
<a name="Imitation-Learning-and-Policy-Learning"></a>

- Ke, Tsung-Wei, Nikolaos Gkanatsios, and Katerina Fragkiadaki. ["3D Diffuser Actor: Policy Diffusion with 3D Scene Representations."](https://arxiv.org/abs/2402.10885) 8th Annual Conference on Robot Learning.

- Djeumou, Franck, et al. ["One Model to Drift Them All: Physics-Informed Conditional Diffusion Model for Driving at the Limits."](https://openreview.net/pdf?id=0gDbaEtVrd) 8th Annual Conference on Robot Learning.

- Blank, Nils, et al. ["Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models."](https://openreview.net/pdf?id=EdVNB2kHv1) 8th Annual Conference on Robot Learning.

- Wang, Yixiao, et al. ["Sparse Diffusion Policy: A Sparse, Reusable, and Flexible Policy for Robot Learning."](https://arxiv.org/pdf/2407.01531) arXiv preprint arXiv:2407.01531 (2024).

- Wang, Dian, et al. ["Equivariant Diffusion Policy."](https://openreview.net/forum?id=wD2kUVLT1g) 8th Annual Conference on Robot Learning.

- Wang, Yixuan, et al. ["GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy."](https://openreview.net/forum?id=7wMlwhCvjS) 8th Annual Conference on Robot Learning.

- Yang, Jingyun, et al. ["EquiBot: SIM (3)-Equivariant Diffusion Policy for Generalizable and Data Efficient Learning."](https://openreview.net/forum?id=ueBmGhLOXP) 8th Annual Conference on Robot Learning.

- Huang, Xiaoyu, et al. ["DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets."](https://openreview.net/forum?id=nVJm2RdPDu) arXiv preprint arXiv:2404.19264 (2024).

- Djeumou, Franck, et al. ["One Model to Drift Them All: Physics-Informed Conditional Diffusion Model for Driving at the Limits."] 8th Annual Conference on Robot Learning.

- Jia, Xiaogang, et al. ["MaIL: Improving Imitation Learning with Mamba."](https://arxiv.org/pdf/2406.08234) 8th Annual Conference on Robot Learning.

- Shridhar, Mohit, Yat Long Lo, and Stephen James. ["Generative Image as Action Models."](https://arxiv.org/pdf/2407.07875) 8th Annual Conference on Robot Learning.

- Zhou, Hongyi, et al. ["Variational Distillation of Diffusion Policies into Mixture of Experts."](https://arxiv.org/pdf/2406.12538) arXiv preprint arXiv:2406.12538 (2024).

- Hao, Ce, et al. ["Language-Guided Manipulation with Diffusion Policies and Constrained Inpainting."](https://arxiv.org/pdf/2406.09767) arXiv preprint arXiv:2406.09767 (2024).

- Høeg, Sigmund H., and Lars Tingelstad. ["TEDi Policy: Temporally Entangled Diffusion for Robotic Control."](https://arxiv.org/pdf/2406.04806) arXiv preprint arXiv:2406.04806 (2024).

- Vosylius, Vitalis, et al. ["Render and Diffuse: Aligning Image and Action Spaces for Diffusion-based Behaviour Cloning."](https://arxiv.org/pdf/2405.18196) _Proceedings of Robotics: Science and Systems (RSS)_ 2024.

- Prasad, Aaditya, et al. ["Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation."](https://arxiv.org/pdf/2405.07503) _Proceedings of Robotics: Science and Systems (RSS)_ 2024.

- Bharadhwaj, Homanga, et al. ["Track2Act: Predicting Point Tracks from Internet Videos enables Diverse Zero-shot Robot Manipulation."](https://arxiv.org/pdf/2405.01527) ECCV  2024

- Reuss, Moritz, et al. ["Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals."]([https://openreview.net/pdf?id=Pt6fLfXMRW](https://arxiv.org/pdf/2407.05996)) _Proceedings of Robotics: Science and Systems (RSS)_ 2024.

- Gupta, Gunshi, et al. ["Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control."](https://openreview.net/pdf?id=9A8jloU3jP) First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024 (2024).

- Ze, Yanjie, et al. ["3D Diffusion Policy."](https://arxiv.org/html/2403.03954v1) _Proceedings of Robotics: Science and Systems (RSS)_ 2024.

- Ma, Xiao, et al. ["Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation."](https://arxiv.org/html/2403.03890v1) arXiv preprint arXiv:2403.03890 (2024).

- Yan, Ge, Yueh-Hua Wu, and Xiaolong Wang. ["DNAct: Diffusion Guided Multi-Task 3D Policy Learning."](https://arxiv.org/html/2403.04115v1) arXiv preprint arXiv:2403.04115 (2024).

- Zhang, Xiaoyu, et al. ["Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning."](https://arxiv.org/html/2402.17768v1) arXiv preprint arXiv:2402.17768 (2024).

- Chen, Kaiqi, et al. ["Behavioral Refinement via Interpolant-based Policy Diffusion."](https://arxiv.org/pdf/2402.16075) arXiv preprint arXiv:2402.16075 (2024).

- Wang, Bingzheng, et al. ["DiffAIL: Diffusion Adversarial Imitation Learning."](https://arxiv.org/pdf/2312.06348) arXiv preprint arXiv:2312.06348 (2023).

- Scheikl, Paul Maria, et al. ["Movement Primitive Diffusion: Learning Gentle Robotic Manipulation of Deformable Objects."](https://arxiv.org/pdf/2312.10008) arXiv preprint arXiv:2312.10008 (2023).

- Octo Model Team et al. [Octo: An Open-Source Generalist Robot Policy](https://octo-models.github.io/paper.pdf) 

- Black, Kevin, et al. ["ZERO-SHOT ROBOTIC MANIPULATION WITH PRETRAINED IMAGE-EDITING DIFFUSION MODELS."](https://arxiv.org/pdf/2310.10639) arXiv preprint arXiv:2310.10639 (2023).

- Reuss, Moritz, and Rudolf Lioutikov. ["Multimodal Diffusion Transformer for Learning from Play."](https://openreview.net/pdf?id=nvtxqMGpn1) 2nd Workshop on Language and Robot Learning: Language as Grounding. 2023.

- Sridhar, Ajay, et al. ["NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration."](https://arxiv.org/pdf/2310.07896) arXiv preprint arXiv:2310.07896 (2023).

- Zhou, Xian, et al. ["Unifying Diffusion Models with Action Detection Transformers for Multi-task Robotic Manipulation."](https://openreview.net/pdf?id=W0zgY2mBTA8) _Conference on Robot Learning._ PMLR, 2023.

- Ze, Yanjie, et al. ["Multi-task real robot learning with generalizable neural feature fields."](https://openreview.net/pdf?id=b1tl3aOt2R2) 7th Annual Conference on Robot Learning. 2023.

- Mishra, Utkarsh Aashu, et al. ["Generative Skill Chaining: Long-Horizon Skill Planning with Diffusion Models."](https://openreview.net/pdf?id=HtJE9ly5dT) _Conference on Robot Learning._ PMLR, 2023.

- Chen, Lili, et al. ["PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play."](https://openreview.net/pdf?id=afF8RGcBBP) _Conference on Robot Learning._ PMLR, 2023.

- Ha, Huy, Pete Florence, and Shuran Song. ["Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition."](https://arxiv.org/pdf/2307.14535) _Conference on Robot Learning._ PMLR, 2023.

- Xu, Mengda, et al. ["XSkill: Cross Embodiment Skill Discovery."](https://arxiv.org/pdf/2307.09955) _Conference on Robot Learning._ PMLR, 2023.

- Li, Xiang, et al. ["Crossway Diffusion: Improving Diffusion-based Visuomotor Policy via Self-supervised Learning."](https://arxiv.org/pdf/2307.01849) arXiv preprint arXiv:2307.01849 (2023).

- Ng, Eley, Ziang Liu, and Monroe Kennedy III. ["Diffusion Co-Policy for Synergistic Human-Robot Collaborative Tasks."](https://arxiv.org/pdf/2305.12171) arXiv preprint arXiv:2305.12171 (2023).

- Chi, Cheng, et al. ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."](https://arxiv.org/pdf/2303.04137) _Proceedings of Robotics: Science and Systems (RSS)_ 2023.

- Reuss, Moritz, et al. ["Goal-Conditioned Imitation Learning using Score-based Diffusion Policies."](https://arxiv.org/pdf/2304.02532) _Proceedings of Robotics: Science and Systems (RSS)_ 2023.

- Yoneda, Takuma, et al. ["To the Noise and Back: Diffusion for Shared Autonomy."](https://arxiv.org/pdf/2302.12244) _Proceedings of Robotics: Science and Systems (RSS)_ 2023.

- Jiang, Chiyu, et al. ["MotionDiffuser: Controllable Multi-Agent Motion Prediction Using Diffusion."](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_MotionDiffuser_Controllable_Multi-Agent_Motion_Prediction_Using_Diffusion_CVPR_2023_paper.pdf) Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

- Kapelyukh, Ivan, et al. ["DALL-E-Bot: Introducing Web-Scale Diffusion Models to Robotics."](https://openreview.net/forum?id=HzOy6lUzPj1) _IEEE Robotics and Automation Letters (RA-L)_ 2023.

- Pearce, Tim, et al. ["Imitating human behaviour with diffusion models."](https://openreview.net/pdf?id=Pv1GPQzRrC8) 
" _International Conference on Learning Representations_. 2023.

- Yu, Tianhe, et al. ["Scaling robot learning with semantically imagined experience."](https://arxiv.org/pdf/2302.11550.pdf) arXiv preprint arXiv:2302.11550 (2023).

---

### Video Diffusion in Robotics
<a name="Video-Diffusion"></a>

The ability of Diffusion models to generate realistic videos over a long horizon has enabled new applications in the context of robotics. 

- Bu, Qingwen, et al. ["Closed-Loop Visuomotor Control with Generative Expectation for Robotic Manipulation."](https://arxiv.org/pdf/2409.09016) arXiv preprint arXiv:2409.09016 (2024).

- Wang, Boyang, et al. ["This&That: Language-Gesture Controlled Video Generation for Robot Planning."](https://arxiv.org/pdf/2407.05530) arXiv:2407.05530 (2024).

- Huang, Shuaiyi, et al. ["ARDuP: Active Region Video Diffusion for Universal Policies."](https://arxiv.org/pdf/2406.13301) arXiv preprint arXiv:2406.13301 (2024).

- Chen, Boyuan, et al. ["Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion."](https://arxiv.org/pdf/2407.01392) arXiv preprint arXiv:2407.01392 (2024).

- Zhou, Siyuan, et al. ["RoboDreamer: Learning Compositional World Models for Robot Imagination."](https://arxiv.org/pdf/2404.12377) arXiv preprint arXiv:2404.12377 (2024).

- McCarthy, Robert, et al. ["Towards Generalist Robot Learning from Internet Video: A Survey."](https://arxiv.org/pdf/2404.19664) arXiv preprint arXiv:2404.19664 (2024).

- He, Haoran, et al. ["Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning."](https://arxiv.org/pdf/2402.14407v1.pdf) arXiv preprint arXiv:2402.14407 (2024).

- Liang, Zhixuan, et al. ["SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution."](https://arxiv.org/pdf/2312.11598) arXiv preprint arXiv:2312.11598 (2023).

- Huang, Tao, et al. ["Diffusion Reward: Learning Rewards via Conditional Video Diffusion."](https://arxiv.org/pdf/2312.14134) arXiv preprint arXiv:2312.14134 (2023).

- Du, Yilun, et al. ["Video Language Planning."](https://arxiv.org/abs/2310.10625) arXiv preprint arXiv:2310.10625 (2023).

- Yang, Mengjiao, et al. ["Learning Interactive Real-World Simulators."](https://arxiv.org/abs/2310.06114) arXiv preprint arXiv:2310.06114 (2023).

- Ko, Po-Chen, et al. ["Learning to Act from Actionless Videos through Dense Correspondences."](https://arxiv.org/pdf/2310.08576) arXiv preprint arXiv:2310.08576 (2023).

- Ajay, Anurag, et al. ["Compositional Foundation Models for Hierarchical Planning."](https://arxiv.org/pdf/2309.08587) _Advances in Neural Information Processing Systems_ 37 (2023)

- Dai, Yilun, et al. ["Learning Universal Policies via Text-Guided Video Generation."](https://arxiv.org/pdf/2302.00111) _Advances in Neural Information Processing Systems_ 37 (2023)


---

### Online RL 
<a name="Online-RL"></a>

The standard policy gradient objective requires the gradient of the log-likelihood, which is only implicitly defined by the underlying Ordinary Differential Equation (ODE) of the diffusion model. 

- Ren, Allen Z., et al. ["Diffusion Policy Policy Optimization."](https://arxiv.org/abs/2409.00588) arXiv preprint arXiv:2409.00588 (2024).

- Shribak, Dmitry, et al. ["Diffusion Spectral Representation for Reinforcement Learning."](https://arxiv.org/pdf/2406.16121) arXiv preprint arXiv:2406.16121 (2024).

- Yang, Long, et al. ["Policy Representation via Diffusion Probability Model for Reinforcement Learning."](https://arxiv.org/pdf/2305.13122) arXiv preprint arXiv:2305.13122 (2023).

- Mazoure, Bogdan, et al. ["Value function estimation using conditional diffusion models for control."](https://arxiv.org/pdf/2306.07290) arXiv preprint arXiv:2306.07290 (2023).
--- 


### Offline RL
<a name="Offline-RL"></a>

- Jackson, Matthew Thomas, et al. ["Policy-guided diffusion."](https://arxiv.org/pdf/2404.06356) arXiv preprint arXiv:2404.06356 (2024).

- Kim, Woo Kyung, Minjong Yoo, and Honguk Woo. ["Robust Policy Learning via Offline Skill Diffusion."](https://arxiv.org/pdf/2403.00225) arXiv preprint arXiv:2403.00225 (2024).

- Kim, Sungyoon, et al. ["Stitching Sub-Trajectories with Conditional Diffusion Model for Goal-Conditioned Offline RL."](https://arxiv.org/pdf/2402.07226.pdf) arXiv preprint arXiv:2402.07226 (2024).

- Psenka, Michael, et al. ["Learning a Diffusion Model Policy from Rewards via Q-Score Matching."](https://arxiv.org/pdf/2312.11752) arXiv preprint arXiv:2312.11752 (2023).

- Chen, Chang, et al. ["Simple Hierarchical Planning with Diffusion."](https://arxiv.org/pdf/2401.02644) arXiv preprint arXiv:2401.02644 (2024).

- Brehmer, Johann, et al. ["EDGI: Equivariant diffusion for planning with embodied agents."](https://proceedings.neurips.cc/paper_files/paper/2023/file/c95c049637c5c549c2a08e8d6dcbca4b-Paper-Conference.pdf) Advances in Neural Information Processing Systems 36 (2024).

- Venkatraman, Siddarth, et al. ["Reasoning with latent diffusion in offline reinforcement learning."](https://arxiv.org/pdf/2309.06599) arXiv preprint arXiv:2309.06599 (2023).

- Chen, Huayu, et al. ["Score Regularized Policy Optimization through Diffusion Behavior."](https://arxiv.org/pdf/2310.07297) arXiv preprint arXiv:2310.07297 (2023).

- Ding, Zihan, and Chi Jin. ["Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning."](https://arxiv.org/pdf/2309.16984) arXiv preprint arXiv:2309.16984 (2023).

- Wang, Zidan, et al. ["Cold Diffusion on the Replay Buffer: Learning to Plan from Known Good States."](https://arxiv.org/pdf/2310.13914) arXiv preprint arXiv:2310.13914 (2023).

- Lee, Kyowoon, Seongun Kim, and Jaesik Choi. ["Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans."](https://arxiv.org/pdf/2310.19427) _Advances in Neural Information Processing Systems_ 37 (2023)

- Liu, Jianwei, Maria Stamatopoulou, and Dimitrios Kanoulas. ["DiPPeR: Diffusion-based 2D Path Planner applied on Legged Robots."](https://arxiv.org/pdf/2310.07842) arXiv preprint arXiv:2310.07842 (2023).

- Zhou, Siyuan, et al. ["Adaptive Online Replanning with Diffusion Models."](https://arxiv.org/pdf/2310.09629) _Advances in Neural Information Processing Systems_ 37 (2023)

- Jain, Vineet, and Siamak Ravanbakhsh. ["Learning to Reach Goals via Diffusion."](https://arxiv.org/pdf/2310.02505) arXiv preprint arXiv:2310.02505 (2023).

- Geng, Jinkun, et al. ["Diffusion Policies as Multi-Agent Reinforcement Learning Strategies."](https://link.springer.com/chapter/10.1007/978-3-031-44213-1_30) International Conference on Artificial Neural Networks. Cham: Springer Nature Switzerland, 2023.

- Suh, H.J., et al. ["Fighting Uncertainty with Gradients: Offline Reinforcement Learning via Diffusion Score Matching."](https://openreview.net/pdf?id=IM8zOC94HF) _Conference on Robot Learning._ PMLR, 2023.

- Yuan, Hui, et al. ["Reward-Directed Conditional Diffusion: Provable Distribution Estimation and Reward Improvement."](https://arxiv.org/pdf/2307.07055) arXiv preprint arXiv:2307.07055 (2023).

- Hu, Jifeng, et al. ["Instructed Diffuser with Temporal Condition Guidance for Offline Reinforcement Learning."](https://arxiv.org/pdf/2306.04875) arXiv preprint arXiv:2306.04875 (2023).

- Hegde, Shashank, et al. ["Generating Behaviorally Diverse Policies with Latent Diffusion Models."](https://arxiv.org/pdf/2305.18738) arXiv preprint arXiv:2305.18738 (2023).

- Xiao, Wei, et al. ["SafeDiffuser: Safe Planning with Diffusion Probabilistic Models."](https://arxiv.org/pdf/2306.00148) arXiv preprint arXiv:2306.00148 (2023).

- Li, Wenhao, et al. ["Hierarchical Diffusion for Offline Decision Making."](https://openreview.net/pdf?id=55kLa7tH9o) _International Conference on Machine Learning_. PMLR, 2023.

- Liang, Zhixuan, et al. ["AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners."](https://arxiv.org/pdf/2302.01877) _International Conference on Machine Learning_. PMLR, 2023.

- Lu, Cheng, et al. ["Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning."](https://arxiv.org/pdf/2304.12824.pdf)  _International Conference on Machine Learning_. PMLR, 2023.

- Zhu, Zhengbang, et al. ["MADiff: Offline Multi-agent Learning with Diffusion Models."](https://arxiv.org/pdf/2305.17330) arXiv preprint arXiv:2305.17330 (2023).

- Kang, Bingyi, et al. ["Efficient Diffusion Policies for Offline Reinforcement Learning."](https://arxiv.org/pdf/2305.20081) arXiv preprint arXiv:2305.20081 (2023).

- Ni, Fei, et al. ["MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL."](https://arxiv.org/pdf/2305.19923) _International Conference on Machine Learning_. PMLR, 2023.

- He, Haoran, et al. ["Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning."](https://arxiv.org/pdf/2305.18459) arXiv preprint arXiv:2305.18459 (2023).

- Ajay, Anurag, et al. ["Is Conditional Generative Modeling all you need for Decision-Making?."](https://arxiv.org/pdf/2211.15657) _International Conference on Learning Representations_. 2023.

- Hansen-Estruch, Philippe, et al. ["IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies."](https://arxiv.org/pdf/2304.10573) arXiv preprint arXiv:2304.10573 (2023).

- Zhu, Zhengbang, et al. ["MADiff: Offline Multi-agent Learning with Diffusion Models."](https://arxiv.org/pdf/2305.17330) arXiv preprint arXiv:2305.17330 (2023).

- Zhang, Edwin, et al. ["LAD: Language Augmented Diffusion for Reinforcement Learning."](https://arxiv.org/pdf/2210.15629) arXiv preprint arXiv:2210.15629 (2022).

- Brehmer, Johann, et al. [EDGI: Equivariant Diffusion for Planning with Embodied Agents](https://openreview.net/forum?id=OrbWCpidbt) _Workshop on Reincarnating Reinforcement Learning at ICLR_ 2023. 

- Janner, Michael, et al. ["Planning with Diffusion for Flexible Behavior Synthesis."](https://arxiv.org/pdf/2205.09991.pdf) _International Conference on Learning Representations_. 2022.

- Wang, Zhendong, et al. ["Diffusion policies as an expressive policy class for offline reinforcement learning."](https://arxiv.org/pdf/2208.06193.pdf)  _International Conference on Learning Representations_. 2023.

- Brehmer, Johann, et al. ["EDGI: Equivariant Diffusion for Planning with Embodied Agents."](https://arxiv.org/pdf/2303.12410) arXiv preprint arXiv:2303.12410 (2023).

- Chen, Huayu, et al. ["Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling."](https://openreview.net/pdf?id=42zs3qa2kpy)" _International Conference on Learning Representations_. 2023.


--- 

### Inverse RL
<a name="Inverse-RL"></a>

- Nuti, Felipe, Tim Franzmeyer, and João F. Henriques. ["Extracting Reward Functions from Diffusion Models."](https://arxiv.org/pdf/2306.01804) _Advances in Neural Information Processing Systems_ 37 (2023)

---

### World Models
<a name="World-Models"></a>

- Valevski, Dani, et al. ["Diffusion Models Are Real-Time Game Engines."](https://arxiv.org/pdf/2408.14837) arXiv preprint arXiv:2408.14837 (2024).

- Yu, Youwei, Junhong Xu, and Lantao Liu. ["Adaptive Diffusion Terrain Generator for Autonomous Uneven Terrain Navigation."](https://openreview.net/forum?id=xYleTh2QhS) 8th Annual Conference on Robot Learning.

- Ding, Zihan, et al. ["Diffusion World Model."](https://arxiv.org/html/2402.03570v1) arXiv preprint arXiv:2402.03570 (2024).

- Rigter, Marc, Jun Yamada, and Ingmar Posner. ["World models via policy-guided trajectory diffusion."](https://arxiv.org/html/2312.08533v2) arXiv preprint arXiv:2312.08533 (2023).

- Zhang, Lunjun, et al. ["Learning unsupervised world models for autonomous driving via discrete diffusion."](https://arxiv.org/abs/2311.01017) arXiv preprint arXiv:2311.01017 (2023).

---

### Task and Motion Planning
<a name="tamp"></a>

- Mishra, Utkarsh Aashu, Yongxin Chen, and Danfei Xu. ["Generative Factor Chaining: Coordinated Manipulation with Diffusion-based Factor Graph."](https://openreview.net/forum?id=p6Wq6TjjHH) 8th Annual Conference on Robot Learning.

- Huang, Huang, et al. ["DiffusionSeeder: Seeding Motion Optimization with Diffusion for Rapid Motion Planning."](https://openreview.net/pdf?id=B7Lf6xEv7l) 8th Annual Conference on Robot Learning.

- Xu, Yiqing, et al. [""Set It Up!": Functional Object Arrangement with Compositional Generative Models."](https://arxiv.org/pdf/2405.11928) arXiv preprint arXiv:2405.11928 (2024).

- Luo, Yunhao, et al. ["Potential Based Diffusion Motion Planning."](https://arxiv.org/pdf/2407.06169) arXiv preprint arXiv:2407.06169 (2024).

- Yang, Cheng-Fu, et al. ["Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty."](https://arxiv.org/pdf/2312.01097) arXiv preprint arXiv:2312.01097 (2023).

- Liu, Jiaqi, et al. ["DDM-Lag: A Diffusion-based Decision-making Model for Autonomous Vehicles with Lagrangian Safety Enhancement."](https://arxiv.org/pdf/2401.03629) arXiv preprint arXiv:2401.03629 (2024).

- Chang, Junwoo, et al. ["Denoising Heat-inspired Diffusion with Insulators for Collision Free Motion Planning."](https://arxiv.org/abs/2310.12609) NeurIPS 2023 Workshop on Diffusion Models

- Ryu, Hyunwoo, et al. ["Diffusion-edfs: Bi-equivariant denoising generative modeling on se (3) for visual robotic manipulation."](https://arxiv.org/pdf/2309.02685.pdf) arXiv preprint arXiv:2309.02685 (2023).

- Yang, Zhutian, et al. ["Compositional Diffusion-Based Continuous Constraint Solvers."](https://openreview.net/forum?id=BimpCf1rT7) 7th Annual Conference on Robot Learning. 2023.

- Carvalho, Joao, et al. ["Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models."](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/2023-iros-carvalho-mpd.pdf), IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). (2023)

- Saha, Kallol, et al. ["EDMP: Ensemble-of-costs-guided Diffusion for Motion Planning."](https://arxiv.org/pdf/2309.11414) arXiv preprint arXiv:2309.11414 (2023).

- Power, Thomas, et al. ["Sampling Constrained Trajectories Using Composable Diffusion Models."](https://openreview.net/pdf?id=UAylEpIMNE) IROS 2023 Workshop on Differentiable Probabilistic Robotics: Emerging Perspectives on Robot Learning. 2023.

- Zhong, Ziyuan, et al. ["Language-Guided Traffic Simulation via Scene-Level Diffusion."](https://openreview.net/pdf?id=nKWQnYkkwX) _Conference on Robot Learning._ PMLR, 2023.

- Fang, Xiaolin, et al. ["DiMSam: Diffusion Models as Samplers for Task and Motion Planning under Partial Observability."](https://arxiv.org/pdf/2306.13196) arXiv preprint arXiv:2306.13196 (2023).

- Liu, Weiyu, et al. ["StructDiffusion: Object-centric diffusion for semantic rearrangement of novel objects."](https://arxiv.org/pdf/2211.04604) _Proceedings of Robotics: Science and Systems (RSS)_ 2023.

- Mishra, Utkarsh A., and Yongxin Chen. ["ReorientDiff: Diffusion Model based Reorientation for Object Manipulation."](https://arxiv.org/pdf/2303.12700) _RSS 2023 Workshop on Learning for Task and Motion Planning_

- Urain, Julen, et al. ["SE (3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion."](https://arxiv.org/pdf/2209.03855) _IEEE International Conference on Robotics and Automation (ICRA)_ 2023

- Carvalho, J. et al. [Conditioned Score-Based Models for Learning Collision-Free Trajectory Generation](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/Conditioned_Score_Based_Models_for_Learning_Collision_Free_Trajectory_Generation.pdf), _NeurIPS 2022 Workshop on Score-Based Methods_ 

---

### Tactile Sensing & Pose Estimation & Grasping & Depth

<a name="Grasping-&-Tactile-Sensing-&-Pose-Estimation"></a>

- Cai, Eric, et al. ["Non-rigid Relative Placement through 3D Dense Diffusion."](https://openreview.net/pdf?id=rvKWXxIvj0) 8th Annual Conference on Robot Learning.

- Song, Pinhao, Pengteng Li, and Renaud Detry. ["Implicit Grasp Diffusion: Bridging the Gap between Dense Prediction and Sampling-based Grasping."](https://openreview.net/pdf?id=VUhlMfEekm) 8th Annual Conference on Robot Learning.

- Wei, Songlin, et al. ["D3RoMa: Disparity Diffusion-based Depth Sensing for Material-Agnostic Robotic Manipulation."](https://openreview.net/forum?id=7E3JAys1xO) 8th Annual Conference on Robot Learning.

- Liao, Ziwei, Binbin Xu, and Steven L. Waslander. ["Toward General Object-level Mapping from Sparse Views with 3D Diffusion Priors."](https://openreview.net/pdf?id=rEteJcq61j) 8th Annual Conference on Robot Learning.

- Yoneda, Takuma, et al. ["6-DoF Stability Field via Diffusion Models."](https://arxiv.org/pdf/2310.17649) arXiv preprint arXiv:2310.17649 (2023).

- Simeonov, Anthony, et al. ["Shelving, Stacking, Hanging: Relational Pose Diffusion for Multi-modal Rearrangement."](https://arxiv.org/pdf/2307.04751) arXiv preprint arXiv:2307.04751 (2023).

- Higuera, Carolina, Byron Boots, and Mustafa Mukadam. ["Learning to Read Braille: Bridging the Tactile Reality Gap with Diffusion Models."](https://arxiv.org/pdf/2304.01182) arXiv preprint arXiv:2304.01182 (2023).


---

### Robot Development and Construction
<a name="Robot-Design-and-Construction"></a>

Excited to see more diffusion papers in this area in the future!
Using generative models to design robots is a very interesting idea, since it allows to generate new robot designs and test them in simulation before building them in the real world. 

- Xu, Xiaomeng, Huy Ha, and Shuran Song. ["Dynamics-Guided Diffusion Model for Sensor-less Robot Manipulator Design."](https://openreview.net/pdf?id=AzP6kSEffm) 8th Annual Conference on Robot Learning.

- Wang, Tsun-Hsuan, et al. ["DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models."](https://openreview.net/forum?id=1zo4iioUEs) Thirty-seventh Conference on Neural Information Processing Systems. 2023.

---

## Code Implementations
<a name="Code-Bases"></a>

There exist numerous implementations of all diffusion models on github. Below you can find a curated list of some clean code variants of the most important diffusion models in general and for robotics:

- [Diffusers](https://github.com/huggingface/diffusers): the main diffusion project from HuggingFaces with numerous pre-trained diffusion models ready to use 

- [k-diffusion](https://github.com/crowsonkb/k-diffusion): while its not the official code-base of the EDM diffusion models from [Karras et al., 2022](https://arxiv.org/pdf/2206.00364), it has very clean code and numerous samplers. Parts of the code have been used in various other projects such as [Consistency Models](https://github.com/openai/consistency_models) from OpenAI and diffusers from HuggingFaces.

- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch): a clean DDPM diffusion model implementation in Pytorch to get a good understanding of all the components 

- [Diffuser](https://github.com/jannerm/diffuser): Variants of this code are used in numerous trajectory diffusion OfflineRL papers listed above

- [diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy): Beautiful Code implementation of Diffusion policies from [Chi et al., 2023](https://diffusion-policy.cs.columbia.edu/#paper) for Imitation Learning with 9 different simulations to test the models on

- [octo-models](https://github.com/octo-models/octo): The first open source foundation behavior diffusion agent, pretrained on 800k trajectories of different embodiements. The JAX code allows you to download their weights and finetune your own Octo-model on your local dataset.

- [3d_diffuser_actor](https://github.com/nickgkan/3d_diffuser_actor): Clean code to get started with 3D-based diffusion policies on the popular RL-bench and CALVIN benchmarks. 

- [flow-diffusion](https://github.com/flow-diffusion/AVDC): If you want to start training your own video-diffusion model, this is the right repository to start! Clean code implementations and available pre-training weights for real world dataset and two simulations.

- [dpm-solver](https://github.com/LuChengTHU/dpm-solver): One of the most widely used ODE samplers for Diffusion models from [Lu et al. 2022](https://arxiv.org/abs/2206.00927) with implementations for all different diffusion models including wrappers for discrete DDPM variants 


--- 

## Diffusion History
<a name="Diffusion-History"></a>
Diffusion models are a type of generative model inspired by non-equilibrium thermodynamics, introduced by [Sohl-Dickstein et al., (2015)](https://arxiv.org/abs/1503.03585). The model learns to invert a diffusion process, that gradually adds noise to a data sample. This process is a Markov chain consisting of diffusion steps, which add random Gaussian noise to a data sample. The diffusion model is used to learn to invert this process. While the paper was presented  in 2015, it took several years for the diffusion models to get widespread attention in the research community. Diffusion models are a type of generative model and in this field, the main focus are vision based applications, thus all theory papers mentioned in the text below are mostly focused on image synthesis or similar tasks related to it. 

There are two perspectives to view diffusion models. The first one is based on the initial idea of  [Sohl-Dickstein et al., (2015)](https://arxiv.org/abs/1503.03585), while the other is based on a different direction of research known as score-based generative models. In 2019 [Song & Ermon, (2019)](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) proposed the _noise-conditioned score network (NCSN)_, which is a predecessor to the score-based diffusion model. The main idea was to learn the score function of the unknown data distribution using a neural network. This approach had been around before, however their paper and the subsequent work [Song & Ermon (2020)](https://arxiv.org/abs/2006.09011) enabled scaling score-based models to high-dimension data distributions and made them competitive on image-generation tasks. The key idea in their work was to perturb the data distribution with various levels of Gaussian noise and learn a noise-conditional score model to predict the score of the perturbed data distributions. 


In 2020, [Ho et al., (2020)](https://arxiv.org/abs/2006.11239) introduced  _denoising diffusion probabilistic models (DDPM)_, which served as the foundation for the success of Diffusion models. At that time, Diffusion models still were not  competitive with state-of-the-art generate models such as GANs. However, this changed rapidly the following year when [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2105.05233) improved upon the previous paper and demonstrated, that Diffusion models are competitive with GANs on image synthesis tasks. Nevertheless, it is important to note, that Diffusion models are not the jack of all trades. Diffusion models still struggle with certain image traits such as generating realistic faces or generating the right amount of fingers. 

Another important idea for diffusion models in the context of image generation has been the introduction of _latent diffusion models_ by [Rombach & Blattman et al., (2022)](https://arxiv.org/abs/2112.1075). By training the diffusion model in the latent space rather than the image space directly, they were able to improve the sampling and training speed and made it possible for everyone to run their own diffusion model on local PCs with a single GPU. Recent AI generated art is mostly based on the stable AI implementation of latent diffusion models and is open source: [Github repo](https://github.com/CompVis/stable-diffusion). Check out some cool Diffusion art on the [stable-diffusion-reddit](https://www.reddit.com/r/StableDiffusion/).


**Conditional Diffusion models**
The initial diffusion models are usually trained on marginal distributions $p(x)$, but conditional image generation is also an research area of great interest. Therefore, we need conditional diffusion models to _guide_ the generation process. Currently, there are three common methods to enable conditional generation with diffusion models:

- Classifier Guided Diffusion by [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233)
- Classifier-Free Guidance (CFG) by [Ho & Salimans, (2021)](https://openreview.net/forum?id=qw8AKxfYbI)
- directly training a conditional diffusion model $p(x|z)$ 

CFG is used in many applications, since it allows to train a conditional diffusion model  and unconditional diffusion model at the same time. During inference, we can combine both models and control the generation process using a guidance weight. 

**Diffusion models perspectives**

As previously mentioned, diffusion models can be viewed from two different perspectives:
- the denoising diffusion probabilistic perspective based on [Ho et al., (2020)](https://arxiv.org/abs/2006.11239) 
- the score-based model perspective based on  [Song & Ermon, (2019)](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf)

There has been a lot of effort to combine these two views into one general framework. The best generalization has been the idea of stochastic differential equations (SDEs) first presented in [Song et al. (2021)](https://arxiv.org/pdf/2011.13456) and further developed to unified framework in [Karras et al. (2022)](https://arxiv.org/pdf/2206.00364).

While diffusion models have mainly been applied in the area of generative modeling, recent work has shown promising applications of diffusion models in robotics. For instance, diffusion models have been used for behavior cloning and offline reinforcement learning, and have also been used to generate more diverse training data for robotics tasks.

Diffusion models offer several useful properties in the context of robotics, including:

- *Expressiveness*: can learn arbitrarily complicated data-distributions 
- *Training stability*: they are easy to train especially in contrast GANs or EBMs
- *Multimodality*: they are able to learn complicated multimodal distributions
- *Compositionality*: Diffusion models can combined in a flexible way to jointly generate new samples

Overall, diffusion models have the potential to be a valuable tool for robotics.

---
