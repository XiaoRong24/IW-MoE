# IW-MoE

**IW-MoE: A Master-Slave Mixture-of-Experts Framework for Competent and Deployable Multi-Task Image Warping**

<div align="center">
  <img src="https://github.com/XiaoRong24/IW-MoE/blob/main/IW-MOE.jpg" alt="IW-MoE Overview" width="85%">
</div>

---

## 🔥 Overview

We propose **IW-MoE**, a unified framework that jointly rectifies **six heterogeneous image warping tasks** — including distortion correction, perspective adjustment, and edge straightening — within a **single inference pass**, with **no need for explicit task-type specification**.

<div align="center">
  <img src="https://github.com/XiaoRong24/IW-MoE/blob/main/ShowImage.jpg" alt="IW-MoE Framework Details" width="85%">
  <br>
  <em>Figure 1: Overall architecture of IW-MoE. A master router dynamically activates task-specific slave experts, enabling synergistic multi-task rectification.</em>
</div>

---
