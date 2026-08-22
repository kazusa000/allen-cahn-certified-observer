# R5：参考 T--K 的联合训练

## 目的

R5-E 原先把 correction 和 certificate 放进同一个 Adam，但两个损失互不相干；这不等价于
参考 ODE T--K 实验中的联合辨识。本方案改为直接让 \(T_{\phi,h}\) 和在线
\(\Gamma_{\theta,h}\) 共同最小化稳定目标残差。

参考实验在双积分器上直接优化

\[
\|TA+BK-A_\lambda T\|_F^2+\|TB-B\|_F^2.
\]

Allen--Cahn 的非线性误差依赖 \(u\)，因此不强行要求一个与 \(u\) 无关的精确 Sylvester
方程，而使用研究冻结中的 state-conditioned fiber certificate 和离散时间稳定目标。

## 数学对象

在线观测器仍写作

\[
\partial_t\hat u_h=F_h(\hat u_h)+
\Gamma_{\theta,h}(\hat u_h,y_{[0,t]},\mu)(y_h-C_h\hat u_h).
\]

离线训练使用 \(e_h=\hat u_h-u_h\)，并令

\[
A_{s,h}=A_h-\lambda I,
\qquad
S_{\Delta t}=\exp(\Delta t A_{s,h}).
\]

每个连续采样对先由当前 \(\Gamma_{\theta,h}\) 做一步离散 observer 更新，得到
\(e_{j+1}\)，再最小化

\[
\mathcal L_{\mathrm{stable}}
=
\left\|
T_{\phi,h}(u_{j+1},e_{j+1})
-S_{\Delta t}T_{\phi,h}(u_j,e_j)
\right\|_{M_h}^{2}.
\]

这使 \(T_{\phi,h}\) 的参数和 \(\Gamma_{\theta,h}\) 的参数出现在同一个损失的计算图中。
certificate 仍不部署；真状态和真误差只用于离线训练和审计。

在此基础上，当前正式版本同时加入连续动力学缺陷。无噪声训练样本上令

\[
r^0_{\theta,\phi,h}(u,e)
=D_uT_{\phi,h}(u,e)F_h(u)
+D_eT_{\phi,h}(u,e)\bigl(F_h(u+e)-F_h(u)+g_{\theta,h}(u,e)\bigr)
-A_{s,h}T_{\phi,h}(u,e),
\]

其中 \(g_{\theta,h}\) 是当前增益作用在创新上的校正项。缺陷项按误差质量归一化：

\[
\mathcal L_{\mathrm{defect}}
=\frac1N\sum_{j=1}^N
\frac{\|r^0_{\theta,\phi,h}(u_j,e_j)\|_{M_h}^2}
{\|e_j\|_{M_h}^2+10^{-8}}.
\]

总训练目标为

\[
\mathcal L_{\mathrm{joint}}
=\mathcal L_{\mathrm{stable}}
+\omega_{\mathrm{defect}}\mathcal L_{\mathrm{defect}}
+\omega_{\mathrm{bi}}\mathcal L_{\mathrm{bi}},
\qquad
\omega_{\mathrm{defect}}=\omega_{\mathrm{bi}}=1.
\]

## 参数化与约束

在线 \(\Gamma_{\theta,h}\) 只接收估计场、当前观测、创新、已知 \(\nu\) 和估计场质量范数，
输出观测残差到状态导数的增益矩阵。默认从固定物理增益

\[
0.02\,C_h^{\mathsf T}/h
\]

附近开始学习 residual gain，但稳定损失而不是 teacher correction target 决定更新。

certificate 使用观测矩阵零空间基的结构化门控，使

\[
T_{\phi,h}(u,0)=0,
\qquad
C_hT_{\phi,h}(u,e)=C_he
\]

在浮点误差内成立。该结构只负责 fiber/direction 约束；动力学是否接近
\(A_{s,h}\) 由 \(\mathcal L_{\mathrm{stable}}+\mathcal L_{\mathrm{defect}}\) 检验。

当前版本的门控只依赖 \(u\)，并把每个零空间方向的缩放因子硬限制在

\[
m_h=0.5\le s_{\phi,h}(u)\le M_h^\sharp=2.0.
\]

因此在观测方向上保持恒等、在零空间方向上保持正的缩放，\(D_eT_{\phi,h}\) 的局部
奇异值下界由结构保证；同时保留

\[
\mathcal L_{\mathrm{bi}}
=\frac1N\sum_{j=1}^N\left[
\operatorname{ReLU}\!\left(m_h\|e_j\|_{M_h}-\|T_{\phi,h}(u_j,e_j)\|_{M_h}\right)^2
+\operatorname{ReLU}\!\left(\|T_{\phi,h}(u_j,e_j)\|_{M_h}-M_h^\sharp\|e_j\|_{M_h}\right)^2
\right]
\]

作为数值安全项。由于硬限制已经满足双 Lipschitz 范数带，正式运行中该项为零是预期的，
不把它误写成连续域证明。

## 预注册结果

每个网格、随机种子和 \(\lambda/\rho_1\) 记录：

- \(\mathcal L_{\mathrm{stable}}\) 的训练/验证值；
- 一步稳定残差的 \(M_h\)-范数；
- \(T_{\phi,h}(u,0)\) 和 \(C_hT_{\phi,h}(u,e)-C_he\)；
- \(T_{\phi,h}\) 对误差方向的 Jacobian 奇异值；
- 在线 test/噪声 rollout 的终点误差；
- 与 R5-D/R5-E 固定 gain 基线的差异。

不把有限样本的一步残差直接写成

\[
(A-\lambda I)T=T(A+B C)
\]

的连续或全局证明；若要进入条件稳定界，还必须进一步估计冻结文件中的
\(\varepsilon_{T,h}\)、\(m_h\)、\(M_h^\sharp\)、\(K_h\)、\(\alpha_h\) 和
\(\beta_h^{\mathrm{stab}}>0\)。

## 与旧 R5-E 的区别

旧实现优化的是 correction target 与 hand-designed nullspace certificate target 的和，
两者参数梯度分离。新实现不使用 hand-designed certificate target，改为直接优化
\(\mathcal L_{\mathrm{stable}}\)，因此它才是 T--K 风格的联合训练。
