### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ ba49a550-fe54-11ed-1d8b-9b5192874bc6
using PlutoUI

# ╔═╡ b76f4412-ef87-4185-912f-060d2891df54
html"""
<style>
body:not(.disable_ui) main {
	max-width: 95%;
	margin-right: 10px;
	align-self: center;
}
</style>
"""

# ╔═╡ 41db52a5-812e-49bb-9d1f-fb7851740a88
md"""
# Computing Eigenvalue Decomposition of Arrowhead and Diagonal-Plus-Rank-k Matrices of Quaternions 

#### by Thaniporn Chaysri, Nevena Jakovčević Stor and Ivan Slapničar

##### from the University of Split, FESB

##### at the 8th Croatian Mathematical Congress, Osijek, July 2-5, 2024.

This work has been fully supported by the Croatian Science Foundation under the project IP-2020-02-2240 - [http://manaa.fesb.unist.hr](http://manaa.fesb.unist.hr).

$(PlutoUI.LocalResource("./HRZZ-eng-170x80-1.jpg"))

"""

# ╔═╡ 2552eeb1-c631-4e58-94f1-3894b9545ab9
md"
# Quaternions

Quaternions are a non-commutative associative number system that extends complex numbers, introduced by Hamilton ( [1853](https://openlibrary.org/books/OL23416635M/Lectures_on_quaternions), [1866](https://openlibrary.org/books/OL7211578M/Elements_of_quaternions.)). For basic quaternions $\mathbf {i}$, 
$\mathbf {j}$, and $\mathbf {k}$, the quaternions have the form

$$q=a+b\ \mathbf {i} +c\ \mathbf {j} +d\ \mathbf {k},\quad a,b,c,d, \in \mathbb{R}.$$

The multiplication table of basic quaternions is the following:

$$\begin{array}{c|ccc}
\times & \mathbf {i} & \mathbf {j} & \mathbf {k} \\ \hline
\mathbf {i} & -1 & \mathbf {k} & -\mathbf {j} \\
\mathbf {j} & -\mathbf {k} & -1 & \mathbf {i} \\
\mathbf {k} & \mathbf {j} & -\mathbf {i} & -1
\end{array}$$

Conjugation is given by 

$$\bar q=a-b\ \mathbf {i} -c\ \mathbf {j} -d\ \mathbf {k}.$$

Then, 

$$\bar q q=q\bar q=|q|^2=a^2+b^2+c^2+d^2.$$

Let $f(x)$ be a complex analytic function. The value $f(q)$, where $q\in\mathbb{Q}$, is computed by evaluating the extension of $f$ to the quaternions at $q$, see ([Sudbery,1979](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/quaternionic-analysis/308CF454034EC347D4D17D1F829F8471)), for example,

$$\sqrt{q}=\pm \left(\sqrt {\frac {\|q\|+a_1}{2}} + \frac {\operatorname{imag} (q)}{\|\operatorname{imag}(q)\|} \sqrt {\frac {\|q\|-a_1}{2}}\right).$$

Basic operations with quaternions and computation of the functions of quaternions are implemented in the package [Quaternions.jl](https://github.com/JuliaGeometry/Quaternions.jl).
"

# ╔═╡ 58a3123e-2359-4af3-bb9d-4f0fdba3c8aa
md"
# Standard form

Quaternions $p$ and $q$ are __similar__ if 

$$\exists x \quad \textrm{s. t.}\quad p=x^{-1} q x.$$

This is *iff* 

$$\operatorname{real}(p)=\operatorname{real}(q)\quad \textrm{and}\quad  
\|p\|=\|q\|.$$

The __standard form__ of the quaternion $q$ is the (unique) similar quaternion $q_s$:

$$
q_s=x^{-1} qx =a + \hat b\,  \mathbf{i},\quad \|x\|=1,\quad \hat b \geq 0,$$

where $x$ is computed as follows:

> if $c=d=0$, then $x=1$,

> if $b<0$, then $x=-\mathbf{j}$, ortherwise,

> if $c^2+d^2>0$, then $x=\hat x/\|\hat x\|$, where $\hat x=\| \operatorname{imag}(q)\|+b-d\,\mathbf{j}+c\,\mathbf{k}$.
"

# ╔═╡ fa9e44d2-5729-45c1-9aa3-a2d35372aebd
md"
# Homomorphism 

Quaternions are homomorphic to $\mathbb{C}^{2\times 2}$:

$$
q\to \begin{bmatrix}a+b\,\mathbf{i} & c+d\, \mathbf{i}\\-c+d\, \mathbf{i} & a-b\, \mathbf{i}\end{bmatrix}\equiv C(q),$$

with eigenvalues $q_s$ and $\bar q_s$. It holds

$$
C(p+q)=C(p)+C(q),\quad C(pq)=C(p)C(q).$$
"

# ╔═╡ 68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
md"
# Matrices
  
__Arrowhead matrix__ (__Arrow__) is a matrix of the form

$$
A=\begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix},$$

where 

$$\mathop{\mathrm{diag}}(D),u, v \in\mathbb{Q}^{n-1},\quad \alpha\in\mathbb{Q},$$

or any symmetric permutation of such a matrix.

__Diagonal-plus-rank-one matrix__ (__DPR1__) is a matrix of the form 

$$
A=\Delta+x ρy^*$$

where 

$$\mathop{\mathrm{diag}}(\Delta),x, y \in\mathbb{Q}^{n},\quad  \rho \in \mathbb{Q}.$$
"

# ╔═╡ 8503c651-2bd3-43f2-9d4c-959962606ff5
md"
# Matrix × vector

Products 

$$
w=Az$$

are computed in $O(n)$ operations. 

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$. Then

$$
\begin{aligned}
w_j&=d_jz_j+u_jz_i, \quad j=1,2,\cdots,i-1\\
% y_i&=\sum_{j=1}^{i-1}\bar v_j x_j+\alpha x_i +\sum_{j=i}^{n-1}\bar v_j x_{j+1}\\
w_i&=v_{1:i-1}^* z_{1:i-1} +\alpha z_i + v_{i:n-1}^* z_{i+1:n} \\
w_j&=u_{j-1}z_i+d_{j-1}z_j,\quad j=i+1,i+2,\cdots,n.
\end{aligned}$$

Let $A=\operatorname{DPRk}(\Delta,x,y,\rho)$ and let $\beta=\rho(y^* z)$. Then

$$
w_i=\delta_i z_i+x_i\beta,\quad i=1,2,\cdots,n.$$
"

# ╔═╡ 47f842fa-063d-4b30-a734-3f7d825b1314
md"""
# Inverses (Arrowhead)

Inverses are computed in $O(n)$ operations.

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$ be nonsingular.

Let $P$ be the permutation matrix of the permutation $p=(1,2,\cdots,i-1,n,i,i+1,\cdots,n-1)$. 

If all $d_j\neq 0$, the inverse of $A$ is a DPRk (DPR1) matrix

$$
A^{-1} =\Delta+x \rho y^*,$$

where 

$$
\Delta=P\begin{bmatrix}D^{-1} & 0\\ 0 & 0\end{bmatrix}P^T,
\quad x=P\begin{bmatrix}D^{-1}u \\-1\end{bmatrix}\rho,\quad
y=P\begin{bmatrix}D^{-*}v \\-1\end{bmatrix},\quad
\rho=(\alpha-v^* D^{-1} u)^{-1}.$$
"""

# ╔═╡ 4d12d9db-66fd-43fb-aa1c-6dafc3c83b75
md"""
# Inverses (Arrowhead, cont.)

If $d_j=0$, the inverse of $A$ is an Arrow with the tip of the arrow at position $(j,j)$ and zero at position $A_{ii}$ (the tip and the zero on the shaft change places). Let $\hat P$ be the permutation matrix of the permutation $\hat p=(1,2,\cdots,j-1,n,j,j+1,\cdots,n-1)$. Partition $D$, $u$ and $v$ as

$$
D=\begin{bmatrix}D_1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & D_2\end{bmatrix},\quad
u=\begin{bmatrix} u_1 \\ u_j \\u_2\end{bmatrix},\quad 
v=\begin{bmatrix} v_1 \\ v_j \\v_2\end{bmatrix}.$$

Then 

$$
A^{-1}=P\begin{bmatrix} \hat D & \hat u\\ \hat v^* & \hat \alpha \end{bmatrix}P^T,
$$

where

$$
\begin{align*}
\hat D&=\begin{bmatrix}D_1^{-1} & 0 & 0 \\ 0 & D_2^{-1} & 0 \\ 0 & 0 & 0\end{bmatrix},\quad
\hat u= \begin{bmatrix}-D_1^{-1}u_1 \\ -D_2^{-1}u_2\\ 1 \end{bmatrix} u_j^{-1},\quad
\hat v= \begin{bmatrix}-D_1^{-*}v_1 \\ -D_2^{-*}v_2\\ 1\end{bmatrix}v_j^{-1},\\
\hat \alpha&=v_j^{-*}\left(-\alpha +v_1^* D_1^{-1} u_1+v_2^* D_2^{-1}u_2\right) u_j^{-1}.
\end{align*}$$
"""

# ╔═╡ 90d86d6c-7896-4a56-b827-1ba2f42d54f9
md"""
# Inverses (DPRk)


Let $A=\operatorname{DPRk}(\Delta,x,y,\rho)$ be nonsingular. 

If all $\delta_j\neq 0$, the inverse of $A$ is a DPRk matrix

$$
A^{-1} =\hat\Delta+\hat x\hat \rho \hat y^*,$$

where 

$$
\hat \Delta=\Delta^{-1},\quad 
\quad \hat x=\Delta^{-1}x,\quad
\hat y=\Delta^{-*}y,\quad
\hat \rho=-\rho(I+y^* \Delta^{-1} x\rho)^{-1}.$$
"""

# ╔═╡ 3122ea8b-77d4-4255-9e82-ac8bd8b0bade
md"""
# Inverses (DPRk, cont.)

If $k=1$ and $\delta_j=0$, the inverse of $A$ is an arrowhead matrix with the tip of the arrow at position $(j,j)$. In particular, let $P$ be the permutation matrix of the permutation  $p=(1,2,\cdots,j-1,n,j,j+1,\cdots,n-1)$. Partition $\Delta$, $x$ and $y$ as

$$
\Delta=\begin{bmatrix}\Delta_1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & \Delta_2\end{bmatrix},\quad
x=\begin{bmatrix} x_1 \\ x_j \\x_2\end{bmatrix},\quad 
y=\begin{bmatrix} y_1 \\ y_j \\y_2\end{bmatrix}.$$

Then, 

$$
A^{-1}=P\begin{bmatrix} D & u \\ v^* & \alpha \end{bmatrix}P^T,$$

where

$$
\begin{align*}
D&=\begin{bmatrix} \Delta_1^{-1} & 0\\ 0 &\Delta_2^{-1}\end{bmatrix},\quad
u= \begin{bmatrix}-\Delta_1^{-1}x_1 \\ -\Delta_2^{-1}x_2\end{bmatrix} x_j^{-1},\quad
v= \begin{bmatrix}-\Delta_1^{-*}y_1 \\ -\Delta_2^{-*}y_2\end{bmatrix}y_j^{-1},\\
\alpha&=(y_j^{-1})^*\left(\rho^{-1} +y_1^* \Delta_1^{-1} x_1+y_2^* \Delta_2^{-1}x_2\right) x_j^{-1}.
\end{align*}$$
"""

# ╔═╡ b299fcf7-7ced-45d1-a55c-74482ecb0c60
md"
# Eigenvalue decomposition

Right eigenpairs $(λ,x)$ satisfy

$$
Ax=xλ, \quad x\neq 0.$$

Usually, $x$ is chosen such that $\lambda$ is the standard form.

Eigenvalues are invariant under similarity.

> Eigenvalues are __NOT__ shift invariant, that is, eigenvalues of the shifted matrix are __NOT__ the shifted eigenvalues. (In general, $X^{-1}qX\neq qX^{-1}X=qI$)

If $\lambda$ is in the standard form, it is invariant under similarity with complex numbers.  
"

# ╔═╡ 82c8a221-3125-42a1-a033-02408688b6ae
md"
# A Quaternion QR algorithm

_by Angelika Bunse-Gerstner, Ralph Byers, and Volker Mehrmann, Numer. Math 55, 83-95 (1989)_

Given $A\in\mathbb{Q}^{n\times n}$, the algorithm has four steps, as usual:

1. Reduce $A$ to Hessenberg form by Householder reflectors:

$$
X^*AX=H,$$

where $X$ is unitary and $H$ is an upper Hessenberg matrix.

2. Compute the Schur decomposition of $H$,

$$
Q^*HQ=T,$$

where $Q$ is unitary and $T$ is upper triangular with eigenvalues of $A$ on the diagonal.

3. Compute the eigenvectors $V$ of $T$ by solving the Sylvester equation:

$$
TV-V\Lambda=0.$$

Then $V^{-1}TV=\Lambda$.

4. Multiply 

$$U=X*Q*V.$$

Then $U^{-1}AU=\Lambda$ is the eigenvalue decomposition of $A$.

The algorithm is derived for general matrices and requires $O(n^3)$ operations. The algorithm is stable and we use it for comparison.
"

# ╔═╡ 02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
md"

#  Computing the Schur decomposition

Given the upper Hessenberg matrix $A \in\mathbb{Q}^{n\times n}$, the method 
applies complex shift $\mu$ to $A$ by using Francis standard double shift on the matrix 

$$
M=A^2-(\mu+\bar\mu)A+\mu\bar\mu I$$

and applying it implicitly on $A$.

If $Ax=x\lambda$, then 

$$
\begin{aligned}
Mx&=(A^2-(\mu+\bar\mu)A+\mu\bar\mu I)x=x\lambda^2-
x(\mu+\bar\mu)\lambda+x\mu\bar\mu\\
&=x(\lambda^2-(\mu+\bar\mu)\lambda+\mu\bar\mu)
\end{aligned}$$

For the perfect shift, $\mu=\lambda$, it holds

$$
\lambda^2-(\mu+\bar\mu)\lambda+\mu\bar\mu=\lambda^2-(\lambda+\bar\lambda)\lambda+\lambda\bar\lambda=0.$$

Details are given in Algorithm 4 in the Appendix of [BGBM89].
"


# ╔═╡ 24724b41-f108-4dc7-b6dd-193a3e39bc37
md"
# RQI with double shifts

We can apply the double shift $\mu$ and $\bar \mu$ similarly as in the [BGGM89
] method. 

The __Rayleigh Quotient Iteration with Double Shifts__ (RQIds) produces sequences of shifts and vectors

$$
\mu_k=\frac{1}{x_k^*x_k} x_k^*Ax_k,\quad 
y_k=(A^2-(\mu_k+\bar\mu_k)A+\mu_k\bar\mu_k I)^{-1} x_k, \quad  
x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

> Due to the arrowhead or DPRk structure of $A$, one step of the method requires $O(n)$ operations:

Here $y_k$ is the solution of the system

$$
(A^2-(\mu_k+\bar\mu_k)A+\mu_k\bar\mu_k I) y_k=x_k.$$
"

# ╔═╡ 0a2d2bb5-2681-472a-b227-ed5b6924062a
md"
# RQIds for Arrowhead

Let

$$
y_k=y,\quad
x_k=\begin{bmatrix} x \\ \xi \end{bmatrix},\quad  \hat\alpha=\mu_k+\bar\mu_k,\quad 
\beta=\mu_k\bar\mu_k.$$

Notice that $\hat \alpha$ and $\beta$ are real. Then:

$$
\left(\begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} -\hat\alpha \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} + \beta \begin{bmatrix} I & 0 \\ 0  & 1 \end{bmatrix} \right) y = \begin{bmatrix} x \\ \chi \end{bmatrix}.$$

Therefore,

$$
\left(\begin{bmatrix} D^2 + uv^* & Du +u\alpha \\ v^*D+\alpha v^* & v^*u+\alpha^2 \end{bmatrix} -\hat\alpha \begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix} + \beta \begin{bmatrix} I & 0 \\ 0  & 1 \end{bmatrix} \right) y = \begin{bmatrix} x \\ \chi \end{bmatrix},$$

so

$$
\begin{bmatrix} D^2 -\hat\alpha D +\beta I +uv^* & 
Du +u(\alpha-\hat\alpha) \\ 
v^*D+(\alpha-\hat\alpha) v^* & 
v^*u+(\alpha-\hat\alpha)\alpha+\beta \end{bmatrix}y = \begin{bmatrix} x \\ \chi \end{bmatrix}. \tag{1}$$
"

# ╔═╡ d3046577-b251-45b0-a743-a9970937811d
md"
# RQIds for Arrowhed (cont.)
The matrix $C=D^2 -\hat\alpha D +\beta I +uv^*$ is a DPRk (DPR1) matrix,

$$
C=\operatorname{DPRk}(D^2 -\hat\alpha D +\beta I,u,v,1).$$

Multiplying of (1) by the block matrix $\begin{bmatrix} C^{-1} & \\ & 1\end{bmatrix}$ from the left yields

$$
My\equiv\begin{bmatrix} I & 
C^{-1}(Du +u(\alpha-\hat\alpha)) \\ 
v^*D+(\alpha-\hat\alpha) v^* & 
v^*u+(\alpha-\hat\alpha)\alpha+\beta \end{bmatrix}y = \begin{bmatrix} C^{-1}x \\ \xi \end{bmatrix}. \tag{2}$$

where $M$ is an arrowhead matrix. Finally, $y=M^{-1}z$, where $z=\begin{bmatrix} C^{-1}x \\ \xi \end{bmatrix}$.

Due to the fast multiplication and computation of inverses, one step requires $O(n)$ operations. 
"

# ╔═╡ 22c35821-40f4-4c64-90b3-2ea2ce4e651c
md"
# Wielandt's deflation

* Let $A$ be a (real, complex, or quaternionic) matrix.
* Let $(\lambda,u)$ be a right eigenpair of $A$.
* Choose $z$ such that $z^* u=1$, say $z^*=\begin{bmatrix} 1/u_1 & 0 & \cdots & 0\end{bmatrix}$.
* Compute the deflated matrix $\tilde A=(I-uz^*)A$.
* Then  $(0,u)$ is an eigenpair of $\tilde A$.  
* Further, if $(\mu,v)$ is an eigenpair of $A$, then $(\mu, \tilde v)$, where $\tilde v=(I-uz^*)v$ is an eigenpair of $\tilde A$.

__Proofs:__ Using $Au=u\lambda$ and $z^*u=1$, the first statement holds since 

$$
\tilde A u= (I-uz^*)A u=Au-uz^*Au=u\lambda-uz^*u\lambda=0.$$

Further,

$$
\begin{aligned}
\tilde A \tilde v &=(I-uz^*)A(I-uz^*)v \\
&= (I-uz^*)Av-Auz^*v+uz^*Auz^*v \\
&= (I-uz^*)v\mu -u\lambda z^*v+uz^*u\lambda z^* v \\
&= \tilde v\mu
\end{aligned}.$$
"

# ╔═╡ fa8ead94-9787-462b-9f41-47fcb41a1a17
md"
# Deflation for Arrowhead

__Lemma 1.__ Let $A$ be an arrowhead matrix partitioned as

$$
A=\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix},$$

where $\chi$, $\upsilon$ and $\alpha$ are scalars, $x$ and $y$ are vectors, and $\Delta$ is a diagonal matrix. 

Let $\bigg(\lambda,\begin{bmatrix}\nu \\ u \\ \psi \end{bmatrix}\bigg)$, where $\nu$ and $\psi$ are scalars, and $u$ is a vector, be an eigenpair of $A$. Then,

$$
\tilde A=\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix},\qquad 
w=\begin{bmatrix}-u\frac{1}{\nu}\delta \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon}\end{bmatrix},\tag{0}$$

and $\hat A$ is an arrowhead matrix

$$
\hat A=\begin{bmatrix} \Delta  & -u\frac{1}{\nu}\chi
+x \\ y^*
& -\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}.\tag{1}$$
"

# ╔═╡ ff113c87-a72d-4556-98f9-e1e42782a1e6
md"
# Deflation for Arrowhead (cont.)

__Lemma 2.__ Let $A$ and $\hat A$ be as in Lemma 1. If $\bigg(\mu, \begin{bmatrix}\hat z \\ \hat \xi\end{bmatrix}\bigg)$ is an eigenpair of $\hat A$, then the eigenpair of $A$ is

$$
\left(\mu, \displaystyle\begin{bmatrix}\zeta \\ \hat z + u\frac{1}{\nu}\zeta \\ \hat \xi + \psi\frac{1}{\nu}\zeta\end{bmatrix}\right), \tag{2}$$

where $\zeta$ is the solution of the scalar Sylvester equation 

$$\bigg(\delta+\chi\psi\frac{1}{\nu}\bigg)\zeta-\zeta \mu=-\chi\hat \xi. \tag{3}$$
"

# ╔═╡ 88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
md"
# Computing the eigenvectors

Let $\left(\lambda,\begin{bmatrix} \nu \\ u \\ \psi\end{bmatrix}\right)$ be an eigenpair of the matrix $A$, that is

$$
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}=\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}\lambda.$$


If $\lambda$ and $\psi$ are known, then the other components of the eigenvector are solutions of scalar Sylvester equations

$$
\begin{aligned}
\delta \nu -\nu \lambda & = -\chi \psi,\qquad\qquad\qquad\qquad\qquad\qquad  (4)\\
\Delta_{ii}u_i-u_i\lambda & =-x_i\psi,\quad i=1,\ldots,n-2.
\end{aligned}$$

By setting

$$
\gamma=\delta+\chi\psi\frac{1}{\nu}$$

the Sylvester equation (3) becomes

$$
\gamma\zeta-\zeta \mu=-\chi\hat \xi.\tag{5}$$

Dividing (4) by $\nu$ from the right gives

$$\gamma=\nu\lambda\frac{1}{\nu}.\tag{6}$$

"

# ╔═╡ a859ec97-d40a-4d35-908d-ccdc16c5fd57
md"
# Algorithm

In the first (forward) pass, in each step the absolutely largest eigenvalue and its eigenvector are computed by the RQIds. 
The first element of the current vector $x$ and the first and the last elements of the current eigenvector are stored. 
The current value $\gamma$ is computed using (6) and stored. The deflation is then performed according to Lemma 1.

The eigenvectors are reconstructed bottom-up, that is from the smallest matrix to the original one (a backward pass). 
In each iteration, we need access to:
* the first element of the vector $x$ which was used to define the current Arrow matrix, 
* its absolutely largest eigenvalue, and 
* the first and the last elements of the corresponding eigenvector.

In the $i$th step, for each $j=i+1,\ldots, n$ the following steps are performed:

1. The equation (5) is solved for $\zeta$ (the first element of the eigenvector of the larger matrix). The quantity $\hat \xi$ is the last element of the eigenvectors and was stored in the forward pass. 
2. The first element of the eigenvector of super-matrix is updated (set to $\zeta$).
3. The last element of the eigenvectors of the super-matrix is updated using (2). 

Iterations are completed in $O(n^2)$ operations.  

After all iterations are completed, we have:

* the computed eigenvalue and its eigenvector (unchanged from the first run of the RQIds), 
* all other eigenvalues and the last elements of their corresponding eigenvectors.  

The rest of the elements of the remaining eigenvectors are computed using the procedure described above. 
This step also requires $O(n^2)$ operations.
"

# ╔═╡ c1cb7779-1b05-44ae-849c-da7e639c34fa
md"

# Corrections

Due to floating-point error in operations with Quaternions, the computed eigenpairs have larger residuals than required. 
This is successfully remedied by running a few steps of the RQIds, starting from the computed eigenvectors. 
This has the effect of using nearly perfect shifts, so typically just a few additional iterations are needed to attain the desired accuracy. 
This step also requires $O(n^2)$ operations.

"

# ╔═╡ 14edcc38-773f-4232-8557-78e1897d6817
md"
# Pseudocode

**Computing all eigenpairs of an Arrow matrix**\
**Require:** an Arrow matrix  $A\in\mathbb{H}^{n \times n}$\
$~~~~$ Compute and store the first eigenpair $(\lambda_1,u)$ using RQIds\
$~~~~$ Compute the deflated matrix $\hat A$ according to Lemma 1\
$~~~~$ Compute $\gamma = {\nu}\lambda \frac{1}{\nu}$\
$~~~~$ Compute and store $\nu, \chi, \psi$ according to Lemma 1\
$~~~~$ **for** $i=2,3,\ldots,n-1$ **do**\
$~~~~~~~~$ Compute $g=\frac{1}{\nu}\chi$\
$~~~~~~~~$ Compute $w$ from (0):  $w = x-ug$\
$~~~~~~~~$ Compute the new matrix $\hat A$ from (1)\
$~~~~~~~~$ Compute and store an eigenpair $(\lambda_i,u)$ of $\hat A$ using RQIds\
$~~~~~~~~$ Update and store $\gamma_i, \nu_i, \chi_i, \psi_i$ according to Lemma 1\
$~~~~$ **end for**\
$~~~~$ Compute and store the last eigenvalue\
$~~~~$ **for** $i=n-1,n-2,\ldots,1$ **do**\
$~~~~~~~~$ **for** $j=i+1,i+2,\ldots,n$ **do**\
$~~~~~~~~~~~~$ Solve the Sylvester equation $\gamma_i \zeta - \zeta \lambda_j = - \chi_i  \psi_j$ for $\zeta$\
$~~~~~~~~~~~~$ Update $\nu_j$ and $\psi_j$, the first and the last element of the eigenvector of 
the super-matrix, respectively:\
$~~~~~~~~~~~~~~~~$ $\nu_j = \zeta,\quad \psi_j = \psi_j+ \psi_i  \frac{\zeta}{\nu_i}$\
$~~~~~~~~$ **end for**\
$~~~~$ **end for**\
$~~~~$ Reconstruct all eigenvectors from the computed eigenvalues and respective first and last elements using (4)\
$~~~~$ Correct the computed eigenpairs by running few steps of RQIds with nearly perfect shifts.
"

# ╔═╡ 78818fc6-997b-4bef-8e8e-679474fd8f06
md"

# DPRk matrices

For DPRk matrices there are analogous results:

* RQIds for DPRk (multiplying by one DPRk matrix ad solving the system with another DPRk matrix with $I$ on the diagonal) - $O(n)$
* Deflation for DPRk - $O(n)$
* Computing the eigenvectors of a DPRk - $O(n^2)$
* Algorithm for DPRk - $O(n^2)$


"

# ╔═╡ d087233c-5e22-40ab-b96f-8bceafd5c72d
md"

# Perturbation theory 

We have the following Bauer-Fike type theorem from
_Sk. Safique Ahmad, Istkhar Ali, and Ivan Slapničar,\
Perturbation analysis of matrices over a quaternion division algebra, ETNA, Volume 54, pp. 128-149, 2021._

__Theorem 1__ Let $A \in \mathbb{H}^{n \times n}$ be a diagonalizable matrix, with $A=XDX^{-1}$, where $X \in \mathbb{H}^{n \times n}$ is invertible and $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ with $\lambda_i$ being the standard right eigenvalues of $A$. If $\mu$ is  a standard right eigenvalue of $A+\Delta A$, then 

$$
\operatorname{dist}(\mu, \Lambda_s(A)) = \min_{\lambda_i \in \Lambda_s(A)} \{ | \lambda_i  - \mu | \} \leq \kappa (X) \| \Delta A \|_2.$$

Moreover, we have

$$
\operatorname{dist}(\xi, \Lambda(A)) = \inf_{\eta_j \in \Lambda(A)} \{ | \eta_j  - \xi | \} \leq \kappa (X) \| \Delta A \|_2,$$

where $\xi \in  \Lambda(A+\Delta A)$ and $\kappa(\cdot)$ is the condition number with respect to the matrix $2$-norm.

"

# ╔═╡ cbea8c8a-3b36-4989-8b3d-ce15a1c13884
md"
# Residual bounds

__Theorem 2__ Let $(\tilde \lambda,\tilde x)$ be the approximate eigenpair of the matrix $A$, where $\|\tilde x\|_2=1$. Let

$$
r=A\tilde x-\tilde x\tilde \lambda,\quad \Delta A=-{r\tilde x^*}.$$

Then, $(\tilde \lambda,\tilde x)$ is the eigenpair of the matrix $A+\Delta A$ and 
$\|\Delta A\|_2\leq \|r\|_2$.

__Theorem 3__ Let $(\tilde \lambda_i,\tilde x_i)$, $i=1,\ldots,m$, be approximate eigenpairs of the matrix $A$, where $\|\tilde x_i\|_2=1$. Set $\tilde \Lambda=\operatorname{diag}(\tilde \lambda_1,\ldots,\tilde \lambda_m)$ and $\tilde X=\begin{bmatrix} \tilde x_1 & \cdots & \tilde x_m\end{bmatrix}$. We assume that eigenvectors are linearly independent. Let

$$
R=A\tilde X-\tilde X\tilde \Lambda,\quad \Delta A=-R(\tilde X^*\tilde X)^{-1}\tilde X^*.$$

Then, $(\tilde \lambda_i,\tilde x_i)$, $i=1,\ldots,m$ are the eigenpairs of the matrix $A+\Delta A$ and

$$
\|\Delta A\|_2\leq \|R\|_2\|(\tilde X^*\tilde X)^{-1}\tilde X^*\|_2.$$
"

# ╔═╡ 0b1cd856-9f43-4e2f-bade-ad01df6aee0e
md"

# Error analysis

An error of the product of two quaternions is bounded as follows (see _Joldes, M.; Muller, J. M., 
Algorithms for manipulating quaternions in floating-point arithmetic.
In IEEE 27th Symposium on Computer Arithmetic (ARITH), Portland, OR, USA, 2020, pp. 48-55_)

__Lemma 3__
Let $p, q \in \mathbb{H}$. Then

$$
| fl(p q) - p q | \leq  (5.75\varepsilon  + \varepsilon^2) |p| |q|.$$

__Lemma 4__
Let $p, q \in \mathbb{H}^n$, that is, $p =(p_1, p_2,\ldots, p_n)$ and 
$q = (q_1, q_2,\ldots, q_n)$, where $p_i,q_i\in\mathbb{H}$ for $i=1,\ldots,n$.
Let $|p|\equiv (|p_1|, |p_2|,\ldots, |p_n|)$ and $|q| \equiv (|q_1|, |q_2|,\ldots, |q_n|)$ denote the corresponding vectors of component-wise absolute values. Then 

$$
| fl(p \cdot q) - p \cdot q | \leq  (2n+5.75)\varepsilon |p| \cdot |q| + \mathcal{O}(\varepsilon^2).$$

__Corollary__
Let $A, B \in \mathbb{H}^{n \times n}$ be matrices of quaternions and $\varepsilon$, and let $|A|$ and $|B|$ denote the corresponding matrices of component-wise absolute values. Then

$$
| fl(A \cdot B) - A \cdot B | \leq  (2n+5.75)\varepsilon |A| \cdot |B| + \mathcal{O}(\varepsilon^2).$$

"

# ╔═╡ ba519f07-3941-4144-b9c4-f293e41bdf23
md"

# Error bounds

For example, let $(\tilde \mu,\tilde x)$ be the computed eigenpair of the matrix $A$, where $\tilde \mu$ is in the standard form and $\|\tilde x\|_2=1$. 
Then we can compute the residual $r$ as in Theorem 2, and Theorem 1 implies that 

$$
\min_{\lambda_i \in \Lambda_s(A)} \{ | \lambda_i  - \tilde \mu | \} \leq \kappa (X) \|r\|_2.$$

We can use the bound effectively if the matrix is diagonalizable and we can approximate the condition of the eigenvector matrix $\kappa (X)$ by the condition of the computed eigenvector matrix $\tilde X$.

If we computed all eigenvalues and all eigenvectors of a diagonalizable matrix, 
$\tilde \Lambda=\operatorname{diag}(\tilde \lambda_1,\ldots,\tilde \lambda_n)$ and $\tilde X$, respectively, then we can compute the residual $R$ as in Theorem 2. Inserting the bound for $\|\Delta A\|_2$ from Theorem 3 into Theorem 1, yields

$$
\max_j \min_{\lambda_i \in \Lambda_s(A)} \{ | \lambda_i  - \tilde \lambda_j | \} \leq \kappa (\tilde X) \|R\|_2\|(\tilde X^*\tilde X)^{-1}\tilde X^*\|_2.$$

If the matrix is normal or Hermitian, then $\kappa(X)=1$, so the bounds are sharper.

"

# ╔═╡ e747a6e4-70df-4aff-993a-e9a9ad51fa03
md"

# Codes and reference

The Julia codes are available at [https://github.com/ivanslapnicar/MANAA](https://github.com/ivanslapnicar/MANAA)

Details, including proofs, are in [Fast Eigenvalue Decomposition of Arrowhead and Diagonal-Plus-Rank-k Matrices of Quaternions,  Mathematics 2024, 12(9), 1327.](https://doi.org/10.3390/math12091327)

"

# ╔═╡ d42c28f0-d33b-4f20-96ba-14309c37d6c7
md"""

# Example 1

Error bounds (green squares), residuals, and actual errors (using `BigFloat`) computed by RQIds (red
dots and diamonds, respectively), and residuals and actual errors computed by QR  (blue
dots and diamonds, respectively). The actual errors are not computed for $n = 40$ and $n = 100$.\
\

$(PlutoUI.LocalResource("../Experiments/Arrow_10_RQIds_12.png", :width=>300))
$(PlutoUI.LocalResource("../Experiments/Arrow_20_RQIds_12.png", :width=>300))\
\
\
$(PlutoUI.LocalResource("../Experiments/Arrow_40_RQIds_12.png", :width=>300))
$(PlutoUI.LocalResource("../Experiments/Arrow_100_RQIds_12.png", :width=>300))

"""

# ╔═╡ a5188d6f-5f7c-499c-b24e-98c61f00743b
md"""

# Example 2

Error bounds (green squares), residuals, and actual errors (using `BigFloat`) computed by RQIds (red
dots and diamonds, respectively), and residuals and actual errors computed by QR  (blue
dots and diamonds, respectively). The actual errors are not computed for $n = 40$ and $n = 100$.\
\

$(PlutoUI.LocalResource("../Experiments/DPRk_10_2_RQIds_12.png", :width=>300))
$(PlutoUI.LocalResource("../Experiments/DPRk_20_2_RQIds_12.png", :width=>300))\
\
\
$(PlutoUI.LocalResource("../Experiments/DPRk_40_3_RQIds_12.png", :width=>300))
$(PlutoUI.LocalResource("../Experiments/DPRk_100_4_RQIds_12.png", :width=>300))

"""

# ╔═╡ a9054ab8-75bd-4960-904a-c404099de35f
md"

# Iterations and running times

Mean number of iterations per eigenvalue and mean total running times for Arrow and DPRk matrices 
of orders $n=10,20,40,100$, using RQIds and QR, respectively.

| **n** | **\# iters Arrow RQIds** | **Time RQIds** | **Time QR** |
| ----- |:---------:|:-------:|:-------:|
| 10    | 8         | 0.00081   | 0.00079 |
| 20    | 9         | 0.0026    | 0.011   |
| 40    | 16        | 0.014     | 0.039   |
| 100   | 32        | 0.17      | 0.47    |


| **n** | **k** | **# iters DPRk RQIds** | **Time RQIds** | **Time QR** |
|-------|-------|:---------:|:-------:|:-------:|
| 10    | 2     | 7         | 0.0018    | 0.00075 |
| 20    | 2     | 9         | 0.0077    | 0.011   |
| 40    | 3     | 16        | 0.031     | 0.071   |
| 100   | 4     | 27        | 0.25      | 0.85    |


"

# ╔═╡ 28ec0511-644d-4c85-af15-fdf42d15c69b
md"

# Conclusions

The key contributions are the following: 

* efficient algorithms for computing eigenvalue decompositions of Arrow and DPRk matrices of quaternions,
* the algorithms require $O(n^2)$ arithmetic operations, $n$ being the order of the matrix,
* algorithms have proven error bounds,
* the computable residual is a good estimate of actual errors, 
* actual errors are even smaller than predicted by the residuals,
* in all experiments errors and residuals are of the order of tolerance from respective algorithms,
* Rayleigh Quotient Iteration with double-shifts is efficient for non-Hermitian matrices,
* RQIds algorithms compare favorably in accuracy and speed to the quaternion QR method for general matrices. 

"

# ╔═╡ a1a4a919-82fa-4a39-9cdc-92ec13b45078
md"

# Thank you!

"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "6e7bcec4be6e95d1f85627422d78f10c0391f199"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─ba49a550-fe54-11ed-1d8b-9b5192874bc6
# ╟─b76f4412-ef87-4185-912f-060d2891df54
# ╟─41db52a5-812e-49bb-9d1f-fb7851740a88
# ╟─2552eeb1-c631-4e58-94f1-3894b9545ab9
# ╟─58a3123e-2359-4af3-bb9d-4f0fdba3c8aa
# ╟─fa9e44d2-5729-45c1-9aa3-a2d35372aebd
# ╟─68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
# ╟─8503c651-2bd3-43f2-9d4c-959962606ff5
# ╟─47f842fa-063d-4b30-a734-3f7d825b1314
# ╟─4d12d9db-66fd-43fb-aa1c-6dafc3c83b75
# ╟─90d86d6c-7896-4a56-b827-1ba2f42d54f9
# ╟─3122ea8b-77d4-4255-9e82-ac8bd8b0bade
# ╟─b299fcf7-7ced-45d1-a55c-74482ecb0c60
# ╟─82c8a221-3125-42a1-a033-02408688b6ae
# ╟─02afde06-04e6-44ae-b8c4-3e8e00d5b7cc
# ╟─24724b41-f108-4dc7-b6dd-193a3e39bc37
# ╟─0a2d2bb5-2681-472a-b227-ed5b6924062a
# ╟─d3046577-b251-45b0-a743-a9970937811d
# ╟─22c35821-40f4-4c64-90b3-2ea2ce4e651c
# ╟─fa8ead94-9787-462b-9f41-47fcb41a1a17
# ╟─ff113c87-a72d-4556-98f9-e1e42782a1e6
# ╟─88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
# ╟─a859ec97-d40a-4d35-908d-ccdc16c5fd57
# ╟─c1cb7779-1b05-44ae-849c-da7e639c34fa
# ╟─14edcc38-773f-4232-8557-78e1897d6817
# ╟─78818fc6-997b-4bef-8e8e-679474fd8f06
# ╟─d087233c-5e22-40ab-b96f-8bceafd5c72d
# ╟─cbea8c8a-3b36-4989-8b3d-ce15a1c13884
# ╟─0b1cd856-9f43-4e2f-bade-ad01df6aee0e
# ╟─ba519f07-3941-4144-b9c4-f293e41bdf23
# ╟─e747a6e4-70df-4aff-993a-e9a9ad51fa03
# ╟─d42c28f0-d33b-4f20-96ba-14309c37d6c7
# ╟─a5188d6f-5f7c-499c-b24e-98c61f00743b
# ╟─a9054ab8-75bd-4960-904a-c404099de35f
# ╟─28ec0511-644d-4c85-af15-fdf42d15c69b
# ╟─a1a4a919-82fa-4a39-9cdc-92ec13b45078
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
