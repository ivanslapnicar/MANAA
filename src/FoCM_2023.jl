### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ba49a550-fe54-11ed-1d8b-9b5192874bc6
using LinearAlgebra, Random, PlutoUI, Quaternions, GenericLinearAlgebra

# ╔═╡ 45cb51a4-9eea-4fa8-8748-8f6ba24e9c48
TableOfContents(title="Contents", aside=true)

# ╔═╡ 484df839-8669-4627-a3c3-919300c1c882
begin
	# Structures
	struct Arrow{T} <: AbstractMatrix{T}
    	D::AbstractVector{T}
    	u::AbstractVecOrMat{T}
    	v::AbstractVecOrMat{T}
    	α
    	i::Int
	end

	struct DPRk{T} <: AbstractMatrix{T}
    	Δ::AbstractVector{T}
    	x::AbstractVecOrMat{T}
    	y::AbstractVecOrMat{T}
		ρ
	end
end

# ╔═╡ 53a198ac-e7bf-4dc2-9ae3-67f94f15a694
begin
	import Base:*,-
	function *(A::Arrow,z::Vector)
	    n=size(A,1)
	    T=typeof(A.u[1])
	    w=Vector{T}(undef,n)
	    i=A.i
	    zi=z[i]
	    for j=1:i-1
	        w[j]=A.D[j]*z[j]+A.u[j]*zi
	    end
		ind=[1:i-1;i+1:n]
		w[i]=A.v⋅z[ind]+A.α*zi
	    # w[i]=adjoint(A.v[1:i-1])*z[1:i-1]+A.α*zi+adjoint(A.v[i:n-1])*z[i+1:n]
	    for j=A.i+1:n
	        w[j]=A.u[j-1]*zi+A.D[j-1]*z[j]
	    end
	    return w
	end
	
	function *(A::DPRk,z::Vector)
	    n=size(A,1)
	    T=typeof(A.x[1])
	    w=Vector{T}(undef,n)
	    β=A.ρ*(adjoint(A.y)*z)
		return Diagonal(A.Δ)*z+A.x*β
	end

	-(A::Arrow,D::Diagonal)=Arrow(A.D-D.diag[1:end-1],A.u,A.v,A.α-D.diag[end],A.i)
	-(A::DPRk,D::Diagonal)=DPRk(A.Δ-D.diag,A.x,A.y,A.ρ)

end

# ╔═╡ 23817579-826c-47fb-aeee-d67712b59ada
begin
	import Base: size, getindex
	import  LinearAlgebra: Matrix, adjoint, transpose

	# Arrowhead
	size(A::Arrow, dim::Integer) = length(A.D)+1
	size(A::Arrow)= size(A,1), size(A,1)
	
	function getindex(A::Arrow,i::Integer,j::Integer)
	    n=size(A,1)
	    if i==j<A.i; return A.D[i]
	    elseif i==j>A.i; return A.D[i-1]
	    elseif i==j==A.i; return A.α
		elseif i==A.i&&j<A.i; return adjoint(A.v[j])
		elseif i==A.i&&j>A.i; return adjoint(A.v[j-1])
		elseif j==A.i&&i<A.i; return A.u[i]
		elseif j==A.i&&i>A.i; return A.u[i-1]
	    else 
	        return zero(A.D[1])
	    # return zeros(size(A.D[i<A.i ? i : i-1],1),size(A.D[j<A.i ? j : j-1],1))
	    end
	end
	
	Matrix(A::Arrow) =[A[i,j] for i=1:size(A,1), j=1:size(A,2)]
	adjoint(A::Arrow)=Arrow(adjoint.(A.D),A.v,A.u, adjoint(A.α),A.i)
	transpose(A::Arrow)=Arrow(A.D, conj.(A.u), conj.(A.v),A.α,A.i)
	
	# DPRk
	size(A::DPRk, dim::Integer) = length(A.Δ)
	size(A::DPRk)= size(A,1), size(A,1)
	
	function getindex(A::DPRk,i::Integer,j::Integer)
		# This is because Julia extracts rows as column vectors
	    Aij=conj.(A.x[i,:])⋅(A.ρ*conj.(A.y[j,:]))
	    return i==j ? A.Δ[i].+Aij : Aij
	end
	
	Matrix(A::DPRk)=[A[i,j] for i=1:size(A,1), j=1:size(A,2)]
	adjoint(A::DPRk)=DPRk(adjoint.(A.Δ),A.y,A.x,adjoint(A.ρ))
end

# ╔═╡ 41db52a5-812e-49bb-9d1f-fb7851740a88
md"""
# Fast computations with arrowhead and diagonal-plus-rank-k matrices over associative fields

#### by Ivan Slapničar, Thaniporn Chaysri and  Nevena Jakovčević Stor

#### from University of Split, FESB

#### presented at FoCM, Workshop III.1 _Numerical Linear Algebra_, Paris, June 19-21, 2023.

This work has been fully supported by Croatian Science Foundation under the project IP-2020-02-2240 - [http://manaa.fesb.unist.hr](http://manaa.fesb.unist.hr).

$(PlutoUI.LocalResource("./HRZZ-eng-170x80-1.jpg"))

"""

# ╔═╡ dd28cb55-9258-4a32-b3ef-d6af94e256cc
md"
> We present efficient $O(n^2)$ eigensolvers for arrowhead and DPRk matrices of quaternions. The eigensolvers use a version of Wielandt deflation. Algorithms are elegantly implemented using Julia's polymorphism.
"

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

Let $f(x)$ be a complex analytic function. The value $f(q)$, where $q\in\mathbb{H}$, is computed by evaluating the extension of $f$ to the quaternions at $q$, see ([Sudbery,1979](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/quaternionic-analysis/308CF454034EC347D4D17D1F829F8471)), for example,

$$\sqrt{q}=\pm \left(\sqrt {\frac {\|q\|+a_1}{2}} + \frac {\operatorname{imag} (q)}{\|\operatorname{imag}(q)\|} \sqrt {\frac {\|q\|-a_1}{2}}\right).$$

Basic operations with quaternions and computation of the functions of quaternions are implemented in the package [Quaternions.jl](https://github.com/JuliaGeometry/Quaternions.jl).

## Standard form

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


## Homomorphism 

Quaternions are homomorphic to $\mathbb{C}^{2\times 2}$:

$$
q\to \begin{bmatrix}a+b\,\mathbf{i} & c+d\, \mathbf{i}\\-c+d\, \mathbf{i} & a-b\, \mathbf{i}\end{bmatrix}\equiv C(q),$$

with eigenvalues $q_s$ and $\bar q_s$.
"

# ╔═╡ 68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
md"
# Matrices

All matrices are in $\mathbb{F}^{n\times n}$ where $\mathbb{F}\in\{\mathbb{R},\mathbb{C},\mathbb{H}\}$.
$\mathbb{H}$ is a non-commutative field of quaternions.
  
__Arrowhead matrix__ (__Arrow__) is a matrix of the form

$$
A=\begin{bmatrix} D & u \\v^* & \alpha \end{bmatrix},$$

where 

$$\mathop{\mathrm{diag}}(D),u, v \in\mathbb{F}^{n-1},\quad \alpha\in\mathbb{F},$$

or any symmetric permutation of such a matrix.

__Diagonal-plus-rank-$k$ matrix__ (__DPRk__) is a matrix of the form 

$$
A=\Delta+x ρy^*$$

where 

$$\mathop{\mathrm{diag}}(\Delta)\in \mathbb{F}^{n},\quad x, y \in\mathbb{F}^{n\times k},\quad  \rho \in \mathbb{F}^{k\times k}.$$
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
# Inverses

Inverses are computed in $O(n)$ operations.

## Arrowhead

Let $A=\operatorname{Arrow}(D,u,v,\alpha)$ be nonsingular.

Let $P$ be the permutation matrix of the permutation $p=(1,2,\cdots,i-1,n,i,i+1,\cdots,n-1)$. 

If all $d_j\neq 0$, the inverse of $A$ is a DPRk (DPR1) matrix

$$
A^{-1} =\Delta+x \rho y^*,$$

where 

$$
\Delta=P\begin{bmatrix}D^{-1} & 0\\ 0 & 0\end{bmatrix}P^T,
\quad x=P\begin{bmatrix}D^{-1}u \\-1\end{bmatrix}\rho,\quad
y=P\begin{bmatrix}D^{-\star}v \\-1\end{bmatrix},\quad
\rho=(\alpha-v^\star D^{-1} u)^{-1}.$$


If $d_j=0$, the inverse of $A$ is an Arrow with the tip of the arrow at position $(j,j)$ and zero at position $A_{ii}$ (the tip and the zero on the shaft change places). In particular, let $\hat P$ be the permutation matrix of the permutation $\hat p=(1,2,\cdots,j-1,n,j,j+1,\cdots,n-1)$. Partition $D$, $u$ and $v$ as

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
\hat v= \begin{bmatrix}-D_1^{-\star}v_1 \\ -D_2^{-\star}v_2\\ 1\end{bmatrix}v_j^{-1},\\
\hat \alpha&=v_j^{-\star}\left(-\alpha +v_1^\star D_1^{-1} u_1+v_2^\star D_2^{-1}u_2\right) u_j^{-1}.
\end{align*}$$

## DPRk

Let $A=\operatorname{DPRk}(\Delta,x,y,\rho)$ be nonsingular. 

If all $\delta_j\neq 0$, the inverse of $A$ is a DPRk matrix

$$
A^{-1} =\hat\Delta+\hat x\hat \rho \hat y^*,$$

where 

$$
\hat \Delta=\Delta^{-1},\quad 
\quad \hat x=\Delta^{-1}x,\quad
\hat y=\Delta^{-*}y,\quad
\hat \rho=-\rho(I-y^* \Delta^{-1} x\rho)^{-1}.$$


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
v= \begin{bmatrix}-\Delta_1^{-\star}y_1 \\ -\Delta_2^{-\star}y_2\end{bmatrix}y_j^{-1},\\
\alpha&=(y_j^{-1})^\star\left(\rho^{-1} +y_1^\star \Delta_1^{-1} x_1+y_2^\star \Delta_2^{-1}x_2\right) x_j^{-1}.
\end{align*}$$

"""

# ╔═╡ b299fcf7-7ced-45d1-a55c-74482ecb0c60
md"
# Eigenvalue decomposition

Right eigenpairs $(λ,x)$ of a quaternionic matrix satisfy

$$
Ax=xλ, \quad x\neq 0.$$

Usually, $x$ is chosen such that $\lambda$ is the standard form.

Eigenvalues are invariant under similarity.

> Eigenvalues are __NOT__ shift invariant, that is, eigenvalues of the shifted matrix are __NOT__ the shifted eigenvalues. 

If $\lambda$ is in the standard form, it is invariant under similarity with complex numbers.  
"

# ╔═╡ f3a47c9d-c3ba-4056-a3d7-bb4050b3175c
md"
## Power method

The power method produces a sequence of vectors

$$
y_k=Ax_k, \quad x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

If $\lambda$ is a dominant eigenvalue, and $x_0$ has a component in the direction of its unit eigenvector $x$, then $x_k\to x$ and $x^*Ax=\lambda$. The convergence is linear.

> If $A$ is an arrowhead or a DPRk matrix, then, due to fast matrix × vector multiplication, one step of the method requires $O(n)$ operations.  
"

# ╔═╡ f6769f8f-19ad-47c5-a1ec-2e3c780f6cd7
md"
## RQI and MRQI

The __Rayleigh Quotient Iteration__ (RQI) produces sequences of shifts and vectors

$$
\mu_k=\frac{1}{x_k^*x_k} x_k^*Ax_k,\quad y_k=(A-\mu_kI)^{-1} x_k, \quad  x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

The __Modified Rayleigh Quotient Iteration__ (MRQI) produces sequences of shifts and vectors

$$
\mu_k=\frac{1}{x^Tx} x^TAx,\quad y_k=(A-\mu_kI)^{-1} x_k, \quad  x_{k+1}=\frac{y_k}{\| y_k\|},\quad k=0,1,2,\ldots$$

Since the eigenvalues are not shift invariant, only real shifts can be used. However, this works fine due to the following: let the matrix $A -\mu I$ have purely imaginary standard eigenvalue:

$$(A-\mu I)x=x (i λ),\qquad \mu,\lambda\in\mathbb{R}.$$

Then 

$$Ax=\mu x+xi\lambda=x(\mu+i\lambda).$$

> If $A$ is an arrowhead or a DPRk matrix, then, due to fast inverses, one step of the methods requires $O(n)$ operations.
"

# ╔═╡ 22c35821-40f4-4c64-90b3-2ea2ce4e651c
md"
## Wielandt's deflation

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
## Arrowhead matrices

__Lemma 1.__ Let $A$ be an arrowhead matrix partitioned as

$$
A=\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix},$$

where $\chi$, $\upsilon$ and $\alpha$ are scalars, $x$ and $y$ are vectors, and $\Delta$ is a diagonal matrix. 

Let $\bigg(\lambda,\begin{bmatrix}\nu \\ u \\ \psi \end{bmatrix}\bigg)$, where $\nu$ and $\psi$ are scalars, and $u$ is a vector, be an eigenpair of $A$. Then, the deflated matrix $\tilde A$ has the form 

$$
\tilde A=\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix}, \tag{1}$$

where 

$$
w=\begin{bmatrix}-u\frac{1}{\nu}\delta \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon}\end{bmatrix},$$

and $\hat A$ is an arrowhead matrix

$$
\hat A=\begin{bmatrix} \Delta  & -u\frac{1}{\nu}\chi
+x \\ y^*
& -\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}.\tag{2}$$

__Proof:__ We have

$$
\begin{aligned}
\tilde A&=\bigg(\begin{bmatrix} 1 & 0^T & 0 \\ 0 & I & 0 \\ 0 & 0^T & 1 \end{bmatrix} - 
\begin{bmatrix} \nu \\ u \\ \psi \end{bmatrix}  \begin{bmatrix} \frac{1}{\nu} & 0^T & 0 \end{bmatrix} \bigg)
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}
\\
&= \begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\\
&= \begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix},
\end{aligned} \tag{3}$$

as desired. $\square$
"

# ╔═╡ ff113c87-a72d-4556-98f9-e1e42782a1e6
md"

__Lemma 2.__ Let $A$ and $\hat A$ be as in Lemma 4. If $\bigg(\mu, \begin{bmatrix}\hat z \\ \hat \xi\end{bmatrix}\bigg)$ is an eigenpair of $\hat A$, then the eigenpair of $A$ is

$$
\left(\mu, \displaystyle\begin{bmatrix}\zeta \\ \hat z + u\frac{1}{\nu}\zeta \\ \hat \xi + \psi\frac{1}{\nu}\zeta\end{bmatrix}\right), \tag{4}$$

where $\zeta$ is the solution of the scalar Sylvester equation 

$$\bigg(\delta+\chi\psi\frac{1}{\nu}\bigg)\zeta-\zeta \mu=-\chi\hat \xi. \tag{5}$$


__Proof:__ If $\mu$ is an eigenvalue of $\hat A$, it is obviously also an eigenvalue of $\tilde A$, and then also of $A$. Assume that the corresponding eigenvector of $A$ is partitioned as $\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}$. By combining (1), (2) and (3), we have 

$$
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu} & I & 
0 \\ -\psi \frac{1}{\nu} & 0^T & 1 \end{bmatrix}
\displaystyle\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}\mu,$$

or

$$
\begin{bmatrix} 0 & 0^T  & 0\\
-u\frac{1}{\nu}\delta & \Delta & 
-u \frac{1}{\nu}\chi+x \\ -\psi\frac{1}{\nu}\delta+\bar{\upsilon} & y^* & 
-\psi\frac{1}{\nu}\chi+\alpha \end{bmatrix}
\begin{bmatrix} 0  \\ -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix}=
\begin{bmatrix} 0  \\ -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix}\mu,$$

Since the bottom right $2\times 2$ matrix is $\hat A$, the above equation implies

$$
\begin{bmatrix}\hat z \\ \hat \xi\end{bmatrix}=\begin{bmatrix} -u\frac{1}{\nu}\zeta+z \\ -\psi \frac{1}{\nu}\zeta + \xi\end{bmatrix},$$

which proves (4). It remains to compute $\zeta$. The first component of the equality 

$$
A\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}=\begin{bmatrix}\zeta \\ z \\ \xi\end{bmatrix}\mu$$

implies

$$
\delta \zeta +\chi \xi=\zeta \mu,$$

or 

$$\delta \zeta + \chi \big(\hat \xi +\psi \frac{1}{\nu}\zeta\big)=\zeta \mu,$$ 

which is exactly the equation (5). $\square$
"

# ╔═╡ 88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
md"
### Computing the eigenvectors

Let $\left(\lambda,\begin{bmatrix} \nu \\ u \\ \psi\end{bmatrix}\right)$ be an eigenpair of the matrix $A$, that is

$$
\begin{bmatrix} \delta & 0 & \chi \\ 0 & \Delta & x\\ \bar \upsilon & y^* & \alpha \end{bmatrix}\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}=\begin{bmatrix}\nu \\ u \\ \psi\end{bmatrix}\lambda.$$


If $\lambda$ and $\psi$ are known, then the other components of the eigenvector are solutions of scalar Sylvester equations

$$
\delta \nu -\nu \lambda = -\chi \psi,\tag{6}$$

$$\Delta_{ii}u_i-u_i\lambda =-x_i\psi,\quad i=1,\ldots,n-2.$$

By setting

$$
\gamma=\delta+\chi\psi\frac{1}{\nu}$$

the Sylvester equation (5) becomes

$$
\gamma\zeta-\zeta \mu=-\chi\hat \xi. \tag{7}$$

Dividing (6) by $\nu$ from the right gives

$$\gamma=\nu\lambda\frac{1}{\nu}. \tag{8}$$
"

# ╔═╡ a859ec97-d40a-4d35-908d-ccdc16c5fd57
md"
### Algorithm

In the first (forward) pass, in each step the absolutely largest eigenvalue and its eigenvector are computed by the power method. The first element of the current vector $x$ and the the first and the last elements of the current eigenvector are stored. The current value $\gamma$ is computed using (8) and stored. The deflation is then performed according to Lemma 1.

The eigenvectors are reconstructed bottom-up, that is from the smallest matrix to the original one (a backward pass). In each iteration we need to have the access to the first element of the vector $x$ which was used to define the current Arrow matrix, its absolutely largest eigenvalue, and the first and the last elements of the corresponding eigenvector.

In the $i$th step, for each $j=i+1,\ldots, n$ the following steps are performed:

1. The equation (5) is solved for $\zeta$ (the first element of the eigenvector of the larger matrix). The quantity $\hat \xi$ is the last element of the eigenvectors and was stored in the forward pass. 
2. The first element of eigenvector of super-matrix is updated (set to $\zeta$).
3. The last element of the eigenvectors of the super matrix is updated using (4). 

Iterations are completed in $O(n^2)$ operations.  

After all iterations are completed, we have:

* the absolutely largest eigenvalue and its eigenvector (unchanged from the first run of the chosen eigensolver), 
* all other eigenvalues and the last elements of their corresponding eigenvectors.  

The rest of the elements of the remaining eigenvectors are computed using the procedure described at the beginning of the previous section. This step also requires $O(n^2)$ operations.
"

# ╔═╡ 09e10ed8-4093-4a43-b452-d2544f29c01c
md"
## DPRk matrices

__Lemma 3.__ Let $A$ be a DPRk matrix partitioned as

$$
A=\begin{bmatrix} \delta & 0^T\\ 0 & \Delta \end{bmatrix} +
\begin{bmatrix} \chi \\ x\end{bmatrix} \rho
\begin{bmatrix} \bar{\upsilon} & y^*\end{bmatrix}.$$

Let $\bigg(\lambda,\begin{bmatrix}\nu \\ u\end{bmatrix}\bigg)$ be the eigenpair of $A$. Then, the deflated matrix $\tilde A$ has the form 

$$
\tilde A=\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix}, \tag{9}$$

where 

$$
w=-u\frac{1}{\nu}\delta-u\frac{1}{\nu}\chi\rho \bar \upsilon +x\rho \bar \upsilon$$

and $\hat A$ is a DPRk matrix

$$
\hat A=\Delta + \hat x \rho y^*,\quad \hat x=x-u\frac{1}{\nu}\chi. \tag{10}$$

__Proof:__ We have

$$
\begin{aligned}
\tilde A&=\bigg(\begin{bmatrix} 1 & 0^T\\ 0 & I \end{bmatrix} - 
\begin{bmatrix} \nu \\ u\end{bmatrix}  \begin{bmatrix} \frac{1}{\nu} & 0^T\end{bmatrix} \bigg)
\bigg(
\begin{bmatrix} \delta & 0^T\\ 0 & \Delta \end{bmatrix} +
\begin{bmatrix} \chi \\ x\end{bmatrix} \rho
\begin{bmatrix} \bar{\upsilon} & y^*\end{bmatrix}\bigg)
\\
&=\begin{bmatrix} 0 & 0^T\\ -u\frac{1}{\nu} & I \end{bmatrix}\cdot
\begin{bmatrix}\delta +\chi \rho \bar\upsilon & \chi \rho y^*\\ 
x\rho \bar \upsilon & \Delta + x\rho y^* \end{bmatrix} \\
&= \begin{bmatrix} 0 & 0^T \\
-u\frac{1}{\nu}\delta-u\frac{1}{\nu}\chi\rho \bar \upsilon +x\rho \bar \upsilon &
-u\frac{1}{\nu}\chi \rho y^* + \Delta + x\rho y^* \end{bmatrix},
\end{aligned} \tag{11}$$

as desired. $\square$
"

# ╔═╡ 0606fb52-9631-48d0-bf62-d5a5a50deaa5
md"
__Lemma 4.__ Let $A$, $\tilde A$, and $\hat A$ be as in Lemma 1. If $(\mu, \hat z)$ is an eigenpair of $\hat A$, then the eigenpair of $A$ is

$$
\bigg(\mu, \displaystyle\begin{bmatrix}\zeta \\ \hat z+u\frac{1}{\nu}\zeta\end{bmatrix}\bigg),$$

where $\zeta$ is the solution of the Sylvester equation 

$$
(\delta +\chi \rho \bar\upsilon+\chi \rho y^*u\frac{1}{\nu})\zeta  - \zeta \mu =  
-\chi\rho y^*\hat z. \tag{12}$$



__Proof:__ If $\mu$ is an eigenvalue of $\hat A$, it is obviously also an eigenvalue of $\tilde A$, and then also of $A$. Assume that the corresponding eigenvector of $A$ is partitioned as $\displaystyle\begin{bmatrix}\zeta \\ z\end{bmatrix}$. By combining (9) and (11) and the previous results, it must hold

$$
\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix} \begin{bmatrix} 0 & 0^T\\ -u\frac{1}{\nu} & I \end{bmatrix}\begin{bmatrix} \zeta \\ z\end{bmatrix}
= \begin{bmatrix} 0 & 0^T\\ -u\frac{1}{\nu} & I \end{bmatrix}\begin{bmatrix} \zeta \\ z\end{bmatrix} \mu$$

or 

$$
\begin{bmatrix} 0 & 0^T \\ w & \hat A\end{bmatrix}\begin{bmatrix} 0 \\ -u\frac{1}{\nu}\zeta+z \end{bmatrix}=\begin{bmatrix} 0 \\ -u\frac{1}{\nu}\zeta+z \end{bmatrix} \mu.$$

Therefore, $\hat z=-u\frac{1}{\nu}\zeta+z$, or

$$
z=\hat z+u\frac{1}{\nu}\zeta, \tag{13}$$

and it remains to compute $\zeta$. From the equality 

$$
\bigg(\begin{bmatrix} \delta & 0^T\\ 0 & \Delta \end{bmatrix} +
\begin{bmatrix} \chi \\ x\end{bmatrix} \rho
\begin{bmatrix} \bar{\upsilon} & y^*\end{bmatrix}\bigg)
\begin{bmatrix} \zeta \\ z\end{bmatrix}=
\begin{bmatrix} \zeta \\ z\end{bmatrix}\mu$$

it follows

$$
\begin{bmatrix}\delta +\chi \rho \bar\upsilon & \chi \rho y^*\\ 
x\rho \bar \upsilon & \Delta + x\rho y^* \end{bmatrix}
\begin{bmatrix} \zeta \\ z\end{bmatrix}=
\begin{bmatrix} \zeta \\ z\end{bmatrix}\mu.$$

Equating the first elements and using (13) gives

$$
(\delta +\chi \rho \bar\upsilon)\zeta + \chi\rho y^*\hat z + \chi \rho y^*u\frac{1}{\nu}\zeta = \zeta \mu,$$

which is exactly the Sylvester equation (12). $\square$
"

# ╔═╡ 6680f116-a563-40e0-8be3-e69f6910eb57
md"
### Computing the eigenvectors

Let the DPRk matrix $A$ and its eigenpair be defined as in Lemma 3. Let $x$ be partitioned row-wise as

$$x=\begin{bmatrix}x_1 \\x_2 \\ \vdots \\x_{n-1} \end{bmatrix}.$$

Set $\alpha=\rho \begin{bmatrix} \bar \upsilon & y^* \end{bmatrix}\begin{bmatrix}\nu \\
u \end{bmatrix}$. From 

$$
\bigg(\begin{bmatrix} \delta & 0^T\\ 0 & \Delta \end{bmatrix} +
\begin{bmatrix} \chi \\ x\end{bmatrix} \rho
\begin{bmatrix} \bar{\upsilon} & y^*\end{bmatrix}\bigg)\begin{bmatrix}\nu \\ u\end{bmatrix} = \begin{bmatrix}\nu \\ u\end{bmatrix} \lambda.$$

it follows that the elements of the eigenvector satisfy scalar Sylvester equations

$$
\begin{aligned}
\delta \nu -\nu \lambda &= -\chi\alpha,\\
\Delta_{ii} u_i-u_i\lambda&=-x_i\alpha, \quad i=1,\ldots,n-1. \qquad\qquad (14) 
\end{aligned}$$

If $\lambda$, $\nu$ and the first $k-1$ components of $u$ are known, then $\alpha$ is computed from the first $k$ equations in (14), that is, by solving the system

$$
\begin{bmatrix}\chi \\
x_{1}\\ \vdots \\ x_{k-1} \end{bmatrix}
\alpha= \begin{bmatrix} \nu\lambda-\delta \nu \\ 
u_1\lambda-\Delta_{11} u_1 \\
\vdots \\
u_{k-1}\lambda-\Delta_{k-1,k-1} u_{k-1}\end{bmatrix},$$

and $u_{i}$, $i=k,\ldots,n-1$, are computed by solving the remaining Sylvester equations in (14).
"

# ╔═╡ 64094f9b-6c1d-4d68-bccc-fd286c121ea1
md"
__Lemma 5.__ Assume $A$ and its eigenpair are given as in Lemma 3, and $\hat A$ and its eigenpair are given as in Lemma 4. 
Set in (12)

$$
\gamma=\delta +\chi \rho \bar\upsilon+\chi \rho y^*u\frac{1}{\nu}, \quad 
\alpha=\rho y^*\hat z.$$

Then, 

$$
\gamma=\nu \lambda \frac{1}{\nu}, \tag{15}$$ 

and $\alpha$ is the solution of the system

$$
\begin{bmatrix}
x_1-u_1\frac{1}{\nu}\chi \\
\vdots\\
x_k-u_k\frac{1}{\nu}\chi
\end{bmatrix} 
\alpha=\begin{bmatrix}\hat z_1\mu-\Delta_{11} \hat z_1 \\
\vdots \\
\hat z_k\mu-\Delta_{kk} \hat z_k
\end{bmatrix}. \tag{16}$$

__Proof:__ The formula for $\gamma$ follows by multiplying the first elements of the equation

$$
\bigg(
\begin{bmatrix} \delta & 0^T\\ 0 & \Delta \end{bmatrix} +
\begin{bmatrix} \chi \\ x\end{bmatrix} \rho
\begin{bmatrix} \bar{\upsilon} & y^*\end{bmatrix}\bigg)\begin{bmatrix}\nu \\ u\end{bmatrix}=\begin{bmatrix}\nu \\ u\end{bmatrix}\lambda$$

with $\frac{1}{\nu}$ from the right.

Consider the equation $\hat A\hat z=\hat z\mu$, that is,

$$
[\Delta +(x-u\frac{1}{\nu}\chi)\rho y^*]\hat z=\hat z\mu.$$

The $i$-th component is

$$
\Delta_{ii}\hat z_i+(x_i-u_i\frac{1}{\nu}\chi)\alpha=\hat z_i\mu,$$

which gives (16). $\square$
"


# ╔═╡ de34d7c6-8632-4ff5-a6ca-082d1f7c1931
md"
### Algorithm

Lemmas 3 and 5 are used as follows. In the first (forward) pass, in each step the absolutely largest eigenvalue and its eigenvector are computed by the power method. The first two elements of the current vector $x$ and the current eigenvector are stored. The current value $\gamma$ is computed using (15) and stored. The deflation is then performed according to Lemma 3.

The eigenvectors are reconstructed bottom-up, that is from the smallest $2\times 2$ matrix to the original one (a backward pass). In each iteration we need to have the access to the first two elements of the vector $x$ which was used to define the current DPRk matrix, its absolutely largest eigenvalue, and the first two elements of the corresponding eigenvector.

In the $i$th step, for each $j=i+1,\ldots, n$, the following steps are performed:

1. The value $\alpha$ is computed from (16).
2. The equation (12), which now reads $\gamma\zeta-\zeta \mu=-\chi \alpha$ is solved for $\zeta$ (the first element of the eigenvector of the larger matrix).
3. First element of eigenvector of super-matrix is updated (set to $\zeta$).

Iterations are completed in $O(n^2)$ operations.  

After all iterations are completed, we have:

* the first computed eigenvalue and its eigenvector (unchanged from the first run of the eigensolver of choice), 
* all other eigenvalues and the first elements of their corresponding eigenvectors.  

The rest of the elements of the remaining eigenvectors are computed using the procedure described at the beginning of the section. This step also requires $O(n^2)$ operations. 
"

# ╔═╡ e747a6e4-70df-4aff-993a-e9a9ad51fa03
md"
# Code
"

# ╔═╡ cf8174d9-035d-4463-ba8b-88b1b6b44317
md"
## Quaternions
"

# ╔═╡ 3f8094c4-93ce-4b01-9b50-c7d66031a610
begin
	import Base: eps, imag
	eps(::Type{Complex{T}}) where T=eps(T)
	eps(::Type{Quaternions.Quaternion{T}}) where T=eps(T)
	const QuaternionF64=Quaternion{Float64}
	im=Quaternion(0,1,0,0)
	jm=Quaternion(0,0,1,0)
	km=Quaternion(0,0,0,1)
	Quaternion{T}(x::Complex) where {T<:Real} = Quaternion(convert(T,real(x)),convert(T,imag(x)),0,0)
	imag(q::Quaternion)=(q-conj(q))/2

	# Quaternion to 2x2 complex
	function q2c(c::T) where T<:QuaternionF64
	    return [complex(c.s, c.v1) complex(c.v2,c.v3);
	        	complex(-c.v2,c.v3) complex(c.s,-c.v1)]
	end
	# For compatibility
	function q2c(c::T) where T
	    return c
	end

	# Converts block matrix to ordinary matrix
	unblock(A) = mapreduce(identity, hcat, [mapreduce(identity, vcat, A[:,i]) 
        for i = 1:size(A,2)])
end

# ╔═╡ a36a7988-5aaf-4a73-8793-b842b00cfbe8
begin
	# Test the arithmetic
	a=randn(QuaternionF64)
	b=√a
	abs(b*b-a)
end

# ╔═╡ 4e9666dc-26f7-429e-afc5-c72f14a34e9a
begin
	function standardformx(a::Vector{T}) where T<:QuaternionF64
		# Computes vector x such that inv.(x).*a .*x is in the standard form
 		n=length(a)
		x=Array{T}(undef,n)
    	for i=1:n
        	x[i]=standardformx(a[i])
    	end
    	x
	end  

	function standardformx(a::QuaternionF64)
		# Return standard form of a
        b=copy(a)
        if norm([b.v2 b.v3])>0.0
            x=norm(Quaternions.imag(b))+b.v1-b.v3*jm+b.v2*km
            x/=abs(x)
        elseif b.v1<0.0
            x=-jm
        else
            x=1.0
        end
        return x
	end

	function standardform(a::QuaternionF64)
		# Return standard form of a
        x=standardformx(a)
        # watch out for the correct division: / and not \
    	return (x\a)*x
	end

	standardform(a::Float64)=a
	standardformx(a::Float64)=one(a)
	standardform(a::ComplexF64)=a
	standardformx(a::ComplexF64)=one(a)
end

# ╔═╡ ddd9a82c-5d26-42be-80e1-087ba826ee10
md"
###  Standard form
"

# ╔═╡ dc11a4fe-c905-4cb5-a90d-8256cb469a39
md"
## Matrices
"

# ╔═╡ bdeab9ab-4838-4332-9b74-7bba33ccb317
md"
## `*()`
"

# ╔═╡ 322273eb-ca50-42b1-866a-3977700e9b63
md"
## `inv()`
"

# ╔═╡ a20a192a-d4c8-4a9e-936f-fac0c8be6b39
begin
	import LinearAlgebra.inv
	function inv(A::Arrow)
	    j=findfirst(iszero.(A.D))
	    if j==nothing
	        p=[1:A.i-1;length(A.D)+1;A.i:length(A.D)]
	        Δ=inv.(A.D)
	        x=Δ.* A.u
			push!(x,-one(x[1]))
			y=adjoint.(Δ) .* A.v
			push!(y,-one(y[1]))
	        ρ=inv(A.α-adjoint(A.v)*(Δ .*A.u))
			push!(Δ,zero(Δ[1]))
	        return DPRk(Δ[p],x[p],y[p],ρ)
	    else
			n=length(A.D)
	        ind=[1:j-1;j+1:n]
	        D=A.D[ind]
	        u=A.u[ind]
	        v=A.v[ind]
	        pₕ=collect(1:n)
	        deleteat!(pₕ,n)
	        iₕ= (j>=A.i) ? A.i : A.i-1
	        insert!(pₕ,iₕ,n)

			# Little bit elaborate to acommodate blocks
			Dₕ=inv.(D)
			uₕ=-Dₕ .* u
			push!(uₕ,one(uₕ[1]))
			uₕ*=inv(A.u[j])
			
	        vₕ=-adjoint.(Dₕ) .* v
			push!(vₕ,one(D[1]))
			vₕ*=inv(A.v[j])
			
	        αₕ=adjoint(inv(A.v[j]))*(-A.α+adjoint(v)*(Dₕ .* u))*inv(A.u[j])
	        
			push!(Dₕ,zero(D[1]))
			jₕ=(j<A.i) ? j : j+1
	        return Arrow(Dₕ[pₕ],uₕ[pₕ],vₕ[pₕ],αₕ,jₕ)
	    end
	end

	function inv(A::DPRk)
    	j=findfirst(iszero.(A.Δ))
		n=length(A.Δ)
    	if j==nothing
			Δₕ=inv.(A.Δ)
        	xₕ=Δₕ .* A.x
        	yₕ=adjoint.(Δₕ) .* A.y
        	ρₕ=-A.ρ*inv(I+adjoint(A.y)*(Δₕ .* (A.x*A.ρ)))
        	return DPRk(Δₕ,xₕ,yₕ,ρₕ)
    	else
        	ind=[1:j-1;j+1:n]
        	Δ=inv.(A.Δ[ind])
        	x=A.x[ind,:]
        	y=A.y[ind,:]
        	uₕ=(-Δ .* x)*inv(A.x[j])
        	vₕ=(-adjoint.(Δ) .* y)*inv(A.y[j])
        	αₕ=adjoint(inv(A.y[j]))*(inv(A.ρ)+adjoint(y)*(Δ .* x)) *inv(A.x[j]) 
			println(" inv else ")
    	    return Arrow(Δ,uₕ,vₕ,αₕ,j)
    	end
	end
end

# ╔═╡ 1c0a18a8-8490-4790-b3b7-7f56edab3d43
md"
## `Power()`
"

# ╔═╡ 15b01358-f35b-4d43-b953-d0f046760db6
function Power(A::AbstractMatrix{T},standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix
	x=normalize!(randn(T,size(A,1)))
	y=A*x
    ν=x⋅y
    steps=1
    while norm(y-x*ν)>tol && steps<3000
        normalize!(y)
		x=y
        y=A*x
        ν=x⋅y
        # println(ν)
        steps+=1
    end
	if standardform
		z=standardformx(ν)
    	ν=inv(z)*ν*z
		y.*=z
	end
	println("Power ", steps)
	normalize!(y)
    ν, y
end

# ╔═╡ 488877d8-1dd7-43a0-97e1-ce12c2555f5d
md"
## `RQI()` and `MRQI()`
"

# ╔═╡ 7570c157-1f63-47e1-9d31-c2a690e5f55b
function RQI(A::AbstractMatrix{T},standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix 
	# using Rayleigh Quotient Iteration
	n=size(A,1)
	x=normalize!(ones(T,n))
	# Only real shifts
    ν=(x'*x)\(x'*(A*x))
	μ=real(ν)
	y=inv(A-μ*I(n))*x
	normalize!(y)
    steps=1
    while norm(A*y-y*ν)>tol && steps<3000
		x=y
		ν=(x'*x)\(x'*(A*x))
		μ=real(ν)
        y=inv(A-μ*I(n))*x
		normalize!(y)
        # println(ν)
        steps+=1
    end
	if standardform
		z=standardformx(ν)
    	ν=inv(z)*ν*z
		y.*=z
	end
	println("RQI ",steps)
	normalize!(y)
    ν, y
end

# ╔═╡ 1bda8dad-13cc-4746-9242-ae4e555fa480
function MRQI(A::AbstractMatrix{T},standardform::Bool=true,tol::Real=1e-12) where T<:Number
	# Right eigenvalue and eigenvector of a (quaternion) Arrow matrix 
	# using Modified Rayleigh Quotient Iteration
	n=size(A,1)
	x=normalize!(ones(T,n))
	# Only real shifts
    ν=(transpose(x)*x)\(transpose(x)*(A*x))
	μ=real(ν)
	y=inv(A-μ*I(n))*x
	normalize!(y)
    steps=1
    while norm(A*y-y*ν)>tol && steps<3000
		x=y
		ν=(transpose(x)*x)\(transpose(x)*(A*x))
		μ=real(ν)
        y=inv(A-μ*I(n))*x
		normalize!(y)
        # println(ν)
        steps+=1
    end
	if standardform
		z=standardformx(ν)
    	ν=inv(z)*ν*z
		y.*=z
	end
	println("MRQI ",steps)
	normalize!(y)
    ν, y
end

# ╔═╡ ce762b41-6522-46d1-a332-eca6756d9687
md"
## `eigvals()`
"

# ╔═╡ 98d415a5-1cdd-48ae-bad2-46230a7df2b9
md"
## `eigvecs()`
"

# ╔═╡ 8d76c56a-60e8-4356-a0e5-3c41d01bc530
begin
	import LinearAlgebra.eigvecs
	function eigvecs(A::Arrow{T}, λ₁::Vector{T}, ψ₁::Vector{T}) where T<:Number
		# Eigenvectors of a (quaternionic) Arrow given eigenvalues λ₁ and last
		# elements ψ₁
		n=length(λ₁)
		# Create matrix for eigenvectors
		U=Matrix{T}(undef,n,n)
		# Temporary vector
		u=Vector{T}(undef,n)
		for i=1:n
			# Compute α=ρ*y'*u from the first element
			ψ=ψ₁[i]
			λ=λ₁[i]
			u[n]=ψ
			for k=1:n-1
				u[k]=sylvester(A.D[k],-λ,A.u[k]*ψ)
			end
			U[:,i]=u
		end
		return U
	end

	function eigvecs(A::DPRk{T}, λ₁::Vector{T}, ζ₁::Vector{Vector{T}}) where T<:Number
		# Eigenvectors of a (quaternionic) DPRk given eigenvalues λ₁ and first k
		# elements ζ₁, it should be n>k
		n=size(A,1)
		m=length(λ₁)
		k=size(A.x,2)
		# Create matrix for eigenvectors
		U=Matrix{T}(undef,n,m)
		# Temporary vector
		u=Vector{T}(undef,n)
		for i=1:m
			# Compute α=ρ*y'*u from the first k elements
			ζ=ζ₁[i]
			λ=λ₁[i]
			# α=-pinv(transpose(A.x[1,:]))*(A.Δ[1]*ζ-ζ*λ)
			# println(α," ", A.ρ*adjoint(A.y)*F.vectors[:,i])
			# α=A.ρ*adjoint(A.y)*F.vectors[:,i]
			α=A.x[1:k,:]\(ζ*λ-A.Δ[1:k].*ζ)
			u[1:k]=ζ
			for l=k+1:n
				u[l]=sylvester(A.Δ[l],-λ,transpose(A.x[l,:])*α)
			end
			normalize!(u)
			U[:,i]=u
		end
		U
	end
end

# ╔═╡ 7e6dcca5-e545-4822-8e49-634514fd60bb
md"
## `eigen()`
"

# ╔═╡ aefbdfff-3cb2-41a4-89be-67915bfb240b
md"
## Examples
"

# ╔═╡ c7252627-e9ca-4087-abdf-dfa4161013d8
begin
	T=QuaternionF64
	# T=ComplexF64
	n=8
	Esolver=MRQI
	# Esolver=Power
	# Esolver=RQI
end

# ╔═╡ 720abc22-e9ec-48c4-a543-c83fd850b56e
begin
	import LinearAlgebra.eigvals
	function eigvals(A₀::Arrow{T}, standardform::Bool=true) where T<:Number
		# Power iteration and Wielandt deflation to compute eigenvalues of 
		# quaternionic Arrow matrix
		A=A₀
		n=size(A,1)
		# Create vector for eigenvalues
		λ=Vector{T}(undef,n)
		# First eigenpair
		λ[1],x=Esolver(A)
		for i=2:n-1
			# Deflated matrix
			g=x[1]\A.u[1]
			w=A.u[2:end]-x[2:end-1]*g
			α=A.α-x[end]*g
			A=Arrow(A.D[2:end],w,A.v[2:end],α,length(w)+1)
			# Eigenpair
			λ[i],x=Esolver(A)
			# println(u[1],A.ρ*A.y'*u)
		end
		# Last eigenvalue
		ν=A.α-x[2]*(x[1]\A.u[1])
		z=[one(T)]
		if standardform
			z=standardformx(ν)
	    	ν=inv(z)*ν*z
		end
		λ[n]=ν
		return λ
	end

	function eigvals(A₀::DPRk{T}, standardform::Bool=true) where T<:Number
		# Power iteration and Wielandt deflation to compute eigenvalues of 
		# quaternionic DPR1 matrix
		A=A₀
		n=size(A,1)
		# Create vector for eigenvalues
		λ=Vector{T}(undef,n)
		# First eigenpair
		λ[1],u=Esolver(A)
		for i=2:n
			# Deflated matrix
			g=Matrix(transpose(u[1]\A.x[1,:]))
			x=A.x[2:end,:]-u[2:end]*g
			A=DPRk(A.Δ[2:end],x,A.y[2:end,:],A.ρ)
			# Eigenpair
			λ[i],u=Esolver(A)
			# println(u[1],A.ρ*A.y'*u)
		end
		return λ
	end
end

# ╔═╡ e5ab4bf6-4d90-4dd5-9dc6-82adb68ce753
begin
	import LinearAlgebra.eigen
	function eigen(A₀::Arrow{T}, standardform::Bool=true) where T<:Number
		# Power iteration and Wielandt deflation to compute eigenvalues of 
		# quaternionic Arrow matrix
		A=A₀
		n=size(A,1)
		# Create arrays for eigenvalues, first element and eigenvectors
		λ=Vector{T}(undef,n)
		γ=Vector{T}(undef,n)
		# First element of A.x
		χ=Vector{T}(undef,n)
		# x₁=Vector{T}(undef,n)
		# First and last elements of current u
		ν=Vector{T}(undef,n)
		ψ=Vector{T}(undef,n)
		# Eigenvector matrix
		U=zeros(T,n,n)
	
		# First eigenvalue
		λ[1],u=Esolver(A)
		γ[1]=u[1]*λ[1]/u[1]
		# U[:,1]=u
		ν[1]=u[1]
		χ[1]=A.u[1]
	    ψ[1]=u[n]
		for i=2:n-1
			# Deflated matrix
			g=u[1]\A.u[1]
			w=A.u[2:end]-u[2:end-1]*g
			α=A.α-u[end]*g
			A=Arrow(A.D[2:end],w,A.v[2:end],α,length(w)+1)
			# Eigenpair
			λ[i],u=Esolver(A)
			γ[i]=u[1]*λ[i]/u[1]
			ν[i]=u[1]
			χ[i]=A.u[1]
			ψ[i]=u[end]
			# println(u[1],A.ρ*A.v'*u)
		end
		# Last eigenvalue
		μ=A.α-u[2]*(u[1]\A.u[1])
		z=[one(T)]
		if standardform
			z=standardformx(μ)
	    	μ=inv(z)*μ*z
		end
		λ[n]=μ
		ψ[n]=z
	
		# Compute the eigenvectors, bottom-up, the formulas are derived 
		# using (4) and known first and last elements of eigenvectors
		for i=n-1:-1:1
			for j=i+1:n
				ζ=sylvester(γ[i],-λ[j],χ[i]*ψ[j])
				ν[j]=ζ
				ψ[j]=ψ[j]+ψ[i]*(ν[i]\ ζ)
			end
		end

		U=eigvecs(A₀,λ,ψ)
		return Eigen(λ,U)
	end

	function eigen(A₀::DPRk{T}, standardform::Bool=true) where T<:Number
		# Power iteration and Wielandt deflation to compute eigenvalues of 
		# quaternionic DPR1 matrix
		A=A₀
		n=size(A,1)
		k=size(A.x,2)
		# Create arrays for eigenvalues, first elements and eigenvectors
		λ=Vector{T}(undef,n)
		γ=Vector{T}(undef,n)
		# First and second elements of A.x
		χ=Vector{Vector{T}}(undef,n)
		x₁=Vector{Matrix{T}}(undef,n)
		# First and second elements of current u
		ν=Vector{T}(undef,n)
		u₁=Vector{Vector{T}}(undef,n)
		# Eigenvector matrix
		U=zeros(T,n,n)
		
		# First eigenvalue
		λ[1],u=Esolver(A)
		γ[1]=u[1]*λ[1]/u[1]
		# Save elements of computed eigenvector
		# U[:,1]=u
		ν[1]=u[1]
		u₁[1]=u[2:k+1]
		χ[1]=A.x[1,:]
	    x₁[1]=A.x[2:k+1,:]

		# Wielandt's deflation
		for i=2:n
			# Deflated matrix
			g=Matrix(transpose(u[1]\χ[i-1]))
			x=A.x[2:end,:]-u[2:end]*g
			A=DPRk(A.Δ[2:end],x,A.y[2:end,:],A.ρ)
			# Eigenpair of the deflated matrix
			λ[i],u=Esolver(A)
			γ[i]=u[1]*λ[i]/u[1]
			ν[i]=u[1]
			χ[i]=A.x[1,:]

			κ=min(k+1,length(u))
			u₁[i]=u[2:κ]
		    x₁[i]=A.x[2:κ,:]
		end

		# Compute the eigenvectors, bottom-up, the formulas are derived 
		# using (14) and known first elements
		for i=n-1:-1:1
			for j=i+1:n
				if length(u₁[j])==k
					# Standard case
					υ=[ν[j];(u₁[j])[1:k-1]]
					α=(x₁[i]-u₁[i]*(1/ν[i])*transpose(χ[i]))\(υ*λ[j]-A₀.Δ[i+1:i+k].*υ)
					ζ=sylvester(γ[i],-λ[j],transpose(χ[i])*α)
					u₁[j]=υ.+u₁[i]*(1/ν[i])*ζ
					ν[j]=ζ
				else
					# Short case - direct formula
					υ=u₁[j]
					α=A.ρ*adjoint(A₀.y[n-length(υ):n,:])*[ν[j];υ]
					ζ=sylvester(γ[i],-λ[j],transpose(χ[i])*α)
					u₁[j]=[ν[j];υ[1:end]].+u₁[i]*(1/ν[i])*ζ
					ν[j]=ζ
				end
			end
		end

		ξ=Vector{Vector{T}}(undef,n)
		[ξ[i]=[ν[i];u₁[i][1:k-1]] for i=1:n]
		U=eigvecs(A₀,λ,ξ)
		return Eigen(λ,U)
	end
end

# ╔═╡ de95310d-ee54-4357-a5d3-ddb69e331f9e
begin
	# Arrow
	Random.seed!(5419)
	D₀=randn(T,n-1)
	u₀=randn(T,n-1)
	v₀=randn(T,n-1)
	α₀=randn(T)
	A=Arrow(D₀,u₀,v₀,α₀,n)
	if T==Float64
		# Treat everything as complex
		A=Arrow(ComplexF64.(A.D),ComplexF64.(A.u),ComplexF64.(A.v),ComplexF64(A.α),A.i)
	end	
	
	# DPRk
	Random.seed!(5477)
	k=3
	Δ₀=randn(T,n)
	x₀=randn(T,n,k)
	y₀=randn(T,n,k)
	ρ₀=randn(T,k,k)
	B=DPRk(Δ₀,x₀,y₀,ρ₀)
	if T==Float64
		# Treat everything as complex
		B=DPRk(map.(ComplexF64,(B.Δ,B.x,B.y,B.ρ))...)
	end	
end

# ╔═╡ 11e3e179-cc1e-4f98-855b-09b4b858e475
Matrix(A)

# ╔═╡ 9e42901c-6700-46b6-bda8-4edfd92c17fd
Matrix(B)

# ╔═╡ ad608357-03db-46a6-a8f8-86f7a2feec7c
Esolver

# ╔═╡ 4afebe6d-1ae1-47e0-9b60-f4f4429c4ce6
ll,yy=Esolver(A)

# ╔═╡ 1d775cf0-79f7-4e3d-926e-6e7d719c0094
norm(A*yy-yy*ll)

# ╔═╡ ca7a2021-818f-40ef-a81d-19814af94654
eigvals(A)

# ╔═╡ 86010eb7-60e7-4373-8dbe-39ab63c81f73
# Check 
eigvals(unblock(q2c.(Matrix(A))))

# ╔═╡ c805f4b8-906e-4da5-90ad-001013704b1c
E=eigen(A)

# ╔═╡ 8c62f964-a446-441a-82f6-92e2e2be7e93
ErrorA=norm(Matrix(A)*E.vectors-E.vectors*Diagonal(E.values))

# ╔═╡ 2f6c66fd-4f62-4aee-9ebc-6be4ec11830a
Esolver(B)

# ╔═╡ 10739160-e285-48eb-943e-c4da3e032363
eigvals(B)

# ╔═╡ 398efe34-3e43-437c-9bdf-3875b58d77b8
# Check 
eigvals(unblock(q2c.(Matrix(B))))

# ╔═╡ 76811f31-51e7-47b5-a4ed-249c02693787
F=eigen(B)

# ╔═╡ 2865d727-ada1-42f7-b7bf-6762d39b08de
ErrorB=norm(Matrix(B)*F.vectors-F.vectors*Diagonal(F.values))

# ╔═╡ 780548c4-f550-40eb-b075-193f3276096c
ErrorA, ErrorB

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Quaternions = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
GenericLinearAlgebra = "~0.3.11"
PlutoUI = "~0.7.51"
Quaternions = "~0.7.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "d17938f1768a37014a2570862d8911c72482d4a6"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

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
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "02be7066f936af6b04669f7c370a31af9036c440"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

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
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
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
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═ba49a550-fe54-11ed-1d8b-9b5192874bc6
# ╠═45cb51a4-9eea-4fa8-8748-8f6ba24e9c48
# ╟─41db52a5-812e-49bb-9d1f-fb7851740a88
# ╟─dd28cb55-9258-4a32-b3ef-d6af94e256cc
# ╟─2552eeb1-c631-4e58-94f1-3894b9545ab9
# ╟─68b2efe8-99b0-40cf-ba11-6eb8cfa0a4d9
# ╟─8503c651-2bd3-43f2-9d4c-959962606ff5
# ╟─47f842fa-063d-4b30-a734-3f7d825b1314
# ╟─b299fcf7-7ced-45d1-a55c-74482ecb0c60
# ╟─f3a47c9d-c3ba-4056-a3d7-bb4050b3175c
# ╟─f6769f8f-19ad-47c5-a1ec-2e3c780f6cd7
# ╟─22c35821-40f4-4c64-90b3-2ea2ce4e651c
# ╟─fa8ead94-9787-462b-9f41-47fcb41a1a17
# ╟─ff113c87-a72d-4556-98f9-e1e42782a1e6
# ╟─88e05838-d19d-45b8-b7ad-ca1fb6d47f7b
# ╟─a859ec97-d40a-4d35-908d-ccdc16c5fd57
# ╟─09e10ed8-4093-4a43-b452-d2544f29c01c
# ╟─0606fb52-9631-48d0-bf62-d5a5a50deaa5
# ╟─6680f116-a563-40e0-8be3-e69f6910eb57
# ╟─64094f9b-6c1d-4d68-bccc-fd286c121ea1
# ╟─de34d7c6-8632-4ff5-a6ca-082d1f7c1931
# ╟─e747a6e4-70df-4aff-993a-e9a9ad51fa03
# ╟─cf8174d9-035d-4463-ba8b-88b1b6b44317
# ╠═3f8094c4-93ce-4b01-9b50-c7d66031a610
# ╠═a36a7988-5aaf-4a73-8793-b842b00cfbe8
# ╟─ddd9a82c-5d26-42be-80e1-087ba826ee10
# ╠═4e9666dc-26f7-429e-afc5-c72f14a34e9a
# ╟─dc11a4fe-c905-4cb5-a90d-8256cb469a39
# ╠═484df839-8669-4627-a3c3-919300c1c882
# ╠═23817579-826c-47fb-aeee-d67712b59ada
# ╟─bdeab9ab-4838-4332-9b74-7bba33ccb317
# ╠═53a198ac-e7bf-4dc2-9ae3-67f94f15a694
# ╟─322273eb-ca50-42b1-866a-3977700e9b63
# ╠═a20a192a-d4c8-4a9e-936f-fac0c8be6b39
# ╟─1c0a18a8-8490-4790-b3b7-7f56edab3d43
# ╠═15b01358-f35b-4d43-b953-d0f046760db6
# ╟─488877d8-1dd7-43a0-97e1-ce12c2555f5d
# ╠═7570c157-1f63-47e1-9d31-c2a690e5f55b
# ╠═1bda8dad-13cc-4746-9242-ae4e555fa480
# ╟─ce762b41-6522-46d1-a332-eca6756d9687
# ╠═720abc22-e9ec-48c4-a543-c83fd850b56e
# ╟─98d415a5-1cdd-48ae-bad2-46230a7df2b9
# ╠═8d76c56a-60e8-4356-a0e5-3c41d01bc530
# ╟─7e6dcca5-e545-4822-8e49-634514fd60bb
# ╠═e5ab4bf6-4d90-4dd5-9dc6-82adb68ce753
# ╟─aefbdfff-3cb2-41a4-89be-67915bfb240b
# ╠═c7252627-e9ca-4087-abdf-dfa4161013d8
# ╠═780548c4-f550-40eb-b075-193f3276096c
# ╠═de95310d-ee54-4357-a5d3-ddb69e331f9e
# ╠═11e3e179-cc1e-4f98-855b-09b4b858e475
# ╠═9e42901c-6700-46b6-bda8-4edfd92c17fd
# ╠═ad608357-03db-46a6-a8f8-86f7a2feec7c
# ╠═4afebe6d-1ae1-47e0-9b60-f4f4429c4ce6
# ╠═1d775cf0-79f7-4e3d-926e-6e7d719c0094
# ╠═ca7a2021-818f-40ef-a81d-19814af94654
# ╠═86010eb7-60e7-4373-8dbe-39ab63c81f73
# ╠═c805f4b8-906e-4da5-90ad-001013704b1c
# ╠═8c62f964-a446-441a-82f6-92e2e2be7e93
# ╠═2f6c66fd-4f62-4aee-9ebc-6be4ec11830a
# ╠═10739160-e285-48eb-943e-c4da3e032363
# ╠═398efe34-3e43-437c-9bdf-3875b58d77b8
# ╠═76811f31-51e7-47b5-a4ed-249c02693787
# ╠═2865d727-ada1-42f7-b7bf-6762d39b08de
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
